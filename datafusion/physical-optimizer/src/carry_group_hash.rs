// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CarryGroupHash`] hashes the group keys of a two-stage aggregation once.

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions,
};

use crate::PhysicalOptimizerRule;

/// Lets a partial aggregation carry the hash of its group keys to the
/// exchange and the final aggregation (see [`AggregateExec::with_group_hash`]),
/// when `datafusion.execution.carry_group_hash` is set.
///
/// ```txt
/// AggregateExec(mode=FinalPartitioned)        reads the hash column, drops it
///   RepartitionExec(Hash([k0, k1], n))        routes by the hash column
///     AggregateExec(mode=Partial)             emits k0, k1, states, hash
/// ```
///
/// The rule sets a flag on each of the three operators
/// ([`AggregateExec::with_group_hash`],
/// [`RepartitionExec::with_carried_group_hash`] and
/// [`AggregateExec::with_read_group_hash`]); without those flags no operator
/// reads the column, whatever a schema's metadata says.
///
/// With `datafusion.execution.carry_group_hash_forward_views` the exchange
/// also forwards the partial's string view buffers
/// ([`RepartitionExec::with_forward_views`]); only for plans that keep the
/// exchange in one process.
///
/// Only aggregations with a variable-width key are changed: hashing a
/// fixed-width key costs less than moving 8 more bytes per row between the
/// stages. The exchange must hash exactly the partial's group keys, in order.
///
/// Runs after every rule that may rebuild the plan between the stages.
#[derive(Default, Debug)]
pub struct CarryGroupHash {}

impl CarryGroupHash {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for CarryGroupHash {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.execution.carry_group_hash
            || !config.execution.enable_migration_aggregate
        {
            return Ok(plan);
        }
        let forward_views = config.execution.carry_group_hash_forward_views;
        plan.transform_down(|plan| carry(plan, forward_views))
            .data()
    }

    fn name(&self) -> &str {
        "CarryGroupHash"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn carry(
    plan: Arc<dyn ExecutionPlan>,
    forward_views: bool,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(final_agg) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(Transformed::no(plan));
    };
    if !final_agg.can_read_group_hash() {
        return Ok(Transformed::no(plan));
    }
    let Some(exchange) = final_agg.input().downcast_ref::<RepartitionExec>() else {
        return Ok(Transformed::no(plan));
    };
    let Partitioning::Hash(exprs, _) = exchange.partitioning() else {
        return Ok(Transformed::no(plan));
    };
    let Some(partial) = exchange.input().downcast_ref::<AggregateExec>() else {
        return Ok(Transformed::no(plan));
    };
    if !partial.can_carry_group_hash()? || partial.carries_group_hash() {
        return Ok(Transformed::no(plan));
    }

    // The exchange hashes the partial's keys, which it outputs first.
    let num_keys = partial.group_expr().expr().len();
    let hashes_keys = exprs.len() == num_keys
        && exprs.iter().enumerate().all(|(index, expr)| {
            expr.downcast_ref::<Column>()
                .is_some_and(|column| column.index() == index)
        });
    let schema = partial.schema();
    let variable_width = schema.fields()[..num_keys].iter().any(|field| {
        let data_type = field.data_type();
        !data_type.is_primitive() && *data_type != DataType::Boolean
    });
    if !hashes_keys || !variable_width {
        return Ok(Transformed::no(plan));
    }

    // Each of the three operators is told explicitly: none of them infers the
    // carried column from a schema.
    let partial: Arc<dyn ExecutionPlan> = Arc::new(partial.clone().with_group_hash()?);
    let recompute = || ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
    let exchange =
        Arc::clone(final_agg.input()).replace_children(vec![partial], recompute())?;
    let Some(exchange) = exchange.downcast_ref::<RepartitionExec>() else {
        return Ok(Transformed::no(plan));
    };
    let exchange = exchange.clone().with_carried_group_hash()?;
    let exchange: Arc<dyn ExecutionPlan> = Arc::new(if forward_views {
        exchange.with_forward_views()
    } else {
        exchange
    });
    let carried = Arc::clone(&plan).replace_children(vec![exchange], recompute())?;
    let Some(carried) = carried.downcast_ref::<AggregateExec>() else {
        return Ok(Transformed::no(plan));
    };
    if !carried.can_read_group_hash() {
        return Ok(Transformed::no(plan));
    }
    Ok(Transformed::yes(Arc::new(
        carried.clone().with_read_group_hash()?,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::compute::SortOptions;
    use arrow::datatypes::{Field, Schema, SchemaRef};
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
    use datafusion_physical_plan::aggregates::{AggregateMode, PhysicalGroupBy};
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::sorts::sort::SortExec;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new(
                "label",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]))
    }

    /// `GROUP BY` the input's columns `keys` as a partial aggregation, a hash
    /// exchange on the partial's output columns `on` and a final aggregation.
    fn two_stage(
        input: Arc<dyn ExecutionPlan>,
        group_by: PhysicalGroupBy,
        on: &[usize],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let partial = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            group_by.clone(),
            vec![],
            vec![],
            input,
            Arc::clone(&schema),
        )?);
        let exprs = on
            .iter()
            .map(|&index| {
                Arc::new(Column::new(partial.schema().field(index).name(), index))
                    as Arc<dyn PhysicalExpr>
            })
            .collect();
        let exchange = Arc::new(RepartitionExec::try_new(
            partial,
            Partitioning::Hash(exprs, 4),
        )?);
        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            group_by.as_final(),
            vec![],
            vec![],
            exchange,
            schema,
        )?))
    }

    fn group_by(keys: &[&str]) -> Result<PhysicalGroupBy> {
        let schema = schema();
        Ok(PhysicalGroupBy::new_single(
            keys.iter()
                .map(|key| Ok((col(key, &schema)?, (*key).to_string())))
                .collect::<Result<_>>()?,
        ))
    }

    fn empty() -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(schema()))
    }

    /// The flags of the final aggregation, the exchange and the partial
    /// aggregation after the rule.
    fn flags(plan: Arc<dyn ExecutionPlan>) -> Result<[bool; 3]> {
        let mut config = ConfigOptions::new();
        config.execution.carry_group_hash = true;
        let plan = CarryGroupHash::new().optimize(plan, &config)?;
        let carries = |plan: &Arc<dyn ExecutionPlan>| {
            plan.downcast_ref::<AggregateExec>()
                .map(AggregateExec::carries_group_hash)
                .or_else(|| {
                    plan.downcast_ref::<RepartitionExec>()
                        .map(RepartitionExec::carries_group_hash)
                })
                .unwrap_or_default()
        };
        let exchange = Arc::clone(plan.children()[0]);
        let partial = Arc::clone(exchange.children()[0]);
        Ok([carries(&plan), carries(&exchange), carries(&partial)])
    }

    #[test]
    fn carries_the_hash_of_variable_width_keys() -> Result<()> {
        assert_eq!(
            flags(two_stage(empty(), group_by(&["name"])?, &[0])?)?,
            [true; 3]
        );
        assert_eq!(
            flags(two_stage(empty(), group_by(&["name", "n"])?, &[0, 1])?)?,
            [true; 3]
        );
        Ok(())
    }

    #[test]
    fn forwards_views_only_when_asked() -> Result<()> {
        let exchange_of = |forward: bool| -> Result<bool> {
            let mut config = ConfigOptions::new();
            config.execution.carry_group_hash = true;
            config.execution.carry_group_hash_forward_views = forward;
            let plan = two_stage(empty(), group_by(&["name"])?, &[0])?;
            let plan = CarryGroupHash::new().optimize(plan, &config)?;
            Ok(plan.children()[0]
                .downcast_ref::<RepartitionExec>()
                .is_some_and(RepartitionExec::forwards_views))
        };
        assert!(!exchange_of(false)?);
        assert!(exchange_of(true)?);
        Ok(())
    }

    #[test]
    fn leaves_fixed_width_keys() -> Result<()> {
        assert_eq!(
            flags(two_stage(empty(), group_by(&["n"])?, &[0])?)?,
            [false; 3]
        );
        Ok(())
    }

    #[test]
    fn leaves_dictionary_keys() -> Result<()> {
        assert_eq!(
            flags(two_stage(empty(), group_by(&["label"])?, &[0])?)?,
            [false; 3]
        );
        assert_eq!(
            flags(two_stage(empty(), group_by(&["name", "label"])?, &[0, 1])?)?,
            [false; 3]
        );
        Ok(())
    }

    #[test]
    fn leaves_grouping_sets() -> Result<()> {
        // ROLLUP(name, n): (name, n), (name), ()
        let schema = schema();
        let rollup = PhysicalGroupBy::new(
            vec![
                (col("name", &schema)?, "name".to_string()),
                (col("n", &schema)?, "n".to_string()),
            ],
            vec![
                (
                    Arc::new(datafusion_physical_expr::expressions::Literal::new(
                        datafusion_common::ScalarValue::Utf8(None),
                    )) as Arc<dyn PhysicalExpr>,
                    "name".to_string(),
                ),
                (
                    Arc::new(datafusion_physical_expr::expressions::Literal::new(
                        datafusion_common::ScalarValue::Int64(None),
                    )),
                    "n".to_string(),
                ),
            ],
            vec![vec![false, false], vec![false, true], vec![true, true]],
            true,
        );
        // The exchange hashes the keys and the grouping id
        assert_eq!(flags(two_stage(empty(), rollup, &[0, 1, 2])?)?, [false; 3]);
        Ok(())
    }

    #[test]
    fn leaves_ordered_input() -> Result<()> {
        let schema = schema();
        let sorted = Arc::new(SortExec::new(
            [PhysicalSortExpr::new(
                col("name", &schema)?,
                SortOptions::default(),
            )]
            .into(),
            empty(),
        ));
        assert_eq!(
            flags(two_stage(sorted, group_by(&["name"])?, &[0])?)?,
            [false; 3]
        );
        Ok(())
    }

    #[test]
    fn leaves_an_exchange_on_other_columns() -> Result<()> {
        let keys = || group_by(&["name", "n"]);
        // A subset of the keys
        assert_eq!(flags(two_stage(empty(), keys()?, &[0])?)?, [false; 3]);
        assert_eq!(flags(two_stage(empty(), keys()?, &[1])?)?, [false; 3]);
        // The keys in another order
        assert_eq!(flags(two_stage(empty(), keys()?, &[1, 0])?)?, [false; 3]);
        Ok(())
    }

    #[test]
    fn exchange_and_final_refuse_an_input_without_the_hash() -> Result<()> {
        let plan = two_stage(empty(), group_by(&["name"])?, &[0])?;
        let exchange = plan.children()[0]
            .downcast_ref::<RepartitionExec>()
            .cloned();
        assert!(
            exchange.is_some_and(|exchange| exchange.with_carried_group_hash().is_err())
        );
        let final_agg = plan.downcast_ref::<AggregateExec>().cloned();
        assert!(final_agg.is_some_and(|agg| agg.with_read_group_hash().is_err()));
        Ok(())
    }
}
