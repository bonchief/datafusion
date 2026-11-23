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
        plan.transform_down(carry).data()
    }

    fn name(&self) -> &str {
        "CarryGroupHash"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn carry(plan: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
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

    let partial: Arc<dyn ExecutionPlan> = Arc::new(partial.clone().with_group_hash()?);
    let recompute = || ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
    let exchange =
        Arc::clone(final_agg.input()).replace_children(vec![partial], recompute())?;
    let carried = Arc::clone(&plan).replace_children(vec![exchange], recompute())?;
    let reads = carried
        .downcast_ref::<AggregateExec>()
        .is_some_and(|agg| agg.can_read_group_hash() && agg.carries_group_hash());
    Ok(if reads {
        Transformed::yes(carried)
    } else {
        Transformed::no(plan)
    })
}
