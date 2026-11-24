// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`SingleDistinctToGroupBy`] replaces `AGG(DISTINCT ..)` with `AGG(..) GROUP BY ..`

use std::sync::Arc;

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

use arrow::datatypes::DataType;
use datafusion_common::{
    DFSchema, DataFusionError, HashSet, Result, assert_or_internal_err,
    tree_node::Transformed,
};
use datafusion_expr::builder::project;
use datafusion_expr::expr::AggregateFunctionParams;
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{
    AggregateUDF, Expr, cast, col,
    expr::AggregateFunction,
    lit,
    logical_plan::{Aggregate, LogicalPlan},
    when,
};

/// single distinct to group by optimizer rule
///  ```text
///    Before:
///    SELECT a, count(DISTINCT b), sum(c)
///    FROM t
///    GROUP BY a
///
///    After:
///    SELECT a, count(alias1), sum(alias2)
///    FROM (
///      SELECT a, b as alias1, sum(c) as alias2
///      FROM t
///      GROUP BY a, b
///    )
///    GROUP BY a
///  ```
///
/// A non-distinct `count` may also appear alongside the distinct aggregate. It
/// is the one supported function whose outer phase is a different function: the
/// inner group-by counts rows per `(group, distinct value)` pair, and the outer
/// phase sums those partial counts.
///
///  ```text
///    Before:
///    SELECT a, count(*), count(DISTINCT b)
///    FROM t
///    GROUP BY a
///
///    After:
///    SELECT a,
///           CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE 0 END,
///           count(alias1)
///    FROM (
///      SELECT a, b as alias1, count(*) as alias2
///      FROM t
///      GROUP BY a, b
///    )
///    GROUP BY a
///  ```
///
/// The `CASE` restores the 0 that `count` reports over an empty input, where a
/// `sum` of no rows is NULL.
///
/// A non-distinct `avg` over a `Float64` argument, which is what type coercion
/// makes of an `avg` over any integer or float column, may appear as well. The
/// inner group-by keeps its sum and its count of non-null values, and the
/// outer phase divides the sum of the sums by the sum of the counts:
///
///  ```text
///    Before:
///    SELECT a, avg(c), count(DISTINCT b)
///    FROM t
///    GROUP BY a
///
///    After:
///    SELECT a, sum(alias2) / CAST(sum(alias3) AS Float64), count(alias1)
///    FROM (
///      SELECT a, b as alias1, sum(c) as alias2, count(c) as alias3
///      FROM t
///      GROUP BY a, b
///    )
///    GROUP BY a
///  ```
///
/// A group whose values are all NULL has a NULL sum, so the quotient is NULL,
/// as `avg` reports. A decimal or duration `avg` is left alone: its result
/// type is not the quotient's.
///
/// A `count` or an `avg` is allowed when the plan runs in several partitions,
/// and in one partition only when `rewrite_pays`.
#[derive(Default, Debug)]
pub struct SingleDistinctToGroupBy {}

const SINGLE_DISTINCT_ALIAS: &str = "alias1";

impl SingleDistinctToGroupBy {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// The functions used to compute a non-distinct `count` or `avg` in two
/// phases: `count` and `avg` identify the aggregates that need the treatment,
/// `sum` combines the partial sums and counts the inner group-by produces.
///
/// They are resolved from the session's function registry, so a plan built
/// without one keeps the previous behaviour of bailing out on `count` and
/// `avg`, and a registry without `avg` bails out on `avg` alone.
struct Rollup {
    count: Arc<AggregateUDF>,
    sum: Arc<AggregateUDF>,
    avg: Option<Arc<AggregateUDF>>,
}

impl Rollup {
    fn try_new(config: &dyn OptimizerConfig) -> Option<Self> {
        let registry = config.function_registry()?;
        Some(Self {
            count: registry.udaf("count").ok()?,
            sum: registry.udaf("sum").ok()?,
            avg: registry.udaf("avg").ok(),
        })
    }

    /// Whether `func` is the registry's `count`, and so decomposes into
    /// `sum` over per-partition counts.
    fn is_count(&self, func: &AggregateUDF) -> bool {
        self.count.as_ref() == func
    }

    /// Whether `func` is the registry's `avg`, and so decomposes into the
    /// sum of partial sums over the sum of partial counts.
    fn is_avg(&self, func: &AggregateUDF) -> bool {
        self.avg.as_deref().is_some_and(|avg| avg == func)
    }
}

/// Whether `args` are the single `Float64` argument an `avg` decomposes over.
fn is_float_avg_arg(args: &[Expr], input_schema: &DFSchema) -> Result<bool> {
    Ok(match args {
        [arg] => arg.get_type(input_schema)? == DataType::Float64,
        _ => false,
    })
}

/// Check whether all aggregate exprs are distinct on a single field.
fn is_single_distinct_agg(
    aggr_expr: &[Expr],
    input_schema: &DFSchema,
    rollup: Option<&Rollup>,
    parallel: bool,
) -> Result<bool> {
    let mut fields_set = HashSet::new();
    let mut aggregate_count = 0;
    let mut distinct_aggs = vec![];
    let mut has_rollup = false;
    for expr in aggr_expr {
        if let Expr::AggregateFunction(AggregateFunction {
            func,
            params:
                AggregateFunctionParams {
                    distinct,
                    args,
                    filter,
                    order_by,
                    null_treatment: _,
                },
        }) = expr
        {
            if filter.is_some() || !order_by.is_empty() {
                return Ok(false);
            }
            aggregate_count += 1;
            if *distinct {
                for e in args {
                    fields_set.insert(e);
                }
                distinct_aggs.push((func, args.as_slice()));
            } else if rollup.is_some_and(|rollup| rollup.is_count(func)) {
                has_rollup = true;
            } else if rollup.is_some_and(|rollup| rollup.is_avg(func)) {
                if !is_float_avg_arg(args, input_schema)? {
                    return Ok(false);
                }
                has_rollup = true;
            } else if func.name() != "sum"
                && func.name().to_lowercase() != "min"
                && func.name().to_lowercase() != "max"
            {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }
    if aggregate_count != aggr_expr.len() || fields_set.len() != 1 {
        return Ok(false);
    }
    if has_rollup && !parallel && !rewrite_pays(&distinct_aggs, input_schema)? {
        return Ok(false);
    }
    Ok(true)
}

/// Whether the rewrite is worth extending to a plan that runs in one partition
/// and only qualifies because of a non-distinct `count` or `avg`.
///
/// A plan that runs in several partitions always takes the rewrite. There
/// the distinct aggregate's own `GroupsAccumulator` splits into a partial and
/// a final phase too, but its partial state is a list of distinct values per
/// group, and the final phase merges every group's lists in the partition the
/// group hashes to. A group that holds a large share of the distinct values
/// then serializes most of the merge in one partition. The rewrite's inner
/// group-by is partitioned on the group keys and the distinct value together,
/// so it spreads that work over every partition whatever the skew, and its
/// state spills where a distinct set cannot. Over 100,000,000 ClickBench rows
/// in 26 partitions, `SELECT "RegionID", SUM("AdvEngineID"), COUNT(*),
/// AVG("ResolutionWidth"), COUNT(DISTINCT "UserID") .. GROUP BY "RegionID"`
/// ran in 0.77x the time rewritten. In one partition there is neither a
/// merge nor its skew, and the rewrite only adds the inner group-by.
///
/// The rewrite is not free: every other aggregate moves down to the inner group
/// by, which has a row per `(group, distinct value)` pair rather than per group,
/// and keeps its state at that finer grain. What pays for that is taking the
/// distinct aggregate off `GroupsAccumulatorAdapter` and its one boxed
/// accumulator per group. An aggregate that already has a specialized
/// `GroupsAccumulator` never went near the adapter, so there is nothing to buy
/// and only the inner group by to pay for.
///
/// `Some(false)` is therefore the only answer that permits the rewrite. `None`,
/// the default for almost every function, is not evidence that it pays.
///
/// This is a proxy: what actually decides the outcome is the cost per distinct
/// value on each side, which this rule cannot see. It is deliberately
/// conservative in the direction that leaves a plan alone. The pre-existing
/// tolerance of a non-distinct `sum`, `min` or `max` is not gated, since
/// narrowing it would change plans that have always been rewritten.
///
/// Measured in <https://github.com/apache/datafusion/pull/24859>.
fn rewrite_pays(
    distinct_aggs: &[(&Arc<AggregateUDF>, &[Expr])],
    input_schema: &DFSchema,
) -> Result<bool> {
    for (func, args) in distinct_aggs {
        let arg_types = args
            .iter()
            .map(|arg| arg.get_type(input_schema))
            .collect::<Result<Vec<_>>>()?;
        if func.groups_accumulator_supported_for_types(&arg_types, true) == Some(false) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if the first expr is [Expr::GroupingSet].
fn contains_grouping_set(expr: &[Expr]) -> bool {
    matches!(expr.first(), Some(Expr::GroupingSet(_)))
}

impl OptimizerRule for SingleDistinctToGroupBy {
    fn name(&self) -> &str {
        "single_distinct_aggregation_to_group_by"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>, DataFusionError> {
        if !matches!(plan, LogicalPlan::Aggregate(_)) {
            return Ok(Transformed::no(plan));
        }
        let rollup = Rollup::try_new(config);
        let parallel = config.options().execution.target_partitions > 1;
        match plan {
            LogicalPlan::Aggregate(Aggregate {
                input,
                aggr_expr,
                schema,
                group_expr,
                ..
            }) if is_single_distinct_agg(
                &aggr_expr,
                input.schema(),
                rollup.as_ref(),
                parallel,
            )? && !contains_grouping_set(&group_expr) =>
            {
                let group_size = group_expr.len();
                // alias all original group_by exprs
                let (mut inner_group_exprs, out_group_expr_with_alias): (
                    Vec<Expr>,
                    Vec<(Expr, _)>,
                ) = group_expr
                    .into_iter()
                    .enumerate()
                    .map(|(i, group_expr)| {
                        if let Expr::Column(_) = group_expr {
                            // For Column expressions we can use existing expression as is.
                            (group_expr.clone(), (group_expr, None))
                        } else {
                            // For complex expression write is as alias, to be able to refer
                            // if from parent operators successfully.
                            // Consider plan below.
                            //
                            // Aggregate: groupBy=[[group_alias_0]], aggr=[[count(alias1)]] [group_alias_0:Int32, count(alias1):Int64;N]\
                            // --Aggregate: groupBy=[[test.a + Int32(1) AS group_alias_0, test.c AS alias1]], aggr=[[]] [group_alias_0:Int32, alias1:UInt32]\
                            // ----TableScan: test [a:UInt32, b:UInt32, c:UInt32]
                            //
                            // First aggregate(from bottom) refers to `test.a` column.
                            // Second aggregate refers to the `group_alias_0` column, Which is a valid field in the first aggregate.

                            // If we were to write plan above as below without alias
                            //
                            // Aggregate: groupBy=[[test.a + Int32(1)]], aggr=[[count(alias1)]] [group_alias_0:Int32, count(alias1):Int64;N]\
                            // --Aggregate: groupBy=[[test.a + Int32(1), test.c AS alias1]], aggr=[[]] [group_alias_0:Int32, alias1:UInt32]\
                            // ----TableScan: test [a:UInt32, b:UInt32, c:UInt32]
                            //
                            // Second aggregate refers to the `test.a + Int32(1)` expression However, its input do not have `test.a` expression in it.
                            let alias_str = format!("group_alias_{i}");
                            let (qualifier, field) = schema.qualified_field(i);
                            (
                                group_expr.alias(alias_str.clone()),
                                (col(alias_str), Some((qualifier, field.name()))),
                            )
                        }
                    })
                    .unzip();

                // replace the distinct arg with alias
                let mut index = 1;
                let mut group_fields_set = HashSet::new();
                let mut inner_aggr_exprs = vec![];
                // Each aggregate yields the expressions the outer `Aggregate`
                // computes and the expression the projection above it selects.
                // They differ for `count`, whose projection restores the zero
                // that `sum` reports as NULL over an empty input, and for
                // `avg`, whose projection divides its two outer sums.
                let outer: Vec<(Vec<Expr>, Expr)> = aggr_expr
                    .into_iter()
                    .map(|aggr_expr| match aggr_expr {
                        Expr::AggregateFunction(AggregateFunction {
                            func,
                            params:
                                AggregateFunctionParams {
                                    mut args,
                                    distinct,
                                    filter,
                                    order_by,
                                    null_treatment,
                                },
                        }) => {
                            if distinct {
                                // `is_single_distinct_agg` checked that every
                                // distinct argument in the plan is the same
                                // expression. A call may still take it more than
                                // once, as `corr(DISTINCT x, x)` does, and keeps
                                // its arity here: its distinct argument tuples
                                // are the distinct values of that expression.
                                assert_or_internal_err!(
                                    args.first()
                                        .is_some_and(|f| args.iter().all(|a| a == f)),
                                    "DISTINCT aggregate arguments should be the same expression"
                                );
                                let arity = args.len();
                                let arg = args.swap_remove(0);

                                if group_fields_set.insert(arg.schema_name().to_string())
                                {
                                    inner_group_exprs
                                        .push(arg.alias(SINGLE_DISTINCT_ALIAS));
                                }
                                let outer =
                                    Expr::AggregateFunction(AggregateFunction::new_udf(
                                        func,
                                        vec![col(SINGLE_DISTINCT_ALIAS); arity],
                                        false, // intentional to remove distinct here
                                        filter,
                                        order_by,
                                        null_treatment,
                                    ));
                                return Ok((vec![outer.clone()], outer));
                            }
                            // if the aggregate function is not distinct, we need to rewrite it like two phase aggregation
                            let mut phase = |inner: Arc<AggregateUDF>,
                                             args: Vec<Expr>,
                                             outer: Arc<AggregateUDF>| {
                                index += 1;
                                let alias = format!("alias{index}");
                                inner_aggr_exprs.push(
                                    Expr::AggregateFunction(AggregateFunction::new_udf(
                                        inner,
                                        args,
                                        false,
                                        filter.clone(),
                                        order_by.clone(),
                                        null_treatment,
                                    ))
                                    .alias(&alias),
                                );
                                Expr::AggregateFunction(AggregateFunction::new_udf(
                                    outer,
                                    vec![col(alias)],
                                    false,
                                    None,
                                    vec![],
                                    None,
                                ))
                            };
                            match rollup.as_ref() {
                                // `count`'s outer phase is `sum`, over the
                                // partial counts the inner group-by produced.
                                Some(rollup) if rollup.is_count(&func) => {
                                    let outer =
                                        phase(func, args, Arc::clone(&rollup.sum));
                                    // `sum` of no rows is NULL where `count` is 0.
                                    // Restoring the 0 also keeps the column
                                    // non-nullable, as `count` had it.
                                    let proj =
                                        when(outer.clone().is_not_null(), outer.clone())
                                            .otherwise(lit(0_i64))?;
                                    Ok((vec![outer], proj))
                                }
                                // `avg` is the sum of the partial sums over the
                                // sum of the partial counts of non-null values.
                                Some(rollup) if rollup.is_avg(&func) => {
                                    let sum = Arc::clone(&rollup.sum);
                                    let total = phase(
                                        Arc::clone(&sum),
                                        args.clone(),
                                        Arc::clone(&sum),
                                    );
                                    let count =
                                        phase(Arc::clone(&rollup.count), args, sum);
                                    let proj = total.clone()
                                        / cast(count.clone(), DataType::Float64);
                                    Ok((vec![total, count], proj))
                                }
                                _ => {
                                    let outer = phase(Arc::clone(&func), args, func);
                                    Ok((vec![outer.clone()], outer))
                                }
                            }
                        }
                        _ => Ok((vec![aggr_expr.clone()], aggr_expr)),
                    })
                    .collect::<Result<_>>()?;
                let (outer_aggr_exprs, outer_proj_exprs): (Vec<Vec<Expr>>, Vec<Expr>) =
                    outer.into_iter().unzip();
                let outer_aggr_exprs = outer_aggr_exprs.into_iter().flatten().collect();

                // construct the inner AggrPlan
                let inner_agg = LogicalPlan::Aggregate(Aggregate::try_new(
                    input,
                    inner_group_exprs,
                    inner_aggr_exprs,
                )?);

                let outer_group_exprs = out_group_expr_with_alias
                    .iter()
                    .map(|(expr, _)| expr.clone())
                    .collect();

                let outer_aggr = LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::new(inner_agg),
                    outer_group_exprs,
                    outer_aggr_exprs,
                )?);

                // so the aggregates are displayed in the same way even after the rewrite
                // this optimizer has two kinds of alias:
                // - group_by aggr
                // - aggr expr
                let group_exprs = out_group_expr_with_alias.into_iter().map(
                    |(group_expr, original_name)| match original_name {
                        Some((qualifier, name)) => {
                            group_expr.alias_qualified(qualifier.cloned(), name)
                        }
                        None => group_expr,
                    },
                );
                // The type each projected expression has over the outer
                // aggregate, which reads its aggregates as columns by name.
                let types = project(outer_aggr.clone(), outer_proj_exprs.clone())?
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.data_type().clone())
                    .collect::<Vec<_>>();
                let alias_expr: Vec<Expr> = group_exprs
                    .chain(outer_proj_exprs.into_iter().zip(types).enumerate().map(
                        |(idx, (expr, current_type))| {
                            let (qualifier, field) =
                                schema.qualified_field(idx + group_size);
                            let expr = if &current_type != field.data_type() {
                                cast(expr, field.data_type().clone())
                            } else {
                                expr
                            };
                            expr.alias_qualified(qualifier.cloned(), field.name())
                        },
                    ))
                    .collect();

                Ok(Transformed::yes(project(outer_aggr, alias_expr)?))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_optimized_plan_eq_display_indent_snapshot;
    use crate::test::*;
    use crate::{Optimizer, OptimizerContext};
    use arrow::datatypes::{DataType, Field, Schema};
    use chrono::{DateTime, Utc};
    use datafusion_common::alias::AliasGenerator;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::ExprFunctionExt;
    use datafusion_expr::expr::GroupingSet;
    use datafusion_expr::function::AccumulatorArgs;
    use datafusion_expr::registry::{FunctionRegistry, MemoryFunctionRegistry};
    use datafusion_expr::{Accumulator, AggregateUDFImpl, Signature, Volatility};
    use datafusion_expr::{
        lit,
        logical_plan::builder::{LogicalPlanBuilder, table_scan},
    };
    use datafusion_functions_aggregate::average::avg_udaf;
    use datafusion_functions_aggregate::correlation::corr_udaf;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_functions_aggregate::expr_fn::{
        avg, count, count_distinct, max, min, sum,
    };
    use datafusion_functions_aggregate::min_max::max_udaf;
    use datafusion_functions_aggregate::sum::sum_udaf;

    fn max_distinct(expr: Expr) -> Expr {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            max_udaf(),
            vec![expr],
            true,
            None,
            vec![],
            None,
        ))
    }

    fn corr_distinct(y: Expr, x: Expr) -> Expr {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            corr_udaf(),
            vec![y, x],
            true,
            None,
            vec![],
            None,
        ))
    }

    fn sum_distinct(expr: Expr) -> Expr {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            sum_udaf(),
            vec![expr],
            true,
            None,
            vec![],
            None,
        ))
    }

    /// An aggregate that leaves
    /// [`AggregateUDFImpl::groups_accumulator_supported_for_types`] at its
    /// default, which is what almost every function in the wild does.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct SilentUdaf {
        signature: Signature,
    }

    impl SilentUdaf {
        fn new() -> Self {
            Self {
                signature: Signature::any(1, Volatility::Immutable),
            }
        }
    }

    impl AggregateUDFImpl for SilentUdaf {
        fn name(&self) -> &str {
            "silent"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::UInt32)
        }

        fn accumulator(
            &self,
            _acc_args: AccumulatorArgs,
        ) -> Result<Box<dyn Accumulator>> {
            unimplemented!("not needed for this test")
        }
    }

    fn silent_distinct(expr: Expr) -> Expr {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            Arc::new(AggregateUDF::from(SilentUdaf::new())),
            vec![expr],
            true,
            None,
            vec![],
            None,
        ))
    }

    fn count_star() -> Expr {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            count_udaf(),
            vec![lit(1_i64)],
            false,
            None,
            vec![],
            None,
        ))
    }

    /// `test` with a `Utf8` `b`, the column the distinct aggregate reads.
    ///
    /// `count(DISTINCT b)` over a string has no specialized
    /// `GroupsAccumulator`, so it is the case a non-distinct `count` may join.
    /// The `UInt32` `b` of [`test_table_scan`] is the case it may not.
    fn test_table_scan_utf8_b() -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::UInt32, false),
        ]);
        table_scan(Some("test"), &schema, None)?.build()
    }

    /// An [`OptimizerConfig`] that resolves functions the way a session does.
    /// The rule needs `count` and `sum` from the registry to rewrite a
    /// non-distinct `count`.
    #[derive(Debug)]
    struct RegistryOptimizerContext {
        inner: OptimizerContext,
        registry: MemoryFunctionRegistry,
    }

    impl RegistryOptimizerContext {
        /// A context planning for one partition, where the rewrite of a
        /// `count` or an `avg` is gated on the distinct aggregate.
        fn new() -> Self {
            Self::with_target_partitions(1)
        }

        fn with_target_partitions(target_partitions: usize) -> Self {
            let mut options = ConfigOptions::new();
            options.execution.target_partitions = target_partitions;
            let mut registry = MemoryFunctionRegistry::new();
            registry.register_udaf(count_udaf()).unwrap();
            registry.register_udaf(sum_udaf()).unwrap();
            registry.register_udaf(avg_udaf()).unwrap();
            Self {
                inner: OptimizerContext::new_with_config_options(Arc::new(options)),
                registry,
            }
        }
    }

    impl OptimizerConfig for RegistryOptimizerContext {
        fn query_execution_start_time(&self) -> Option<DateTime<Utc>> {
            self.inner.query_execution_start_time()
        }

        fn alias_generator(&self) -> &Arc<AliasGenerator> {
            self.inner.alias_generator()
        }

        fn options(&self) -> Arc<ConfigOptions> {
            self.inner.options()
        }

        fn function_registry(&self) -> Option<&dyn FunctionRegistry> {
            Some(&self.registry)
        }
    }

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {
            assert_optimized_plan_equal!($plan, partitions = 1, @ $expected)
        };
        (
            $plan:expr,
            partitions = $partitions:expr,
            @ $expected:literal $(,)?
        ) => {{
            let rule: Arc<dyn crate::OptimizerRule + Send + Sync> =
                Arc::new(SingleDistinctToGroupBy::new());
            let optimizer = Optimizer::with_rules(vec![rule]);
            let context = RegistryOptimizerContext::with_target_partitions($partitions);
            let optimized_plan = optimizer
                .optimize($plan, &context, |_, _| {})
                .expect("failed to optimize plan");
            insta::assert_snapshot!(optimized_plan.display_indent_schema(), @ $expected);

            Ok::<(), DataFusionError>(())
        }};
    }

    #[test]
    fn not_exist_distinct() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(Vec::<Expr>::new(), vec![max(col("b"))])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[]], aggr=[[max(test.b)]] [max(test.b):UInt32;N]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn single_distinct() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(Vec::<Expr>::new(), vec![count_distinct(col("b"))])?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: count(alias1) AS count(DISTINCT test.b) [count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[]], aggr=[[count(alias1)]] [count(alias1):Int64]
            Aggregate: groupBy=[[test.b AS alias1]], aggr=[[]] [alias1:UInt32]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    // Currently this optimization is disabled for CUBE/ROLLUP/GROUPING SET
    #[test]
    fn single_distinct_and_grouping_set() -> Result<()> {
        let table_scan = test_table_scan()?;

        let grouping_set = Expr::GroupingSet(GroupingSet::GroupingSets(vec![
            vec![col("a")],
            vec![col("b")],
        ]));

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![grouping_set], vec![count_distinct(col("c"))])?
            .build()?;

        // Should not be optimized
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[GROUPING SETS ((test.a), (test.b))]], aggr=[[count(DISTINCT test.c)]] [a:UInt32;N, b:UInt32;N, __grouping_id:UInt8, count(DISTINCT test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    // Currently this optimization is disabled for CUBE/ROLLUP/GROUPING SET
    #[test]
    fn single_distinct_and_cube() -> Result<()> {
        let table_scan = test_table_scan()?;

        let grouping_set = Expr::GroupingSet(GroupingSet::Cube(vec![col("a"), col("b")]));

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![grouping_set], vec![count_distinct(col("c"))])?
            .build()?;

        // Should not be optimized
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[CUBE (test.a, test.b)]], aggr=[[count(DISTINCT test.c)]] [a:UInt32;N, b:UInt32;N, __grouping_id:UInt8, count(DISTINCT test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    // Currently this optimization is disabled for CUBE/ROLLUP/GROUPING SET
    #[test]
    fn single_distinct_and_rollup() -> Result<()> {
        let table_scan = test_table_scan()?;

        let grouping_set =
            Expr::GroupingSet(GroupingSet::Rollup(vec![col("a"), col("b")]));

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![grouping_set], vec![count_distinct(col("c"))])?
            .build()?;

        // Should not be optimized
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[ROLLUP (test.a, test.b)]], aggr=[[count(DISTINCT test.c)]] [a:UInt32;N, b:UInt32;N, __grouping_id:UInt8, count(DISTINCT test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn single_distinct_expr() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(Vec::<Expr>::new(), vec![count_distinct(lit(2) * col("b"))])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: count(alias1) AS count(DISTINCT Int32(2) * test.b) [count(DISTINCT Int32(2) * test.b):Int64]
          Aggregate: groupBy=[[]], aggr=[[count(alias1)]] [count(alias1):Int64]
            Aggregate: groupBy=[[Int32(2) * test.b AS alias1]], aggr=[[]] [alias1:Int64]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn single_distinct_and_groupby() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![count_distinct(col("b"))])?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, count(alias1) AS count(DISTINCT test.b) [a:UInt32, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[count(alias1)]] [a:UInt32, count(alias1):Int64]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[]] [a:UInt32, alias1:UInt32]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn two_distinct_and_groupby() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), count_distinct(col("c"))],
            )?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[count(DISTINCT test.b), count(DISTINCT test.c)]] [a:UInt32, count(DISTINCT test.b):Int64, count(DISTINCT test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn single_distinct_repeated_arg_and_groupby() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![corr_distinct(col("b"), col("b"))])?
            .build()?;

        // Should work: the call takes the distinct argument twice, so it keeps
        // both arguments after the rewrite.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, corr(alias1,alias1) AS corr(DISTINCT test.b,test.b) [a:UInt32, corr(DISTINCT test.b,test.b):Float64;N]
          Aggregate: groupBy=[[test.a]], aggr=[[corr(alias1, alias1)]] [a:UInt32, corr(alias1,alias1):Float64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[]] [a:UInt32, alias1:UInt32]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn single_distinct_two_args_and_groupby() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![corr_distinct(col("b"), col("c"))])?
            .build()?;

        // Do nothing: the arguments are different expressions, so there is no
        // single field to group by.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[corr(DISTINCT test.b, test.c)]] [a:UInt32, corr(DISTINCT test.b,test.c):Float64;N]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn one_field_two_distinct_and_groupby() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), max_distinct(col("b"))],
            )?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, count(alias1) AS count(DISTINCT test.b), max(alias1) AS max(DISTINCT test.b) [a:UInt32, count(DISTINCT test.b):Int64, max(DISTINCT test.b):UInt32;N]
          Aggregate: groupBy=[[test.a]], aggr=[[count(alias1), max(alias1)]] [a:UInt32, count(alias1):Int64, max(alias1):UInt32;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[]] [a:UInt32, alias1:UInt32]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_and_common() -> Result<()> {
        let table_scan = test_table_scan_utf8_b()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should work: the non-distinct count becomes a sum of the counts the
        // inner group-by produces per (test.a, test.b) pair
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, count(alias1) AS count(DISTINCT test.b), CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(test.c) [a:UInt32, count(DISTINCT test.b):Int64, count(test.c):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[count(alias1), sum(alias2)]] [a:UInt32, count(alias1):Int64, sum(alias2):Int64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[count(test.c) AS alias2]] [a:UInt32, alias1:Utf8, alias2:Int64]
              TableScan: test [a:UInt32, b:Utf8, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_and_common_over_a_natively_supported_type() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should not work: `count(DISTINCT b)` over a `UInt32` has its own
        // `GroupsAccumulator`, so the rewrite would add an inner group-by
        // without taking anything off `GroupsAccumulatorAdapter`
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[count(DISTINCT test.b), count(test.c)]] [a:UInt32, count(DISTINCT test.b):Int64, count(test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_over_a_natively_supported_type_without_a_count() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), sum(col("c"))],
            )?
            .build()?;

        // Should work: the gate covers only the `count` this change added, so a
        // plan that already qualified through `sum`, `min` or `max` is rewritten
        // exactly as before
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, count(alias1) AS count(DISTINCT test.b), sum(alias2) AS sum(test.c) [a:UInt32, count(DISTINCT test.b):Int64, sum(test.c):UInt64;N]
          Aggregate: groupBy=[[test.a]], aggr=[[count(alias1), sum(alias2)]] [a:UInt32, count(alias1):Int64, sum(alias2):UInt64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[sum(test.c) AS alias2]] [a:UInt32, alias1:UInt32, alias2:UInt64;N]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_over_a_function_with_no_opinion_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![silent_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should not work: `silent` leaves
        // `groups_accumulator_supported_for_types` at its default, so it
        // reports nothing about `GroupsAccumulatorAdapter`. Silence is not
        // evidence that the rewrite pays, and the rule must not read it as
        // permission
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[silent(DISTINCT test.b), count(test.c)]] [a:UInt32, silent(DISTINCT test.b):UInt32;N, count(test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_sum_and_count_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![sum_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should not work: `sum` reports nothing about
        // `GroupsAccumulatorAdapter` either, so the rewrite is not known to pay
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[sum(DISTINCT test.b), count(test.c)]] [a:UInt32, sum(DISTINCT test.b):UInt64;N, count(test.c):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn group_by_with_expr() -> Result<()> {
        let table_scan = test_table_scan().unwrap();

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a") + lit(1)], vec![count_distinct(col("c"))])?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: group_alias_0 AS test.a + Int32(1), count(alias1) AS count(DISTINCT test.c) [test.a + Int32(1):Int64, count(DISTINCT test.c):Int64]
          Aggregate: groupBy=[[group_alias_0]], aggr=[[count(alias1)]] [group_alias_0:Int64, count(alias1):Int64]
            Aggregate: groupBy=[[test.a + Int32(1) AS group_alias_0, test.c AS alias1]], aggr=[[]] [group_alias_0:Int64, alias1:UInt32]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn two_distinct_and_one_common() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![
                    sum(col("c")),
                    count_distinct(col("b")),
                    max_distinct(col("b")),
                ],
            )?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, sum(alias2) AS sum(test.c), count(alias1) AS count(DISTINCT test.b), max(alias1) AS max(DISTINCT test.b) [a:UInt32, sum(test.c):UInt64;N, count(DISTINCT test.b):Int64, max(DISTINCT test.b):UInt32;N]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias2), count(alias1), max(alias1)]] [a:UInt32, sum(alias2):UInt64;N, count(alias1):Int64, max(alias1):UInt32;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[sum(test.c) AS alias2]] [a:UInt32, alias1:UInt32, alias2:UInt64;N]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn one_distinct_and_two_common() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![sum(col("c")), max(col("c")), count_distinct(col("b"))],
            )?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, sum(alias2) AS sum(test.c), max(alias3) AS max(test.c), count(alias1) AS count(DISTINCT test.b) [a:UInt32, sum(test.c):UInt64;N, max(test.c):UInt32;N, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias2), max(alias3), count(alias1)]] [a:UInt32, sum(alias2):UInt64;N, max(alias3):UInt32;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[sum(test.c) AS alias2, max(test.c) AS alias3]] [a:UInt32, alias1:UInt32, alias2:UInt64;N, alias3:UInt32;N]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn one_distinct_and_one_common() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("c")],
                vec![min(col("a")), count_distinct(col("b"))],
            )?
            .build()?;

        // Should work
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.c, min(alias2) AS min(test.a), count(alias1) AS count(DISTINCT test.b) [c:UInt32, min(test.a):UInt32;N, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[test.c]], aggr=[[min(alias2), count(alias1)]] [c:UInt32, min(alias2):UInt32;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.c, test.b AS alias1]], aggr=[[min(test.a) AS alias2]] [c:UInt32, alias1:UInt32, alias2:UInt32;N]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn common_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;

        // sum(a) FILTER (WHERE a > 5)
        let expr = Expr::AggregateFunction(AggregateFunction::new_udf(
            sum_udaf(),
            vec![col("a")],
            false,
            Some(Box::new(col("a").gt(lit(5)))),
            vec![],
            None,
        ));
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[sum(test.a) FILTER (WHERE test.a > Int32(5)), count(DISTINCT test.b)]] [c:UInt32, sum(test.a) FILTER (WHERE test.a > Int32(5)):UInt64;N, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(DISTINCT a) FILTER (WHERE a > 5)
        let expr = count_udaf()
            .call(vec![col("a")])
            .distinct()
            .filter(col("a").gt(lit(5)))
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![sum(col("a")), expr])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[sum(test.a), count(DISTINCT test.a) FILTER (WHERE test.a > Int32(5))]] [c:UInt32, sum(test.a):UInt64;N, count(DISTINCT test.a) FILTER (WHERE test.a > Int32(5)):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn common_with_order_by() -> Result<()> {
        let table_scan = test_table_scan()?;

        // SUM(a ORDER BY a)
        let expr = Expr::AggregateFunction(AggregateFunction::new_udf(
            sum_udaf(),
            vec![col("a")],
            false,
            None,
            vec![col("a").sort(true, false)],
            None,
        ));
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[sum(test.a) ORDER BY [test.a ASC NULLS LAST], count(DISTINCT test.b)]] [c:UInt32, sum(test.a) ORDER BY [test.a ASC NULLS LAST]:UInt64;N, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_with_order_by() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(DISTINCT a ORDER BY a)
        let expr = count_udaf()
            .call(vec![col("a")])
            .distinct()
            .order_by(vec![col("a").sort(true, false)])
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![sum(col("a")), expr])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[sum(test.a), count(DISTINCT test.a) ORDER BY [test.a ASC NULLS LAST]]] [c:UInt32, sum(test.a):UInt64;N, count(DISTINCT test.a) ORDER BY [test.a ASC NULLS LAST]:Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn aggregate_with_filter_and_order_by() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(DISTINCT a ORDER BY a) FILTER (WHERE a > 5)
        let expr = count_udaf()
            .call(vec![col("a")])
            .distinct()
            .filter(col("a").gt(lit(5)))
            .order_by(vec![col("a").sort(true, false)])
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![sum(col("a")), expr])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[sum(test.a), count(DISTINCT test.a) FILTER (WHERE test.a > Int32(5)) ORDER BY [test.a ASC NULLS LAST]]] [c:UInt32, sum(test.a):UInt64;N, count(DISTINCT test.a) FILTER (WHERE test.a > Int32(5)) ORDER BY [test.a ASC NULLS LAST]:Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn count_star_and_distinct_without_groupby() -> Result<()> {
        let table_scan = test_table_scan_utf8_b()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                Vec::<Expr>::new(),
                vec![count_star(), count_distinct(col("b"))],
            )?
            .build()?;

        // Should work. Without a group by the outer aggregate sees no rows at
        // all for an empty input, so the projection has to turn the NULL that
        // `sum` reports there back into the 0 `count` reports.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(Int64(1)), count(alias1) AS count(DISTINCT test.b) [count(Int64(1)):Int64, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[]], aggr=[[sum(alias2), count(alias1)]] [sum(alias2):Int64;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.b AS alias1]], aggr=[[count(Int64(1)) AS alias2]] [alias1:Utf8, alias2:Int64]
              TableScan: test [a:UInt32, b:Utf8, c:UInt32]
        "
        )
    }

    #[test]
    fn count_star_min_max_sum_and_distinct_with_groupby() -> Result<()> {
        let table_scan = test_table_scan_utf8_b()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![
                    count_star(),
                    count_distinct(col("b")),
                    min(col("c")),
                    max(col("c")),
                    sum(col("c")),
                ],
            )?
            .build()?;

        // Should work: this is the shape a `count(*)` alongside a
        // `count(DISTINCT ..)`, a min, a max and a sum produces
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(Int64(1)), count(alias1) AS count(DISTINCT test.b), min(alias3) AS min(test.c), max(alias4) AS max(test.c), sum(alias5) AS sum(test.c) [a:UInt32, count(Int64(1)):Int64, count(DISTINCT test.b):Int64, min(test.c):UInt32;N, max(test.c):UInt32;N, sum(test.c):UInt64;N]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias2), count(alias1), min(alias3), max(alias4), sum(alias5)]] [a:UInt32, sum(alias2):Int64;N, count(alias1):Int64, min(alias3):UInt32;N, max(alias4):UInt32;N, sum(alias5):UInt64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[count(Int64(1)) AS alias2, min(test.c) AS alias3, max(test.c) AS alias4, sum(test.c) AS alias5]] [a:UInt32, alias1:Utf8, alias2:Int64, alias3:UInt32;N, alias4:UInt32;N, alias5:UInt64;N]
              TableScan: test [a:UInt32, b:Utf8, c:UInt32]
        "
        )
    }

    #[test]
    fn count_with_filter_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(a) FILTER (WHERE a > 5)
        let expr = count_udaf()
            .call(vec![col("a")])
            .filter(col("a").gt(lit(5)))
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing: the filter would have to be applied per input row, but
        // the inner aggregate has already collapsed them
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[count(test.a) FILTER (WHERE test.a > Int32(5)), count(DISTINCT test.b)]] [c:UInt32, count(test.a) FILTER (WHERE test.a > Int32(5)):Int64, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn count_with_order_by_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(a ORDER BY a)
        let expr = count_udaf()
            .call(vec![col("a")])
            .order_by(vec![col("a").sort(true, false)])
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[count(test.a) ORDER BY [test.a ASC NULLS LAST], count(DISTINCT test.b)]] [c:UInt32, count(test.a) ORDER BY [test.a ASC NULLS LAST]:Int64, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn count_without_function_registry_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![count_star(), count_distinct(col("b"))])?
            .build()?;

        // Do nothing: without a registry the rule cannot resolve the `sum` the
        // outer phase needs
        let rule: Arc<dyn OptimizerRule + Send + Sync> =
            Arc::new(SingleDistinctToGroupBy::new());
        assert_optimized_plan_eq_display_indent_snapshot!(
            rule,
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[count(Int64(1)), count(DISTINCT test.b)]] [a:UInt32, count(Int64(1)):Int64, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        ",
        )
    }

    /// `test` with the `distinct` column `b` of the given type, a `Float64`
    /// `c`, the type coercion gives an `avg` over any integer or float, and a
    /// decimal `d`.
    fn test_table_scan_avg(b: DataType) -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", b, true),
            Field::new("c", DataType::Float64, true),
            Field::new("d", DataType::Decimal128(10, 2), true),
        ]);
        table_scan(Some("test"), &schema, None)?.build()
    }

    #[test]
    fn avg_count_star_and_distinct_with_groupby() -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Utf8)?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![
                    sum(col("a")),
                    count_star(),
                    avg(col("c")),
                    count_distinct(col("b")),
                ],
            )?
            .build()?;

        // Should work: the `avg` becomes the sum of the partial sums over the
        // sum of the partial counts the inner group-by produces per
        // (test.a, test.b) pair
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a, sum(alias2) AS sum(test.a), CASE WHEN sum(alias3) IS NOT NULL THEN sum(alias3) ELSE Int64(0) END AS count(Int64(1)), sum(alias4) / CAST(sum(alias5) AS Float64) AS avg(test.c), count(alias1) AS count(DISTINCT test.b) [a:UInt32, sum(test.a):UInt64;N, count(Int64(1)):Int64, avg(test.c):Float64;N, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias2), sum(alias3), sum(alias4), sum(alias5), count(alias1)]] [a:UInt32, sum(alias2):UInt64;N, sum(alias3):Int64;N, sum(alias4):Float64;N, sum(alias5):Int64;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[sum(test.a) AS alias2, count(Int64(1)) AS alias3, sum(test.c) AS alias4, count(test.c) AS alias5]] [a:UInt32, alias1:Utf8;N, alias2:UInt64;N, alias3:Int64, alias4:Float64;N, alias5:Int64]
              TableScan: test [a:UInt32, b:Utf8;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn avg_and_distinct_without_groupby() -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Utf8)?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                Vec::<Expr>::new(),
                vec![avg(col("c")), count_distinct(col("b"))],
            )?
            .build()?;

        // Should work. Over an empty input both outer sums are NULL, and so is
        // their quotient, as `avg` reports
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: sum(alias2) / CAST(sum(alias3) AS Float64) AS avg(test.c), count(alias1) AS count(DISTINCT test.b) [avg(test.c):Float64;N, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[]], aggr=[[sum(alias2), sum(alias3), count(alias1)]] [sum(alias2):Float64;N, sum(alias3):Int64;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.b AS alias1]], aggr=[[sum(test.c) AS alias2, count(test.c) AS alias3]] [alias1:Utf8;N, alias2:Float64;N, alias3:Int64]
              TableScan: test [a:UInt32, b:Utf8;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn avg_decimal_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Utf8)?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![avg(col("d")), count_distinct(col("b"))],
            )?
            .build()?;

        // Do nothing: a decimal `avg` scales its result past the quotient of
        // its sum and count
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[avg(test.d), count(DISTINCT test.b)]] [a:UInt32, avg(test.d):Decimal128(14, 6);N, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:Utf8;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn avg_with_filter_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Utf8)?;

        // avg(c) FILTER (WHERE a > 5)
        let expr = avg_udaf()
            .call(vec![col("c")])
            .filter(col("a").gt(lit(5)))
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[avg(test.c) FILTER (WHERE test.a > Int32(5)), count(DISTINCT test.b)]] [a:UInt32, avg(test.c) FILTER (WHERE test.a > Int32(5)):Float64;N, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:Utf8;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn avg_and_distinct_over_a_natively_supported_type() -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Int64)?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![avg(col("c")), count_distinct(col("b"))],
            )?
            .build()?;

        // Should not work: like a `count`, an `avg` joins the rewrite only when
        // the distinct aggregate lacks its own `GroupsAccumulator`
        assert_optimized_plan_equal!(
            plan,
            @r"
        Aggregate: groupBy=[[test.a]], aggr=[[avg(test.c), count(DISTINCT test.b)]] [a:UInt32, avg(test.c):Float64;N, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:Int64;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn distinct_and_common_over_a_natively_supported_type_in_parallel() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should work: in several partitions the native accumulator would
        // merge each group's distinct values in the one partition the group
        // hashes to, where the inner group-by spreads them over all of them
        assert_optimized_plan_equal!(
            plan,
            partitions = 4,
            @r"
        Projection: test.a, count(alias1) AS count(DISTINCT test.b), CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(test.c) [a:UInt32, count(DISTINCT test.b):Int64, count(test.c):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[count(alias1), sum(alias2)]] [a:UInt32, count(alias1):Int64, sum(alias2):Int64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[count(test.c) AS alias2]] [a:UInt32, alias1:UInt32, alias2:Int64]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn avg_count_star_and_distinct_over_a_natively_supported_type_in_parallel()
    -> Result<()> {
        let table_scan = test_table_scan_avg(DataType::Int64)?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![count_star(), avg(col("c")), count_distinct(col("b"))],
            )?
            .build()?;

        // Should work: the shape of ClickBench Q9
        assert_optimized_plan_equal!(
            plan,
            partitions = 4,
            @r"
        Projection: test.a, CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(Int64(1)), sum(alias3) / CAST(sum(alias4) AS Float64) AS avg(test.c), count(alias1) AS count(DISTINCT test.b) [a:UInt32, count(Int64(1)):Int64, avg(test.c):Float64;N, count(DISTINCT test.b):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias2), sum(alias3), sum(alias4), count(alias1)]] [a:UInt32, sum(alias2):Int64;N, sum(alias3):Float64;N, sum(alias4):Int64;N, count(alias1):Int64]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[count(Int64(1)) AS alias2, sum(test.c) AS alias3, count(test.c) AS alias4]] [a:UInt32, alias1:Int64;N, alias2:Int64, alias3:Float64;N, alias4:Int64]
              TableScan: test [a:UInt32, b:Int64;N, c:Float64;N, d:Decimal128(10, 2);N]
        "
        )
    }

    #[test]
    fn distinct_with_filter_in_parallel_is_not_rewritten() -> Result<()> {
        let table_scan = test_table_scan()?;

        // count(a) FILTER (WHERE a > 5)
        let expr = count_udaf()
            .call(vec![col("a")])
            .filter(col("a").gt(lit(5)))
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], vec![expr, count_distinct(col("b"))])?
            .build()?;

        // Do nothing: running in parallel lifts only the gate on the distinct
        // aggregate, not the refusal of a FILTER
        assert_optimized_plan_equal!(
            plan,
            partitions = 4,
            @r"
        Aggregate: groupBy=[[test.c]], aggr=[[count(test.a) FILTER (WHERE test.a > Int32(5)), count(DISTINCT test.b)]] [c:UInt32, count(test.a) FILTER (WHERE test.a > Int32(5)):Int64, count(DISTINCT test.b):Int64]
          TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn distinct_sum_and_count_in_parallel() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![sum_distinct(col("b")), count(col("c"))],
            )?
            .build()?;

        // Should work: in parallel the gate does not ask the distinct
        // aggregate at all
        assert_optimized_plan_equal!(
            plan,
            partitions = 4,
            @r"
        Projection: test.a, sum(alias1) AS sum(DISTINCT test.b), CASE WHEN sum(alias2) IS NOT NULL THEN sum(alias2) ELSE Int64(0) END AS count(test.c) [a:UInt32, sum(DISTINCT test.b):UInt64;N, count(test.c):Int64]
          Aggregate: groupBy=[[test.a]], aggr=[[sum(alias1), sum(alias2)]] [a:UInt32, sum(alias1):UInt64;N, sum(alias2):Int64;N]
            Aggregate: groupBy=[[test.a, test.b AS alias1]], aggr=[[count(test.c) AS alias2]] [a:UInt32, alias1:UInt32, alias2:Int64]
              TableScan: test [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn sum_decimal_with_distinct_preserves_declared_type() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("d", DataType::Decimal128(5, 2), false),
            Field::new("x", DataType::Int32, false),
        ]);
        let table_scan = table_scan(Some("test"), &schema, None)?.build()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                Vec::<Expr>::new(),
                vec![sum(col("d")), count_distinct(col("x"))],
            )?
            .build()?;

        let rule: Arc<dyn OptimizerRule + Send + Sync> =
            Arc::new(SingleDistinctToGroupBy::new());
        let optimizer = Optimizer::with_rules(vec![rule]);
        let optimized_plan = optimizer
            .optimize(plan.clone(), &RegistryOptimizerContext::new(), |_, _| {})
            .expect("failed to optimize plan");

        assert_eq!(plan.schema().as_ref(), optimized_plan.schema().as_ref());
        Ok(())
    }
}
