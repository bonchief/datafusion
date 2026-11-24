// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `AggregateExec`.

use super::{roundtrip_test, roundtrip_test_with_context};
use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::compute::kernels::sort::SortOptions;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Volatility;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::update_aggr_exprs::OptimizeAggregateOrder;
use datafusion::physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::expressions::{PhysicalSortExpr, col, lit};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    Accumulator, AccumulatorFactoryFunction, AggregateUDF, Signature, SimpleAggregateUDF,
};
use datafusion_functions_aggregate::approx_percentile_cont::approx_percentile_cont_udaf;
use datafusion_functions_aggregate::array_agg::array_agg_udaf;
use datafusion_functions_aggregate::average::avg_udaf;
use datafusion_functions_aggregate::first_last::first_value_udaf;
use datafusion_functions_aggregate::min_max::min_udaf;
use datafusion_functions_aggregate::nth_value::nth_value_udaf;
use datafusion_functions_aggregate::string_agg::string_agg_udaf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_proto::physical_plan::{AsExecutionPlan, DefaultPhysicalExtensionCodec};
use datafusion_proto::protobuf;
use datafusion_proto::protobuf::PhysicalPlanNode;
use prost::Message;
use std::sync::Arc;
use std::vec;

#[test]
fn roundtrip_aggregate() -> Result<()> {
    let field_a = Field::new("a", DataType::Int64, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));

    let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
        vec![(col("a", &schema)?, "unused".to_string())];

    let avg_expr = AggregateExprBuilder::new(avg_udaf(), vec![col("b", &schema)?])
        .schema(Arc::clone(&schema))
        .alias("AVG(b)")
        .build()?;
    let nth_expr =
        AggregateExprBuilder::new(nth_value_udaf(), vec![col("b", &schema)?, lit(1u64)])
            .schema(Arc::clone(&schema))
            .alias("NTH_VALUE(b, 1)")
            .build()?;
    let str_agg_expr =
        AggregateExprBuilder::new(string_agg_udaf(), vec![col("b", &schema)?, lit(1u64)])
            .schema(Arc::clone(&schema))
            .alias("NTH_VALUE(b, 1)")
            .build()?;

    let test_cases = vec![
        // AVG
        vec![Arc::new(avg_expr)],
        // NTH_VALUE
        vec![Arc::new(nth_expr)],
        // STRING_AGG
        vec![Arc::new(str_agg_expr)],
    ];

    for aggregates in test_cases {
        let schema = schema.clone();
        roundtrip_test(Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            PhysicalGroupBy::new_single(groups.clone()),
            aggregates,
            vec![None],
            Arc::new(EmptyExec::new(schema.clone())),
            schema,
        )?))?;
    }

    Ok(())
}

#[test]
fn roundtrip_aggregate_preserves_optimizer_schema_and_reversed_state() -> Result<()> {
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("b", DataType::Int64, true)]));
    let input_ordering = LexOrdering::new(vec![PhysicalSortExpr {
        expr: col("b", &input_schema)?,
        options: SortOptions::new(true, true),
    }])
    .expect("single sort expression should form an ordering");
    let input: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(
        input_ordering,
        Arc::new(EmptyExec::new(Arc::clone(&input_schema))),
    ));
    let original_name = "first_value(b) ORDER BY [b ASC NULLS LAST]";
    let aggregate_expr = Arc::new(
        AggregateExprBuilder::new(first_value_udaf(), vec![col("b", &input_schema)?])
            .order_by(vec![PhysicalSortExpr {
                expr: col("b", &input_schema)?,
                options: SortOptions::new(false, false),
            }])
            .schema(Arc::clone(&input_schema))
            .alias(original_name)
            .build()?,
    );
    let aggregate = AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new(vec![], vec![], vec![], false),
        vec![aggregate_expr],
        vec![None],
        input,
        input_schema,
    )?;

    let optimized = OptimizeAggregateOrder::new()
        .optimize(Arc::new(aggregate), &ConfigOptions::new())?;
    let optimized_aggregate = optimized
        .downcast_ref::<AggregateExec>()
        .expect("expected optimized AggregateExec");
    assert_eq!(optimized.schema().field(0).name(), original_name);
    assert_eq!(
        optimized_aggregate.aggr_expr()[0].name(),
        "last_value(b) ORDER BY [b DESC NULLS FIRST]"
    );
    assert!(optimized_aggregate.aggr_expr()[0].is_reversed());

    let codec = DefaultPhysicalExtensionCodec {};
    let node = PhysicalPlanNode::try_from_physical_plan(Arc::clone(&optimized), &codec)?;
    let node = PhysicalPlanNode::decode(node.encode_to_vec().as_slice())
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let ctx = SessionContext::new();
    let decoded = node.try_into_physical_plan(&ctx.task_ctx(), &codec)?;
    let decoded_aggregate = decoded
        .downcast_ref::<AggregateExec>()
        .expect("expected decoded AggregateExec");

    assert_eq!(optimized.schema(), decoded.schema());
    assert!(decoded_aggregate.aggr_expr()[0].is_reversed());
    Ok(())
}

#[test]
fn decode_aggregate_without_output_schema() -> Result<()> {
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("b", DataType::Int64, true)]));
    let aggregate_expr = Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![col("b", &input_schema)?])
            .schema(Arc::clone(&input_schema))
            .alias("SUM(b)")
            .build()?,
    );
    let plan: Arc<dyn ExecutionPlan> = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new(vec![], vec![], vec![], false),
        vec![aggregate_expr],
        vec![None],
        Arc::new(EmptyExec::new(Arc::clone(&input_schema))),
        input_schema,
    )?);

    let codec = DefaultPhysicalExtensionCodec {};
    let mut node = PhysicalPlanNode::try_from_physical_plan(Arc::clone(&plan), &codec)?;
    let Some(protobuf::physical_plan_node::PhysicalPlanType::Aggregate(aggregate)) =
        node.physical_plan_type.as_mut()
    else {
        panic!("expected AggregateExecNode");
    };
    assert!(aggregate.schema.take().is_some());

    let ctx = SessionContext::new();
    let decoded = node.try_into_physical_plan(&ctx.task_ctx(), &codec)?;
    assert_eq!(plan.schema(), decoded.schema());
    Ok(())
}

#[test]
fn roundtrip_aggregate_with_limit() -> Result<()> {
    let field_a = Field::new("a", DataType::Int64, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));

    let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
        vec![(col("a", &schema)?, "unused".to_string())];

    let aggregates = vec![
        AggregateExprBuilder::new(min_udaf(), vec![col("b", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("MIN(b)")
            .build()
            .map(Arc::new)?,
    ];

    let agg = AggregateExec::try_new(
        AggregateMode::Final,
        PhysicalGroupBy::new_single(groups.clone()),
        aggregates,
        vec![None],
        Arc::new(EmptyExec::new(schema.clone())),
        schema,
    )?;
    let agg = agg
        .try_optimize_topk(12, "MIN(b)", SortOptions::new(false, false))
        .unwrap()
        .data;
    roundtrip_test(Arc::new(agg))
}

#[test]
fn roundtrip_aggregate_with_approx_pencentile_cont() -> Result<()> {
    let field_a = Field::new("a", DataType::Int64, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));

    let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
        vec![(col("a", &schema)?, "unused".to_string())];

    let aggregates = vec![
        AggregateExprBuilder::new(
            approx_percentile_cont_udaf(),
            vec![col("b", &schema)?, lit(0.5)],
        )
        .schema(Arc::clone(&schema))
        .alias("APPROX_PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b)")
        .build()
        .map(Arc::new)?,
    ];

    let agg = AggregateExec::try_new(
        AggregateMode::Final,
        PhysicalGroupBy::new_single(groups.clone()),
        aggregates,
        vec![None],
        Arc::new(EmptyExec::new(schema.clone())),
        schema,
    )?;
    roundtrip_test(Arc::new(agg))
}

#[test]
fn roundtrip_aggregate_with_sort() -> Result<()> {
    let field_a = Field::new("a", DataType::Int64, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));

    let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
        vec![(col("a", &schema)?, "unused".to_string())];
    let sort_exprs = vec![PhysicalSortExpr {
        expr: col("b", &schema)?,
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    }];

    let aggregates = vec![
        AggregateExprBuilder::new(array_agg_udaf(), vec![col("b", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("ARRAY_AGG(b)")
            .order_by(sort_exprs)
            .build()
            .map(Arc::new)?,
    ];

    let agg = AggregateExec::try_new(
        AggregateMode::Final,
        PhysicalGroupBy::new_single(groups.clone()),
        aggregates,
        vec![None],
        Arc::new(EmptyExec::new(schema.clone())),
        schema,
    )?;
    roundtrip_test(Arc::new(agg))
}

#[test]
fn roundtrip_aggregate_udaf() -> Result<()> {
    let field_a = Field::new("a", DataType::Int64, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));

    #[derive(Debug)]
    struct Example;
    impl Accumulator for Example {
        fn state(&mut self) -> Result<Vec<ScalarValue>> {
            Ok(vec![ScalarValue::Int64(Some(0))])
        }

        fn update_batch(&mut self, _values: &[ArrayRef]) -> Result<()> {
            Ok(())
        }

        fn merge_batch(&mut self, _states: &[ArrayRef]) -> Result<()> {
            Ok(())
        }

        fn evaluate(&mut self) -> Result<ScalarValue> {
            Ok(ScalarValue::Int64(Some(0)))
        }

        fn size(&self) -> usize {
            0
        }
    }

    let return_type = DataType::Int64;
    let accumulator: AccumulatorFactoryFunction = Arc::new(|_| Ok(Box::new(Example)));

    let udaf = AggregateUDF::from(SimpleAggregateUDF::new_with_signature(
        "example",
        Signature::exact(vec![DataType::Int64], Volatility::Immutable),
        return_type,
        accumulator,
        vec![Field::new("value", DataType::Int64, true).into()],
    ));

    let ctx = SessionContext::new();
    ctx.register_udaf(udaf.clone());

    let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
        vec![(col("a", &schema)?, "unused".to_string())];

    let aggregates = vec![
        AggregateExprBuilder::new(Arc::new(udaf), vec![col("b", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("example_agg")
            .build()
            .map(Arc::new)?,
    ];

    roundtrip_test_with_context(
        Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            PhysicalGroupBy::new_single(groups.clone()),
            aggregates,
            vec![None],
            Arc::new(EmptyExec::new(schema.clone())),
            schema,
        )?),
        &ctx,
    )
}

/// A partial aggregation that carries its group hash, the exchange that routes
/// by it and the final aggregation that reads it keep their flags across the
/// wire, and the partial's schema keeps exactly one hash column.
#[test]
fn roundtrip_carried_group_hash() -> Result<()> {
    use datafusion::physical_expr::Partitioning;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion_functions_aggregate::count::count_udaf;

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("n", DataType::Int64, true),
    ]));
    let group_by = PhysicalGroupBy::new_single(vec![
        (col("name", &schema)?, "name".to_string()),
        (col("n", &schema)?, "n".to_string()),
    ]);
    let count = Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![col("n", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("count(n)")
            .build()?,
    );
    let partial = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Partial,
            group_by.clone(),
            vec![Arc::clone(&count)],
            vec![None],
            Arc::new(EmptyExec::new(Arc::clone(&schema))),
            Arc::clone(&schema),
        )?
        .with_group_hash()?,
    );
    let partial_fields = partial.schema().fields().len();
    let on = vec![
        col("name", &partial.schema())?,
        col("n", &partial.schema())?,
    ];
    let exchange = Arc::new(
        RepartitionExec::try_new(partial, Partitioning::Hash(on, 4))?
            .with_carried_group_hash()?
            .with_forward_views(),
    );
    let final_agg = Arc::new(
        AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            group_by.as_final(),
            vec![count],
            vec![None],
            exchange,
            Arc::clone(&schema),
        )?
        .with_read_group_hash()?,
    );

    let ctx = SessionContext::new();
    let decoded = super::roundtrip_test_and_return(
        final_agg,
        &ctx,
        &DefaultPhysicalExtensionCodec {},
        &datafusion_proto::physical_plan::DefaultPhysicalProtoConverter {},
    )?;
    let final_agg = decoded
        .downcast_ref::<AggregateExec>()
        .expect("decoded final aggregation");
    assert!(final_agg.carries_group_hash());
    let exchange = final_agg
        .input()
        .downcast_ref::<RepartitionExec>()
        .expect("decoded exchange");
    assert!(exchange.carries_group_hash());
    assert!(exchange.forwards_views());
    let partial = exchange
        .input()
        .downcast_ref::<AggregateExec>()
        .expect("decoded partial aggregation");
    assert!(partial.carries_group_hash());
    assert_eq!(partial.schema().fields().len(), partial_fields);
    Ok(())
}
