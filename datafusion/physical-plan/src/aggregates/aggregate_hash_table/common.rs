// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::BooleanBufferBuilder;
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, new_empty_array, new_null_array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::cmp::distinct;
use arrow::compute::{filter, filter_record_batch, prep_null_mask_filter};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::{AggregateMetrics, EmitTo, GroupsAccumulator};
use datafusion_physical_expr::GroupsAccumulatorAdapter;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use log::debug;

use crate::PhysicalExpr;
use crate::aggregates::aggregate_argument::AggregateArgument;
use crate::aggregates::group_values::{
    AccumulatorPhase, AggregateAccumulatorMetrics, AggregateArgumentMetrics,
    GroupByMetrics, GroupValues, new_group_values,
};
use crate::aggregates::order::GroupClustering;
use crate::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy, aggregate_expressions,
    evaluate_group_by, group_id_array, max_duplicate_ordinal,
};
use crate::metrics;

use super::AggregateTableMetrics;

/// Marker for raw rows -> partial state aggregation.
pub(in crate::aggregates) struct PartialMarker;
/// Marker for raw rows -> final value aggregation.
pub(in crate::aggregates) struct SingleMarker;
/// Marker for partial state -> partial state aggregation.
pub(in crate::aggregates) struct PartialReduceMarker;
/// Marker for raw rows -> partial state conversion without aggregation.
pub(in crate::aggregates) struct PartialSkipMarker;
/// Marker for partial state -> final value aggregation.
pub(in crate::aggregates) struct FinalMarker;

/// Create an accumulator for `agg_expr` -- a [`GroupsAccumulator`] if
/// that is supported by the aggregate, or a
/// [`GroupsAccumulatorAdapter`] if not.
pub(in crate::aggregates) fn create_group_accumulator(
    agg_expr: &Arc<AggregateFunctionExpr>,
    metrics: Arc<dyn AggregateMetrics>,
) -> Result<Box<dyn GroupsAccumulator>> {
    if agg_expr.groups_accumulator_supported() {
        agg_expr.create_groups_accumulator_with_metrics(metrics)
    } else {
        // Note in the log when the slow path is used
        debug!(
            "Creating GroupsAccumulatorAdapter for {}: {agg_expr:?}",
            agg_expr.name()
        );
        let agg_expr = Arc::clone(agg_expr);
        let mut adapter =
            GroupsAccumulatorAdapter::new(move || agg_expr.create_accumulator());
        adapter.set_metrics(metrics);
        Ok(Box::new(adapter))
    }
}

/// Grouped hash table shared by the partial and final paths.
///
/// While building, it consumes input batches and updates group / accumulator
/// state. While outputting, it incrementally drains that state into output
/// batches.
///
/// # Logical and Physical Model
///
/// Logically, this is a hash table that maps { group keys -> accumulator states }
/// For example, `AVG(v) GROUP BY k` stores one entry per `k`, where each
/// entry owns the `sum(v)` and `count(v)` state needed to compute the final
/// average.
///
/// Physically, the group keys and accumulators are backed by [`GroupValues`] and
/// [`GroupsAccumulator`]. Both use columnar storage so aggregation can stay
/// vectorized.
///
/// # Marker Type
/// `AggrMode` selects the aggregate semantics.
///
/// e.g. `AggregateHashTable::<PartialMarker>::new(...)` creates an aggregate hash table
/// for the partial hash aggregate stage, the input schema is raw rows and output
/// schema is intermediate states.
///
/// It is a zero-sized compile-time marker, so each stage keeps its update logic
/// in a separate impl block, to make the behavior difference explicit.
pub(in crate::aggregates) struct AggregateHashTable<AggrMode> {
    /// Grouping and accumulator-specific timing metrics.
    pub(super) group_by_metrics: GroupByMetrics,

    /// Per-aggregate timing metrics for evaluating aggregate arguments.
    pub(super) aggregate_argument_metrics: AggregateArgumentMetrics,

    /// Per-aggregate timing metrics for accumulator operations.
    pub(super) aggregate_accumulator_metrics: Arc<AggregateAccumulatorMetrics>,

    /// Optional internal metrics owned by each aggregate expression.
    pub(super) aggregate_submetrics: Vec<Arc<dyn AggregateMetrics>>,

    /// Raw input schema, used to evaluate expressions and synthesize empty
    /// grouping-set rows.
    pub(super) input_schema: SchemaRef,

    /// Output schema: group columns followed by aggregate state or final values.
    pub(super) output_schema: SchemaRef,

    /// Intermediate-state schema used when memory pressure requires the table
    /// to spill its current state.
    pub(super) state_schema: SchemaRef,

    /// Maximum rows per emitted output batch, from config `batch_size`.
    pub(super) batch_size: usize,

    /// Lifecycle-specific state: building stage / outputting stage.
    pub(super) state: AggregateHashTableState,

    /// Set by tables that are used for one set of groups after another (see
    /// `AggregateHashTable::<FinalMarker>::restart`): when the output is
    /// materialized, the emptied buffer is kept in `recycled_buffer` with
    /// its allocations instead of being dropped.
    pub(super) recycle_buffer: bool,
    pub(super) recycled_buffer: Option<AggregateHashTableBuffer>,

    pub(super) _mode: PhantomData<AggrMode>,
}

/// Methods shared by all aggregate hash table modes.
impl<AggrMode> AggregateHashTable<AggrMode> {
    pub(super) fn new_with_filters(
        agg: &AggregateExec,
        partition: usize,
        output_schema: SchemaRef,
        state_schema: SchemaRef,
        batch_size: usize,
        filters: Vec<Option<Arc<dyn PhysicalExpr>>>,
    ) -> Result<Self> {
        Self::new_for_input(
            agg,
            agg.input().schema(),
            &agg.mode,
            partition,
            output_schema,
            state_schema,
            batch_size,
            filters,
        )
    }

    /// Like [`Self::new_with_filters`], for a table whose input is not the
    /// input of `agg`: `input_schema` is the schema of the batches it
    /// aggregates and `mode` decides how they are read, as raw rows or as
    /// partial state. `agg.group_by` must refer to `input_schema`.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new_for_input(
        agg: &AggregateExec,
        input_schema: SchemaRef,
        mode: &AggregateMode,
        partition: usize,
        output_schema: SchemaRef,
        state_schema: SchemaRef,
        batch_size: usize,
        filters: Vec<Option<Arc<dyn PhysicalExpr>>>,
    ) -> Result<Self> {
        if batch_size == 0 {
            return internal_err!("AggregateHashTable requires config batch_size >= 1");
        }

        let metrics = AggregateTableMetrics::new(agg, partition);
        // `mode` is a parameter rather than `agg.mode`: a bucket's table reads
        // the partial state of an aggregation that is not its own.
        let aggregate_arguments = aggregate_expressions(
            agg.aggr_expr(),
            mode,
            agg.group_by().num_group_exprs(),
        )?;
        let accumulators: Vec<_> = agg
            .aggr_expr()
            .iter()
            .zip(aggregate_arguments)
            .zip(filters)
            .zip(metrics.submetrics.iter())
            .map(|(((agg_expr, arguments), filter), submetrics)| {
                let accumulator =
                    create_group_accumulator(agg_expr, Arc::clone(submetrics))?;
                Ok(HashAggregateAccumulator::new(
                    Arc::clone(agg_expr),
                    arguments,
                    filter,
                    accumulator,
                    Arc::clone(submetrics),
                ))
            })
            .collect::<Result<_>>()?;

        let group_schema = agg.group_by().group_schema(&input_schema)?;
        let group_values = new_group_values(group_schema, &GroupClustering::None)?;

        Ok(Self {
            group_by_metrics: metrics.group_by,
            aggregate_argument_metrics: metrics.aggregate_arguments,
            aggregate_accumulator_metrics: metrics.accumulator,
            aggregate_submetrics: metrics.submetrics,
            input_schema,
            output_schema,
            state_schema,
            batch_size,
            state: AggregateHashTableState::Building(AggregateHashTableBuffer {
                group_by: Arc::clone(agg.group_by()),
                group_values,
                batch_group_indices: Default::default(),
                accumulators,
            }),
            recycle_buffer: false,
            recycled_buffer: None,
            _mode: PhantomData,
        })
    }

    /// See comments in [`EvaluatedAggregateBatch`]
    pub(super) fn evaluate_batch(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<EvaluatedAggregateBatch> {
        let state = self.state.building_mut();
        // Outer vec: one per grouping set; inner vec: group-by expressions.
        let grouping_set_args = self
            .group_by_metrics
            .time_group_key_preparation(|| evaluate_group_by(&state.group_by, batch))?;

        // The evaluated args for each accumulator.
        let accumulator_args = self.group_by_metrics.time_aggregate_arguments(|| {
            state
                .accumulators
                .iter_mut()
                .enumerate()
                .map(|(idx, acc)| {
                    self.aggregate_argument_metrics
                        .time(idx, || acc.evaluate_compacted_args(batch))
                })
                .collect::<Result<Vec<_>>>()
        })?;

        Ok(EvaluatedAggregateBatch {
            grouping_set_args,
            accumulator_args,
        })
    }

    /// Aggregates one input batch after selecting the mode-specific accumulator
    /// operation.
    ///
    /// Each aggregation mode chooses a different `aggregate_fn` according to its
    /// semantics. For example, partial aggregation takes raw inputs, and update them
    /// into stored partial states, so [`GroupsAccumulator::update_batch`] is used.
    ///
    /// With `collapse`, a single grouping set's batch is interned by runs (see
    /// [`intern_runs`]) and the rows that took their predecessor's group are
    /// added to the counter.
    pub(super) fn aggregate_batch_inner(
        &mut self,
        batch: &RecordBatch,
        aggregate_fn: AggregateBatchFn,
        accumulator_phase: AccumulatorPhase,
        collapse: Option<&metrics::Count>,
    ) -> Result<()> {
        let evaluated_batch = self.evaluate_batch(batch)?;
        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let group_by_metrics = self.group_by_metrics.clone();
        let state = self.state.building_mut();
        let collapse = collapse.filter(|_| evaluated_batch.grouping_set_args.len() == 1);

        for group_values in &evaluated_batch.grouping_set_args {
            group_by_metrics.time_group_key_preparation(|| match collapse {
                Some(collapsed) => {
                    let rows = intern_runs(
                        state.group_values.as_mut(),
                        group_values,
                        &mut state.batch_group_indices,
                    )?;
                    collapsed.add(rows);
                    Ok(())
                }
                None => state
                    .group_values
                    .intern(group_values, &mut state.batch_group_indices),
            })?;

            // Register groups from the full input. Each filtered aggregate compacts
            // this row-aligned vector independently immediately before its update.
            let group_indices = &state.batch_group_indices;
            let total_num_groups = state.group_values.len();
            group_by_metrics.time_aggregation(|| {
                for (idx, (acc, values)) in state
                    .accumulators
                    .iter_mut()
                    .zip(evaluated_batch.accumulator_args.iter())
                    .enumerate()
                {
                    accumulator_metrics.time(idx, accumulator_phase, || {
                        aggregate_fn(acc, values, group_indices, total_num_groups)
                    })?;
                }
                Ok::<(), datafusion_common::DataFusionError>(())
            })?;
        }

        Ok(())
    }

    /// Materializes the full output once, then returns it downstream incrementally
    /// by slicing it into `batch_size` chunks.
    ///
    /// Each aggregation mode chooses a different `materialize_accumulator_fn`
    /// according to its semantics. For example, partial aggregation emits
    /// partial states to feed the final stage, so it uses [`GroupsAccumulator::state`].
    ///
    /// This is a temporary solution until blocked state management is implemented:
    /// Issue: <https://github.com/apache/datafusion/issues/7065>
    pub(super) fn next_output_batch_inner(
        &mut self,
        materialize_accumulator_fn: MaterializeAccumulatorFn,
        accumulator_phase: AccumulatorPhase,
    ) -> Result<Option<RecordBatch>> {
        let output_schema = Arc::clone(&self.output_schema);
        let batch_size = self.batch_size;
        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);

        let mut output =
            match std::mem::replace(&mut self.state, AggregateHashTableState::Done) {
                AggregateHashTableState::Outputting(mut state) => {
                    if state.group_values.is_empty() {
                        return Ok(None);
                    }

                    // Accumulator output consumes internal state. Materialize all
                    // groups once, then slice the materialized batch on later polls.
                    let emit_to = EmitTo::All;
                    let columns = self.group_by_metrics.time_emitting(|| {
                        let mut columns = state.group_values.emit(emit_to)?;
                        for (idx, acc) in state.accumulators.iter_mut().enumerate() {
                            columns.extend(accumulator_metrics.time(
                                idx,
                                accumulator_phase,
                                || materialize_accumulator_fn(acc, emit_to),
                            )?);
                        }
                        Ok::<_, datafusion_common::DataFusionError>(columns)
                    })?;

                    let batch = RecordBatch::try_new(output_schema, columns)?;
                    debug_assert!(batch.num_rows() > 0);
                    if self.recycle_buffer {
                        // Keep the hash table's capacity for as many groups
                        // as it just held: the next set is likely as large.
                        state.group_values.clear_shrink(batch.num_rows());
                        state.batch_group_indices.clear();
                        self.recycled_buffer = Some(state);
                    }
                    MaterializedAggregateOutput::new(batch)
                }
                AggregateHashTableState::OutputtingMaterialized(output) => output,
                AggregateHashTableState::Done => return Ok(None),
                AggregateHashTableState::Building(_) => {
                    return internal_err!(
                        "next_output_batch must be called in the outputting state"
                    );
                }
            };

        let batch = output.next_batch(batch_size);
        if output.is_exhausted() {
            self.state = AggregateHashTableState::Done;
        } else {
            self.state = AggregateHashTableState::OutputtingMaterialized(output);
        }
        Ok(batch)
    }

    pub(in crate::aggregates) fn memory_size(&self) -> usize {
        let buffer_size = |state: &AggregateHashTableBuffer| {
            let acc = state
                .accumulators
                .iter()
                .map(|acc| acc.accumulator.size())
                .sum::<usize>();

            acc + state.group_values.size() + state.batch_group_indices.allocated_size()
        };
        let recycled = self.recycled_buffer.as_ref().map_or(0, buffer_size);
        recycled
            + match &self.state {
                AggregateHashTableState::Building(state)
                | AggregateHashTableState::Outputting(state) => buffer_size(state),
                AggregateHashTableState::OutputtingMaterialized(output) => {
                    output.memory_size()
                }
                AggregateHashTableState::Done => 0,
            }
    }

    /// Returns the number of distinct groups accumulated so far.
    pub(in crate::aggregates) fn building_group_count(&self) -> usize {
        self.state.building().group_values.len()
    }

    /// Takes every intermediate aggregate state and resets the table so it can
    /// continue accumulating raw input.
    ///
    /// Unlike normal single aggregation output, this materializes intermediate
    /// states rather than final values. The states can therefore be merged after
    /// spilling without finalizing the same group more than once.
    pub(in crate::aggregates) fn take_state_batch(
        &mut self,
    ) -> Result<Option<RecordBatch>> {
        self.take_state_batch_inner(false, 0)
    }

    /// Like [`Self::take_state_batch`], but keeps the table's capacity for as
    /// many groups as it held, for a table that is filled again right away.
    pub(in crate::aggregates) fn take_state_batch_keep_capacity(
        &mut self,
    ) -> Result<Option<RecordBatch>> {
        self.take_state_batch_inner(true, 0)
    }

    /// Reserves fixed row capacity without changing the number of groups.
    pub(in crate::aggregates) fn reserve_groups(&mut self, capacity: usize) {
        let state = self.state.building_mut();
        state.group_values.reserve_groups(capacity);
        state
            .batch_group_indices
            .reserve_exact(capacity.saturating_sub(state.batch_group_indices.len()));
        for acc in &mut state.accumulators {
            acc.accumulator.reserve_groups(capacity);
        }
    }

    /// Emits all states and reserves a fixed capacity for the next table.
    pub(in crate::aggregates) fn take_state_batch_with_capacity(
        &mut self,
        capacity: usize,
    ) -> Result<Option<RecordBatch>> {
        self.take_state_batch_inner(false, capacity)
    }

    fn take_state_batch_inner(
        &mut self,
        keep_capacity: bool,
        capacity: usize,
    ) -> Result<Option<RecordBatch>> {
        let state_schema = Arc::clone(&self.state_schema);
        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let group_by_metrics = self.group_by_metrics.clone();
        let state = self.state.building_mut();
        if state.group_values.is_empty() {
            return Ok(None);
        }

        let output = group_by_metrics.time_emitting(|| {
            let mut output = if capacity > 0 {
                state.group_values.emit_with_capacity(capacity)?
            } else {
                state.group_values.emit(EmitTo::All)?
            };
            for (idx, acc) in state.accumulators.iter_mut().enumerate() {
                output.extend(accumulator_metrics.time(
                    idx,
                    AccumulatorPhase::State,
                    || acc.state(EmitTo::All),
                )?);
            }
            Ok::<_, datafusion_common::DataFusionError>(output)
        })?;

        let batch = RecordBatch::try_new(state_schema, output)?;
        debug_assert!(batch.num_rows() > 0);

        // `emit(EmitTo::All)` resets accumulator state. Explicitly shrink the
        // key/index buffers and accumulator scratch too so the memory
        // reservation can be released before the batch is sorted for spilling,
        // unless the table is about to be refilled.
        state.batch_group_indices.clear();
        if capacity > 0 {
            for acc in &mut state.accumulators {
                acc.accumulator.reserve_groups(capacity);
            }
            return Ok(Some(batch));
        }
        state
            .group_values
            .clear_shrink(if keep_capacity { batch.num_rows() } else { 0 });
        state.batch_group_indices.shrink_to_fit();
        if !keep_capacity {
            for acc in &mut state.accumulators {
                acc.shrink_to_fit();
            }
        }

        Ok(Some(batch))
    }

    pub(in crate::aggregates) fn is_building(&self) -> bool {
        matches!(self.state, AggregateHashTableState::Building(_))
    }

    pub(super) fn start_outputting(&mut self) {
        let AggregateHashTableState::Building(mut state) =
            std::mem::replace(&mut self.state, AggregateHashTableState::Done)
        else {
            unreachable!("hash aggregate table is not building")
        };

        state.batch_group_indices = Vec::new();
        self.state = AggregateHashTableState::Outputting(state);
    }

    /// Creates the required empty grouping-set rows when the input is empty.
    ///
    /// For example, this query must still produce one grand-total group even if
    /// `t` has no rows:
    ///
    /// ```sql
    /// SELECT COUNT(v)
    /// FROM t
    /// GROUP BY GROUPING SETS (());
    /// ```
    ///
    /// Accumulators receive zero argument rows and zero group IDs, together with the
    /// full registered group count, so they produce the same state as empty input.
    ///
    /// Only the raw-input tables (partial and single aggregation) call this
    /// method: grouping sets are expanded while consuming raw rows, so the
    /// state-input stages (final and partial-reduce aggregation) receive the
    /// already expanded keys as plain group columns (see
    /// [`PhysicalGroupBy::as_final`]) and never own a grouping set.
    pub(super) fn init_empty_grouping_sets(&mut self) -> Result<()> {
        let group_by_metrics = self.group_by_metrics.clone();
        let state = self.state.building_mut();
        if !state.group_by.has_grouping_set() || !state.group_values.is_empty() {
            return Ok(());
        }

        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let any_interned = group_by_metrics.time_group_key_preparation(|| {
            let max_ordinal = max_duplicate_ordinal(state.group_by.groups());
            let mut ordinals: HashMap<&[bool], usize> =
                HashMap::with_capacity(state.group_by.groups().len());
            let group_schema = state.group_by.group_schema(&self.input_schema)?;
            let n_expr = state.group_by.expr().len();
            let mut any_interned = false;

            for group in state.group_by.groups() {
                let ordinal = {
                    let entry = ordinals.entry(group.as_slice()).or_insert(0);
                    let ordinal = *entry;
                    *entry += 1;
                    ordinal
                };

                if !group.iter().all(|&is_null| is_null) {
                    continue;
                }

                let mut cols: Vec<ArrayRef> = group_schema
                    .fields()
                    .iter()
                    .take(n_expr)
                    .map(|field| new_null_array(field.data_type(), 1))
                    .collect();
                cols.push(group_id_array(group, ordinal, max_ordinal, 1)?);

                state
                    .group_values
                    .intern(&cols, &mut state.batch_group_indices)?;
                any_interned = true;
            }
            Ok::<_, datafusion_common::DataFusionError>(any_interned)
        })?;

        if any_interned {
            let total_groups = state.group_values.len();
            let values = state
                .accumulators
                .iter()
                .map(|acc| {
                    Ok(CompactedAccumulatorArgs {
                        arguments: acc.null_arguments(&self.input_schema, 0)?,
                        selection: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            group_by_metrics.time_aggregation(|| {
                for (idx, (acc, values)) in
                    state.accumulators.iter_mut().zip(values.iter()).enumerate()
                {
                    accumulator_metrics.time(idx, AccumulatorPhase::Update, || {
                        acc.update_batch(values, &[], total_groups)
                    })?;
                }
                Ok::<(), datafusion_common::DataFusionError>(())
            })?;
        }

        Ok(())
    }
}

/// State and argument information for a single Aggregate
///
/// For example, for `SELECT COUNT(x), SUM(y WHERE z > 10) ...`  there would be two
/// `HashAggregateAccumulator`, one each for `COUNT(x)` and `SUM(y WHERE z > 10)`
pub(super) struct HashAggregateAccumulator {
    /// Aggregate expression used to create a fresh accumulator for related
    /// hash tables, such as the partial-skip table.
    aggregate_expr: Arc<AggregateFunctionExpr>,

    /// Arguments to pass to this accumulator.
    ///
    /// Example: `CORR(x, y)` stores two arguments here, while `SUM(x)` stores one.
    arguments: Vec<AggregateArgument>,

    /// Optional `FILTER` expression for this accumulator.
    ///
    /// Example: `SUM(x) FILTER (WHERE x > 10)` stores the `x > 10` predicate.
    filter: Option<Arc<dyn PhysicalExpr>>,

    /// Accumulator state for all groups for one aggregate expression.
    accumulator: Box<dyn GroupsAccumulator>,

    /// Optional internal metrics owned by this aggregate expression.
    submetrics: Arc<dyn AggregateMetrics>,
}

pub(super) type AggregateAccumulator = HashAggregateAccumulator;

/// Function used by [`AggregateHashTable::aggregate_batch_inner`] to update one
/// accumulator with one evaluated input batch.
///
/// Arguments:
/// * accumulator to update.
/// * accumulator's compacted arguments and optional row-aligned selection.
/// * one group index per input row, mapping each row to its interned group.
/// * total number of groups currently interned in that buffer, including newly
///   interned groups.
pub(super) type AggregateBatchFn = fn(
    &mut AggregateAccumulator,
    &CompactedAccumulatorArgs,
    &[usize],
    usize,
) -> Result<()>;

/// Function used by [`AggregateHashTable::next_output_batch_inner`] to
/// materialize one accumulator's output columns.
///
/// Arguments:
/// * accumulator to materialize.
/// * group range to emit from the accumulator.
pub(super) type MaterializeAccumulatorFn =
    fn(&mut AggregateAccumulator, EmitTo) -> Result<Vec<ArrayRef>>;

/// Aggregate arguments compacted according to one aggregate's `FILTER`.
pub(super) struct CompactedAccumulatorArgs {
    /// Argument arrays containing only selected rows. Some aggregate functions take
    /// multiple arguments.
    pub(super) arguments: Vec<ArrayRef>,
    /// Original row-aligned selection used only to compact the matching group IDs.
    pub(super) selection: Option<BooleanArray>,
}

/// Evaluated aggregate arguments that preserve one output row per input row.
pub(super) struct RowAlignedAccumulatorArgs {
    /// Row-aligned argument arrays. Rejected rows are represented as nulls.
    pub(super) arguments: Vec<ArrayRef>,
    /// Original row-aligned filter passed through to state conversion.
    pub(super) filter: Option<BooleanArray>,
}

/// Evaluated all group by keys and accumulator args.
///
/// e.g., `select k+1, sum(v*v) from t group by (k+1)`, this function evaluates
/// `k+1`, `v*v`
pub(super) struct EvaluatedAggregateBatch {
    /// One entry per grouping set; each entry contains all evaluated group key
    /// arrays for the current input batch.
    pub(super) grouping_set_args: Vec<Vec<ArrayRef>>,

    /// Compacted arguments and selections, one entry per aggregate expression.
    pub(super) accumulator_args: Vec<CompactedAccumulatorArgs>,
}

/// Buffer for the aggregate hash table's group keys and accumulator states.
///
/// It accumulates input during aggregation and emits final results during the
/// outputting stage.
///
/// [`GroupValues`] stores the physical group-key layout, while
/// [`GroupsAccumulator`] stores per-group aggregate state.
pub(super) struct AggregateHashTableBuffer {
    /// GROUP BY expressions evaluated for each input batch.
    pub(super) group_by: Arc<PhysicalGroupBy>,

    /// Interned group keys. Accumulator state is stored separately by group index.
    pub(super) group_values: Box<dyn GroupValues>,

    /// Group index for each row in the current input batch.
    ///
    /// Each value indexes into `group_values`, and the same index is used by every
    /// accumulator to update that group's aggregate state.
    pub(super) batch_group_indices: Vec<usize>,

    /// One item per aggregate expression.
    ///
    /// Example: `COUNT(x), SUM(y)` creates two items. Each item owns the input
    /// expressions, optional filter, and accumulator state for all groups.
    pub(super) accumulators: Vec<HashAggregateAccumulator>,
}

pub(super) enum AggregateHashTableState {
    /// Accumulating input rows into group keys and aggregate state.
    Building(AggregateHashTableBuffer),
    /// Emitting results directly from group keys and aggregate state.
    Outputting(AggregateHashTableBuffer),
    /// Materialize all the output results, and then incrementally output in the `OutputtingMaterialized` state.
    ///
    /// Note this is a temporary solution until the `GroupValues` issue is solved:
    /// Issue: <https://github.com/apache/datafusion/issues/23178>
    OutputtingMaterialized(MaterializedAggregateOutput),
    Done,
}

/// Fully evaluated aggregate output and the next row offset to emit.
///
/// Final aggregate evaluation consumes accumulator state, and partial terminal
/// output should not repeatedly renumber group values with `EmitTo::First`.
/// Materialize once and then slice to honor `batch_size` across output polls.
pub(super) struct MaterializedAggregateOutput {
    batch: RecordBatch,
    offset: usize,
}

impl MaterializedAggregateOutput {
    pub(super) fn new(batch: RecordBatch) -> Self {
        Self { batch, offset: 0 }
    }

    pub(super) fn next_batch(&mut self, batch_size: usize) -> Option<RecordBatch> {
        debug_assert!(batch_size > 0);
        if self.is_exhausted() {
            return None;
        }

        let length = batch_size.min(self.batch.num_rows() - self.offset);
        let batch = self.batch.slice(self.offset, length);
        self.offset += length;
        Some(batch)
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.offset >= self.batch.num_rows()
    }

    pub(super) fn memory_size(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}

/// A batch is interned by runs only when at least one row in this many repeats
/// the row before it. Below that, the comparison is all the batch pays: taking
/// the run starts would copy nearly every key for few saved lookups.
const RUN_REPEAT_SHARE: usize = 8;

/// Interns `cols` into `group_values` and writes one group index per row to
/// `groups`, giving a row whose keys all equal the previous row's that row's
/// group without hashing or probing it. Returns how many rows did so.
///
/// Input that is sorted or clustered on a prefix that includes or determines
/// the keys arrives in runs of equal keys; such a row would be a table hit
/// anyway, so the groups and their order are exactly those of
/// [`GroupValues::intern`] over the whole batch. Runs never cross batches: the
/// first row of every batch is interned.
fn intern_runs(
    group_values: &mut dyn GroupValues,
    cols: &[ArrayRef],
    groups: &mut Vec<usize>,
) -> Result<usize> {
    let Some(starts) = run_starts(cols)? else {
        group_values.intern(cols, groups)?;
        return Ok(0);
    };
    let rows = starts.len();
    let heads = starts.count_set_bits();
    let repeats = rows - heads;
    if repeats * RUN_REPEAT_SHARE < rows {
        group_values.intern(cols, groups)?;
        return Ok(0);
    }

    let starts = BooleanArray::new(starts, None);
    let keys = cols
        .iter()
        .map(|col| filter(col, &starts))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    group_values.intern(&keys, groups)?;

    // Spread the run starts' groups over their runs in place, from the back:
    // the start of row `row`'s run sits at `run <= row`, which is not yet
    // overwritten.
    groups.resize(rows, 0);
    let mut run = heads;
    for row in (0..rows).rev() {
        groups[row] = groups[run - 1];
        if starts.value(row) {
            run -= 1;
        }
    }
    Ok(repeats)
}

/// Marks the rows whose keys differ from the previous row's in any column,
/// and always the first row. Returns `None` for fewer than two rows, when a
/// key type is not compared, or when a key's values are long (see
/// [`LONG_KEY_BYTES`]), so the batch is interned as it is.
///
/// Keys are compared with [`distinct`], which treats two nulls as equal, as
/// grouping does, and compares floats by their bits, so it never joins two
/// rows that [`GroupValues`] would keep apart.
fn run_starts(cols: &[ArrayRef]) -> Result<Option<BooleanBuffer>> {
    let rows = cols.first().map_or(0, |col| col.len());
    if rows < 2
        || !cols
            .iter()
            .all(|col| compares_runs(col.data_type()) && !long_values(col))
    {
        return Ok(None);
    }
    let mut differs: Option<BooleanBuffer> = None;
    for col in cols {
        let next = distinct(&col.slice(1, rows - 1), &col.slice(0, rows - 1))?;
        let next = next.values();
        differs = Some(match differs {
            Some(differs) => &differs | next,
            None => next.clone(),
        });
    }
    let mut starts = BooleanBufferBuilder::new(rows);
    starts.append(true);
    if let Some(differs) = differs {
        starts.append_buffer(&differs);
    }
    Ok(Some(starts.finish()))
}

/// Average bytes per value above which a batch's variable-width key column is
/// not compared for runs, and the batch is interned as it is.
///
/// Comparing a row with its predecessor reads the bytes of every row, while
/// it saves a lookup only on the rows that repeat. For short values the
/// compare is a view or offset check and the saved lookups win at any repeat
/// share run collapse accepts; for long values that share a prefix, as URLs
/// do, the compare runs through the bytes and costs more than the lookups it
/// saves. On ClickBench's hits table at 100M rows, collapsing URL (86 bytes on
/// average, a fifth of rows repeating) made the partial aggregate 10% slower,
/// while (UserID, SearchPhrase), whose phrase averages 8 bytes, kept a 13%
/// gain. The length is read from the offsets or the views, before any compare.
const LONG_KEY_BYTES: usize = 32;

/// Whether `col` is a variable-width column whose values average more than
/// [`LONG_KEY_BYTES`] bytes; fixed-width columns never are.
fn long_values(col: &ArrayRef) -> bool {
    let bytes = match col.data_type() {
        DataType::Utf8 => offset_bytes(col.as_string::<i32>().value_offsets()),
        DataType::LargeUtf8 => offset_bytes(col.as_string::<i64>().value_offsets()),
        DataType::Binary => offset_bytes(col.as_binary::<i32>().value_offsets()),
        DataType::LargeBinary => offset_bytes(col.as_binary::<i64>().value_offsets()),
        DataType::Utf8View => view_bytes(col.as_string_view().views()),
        DataType::BinaryView => view_bytes(col.as_binary_view().views()),
        _ => return false,
    };
    bytes > LONG_KEY_BYTES.saturating_mul(col.len())
}

/// Bytes between an array's first and last offset.
fn offset_bytes<O: arrow::array::OffsetSizeTrait>(offsets: &[O]) -> usize {
    match (offsets.first(), offsets.last()) {
        (Some(first), Some(last)) => (*last - *first).as_usize(),
        _ => 0,
    }
}

/// Sum of the lengths a view array's views hold, in their low 32 bits.
fn view_bytes(views: &[u128]) -> usize {
    views.iter().map(|view| *view as u32 as usize).sum()
}

/// Key types whose equality under [`distinct`] is the equality of their bytes.
fn compares_runs(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

/// Compacts row-aligned group indices using an aggregate filter.
///
/// Returns `None` when every row is selected so callers can reuse the input
/// slice without allocating. At high selectivity, copying contiguous selected
/// ranges avoids branching once per row.
fn compact_group_indices(
    group_indices: &[usize],
    filter: &BooleanArray,
) -> Option<Vec<usize>> {
    debug_assert_eq!(group_indices.len(), filter.len());

    let filter = match filter.null_count() {
        0 => Cow::Borrowed(filter),
        _ => Cow::Owned(prep_null_mask_filter(filter)),
    };
    let mask = filter.values();
    let selected_rows = mask.count_set_bits();

    if selected_rows == group_indices.len() {
        return None;
    }

    let mut compacted = Vec::with_capacity(selected_rows);
    // Match scatter's strategy: above 80% selectivity, copy contiguous ranges.
    if selected_rows * 5 > group_indices.len() * 4 {
        for (start, end) in mask.set_slices() {
            compacted.extend_from_slice(&group_indices[start..end]);
        }
    } else {
        compacted.extend(mask.set_indices().map(|index| group_indices[index]));
    }
    debug_assert_eq!(compacted.len(), selected_rows);

    Some(compacted)
}

impl HashAggregateAccumulator {
    pub(super) fn new(
        aggregate_expr: Arc<AggregateFunctionExpr>,
        arguments: Vec<Arc<dyn PhysicalExpr>>,
        filter: Option<Arc<dyn PhysicalExpr>>,
        accumulator: Box<dyn GroupsAccumulator>,
        submetrics: Arc<dyn AggregateMetrics>,
    ) -> Self {
        Self {
            aggregate_expr,
            arguments: arguments.into_iter().map(AggregateArgument::new).collect(),
            filter,
            accumulator,
            submetrics,
        }
    }

    /// Construct a new accumulator with the same definition, but with empty internal
    /// state buffers (empty [`GroupsAccumulator`]).
    pub(super) fn empty_like(&self) -> Result<Self> {
        let accumulator =
            create_group_accumulator(&self.aggregate_expr, Arc::clone(&self.submetrics))?;
        Ok(Self::new(
            Arc::clone(&self.aggregate_expr),
            self.arguments
                .iter()
                .map(|argument| Arc::clone(argument.expr()))
                .collect(),
            self.filter.clone(),
            accumulator,
            Arc::clone(&self.submetrics),
        ))
    }

    /// Evaluate aggregate arguments and filter for one input batch.
    ///
    /// For example, `AVG(2 / x) FILTER (WHERE x > 0)` evaluates `x > 0`
    /// first, then evaluates `2 / x` against a compact batch containing only
    /// selected rows. Filtered rows won't trigger errors such as divide by zero.
    ///
    /// Before updating [`GroupsAccumulator`], the retained selection is used to
    /// compact the matching group IDs and is not passed through.
    pub(super) fn evaluate_compacted_args(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<CompactedAccumulatorArgs> {
        let selection = self.evaluate_filter(batch)?;
        let selected_rows = selection.as_ref().map(|selection| selection.true_count());
        let filtered_batch = match (selection.as_ref(), selected_rows) {
            (Some(selection), Some(selected_rows))
                if selected_rows > 0 && selected_rows < batch.num_rows() =>
            {
                Some(filter_record_batch(batch, selection)?)
            }
            _ => None,
        };
        let argument_batch = match selected_rows {
            None => Some(batch),
            Some(0) => None,
            Some(selected_rows) if selected_rows == batch.num_rows() => Some(batch),
            Some(_) => filtered_batch.as_ref(),
        };
        let arguments = self
            .arguments
            .iter_mut()
            .map(|argument| {
                if let Some(argument_batch) = argument_batch {
                    argument.evaluate(argument_batch)
                } else {
                    let data_type =
                        argument.expr().data_type(batch.schema_ref().as_ref())?;
                    Ok(new_empty_array(&data_type))
                }
            })
            .collect::<Result<_>>()?;

        Ok(CompactedAccumulatorArgs {
            arguments,
            selection,
        })
    }

    /// Evaluates selected arguments while preserving the input batch row count.
    ///
    /// Skip-partial conversion produces one state row per input row, so rejected
    /// rows remain as null argument values and the filter is passed to
    /// [`GroupsAccumulator::convert_to_state`].
    pub(super) fn evaluate_row_aligned_args(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<RowAlignedAccumulatorArgs> {
        let filter = self.evaluate_filter(batch)?;
        let selection = filter.as_ref();
        let arguments = self
            .arguments
            .iter_mut()
            .map(|argument| match selection {
                Some(selection) => argument.evaluate_selection(batch, selection),
                None => argument.evaluate(batch),
            })
            .collect::<Result<_>>()?;

        Ok(RowAlignedAccumulatorArgs { arguments, filter })
    }

    fn evaluate_filter(&self, batch: &RecordBatch) -> Result<Option<BooleanArray>> {
        self.filter
            .as_ref()
            .map(|filter| {
                filter
                    .evaluate(batch)
                    .and_then(|value| value.into_array(batch.num_rows()))
                    .map(|filter| filter.as_boolean().clone())
            })
            .transpose()
    }

    pub(super) fn size(&self) -> usize {
        self.accumulator.size()
    }

    /// See [`GroupsAccumulator::shrink_to_fit`].
    pub(super) fn shrink_to_fit(&mut self) {
        self.accumulator.shrink_to_fit();
    }

    pub(super) fn update_batch(
        &mut self,
        values: &CompactedAccumulatorArgs,
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        let filtered_group_indices = values
            .selection
            .as_ref()
            .and_then(|selection| compact_group_indices(group_indices, selection));
        let group_indices = filtered_group_indices.as_deref().unwrap_or(group_indices);
        self.accumulator.update_batch(
            &values.arguments,
            group_indices,
            None,
            total_num_groups,
        )
    }

    pub(super) fn merge_batch(
        &mut self,
        values: &CompactedAccumulatorArgs,
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert!(values.selection.is_none());
        self.accumulator
            .merge_batch(&values.arguments, group_indices, total_num_groups)
    }

    /// Evaluating final aggregate results according to `EmitTo`, and reset inner
    /// states. (e.g. after `evaluate(EmitTo::All)`, it returns all accumulated groups
    /// , and clear the inner buffers)
    pub(super) fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.accumulator.evaluate(emit_to)
    }

    pub(super) fn evaluate_to_columns(
        &mut self,
        emit_to: EmitTo,
    ) -> Result<Vec<ArrayRef>> {
        Ok(vec![self.evaluate(emit_to)?])
    }

    /// Evaluating partial aggregate results according to `EmitTo`, and reset inner
    /// states. (e.g. after `state(EmitTo::All)`, it returns all accumulated groups
    /// , and clear the inner buffers)
    pub(super) fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.accumulator.state(emit_to)
    }

    /// Converts evaluated row-aligned arguments directly to partial state.
    pub(super) fn convert_to_state(
        &self,
        values: &RowAlignedAccumulatorArgs,
    ) -> Result<Vec<ArrayRef>> {
        self.accumulator
            .convert_to_state(&values.arguments, values.filter.as_ref())
    }

    pub(super) fn null_arguments(
        &self,
        input_schema: &SchemaRef,
        num_rows: usize,
    ) -> Result<Vec<ArrayRef>> {
        self.arguments
            .iter()
            .map(|argument| {
                let data_type = argument.expr().data_type(input_schema)?;
                Ok(new_null_array(&data_type, num_rows))
            })
            .collect()
    }
}

impl AggregateHashTableState {
    pub(super) fn building(&self) -> &AggregateHashTableBuffer {
        let Self::Building(state) = self else {
            unreachable!("hash aggregate table is not building")
        };
        state
    }

    pub(super) fn building_mut(&mut self) -> &mut AggregateHashTableBuffer {
        let Self::Building(state) = self else {
            unreachable!("hash aggregate table is not building")
        };
        state
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, BooleanArray, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_functions_aggregate::sum::sum_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::Column;

    use super::*;
    use crate::aggregates::group_values::aggregate_sub_metrics;
    use crate::metrics::ExecutionPlanMetricsSet;

    #[test]
    fn compact_group_indices_uses_filter_bitmap() {
        let group_indices = (0..10).collect::<Vec<_>>();
        let all_true = BooleanArray::from(vec![true; 10]);
        assert_eq!(compact_group_indices(&group_indices, &all_true), None);

        let high_selectivity =
            BooleanArray::from((0..10).map(|index| index != 4).collect::<Vec<_>>());
        assert_eq!(
            compact_group_indices(&group_indices, &high_selectivity),
            Some(vec![0, 1, 2, 3, 5, 6, 7, 8, 9])
        );

        let with_nulls =
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true), None]);
        assert_eq!(
            compact_group_indices(&group_indices[..5], &with_nulls),
            Some(vec![0, 3])
        );
    }

    #[test]
    fn materialized_aggregate_output_slices_batches_until_exhausted() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "group_col",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )?;
        let mut output = MaterializedAggregateOutput::new(batch);

        assert_eq!(int32_values(&output.next_batch(2).unwrap(), 0), vec![1, 2]);
        assert_eq!(int32_values(&output.next_batch(2).unwrap(), 0), vec![3, 4]);
        assert_eq!(int32_values(&output.next_batch(2).unwrap(), 0), vec![5]);
        assert!(output.next_batch(2).is_none());
        assert!(output.is_exhausted());

        Ok(())
    }

    #[test]
    fn convert_to_state_preserves_rows_and_metrics() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
            Field::new("include", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
                Arc::new(BooleanArray::from(vec![true, false, true, false])),
            ],
        )?;
        let metrics = ExecutionPlanMetricsSet::new();
        let submetrics = aggregate_sub_metrics(&metrics, 0, ["SUM(value)"])
            .pop()
            .expect("one aggregate submetric factory");
        let mut accumulator = sum_accumulator(&schema, "include", 1, submetrics)?;
        let group_by_metrics = GroupByMetrics::new(&metrics, 0);
        let argument_metrics = AggregateArgumentMetrics::new(&metrics, 0, ["SUM(value)"]);
        let accumulator_metrics = AggregateAccumulatorMetrics::new(
            &metrics,
            0,
            ["SUM(value)"],
            &[AccumulatorPhase::ConvertToState],
        );

        let values = group_by_metrics.time_aggregate_arguments(|| {
            argument_metrics.time(0, || accumulator.evaluate_row_aligned_args(&batch))
        })?;
        let state =
            accumulator_metrics.time(0, AccumulatorPhase::ConvertToState, || {
                accumulator.convert_to_state(&values)
            })?;

        assert_eq!(
            int64_options(&state[0]),
            vec![Some(10), None, Some(30), None]
        );
        let metrics = metrics.clone_inner();
        for metric_name in [
            "aggregate_arguments_time",
            "agg_expr_0_arguments_time",
            "agg_expr_0_convert_to_state_time",
        ] {
            assert!(
                metrics
                    .sum_by_name(metric_name)
                    .is_some_and(|time| { time.as_usize() > 0 })
            );
        }

        Ok(())
    }

    fn sum_accumulator(
        schema: &SchemaRef,
        filter_name: &str,
        filter_index: usize,
        submetrics: Arc<dyn AggregateMetrics>,
    ) -> Result<HashAggregateAccumulator> {
        let argument: Arc<dyn PhysicalExpr> = Arc::new(Column::new("value", 0));
        let aggregate_expr = Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![Arc::clone(&argument)])
                .schema(Arc::clone(schema))
                .alias("SUM(value)")
                .build()?,
        );
        let accumulator =
            create_group_accumulator(&aggregate_expr, Arc::clone(&submetrics))?;
        Ok(HashAggregateAccumulator::new(
            aggregate_expr,
            vec![argument],
            Some(Arc::new(Column::new(filter_name, filter_index))),
            accumulator,
            submetrics,
        ))
    }

    fn int64_options(array: &ArrayRef) -> Vec<Option<i64>> {
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    fn int32_values(batch: &RecordBatch, column: usize) -> Vec<i32> {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        (0..array.len()).map(|idx| array.value(idx)).collect()
    }

    /// Interns `batches` in order, by runs and as they are, into fresh group
    /// values for `schema`, and asserts both give the same group of every row
    /// and the same groups. Returns the rows collapsed per batch.
    fn assert_runs_match(schema: Schema, batches: &[Vec<ArrayRef>]) -> Vec<usize> {
        let schema = Arc::new(schema);
        let mut runs =
            new_group_values(Arc::clone(&schema), &GroupClustering::None).unwrap();
        let mut plain = new_group_values(schema, &GroupClustering::None).unwrap();
        let (mut by_runs, mut as_is) = (vec![], vec![]);
        let collapsed = batches
            .iter()
            .map(|cols| {
                let collapsed = intern_runs(runs.as_mut(), cols, &mut by_runs).unwrap();
                plain.intern(cols, &mut as_is).unwrap();
                assert_eq!(by_runs, as_is);
                collapsed
            })
            .collect();
        let (runs, plain) = (
            runs.emit(EmitTo::All).unwrap(),
            plain.emit(EmitTo::All).unwrap(),
        );
        assert_eq!(runs.len(), plain.len());
        for (runs, plain) in runs.iter().zip(&plain) {
            assert_eq!(runs.to_data(), plain.to_data());
        }
        collapsed
    }

    #[test]
    fn intern_runs_matches_intern_with_nulls_and_two_keys() {
        use arrow::array::StringViewArray;
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ]);
        let a: ArrayRef = Arc::new(Int32Array::from(vec![
            None,
            None,
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            None,
            Some(2),
            Some(2),
        ]));
        let b: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            Some("x"),
            None,
            None,
            Some("a value longer than twelve bytes"),
            Some("a value longer than twelve bytes"),
            Some("a value longer than twelve bytes"),
            Some("x"),
            None,
            Some("x"),
        ]));
        // Repeats: rows 1, 3, 5. Row 6 differs in `a` only, row 9 in `b` only.
        assert_eq!(assert_runs_match(schema, &[vec![a, b]]), vec![3]);
    }

    #[test]
    fn intern_runs_does_not_collapse_across_batches() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, true)]);
        let first: ArrayRef = Arc::new(Int64Array::from(vec![1, 1, 2, 2]));
        let second: ArrayRef = Arc::new(Int64Array::from(vec![2, 2, 3, 3]));
        // The second batch's first row repeats the first batch's last row but
        // is interned; it finds the group the first batch made.
        assert_eq!(
            assert_runs_match(schema, &[vec![first], vec![second]]),
            vec![2, 2]
        );
    }

    #[test]
    fn intern_runs_interns_a_batch_that_barely_repeats_as_it_is() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        // One repeat in 16 rows is under an eighth.
        let mut values = (0..16).collect::<Vec<i64>>();
        values[8] = 7;
        let barely: ArrayRef = Arc::new(Int64Array::from(values));
        // Two repeats in 16 rows is an eighth.
        let mut values = (0..16).collect::<Vec<i64>>();
        values[8] = 7;
        values[12] = 11;
        let enough: ArrayRef = Arc::new(Int64Array::from(values));
        let single: ArrayRef = Arc::new(Int64Array::from(vec![5]));
        assert_eq!(
            assert_runs_match(schema, &[vec![barely], vec![enough], vec![single]]),
            vec![0, 2, 0]
        );
    }

    #[test]
    fn intern_runs_compares_floats_by_their_bits() {
        use arrow::array::Float64Array;
        let schema = Schema::new(vec![Field::new("a", DataType::Float64, true)]);
        let a: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(f64::NAN),
            None,
            None,
            Some(1.5),
            Some(1.5),
        ]));
        assert_eq!(assert_runs_match(schema, &[vec![a]]), vec![3]);
    }

    #[test]
    fn intern_runs_does_not_compare_long_values() {
        use arrow::array::{StringArray, StringViewArray};
        let long = "https://example.com/a/path/that/is/longer/than/thirty-two/bytes";
        let short = "phrase";
        // Every row repeats its predecessor; only the length decides.
        let rows = |value: &'static str| vec![value; 8];
        let view =
            |value| Schema::new(vec![Field::new(value, DataType::Utf8View, false)]);
        let long_view: ArrayRef = Arc::new(StringViewArray::from(rows(long)));
        let short_view: ArrayRef = Arc::new(StringViewArray::from(rows(short)));
        assert_eq!(assert_runs_match(view("v"), &[vec![long_view]]), vec![0]);
        assert_eq!(assert_runs_match(view("v"), &[vec![short_view]]), vec![7]);

        // Offsets: a slice is measured by its own values, not its buffer's.
        let mut values = rows(short);
        values.extend(rows(long));
        let utf8: ArrayRef = Arc::new(StringArray::from(values));
        let schema = || Schema::new(vec![Field::new("s", DataType::Utf8, false)]);
        assert_eq!(
            assert_runs_match(schema(), &[vec![utf8.slice(0, 8)]]),
            vec![7]
        );
        assert_eq!(
            assert_runs_match(schema(), &[vec![utf8.slice(8, 8)]]),
            vec![0]
        );

        // One long key column stops the whole batch, beside a short one.
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("v", DataType::Utf8View, false),
        ]);
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1; 8]));
        let v: ArrayRef = Arc::new(StringViewArray::from(rows(long)));
        assert_eq!(assert_runs_match(schema, &[vec![a, v]]), vec![0]);
    }
}
