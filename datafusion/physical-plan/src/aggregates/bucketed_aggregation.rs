// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Aggregating partial state that was split into [`FinalBuckets`].
//!
//! See [`BucketedAggregation`].

use std::sync::Arc;

use arrow::compute::BatchCoalescer;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_execution::{TryEmitter, async_try_stream};
use futures::StreamExt;
use futures::future::BoxFuture;

use super::AggregateExec;
use super::aggregate_hash_table::{AggregateHashTable, FinalMarker};
use super::final_buckets::{
    FIXED_WIDTH_BITS, FinalBuckets, MAX_FANOUT_BITS, MIN_FANOUT_BITS, RouteMetrics,
};
use crate::SendableRecordBatchStream;
use crate::metrics;
use crate::metrics::{BaselineMetrics, MetricBuilder, MetricCategory, RecordOutput};
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;

/// What a hash aggregation stream needs to split the groups of a hash table
/// that has reached its [`BucketTrigger`] into [`FinalBuckets`],
/// and to aggregate those buckets one after another.
///
/// The buckets hold partial state rows, whatever the input of the stream is:
/// a final aggregation routes its input as is, a single stage aggregation
/// routes the state of the table that aggregates its raw input.
pub(super) struct BucketedAggregation {
    /// When one hash table is split into buckets
    trigger: BucketTrigger,
    /// Aggregate configuration used to construct the table of each bucket:
    /// its `group_by` refers to the columns of `state_schema`.
    agg: AggregateExec,
    /// Original partition index.
    partition: usize,
    /// Target batch size from configuration.
    batch_size: usize,
    /// Schema of the partial state rows held by the buckets.
    state_schema: SchemaRef,
    /// Schema of the aggregation's output.
    output_schema: SchemaRef,
    /// `None` if spilling is not supported by the configured `DiskManager`.
    spill_manager: Option<SpillManager>,
    /// Bytes of table a bucket is meant to hold:
    /// `hash_aggregate_piece_bytes`.
    piece_bytes: usize,
    /// Rows the partition's input is expected to hold, if known
    expected_rows: Option<usize>,
    /// Number of times a hash table was split into buckets
    bucket_splits: metrics::Count,
    /// Number of times the rows of a bucket were replaced by their aggregated state
    bucket_compactions: metrics::Count,
    /// Time and rows of routing rows into buckets
    route_metrics: RouteMetrics,
    /// Time spent compacting buckets
    compaction_time: metrics::Time,
    /// Rows aggregated again by compactions
    compacted_rows: metrics::Count,
    /// Time spent aggregating the rows of finished buckets
    bucket_aggregate_time: metrics::Time,
    /// Rows aggregated from finished buckets
    bucket_aggregated_rows: metrics::Count,
    /// Time spent taking the state of a table that is split
    bucket_state_time: metrics::Time,
    /// Rows of state taken from tables that were split
    bucket_state_rows: metrics::Count,
    /// Time spent emitting the groups of finished buckets
    bucket_emit_time: metrics::Time,
    /// Time spent clearing the table between buckets
    bucket_reset_time: metrics::Time,
}

/// What [`BucketedAggregation::output_stream`] works with while it runs.
struct BucketOutput {
    reservation: MemoryReservation,
    baseline_metrics: BaselineMetrics,
    /// The stream's `peak_mem_used`
    peak_mem_used: metrics::Gauge,
    /// Combines the output of small buckets into batches of the target size.
    coalescer: BatchCoalescer,
}

impl BucketOutput {
    fn record_peak(&self) {
        self.peak_mem_used.set_max(self.reservation.size());
    }
}

/// How many times its size a table that reached the byte trigger must be
/// expected to grow to be split: see [`BucketedAggregation::grows`].
const MIN_GROWTH: f64 = 2.0;

/// When the hash table of one partition stops growing and is split into
/// [`FinalBuckets`]. See [`BucketedAggregation::trigger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BucketTrigger {
    /// Fixed-width keys and state: once the table holds this many groups.
    Groups(usize),
    /// Variable-width keys or state: once the table, its group keys and its
    /// accumulators take this many bytes.
    Bytes(usize),
}

pub(super) fn is_fixed_width(data_type: &DataType) -> bool {
    data_type.is_primitive() || *data_type == DataType::Boolean
}

impl BucketedAggregation {
    /// `agg` must aggregate rows of `state_schema` as partial state: see
    /// [`AggregateHashTable::<FinalMarker>::new_over_state`].
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new(
        trigger: BucketTrigger,
        agg: AggregateExec,
        partition: usize,
        batch_size: usize,
        state_schema: SchemaRef,
        output_schema: SchemaRef,
        spill_manager: Option<SpillManager>,
        piece_bytes: usize,
        expected_rows: Option<usize>,
    ) -> Self {
        let bucket_splits =
            MetricBuilder::new(&agg.metrics).counter("bucket_splits", partition);
        let bucket_compactions =
            MetricBuilder::new(&agg.metrics).counter("bucket_compactions", partition);
        let route_metrics = RouteMetrics {
            hash_time: MetricBuilder::new(&agg.metrics)
                .subset_time("bucket_route_hash_time", partition),
            move_time: MetricBuilder::new(&agg.metrics)
                .subset_time("bucket_route_move_time", partition),
            rows: MetricBuilder::new(&agg.metrics)
                .with_category(MetricCategory::Rows)
                .counter("bucket_routed_rows", partition),
        };
        let compaction_time = MetricBuilder::new(&agg.metrics)
            .subset_time("bucket_compaction_time", partition);
        let compacted_rows = MetricBuilder::new(&agg.metrics)
            .with_category(MetricCategory::Rows)
            .counter("bucket_compacted_rows", partition);
        let bucket_aggregate_time = MetricBuilder::new(&agg.metrics)
            .subset_time("bucket_aggregate_time", partition);
        let bucket_aggregated_rows = MetricBuilder::new(&agg.metrics)
            .with_category(MetricCategory::Rows)
            .counter("bucket_aggregated_rows", partition);
        let bucket_state_time =
            MetricBuilder::new(&agg.metrics).subset_time("bucket_state_time", partition);
        let bucket_state_rows = MetricBuilder::new(&agg.metrics)
            .with_category(MetricCategory::Rows)
            .counter("bucket_state_rows", partition);
        let bucket_emit_time =
            MetricBuilder::new(&agg.metrics).subset_time("bucket_emit_time", partition);
        let bucket_reset_time =
            MetricBuilder::new(&agg.metrics).subset_time("bucket_reset_time", partition);
        Self {
            trigger,
            agg,
            partition,
            batch_size,
            state_schema,
            output_schema,
            spill_manager,
            piece_bytes,
            expected_rows,
            bucket_splits,
            bucket_compactions,
            route_metrics,
            compaction_time,
            compacted_rows,
            bucket_aggregate_time,
            bucket_aggregated_rows,
            bucket_state_time,
            bucket_state_rows,
            bucket_emit_time,
            bucket_reset_time,
        }
    }

    /// When a table of a final hash aggregation over rows of `state_schema`
    /// is split into buckets, if ever: `groups` is
    /// `hash_aggregate_bucket_threshold`, `bytes` is
    /// `hash_aggregate_bucket_threshold_bytes`, and 0 turns either off.
    ///
    /// Bucketing moves every row once more before it is aggregated, which
    /// only pays when the cache misses of one large table cost more than the
    /// move. A fixed-width row is cheap to move, so such a table is bucketed
    /// by its group count. A string costs several times as much to move (it
    /// is hashed, gathered and its bytes copied into the bucket), and how
    /// large its table is depends on the strings as much as on the groups,
    /// so a table with variable-width keys or state is bucketed by its bytes
    /// instead. Nested state (the lists of `count(distinct)`, `array_agg` or
    /// `median`) is costly to rebuild and takes more memory as rows than
    /// inside its accumulator, so such aggregations keep their single table.
    pub(super) fn trigger(
        state_schema: &SchemaRef,
        groups: usize,
        bytes: usize,
    ) -> Option<BucketTrigger> {
        let types = || state_schema.fields().iter().map(|field| field.data_type());
        if types().all(is_fixed_width) {
            return (groups > 0).then_some(BucketTrigger::Groups(groups));
        }
        let movable = types().all(|data_type| {
            is_fixed_width(data_type)
                || matches!(
                    data_type,
                    DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::BinaryView
                        | DataType::FixedSizeBinary(_)
                )
        });
        (movable && bytes > 0).then_some(BucketTrigger::Bytes(bytes))
    }

    /// True if `table` has grown large enough to be split into buckets.
    pub(super) fn due(&self, table: &AggregateHashTable<FinalMarker>) -> bool {
        match self.trigger {
            BucketTrigger::Groups(groups) => table.building_group_count() >= groups,
            BucketTrigger::Bytes(bytes) => table.memory_size() >= bytes,
        }
    }

    fn new_table(&self) -> Result<AggregateHashTable<FinalMarker>> {
        AggregateHashTable::<FinalMarker>::new_over_state(
            &self.agg,
            &self.state_schema,
            self.partition,
            Arc::clone(&self.output_schema),
            self.batch_size,
        )
    }

    /// Bucket bits for the first split of a table of `table_bytes`, which
    /// has just reached the trigger, after aggregating `table_rows` input
    /// rows: enough buckets for the table the whole input is expected to
    /// build, if every further row were a new group, to come in pieces of
    /// about `piece_bytes`, and of at most half the trigger, so that buckets
    /// are not split again. 64 buckets if the input's row count is unknown,
    /// and always [`FIXED_WIDTH_BITS`] for fixed-width rows, which take the
    /// coalesced route (see [`FinalBuckets`]).
    pub(super) fn fanout_bits(&self, table_bytes: usize, table_rows: usize) -> u32 {
        if matches!(self.trigger, BucketTrigger::Groups(_)) {
            return FIXED_WIDTH_BITS;
        }
        let Some(expected_rows) = self.expected_rows else {
            return 6;
        };
        let growth = expected_rows.max(table_rows) as f64 / table_rows.max(1) as f64;
        let expected_bytes = table_bytes as f64 * growth;
        let pieces = (expected_bytes / self.piece_bytes.max(1) as f64)
            .max(2.0 * growth)
            .max(1.0);
        (pieces.log2().ceil() as u32).clamp(MIN_FANOUT_BITS, MAX_FANOUT_BITS)
    }

    /// True if a table that reached the byte trigger after aggregating
    /// `table_rows` input rows is expected to grow at least [`MIN_GROWTH`]
    /// times past it, or if that cannot be told. A table that has taken most
    /// of its input already gains little from buckets: its rows would be
    /// moved once more to save misses on the few rows left (ClickBench Q12
    /// and Q13 at 8 partitions split near their final size and ran 1.8–1.9x
    /// slower). The group trigger of fixed-width rows keeps splitting as it
    /// did before this rule.
    pub(super) fn grows(&self, table_rows: usize) -> bool {
        Self::expected_to_grow(self.trigger, self.expected_rows, table_rows)
    }

    /// See [`Self::grows`]: `expected_rows` are the input rows the table is
    /// expected to take in all.
    fn expected_to_grow(
        trigger: BucketTrigger,
        expected_rows: Option<usize>,
        table_rows: usize,
    ) -> bool {
        match (trigger, expected_rows) {
            (BucketTrigger::Bytes(_), Some(expected)) => {
                expected as f64 >= MIN_GROWTH * table_rows as f64
            }
            _ => true,
        }
    }

    /// Starts buckets that read `bits` bucket bits from `offset` with the
    /// state of `table`, which is split and left empty.
    pub(super) fn split(
        &self,
        offset: u32,
        bits: u32,
        table: &mut AggregateHashTable<FinalMarker>,
    ) -> Result<FinalBuckets> {
        let state = {
            let _timer = self.bucket_state_time.timer();
            table.take_state_batch()?
        };
        let mut buckets = FinalBuckets::new(
            &self.state_schema,
            self.agg.group_by().num_group_exprs(),
            self.batch_size,
            offset,
            bits,
            self.spill_manager.clone(),
            self.route_metrics.clone(),
        );
        self.bucket_splits.add(1);
        if let Some(state) = state {
            self.bucket_state_rows.add(state.num_rows());
            buckets.route(&state)?;
        }
        Ok(buckets)
    }

    /// Replaces the rows of bucket `index` by their aggregated state.
    fn compact(
        &self,
        buckets: &mut FinalBuckets,
        index: usize,
        table: &mut Option<AggregateHashTable<FinalMarker>>,
    ) -> Result<()> {
        let _timer = self.compaction_time.timer();
        let table = match table {
            Some(table) => table,
            None => table.insert(self.new_table()?),
        };
        let mut input_rows = 0;
        for batch in buckets.take_bucket(index) {
            input_rows += batch.num_rows();
            table.aggregate_batch(&batch)?;
        }
        buckets.put_compacted(
            index,
            table.take_state_batch_keep_capacity()?,
            input_rows,
        )?;
        self.bucket_compactions.add(1);
        self.compacted_rows.add(input_rows);
        Ok(())
    }

    /// Reserves `other_bytes` plus the memory of `buckets`. For as long as
    /// that does not fit, gathers the routed rows, then compacts or spills
    /// the largest bucket: see the memory pressure section of
    /// [`FinalBuckets`].
    pub(super) fn reserve(
        &self,
        reservation: &MemoryReservation,
        other_bytes: usize,
        buckets: &mut FinalBuckets,
        table: &mut Option<AggregateHashTable<FinalMarker>>,
    ) -> Result<()> {
        loop {
            let size = other_bytes.saturating_add(buckets.memory_size());
            match reservation.try_resize(size) {
                Ok(()) => return Ok(()),
                Err(e @ DataFusionError::ResourcesExhausted(_)) => {
                    if buckets.has_routed() {
                        buckets.gather()?;
                        continue;
                    }
                    if let Some(index) = buckets.largest() {
                        if buckets.should_compact(index) {
                            self.compact(buckets, index, table)?;
                            continue;
                        }
                        if buckets.spill(index)? {
                            continue;
                        }
                    }
                    // Every bucket is on disk. What is left is the fixed cost
                    // of routing a batch, which no spill can release, so go on
                    // like the sort based spill path does after it has spilled
                    // its table: with what the pool still grants, which covers
                    // at least the memory held besides these buckets.
                    if buckets.is_fully_spilled()
                        && reservation.try_resize(other_bytes).is_ok()
                    {
                        return Ok(());
                    }
                    return Err(e.context("Hash aggregate has no more buckets to spill"));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Aggregates the buckets one after another, each with a table of its
    /// own size, and emits the groups of a bucket before reading the next one.
    ///
    /// A bucket whose table reaches the bucketing threshold again, or does
    /// not fit in memory, is split into buckets of the next level.
    pub(super) fn output_stream(
        self: Arc<Self>,
        buckets: FinalBuckets,
        reservation: MemoryReservation,
        baseline_metrics: BaselineMetrics,
        peak_mem_used: metrics::Gauge,
    ) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.output_schema);
        let batch_size = self.batch_size;
        let stream = async_try_stream(move |mut emitter| async move {
            let mut output = BucketOutput {
                reservation,
                baseline_metrics,
                peak_mem_used,
                coalescer: BatchCoalescer::new(
                    Arc::clone(&self.output_schema),
                    batch_size,
                )
                .with_biggest_coalesce_batch_size(Some(batch_size / 2)),
            };
            self.produce_output(buckets, &mut output, &mut emitter)
                .await?;
            output.reservation.try_resize(0)?;

            output.coalescer.finish_buffered_batch()?;
            while let Some(batch) = output.coalescer.next_completed_batch() {
                emitter
                    .emit(batch.record_output(&output.baseline_metrics))
                    .await;
            }
            Ok(())
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    fn produce_output<'a>(
        &'a self,
        buckets: FinalBuckets,
        output: &'a mut BucketOutput,
        emitter: &'a mut TryEmitter<RecordBatch, DataFusionError>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let elapsed_compute = output.baseline_metrics.elapsed_compute().clone();
            let next_split = buckets.next_split();

            let (sources, routed_bytes) = buckets.into_sources()?;
            // Memory of the buckets that wait for their turn, and of the
            // routed batches they share until the last of them is done
            let mut waiting_bytes: usize = routed_bytes
                + sources
                    .iter()
                    .map(|source| source.memory_size())
                    .sum::<usize>();

            // One table aggregates all the buckets, one after another
            let mut reusable_table = None;
            for source in sources {
                let source_bytes = source.memory_size();
                waiting_bytes -= source_bytes;
                let mut input = source.into_stream(&self.state_schema);

                let mut timer = elapsed_compute.timer();
                let mut hash_table = match reusable_table.take() {
                    Some(hash_table) => hash_table,
                    None => self.new_table()?.with_restart(),
                };
                let mut sub_buckets: Option<FinalBuckets> = None;
                let mut compaction_table = None;

                while let Some(batch) = input.next().await.transpose()? {
                    // The bucket's rows are held until it is done
                    let held_bytes = waiting_bytes + source_bytes;

                    if let Some(sub_buckets) = sub_buckets.as_mut() {
                        sub_buckets.route(&batch)?;
                        self.reserve(
                            &output.reservation,
                            held_bytes,
                            sub_buckets,
                            &mut compaction_table,
                        )?;
                        output.record_peak();
                        continue;
                    }

                    let aggregate_timer = self.bucket_aggregate_time.timer();
                    hash_table.aggregate_batch(&batch)?;
                    aggregate_timer.done();
                    self.bucket_aggregated_rows.add(batch.num_rows());

                    let can_split = next_split.is_some();
                    let split = match output
                        .reservation
                        .try_resize(held_bytes + hash_table.memory_size())
                    {
                        Ok(()) => can_split && self.due(&hash_table),
                        Err(DataFusionError::ResourcesExhausted(_)) if can_split => true,
                        Err(e) => return Err(e),
                    };
                    if let (true, Some((offset, bits))) = (split, next_split) {
                        let mut new_buckets =
                            self.split(offset, bits, &mut hash_table)?;
                        self.reserve(
                            &output.reservation,
                            held_bytes + hash_table.memory_size(),
                            &mut new_buckets,
                            &mut compaction_table,
                        )?;
                        sub_buckets = Some(new_buckets);
                    }
                    output.record_peak();
                }
                drop(input);

                if let Some(sub_buckets) = sub_buckets {
                    // The table handed its groups over and is empty again
                    reusable_table = Some(hash_table);
                    timer.done();
                    self.produce_output(sub_buckets, output, emitter).await?;
                    continue;
                }

                let mut emit_timer = self.bucket_emit_time.timer();
                hash_table.start_output()?;
                while let Some(batch) = hash_table.next_output_batch()? {
                    output
                        .reservation
                        .try_resize(waiting_bytes + hash_table.memory_size())?;
                    output.coalescer.push_batch(batch)?;
                    while let Some(batch) = output.coalescer.next_completed_batch() {
                        timer.done();
                        emit_timer.done();
                        emitter
                            .emit(batch.record_output(&output.baseline_metrics))
                            .await;
                        timer = elapsed_compute.timer();
                        emit_timer = self.bucket_emit_time.timer();
                    }
                }
                emit_timer.done();
                let reset_timer = self.bucket_reset_time.timer();
                if hash_table.restart() {
                    reusable_table = Some(hash_table);
                }
                reset_timer.done();
                timer.done();
            }

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{Field, Fields, Schema};

    fn schema(types: Vec<DataType>) -> SchemaRef {
        let fields: Fields = types
            .into_iter()
            .enumerate()
            .map(|(i, data_type)| Field::new(format!("c{i}"), data_type, true))
            .collect();
        Arc::new(Schema::new(fields))
    }

    #[test]
    fn fixed_width_state_is_bucketed_by_groups_and_strings_by_bytes() {
        let fixed = schema(vec![
            DataType::Int64,
            DataType::Boolean,
            DataType::Decimal128(38, 10),
            DataType::Float64,
            DataType::Date32,
        ]);
        assert_eq!(
            BucketedAggregation::trigger(&fixed, 100, 1_000),
            Some(BucketTrigger::Groups(100))
        );
        assert_eq!(BucketedAggregation::trigger(&fixed, 0, 1_000), None);

        // A variable-length group key or state column is bucketed by bytes
        for variable_length in [
            DataType::Utf8,
            DataType::Utf8View,
            DataType::LargeBinary,
            DataType::BinaryView,
        ] {
            for types in [
                vec![DataType::Int64, variable_length.clone()],
                vec![variable_length, DataType::Int64],
            ] {
                let state = schema(types);
                assert_eq!(
                    BucketedAggregation::trigger(&state, 100, 1_000),
                    Some(BucketTrigger::Bytes(1_000))
                );
                assert_eq!(BucketedAggregation::trigger(&state, 100, 0), None);
            }
        }

        // Nested state keeps its single table
        let nested = schema(vec![
            DataType::Utf8View,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
        ]);
        assert_eq!(BucketedAggregation::trigger(&nested, 100, 1_000), None);
    }

    #[test]
    fn a_byte_triggered_table_splits_only_if_expected_to_double() {
        let grows = BucketedAggregation::expected_to_grow;
        let bytes = BucketTrigger::Bytes(1_000);
        assert!(!grows(bytes, Some(150), 100), "most of the input is in");
        assert!(grows(bytes, Some(200), 100));
        assert!(grows(bytes, None, 100), "unknown input keeps splitting");
        // The group trigger of fixed-width rows splits as before
        assert!(grows(BucketTrigger::Groups(100), Some(101), 100));
    }
}
