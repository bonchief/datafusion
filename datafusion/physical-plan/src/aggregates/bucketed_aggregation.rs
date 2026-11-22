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
use super::final_buckets::{FinalBuckets, MAX_BUCKET_LEVELS};
use crate::SendableRecordBatchStream;
use crate::metrics;
use crate::metrics::{BaselineMetrics, MetricBuilder, RecordOutput};
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;

/// Upper bound of the hash map's bytes per group of a presized table: the
/// map keeps between 8/7 and 16/7 slots per entry, a slot is a 16 byte entry
/// plus a control byte, and the table keeps a group index per row.
const MAP_BYTES_PER_GROUP: usize = 48;

/// What a hash aggregation stream needs to split the groups of a hash table
/// that has reached `hash_aggregate_bucket_threshold` into [`FinalBuckets`],
/// and to aggregate those buckets one after another.
///
/// The buckets hold partial state rows, whatever the input of the stream is:
/// a final aggregation routes its input as is, a single stage aggregation
/// routes the state of the table that aggregates its raw input.
///
/// # When a bucket is aggregated
///
/// While input arrives, an eager bucket (see [`FinalBuckets`]) aggregates a
/// chunk of its rows into its own table when the stream finds its input
/// with no batch ready ([`Self::aggregate_idle`]), and once it has a table,
/// also whenever [`Self::compact`] finds it due after a routed batch. Both
/// run in the stream's own task, within the turns it is given: no other task
/// or thread is started. After input ends each bucket, in bucket order,
/// aggregates its remaining rows into its table, if it has one, and emits
/// it. A bucket without a table is aggregated as a whole after input ends,
/// as it was before eager buckets existed.
///
/// # Memory
///
/// The stream's reservation covers, at every point where it is resized, the
/// memory held besides the buckets plus every bucket's buffered rows and
/// table ([`FinalBuckets::memory_size`]). Before a table is presized for a
/// chunk, the reservation is grown to cover the presized table
/// ([`Self::presize`]); after each routed batch it is set to what is held
/// ([`Self::reserve`]), spilling the largest bucket, rows and table, for as
/// long as the pool refuses it. While output is produced, the buckets that
/// wait for their turn stay charged until they are aggregated, and each
/// table is released once it has been emitted.
pub(super) struct BucketedAggregation {
    /// Number of groups in one hash table that triggers bucketing
    threshold: usize,
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
    /// Number of times a hash table was split into buckets
    bucket_splits: metrics::Count,
    /// Number of times the rows of a bucket were replaced by their aggregated state
    bucket_compactions: metrics::Count,
    /// Number of times the buffered rows of an eager bucket were aggregated
    /// into its table while input was still arriving
    bucket_chunks: metrics::Count,
}

/// What [`BucketedAggregation::output_stream`] works with while it runs.
struct BucketOutput {
    reservation: MemoryReservation,
    baseline_metrics: BaselineMetrics,
    /// Combines the output of small buckets into batches of the target size.
    coalescer: BatchCoalescer,
}

impl BucketedAggregation {
    /// `agg` must aggregate rows of `state_schema` as partial state: see
    /// [`AggregateHashTable::<FinalMarker>::new_over_state`].
    pub(super) fn new(
        threshold: usize,
        agg: AggregateExec,
        partition: usize,
        batch_size: usize,
        state_schema: SchemaRef,
        output_schema: SchemaRef,
        spill_manager: Option<SpillManager>,
    ) -> Self {
        let bucket_splits =
            MetricBuilder::new(&agg.metrics).counter("bucket_splits", partition);
        let bucket_compactions =
            MetricBuilder::new(&agg.metrics).counter("bucket_compactions", partition);
        let bucket_chunks =
            MetricBuilder::new(&agg.metrics).counter("bucket_chunks", partition);
        Self {
            threshold,
            agg,
            partition,
            batch_size,
            state_schema,
            output_schema,
            spill_manager,
            bucket_splits,
            bucket_compactions,
            bucket_chunks,
        }
    }

    /// True if a schema of partial state rows can be bucketed.
    ///
    /// Bucketing moves every row once more before it is aggregated, which only
    /// pays when moving a row is cheap compared to the cache miss it saves.
    /// That holds for fixed-width columns. Variable-length keys or state
    /// (strings, lists, ...) cost as much to move as the smaller tables save,
    /// so such aggregations keep their single table.
    pub(super) fn supports_state(state_schema: &SchemaRef) -> bool {
        state_schema.fields().iter().all(|field| {
            let data_type = field.data_type();
            data_type.is_primitive() || *data_type == DataType::Boolean
        })
    }

    pub(super) fn threshold(&self) -> usize {
        self.threshold
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

    /// Starts the buckets of `level` with `state`, the state of the table
    /// that is split. `kept` is the share of its input rows which that table
    /// kept as groups.
    pub(super) fn split(
        &self,
        level: u32,
        state: Option<RecordBatch>,
        kept: f64,
    ) -> Result<FinalBuckets> {
        let mut buckets = FinalBuckets::new(
            &self.state_schema,
            self.agg.group_by().num_group_exprs(),
            self.batch_size,
            level,
            self.spill_manager.clone(),
        );
        self.bucket_splits.add(1);
        buckets.expect_kept(kept);
        if let Some(state) = state {
            buckets.route(&state)?;
        }
        Ok(buckets)
    }

    /// Aggregates the buffered rows of every bucket with a table that is due
    /// for it into that table, then replaces the rows of every bucket without
    /// a table that is due for it by their aggregated state. See the sections
    /// on aggregating as input arrives and on compaction of [`FinalBuckets`].
    ///
    /// `table` is the table that compacts buckets, and the template of the
    /// buckets' tables. `other_bytes` is the memory held besides `buckets`,
    /// for the reservation that presizes a bucket's table.
    pub(super) fn compact(
        &self,
        buckets: &mut FinalBuckets,
        table: &mut Option<AggregateHashTable<FinalMarker>>,
        reservation: &MemoryReservation,
        other_bytes: usize,
    ) -> Result<()> {
        while let Some(index) = buckets.bucket_to_aggregate() {
            self.aggregate_chunk(buckets, index, table, reservation, other_bytes)?;
        }

        while let Some(index) = buckets.bucket_to_compact() {
            let table = match table {
                Some(table) => table,
                None => table.insert(self.new_table()?),
            };
            let mut input_rows = 0;
            for batch in buckets.take_bucket(index)? {
                input_rows += batch.num_rows();
                table.aggregate_batch(&batch)?;
            }
            buckets.put_compacted(
                index,
                table.take_state_batch_keep_capacity()?,
                input_rows,
            );
            self.bucket_compactions.add(1);
        }
        Ok(())
    }

    /// Aggregates one chunk of an eager bucket while the stream's input has
    /// no batch ready. Returns false if no bucket buffers a batch of rows.
    /// See [`Self::compact`] for the arguments.
    pub(super) fn aggregate_idle(
        &self,
        buckets: &mut FinalBuckets,
        table: &mut Option<AggregateHashTable<FinalMarker>>,
        reservation: &MemoryReservation,
        other_bytes: usize,
    ) -> Result<bool> {
        let Some(index) = buckets.bucket_to_aggregate_idle() else {
            return Ok(false);
        };
        self.aggregate_chunk(buckets, index, table, reservation, other_bytes)?;
        Ok(true)
    }

    /// Aggregates the buffered rows of the eager bucket `index` into its
    /// table, which is started from `template` if it has none yet.
    fn aggregate_chunk(
        &self,
        buckets: &mut FinalBuckets,
        index: usize,
        template: &mut Option<AggregateHashTable<FinalMarker>>,
        reservation: &MemoryReservation,
        other_bytes: usize,
    ) -> Result<()> {
        let can_split = buckets.level() + 1 < MAX_BUCKET_LEVELS;
        let template = match template {
            Some(table) => table,
            None => template.insert(self.new_table()?),
        };
        let mut bucket_table = match buckets.take_table(index) {
            Some(bucket_table) => bucket_table,
            None => template.empty_like()?,
        };
        let batches = buckets.take_bucket(index)?;
        let rows = batches.iter().map(RecordBatch::num_rows).sum();
        let bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        let held = other_bytes + buckets.memory_size() + bytes;
        // A table that cannot be split any more grows past the threshold
        let limit = if can_split {
            self.threshold
        } else {
            usize::MAX
        };
        self.presize(&mut bucket_table, rows, bytes, limit, reservation, held)?;
        for batch in &batches {
            bucket_table.aggregate_batch(batch)?;
        }
        drop(batches);
        // The table waits for its next chunk without the scratch of this one
        bucket_table.release_scratch();
        if can_split && bucket_table.building_group_count() >= self.threshold {
            buckets.stop_eager(index, bucket_table.take_state_batch()?, rows);
        } else {
            buckets.put_table(index, bucket_table, rows);
        }
        self.bucket_chunks.add(1);
        Ok(())
    }

    /// Makes room in `table` for the groups of `rows` more state rows of
    /// `bytes` bytes, at most `limit` groups in all, so that aggregating them
    /// neither grows nor rehashes the table: the rows are an upper bound on
    /// the groups they add.
    ///
    /// The memory is reserved before it is allocated: `held` plus the table
    /// plus an upper bound of its growth (the state row of each new group,
    /// which mirrors the table's group and accumulator columns for the fixed
    /// width state that is bucketed, and [`MAP_BYTES_PER_GROUP`]). If the
    /// pool refuses it, the table is left to grow as it aggregates, and the
    /// caller's reservation after the aggregation decides what to spill.
    fn presize(
        &self,
        table: &mut AggregateHashTable<FinalMarker>,
        rows: usize,
        bytes: usize,
        limit: usize,
        reservation: &MemoryReservation,
        held: usize,
    ) -> Result<()> {
        let groups = table.building_group_count();
        let target = groups.saturating_add(rows).min(limit.max(groups));
        if target <= groups {
            return Ok(());
        }
        let row_bytes = bytes / rows.max(1);
        let growth = (target - groups).saturating_mul(row_bytes + MAP_BYTES_PER_GROUP);
        let size = held
            .saturating_add(table.memory_size())
            .saturating_add(growth);
        match reservation.try_resize(size) {
            Ok(()) => {
                table.reserve_groups(target);
                Ok(())
            }
            Err(DataFusionError::ResourcesExhausted(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Reserves `other_bytes` plus the memory of `buckets`, spilling buckets
    /// for as long as the reservation does not fit.
    pub(super) fn reserve(
        &self,
        reservation: &MemoryReservation,
        other_bytes: usize,
        buckets: &mut FinalBuckets,
    ) -> Result<()> {
        loop {
            let size = other_bytes.saturating_add(buckets.memory_size());
            match reservation.try_resize(size) {
                Ok(()) => return Ok(()),
                Err(e @ DataFusionError::ResourcesExhausted(_)) => {
                    if buckets.spill_largest()? {
                        continue;
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
    /// An eager bucket finishes in the table that aggregated its rows while
    /// input arrived; the others share one table, which keeps its capacity
    /// from one bucket to the next. Every table is presized for the rows that
    /// are left in the bucket, if they are in memory.
    ///
    /// A bucket whose table reaches the bucketing threshold again, or does
    /// not fit in memory, is split into buckets of the next level.
    pub(super) fn output_stream(
        self: Arc<Self>,
        buckets: FinalBuckets,
        reservation: MemoryReservation,
        baseline_metrics: BaselineMetrics,
    ) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.output_schema);
        let batch_size = self.batch_size;
        let stream = async_try_stream(move |mut emitter| async move {
            let mut output = BucketOutput {
                reservation,
                baseline_metrics,
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
            let next_level = buckets.level() + 1;

            let sources = buckets.into_sources()?;
            // Memory of the buckets that wait for their turn
            let mut waiting_bytes: usize =
                sources.iter().map(|source| source.memory_size()).sum();

            let can_split = next_level < MAX_BUCKET_LEVELS;
            let limit = if can_split {
                self.threshold
            } else {
                usize::MAX
            };

            // An eager bucket finishes in its own table. One table aggregates
            // all the other buckets, one after another.
            let mut reusable_table = None;
            for mut source in sources {
                waiting_bytes -= source.memory_size();
                let eager = source.take_table();
                let mut source_bytes = source.memory_size();
                let rows = source.rows();
                let mut input = source.into_stream(&self.state_schema);

                let mut timer = elapsed_compute.timer();
                let is_eager = eager.is_some();
                let (mut hash_table, mut table_rows) = match eager {
                    Some(eager) => (eager.table, eager.rows),
                    None => match reusable_table.take() {
                        Some(hash_table) => (hash_table, 0),
                        None => (self.new_table()?.with_restart(), 0),
                    },
                };
                if let Some(rows) = rows {
                    self.presize(
                        &mut hash_table,
                        rows,
                        source_bytes,
                        limit,
                        &output.reservation,
                        waiting_bytes + source_bytes,
                    )?;
                }
                let mut sub_buckets: Option<FinalBuckets> = None;
                let mut compaction_table = None;

                while let Some(batch) = input.next().await.transpose()? {
                    // Batches of an in-memory bucket are released as they are read
                    source_bytes =
                        source_bytes.saturating_sub(batch.get_array_memory_size());
                    let held_bytes = waiting_bytes + source_bytes;

                    if let Some(sub_buckets) = sub_buckets.as_mut() {
                        sub_buckets.route(&batch)?;
                        self.compact(
                            sub_buckets,
                            &mut compaction_table,
                            &output.reservation,
                            held_bytes,
                        )?;
                        self.reserve(&output.reservation, held_bytes, sub_buckets)?;
                        continue;
                    }

                    hash_table.aggregate_batch(&batch)?;
                    table_rows += batch.num_rows();

                    let split = match output
                        .reservation
                        .try_resize(held_bytes + hash_table.memory_size())
                    {
                        Ok(()) => {
                            can_split
                                && hash_table.building_group_count() >= self.threshold
                        }
                        Err(DataFusionError::ResourcesExhausted(_)) if can_split => true,
                        Err(e) => return Err(e),
                    };
                    if split {
                        let kept = hash_table.building_group_count() as f64
                            / table_rows.max(1) as f64;
                        let mut new_buckets =
                            self.split(next_level, hash_table.take_state_batch()?, kept)?;
                        self.reserve(
                            &output.reservation,
                            held_bytes + hash_table.memory_size(),
                            &mut new_buckets,
                        )?;
                        sub_buckets = Some(new_buckets);
                    }
                }
                drop(input);

                if let Some(sub_buckets) = sub_buckets {
                    // The table handed its groups over and is empty again
                    if !is_eager {
                        reusable_table = Some(hash_table);
                    }
                    timer.done();
                    self.produce_output(sub_buckets, output, emitter).await?;
                    continue;
                }

                hash_table.start_output()?;
                while let Some(batch) = hash_table.next_output_batch()? {
                    output
                        .reservation
                        .try_resize(waiting_bytes + hash_table.memory_size())?;
                    output.coalescer.push_batch(batch)?;
                    while let Some(batch) = output.coalescer.next_completed_batch() {
                        timer.done();
                        emitter
                            .emit(batch.record_output(&output.baseline_metrics))
                            .await;
                        timer = elapsed_compute.timer();
                    }
                }
                if hash_table.restart() {
                    reusable_table = Some(hash_table);
                }
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
    fn only_fixed_width_state_is_bucketed() {
        assert!(BucketedAggregation::supports_state(&schema(vec![
            DataType::Int64,
            DataType::Boolean,
            DataType::Decimal128(38, 10),
            DataType::Float64,
            DataType::Date32,
        ])));

        // A variable-length group key or state column turns bucketing off
        for variable_length in [
            DataType::Utf8,
            DataType::Utf8View,
            DataType::LargeBinary,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
        ] {
            assert!(!BucketedAggregation::supports_state(&schema(vec![
                DataType::Int64,
                variable_length.clone(),
            ])));
            assert!(!BucketedAggregation::supports_state(&schema(vec![
                variable_length,
                DataType::Int64,
            ])));
        }
    }
}
