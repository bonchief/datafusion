// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Hash buckets of partial aggregate state for the final hash aggregation.
//!
//! See [`FinalBuckets`].

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::compute::{BatchCoalescer, take_arrays};
use arrow::datatypes::{SchemaRef, UInt32Type};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion_common::Result;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_execution::async_try_stream;
use futures::StreamExt;

use super::bucket_scatter::{ScatterColumn, scatter_column};
use crate::SendableRecordBatchStream;
use crate::metrics::Count;
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;

/// Number of bits of the routing hash consumed by one bucketing level.
const BUCKET_BITS: u32 = 6;

/// Number of buckets rows are split into at each level.
pub(super) const NUM_BUCKETS: usize = 1 << BUCKET_BITS;

/// A bucket that is still too large is split again with the next hash bits,
/// up to this many levels.
pub(super) const MAX_BUCKET_LEVELS: u32 = 4;

/// Seed of the routing hash. It differs from both the `RepartitionExec` seed,
/// whose hash is the same for every row of one final partition modulo the
/// partition count, and the aggregation seed, so that the rows of one bucket
/// still spread evenly over the bucket's own hash table.
const BUCKET_HASH_SEED: RandomState = RandomState::with_seed(5364907223173859721);

/// A bucket is compacted for the first time once it holds this many batches
/// worth of rows. With 64 buckets per partition this floor is what the
/// buckets hold when every bucket has few groups, so it is kept small.
const MIN_COMPACTION_BATCHES: usize = 1;

/// A compaction that kept more than this share of its rows found (almost) no
/// repeated groups.
const POOR_COMPACTION: f64 = 0.8;

/// After a compaction that paid off, the bucket is compacted again once it
/// has received at most this many times its own rows. Measured on unreduced
/// input with 5 to 8 rows per group: 2 holds peak memory at about 1.35x of a
/// single table, 16 lets it reach 2.3x.
const MAX_COMPACTION_FACTOR: f64 = 2.0;

/// After a poor compaction the bucket waits for this many times its own rows.
/// Aggregating rows that are already unique again and again is pure rework
/// (1.1x -> 1.5x run time on a wide string key with a factor of 2). But a
/// small sample of a bucket with many groups looks unique even when every
/// group repeats, so the wait must stay short enough to find that out before
/// the whole input is buffered (16 let peak memory reach 1.8x of a single
/// table on input with 5 rows per group).
const POOR_COMPACTION_FACTOR: f64 = 4.0;

/// One hash bucket: its completed batches, held in memory or appended to a
/// spill file, plus the rows not yet forming a full batch.
struct Bucket {
    /// The rows not yet forming a full batch, when they are routed the
    /// general way. `None` when [`FinalBuckets`] scatters them, and then
    /// `pending` of them are held by its scatter columns.
    coalescer: Option<BatchCoalescer>,
    /// Rows in progress in the scatter columns
    pending: usize,
    batches: VecDeque<RecordBatch>,
    /// Memory held by `batches`
    batches_bytes: usize,
    /// Set once the bucket has been spilled: completed batches go here
    spill_file: Option<InProgressSpillFile>,
    /// Rows routed to the bucket that are still held in memory
    rows: usize,
    /// The bucket is due for compaction once it holds this many rows
    compact_at: usize,
}

/// Rows of partial aggregate state, split by a hash of their group keys.
///
/// Once the table of a final hash aggregation has outgrown the CPU caches,
/// every probe and every accumulator update of a further row is a cache
/// miss. The stream then stops growing that table: it moves the table's
/// state in here, routes the rest of its input here without aggregating it,
/// and finally aggregates the buckets one after another, each with a table
/// that is `NUM_BUCKETS` times smaller. Buckets are independent, so they can
/// be spilled, read back and released one at a time.
///
/// # Compaction
///
/// The input may hold the same group many times (for example when the partial
/// aggregation was skipped), so buffering it as is could take far more memory
/// than the single table it replaces. A bucket that has grown enough is
/// therefore *compacted*: the stream aggregates its rows with a small table
/// and puts the resulting state, one row per group, back in their place. The
/// next compaction waits for as many new rows as made the last one pay off,
/// so the buffered rows stay within a small multiple of the groups they hold,
/// and waits much longer after a compaction that found nothing to merge, so
/// input that does not reduce is not aggregated over and over.
///
/// # Routing
///
/// A row's bucket is picked by bits of a hash of its group keys. When every
/// column of the schema is a fixed-width primitive, a boolean or a view, a
/// routed batch is read once in row order and each row is appended to its
/// bucket's rows in progress ([`ScatterColumn`]), which are handed over as
/// one of the bucket's batches when they reach `batch_size` rows. Otherwise
/// the batch is sorted by bucket with one `take` and each bucket's slice is
/// pushed into a `BatchCoalescer` of its own. Both write the same rows to the
/// same bucket, in the order they arrive, in batches of `batch_size` rows
/// (except a bucket's last one), so a bucket's aggregation sees the same
/// batches either way.
pub(super) struct FinalBuckets {
    /// Bucketing level, selects which bits of the routing hash are used
    level: u32,
    /// Number of leading columns of a batch that are the group keys
    num_group_columns: usize,
    buckets: Vec<Bucket>,
    /// Schema of the rows held, and of the batches handed over
    schema: SchemaRef,
    /// The rows in progress of every bucket, one per column, if every column
    /// can be scattered. See the routing section above.
    scatter: Option<Vec<Box<dyn ScatterColumn>>>,
    /// `None` if spilling is not supported by the configured `DiskManager`
    spill_manager: Option<SpillManager>,
    batch_size: usize,
    min_compaction_rows: usize,
    metrics: RouteMetrics,
    /// Reused buffers
    hashes: Vec<u64>,
    /// The bucket of every row of the batch being routed
    ids: Vec<u8>,
    /// Rows after which a bucket is full, with that bucket
    cuts: Vec<(usize, usize)>,
    bucket_sizes: Vec<u32>,
    reordered_indices: Vec<u32>,
}

/// What [`FinalBuckets::route`] reports, so a plan shows the payload that
/// moves through the buckets.
#[derive(Clone, Debug, Default)]
pub(super) struct RouteMetrics {
    /// Rows routed into buckets
    pub(super) rows: Count,
    /// Bytes the routing wrote: each scattered row's values, validity bits
    /// and long string bytes once; for a batch routed the general way, an
    /// estimate of twice the memory of its sorted copy (the `take`, then the
    /// coalescer).
    pub(super) bytes: Count,
}

impl FinalBuckets {
    pub(super) fn new(
        schema: &SchemaRef,
        num_group_columns: usize,
        batch_size: usize,
        level: u32,
        spill_manager: Option<SpillManager>,
        metrics: RouteMetrics,
    ) -> Self {
        let scatter = schema
            .fields()
            .iter()
            .map(|field| scatter_column(field.data_type()))
            .collect();
        Self::with_scatter(
            schema,
            num_group_columns,
            batch_size,
            level,
            spill_manager,
            metrics,
            scatter,
        )
    }

    fn with_scatter(
        schema: &SchemaRef,
        num_group_columns: usize,
        batch_size: usize,
        level: u32,
        spill_manager: Option<SpillManager>,
        metrics: RouteMetrics,
        scatter: Option<Vec<Box<dyn ScatterColumn>>>,
    ) -> Self {
        debug_assert!(level < MAX_BUCKET_LEVELS);
        let min_compaction_rows = MIN_COMPACTION_BATCHES * batch_size;
        let buckets = (0..NUM_BUCKETS)
            .map(|_| Bucket {
                coalescer: scatter
                    .is_none()
                    .then(|| BatchCoalescer::new(Arc::clone(schema), batch_size)),
                pending: 0,
                batches: VecDeque::new(),
                batches_bytes: 0,
                spill_file: None,
                rows: 0,
                compact_at: min_compaction_rows,
            })
            .collect();
        Self {
            level,
            num_group_columns,
            buckets,
            schema: Arc::clone(schema),
            scatter,
            spill_manager,
            batch_size,
            min_compaction_rows,
            metrics,
            hashes: vec![],
            ids: vec![],
            cuts: vec![],
            bucket_sizes: vec![0; NUM_BUCKETS],
            reordered_indices: vec![],
        }
    }

    pub(super) fn level(&self) -> u32 {
        self.level
    }

    /// Tells the buckets what share of its input rows the table they are
    /// split from kept as groups. If that table found (almost) no repeated
    /// groups, the first compaction of a bucket would most likely find none
    /// either, so it waits as long as after a poor compaction.
    pub(super) fn expect_kept(&mut self, kept: f64) {
        if kept > POOR_COMPACTION {
            let compact_at =
                (self.min_compaction_rows as f64 * POOR_COMPACTION_FACTOR) as usize;
            for bucket in &mut self.buckets {
                bucket.compact_at = compact_at;
            }
        }
    }

    /// Appends every row of `batch` to the bucket of its group key.
    pub(super) fn route(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }

        let group_columns: &[ArrayRef] = &batch.columns()[..self.num_group_columns];
        self.hashes.clear();
        self.hashes.resize(num_rows, 0);
        create_hashes(group_columns, &BUCKET_HASH_SEED, &mut self.hashes)?;
        let shift = self.level * BUCKET_BITS;
        self.ids.clear();
        self.ids.extend(
            self.hashes
                .iter()
                .map(|&hash| (((hash >> shift) as usize) & (NUM_BUCKETS - 1)) as u8),
        );

        let bytes = match self.scatter.is_some() {
            true => self.scatter_batch(batch)?,
            false => self.take_batch(batch)?,
        };
        self.metrics.rows.add(num_rows);
        self.metrics.bytes.add(bytes);
        Ok(())
    }

    /// Writes every row of `batch` once, into the rows in progress of its
    /// bucket. Returns the bytes written.
    fn scatter_batch(&mut self, batch: &RecordBatch) -> Result<usize> {
        let num_rows = batch.num_rows();
        let batch_size = self.batch_size;

        // The rows after which an in-memory bucket holds a full batch. A
        // spilled bucket takes this batch's rows as one batch of its own.
        let mut fill = [0usize; NUM_BUCKETS];
        let mut limit = [usize::MAX; NUM_BUCKETS];
        for ((fill, limit), bucket) in fill.iter_mut().zip(&mut limit).zip(&self.buckets)
        {
            if bucket.spill_file.is_none() {
                *fill = bucket.pending;
                *limit = batch_size;
            }
        }
        let mut counts = [0usize; NUM_BUCKETS];
        self.cuts.clear();
        for (row, &id) in self.ids.iter().enumerate() {
            let id = id as usize;
            counts[id] += 1;
            fill[id] += 1;
            if fill[id] == limit[id] {
                fill[id] = 0;
                self.cuts.push((row + 1, id));
            }
        }

        let mut bytes = 0;
        let mut start = 0;
        let cuts = std::mem::take(&mut self.cuts);
        let ends = cuts
            .iter()
            .map(|&(end, full)| (end, Some(full)))
            .chain(std::iter::once((num_rows, None)));
        for (end, full) in ends {
            if end > start {
                bytes += self.scatter_rows(batch, start, end, &counts);
            }
            if let Some(full) = full {
                debug_assert_eq!(self.buckets[full].pending, batch_size);
                self.hand_off(full)?;
            }
            start = end;
        }
        self.cuts = cuts;

        // A spilled bucket holds no memory: its rows go straight to the file
        // and are combined into batches when read back.
        for (index, &count) in counts.iter().enumerate() {
            if self.buckets[index].spill_file.is_some() && count > 0 {
                let rows = self.finish_pending(index)?;
                if let Some(spill_file) = &mut self.buckets[index].spill_file {
                    spill_file.append_batch(&rows)?;
                }
            }
        }
        Ok(bytes)
    }

    /// Appends the rows `start..end` of `batch` to their buckets, none of
    /// which reaches `batch_size` rows before the last of them.
    fn scatter_rows(
        &mut self,
        batch: &RecordBatch,
        start: usize,
        end: usize,
        batch_counts: &[usize; NUM_BUCKETS],
    ) -> usize {
        let ids = &self.ids[start..end];
        let mut counts = [0usize; NUM_BUCKETS];
        for &id in ids {
            counts[id as usize] += 1;
        }
        let Some(columns) = self.scatter.as_mut() else {
            return 0;
        };
        let mut bytes = 0;
        for (column, source) in columns.iter_mut().zip(batch.columns()) {
            for (index, bucket) in self.buckets.iter().enumerate() {
                if counts[index] > 0 {
                    // A first allocation takes a whole batch in memory, and
                    // all of this batch's rows of a spilled bucket
                    let at_least = match bucket.spill_file {
                        None => self.batch_size,
                        Some(_) => batch_counts[index],
                    };
                    column.reserve(index, counts[index], at_least);
                }
            }
            bytes += column.scatter(source, start, ids, &counts);
        }
        for (bucket, &count) in self.buckets.iter_mut().zip(&counts) {
            bucket.pending += count;
            if bucket.spill_file.is_none() {
                bucket.rows += count;
            }
        }
        bytes
    }

    /// Hands the rows in progress of a bucket over as its next batch.
    fn hand_off(&mut self, index: usize) -> Result<()> {
        if self.buckets[index].pending > 0 {
            let rows = self.finish_pending(index)?;
            let bucket = &mut self.buckets[index];
            bucket.batches_bytes += rows.get_array_memory_size();
            bucket.batches.push_back(rows);
        }
        Ok(())
    }

    /// The rows in progress of a bucket, as one batch.
    fn finish_pending(&mut self, index: usize) -> Result<RecordBatch> {
        let bucket = &mut self.buckets[index];
        let rows = std::mem::take(&mut bucket.pending);
        let columns = self
            .scatter
            .iter_mut()
            .flatten()
            .map(|column| column.finish(index))
            .collect();
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?)
    }

    /// Ends the bucket's rows in progress as a batch, wherever they are.
    fn finish_buffered(&mut self, index: usize) -> Result<()> {
        match self.buckets[index].coalescer.as_mut() {
            Some(coalescer) => coalescer.finish_buffered_batch()?,
            None => self.hand_off(index)?,
        }
        self.buckets[index].collect_completed()
    }

    /// Memory held by the rows in progress of a bucket
    fn pending_size(&self, index: usize) -> usize {
        match (&self.buckets[index].coalescer, &self.scatter) {
            (Some(coalescer), _) => coalescer.size(),
            (None, Some(columns)) => {
                columns.iter().map(|column| column.size(index)).sum()
            }
            (None, None) => 0,
        }
    }

    /// Routes `batch` the general way: sorts its rows by bucket with one
    /// `take` and pushes each bucket's slice into the bucket's coalescer,
    /// which copies it. Returns an estimate of the bytes written.
    fn take_batch(&mut self, batch: &RecordBatch) -> Result<usize> {
        let num_rows = batch.num_rows();

        // Counting sort of the row indices by bucket
        self.bucket_sizes.fill(0);
        for &id in &self.ids {
            self.bucket_sizes[id as usize] += 1;
        }
        let mut starts = [0u32; NUM_BUCKETS];
        let mut next = 0;
        for (start, size) in starts.iter_mut().zip(&self.bucket_sizes) {
            *start = next;
            next += size;
        }
        self.reordered_indices.clear();
        self.reordered_indices.resize(num_rows, 0);
        let mut cursors = starts;
        for (row, &id) in self.ids.iter().enumerate() {
            let cursor = &mut cursors[id as usize];
            self.reordered_indices[*cursor as usize] = row as u32;
            *cursor += 1;
        }

        // One gather for the whole batch, then a slice per bucket. The
        // coalescer copies the slice, so a bucket owns its batches.
        let indices: PrimitiveArray<UInt32Type> =
            std::mem::take(&mut self.reordered_indices).into();
        let columns = take_arrays(batch.columns(), &indices, None)?;
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        let reordered =
            RecordBatch::try_new_with_options(batch.schema(), columns, &options)?;
        let bytes = 2 * reordered.get_array_memory_size();

        for (bucket, (&start, &size)) in self
            .buckets
            .iter_mut()
            .zip(starts.iter().zip(&self.bucket_sizes))
        {
            if size == 0 {
                continue;
            }
            let slice = reordered.slice(start as usize, size as usize);
            match (&mut bucket.spill_file, &mut bucket.coalescer) {
                // A spilled bucket holds no memory: its rows go straight to
                // the file and are combined into batches when read back.
                (Some(spill_file), _) => {
                    spill_file.append_batch(&slice)?;
                }
                (None, Some(coalescer)) => {
                    coalescer.push_batch(slice)?;
                    bucket.rows += size as usize;
                    bucket.collect_completed()?;
                }
                (None, None) => unreachable!("a bucket routed by take has a coalescer"),
            }
        }
        Ok(bytes)
    }

    /// Returns a bucket that is due for compaction, if any.
    pub(super) fn bucket_to_compact(&self) -> Option<usize> {
        self.buckets.iter().position(|bucket| {
            bucket.spill_file.is_none() && bucket.rows >= bucket.compact_at
        })
    }

    /// Removes and returns the rows of an in-memory bucket, to be followed by
    /// [`Self::put_compacted`].
    pub(super) fn take_bucket(&mut self, index: usize) -> Result<Vec<RecordBatch>> {
        debug_assert!(self.buckets[index].spill_file.is_none());
        self.finish_buffered(index)?;
        let bucket = &mut self.buckets[index];
        bucket.batches_bytes = 0;
        Ok(bucket.batches.drain(..).collect())
    }

    /// Puts back the aggregated state of the `input_rows` rows that
    /// [`Self::take_bucket`] returned.
    pub(super) fn put_compacted(
        &mut self,
        index: usize,
        state: Option<RecordBatch>,
        input_rows: usize,
    ) {
        let bucket = &mut self.buckets[index];
        let state_rows = state.as_ref().map_or(0, |state| state.num_rows());
        if let Some(state) = state {
            bucket.batches_bytes += state.get_array_memory_size();
            bucket.batches.push_back(state);
        }
        bucket.rows = state_rows;

        // Compacting again costs about `state_rows + new_rows` and removes
        // about `(1 - kept) * new_rows` rows, where `kept` is the share of rows
        // that survived this time. Wait for enough new rows to pay for it.
        let kept = state_rows as f64 / input_rows.max(1) as f64;
        let factor = if kept > POOR_COMPACTION {
            POOR_COMPACTION_FACTOR
        } else {
            (kept / (1.0 - kept)).clamp(1.0, MAX_COMPACTION_FACTOR)
        };
        let new_rows =
            ((state_rows as f64 * factor) as usize).max(self.min_compaction_rows);
        bucket.compact_at = state_rows + new_rows;
    }

    /// Memory held by the buckets
    pub(super) fn memory_size(&self) -> usize {
        (0..NUM_BUCKETS)
            .map(|index| self.buckets[index].batches_bytes + self.pending_size(index))
            .sum::<usize>()
            + self.hashes.capacity() * size_of::<u64>()
            + self.ids.capacity()
    }

    /// Spills the in-memory batches of the largest bucket, which from then on
    /// appends its batches to the spill file. Returns false if nothing can be
    /// spilled.
    pub(super) fn spill_largest(&mut self) -> Result<bool> {
        let Some(spill_manager) = &self.spill_manager else {
            return Ok(false);
        };
        let Some(index) = (0..NUM_BUCKETS)
            .filter(|&index| {
                let bucket = &self.buckets[index];
                bucket.spill_file.is_none() && bucket.rows > 0
            })
            .max_by_key(|&index| {
                self.buckets[index].batches_bytes + self.pending_size(index)
            })
        else {
            return Ok(false);
        };
        self.buckets[index].spill_file =
            Some(spill_manager.create_in_progress_file("FinalHashAggregateBucket")?);
        // Rows that do not form a full batch yet are spilled as well, and the
        // coalescer is replaced to give up the buffers it allocated. Finished
        // scatter columns hold no buffers for the bucket.
        self.finish_buffered(index)?;
        let bucket = &mut self.buckets[index];
        if let Some(coalescer) = &mut bucket.coalescer {
            *coalescer = BatchCoalescer::new(coalescer.schema(), 1);
        }
        Ok(true)
    }

    /// True if no bucket holds rows in memory.
    pub(super) fn is_fully_spilled(&self) -> bool {
        self.buckets.iter().all(|bucket| bucket.rows == 0)
    }

    /// Finishes the buckets and returns the non-empty ones.
    pub(super) fn into_sources(mut self) -> Result<Vec<BucketSource>> {
        let mut sources = Vec::with_capacity(NUM_BUCKETS);
        for index in 0..NUM_BUCKETS {
            self.finish_buffered(index)?;
        }
        for bucket in self.buckets {
            match bucket.spill_file {
                Some(mut spill_file) => {
                    let spill_manager = self
                        .spill_manager
                        .as_ref()
                        .expect("a spilled bucket has a spill manager");
                    if let Some(file) = spill_file.finish()? {
                        sources.push(BucketSource::Spilled {
                            stream: spill_manager.read_spill_as_stream(file, None)?,
                            batch_size: self.batch_size,
                        });
                    }
                }
                None if bucket.batches.is_empty() => {}
                None => sources.push(BucketSource::Memory {
                    batches: bucket.batches,
                    bytes: bucket.batches_bytes,
                }),
            }
        }
        Ok(sources)
    }
}

impl Bucket {
    /// Moves the batches the coalescer has completed to where the bucket
    /// lives: the spill file if it has been spilled, memory otherwise.
    fn collect_completed(&mut self) -> Result<()> {
        if let Some(coalescer) = &mut self.coalescer {
            while let Some(batch) = coalescer.next_completed_batch() {
                self.batches_bytes += batch.get_array_memory_size();
                self.batches.push_back(batch);
            }
        }
        if let Some(spill_file) = &mut self.spill_file {
            for batch in self.batches.drain(..) {
                spill_file.append_batch(&batch)?;
            }
            self.batches_bytes = 0;
            self.rows = 0;
        }
        Ok(())
    }
}

/// The rows of one finished bucket
pub(super) enum BucketSource {
    Memory {
        batches: VecDeque<RecordBatch>,
        bytes: usize,
    },
    /// Rows were appended to the file in pieces much smaller than a batch
    Spilled {
        stream: SendableRecordBatchStream,
        batch_size: usize,
    },
}

impl BucketSource {
    /// Memory held by the bucket's rows
    pub(super) fn memory_size(&self) -> usize {
        match self {
            Self::Memory { bytes, .. } => *bytes,
            Self::Spilled { .. } => 0,
        }
    }

    pub(super) fn into_stream(self, schema: &SchemaRef) -> SendableRecordBatchStream {
        match self {
            Self::Memory { batches, .. } => Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(schema),
                futures::stream::iter(batches.into_iter().map(Ok)),
            )),
            Self::Spilled { stream, batch_size } => {
                coalesce_stream(stream, Arc::clone(schema), batch_size)
            }
        }
    }
}

/// Combines the batches of `input` into batches of `batch_size` rows.
fn coalesce_stream(
    mut input: SendableRecordBatchStream,
    schema: SchemaRef,
    batch_size: usize,
) -> SendableRecordBatchStream {
    let mut coalescer = BatchCoalescer::new(Arc::clone(&schema), batch_size);
    let stream = async_try_stream(|mut emitter| async move {
        while let Some(batch) = input.next().await.transpose()? {
            coalescer.push_batch(batch)?;
            while let Some(batch) = coalescer.next_completed_batch() {
                emitter.emit(batch).await;
            }
        }
        coalescer.finish_buffered_batch()?;
        while let Some(batch) = coalescer.next_completed_batch() {
            emitter.emit(batch).await;
        }
        Ok(())
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use std::collections::HashMap;

    fn test_batch(schema: &SchemaRef, keys: std::ops::Range<i64>) -> RecordBatch {
        let names: StringArray = keys
            .clone()
            .map(|k| Some(format!("key-{}", k % 97)))
            .collect();
        let values: Int64Array = keys.clone().map(|k| k * 10).collect::<Vec<_>>().into();
        let keys: Int64Array = keys.collect::<Vec<_>>().into();
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![Arc::new(keys), Arc::new(names), Arc::new(values)],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn routes_every_row_once_and_keeps_keys_together() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut buckets =
            FinalBuckets::new(&schema, 2, 1024, 0, None, RouteMetrics::default());
        // the same keys arrive in two different batches
        buckets.route(&test_batch(&schema, 0..20_000))?;
        buckets.route(&test_batch(&schema, 10_000..30_000))?;
        assert!(buckets.memory_size() > 0);
        assert!(
            !buckets.spill_largest()?,
            "no spill manager, nothing spilled"
        );

        let sources = buckets.into_sources()?;
        assert!(
            sources.len() > NUM_BUCKETS / 2,
            "rows spread over the buckets"
        );

        let mut bucket_of_key: HashMap<i64, usize> = HashMap::new();
        let mut rows = 0;
        for (bucket, source) in sources.into_iter().enumerate() {
            let mut stream = source.into_stream(&schema);
            while let Some(batch) = stream.next().await.transpose()? {
                rows += batch.num_rows();
                let keys = batch.column(0).as_primitive::<Int64Type>();
                let values = batch.column(2).as_primitive::<Int64Type>();
                for (key, value) in keys.values().iter().zip(values.values()) {
                    assert_eq!(*value, key * 10, "columns stay aligned");
                    let seen = *bucket_of_key.entry(*key).or_insert(bucket);
                    assert_eq!(seen, bucket, "key {key} in two buckets");
                }
            }
        }
        assert_eq!(rows, 40_000);
        assert_eq!(bucket_of_key.len(), 30_000);
        Ok(())
    }

    #[test]
    fn levels_use_different_hash_bits() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));
        // Rows of one level-0 bucket must spread again at level 1
        let mut level0 =
            FinalBuckets::new(&schema, 1, 1024, 0, None, RouteMetrics::default());
        level0.route(&test_batch(&schema, 0..50_000))?;
        let source = level0.into_sources()?.swap_remove(0);
        let BucketSource::Memory { batches, .. } = source else {
            unreachable!()
        };
        let mut level1 =
            FinalBuckets::new(&schema, 1, 1024, 1, None, RouteMetrics::default());
        for batch in &batches {
            level1.route(batch)?;
        }
        assert!(level1.into_sources()?.len() > NUM_BUCKETS / 2);
        Ok(())
    }

    /// A batch of every column type the scatter writes, with nulls, empty
    /// strings, inline strings and strings longer than a block
    fn typed_batch(schema: &SchemaRef, keys: std::ops::Range<i64>) -> RecordBatch {
        use arrow::array::{
            BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float64Array,
            Int8Array, Int32Array, StringViewArray, TimestampMicrosecondArray,
            UInt64Array,
        };
        let text = |k: i64| match k % 7 {
            0 => None,
            1 => Some(String::new()),
            2 => Some(format!("s{k}")),
            3 => Some("x".repeat(20_000 + (k % 13) as usize)),
            _ => Some(format!("a string longer than twelve bytes {k}")),
        };
        let nullable = |k: i64| (k % 5 != 0).then_some(k);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(keys.clone().map(|k| k % 3001))),
            Arc::new(StringViewArray::from_iter(keys.clone().map(text))),
            Arc::new(Int8Array::from_iter(
                keys.clone().map(|k| nullable(k).map(|k| k as i8)),
            )),
            Arc::new(Int32Array::from_iter(
                keys.clone().map(|k| nullable(k).map(|k| k as i32)),
            )),
            Arc::new(UInt64Array::from_iter_values(
                keys.clone().map(|k| k as u64),
            )),
            Arc::new(Float64Array::from_iter(
                keys.clone().map(|k| nullable(k).map(|k| k as f64 / 3.0)),
            )),
            Arc::new(
                Decimal128Array::from_iter(
                    keys.clone().map(|k| nullable(k).map(i128::from)),
                )
                .with_precision_and_scale(38, 4)
                .unwrap(),
            ),
            Arc::new(
                TimestampMicrosecondArray::from_iter(keys.clone().map(nullable))
                    .with_timezone("+01:00"),
            ),
            Arc::new(Date32Array::from_iter_values(
                keys.clone().map(|k| k as i32),
            )),
            Arc::new(BooleanArray::from_iter(
                keys.clone().map(|k| (k % 3 != 0).then_some(k % 2 == 0)),
            )),
            Arc::new(BinaryViewArray::from_iter(
                keys.clone().map(|k| text(k + 1).map(String::into_bytes)),
            )),
        ];
        RecordBatch::try_new(Arc::clone(schema), columns).unwrap()
    }

    fn typed_schema() -> SchemaRef {
        use arrow::datatypes::TimeUnit;
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, true),
            Field::new("i8", DataType::Int8, true),
            Field::new("i32", DataType::Int32, true),
            Field::new("u64", DataType::UInt64, false),
            Field::new("f64", DataType::Float64, true),
            Field::new("dec", DataType::Decimal128(38, 4), true),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("+01:00".into())),
                true,
            ),
            Field::new("date", DataType::Date32, false),
            Field::new("flag", DataType::Boolean, true),
            Field::new("bin", DataType::BinaryView, true),
        ]))
    }

    /// The buckets of `schema`, routed by scatter or the general way
    fn buckets_of(
        schema: &SchemaRef,
        batch_size: usize,
        level: u32,
        scatter: bool,
        spill_manager: Option<SpillManager>,
    ) -> FinalBuckets {
        let columns = scatter.then(|| {
            schema
                .fields()
                .iter()
                .map(|field| scatter_column(field.data_type()).unwrap())
                .collect()
        });
        FinalBuckets::with_scatter(
            schema,
            2,
            batch_size,
            level,
            spill_manager,
            RouteMetrics::default(),
            columns,
        )
    }

    /// Every bucket's batches, in order
    async fn read_buckets(
        buckets: FinalBuckets,
        schema: &SchemaRef,
    ) -> Result<Vec<Vec<RecordBatch>>> {
        let mut out = vec![];
        for source in buckets.into_sources()? {
            let mut stream = source.into_stream(schema);
            let mut batches = vec![];
            while let Some(batch) = stream.next().await.transpose()? {
                batches.push(batch);
            }
            out.push(batches);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn scattered_rows_equal_the_general_route() -> Result<()> {
        let schema = typed_schema();
        // 7 rows per batch: a bucket hands off several batches per input.
        // 100: a bucket fills over several inputs.
        for batch_size in [7, 100] {
            let mut scattered = buckets_of(&schema, batch_size, 0, true, None);
            let mut taken = buckets_of(&schema, batch_size, 0, false, None);
            for keys in [0..3000, 3000..3001, 1500..6500] {
                let batch = typed_batch(&schema, keys);
                scattered.route(&batch)?;
                taken.route(&batch)?;
            }
            scattered.route(&typed_batch(&schema, 0..0))?;

            // The rows of a bucket taken for compaction, mid-input
            let index = scattered.bucket_to_compact().unwrap_or(0);
            assert_eq!(
                scattered.take_bucket(index)?,
                taken.take_bucket(index)?,
                "taken bucket {index} at batch size {batch_size}"
            );
            scattered.put_compacted(index, None, 0);
            taken.put_compacted(index, None, 0);

            let batch = typed_batch(&schema, 9000..12_000);
            scattered.route(&batch)?;
            taken.route(&batch)?;

            let scattered = read_buckets(scattered, &schema).await?;
            let taken = read_buckets(taken, &schema).await?;
            assert_eq!(scattered.len(), taken.len());
            for (scattered, taken) in scattered.iter().zip(&taken) {
                assert_eq!(scattered, taken, "batch size {batch_size}");
                // Every batch but the last is full
                for batch in &scattered[..scattered.len() - 1] {
                    assert_eq!(batch.num_rows(), batch_size);
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_resplit_scatters_as_the_general_route_does() -> Result<()> {
        let schema = typed_schema();
        let mut level0 = buckets_of(&schema, 64, 0, true, None);
        level0.route(&typed_batch(&schema, 0..20_000))?;
        let bucket = read_buckets(level0, &schema).await?.swap_remove(0);

        let mut scattered = buckets_of(&schema, 64, 1, true, None);
        let mut taken = buckets_of(&schema, 64, 1, false, None);
        for batch in &bucket {
            scattered.route(batch)?;
            taken.route(batch)?;
        }
        let scattered = read_buckets(scattered, &schema).await?;
        assert!(scattered.len() > NUM_BUCKETS / 4, "rows spread again");
        assert_eq!(scattered, read_buckets(taken, &schema).await?);
        Ok(())
    }

    #[tokio::test]
    async fn spilled_buckets_get_the_same_rows() -> Result<()> {
        use crate::metrics::ExecutionPlanMetricsSet;
        use crate::spill::spill_manager::SpillManager;
        use datafusion_execution::runtime_env::RuntimeEnv;

        let schema = typed_schema();
        let spill_manager = || {
            Some(SpillManager::new(
                Arc::new(RuntimeEnv::default()),
                crate::metrics::SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
                Arc::clone(&schema),
            ))
        };
        let mut scattered = buckets_of(&schema, 50, 0, true, spill_manager());
        let mut taken = buckets_of(&schema, 50, 0, false, spill_manager());
        for keys in [0..4000, 4000..8000, 8000..9000] {
            let batch = typed_batch(&schema, keys);
            scattered.route(&batch)?;
            taken.route(&batch)?;
            assert!(scattered.spill_largest()?);
            assert!(taken.spill_largest()?);
        }
        let scattered = read_buckets(scattered, &schema).await?;
        let taken = read_buckets(taken, &schema).await?;
        assert_eq!(scattered.len(), taken.len());
        for (scattered, taken) in scattered.iter().zip(&taken) {
            let concat = |batches: &[RecordBatch]| {
                arrow::compute::concat_batches(&schema, batches).unwrap()
            };
            assert_eq!(concat(scattered), concat(taken));
        }
        Ok(())
    }

    #[test]
    fn a_column_the_scatter_cannot_write_routes_the_general_way() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut buckets =
            FinalBuckets::new(&schema, 2, 1024, 0, None, RouteMetrics::default());
        assert!(buckets.scatter.is_none());
        buckets.route(&test_batch(&schema, 0..1000))?;
        assert!(
            buckets
                .buckets
                .iter()
                .all(|bucket| bucket.coalescer.is_some())
        );

        let typed =
            FinalBuckets::new(&typed_schema(), 2, 1024, 0, None, RouteMetrics::default());
        assert!(typed.scatter.is_some());
        Ok(())
    }

    #[test]
    fn route_metrics_count_rows_and_bytes() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let metrics = RouteMetrics::default();
        let mut buckets = FinalBuckets::new(&schema, 1, 128, 0, None, metrics.clone());
        let keys: Int64Array = (0..1000).collect::<Vec<i64>>().into();
        let counts = arrow::array::Int32Array::from_iter_values(0..1000);
        buckets.route(&RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(keys), Arc::new(counts)],
        )?)?;
        assert_eq!(metrics.rows.value(), 1000);
        assert_eq!(metrics.bytes.value(), 1000 * 12);
        Ok(())
    }
}
