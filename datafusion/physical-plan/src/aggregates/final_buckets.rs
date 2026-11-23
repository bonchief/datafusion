// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Hash buckets of partial aggregate state for the final hash aggregation.
//!
//! See [`FinalBuckets`].

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, GenericByteViewArray};
use arrow::compute::{BatchCoalescer, interleave_record_batch};
use arrow::datatypes::{ByteViewType, DataType, SchemaRef, UInt64Type};
use arrow::record_batch::RecordBatch;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{Result, internal_err};
use datafusion_execution::async_try_stream;
use datafusion_physical_expr_common::group_hash::GroupHash;
use futures::StreamExt;

use super::aggregate_hash_table::carries_group_hash;
use crate::SendableRecordBatchStream;
use crate::metrics;
use crate::spill::gc_view_arrays;
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;

/// Fewest and most hash bits the first split of a table reads: 16 to 1,024
/// buckets.
pub(super) const MIN_FANOUT_BITS: u32 = 4;
pub(super) const MAX_FANOUT_BITS: u32 = 10;

/// Hash bits a bucket that is split again reads, when that many are left.
const SPLIT_BITS: u32 = 6;

/// A compaction that kept more than this share of its rows found (almost) no
/// repeated groups: the bucket is spilled the next time instead.
const POOR_COMPACTION: f64 = 0.8;

/// Time and rows of [`FinalBuckets::route`], shared by the buckets of every
/// level of one aggregation partition.
#[derive(Debug, Clone, Default)]
pub(super) struct RouteMetrics {
    /// Reading or computing the rows' hashes and recording each row's bucket
    pub(super) hash_time: metrics::Time,
    /// Gathering the rows of a bucket into batches of its own, which happens
    /// when the bucket is aggregated, or earlier under memory pressure
    pub(super) move_time: metrics::Time,
    /// Rows routed into buckets, counted again at every level
    pub(super) rows: metrics::Count,
}

/// One hash bucket.
#[derive(Default)]
struct Bucket {
    /// Rows routed here and not gathered yet: `batch << 32 | row`, where
    /// `batch` indexes [`FinalBuckets::routed`].
    pending: Vec<u64>,
    /// Gathered rows, or the state a compaction left of them
    batches: VecDeque<RecordBatch>,
    /// Memory held by `batches`
    batches_bytes: usize,
    /// Rows in `batches`
    rows: usize,
    /// Set once the bucket has been spilled: gathered rows go here
    spill_file: Option<InProgressSpillFile>,
    /// Rows the last compaction left, if any
    compacted_rows: Option<usize>,
    /// The last compaction kept more than [`POOR_COMPACTION`] of its rows
    poor: bool,
}

/// Rows of partial aggregate state, split by the bits of their
/// [`GroupHash`].
///
/// Once the table of a final hash aggregation has outgrown the CPU caches,
/// every probe and every accumulator update of a further row is a cache
/// miss. The stream then stops growing that table: it moves the table's
/// state in here, routes the rest of its input here without aggregating it,
/// and finally aggregates the buckets one after another, each with a table
/// a fraction of the size. Buckets are independent, so they can be spilled,
/// read back and released one at a time.
///
/// # Routing
///
/// A row's bucket is read from bits of its [`GroupHash`] (see its bit layout),
/// which the row carries in its last column when the partial aggregation
/// computed it (see `AggregateExec::with_group_hash`), and which is
/// computed here otherwise. Routing records the bucket of every row and keeps
/// the batch as it came: a bucket's rows are gathered into batches of its own
/// only when the bucket is aggregated, by one `interleave` that copies the
/// 16-byte view of a string and not its bytes.
///
/// # Memory pressure
///
/// Routed batches are charged at the bytes their rows reference. Under memory
/// pressure every bucket gathers its rows into batches that own their bytes,
/// releasing the routed batches. Then the largest bucket is compacted (its
/// rows are aggregated with a small table and replaced by the state, one row
/// per group) unless its last compaction found nothing to merge or it has
/// not doubled since, in which case it is spilled. Nothing is compacted
/// while memory suffices, so input that does not repeat its groups is
/// aggregated once.
pub(super) struct FinalBuckets {
    /// First bucket bit this split reads, and how many: see [`GroupHash::bucket`]
    offset: u32,
    bits: u32,
    /// Number of leading columns of a batch that are the group keys
    num_group_columns: usize,
    /// The rows carry their [`GroupHash`] in their last column
    carried: bool,
    /// Batches routed since rows were last gathered
    routed: Vec<RecordBatch>,
    /// Memory charged for `routed`
    routed_bytes: usize,
    buckets: Vec<Bucket>,
    /// `None` if spilling is not supported by the configured `DiskManager`
    spill_manager: Option<SpillManager>,
    batch_size: usize,
    metrics: RouteMetrics,
    /// Reused for hashes computed here
    hashes: Vec<u64>,
}

impl FinalBuckets {
    /// Buckets of rows of `schema` that read `bits` bucket bits from `offset`.
    pub(super) fn new(
        schema: &SchemaRef,
        num_group_columns: usize,
        batch_size: usize,
        offset: u32,
        bits: u32,
        spill_manager: Option<SpillManager>,
        metrics: RouteMetrics,
    ) -> Self {
        debug_assert!(bits > 0 && offset + bits <= GroupHash::BUCKET_BITS);
        Self {
            offset,
            bits,
            num_group_columns,
            carried: carries_group_hash(schema),
            routed: vec![],
            routed_bytes: 0,
            buckets: (0..1 << bits).map(|_| Bucket::default()).collect(),
            spill_manager,
            batch_size,
            metrics,
            hashes: vec![],
        }
    }

    /// The bit offset and width of a split of one of these buckets, if
    /// enough hash bits are left for one.
    pub(super) fn next_split(&self) -> Option<(u32, u32)> {
        let offset = self.offset + self.bits;
        let bits = SPLIT_BITS.min(GroupHash::BUCKET_BITS - offset);
        (bits >= 2).then_some((offset, bits))
    }

    /// Records the bucket of every row of `batch` and keeps the batch.
    pub(super) fn route(&mut self, batch: RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }
        let Ok(index) = u32::try_from(self.routed.len()) else {
            return internal_err!("Too many batches routed into hash buckets");
        };
        self.metrics.rows.add(num_rows);
        let hash_timer = self.metrics.hash_time.timer();

        let hashes: &[u64] = if self.carried {
            match batch
                .columns()
                .last()
                .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            {
                Some(hashes) => hashes.values(),
                None => return internal_err!("Bucketed rows have no group hash column"),
            }
        } else {
            self.hashes.clear();
            self.hashes.resize(num_rows, 0);
            let keys: &[ArrayRef] = &batch.columns()[..self.num_group_columns];
            create_hashes(keys, &GroupHash::STATE, &mut self.hashes)?;
            &self.hashes
        };
        let index = u64::from(index) << 32;
        for (row, &hash) in hashes.iter().enumerate() {
            let bucket = GroupHash::new(hash).bucket(self.offset, self.bits);
            self.buckets[bucket].pending.push(index | row as u64);
        }
        hash_timer.done();

        self.routed_bytes += referenced_bytes(&batch);
        self.routed.push(batch);
        Ok(())
    }

    /// True if rows were routed since they were last gathered.
    pub(super) fn has_routed(&self) -> bool {
        !self.routed.is_empty()
    }

    /// Gathers the routed rows of every bucket into batches that own their
    /// bytes, or into the bucket's spill file, and releases the routed
    /// batches.
    pub(super) fn gather(&mut self) -> Result<()> {
        if self.routed.is_empty() {
            return Ok(());
        }
        let _timer = self.metrics.move_time.timer();
        let routed: Vec<&RecordBatch> = self.routed.iter().collect();
        for bucket in &mut self.buckets {
            let pending = std::mem::take(&mut bucket.pending);
            for rows in pending.chunks(self.batch_size) {
                let batch = gc_view_arrays(&interleave(&routed, rows)?)?;
                bucket.push(batch)?;
            }
        }
        self.routed.clear();
        self.routed_bytes = 0;
        Ok(())
    }

    /// The largest bucket held in memory, if any. Call [`Self::gather`] first.
    pub(super) fn largest(&self) -> Option<usize> {
        self.buckets
            .iter()
            .enumerate()
            .filter(|(_, bucket)| bucket.spill_file.is_none() && bucket.rows > 0)
            .max_by_key(|(_, bucket)| bucket.batches_bytes)
            .map(|(index, _)| index)
    }

    /// True if compacting bucket `index` may pay: its last compaction, if
    /// any, merged rows, and it has received at least as many rows since.
    pub(super) fn should_compact(&self, index: usize) -> bool {
        let bucket = &self.buckets[index];
        !bucket.poor
            && bucket.rows
                >= self
                    .batch_size
                    .max(2 * bucket.compacted_rows.unwrap_or_default())
    }

    /// Removes and returns the gathered rows of an in-memory bucket, to be
    /// followed by [`Self::put_compacted`].
    pub(super) fn take_bucket(&mut self, index: usize) -> Vec<RecordBatch> {
        let bucket = &mut self.buckets[index];
        debug_assert!(bucket.spill_file.is_none() && bucket.pending.is_empty());
        bucket.batches_bytes = 0;
        bucket.rows = 0;
        bucket.batches.drain(..).collect()
    }

    /// Puts back the aggregated state of the `input_rows` rows that
    /// [`Self::take_bucket`] returned.
    pub(super) fn put_compacted(
        &mut self,
        index: usize,
        state: Option<RecordBatch>,
        input_rows: usize,
    ) -> Result<()> {
        let bucket = &mut self.buckets[index];
        let state_rows = state.as_ref().map_or(0, |state| state.num_rows());
        if let Some(state) = state {
            bucket.push(state)?;
        }
        bucket.compacted_rows = Some(state_rows);
        bucket.poor = state_rows as f64 > POOR_COMPACTION * input_rows as f64;
        Ok(())
    }

    /// Spills the gathered rows of bucket `index`, which from then on gathers
    /// its rows into the spill file. Returns false if spilling is not
    /// supported.
    pub(super) fn spill(&mut self, index: usize) -> Result<bool> {
        let Some(spill_manager) = &self.spill_manager else {
            return Ok(false);
        };
        let bucket = &mut self.buckets[index];
        let mut spill_file =
            spill_manager.create_in_progress_file("FinalHashAggregateBucket")?;
        for batch in bucket.batches.drain(..) {
            spill_file.append_batch(&batch)?;
        }
        bucket.spill_file = Some(spill_file);
        bucket.batches_bytes = 0;
        bucket.rows = 0;
        Ok(true)
    }

    /// Memory held by the buckets
    pub(super) fn memory_size(&self) -> usize {
        self.routed_bytes
            + self
                .buckets
                .iter()
                .map(|bucket| {
                    bucket.batches_bytes + bucket.pending.capacity() * size_of::<u64>()
                })
                .sum::<usize>()
            + self.hashes.capacity() * size_of::<u64>()
    }

    /// True if no bucket holds rows in memory.
    pub(super) fn is_fully_spilled(&self) -> bool {
        self.routed.is_empty() && self.buckets.iter().all(|bucket| bucket.rows == 0)
    }

    /// Finishes the buckets and returns the non-empty ones, and the memory
    /// of the routed batches they share, held until the last of them is
    /// aggregated.
    pub(super) fn into_sources(mut self) -> Result<(Vec<BucketSource>, usize)> {
        // A spilled bucket's routed rows go to its file first
        if self
            .buckets
            .iter()
            .any(|bucket| bucket.spill_file.is_some() && !bucket.pending.is_empty())
        {
            let _timer = self.metrics.move_time.timer();
            let routed: Vec<&RecordBatch> = self.routed.iter().collect();
            for bucket in &mut self.buckets {
                if bucket.spill_file.is_none() {
                    continue;
                }
                let pending = std::mem::take(&mut bucket.pending);
                for rows in pending.chunks(self.batch_size) {
                    bucket.push(interleave(&routed, rows)?)?;
                }
            }
        }

        let routed = Arc::new(std::mem::take(&mut self.routed));
        let mut sources = Vec::with_capacity(self.buckets.len());
        for bucket in self.buckets {
            match bucket.spill_file {
                Some(mut spill_file) => {
                    let Some(spill_manager) = &self.spill_manager else {
                        return internal_err!("A spilled bucket has no spill manager");
                    };
                    if let Some(file) = spill_file.finish()? {
                        sources.push(BucketSource::Spilled {
                            stream: spill_manager.read_spill_as_stream(file, None)?,
                            batch_size: self.batch_size,
                        });
                    }
                }
                None if bucket.batches.is_empty() && bucket.pending.is_empty() => {}
                None => sources.push(BucketSource::Memory {
                    batches: bucket.batches,
                    bytes: bucket.batches_bytes,
                    pending: bucket.pending,
                    routed: Arc::clone(&routed),
                    batch_size: self.batch_size,
                    move_time: self.metrics.move_time.clone(),
                }),
            }
        }
        Ok((sources, self.routed_bytes))
    }
}

impl Bucket {
    /// Adds gathered rows to where the bucket lives: its spill file if it has
    /// been spilled, memory otherwise.
    fn push(&mut self, batch: RecordBatch) -> Result<()> {
        match &mut self.spill_file {
            Some(spill_file) => {
                spill_file.append_batch(&batch)?;
            }
            None => {
                self.rows += batch.num_rows();
                self.batches_bytes += batch.get_array_memory_size();
                self.batches.push_back(batch);
            }
        }
        Ok(())
    }
}

/// The rows `rows` (`batch << 32 | row`) of `routed`, as one batch.
fn interleave(routed: &[&RecordBatch], rows: &[u64]) -> Result<RecordBatch> {
    let indices: Vec<(usize, usize)> = rows
        .iter()
        .map(|&row| ((row >> 32) as usize, (row & u64::from(u32::MAX)) as usize))
        .collect();
    Ok(interleave_record_batch(routed, &indices)?)
}

/// Bytes the rows of `batch` take, counting only the bytes of a string or
/// binary view that its row references: a routed batch shares the data
/// buffers of the batch it was taken from with the other partitions' rows.
fn referenced_bytes(batch: &RecordBatch) -> usize {
    fn views<T: ByteViewType + ?Sized>(array: &GenericByteViewArray<T>) -> usize {
        let outlined: usize = array
            .views()
            .iter()
            .map(|&view| match view as u32 {
                len if len > 12 => len as usize,
                _ => 0,
            })
            .sum();
        array.views().inner().len()
            + outlined
            + array.nulls().map_or(0, |nulls| nulls.buffer().len())
    }
    batch
        .columns()
        .iter()
        .map(|column| match column.data_type() {
            DataType::Utf8View => views(column.as_string_view()),
            DataType::BinaryView => views(column.as_binary_view()),
            _ => column
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| column.get_array_memory_size()),
        })
        .sum()
}

/// The rows of one finished bucket
pub(super) enum BucketSource {
    /// Gathered rows, then rows still to be gathered from the routed batches
    Memory {
        batches: VecDeque<RecordBatch>,
        bytes: usize,
        pending: Vec<u64>,
        routed: Arc<Vec<RecordBatch>>,
        batch_size: usize,
        move_time: metrics::Time,
    },
    /// Rows were appended to the file in pieces smaller than a batch
    Spilled {
        stream: SendableRecordBatchStream,
        batch_size: usize,
    },
}

impl BucketSource {
    /// Memory held by the bucket's own rows: the routed batches it shares
    /// are counted by [`FinalBuckets::into_sources`].
    pub(super) fn memory_size(&self) -> usize {
        match self {
            Self::Memory { bytes, pending, .. } => {
                bytes + pending.capacity() * size_of::<u64>()
            }
            Self::Spilled { .. } => 0,
        }
    }

    pub(super) fn into_stream(self, schema: &SchemaRef) -> SendableRecordBatchStream {
        match self {
            Self::Memory {
                batches,
                pending,
                routed,
                batch_size,
                move_time,
                ..
            } => {
                let gathered = futures::stream::iter(
                    (0..pending.len())
                        .step_by(batch_size.max(1))
                        .map(move |start| {
                            let _timer = move_time.timer();
                            let rows =
                                &pending[start..(start + batch_size).min(pending.len())];
                            let routed: Vec<&RecordBatch> = routed.iter().collect();
                            interleave(&routed, rows)
                        }),
                );
                Box::pin(RecordBatchStreamAdapter::new(
                    Arc::clone(schema),
                    futures::stream::iter(batches.into_iter().map(Ok)).chain(gathered),
                ))
            }
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

    use arrow::array::{Int64Array, StringArray, UInt64Array};
    use arrow::datatypes::{Field, Int64Type, Schema};
    use datafusion_physical_expr_common::group_hash::group_hash_field;
    use std::collections::HashMap;

    fn schema(carried: bool) -> SchemaRef {
        let mut fields = vec![
            Field::new("k", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ];
        if carried {
            fields.push(group_hash_field(&[0, 1]));
        }
        Arc::new(Schema::new(fields))
    }

    fn test_batch(schema: &SchemaRef, keys: std::ops::Range<i64>) -> RecordBatch {
        let names: StringArray = keys
            .clone()
            .map(|k| Some(format!("key-{}", k % 97)))
            .collect();
        let values: Int64Array = keys.clone().map(|k| k * 10).collect::<Vec<_>>().into();
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(keys.clone().collect::<Vec<_>>())),
            Arc::new(names),
            Arc::new(values),
        ];
        if carries_group_hash(schema) {
            // A hash that puts key k in bucket k % 64 of the first 6 bits
            let hashes: UInt64Array = keys
                .map(|k| ((k as u64) % 64) << GroupHash::BUCKET_SHIFT)
                .collect::<Vec<_>>()
                .into();
            columns.push(Arc::new(hashes));
        }
        RecordBatch::try_new(Arc::clone(schema), columns).unwrap()
    }

    /// Every row of every source, as `(bucket, key, value)`.
    async fn drain(
        sources: Vec<BucketSource>,
        schema: &SchemaRef,
    ) -> Result<Vec<(usize, i64, i64)>> {
        let mut rows = vec![];
        for (bucket, source) in sources.into_iter().enumerate() {
            let mut stream = source.into_stream(schema);
            while let Some(batch) = stream.next().await.transpose()? {
                let keys = batch.column(0).as_primitive::<Int64Type>();
                let values = batch.column(2).as_primitive::<Int64Type>();
                for (key, value) in keys.values().iter().zip(values.values()) {
                    rows.push((bucket, *key, *value));
                }
            }
        }
        Ok(rows)
    }

    #[tokio::test]
    async fn routes_every_row_once_and_keeps_keys_together() -> Result<()> {
        let schema = schema(false);
        let mut buckets =
            FinalBuckets::new(&schema, 2, 1024, 0, 6, None, RouteMetrics::default());
        // the same keys arrive in two different batches
        buckets.route(test_batch(&schema, 0..20_000))?;
        buckets.route(test_batch(&schema, 10_000..30_000))?;
        assert!(buckets.memory_size() > 0);
        // Half of the rows are gathered early, as under memory pressure
        buckets.gather()?;
        assert!(!buckets.has_routed());
        buckets.route(test_batch(&schema, 0..10_000))?;

        let (sources, routed_bytes) = buckets.into_sources()?;
        assert!(routed_bytes > 0);
        assert!(sources.len() > 32, "rows spread over the buckets");

        let rows = drain(sources, &schema).await?;
        assert_eq!(rows.len(), 50_000);
        let mut bucket_of_key: HashMap<i64, usize> = HashMap::new();
        for (bucket, key, value) in rows {
            assert_eq!(value, key * 10, "columns stay aligned");
            let seen = *bucket_of_key.entry(key).or_insert(bucket);
            assert_eq!(seen, bucket, "key {key} in two buckets");
        }
        assert_eq!(bucket_of_key.len(), 30_000);
        Ok(())
    }

    #[tokio::test]
    async fn routes_by_the_carried_hash() -> Result<()> {
        let schema = schema(true);
        let mut buckets =
            FinalBuckets::new(&schema, 2, 1024, 0, 6, None, RouteMetrics::default());
        buckets.route(test_batch(&schema, 0..6_400))?;
        let (sources, _) = buckets.into_sources()?;
        assert_eq!(sources.len(), 64);
        for (bucket, key, _) in drain(sources, &schema).await? {
            assert_eq!(key as usize % 64, bucket);
        }
        Ok(())
    }

    #[tokio::test]
    async fn splits_read_the_next_bits() -> Result<()> {
        // Rows of one bucket of the first split spread again at the next one
        let schema = schema(false);
        let mut first =
            FinalBuckets::new(&schema, 1, 1024, 0, 10, None, RouteMetrics::default());
        let (offset, bits) = first.next_split().unwrap();
        assert_eq!((offset, bits), (10, 6));
        first.route(test_batch(&schema, 0..200_000))?;
        let (mut sources, _) = first.into_sources()?;
        let mut stream = sources.swap_remove(0).into_stream(&schema);
        let mut second = FinalBuckets::new(
            &schema,
            1,
            1024,
            offset,
            bits,
            None,
            RouteMetrics::default(),
        );
        assert_eq!(second.next_split(), None, "16 bucket bits are used up");
        while let Some(batch) = stream.next().await.transpose()? {
            second.route(batch)?;
        }
        assert!(second.into_sources()?.0.len() > 32);
        Ok(())
    }
}
