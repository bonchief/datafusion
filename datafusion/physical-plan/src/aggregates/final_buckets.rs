// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Hash buckets of partial aggregate state for the final hash aggregation.
//!
//! See [`FinalBuckets`].

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, GenericByteViewArray, UInt32Array};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow::compute::{BatchCoalescer, concat_batches, take};
use arrow::datatypes::{ByteViewType, DataType, SchemaRef, UInt64Type};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{Result, internal_err};
use datafusion_execution::async_try_stream;
use datafusion_physical_expr_common::group_hash::GroupHash;
use futures::StreamExt;

use super::bucketed_aggregation::is_fixed_width;
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

/// Hash bits the first split of fixed-width rows reads: 64 buckets, few
/// enough that each bucket's slice of a routed batch is worth a copy into its
/// coalescer. See [`FinalBuckets`].
pub(super) const FIXED_WIDTH_BITS: u32 = 6;

/// Hash bits a bucket that is split again reads, when that many are left.
const SPLIT_BITS: u32 = 6;

/// A compaction that kept more than this share of its rows found (almost) no
/// repeated groups: the bucket is spilled the next time instead.
const POOR_COMPACTION: f64 = 0.8;

/// Time and rows of [`FinalBuckets::route`], shared by the buckets of every
/// level of one aggregation partition.
#[derive(Debug, Clone, Default)]
pub(super) struct RouteMetrics {
    /// Reading or computing the rows' hashes and sorting the row indices by
    /// bucket
    pub(super) hash_time: metrics::Time,
    /// Gathering each routed batch's rows by bucket, and concatenating a
    /// bucket's pieces when it is aggregated or under memory pressure
    pub(super) move_time: metrics::Time,
    /// Rows routed into buckets, counted again at every level
    pub(super) rows: metrics::Count,
}

/// One hash bucket.
#[derive(Default)]
struct Bucket {
    /// Rows routed here and not gathered yet: one slice of each routed
    /// batch, sharing that batch's buffers. Empty if rows are coalesced.
    pending: Vec<RecordBatch>,
    /// Gathers each routed slice into batches of the bucket's own while the
    /// routed batch is in cache, if the rows are fixed width
    coalescer: Option<BatchCoalescer>,
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
/// computed here otherwise. While a routed batch is still in cache, its row
/// indices are sorted by bucket and its rows are copied in that order, the
/// bytes of a string or binary view longer than its inline 12 included (see
/// [`scatter_views`]); each bucket keeps its slice of the result. A bucket's
/// rows thus own their bytes, which lie together, and its aggregate does not
/// reach back into the batches the exchange sent. A bucket's slices are
/// concatenated, in order, when the bucket is aggregated.
///
/// Rows of fixed-width columns only take a second route: each bucket's
/// slice is copied into a [`BatchCoalescer`] of its own while the routed
/// batch is in cache, and they are split into at most
/// [`FIXED_WIDTH_BITS`] buckets. Concatenating a bucket's slices when it is
/// aggregated costs a fixed amount per slice, which a string's bytes
/// amortise and a fixed-width row does not: on ClickBench Q32 (10 M groups of
/// two integers) at 8 partitions, slices of about 64 rows from 128 buckets
/// made the final 1.23x slower than the coalesced route, and 64 buckets
/// 1.07x. A coalescer, in turn, would copy a string's bytes a second time.
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
    /// The rows' [`GroupHash`] is read from their last column, set by
    /// [`Self::read_carried`]; otherwise it is computed from their keys
    carried: bool,
    /// Memory charged for the routed batches that pending slices share
    routed_bytes: usize,
    buckets: Vec<Bucket>,
    /// `None` if spilling is not supported by the configured `DiskManager`
    spill_manager: Option<SpillManager>,
    batch_size: usize,
    metrics: RouteMetrics,
    /// Reused for hashes computed here
    hashes: Vec<u64>,
    /// Reused for the counting sort of a routed batch's rows
    starts: Vec<u32>,
    /// Every column is fixed width: buckets coalesce their rows as routed
    coalesced: bool,
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
        let coalesced = schema
            .fields()
            .iter()
            .all(|field| is_fixed_width(field.data_type()));
        let bucket = || Bucket {
            coalescer: coalesced
                .then(|| BatchCoalescer::new(Arc::clone(schema), batch_size)),
            ..Default::default()
        };
        Self {
            offset,
            bits,
            num_group_columns,
            carried: false,
            routed_bytes: 0,
            buckets: (0..1 << bits).map(|_| bucket()).collect(),
            spill_manager,
            batch_size,
            metrics,
            hashes: vec![],
            starts: vec![],
            coalesced,
        }
    }

    /// Route rows by the [`GroupHash`] they carry in their last column, for
    /// a final aggregation whose plan reads it
    /// (`AggregateExec::with_read_group_hash`), instead of hashing their keys.
    pub(super) fn read_carried(mut self) -> Self {
        self.carried = true;
        self
    }

    /// The bit offset and width of a split of one of these buckets, if
    /// enough hash bits are left for one.
    pub(super) fn next_split(&self) -> Option<(u32, u32)> {
        let offset = self.offset + self.bits;
        let bits = SPLIT_BITS.min(GroupHash::BUCKET_BITS - offset);
        (bits >= 2).then_some((offset, bits))
    }

    /// Appends every row of `batch` to its bucket.
    pub(super) fn route(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }
        if u32::try_from(num_rows).is_err() {
            return internal_err!("A batch routed into hash buckets has too many rows");
        }
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

        // Counting sort of the row indices by bucket: `starts[b]` ends as the
        // first sorted position of bucket `b`, `starts[b + 1]` as its end
        let (offset, bits) = (self.offset, self.bits);
        let bucket_of = |hash: u64| GroupHash::new(hash).bucket(offset, bits);
        let starts = &mut self.starts;
        starts.clear();
        starts.resize(self.buckets.len() + 1, 0);
        for &hash in hashes {
            starts[bucket_of(hash) + 1] += 1;
        }
        for bucket in 1..starts.len() {
            starts[bucket] += starts[bucket - 1];
        }
        let mut sorted = vec![0u32; num_rows];
        let mut cursors = starts[..self.buckets.len()].to_vec();
        for (row, &hash) in hashes.iter().enumerate() {
            let cursor = &mut cursors[bucket_of(hash)];
            sorted[*cursor as usize] = row as u32;
            *cursor += 1;
        }
        hash_timer.done();

        let _timer = self.metrics.move_time.timer();
        let indices = UInt32Array::from(sorted);
        let columns = batch
            .columns()
            .iter()
            .map(|column| match column.data_type() {
                DataType::Utf8View => scatter_views(column.as_string_view(), &indices),
                DataType::BinaryView => scatter_views(column.as_binary_view(), &indices),
                _ => Ok(take(column, &indices, None)?),
            })
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        let sorted =
            RecordBatch::try_new_with_options(batch.schema(), columns, &options)?;
        if !self.coalesced {
            self.routed_bytes += referenced_bytes(&sorted);
        }
        for (bucket, range) in self.buckets.iter_mut().zip(starts.windows(2)) {
            let (start, end) = (range[0] as usize, range[1] as usize);
            if end > start {
                bucket.append(sorted.slice(start, end - start))?;
            }
        }
        Ok(())
    }

    /// True if rows were routed since they were last gathered.
    pub(super) fn has_routed(&self) -> bool {
        self.routed_bytes > 0
            || self.buckets.iter().any(|bucket| {
                bucket
                    .coalescer
                    .as_ref()
                    .is_some_and(|coalescer| coalescer.get_buffered_rows() > 0)
            })
    }

    /// Gathers the routed rows of every bucket into batches that own their
    /// bytes, or into the bucket's spill file, and releases the routed
    /// batches.
    pub(super) fn gather(&mut self) -> Result<()> {
        if !self.has_routed() {
            return Ok(());
        }
        let _timer = self.metrics.move_time.timer();
        for bucket in &mut self.buckets {
            bucket.flush()?;
            let pending = std::mem::take(&mut bucket.pending);
            for batch in concat_pieces(pending, self.batch_size) {
                bucket.push(gc_view_arrays(&batch?)?)?;
            }
        }
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
                    bucket.batches_bytes
                        + bucket.coalescer.as_ref().map_or(0, BatchCoalescer::size)
                })
                .sum::<usize>()
            + self.hashes.capacity() * size_of::<u64>()
            + self.starts.capacity() * size_of::<u32>()
    }

    /// True if no bucket holds rows in memory.
    pub(super) fn is_fully_spilled(&self) -> bool {
        !self.has_routed() && self.buckets.iter().all(|bucket| bucket.rows == 0)
    }

    /// Finishes the buckets and returns the non-empty ones, and the memory
    /// of the routed batches they share, held until the last of them is
    /// aggregated.
    pub(super) fn into_sources(mut self) -> Result<(Vec<BucketSource>, usize)> {
        // A spilled bucket's routed rows go to its file first
        for bucket in &mut self.buckets {
            {
                let _timer = self.metrics.move_time.timer();
                bucket.flush()?;
            }
            if bucket.spill_file.is_some() && !bucket.pending.is_empty() {
                let _timer = self.metrics.move_time.timer();
                let pending = std::mem::take(&mut bucket.pending);
                for batch in concat_pieces(pending, self.batch_size) {
                    bucket.push(gc_view_arrays(&batch?)?)?;
                }
            }
        }

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
                    batch_size: self.batch_size,
                    move_time: self.metrics.move_time.clone(),
                }),
            }
        }
        Ok((sources, self.routed_bytes))
    }
}

impl Bucket {
    /// Adds a routed slice: through the coalescer, if any, or as pending.
    fn append(&mut self, slice: RecordBatch) -> Result<()> {
        match &mut self.coalescer {
            Some(coalescer) => {
                coalescer.push_batch(slice)?;
                self.drain()
            }
            None => {
                self.pending.push(slice);
                Ok(())
            }
        }
    }

    /// Finishes the coalescer's partial batch, if any.
    fn flush(&mut self) -> Result<()> {
        if let Some(coalescer) = &mut self.coalescer {
            coalescer.finish_buffered_batch()?;
        }
        self.drain()
    }

    /// Moves the coalescer's completed batches to where the bucket lives.
    fn drain(&mut self) -> Result<()> {
        while let Some(batch) = self
            .coalescer
            .as_mut()
            .and_then(BatchCoalescer::next_completed_batch)
        {
            self.push(batch)?;
        }
        Ok(())
    }

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

/// `pieces` combined into batches of at most `batch_size` rows (or one piece,
/// if larger), keeping their order.
fn concat_pieces(
    pieces: Vec<RecordBatch>,
    batch_size: usize,
) -> impl Iterator<Item = Result<RecordBatch>> {
    let mut pieces = pieces.into_iter().peekable();
    std::iter::from_fn(move || {
        let first = pieces.next()?;
        let mut rows = first.num_rows();
        let mut group = vec![first];
        while let Some(piece) =
            pieces.next_if(|piece| rows + piece.num_rows() <= batch_size)
        {
            rows += piece.num_rows();
            group.push(piece);
        }
        Some(match group.as_slice() {
            [single] => Ok(single.clone()),
            _ => concat_batches(&group[0].schema(), &group).map_err(Into::into),
        })
    })
}

/// The rows of `array` in the order `rows` lists them, with the bytes of
/// every view longer than the 12 a view holds inline copied into one buffer
/// of their own. Unlike `take`, which keeps referencing the source's buffers,
/// the result references nothing of `array`. `rows` lists every row of
/// `array` once.
fn scatter_views<T: ByteViewType + ?Sized>(
    array: &GenericByteViewArray<T>,
    rows: &UInt32Array,
) -> Result<ArrayRef> {
    const INLINE: u32 = 12;
    let views = array.views();
    let sources = array.data_buffers();
    let nulls = array.nulls().filter(|nulls| nulls.null_count() > 0);
    let valid = |row: usize| nulls.is_none_or(|nulls| nulls.is_valid(row));
    let length = |view: u128| view as u32;

    // The length of a valid view's bytes held outside the view, else 0
    let outlined = |row: usize| {
        let len = length(views[row]);
        if len > INLINE && valid(row) {
            len as usize
        } else {
            0
        }
    };

    // Where each row goes, and where its bytes go, in bucket order
    let rows = rows.values();
    if rows.len() != array.len() {
        return internal_err!("Routed rows are not a permutation of the batch");
    }
    let mut positions = vec![u32::MAX; rows.len()];
    let mut offsets = Vec::with_capacity(rows.len());
    let mut bytes = 0usize;
    for (position, &row) in rows.iter().enumerate() {
        let Some(slot) = positions
            .get_mut(row as usize)
            .filter(|slot| **slot == u32::MAX)
        else {
            return internal_err!("Routed rows are not a permutation of the batch");
        };
        *slot = position as u32;
        offsets.push(bytes);
        bytes += outlined(row as usize);
    }
    if u32::try_from(bytes).is_err() {
        return internal_err!(
            "A batch routed into hash buckets holds over 4 GiB of strings"
        );
    }

    // Read the rows in the order they lie in the source, which is the order
    // their bytes lie in, and write each where it goes
    let mut data = vec![0u8; bytes];
    let mut out = vec![0u128; rows.len()];
    for (row, &view) in views.iter().enumerate() {
        let position = positions[row] as usize;
        let len = outlined(row);
        out[position] = if len > 0 {
            // Long view: length and prefix in the low 64 bits, then the
            // buffer index and the offset into that buffer
            let source = (view >> 64) as u32 as usize;
            let start = (view >> 96) as u32 as usize;
            let offset = offsets[position];
            data[offset..offset + len]
                .copy_from_slice(&sources[source][start..start + len]);
            (view & u128::from(u64::MAX)) | ((offset as u128) << 96)
        } else if valid(row) {
            view
        } else {
            0
        };
    }
    let nulls = nulls.map(|nulls| {
        NullBuffer::new(BooleanBuffer::collect_bool(rows.len(), |i| {
            nulls.is_valid(rows[i] as usize)
        }))
    });
    let buffers = match data.is_empty() {
        true => vec![],
        false => vec![Buffer::from_vec(data)],
    };
    // SAFETY: `rows` is a permutation of `array`'s rows (the counting sort of
    // `FinalBuckets::route`), so every view is written once. Each is a valid
    // view of `array`, or zero for a null, and every long view now points at
    // an identical copy of its bytes in buffer 0, so the views are as valid
    // as `array`'s.
    let array = unsafe {
        GenericByteViewArray::<T>::new_unchecked(
            ScalarBuffer::from(out),
            buffers.into(),
            nulls,
        )
    };
    Ok(Arc::new(array))
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
        pending: Vec<RecordBatch>,
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
            Self::Memory { bytes, .. } => *bytes,
            Self::Spilled { .. } => 0,
        }
    }

    pub(super) fn into_stream(self, schema: &SchemaRef) -> SendableRecordBatchStream {
        match self {
            Self::Memory {
                batches,
                pending,
                batch_size,
                move_time,
                ..
            } => {
                let mut pieces = concat_pieces(pending, batch_size);
                let gathered = futures::stream::iter(std::iter::from_fn(move || {
                    let _timer = move_time.timer();
                    pieces.next()
                }));
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

    use arrow::array::{
        BooleanArray, Int64Array, StringArray, StringViewArray, UInt64Array,
    };
    use arrow::datatypes::{Field, Int64Type, Schema};
    use datafusion_physical_expr_common::group_hash::{
        group_hash_field, group_hash_keys,
    };
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
        if schema
            .fields()
            .last()
            .is_some_and(|field| group_hash_keys(field).is_some())
        {
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
        buckets.route(&test_batch(&schema, 0..20_000))?;
        buckets.route(&test_batch(&schema, 10_000..30_000))?;
        assert!(buckets.memory_size() > 0);
        // Half of the rows are gathered early, as under memory pressure
        buckets.gather()?;
        assert!(!buckets.has_routed());
        buckets.route(&test_batch(&schema, 0..10_000))?;

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
    async fn routed_strings_own_their_bytes() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, true),
            Field::new("v", DataType::Int64, false),
        ]));
        // Short (inline), long and null names
        let name = |k: i64| match k % 3 {
            0 => Some(format!("k{k}")),
            1 => Some(format!("a string longer than twelve bytes {k}")),
            _ => None,
        };
        let names: StringViewArray = (0..5_000).map(name).collect();
        let source: Vec<_> = names
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_ptr_range())
            .collect();
        let values: Int64Array = (0..5_000).collect::<Vec<_>>().into();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(names), Arc::new(values)],
        )?;
        let mut buckets =
            FinalBuckets::new(&schema, 1, 1024, 0, 6, None, RouteMetrics::default());
        buckets.route(&batch)?;
        drop(batch);

        let mut seen = 0;
        for source_rows in buckets.into_sources()?.0 {
            let mut stream = source_rows.into_stream(&schema);
            while let Some(batch) = stream.next().await.transpose()? {
                let names = batch.column(0).as_string_view();
                for buffer in names.data_buffers().iter() {
                    let start = buffer.as_ptr();
                    assert!(
                        !source.iter().any(|range| range.contains(&start)),
                        "a routed string references the batch it came from"
                    );
                }
                let values = batch.column(1).as_primitive::<Int64Type>();
                for (got, k) in names.iter().zip(values.values()) {
                    assert_eq!(got.map(str::to_owned), name(*k));
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 5_000);
        Ok(())
    }

    #[tokio::test]
    async fn fixed_width_rows_are_coalesced_as_routed() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("flag", DataType::Boolean, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = |keys: std::ops::Range<i64>| {
            let flags: BooleanArray = keys.clone().map(|k| Some(k % 2 == 0)).collect();
            let values: Int64Array =
                keys.clone().map(|k| k * 10).collect::<Vec<_>>().into();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys.collect::<Vec<_>>())),
                    Arc::new(flags),
                    Arc::new(values),
                ],
            )
        };
        let mut buckets = FinalBuckets::new(
            &schema,
            1,
            1024,
            0,
            FIXED_WIDTH_BITS,
            None,
            RouteMetrics::default(),
        );
        buckets.route(&batch(0..20_000)?)?;
        // Rows wait in the coalescers, charged as memory, until gathered
        assert!(buckets.has_routed());
        assert!(buckets.memory_size() > 0);
        buckets.gather()?;
        assert!(!buckets.has_routed());
        buckets.route(&batch(20_000..30_000)?)?;

        let (sources, routed_bytes) = buckets.into_sources()?;
        assert_eq!(routed_bytes, 0, "coalesced rows share no routed batch");
        let rows = drain(sources, &schema).await?;
        assert_eq!(rows.len(), 30_000);
        let mut bucket_of_key: HashMap<i64, usize> = HashMap::new();
        for (bucket, key, value) in rows {
            assert_eq!(value, key * 10, "columns stay aligned");
            assert!(
                bucket_of_key.insert(key, bucket).is_none(),
                "key {key} twice"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn routes_by_the_carried_hash() -> Result<()> {
        let schema = schema(true);
        let mut buckets =
            FinalBuckets::new(&schema, 2, 1024, 0, 6, None, RouteMetrics::default())
                .read_carried();
        buckets.route(&test_batch(&schema, 0..6_400))?;
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
        first.route(&test_batch(&schema, 0..200_000))?;
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
            second.route(&batch)?;
        }
        assert!(second.into_sources()?.0.len() > 32);
        Ok(())
    }
}
