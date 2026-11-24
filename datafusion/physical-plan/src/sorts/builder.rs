// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::spill::get_record_batch_memory_size;
use arrow::array::{Array, ArrayData, ArrayRef};
use arrow::compute::interleave;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow::util::bit_util;
use datafusion_common::utils::memory::RecordBatchMemoryTracker;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_execution::memory_pool::MemoryReservation;
use log::warn;
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
struct BatchCursor {
    /// The index into BatchBuilder::batches
    batch_idx: usize,
    /// The row index within the given batch
    row_idx: usize,
}

impl BatchCursor {
    /// A cursor whose batch has been released. `push_row` must not be called
    /// for the stream until `push_batch` installs a new cursor.
    const RELEASED: Self = Self {
        batch_idx: usize::MAX,
        row_idx: 0,
    };
}

/// Provides an API to incrementally build a [`RecordBatch`] from partitioned [`RecordBatch`]
///
/// # Memory accounting
///
/// The builder's reservation covers what the merge keeps alive and what it
/// allocates, each exactly once:
///
/// * **Held input.** Every buffer of every live input batch is charged once,
///   keyed by the address of its allocation, at the largest capacity seen
///   for it. Slices, `take` and filter results of string view columns keep
///   all variadic data buffers of their source, so many live batches (several
///   sorted runs, or the slices of one run) can point at the same
///   allocation; it is charged once, not once per batch. The charge belongs
///   to the builder's buffer table, not to the batch that first brought the
///   buffer: when that batch is released while another live batch still
///   points at the buffer, the charge stays until the last live batch that
///   references it is released. So nothing the builder keeps alive is ever
///   uncharged, and nothing is charged twice.
/// * **Output.** Before interleaving, the output is admitted at what
///   interleaving newly allocates ([`fresh`]): a view column allocates its
///   16-byte views and nulls and shares the data buffers already charged as
///   held input. The admission is released once the batch is built; its
///   consumer charges the batch on receipt, as it does any input batch.
///   When the pool cannot admit the output, the builder emits half as many
///   rows, down to one, before it fails. It halves first against the room
///   the pool reports ([`MemoryReservation::room`]), but not below an eighth
///   of the batch size: an output it can see does not fit is not asked for,
///   since a pool may record a refused ask (as the ask its next attempt
///   needs) even when the caller recovers, while an output of that floor is
///   asked for whatever the room, since a pool may fund an ask past its room
///   (from its node) and a merge starved into one-row batches would leave
///   every consumer charging the shared buffers each tiny batch points at.
///   Only a refused ask halves below the floor.
#[derive(Debug)]
pub struct BatchBuilder {
    /// The schema of the RecordBatches yielded by this stream
    schema: SchemaRef,

    /// Maintain a list of [`RecordBatch`] and their corresponding stream
    /// Maintain a list of [`RecordBatch`], their corresponding stream and the
    /// bytes per row that interleaving one of their rows allocates
    batches: Vec<(usize, RecordBatch, usize)>,

    /// Accounts for memory used by buffered batches.
    ///
    /// May include pre-reserved bytes (from `sort_spill_reservation_bytes`)
    /// that were transferred via [`MemoryReservation::take()`] to prevent
    /// starvation when concurrent sort partitions compete for pool memory.
    reservation: MemoryReservation,

    /// The buffers of the live batches, each counted once however many
    /// live batches share it (not including pre-reserved bytes). Its total
    /// lets [`Self::push_batch`] skip pool allocation requests when the
    /// pre-reserved bytes cover the batch.
    held: RecordBatchMemoryTracker,

    /// The initial reservation size at construction time. When the reservation
    /// is pre-loaded with `sort_spill_reservation_bytes` (via `take()`), this
    /// records that amount so we never shrink below it, maintaining the
    /// anti-starvation guarantee throughout the merge.
    initial_reservation: usize,

    /// The current [`BatchCursor`] for each stream
    cursors: Vec<BatchCursor>,

    /// The accumulated stream indexes from which to pull rows
    /// Consists of a tuple of `(batch_idx, row_idx)`
    indices: Vec<(usize, usize)>,

    /// The fewest rows an output is halved to against the pool's room
    /// before it is asked for: an eighth of the batch size.
    floor: usize,
}

impl BatchBuilder {
    /// Create a new [`BatchBuilder`] with the provided `stream_count` and `batch_size`
    pub fn new(
        schema: SchemaRef,
        stream_count: usize,
        batch_size: usize,
        reservation: MemoryReservation,
    ) -> Self {
        let initial_reservation = reservation.size();
        Self {
            schema,
            batches: Vec::with_capacity(stream_count * 2),
            cursors: vec![BatchCursor::RELEASED; stream_count],
            indices: Vec::with_capacity(batch_size),
            reservation,
            held: RecordBatchMemoryTracker::new(),
            initial_reservation,
            floor: (batch_size / 8).max(1),
        }
    }

    /// Append a new batch in `stream_idx`, returning its own memory size
    /// (its buffers counted once, whether or not other live batches share
    /// them). The reservation grows only by the buffers no live batch held.
    pub fn push_batch(&mut self, stream_idx: usize, batch: RecordBatch) -> Result<usize> {
        let size = get_record_batch_memory_size(&batch);
        let width = fresh(&batch)?.div_ceil(batch.num_rows().max(1));
        self.held.count_batch(&batch);
        // Only request additional memory from the pool when actual batch
        // usage exceeds the current reservation (which may include
        // pre-reserved bytes from sort_spill_reservation_bytes).
        let held = self.held_bytes();
        if let Err(e) = try_grow_reservation_to_at_least(&mut self.reservation, held) {
            // The batch is not kept, so it no longer holds its buffers.
            self.held.uncount_batch(&batch);
            return Err(e);
        }
        let batch_idx = self.batches.len();
        self.batches.push((stream_idx, batch, width));
        self.cursors[stream_idx] = BatchCursor {
            batch_idx,
            row_idx: 0,
        };
        Ok(size)
    }

    /// Append the next row from `stream_idx`
    pub fn push_row(&mut self, stream_idx: usize) {
        debug_assert!(
            self.batches
                .get(self.cursors[stream_idx].batch_idx)
                .is_some_and(|(_, batch, _)| {
                    self.cursors[stream_idx].row_idx < batch.num_rows()
                }),
            "push_row on stream {stream_idx} with no live batch"
        );
        let cursor = &mut self.cursors[stream_idx];
        let row_idx = cursor.row_idx;
        cursor.row_idx += 1;
        self.indices.push((cursor.batch_idx, row_idx));
    }

    /// Bytes of the buffers the live batches hold, each counted once
    fn held_bytes(&self) -> usize {
        self.held.memory_usage()
    }

    /// Returns the number of in-progress rows in this [`BatchBuilder`]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this [`BatchBuilder`] contains no in-progress rows
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the schema of this [`BatchBuilder`]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Try to interleave all columns using the given index slice.
    fn try_interleave_columns(
        &self,
        indices: &[(usize, usize)],
    ) -> Result<Vec<ArrayRef>> {
        let mut arrays: Vec<&dyn Array> = Vec::with_capacity(self.batches.len());
        (0..self.schema.fields.len())
            .map(|column_idx| {
                arrays.clear();
                arrays.extend(
                    self.batches
                        .iter()
                        .map(|(_, batch, _)| batch.column(column_idx).as_ref()),
                );
                // Arrow 58.1.0+ returns OffsetOverflowError directly from
                // interleave, allowing retry_interleave to shrink the batch.
                interleave(&arrays, indices).map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Builds a record batch from the first `rows_to_emit` buffered rows.
    fn finish_record_batch(
        &mut self,
        rows_to_emit: usize,
        columns: Vec<ArrayRef>,
    ) -> Result<RecordBatch> {
        // Remove consumed indices, keeping any remaining for the next call.
        self.indices.drain(..rows_to_emit);

        if self.indices.is_empty() {
            self.retain_cursor_batches();
        } else {
            self.retain_live_batches();
        }

        // Release excess memory back to the pool, but never shrink below
        // initial_reservation to maintain the anti-starvation guarantee
        // for the merge phase. The output's admission ends here: its
        // consumer charges it on receipt, as it does any input batch.
        let target = self.held_bytes().max(self.initial_reservation);
        if self.reservation.size() > target {
            self.reservation.shrink(self.reservation.size() - target);
        }

        RecordBatch::try_new(Arc::clone(&self.schema), columns).map_err(Into::into)
    }

    fn retain_cursor_batches(&mut self) {
        // New cursors are only created once the previous cursor for the stream
        // is finished. This means all remaining rows from all but the last batch
        // for each stream have been yielded to the newly created record batch
        //
        // We can therefore drop all but the last live cursor batch for each stream
        let mut batch_idx = 0;
        let mut retained = 0;
        self.batches.retain(|(stream_idx, batch, _)| {
            let stream_cursor = &mut self.cursors[*stream_idx];
            let is_cursor_batch = stream_cursor.batch_idx == batch_idx;
            let retain = is_cursor_batch && stream_cursor.row_idx < batch.num_rows();
            batch_idx += 1;

            if retain {
                stream_cursor.batch_idx = retained;
                retained += 1;
            } else {
                if is_cursor_batch {
                    *stream_cursor = BatchCursor::RELEASED;
                }
                self.held.uncount_batch(batch);
            }
            retain
        });
    }

    fn retain_live_batches(&mut self) {
        let mut retain_batch = vec![false; self.batches.len()];
        for (batch_idx, _) in &self.indices {
            retain_batch[*batch_idx] = true;
        }

        for cursor in &self.cursors {
            if self
                .batches
                .get(cursor.batch_idx)
                .is_some_and(|(_, batch, _)| cursor.row_idx < batch.num_rows())
            {
                retain_batch[cursor.batch_idx] = true;
            }
        }

        let mut batch_idx = 0;
        let mut retained = 0;
        let mut remap = vec![usize::MAX; self.batches.len()];
        self.batches.retain(|(_, batch, _)| {
            let retain = retain_batch[batch_idx];
            if retain {
                remap[batch_idx] = retained;
                retained += 1;
            } else {
                self.held.uncount_batch(batch);
            }
            batch_idx += 1;
            retain
        });

        for (batch_idx, _) in &mut self.indices {
            *batch_idx = remap[*batch_idx];
        }
        for cursor in &mut self.cursors {
            if let Some(new_idx) = remap.get(cursor.batch_idx) {
                // `usize::MAX` means the cursor's batch was released.
                cursor.batch_idx = *new_idx;
            }
        }
    }

    /// Bytes the reservation must hold to emit the first `rows` rows: the
    /// held input plus the output interleaving them allocates.
    fn needed(&self, rows: usize) -> Result<usize> {
        let estimate = self.indices[..rows]
            .iter()
            .try_fold(0usize, |bytes, (batch_idx, _)| {
                bytes.checked_add(self.batches[*batch_idx].2)
            })
            .ok_or_else(|| internal_datafusion_err!("merge output size overflow"))?;
        Ok(self.held_bytes().saturating_add(estimate))
    }

    /// Drains the in_progress row indexes, and builds a new RecordBatch from them
    ///
    /// Will then drop any batches for which all rows have been yielded to the output.
    /// If an offset overflow occurs (e.g. string/list offsets exceed i32::MAX),
    /// retries with progressively fewer rows until it succeeds.
    ///
    /// Returns `None` if no pending rows
    pub fn build_record_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.is_empty() {
            return Ok(None);
        }

        // Interleaving allocates the output before any consumer can charge
        // it. Admit it first, from the per-row bytes of each source batch,
        // so a pool that cannot hold it refuses instead of being exceeded.
        // An output the pool's room cannot hold is halved before it is
        // asked for, down to the floor; a refused admission retries with
        // half the rows, down to one.
        let mut rows = self.indices.len();
        if let Some(room) = self.reservation.room() {
            while rows > self.floor
                && self.needed(rows)?.saturating_sub(self.reservation.size()) > room
            {
                rows = (rows / 2).max(self.floor);
            }
        }
        loop {
            let needed = self.needed(rows)?;
            match try_grow_reservation_to_at_least(&mut self.reservation, needed) {
                Ok(()) => break,
                Err(e @ DataFusionError::ResourcesExhausted(_)) if rows > 1 => {
                    rows /= 2;
                    warn!(
                        "Merge output of {} rows refused memory, retrying with {rows}: {e}",
                        self.indices.len()
                    );
                }
                Err(e) => return Err(e),
            }
        }

        let (rows_to_emit, columns) =
            retry_interleave(rows, self.indices.len(), |rows_to_emit| {
                self.try_interleave_columns(&self.indices[..rows_to_emit])
            })?;

        Ok(Some(self.finish_record_batch(rows_to_emit, columns)?))
    }
}

/// Bytes a batch's rows occupy in the buffers interleaving them allocates.
///
/// Interleaving copies offsets, values and child arrays, but a view column
/// shares its data buffers with its inputs (`interleave_views` keeps each
/// referenced buffer by pointer) and allocates only its 16-byte views and
/// nulls. A struct interleaves each child the same way. Every other type
/// counts its slice size, which covers what interleaving copies.
///
/// The shared data buffers are not lost to the count: the builder charges
/// them as held input while any live batch points at them, and the output's
/// consumer charges them on receipt.
fn fresh(batch: &RecordBatch) -> Result<usize> {
    batch.columns().iter().try_fold(0usize, |bytes, array| {
        Ok(bytes + fresh_data(&array.to_data())?)
    })
}

fn fresh_data(data: &ArrayData) -> Result<usize> {
    let nulls = if data.nulls().is_some() {
        bit_util::ceil(data.len(), 8)
    } else {
        0
    };
    match data.data_type() {
        DataType::Utf8View | DataType::BinaryView => {
            Ok(data.len() * size_of::<u128>() + nulls)
        }
        DataType::Struct(_) => data
            .child_data()
            .iter()
            .try_fold(nulls, |bytes, child| Ok(bytes + fresh_data(child)?)),
        _ => Ok(data.get_slice_memory_size()?),
    }
}

/// Try to grow `reservation` so it covers at least `needed` bytes.
///
/// When a reservation has been pre-loaded with bytes (e.g. via
/// [`MemoryReservation::take()`]), this avoids redundant pool
/// allocations: if the reservation already covers `needed`, this is
/// a no-op; otherwise only the deficit is requested from the pool.
pub(crate) fn try_grow_reservation_to_at_least(
    reservation: &mut MemoryReservation,
    needed: usize,
) -> Result<()> {
    if needed > reservation.size() {
        reservation.try_grow(needed - reservation.size())?;
    }
    Ok(())
}

/// Returns true if the error is an Arrow offset overflow.
fn is_offset_overflow(e: &DataFusionError) -> bool {
    matches!(
        e,
        DataFusionError::ArrowError(boxed, _)
            if matches!(boxed.as_ref(), ArrowError::OffsetOverflowError(_))
    )
}

#[cfg(test)]
fn offset_overflow_error() -> DataFusionError {
    DataFusionError::ArrowError(Box::new(ArrowError::OffsetOverflowError(0)), None)
}

fn retry_interleave<T, F>(
    mut rows_to_emit: usize,
    total_rows: usize,
    mut interleave: F,
) -> Result<(usize, T)>
where
    F: FnMut(usize) -> Result<T>,
{
    loop {
        match interleave(rows_to_emit) {
            Ok(value) => return Ok((rows_to_emit, value)),
            // Only offset overflow is recoverable by emitting fewer rows.
            Err(e) if is_offset_overflow(&e) => {
                rows_to_emit /= 2;
                if rows_to_emit == 0 {
                    return Err(e);
                }
                warn!(
                    "Interleave offset overflow with {total_rows} rows, retrying with {rows_to_emit}"
                );
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, ArrayDataBuilder, AsArray, Int32Array, ListArray, StringViewArray,
        UInt32Array,
    };
    use arrow::buffer::Buffer;
    use arrow::datatypes::Int32Type;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool, MergeMemoryPool,
        UnboundedMemoryPool,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A greedy pool that counts the asks it refuses, as a pool that records
    /// a refusal would see them.
    #[derive(Debug)]
    struct Refusals {
        pool: GreedyMemoryPool,
        refused: AtomicUsize,
    }

    impl std::fmt::Display for Refusals {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "refusals")
        }
    }

    impl MemoryPool for Refusals {
        fn name(&self) -> &str {
            "refusals"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            self.pool.grow(reservation, additional)
        }

        fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
            self.pool.shrink(reservation, shrink)
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.pool
                .try_grow(reservation, additional)
                .inspect_err(|_| {
                    self.refused.fetch_add(1, Ordering::Relaxed);
                })
        }

        fn reserved(&self) -> usize {
            self.pool.reserved()
        }

        fn memory_limit(&self) -> MemoryLimit {
            self.pool.memory_limit()
        }

        fn room(&self, reservation: &MemoryReservation) -> Option<usize> {
            self.pool.room(reservation)
        }
    }

    fn refusals(size: usize) -> Arc<Refusals> {
        Arc::new(Refusals {
            pool: GreedyMemoryPool::new(size),
            refused: AtomicUsize::new(0),
        })
    }

    fn overflow_list_batch() -> RecordBatch {
        let values_field = Arc::new(Field::new_list_field(DataType::Int32, true));
        // SAFETY: This intentionally constructs an invalid child length so
        // Arrow's interleave hits offset overflow before touching child data.
        let list = ListArray::from(unsafe {
            ArrayDataBuilder::new(DataType::List(Arc::clone(&values_field)))
                .len(1)
                .add_buffer(Buffer::from_slice_ref([0_i32, i32::MAX]))
                .add_child_data(Int32Array::from(Vec::<i32>::new()).to_data())
                .build_unchecked()
        });
        let schema = Arc::new(Schema::new(vec![Field::new(
            "list_col",
            DataType::List(values_field),
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(list)]).unwrap()
    }

    fn reservation() -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        MemoryConsumer::new("test").register(&pool)
    }

    fn int_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    fn push_n_rows(builder: &mut BatchBuilder, stream_idx: usize, n: usize) {
        for _ in 0..n {
            builder.push_row(stream_idx);
        }
    }

    fn emit_n_rows(builder: &mut BatchBuilder, n: usize) -> RecordBatch {
        let columns = builder
            .try_interleave_columns(&builder.indices[..n])
            .unwrap();
        builder.finish_record_batch(n, columns).unwrap()
    }

    fn assert_int_output(batch: &RecordBatch, expected: &[i32]) {
        let actual = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_partial_emit_releases_unreferenced_and_retains_live_batches() {
        let batch0 = int_batch(vec![10, 11]);
        let batch1 = int_batch(vec![20, 21]);
        let batch2 = int_batch(vec![30, 31]);
        let batch1_size = get_record_batch_memory_size(&batch1);
        let batch2_size = get_record_batch_memory_size(&batch2);
        let schema = batch0.schema();
        let mut builder = BatchBuilder::new(Arc::clone(&schema), 3, 6, reservation());

        builder.push_batch(0, batch0).unwrap();
        push_n_rows(&mut builder, 0, 2);
        builder.push_batch(1, batch1).unwrap();
        push_n_rows(&mut builder, 1, 2);
        // Keep one stream empty so an unloaded cursor cannot retain consumed batches.
        builder.push_batch(0, batch2).unwrap();

        let output = emit_n_rows(&mut builder, 2);
        assert_int_output(&output, &[10, 11]);

        assert_eq!(builder.len(), 2);
        assert_eq!(builder.batches.len(), 2);
        assert_eq!(builder.held_bytes(), batch1_size + batch2_size);
        assert_eq!(builder.reservation.size(), batch1_size + batch2_size);

        push_n_rows(&mut builder, 0, 2);
        let output = emit_n_rows(&mut builder, 4);
        assert_int_output(&output, &[20, 21, 30, 31]);

        assert!(builder.is_empty());
        assert!(builder.batches.is_empty());
        assert_eq!(builder.held_bytes(), 0);
        assert_eq!(builder.reservation.size(), 0);
    }

    #[test]
    fn test_released_cursor_accepts_new_batch_for_stream() {
        let batch0 = int_batch(vec![10]);
        let batch1 = int_batch(vec![20]);
        let schema = batch0.schema();
        let mut builder = BatchBuilder::new(Arc::clone(&schema), 1, 1, reservation());

        builder.push_batch(0, batch0).unwrap();
        builder.push_row(0);
        let output = emit_n_rows(&mut builder, 1);
        assert_int_output(&output, &[10]);
        assert!(builder.batches.is_empty());

        builder.push_batch(0, batch1).unwrap();
        builder.push_row(0);
        let output = emit_n_rows(&mut builder, 1);
        assert_int_output(&output, &[20]);
        assert!(builder.batches.is_empty());
        assert_eq!(builder.held_bytes(), 0);
        assert_eq!(builder.reservation.size(), 0);
    }

    #[test]
    fn test_retry_interleave_halves_rows_until_success() {
        let mut attempts = Vec::new();

        let (rows_to_emit, result) = retry_interleave(4, 4, |rows_to_emit| {
            attempts.push(rows_to_emit);
            if rows_to_emit > 1 {
                Err(offset_overflow_error())
            } else {
                Ok("ok")
            }
        })
        .unwrap();

        assert_eq!(rows_to_emit, 1);
        assert_eq!(result, "ok");
        assert_eq!(attempts, vec![4, 2, 1]);
    }

    #[test]
    fn test_is_offset_overflow_matches_arrow_error() {
        assert!(is_offset_overflow(&offset_overflow_error()));
    }

    #[test]
    fn test_retry_interleave_does_not_retry_non_offset_errors() {
        let mut attempts = Vec::new();

        let error = retry_interleave(4, 4, |rows_to_emit| {
            attempts.push(rows_to_emit);
            Err::<(), _>(DataFusionError::Execution("boom".into()))
        })
        .unwrap_err();

        assert_eq!(attempts, vec![4]);
        assert!(matches!(error, DataFusionError::Execution(msg) if msg == "boom"));
    }

    #[test]
    fn test_try_interleave_columns_surfaces_arrow_offset_overflow() {
        let batch = overflow_list_batch();
        let schema = batch.schema();
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(schema, 1, 2, reservation);
        builder.push_batch(0, batch).unwrap();

        let error = builder
            .try_interleave_columns(&[(0, 0), (0, 0)])
            .unwrap_err();

        assert!(is_offset_overflow(&error));
    }

    /// Interleaving allocates the output before its consumer can charge it,
    /// so a pool without room for it admits fewer rows instead.
    #[test]
    fn test_build_record_batch_admits_its_output_first() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = |values: Vec<i32>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(values))],
            )
            .unwrap()
        };
        let (left, right) = (batch((0..1024).collect()), batch((0..1024).collect()));
        let held =
            get_record_batch_memory_size(&left) + get_record_batch_memory_size(&right);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(held + 1024));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(Arc::clone(&schema), 2, 1024, reservation);
        builder.push_batch(0, left).unwrap();
        builder.push_batch(1, right).unwrap();
        for _ in 0..512 {
            builder.push_row(0);
            builder.push_row(1);
        }

        let output = builder.build_record_batch().unwrap().unwrap();

        // 1024 bytes of room hold 256 of the 1024 pending Int32 rows.
        assert_eq!(output.num_rows(), 256);
        assert_eq!(builder.len(), 768);
    }

    /// A string view column of `rows` 64-byte values in one data buffer, cut
    /// into `parts` slices that each keep that whole buffer.
    fn shared_view_slices(rows: usize, parts: usize) -> (Vec<RecordBatch>, usize) {
        let values =
            StringViewArray::from_iter_values((0..rows).map(|row| format!("{row:064}")));
        let data = values.data_buffers().iter().map(|b| b.capacity()).sum();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8View,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let part = rows / parts;
        let slices = (0..parts)
            .map(|i| batch.slice(i * part, part))
            .collect::<Vec<_>>();
        (slices, data)
    }

    fn view_values(batch: &RecordBatch) -> Vec<String> {
        batch
            .column(0)
            .as_string_view()
            .iter()
            .map(|value| value.unwrap().to_string())
            .collect()
    }

    /// Slices that share one large view data buffer are held at that buffer
    /// once, and their output is admitted at its views, not at the buffer:
    /// a merge of eight slices fits a pool of about twice the buffer.
    #[test]
    fn test_shared_view_buffers_are_charged_once() {
        let (slices, data) = shared_view_slices(8192, 8);
        let views = 8192 * size_of::<u128>();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(data + 2 * views));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let schema = slices[0].schema();
        let mut builder = BatchBuilder::new(schema, slices.len(), 8192, reservation);
        // Charging each slice its buffers would ask for eight times `data`.
        for (stream, slice) in slices.iter().enumerate() {
            builder.push_batch(stream, slice.clone()).unwrap();
        }
        // The slices share the views buffer as well as the data buffers.
        assert_eq!(builder.held_bytes(), data + views);
        // Interleave the streams so the output draws from every slice.
        for _ in 0..1024 {
            for stream in (0..slices.len()).rev() {
                builder.push_row(stream);
            }
        }

        let output = builder.build_record_batch().unwrap().unwrap();

        let part = |stream: usize, row: usize| format!("{:064}", stream * 1024 + row);
        let expected = (0..1024)
            .flat_map(|row| (0..8).rev().map(move |stream| part(stream, row)))
            .collect::<Vec<_>>();
        assert_eq!(view_values(&output), expected);
        assert!(builder.batches.is_empty());
        assert_eq!(builder.held_bytes(), 0);
        assert_eq!(builder.reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    /// A buffer stays charged while any live batch points at it, even after
    /// the batch that first brought it is released.
    #[test]
    fn test_shared_buffer_stays_charged_until_its_last_batch() {
        let (slices, data) = shared_view_slices(1024, 2);
        let schema = slices[0].schema();
        let mut builder = BatchBuilder::new(schema, 2, 512, reservation());
        builder.push_batch(0, slices[0].clone()).unwrap();
        let first = builder.held_bytes();
        assert_eq!(first, data + 1024 * size_of::<u128>());
        // The second slice shares the first's views and data buffers.
        builder.push_batch(1, slices[1].clone()).unwrap();
        assert_eq!(builder.held_bytes(), first);

        // Emitting all of stream 0 releases the first slice; stream 1 still
        // points at the data buffer, so it stays charged.
        push_n_rows(&mut builder, 0, 512);
        let output = emit_n_rows(&mut builder, 512);
        assert_eq!(output.num_rows(), 512);
        assert_eq!(builder.batches.len(), 1);
        assert_eq!(builder.held_bytes(), first);
        assert_eq!(builder.reservation.size(), first);

        push_n_rows(&mut builder, 1, 512);
        let output = emit_n_rows(&mut builder, 512);
        assert_eq!(view_values(&output)[0], format!("{:064}", 512));
        assert_eq!(builder.held_bytes(), 0);
        assert_eq!(builder.reservation.size(), 0);
    }

    /// An output the pool cannot admit whole is emitted in halves, and the
    /// rows it keeps are emitted next, in order.
    #[test]
    fn test_refused_output_emits_fewer_rows() {
        let batch = int_batch((0..1024).collect());
        let held = get_record_batch_memory_size(&batch);
        // Room for the held batch and a quarter of its rows as output.
        let refusals = Arc::new(Refusals {
            pool: GreedyMemoryPool::new(held + 256 * size_of::<i32>()),
            refused: AtomicUsize::new(0),
        });
        let pool: Arc<dyn MemoryPool> = Arc::clone(&refusals) as _;
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(batch.schema(), 1, 1024, reservation);
        builder.push_batch(0, batch).unwrap();
        push_n_rows(&mut builder, 0, 1024);

        let mut emitted = vec![];
        while let Some(output) = builder.build_record_batch().unwrap() {
            assert!(output.num_rows() <= 256, "{}", output.num_rows());
            emitted
                .extend_from_slice(output.column(0).as_primitive::<Int32Type>().values());
        }

        assert_eq!(emitted, (0..1024).collect::<Vec<_>>());
        assert_eq!(builder.held_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
        // The halving sized itself to the pool's room, so the pool never
        // saw an ask it refused.
        assert_eq!(refusals.refused.load(Ordering::Relaxed), 0);
    }

    /// When not even one row fits, the merge still fails.
    #[test]
    fn test_refused_single_row_fails() {
        let batch = int_batch((0..1024).collect());
        let held = get_record_batch_memory_size(&batch);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(held));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(batch.schema(), 1, 1024, reservation);
        builder.push_batch(0, batch).unwrap();
        push_n_rows(&mut builder, 0, 1024);

        let error = builder.build_record_batch().unwrap_err();

        assert!(
            matches!(error, DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        assert_eq!(builder.len(), 1024);
    }

    /// A merge registered on a sorter's [`MergeMemoryPool`] sizes its output
    /// to what the parent pool would grant, not to the parent's whole budget:
    /// with the budget nearly full of another consumer, the parent never sees
    /// an ask it refuses.
    #[test]
    fn test_merge_pool_room_counts_other_consumers() {
        let batch = int_batch((0..1024).collect());
        let held = get_record_batch_memory_size(&batch);
        let other = 1 << 20;
        let parent = refusals(other + held + 256 * size_of::<i32>());
        let parent_pool: Arc<dyn MemoryPool> = Arc::clone(&parent) as _;
        let contender = MemoryConsumer::new("other sorter").register(&parent_pool);
        contender.try_grow(other).unwrap();
        let merge = Arc::new(MergeMemoryPool::new(
            Arc::clone(&parent_pool),
            MemoryConsumer::new("ExternalSorterMerge"),
        ));
        let merge_pool: Arc<dyn MemoryPool> = Arc::clone(&merge) as _;
        let reservation = MemoryConsumer::new("merge").register(&merge_pool);
        let mut builder = BatchBuilder::new(batch.schema(), 1, 1024, reservation);
        builder.push_batch(0, batch).unwrap();
        push_n_rows(&mut builder, 0, 1024);
        assert_eq!(
            builder.reservation.room(),
            Some(256 * size_of::<i32>()),
            "the parent's room, not its budget less this merge"
        );

        let mut emitted = 0;
        while let Some(output) = builder.build_record_batch().unwrap() {
            assert!(output.num_rows() <= 256, "{}", output.num_rows());
            emitted += output.num_rows();
        }

        assert_eq!(emitted, 1024);
        assert_eq!(parent.refused.load(Ordering::Relaxed), 0);
        assert_eq!(parent_pool.reserved(), other);
    }

    /// Halving against the room stops at an eighth of the batch size, which
    /// is asked for whatever the room, so a pool that funds asks past its
    /// room can grant it; only its refusal halves further.
    #[test]
    fn test_halving_asks_at_its_floor() {
        let batch = int_batch((0..1024).collect());
        let held = get_record_batch_memory_size(&batch);
        // Room for 64 rows of output, below the floor of 1024 / 8 = 128.
        let pool = refusals(held + 64 * size_of::<i32>());
        let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
        let reservation = MemoryConsumer::new("test").register(&dyn_pool);
        let mut builder = BatchBuilder::new(batch.schema(), 1, 1024, reservation);
        builder.push_batch(0, batch).unwrap();
        push_n_rows(&mut builder, 0, 1024);

        let output = builder.build_record_batch().unwrap().unwrap();

        assert_eq!(output.num_rows(), 64);
        // One ask, at the floor; nothing larger.
        assert_eq!(pool.refused.load(Ordering::Relaxed), 1);
    }

    /// A batch whose admission is refused is rolled back without uncharging
    /// the buffers it shares with a live batch.
    #[test]
    fn test_refused_push_keeps_shared_buffers_charged() {
        let (slices, data) = shared_view_slices(1024, 2);
        // A batch with its own views over the first slice's data buffers.
        let rows = UInt32Array::from_iter_values(0..512);
        let taken = arrow::compute::take_record_batch(&slices[0], &rows).unwrap();
        let first = data + 1024 * size_of::<u128>();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(first + 1024));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(slices[0].schema(), 2, 512, reservation);
        builder.push_batch(0, slices[0].clone()).unwrap();
        assert_eq!(builder.held_bytes(), first);

        assert!(builder.push_batch(1, taken).is_err());

        assert_eq!(builder.held_bytes(), first);
        assert_eq!(builder.reservation.size(), first);
        assert_eq!(pool.reserved(), first);
    }

    /// A buffer released with its last batch is charged again when a later
    /// batch brings it back.
    #[test]
    fn test_released_buffer_is_charged_again() {
        let (slices, data) = shared_view_slices(1024, 2);
        let mut builder = BatchBuilder::new(slices[0].schema(), 1, 512, reservation());
        builder.push_batch(0, slices[0].clone()).unwrap();
        let first = builder.held_bytes();
        push_n_rows(&mut builder, 0, 512);
        emit_n_rows(&mut builder, 512);
        assert_eq!(builder.held_bytes(), 0);

        builder.push_batch(0, slices[1].clone()).unwrap();

        assert_eq!(builder.held_bytes(), first);
        assert_eq!(first, data + 1024 * size_of::<u128>());
        assert_eq!(builder.reservation.size(), first);
    }

    /// What interleaving a view column with nulls allocates stays within
    /// what the builder admitted for it.
    #[test]
    fn test_interleaved_views_with_nulls_fit_their_admission() {
        let values = StringViewArray::from_iter(
            (0..1024).map(|row| (row % 3 != 0).then(|| format!("{row:064}"))),
        );
        let schema =
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();
        let (first, second) = (batch.slice(0, 512), batch.slice(512, 512));
        let mut builder = BatchBuilder::new(batch.schema(), 2, 1024, reservation());
        builder.push_batch(0, first).unwrap();
        builder.push_batch(1, second).unwrap();
        for _ in 0..512 {
            builder.push_row(1);
            builder.push_row(0);
        }
        let admitted = builder.needed(1024).unwrap() - builder.held_bytes();

        let output = builder.build_record_batch().unwrap().unwrap();

        let column = output.column(0).as_string_view();
        let allocated = column.views().inner().capacity()
            + column.nulls().map_or(0, |nulls| nulls.buffer().capacity());
        assert!(column.null_count() > 0);
        assert!(
            allocated <= admitted,
            "allocated {allocated} above the admitted {admitted}"
        );
    }
}
