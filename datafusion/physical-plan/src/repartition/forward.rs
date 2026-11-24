// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Forwarded string views: output batches of a [`RepartitionExec`] whose view
//! columns reference the buffers of the batches its input produced, instead
//! of copies of the bytes they reference.
//!
//! A hash repartition splits each input batch into one slice per output, so
//! each slice of a view column references the whole of the input's buffers
//! while it uses a fraction of their bytes. Arrow's coalescer copies such a
//! sparse column's bytes into new buffers. A consumer in the same process can
//! read the input's buffers in place, so with
//! [`RepartitionExec::with_forward_views`] the output coalescer here
//! ([`ViewForwarder`]) concatenates the slices without copying a byte, and
//! the bytes are charged once ([`BufferLedger`]) however many outputs share
//! them.
//!
//! [`RepartitionExec`]: super::RepartitionExec
//! [`RepartitionExec::with_forward_views`]: super::RepartitionExec::with_forward_views

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryViewArray, GenericByteViewArray, RecordBatch,
    RecordBatchOptions, StringViewArray,
};
use arrow::buffer::{Buffer, NullBuffer, ScalarBuffer};
use arrow::compute::concat;
use arrow::datatypes::{
    BinaryViewType, ByteViewType, DataType, SchemaRef, StringViewType,
};
use datafusion_common::Result;
#[cfg(test)]
use datafusion_common::internal_err;
use datafusion_execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;

/// The output coalescer of a repartition that forwards string views.
///
/// It gathers the slices routed to one output until they hold
/// `target_batch_size` rows and then concatenates them into one batch whose
/// view columns reference the slices' buffers, each buffer once
/// ([`concat_forwarded`]). Unlike arrow's coalescer it never copies the bytes
/// of a view column, and its batches may hold more than `target_batch_size`
/// rows: up to `target_batch_size` plus one slice less one row.
#[derive(Debug)]
pub(super) struct ViewForwarder {
    schema: SchemaRef,
    target_batch_size: usize,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    completed: Vec<RecordBatch>,
}

impl ViewForwarder {
    pub(super) fn new(schema: SchemaRef, target_batch_size: usize) -> Self {
        Self {
            schema,
            target_batch_size,
            pending: vec![],
            pending_rows: 0,
            completed: vec![],
        }
    }

    pub(super) fn push_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.pending_rows += batch.num_rows();
        self.pending.push(batch);
        if self.pending_rows >= self.target_batch_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Completes the rows gathered so far as one batch.
    pub(super) fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let batch = concat_forwarded(&self.schema, &self.pending)?;
        self.pending.clear();
        self.pending_rows = 0;
        self.completed.push(batch);
        Ok(())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.completed.is_empty()
    }

    pub(super) fn next_completed_batch(&mut self) -> Option<RecordBatch> {
        (!self.completed.is_empty()).then(|| self.completed.remove(0))
    }
}

/// `batches` concatenated. A view column of the result references the data
/// buffers of the input columns, each distinct buffer once, without copying
/// their bytes; the other columns are concatenated by arrow.
pub(super) fn concat_forwarded(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch> {
    if let [batch] = batches {
        return Ok(batch.clone());
    }
    let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
    let columns = (0..schema.fields().len())
        .map(|index| {
            let arrays: Vec<&ArrayRef> =
                batches.iter().map(|batch| batch.column(index)).collect();
            match schema.field(index).data_type() {
                DataType::Utf8View => {
                    Ok(Arc::new(concat_views::<StringViewType>(&arrays)) as ArrayRef)
                }
                DataType::BinaryView => {
                    Ok(Arc::new(concat_views::<BinaryViewType>(&arrays)) as ArrayRef)
                }
                _ => {
                    let arrays: Vec<&dyn Array> =
                        arrays.iter().map(|array| array.as_ref()).collect();
                    Ok(concat(&arrays)?)
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &options,
    )?)
}

/// View arrays concatenated over the distinct data buffers they reference.
fn concat_views<T: ByteViewType>(arrays: &[&ArrayRef]) -> GenericByteViewArray<T> {
    let arrays: Vec<&GenericByteViewArray<T>> = arrays
        .iter()
        .map(|array| array.as_byte_view::<T>())
        .collect();
    let len = arrays.iter().map(|array| array.len()).sum();
    let mut buffers: Vec<Buffer> = vec![];
    let mut index_of: HashMap<(usize, usize), u32> = HashMap::new();
    let mut views = Vec::with_capacity(len);
    for array in &arrays {
        // Where each of this array's buffers lies among the result's
        let indices: Vec<u32> = array
            .data_buffers()
            .iter()
            .map(|buffer| {
                *index_of
                    .entry((buffer.as_ptr() as usize, buffer.len()))
                    .or_insert_with(|| {
                        buffers.push(buffer.clone());
                        (buffers.len() - 1) as u32
                    })
            })
            .collect();
        views.extend(array.views().iter().map(|&view| {
            let len = view as u32;
            if len <= 12 {
                return view;
            }
            // The buffer index is bits 64..96 of a view longer than 12 bytes
            let buffer = (view >> 64) as u32 as usize;
            let remapped = u128::from(indices[buffer]);
            (view & !(u128::from(u32::MAX) << 64)) | (remapped << 64)
        }));
    }
    let nulls = arrays.iter().any(|array| array.null_count() > 0).then(|| {
        NullBuffer::from_iter(
            arrays
                .iter()
                .flat_map(|array| (0..array.len()).map(|row| array.is_valid(row))),
        )
    });
    // SAFETY: every view is a view of one of `arrays`, valid for its array's
    // buffers. An inline view is unchanged; a view longer than 12 bytes gets
    // the index its buffer has in `buffers`, which holds that very buffer, so
    // its offset and length still lie within it and its bytes are the same
    // valid UTF-8 (for strings) they were. The nulls have one bit per view.
    unsafe {
        GenericByteViewArray::new_unchecked(
            ScalarBuffer::from(views),
            buffers.into(),
            nulls,
        )
    }
}

/// The distinct data buffers of the view columns of `batch`, by address.
fn view_buffers(batch: &RecordBatch) -> Vec<Buffer> {
    let mut seen = HashMap::new();
    for column in batch.columns() {
        let buffers = match column.data_type() {
            DataType::Utf8View => column
                .as_any()
                .downcast_ref::<StringViewArray>()
                .map(|array| array.data_buffers()),
            DataType::BinaryView => column
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .map(|array| array.data_buffers()),
            _ => None,
        };
        for buffer in buffers.iter().flat_map(|buffers| buffers.iter()) {
            seen.entry(buffer.as_ptr() as usize)
                .or_insert_with(|| buffer.clone());
        }
    }
    seen.into_values().collect()
}

/// The bytes of the view data buffers that several queued batches of a
/// repartition share, charged to one reservation once each.
///
/// A forwarded batch is charged in two parts: what it owns (views, nulls and
/// its other columns) to its output's reservation, and the data buffers of
/// its view columns here. A buffer is charged when the first queued batch
/// that references it is charged and released when the last one is taken,
/// so a buffer the slices of one input batch share across every output
/// counts once, not once per output.
///
/// The ledger counts a buffer only while a batch that references it is
/// queued. A buffer stays pinned longer than that, while:
/// - an output's coalescer holds a slice that references it, until the
///   coalescer has gathered `target_batch_size` rows for that output, its
///   consumer asks for a flush, or the input ends. That is at most the input
///   batches whose rows make up one output batch per output, as when arrow's
///   coalescer holds its in-progress batch, which is not charged either;
/// - a consumer holds a batch it has taken: a final aggregation copies the
///   keys it keeps and drops the batch once it has aggregated it.
#[derive(Debug)]
pub(super) struct BufferLedger {
    reservation: MemoryReservation,
    /// For each charged buffer, by address: its bytes and the queued
    /// batches that reference it
    held: Mutex<HashMap<usize, (usize, usize)>>,
}

impl BufferLedger {
    pub(super) fn new(reservation: MemoryReservation) -> Self {
        Self {
            reservation,
            held: Mutex::new(HashMap::new()),
        }
    }

    /// Bytes charged
    #[cfg(test)]
    pub(super) fn size(&self) -> usize {
        self.reservation.size()
    }

    /// Charges the buffers of `buffers` that no queued batch references yet.
    /// Returns `false`, charging nothing, if the pool refuses them.
    fn acquire(&self, buffers: &[Buffer]) -> bool {
        let mut held = self.held.lock();
        let new: usize = buffers
            .iter()
            .filter(|buffer| !held.contains_key(&(buffer.as_ptr() as usize)))
            .map(|buffer| buffer.capacity())
            .sum();
        if self.reservation.try_grow(new).is_err() {
            return false;
        }
        for buffer in buffers {
            held.entry(buffer.as_ptr() as usize)
                .or_insert_with(|| (buffer.capacity(), 0))
                .1 += 1;
        }
        true
    }

    /// Releases one reference to each of `addresses`, freeing the bytes of a
    /// buffer no queued batch references any more.
    fn release(&self, addresses: &[usize]) {
        let mut held = self.held.lock();
        let mut freed = 0;
        for address in addresses {
            if let Some((bytes, refs)) = held.get_mut(address) {
                *refs -= 1;
                if *refs == 0 {
                    freed += *bytes;
                    held.remove(address);
                }
            }
        }
        self.reservation.shrink(freed);
    }
}

/// The memory a queued forwarded batch holds: its own bytes in its output's
/// reservation and a reference to each of its view buffers in the
/// [`BufferLedger`]. Dropping it releases both, wherever the batch leaves the
/// queue.
#[derive(Debug)]
pub(super) struct ForwardedBytes {
    own: usize,
    reservation: Arc<MemoryReservation>,
    addresses: Vec<usize>,
    ledger: Arc<BufferLedger>,
}

impl ForwardedBytes {
    /// Charges `batch`, or returns `None`, charging nothing, if the pool
    /// refuses it.
    pub(super) fn charge(
        batch: &RecordBatch,
        reservation: &Arc<MemoryReservation>,
        ledger: &Arc<BufferLedger>,
    ) -> Option<Self> {
        let buffers = view_buffers(batch);
        let shared: usize = buffers.iter().map(Buffer::capacity).sum();
        let own = batch.get_array_memory_size().saturating_sub(shared);
        reservation.try_grow(own).ok()?;
        if !ledger.acquire(&buffers) {
            reservation.shrink(own);
            return None;
        }
        Some(Self {
            own,
            reservation: Arc::clone(reservation),
            addresses: buffers.iter().map(|b| b.as_ptr() as usize).collect(),
            ledger: Arc::clone(ledger),
        })
    }
}

impl Drop for ForwardedBytes {
    fn drop(&mut self) {
        self.reservation.shrink(self.own);
        self.ledger.release(&self.addresses);
    }
}

/// Fails unless every view of `batch` that is longer than 12 bytes names one
/// of its column's buffers: the check a consumer could make that a batch was
/// forwarded intact.
#[cfg(test)]
pub(super) fn check_views(batch: &RecordBatch) -> Result<()> {
    for column in batch.columns() {
        if let Some(array) = column.as_any().downcast_ref::<StringViewArray>() {
            let buffers = array.data_buffers().len();
            for &view in array.views().iter() {
                if view as u32 > 12 && (view >> 64) as u32 as usize >= buffers {
                    return internal_err!("A view names a buffer the array lacks");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool,
    };

    fn strings(rows: std::ops::Range<usize>) -> StringViewArray {
        rows.map(|row| Some(format!("a string longer than twelve bytes {row}")))
            .collect()
    }

    #[test]
    fn concatenates_without_copying_bytes() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, true),
            Field::new("n", DataType::Int64, false),
        ]));
        let column: ArrayRef = Arc::new(strings(0..100));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::clone(&column),
                Arc::new(arrow::array::Int64Array::from_iter_values(0..100)),
            ],
        )?;
        let other = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(strings(100..150)),
                Arc::new(arrow::array::Int64Array::from_iter_values(100..150)),
            ],
        )?;
        let slices = [batch.slice(90, 10), other, batch.slice(0, 10)];
        let joined = concat_forwarded(&schema, &slices)?;
        check_views(&joined)?;
        let views = joined.column(0).as_string_view();
        // The buffer the two slices of `batch` share appears once
        assert_eq!(views.data_buffers().len(), 2);
        let source = column.as_string_view().data_buffers()[0].as_ptr();
        assert!(views.data_buffers().iter().any(|b| b.as_ptr() == source));
        let expected: Vec<String> = (90..150)
            .chain(0..10)
            .map(|row| format!("a string longer than twelve bytes {row}"))
            .collect();
        let actual: Vec<&str> = views.iter().map(|v| v.unwrap_or_default()).collect();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn a_shared_buffer_is_charged_once() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let ledger = Arc::new(BufferLedger::new(
            MemoryConsumer::new("ledger").register(&pool),
        ));
        let output = |name: &str| Arc::new(MemoryConsumer::new(name).register(&pool));
        let (first, second) = (output("first"), output("second"));
        let schema =
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(strings(0..1000))])?;
        let shared: usize = batch
            .column(0)
            .as_string_view()
            .data_buffers()
            .iter()
            .map(Buffer::capacity)
            .sum();

        let a = ForwardedBytes::charge(&batch.slice(0, 500), &first, &ledger);
        let b = ForwardedBytes::charge(&batch.slice(500, 500), &second, &ledger);
        assert!(a.is_some() && b.is_some());
        assert_eq!(ledger.size(), shared);
        assert!(first.size() < shared && second.size() < shared);
        drop(a);
        assert_eq!(ledger.size(), shared, "the second slice still holds it");
        drop(b);
        assert_eq!(ledger.size(), 0);
        assert_eq!(first.size() + second.size(), 0);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn a_refused_charge_charges_nothing() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
        let ledger = Arc::new(BufferLedger::new(
            MemoryConsumer::new("ledger").register(&pool),
        ));
        let output = Arc::new(MemoryConsumer::new("output").register(&pool));
        let schema =
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(strings(0..10_000))])?;
        assert!(ForwardedBytes::charge(&batch, &output, &ledger).is_none());
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }
}
