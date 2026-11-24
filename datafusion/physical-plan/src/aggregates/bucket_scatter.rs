// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Writing the rows routed into [`FinalBuckets`] straight into each bucket's
//! rows in progress.
//!
//! [`FinalBuckets`]: super::final_buckets::FinalBuckets

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, BooleanBufferBuilder,
    GenericByteViewArray, PrimitiveArray, downcast_primitive,
};
use arrow::buffer::{Buffer, NullBuffer, NullBufferBuilder, ScalarBuffer};
use arrow::datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType};

use super::final_buckets::NUM_BUCKETS;

/// Bytes a view holds inline. A longer value's bytes sit in a data buffer.
const INLINE: u32 = 12;

/// A string block starts at 8 KiB and doubles up to 1 MiB, as in arrow's
/// `BatchCoalescer`, so a bucket holds as much as the coalescer it replaces.
const FIRST_BLOCK: usize = 8 * 1024;
const MAX_BLOCK: usize = 1024 * 1024;

/// The rows in progress of one column, for every bucket.
///
/// A routed row is written once: read from the routed batch in row order and
/// appended to its bucket's buffers, which become the bucket's next batch as
/// they are. It replaces a `take` of the batch followed by a
/// `BatchCoalescer` per bucket, which wrote every row twice.
pub(super) trait ScatterColumn: Send {
    /// Makes room for `rows` more rows of `bucket`. A bucket with no
    /// allocation takes at least `at_least` rows, which is what the
    /// coalescer allocates on its first write after a hand-off.
    fn reserve(&mut self, bucket: usize, rows: usize, at_least: usize);

    /// Appends row `offset + i` of `source` to bucket `ids[i]`, in row order.
    /// `counts[b]` is the number of `ids` equal to `b`, and every bucket in
    /// `ids` has room for them (see [`Self::reserve`]). Returns the bytes
    /// written.
    fn scatter(
        &mut self,
        source: &ArrayRef,
        offset: usize,
        ids: &[u8],
        counts: &[usize; NUM_BUCKETS],
    ) -> usize;

    /// Hands over the rows of `bucket` as an array, leaving it with no rows
    /// and no allocation.
    fn finish(&mut self, bucket: usize) -> ArrayRef;

    /// Memory held by the rows in progress of `bucket`
    fn size(&self, bucket: usize) -> usize;
}

/// A [`ScatterColumn`] for `data_type`, or `None` if the rows of such a
/// column are routed the general way (`take` and a coalescer).
///
/// Fixed-width primitives, booleans and views are written directly. The
/// partial state the buckets hold today is only of the first two kinds (see
/// `BucketedAggregation::supports_state`).
pub(super) fn scatter_column(data_type: &DataType) -> Option<Box<dyn ScatterColumn>> {
    macro_rules! primitive {
        ($t:ty) => {
            Some(Box::new(PrimitiveScatter::<$t>::new(data_type)))
        };
    }
    downcast_primitive! {
        data_type => (primitive),
        DataType::Boolean => Some(Box::new(BooleanScatter::new())),
        DataType::Utf8View => Some(Box::new(ViewScatter::<StringViewType>::new())),
        DataType::BinaryView => Some(Box::new(ViewScatter::<BinaryViewType>::new())),
        _ => None,
    }
}

fn new_nulls() -> Vec<NullBufferBuilder> {
    (0..NUM_BUCKETS).map(|_| NullBufferBuilder::new(0)).collect()
}

/// Appends the validity of the routed rows to their buckets. Returns the
/// bytes written.
fn scatter_nulls(
    nulls: &mut [NullBufferBuilder],
    source: Option<&NullBuffer>,
    offset: usize,
    ids: &[u8],
    counts: &[usize; NUM_BUCKETS],
) -> usize {
    match source {
        Some(source) => {
            for (row, &id) in ids.iter().enumerate() {
                nulls[id as usize].append(source.is_valid(offset + row));
            }
            ids.len().div_ceil(8)
        }
        // A builder that has seen no null holds no buffer
        None => {
            for (nulls, &count) in nulls.iter_mut().zip(counts) {
                if count > 0 {
                    nulls.append_n_non_nulls(count);
                }
            }
            0
        }
    }
}

/// Reserves exactly, so a bucket's first allocation is the `at_least` rows
/// the coalescer would allocate.
fn reserve_vec<T>(values: &mut Vec<T>, rows: usize, at_least: usize) {
    let wanted = values.len() + rows;
    if values.capacity() < wanted {
        values.reserve_exact(wanted.max(at_least) - values.len());
    }
}

struct PrimitiveScatter<T: ArrowPrimitiveType> {
    data_type: DataType,
    values: Vec<Vec<T::Native>>,
    nulls: Vec<NullBufferBuilder>,
}

impl<T: ArrowPrimitiveType> PrimitiveScatter<T> {
    fn new(data_type: &DataType) -> Self {
        Self {
            data_type: data_type.clone(),
            values: (0..NUM_BUCKETS).map(|_| Vec::new()).collect(),
            nulls: new_nulls(),
        }
    }
}

impl<T: ArrowPrimitiveType> ScatterColumn for PrimitiveScatter<T> {
    fn reserve(&mut self, bucket: usize, rows: usize, at_least: usize) {
        reserve_vec(&mut self.values[bucket], rows, at_least);
    }

    fn scatter(
        &mut self,
        source: &ArrayRef,
        offset: usize,
        ids: &[u8],
        counts: &[usize; NUM_BUCKETS],
    ) -> usize {
        let source = source.as_primitive::<T>();
        let values = &source.values()[offset..offset + ids.len()];

        // One write cursor per bucket, at the end of its rows
        let mut cursors: [*mut T::Native; NUM_BUCKETS] = std::array::from_fn(|bucket| {
            let buffer = &mut self.values[bucket];
            debug_assert!(buffer.len() + counts[bucket] <= buffer.capacity());
            // SAFETY: `len` is within the allocation (or 0 for none)
            unsafe { buffer.as_mut_ptr().add(buffer.len()) }
        });
        for (&id, &value) in ids.iter().zip(values) {
            // SAFETY: `id < NUM_BUCKETS`, and the caller reserved room for
            // `counts[id]` rows of bucket `id`, which this loop writes.
            unsafe {
                let cursor = cursors.get_unchecked_mut(id as usize);
                cursor.write(value);
                *cursor = cursor.add(1);
            }
        }
        for (buffer, &count) in self.values.iter_mut().zip(counts) {
            // SAFETY: the loop above initialized the next `count` values
            unsafe { buffer.set_len(buffer.len() + count) };
        }

        ids.len() * size_of::<T::Native>()
            + scatter_nulls(&mut self.nulls, source.nulls(), offset, ids, counts)
    }

    fn finish(&mut self, bucket: usize) -> ArrayRef {
        let values = std::mem::take(&mut self.values[bucket]);
        let nulls = self.nulls[bucket].finish();
        Arc::new(
            PrimitiveArray::<T>::new(ScalarBuffer::from(values), nulls)
                .with_data_type(self.data_type.clone()),
        )
    }

    fn size(&self, bucket: usize) -> usize {
        self.values[bucket].capacity() * size_of::<T::Native>()
            + self.nulls[bucket].allocated_size()
    }
}

struct BooleanScatter {
    values: Vec<BooleanBufferBuilder>,
    nulls: Vec<NullBufferBuilder>,
}

impl BooleanScatter {
    fn new() -> Self {
        Self {
            values: (0..NUM_BUCKETS)
                .map(|_| BooleanBufferBuilder::new(0))
                .collect(),
            nulls: new_nulls(),
        }
    }
}

impl ScatterColumn for BooleanScatter {
    fn reserve(&mut self, bucket: usize, rows: usize, at_least: usize) {
        let values = &mut self.values[bucket];
        let wanted = values.len() + rows;
        if values.capacity() < wanted {
            values.reserve(wanted.max(at_least) - values.len());
        }
    }

    fn scatter(
        &mut self,
        source: &ArrayRef,
        offset: usize,
        ids: &[u8],
        counts: &[usize; NUM_BUCKETS],
    ) -> usize {
        let source = source.as_boolean();
        let values = source.values();
        for (row, &id) in ids.iter().enumerate() {
            self.values[id as usize].append(values.value(offset + row));
        }
        ids.len().div_ceil(8)
            + scatter_nulls(&mut self.nulls, source.nulls(), offset, ids, counts)
    }

    fn finish(&mut self, bucket: usize) -> ArrayRef {
        let values = std::mem::replace(
            &mut self.values[bucket],
            BooleanBufferBuilder::new(0),
        )
        .finish();
        Arc::new(BooleanArray::new(values, self.nulls[bucket].finish()))
    }

    fn size(&self, bucket: usize) -> usize {
        self.values[bucket].capacity() / 8 + self.nulls[bucket].allocated_size()
    }
}

/// The rows in progress of one bucket of a view column. A long value's bytes
/// are copied into the bucket's own blocks, so the bucket's batches
/// reference nothing of the routed batch.
#[derive(Default)]
struct ViewBucket {
    views: Vec<u128>,
    /// The block being filled
    block: Vec<u8>,
    /// Full blocks of this batch
    blocks: Vec<Buffer>,
    blocks_bytes: usize,
    /// Capacity of the last block allocated
    block_size: usize,
}

impl ViewBucket {
    /// Copies `bytes` into the bucket's blocks and returns the index of
    /// their block and their offset in it.
    fn push_bytes(&mut self, bytes: &[u8]) -> (u32, u32) {
        if self.block.capacity() - self.block.len() < bytes.len() {
            self.seal_block();
            self.block_size = match self.block_size {
                0 => FIRST_BLOCK,
                size => (size * 2).min(MAX_BLOCK),
            };
            self.block = Vec::with_capacity(self.block_size.max(bytes.len()));
        }
        let offset = self.block.len() as u32;
        self.block.extend_from_slice(bytes);
        (self.blocks.len() as u32, offset)
    }

    fn seal_block(&mut self) {
        if !self.block.is_empty() {
            let block = Buffer::from_vec(std::mem::take(&mut self.block));
            self.blocks_bytes += block.capacity();
            self.blocks.push(block);
        }
    }
}

struct ViewScatter<T: ByteViewType + ?Sized> {
    buckets: Vec<ViewBucket>,
    nulls: Vec<NullBufferBuilder>,
    _type: PhantomData<T>,
}

impl<T: ByteViewType + ?Sized> ViewScatter<T> {
    fn new() -> Self {
        Self {
            buckets: (0..NUM_BUCKETS).map(|_| ViewBucket::default()).collect(),
            nulls: new_nulls(),
            _type: PhantomData,
        }
    }
}

impl<T: ByteViewType + ?Sized> ScatterColumn for ViewScatter<T> {
    fn reserve(&mut self, bucket: usize, rows: usize, at_least: usize) {
        reserve_vec(&mut self.buckets[bucket].views, rows, at_least);
    }

    fn scatter(
        &mut self,
        source: &ArrayRef,
        offset: usize,
        ids: &[u8],
        counts: &[usize; NUM_BUCKETS],
    ) -> usize {
        let source = source.as_byte_view::<T>();
        let views = &source.views()[offset..offset + ids.len()];
        let buffers = source.data_buffers();
        let nulls = source.nulls();

        // Rows are read in the order they, and their bytes, lie in the batch
        let mut bytes = ids.len() * size_of::<u128>();
        for (row, (&id, &view)) in ids.iter().zip(views).enumerate() {
            let bucket = &mut self.buckets[id as usize];
            let len = view as u32;
            let routed = if nulls.is_some_and(|nulls| nulls.is_null(offset + row)) {
                // A null's view may point anywhere: write an empty one
                0
            } else if len <= INLINE {
                view
            } else {
                // Long view: length and prefix in the low 64 bits, then the
                // buffer index and the offset into that buffer
                let buffer = (view >> 64) as u32 as usize;
                let start = (view >> 96) as u32 as usize;
                let (block, at) =
                    bucket.push_bytes(&buffers[buffer][start..start + len as usize]);
                bytes += len as usize;
                (view & u128::from(u64::MAX))
                    | (u128::from(block) << 64)
                    | (u128::from(at) << 96)
            };
            debug_assert!(bucket.views.len() < bucket.views.capacity());
            bucket.views.push(routed);
        }
        bytes + scatter_nulls(&mut self.nulls, nulls, offset, ids, counts)
    }

    fn finish(&mut self, bucket: usize) -> ArrayRef {
        let nulls = self.nulls[bucket].finish();
        let bucket = &mut self.buckets[bucket];
        bucket.seal_block();
        let views = std::mem::take(&mut bucket.views);
        let blocks = std::mem::take(&mut bucket.blocks);
        bucket.blocks_bytes = 0;
        // SAFETY: every view is a valid view of a routed array (or zero, for
        // a null), and every long view points at an identical copy of its
        // bytes in `blocks`: valid UTF-8 for a string view, as the source's
        // were.
        Arc::new(unsafe {
            GenericByteViewArray::<T>::new_unchecked(
                ScalarBuffer::from(views),
                blocks.into(),
                nulls,
            )
        })
    }

    fn size(&self, bucket: usize) -> usize {
        let nulls = self.nulls[bucket].allocated_size();
        let bucket = &self.buckets[bucket];
        bucket.views.capacity() * size_of::<u128>()
            + bucket.block.capacity()
            + bucket.blocks_bytes
            + nulls
    }
}
