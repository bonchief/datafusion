// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A per-batch memo from a 16-byte view to the first row carrying it, so a
//! group map is probed once per distinct view rather than once per row.
//!
//! A byte view array decoded from a dictionary, such as a Parquet dictionary
//! page or a dictionary-encoded column exported to views, gives every row of
//! the same value the same view: the same length, prefix, buffer and offset.
//! Two rows with bit-identical views therefore hold equal bytes, whether the
//! value is inlined in the view (up to 12 bytes) or not. The converse does
//! not hold: equal bytes can sit at different offsets, and such rows simply
//! stay separate here and meet in the group map, which decides equality. So
//! deduplicating by view is exact, and only ever saves lookups.
//!
//! The memo costs a hash and a probe of a small table per row, which pays
//! only when enough rows repeat a view. Like DuckDB's dictionary aggregation
//! (`GroupedAggregateHashTable::TryAddDictionaryGroups`), it is gated per
//! batch: it gives up after the first [`SAMPLE`] rows when fewer than an
//! eighth of them repeat, and a batch where fewer than a quarter of the rows
//! repeat backs the memo off
//! for a doubling number of batches, up to [`MAX_BACKOFF`], so a stream
//! without shared views (plain decoding, high-distinct values, a final
//! aggregate's merged input) pays only the occasional sample.

use arrow::array::{Array, ArrayRef, AsArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType};
use datafusion_common::Result;
use std::mem::size_of;

/// Rows probed before deciding whether the rest of a batch is worth it.
const SAMPLE: usize = 256;

/// Batches shorter than this are passed straight through.
const MIN_ROWS: usize = SAMPLE * 2;

/// The longest run of batches the memo sits out after a batch that did not pay.
const MAX_BACKOFF: u32 = 64;

/// The sample goes on to the whole batch when at least one row in
/// `SAMPLE_REPEATS` repeats an earlier view. Repeats spread across a batch
/// show less in its first rows, so this is looser than [`REPEATS`].
const SAMPLE_REPEATS: usize = 8;

/// A batch pays when at least one row in `REPEATS` repeats an earlier view.
const REPEATS: usize = 4;

/// See the [module documentation](self).
#[derive(Debug, Default)]
pub(super) struct ViewDedupe {
    /// Open addressing table: the batch's stamp in the high 32 bits of a
    /// slot, the index into `firsts` in the low 32 bits. A slot with an older
    /// stamp is empty, so the table is never cleared between batches.
    slots: Vec<u64>,
    stamp: u32,
    /// The row of each distinct view, in order of first appearance.
    firsts: Vec<u32>,
    /// For each row of the batch, its index into `firsts`.
    rows: Vec<u32>,
    /// Batches still to pass through, and the length of the current back-off.
    idle: u32,
    backoff: u32,
    /// Batches deduplicated, for tests.
    #[cfg(test)]
    used: usize,
}

impl ViewDedupe {
    /// The distinct views of `values` as an array of one row per distinct
    /// view, in order of first appearance, with [`Self::rows`] mapping each
    /// row of `values` to its row in it; or `None` when this batch is not
    /// worth deduplicating and should be looked up as it is.
    pub(super) fn distinct(&mut self, values: &ArrayRef) -> Result<Option<ArrayRef>> {
        if values.len() < MIN_ROWS {
            return Ok(None);
        }
        if self.idle > 0 {
            self.idle -= 1;
            return Ok(None);
        }
        let paid = match values.data_type() {
            DataType::BinaryView => self.scan::<BinaryViewType>(values),
            DataType::Utf8View => self.scan::<StringViewType>(values),
            _ => return Ok(None),
        };
        let Some(paid) = paid else {
            self.back_off();
            return Ok(None);
        };
        if paid {
            self.backoff = 0;
        } else {
            // The work is done, so use it, but do not try again soon.
            self.back_off();
        }
        #[cfg(test)]
        {
            self.used += 1;
        }
        let firsts = UInt32Array::from(self.firsts.clone());
        Ok(Some(take(values.as_ref(), &firsts, None)?))
    }

    /// For each row of the batch [`Self::distinct`] last deduplicated, its row
    /// in the array it returned.
    pub(super) fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// The memory this memo holds.
    pub(super) fn size(&self) -> usize {
        self.slots.capacity() * size_of::<u64>()
            + (self.firsts.capacity() + self.rows.capacity()) * size_of::<u32>()
    }

    /// Releases the memo's buffers.
    pub(super) fn release(&mut self) {
        *self = Self::default();
    }

    #[cfg(test)]
    pub(super) fn used(&self) -> usize {
        self.used
    }

    fn back_off(&mut self) {
        self.backoff = (self.backoff * 2).clamp(1, MAX_BACKOFF);
        self.idle = self.backoff;
    }

    /// Fills `firsts` and `rows` for `values`. `None` when the sample shows
    /// too few repeats; otherwise whether the whole batch repeated enough.
    fn scan<B: ByteViewType>(&mut self, values: &ArrayRef) -> Option<bool> {
        let array = values.as_byte_view::<B>();
        let views = array.views();
        let nulls = array.nulls().filter(|nulls| nulls.null_count() > 0);
        let len = views.len();

        let capacity = (len * 2).next_power_of_two();
        if self.slots.len() < capacity {
            self.slots.resize(capacity, 0);
        }
        // Only the first `capacity` slots are used, so the hash keeps the
        // bits that index them.
        let bits = capacity.trailing_zeros();
        let mask = capacity - 1;
        if self.stamp == u32::MAX {
            self.slots.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        let stamp = u64::from(self.stamp) << 32;
        self.firsts.clear();
        self.rows.clear();
        self.rows.reserve(len);
        let mut null_first: Option<u32> = None;

        for (row, &view) in views.iter().enumerate() {
            if row == SAMPLE && (row - self.firsts.len()) * SAMPLE_REPEATS < row {
                return None;
            }
            if nulls.is_some_and(|nulls| nulls.is_null(row)) {
                let first = *null_first.get_or_insert_with(|| {
                    self.firsts.push(row as u32);
                    (self.firsts.len() - 1) as u32
                });
                self.rows.push(first);
                continue;
            }
            let mut slot = hash(view, bits) & mask;
            let first = loop {
                let entry = self.slots[slot];
                if entry & !u64::from(u32::MAX) != stamp {
                    let first = self.firsts.len() as u32;
                    self.slots[slot] = stamp | u64::from(first);
                    self.firsts.push(row as u32);
                    break first;
                }
                let first = entry as u32;
                if views[self.firsts[first as usize] as usize] == view {
                    break first;
                }
                slot = (slot + 1) & mask;
            };
            self.rows.push(first);
        }
        Some((len - self.firsts.len()) * REPEATS >= len)
    }
}

/// The top `bits` bits of a multiplicative mix of both halves of `view`: its
/// length and prefix, and its buffer and offset or the rest of its bytes.
#[inline]
fn hash(view: u128, bits: u32) -> usize {
    let mixed = ((view as u64) ^ ((view >> 64) as u64).rotate_left(29))
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (mixed >> (64 - bits)) as usize
}
