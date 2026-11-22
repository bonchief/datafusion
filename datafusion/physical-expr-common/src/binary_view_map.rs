// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ArrowBytesViewMap`] and [`ArrowBytesViewSet`] for storing maps/sets of values from
//! `StringViewArray`/`BinaryViewArray`.
use crate::binary_map::{OutputType, single_null_buffer};
use crate::prefetch::{DISTANCE, l2_cache_bytes, prefetch};
use arrow::array::cast::AsArray;
use arrow::array::{
    Array, ArrayRef, BinaryViewArray, BinaryViewBuilder, ByteView, GenericByteViewArray,
    StringViewBuilder, make_view,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType};
use datafusion_common::hash_utils::RandomState;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_common::{Result, exec_err};
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;

const BYTE_VIEW_STARTING_BLOCK_SIZE: usize = 8 * 1024;
const BYTE_VIEW_MAX_BLOCK_SIZE: usize = 2 * 1024 * 1024;

/// HashSet optimized for storing string or binary values that can produce that
/// the final set as a `GenericBinaryViewArray` with minimal copies.
#[derive(Debug)]
pub struct ArrowBytesViewSet(ArrowBytesViewMap<()>);

impl ArrowBytesViewSet {
    /// Creates a set that does not pre-allocate its hash table.
    ///
    /// See [`ArrowBytesViewMap::new`] for when to prefer this over
    /// [`Self::with_capacity`].
    pub fn new(output_type: OutputType) -> Self {
        Self(ArrowBytesViewMap::new(output_type))
    }

    /// Creates a set with room for `map_capacity` entries.
    ///
    /// See [`ArrowBytesViewMap::with_capacity`].
    pub fn with_capacity(output_type: OutputType, map_capacity: usize) -> Self {
        Self(ArrowBytesViewMap::with_capacity(output_type, map_capacity))
    }

    /// Inserts each value from `values` into the set
    pub fn insert(&mut self, values: &ArrayRef) {
        fn make_payload_fn(_value: Option<&[u8]>) {}
        fn observe_payload_fn(_payload: ()) {}
        self.0
            .insert_if_new(values, make_payload_fn, observe_payload_fn);
    }

    /// Return the contents of this map and replace it with a new empty map with
    /// the same output type
    pub fn take(&mut self) -> Self {
        Self(self.0.take())
    }

    /// Converts this set into a `StringViewArray` or `BinaryViewArray`
    /// containing each distinct value that was interned.
    /// This is done without copying the values.
    pub fn into_state(self) -> ArrayRef {
        self.0.into_state()
    }

    /// Returns the total number of distinct values (including nulls) seen so far
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// returns the total number of distinct values (not including nulls) seen so far
    pub fn non_null_len(&self) -> usize {
        self.0.non_null_len()
    }

    /// Return the total size, in bytes, of memory used to store the data in
    /// this set, not including `self`
    pub fn size(&self) -> usize {
        self.0.size()
    }
}

/// Optimized map for storing Arrow "byte view" types (`StringView`, `BinaryView`)
/// values that can produce the set of keys on
/// output as `GenericBinaryViewArray` without copies.
///
/// Equivalent to `HashSet<String, V>` but with better performance if you need
/// to emit the keys as an Arrow `StringViewArray` / `BinaryViewArray`. For other
/// purposes it is the same as a `HashMap<String, V>`
///
/// # Generic Arguments
///
/// * `V`: payload type
///
/// # Description
///
/// This is a specialized HashMap with the following properties:
///
/// 1. Optimized for storing and emitting Arrow byte types  (e.g.
///    `StringViewArray` / `BinaryViewArray`) very efficiently by minimizing copying of
///    the string values themselves, both when inserting and when emitting the
///    final array.
///
/// 2. Retains the insertion order of entries in the final array. The values are
///    in the same order as they were inserted.
///
/// Note this structure can be used as a `HashSet` by specifying the value type
/// as `()`, as is done by [`ArrowBytesViewSet`].
///
/// This map is used by the special `COUNT DISTINCT` aggregate function to
/// store the distinct values, and by the `GROUP BY` operator to store
/// group values when they are a single string array.
///
/// # Layout and invariants
///
/// Each distinct non-null key has an index: its position in `views`, which is
/// the order it was first seen. `views`, `hashes` and `payloads` all have one
/// element per index. The NULL key, if seen, also takes an index (so it is
/// emitted in its place), with hash 0 and a default payload there; its real
/// payload is in `null`, and no slot names it.
///
/// `slots` is an open-addressed table with linear probing. Its length is zero
/// or a power of two, and at most three quarters of it is occupied, so every
/// probe ends at an empty slot. A slot is zero (empty) or `tag | (index + 1)`:
/// the top 16 bits of the key's hash, and the index in the low 48 bits (an
/// index cannot reach 2^48 − 1, since `views` would then hold 4 PiB). A key's
/// probe starts at `hash & (len − 1)`. Every distinct non-null key is in
/// exactly one slot, and `items` counts them.
///
/// A key matches a slot when the tag, then the stored view's length, then (for
/// a key longer than 12 bytes) its 4-byte prefix and finally its bytes are
/// equal; a key of 12 bytes or fewer is wholly inside its view, so the views
/// are compared instead. Because the hash is kept in `hashes`, growth places
/// every index again without reading or hashing a key.
///
/// # Staged probe
///
/// Probing one row at a time takes up to three dependent cache misses per row:
/// the slot, the stored view (and payload), and the stored key bytes. When a
/// batch is longer than `3 * DISTANCE` rows and the bytes a probe touches
/// (slots, views, payloads and key bytes) exceed the core's L2 cache, the batch
/// is probed in stages instead: while row `i` is looked up, the slot of row
/// `i + 3D` is prefetched, the view and payload that row `i + 2D`'s tag selects
/// are prefetched, and the key bytes of row `i + D`'s candidate are prefetched
/// when its length and prefix already match, so the misses of different rows
/// overlap. The earlier stages only prefetch; every lookup, insert and payload
/// call still happens for row `i` in row order, exactly as on the plain path.
/// Results, payloads, group order and memory use are therefore identical on
/// both paths.
pub struct ArrowBytesViewMap<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// Should the output be StringView or BinaryView?
    output_type: OutputType,
    /// Open-addressed table of `tag | (index + 1)`, see the type docs
    slots: Vec<u64>,
    /// Number of occupied slots: the distinct non-null keys
    items: usize,
    /// Hash table capacity to re-create the map with in [`Self::take`], so a
    /// map built with [`Self::with_capacity`] keeps its pre-allocation when it
    /// is emptied and reused
    initial_map_capacity: usize,

    /// Views for all stored values (in insertion order)
    views: Vec<u128>,
    /// Hash of the key at each index, read only to place the keys on growth
    hashes: Vec<u64>,
    /// Payload of the key at each index
    payloads: Vec<V>,
    /// In-progress buffer for out-of-line string data
    in_progress: Vec<u8>,
    /// Completed buffers containing string data
    completed: Vec<Buffer>,
    /// Allocation target, doubled for each new payload block up to 2 MiB.
    /// Zero keeps lazy payload growth for the small per-group distinct maps.
    block_size: usize,

    /// random state used to generate hashes
    random_state: RandomState,
    /// buffer that stores hash values (reused across batches to save allocations)
    hashes_buffer: Vec<u64>,
    /// `(payload, null_index)` for the 'null' value, if any
    /// NOTE null_index is the logical index in the final array, not the index
    /// in the buffer
    null: Option<(V, usize)>,
}

/// The size, in number of entries, of the hash table pre-allocated by
/// [`ArrowBytesViewMap::with_capacity`]. It is a warm up size for maps that go
/// on to hold many values, not a bound on what the map can hold.
pub const INITIAL_MAP_CAPACITY: usize = 512;

impl<V> ArrowBytesViewMap<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// Creates a map that does not pre-allocate its hash table.
    ///
    /// Use this when maps are created in large numbers and most of them stay
    /// small, such as the per group `COUNT(DISTINCT)` accumulators that
    /// `GroupsAccumulatorAdapter` creates one of per group. There the
    /// pre-allocation dwarfs the values the map actually holds.
    pub fn new(output_type: OutputType) -> Self {
        Self::with_capacity(output_type, 0)
    }

    /// Creates a map whose hash table is pre-allocated for `map_capacity`
    /// entries.
    ///
    /// Use this for the few long lived maps that are each expected to hold many
    /// values, such as the single map backing a `GROUP BY` on one string
    /// column. The capacity is preserved across [`Self::take`].
    pub fn with_capacity(output_type: OutputType, map_capacity: usize) -> Self {
        Self {
            output_type,
            slots: vec![EMPTY; slots_for(map_capacity)],
            items: 0,
            initial_map_capacity: map_capacity,
            views: Vec::new(),
            hashes: Vec::new(),
            payloads: Vec::new(),
            in_progress: Vec::new(),
            completed: Vec::new(),
            block_size: if map_capacity == 0 {
                0
            } else {
                BYTE_VIEW_STARTING_BLOCK_SIZE
            },
            random_state: RandomState::default(),
            hashes_buffer: vec![],
            null: None,
        }
    }

    /// Reserves group slots without preallocating variable-length string payloads.
    pub fn reserve_groups(&mut self, capacity: usize) {
        if self.block_size == 0 && capacity != 0 {
            self.block_size = BYTE_VIEW_STARTING_BLOCK_SIZE;
        }
        let slots = slots_for(capacity);
        if slots > self.slots.len() {
            self.resize(slots);
        }
        let more = capacity.saturating_sub(self.views.len());
        self.views.reserve_exact(more);
        self.hashes.reserve_exact(more);
        self.payloads.reserve_exact(more);
        self.hashes_buffer
            .reserve_exact(capacity.saturating_sub(self.hashes_buffer.len()));
    }

    /// Emits keys, retains the hash table and scratch buffer, and reserves fixed
    /// view capacity. Payload blocks restart their exponential allocation growth.
    pub fn take_with_capacity(&mut self, capacity: usize) -> ArrayRef {
        let mut outgoing = Self::new(self.output_type);
        mem::swap(self, &mut outgoing);
        mem::swap(&mut self.slots, &mut outgoing.slots);
        self.slots.fill(EMPTY);
        mem::swap(&mut self.hashes, &mut outgoing.hashes);
        self.hashes.clear();
        mem::swap(&mut self.payloads, &mut outgoing.payloads);
        self.payloads.clear();
        mem::swap(&mut self.hashes_buffer, &mut outgoing.hashes_buffer);
        self.hashes_buffer.clear();
        self.initial_map_capacity = outgoing.initial_map_capacity;
        mem::swap(&mut self.random_state, &mut outgoing.random_state);
        self.reserve_groups(capacity);
        outgoing.into_state()
    }

    /// Return the contents of this map and replace it with a new empty map with
    /// the same output type
    pub fn take(&mut self) -> Self {
        let mut new_self =
            Self::with_capacity(self.output_type, self.initial_map_capacity);
        mem::swap(self, &mut new_self);
        new_self
    }

    /// Empties this map and releases every allocation it holds, so
    /// [`Self::size`] drops to approximately zero.
    ///
    /// This is the difference from [`Self::take`]: `take` restores the capacity
    /// the map was configured with so the emptied map stays warm for continued
    /// use, which is what emitting wants, whereas here the point is to hand the
    /// memory back, as before spilling or before a downstream sort. The
    /// configured capacity is remembered, so the next [`Self::take`] warms the
    /// map up again.
    pub fn clear_and_release(&mut self) {
        let mut released = Self::with_capacity(self.output_type, 0);
        released.initial_map_capacity = self.initial_map_capacity;
        *self = released;
    }

    /// Inserts each value from `values` into the map, invoking `payload_fn` for
    /// each value if *not* already present, deferring the allocation of the
    /// payload until it is needed.
    ///
    /// Note that this is different than a normal map that would replace the
    /// existing entry
    ///
    /// # Arguments:
    ///
    /// `values`: array whose values are inserted
    ///
    /// `make_payload_fn`:  invoked for each value that is not already present
    /// to create the payload, in order of the values in `values`
    ///
    /// `observe_payload_fn`: invoked once, for each value in `values`, that was
    /// already present in the map, with corresponding payload value.
    ///
    /// # Returns
    ///
    /// The payload value for the entry, either the existing value or
    /// the newly inserted value
    ///
    /// # Safety:
    ///
    /// Note that `make_payload_fn` and `observe_payload_fn` are only invoked
    /// with valid values from `values`, not for the `NULL` value.
    pub fn insert_if_new<MP, OP>(
        &mut self,
        values: &ArrayRef,
        make_payload_fn: MP,
        observe_payload_fn: OP,
    ) where
        MP: FnMut(Option<&[u8]>) -> V,
        OP: FnMut(V),
    {
        self.insert(values, None, make_payload_fn, observe_payload_fn)
    }

    /// [`Self::insert_if_new`], probing in stages when `staged` says so, or
    /// when it is `None` and the staged probe pays (see the type docs)
    fn insert<MP, OP>(
        &mut self,
        values: &ArrayRef,
        staged: Option<bool>,
        make_payload_fn: MP,
        observe_payload_fn: OP,
    ) where
        MP: FnMut(Option<&[u8]>) -> V,
        OP: FnMut(V),
    {
        // Sanity check array type
        match self.output_type {
            OutputType::BinaryView => {
                assert!(matches!(values.data_type(), DataType::BinaryView));
                self.insert_if_new_inner::<MP, OP, BinaryViewType>(
                    values,
                    staged,
                    make_payload_fn,
                    observe_payload_fn,
                )
            }
            OutputType::Utf8View => {
                assert!(matches!(values.data_type(), DataType::Utf8View));
                self.insert_if_new_inner::<MP, OP, StringViewType>(
                    values,
                    staged,
                    make_payload_fn,
                    observe_payload_fn,
                )
            }
            _ => unreachable!("Utf8/Binary should use `ArrowBytesSet`"),
        }
    }

    /// Generic version of [`Self::insert_if_new`] that handles `ByteViewType`
    /// (both StringView and BinaryView)
    ///
    /// Note this is the only function that is generic on [`ByteViewType`], which
    /// avoids having to template the entire structure,  making the code
    /// simpler and understand and reducing code bloat due to duplication.
    ///
    /// See comments on `insert_if_new` for more details
    fn insert_if_new_inner<MP, OP, B>(
        &mut self,
        values: &ArrayRef,
        staged: Option<bool>,
        mut make_payload_fn: MP,
        mut observe_payload_fn: OP,
    ) where
        MP: FnMut(Option<&[u8]>) -> V,
        OP: FnMut(V),
        B: ByteViewType,
    {
        // step 1: compute hashes
        let mut hashes = mem::take(&mut self.hashes_buffer);
        hashes.clear();
        hashes.resize(values.len(), 0);
        create_hashes([values], &self.random_state, &mut hashes)
            // hash is supported for all types and create_hashes only
            // returns errors for unsupported types
            .unwrap();

        // step 2: insert each value into the set, if not already present
        let values = values.as_byte_view::<B>();

        // Ensure lengths are equivalent
        assert_eq!(values.len(), hashes.len());

        let staged = staged.unwrap_or_else(|| self.stage_pays(values.len()));
        if staged {
            let views = values.views();
            for (i, &hash) in hashes.iter().enumerate() {
                if let Some(&ahead) = hashes.get(i + 3 * DISTANCE) {
                    self.prefetch_slot(ahead);
                }
                if let Some(&ahead) = hashes.get(i + 2 * DISTANCE) {
                    self.prefetch_entry(ahead);
                }
                if let Some(&ahead) = hashes.get(i + DISTANCE) {
                    self.prefetch_key(ahead, views[i + DISTANCE]);
                }
                self.insert_row(
                    values,
                    i,
                    hash,
                    &mut make_payload_fn,
                    &mut observe_payload_fn,
                );
            }
        } else {
            for (i, &hash) in hashes.iter().enumerate() {
                self.insert_row(
                    values,
                    i,
                    hash,
                    &mut make_payload_fn,
                    &mut observe_payload_fn,
                );
            }
        }
        self.hashes_buffer = hashes;
    }

    /// Looks up row `i` of `values` and inserts it if it is new: the one step
    /// both the plain and the staged probe take for every row, in row order.
    #[inline(always)]
    fn insert_row<MP, OP, B>(
        &mut self,
        values: &GenericByteViewArray<B>,
        i: usize,
        hash: u64,
        make_payload_fn: &mut MP,
        observe_payload_fn: &mut OP,
    ) where
        MP: FnMut(Option<&[u8]>) -> V,
        OP: FnMut(V),
        B: ByteViewType,
    {
        // handle null value via validity bitmap check
        if values.is_null(i) {
            let payload = if let Some(&(payload, _offset)) = self.null.as_ref() {
                payload
            } else {
                let payload = make_payload_fn(None);
                let null_index = self.views.len();
                self.views.push(0);
                self.hashes.push(0);
                self.payloads.push(V::default());
                self.null = Some((payload, null_index));
                payload
            };
            observe_payload_fn(payload);
            return;
        }

        let view = values.views()[i];
        // Extract length from the view (first 4 bytes of u128 in little-endian)
        let len = view as u32;
        let value: &[u8] = if len <= 12 {
            &[]
        } else {
            values.value(i).as_ref()
        };

        let payload = match self.find(hash, view, value) {
            Ok(index) => self.payloads[index],
            Err(vacant) => {
                let slot = if self.items < max_items(self.slots.len()) {
                    vacant
                } else {
                    self.resize(slots_for(self.items + 1));
                    self.vacant(hash)
                };
                let index = self.views.len();
                let payload = if len <= 12 {
                    // Inline path: bytes are already packed in view.
                    // The inline ByteView format is [len:u32 LE][data:12 bytes zero-padded],
                    // so extracting bytes from the u128 avoids a round-trip through
                    // values.value(i) (which reads the views buffer and returns the same slice).
                    let view_bytes = view.to_le_bytes();
                    let payload = make_payload_fn(Some(&view_bytes[4..4 + len as usize]));
                    // For inline strings, the stored view is identical to the input view:
                    // make_view(value, 0, 0) produces the same u128 as view.
                    //
                    // SAFETY: view was a valid view, and the enclosing `len <= 12`
                    // ensures it is inline
                    unsafe { self.append_inline_view(view) };
                    payload
                } else {
                    let payload = make_payload_fn(Some(value));
                    self.append_value(value);
                    payload
                };
                self.hashes.push(hash);
                self.payloads.push(payload);
                self.slots[slot] = slot_of(hash, index);
                self.items += 1;
                payload
            }
        };
        observe_payload_fn(payload);
    }

    /// Finds the index of the key with `hash`, input `view` and, when it is
    /// longer than 12 bytes, bytes `value`; or, if it is absent, the empty slot
    /// its probe ended at. With no slots the position is meaningless, and the
    /// caller grows the table before it inserts.
    #[inline(always)]
    fn find(&self, hash: u64, view: u128, value: &[u8]) -> Result<usize, usize> {
        if self.slots.is_empty() {
            return Err(0);
        }
        let mask = self.slots.len() - 1;
        let tag = tag(hash);
        let mut pos = hash as usize & mask;
        loop {
            let slot = self.slots[pos];
            if slot == EMPTY {
                return Err(pos);
            }
            if slot & !INDEX_MASK == tag {
                let index = index_of(slot);
                if self.matches(self.views[index], view, value) {
                    return Ok(index);
                }
            }
            pos = (pos + 1) & mask;
        }
    }

    /// Whether the key stored with view `stored` is the input key with `view`
    /// and, for a key longer than 12 bytes, bytes `value`.
    #[inline(always)]
    fn matches(&self, stored: u128, view: u128, value: &[u8]) -> bool {
        let len = view as u32;
        if stored as u32 != len {
            return false;
        }
        // Fast path: inline strings can be compared directly
        if len <= 12 {
            return stored == view;
        }
        // For larger strings: first compare the 4-byte prefix
        if (stored >> 32) as u32 != (view >> 32) as u32 {
            return false;
        }
        // Prefix matched - compare full bytes
        self.stored_value(stored) == value
    }

    /// The bytes of a stored view longer than 12 bytes.
    fn stored_value(&self, stored: u128) -> &[u8] {
        let byte_view = ByteView::from(stored);
        let len = byte_view.length as usize;
        let buffer_index = byte_view.buffer_index as usize;
        let offset = byte_view.offset as usize;
        if buffer_index < self.completed.len() {
            &self.completed[buffer_index].as_slice()[offset..offset + len]
        } else {
            &self.in_progress[offset..offset + len]
        }
    }

    /// The first empty slot of `hash`'s probe. The table has a slot and room.
    fn vacant(&self, hash: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut pos = hash as usize & mask;
        while self.slots[pos] != EMPTY {
            pos = (pos + 1) & mask;
        }
        pos
    }

    /// Replaces the table with `len` slots, a power of two that holds every
    /// key, and places each key again from its stored hash, in index order.
    fn resize(&mut self, len: usize) {
        let mut slots = vec![EMPTY; len];
        let mask = len - 1;
        let null = self.null.map(|(_, index)| index);
        for (index, &hash) in self.hashes.iter().enumerate() {
            if Some(index) == null {
                continue;
            }
            let mut pos = hash as usize & mask;
            while slots[pos] != EMPTY {
                pos = (pos + 1) & mask;
            }
            slots[pos] = slot_of(hash, index);
        }
        self.slots = slots;
    }

    /// Whether a batch of `rows` should take the staged probe: it has rows
    /// beyond the look-ahead, and the memory a probe reads is larger than L2.
    fn stage_pays(&self, rows: usize) -> bool {
        if rows <= 3 * DISTANCE {
            return false;
        }
        let keys = self.in_progress.len()
            + self.completed.iter().map(Buffer::len).sum::<usize>();
        let probed = self.slots.len() * size_of::<u64>()
            + self.views.len() * size_of::<u128>()
            + self.payloads.len() * size_of::<V>()
            + keys;
        probed > l2_cache_bytes()
    }

    /// Staged probe, first stage: prefetch the home slot of `hash`.
    #[inline(always)]
    fn prefetch_slot(&self, hash: u64) {
        if !self.slots.is_empty() {
            let pos = hash as usize & (self.slots.len() - 1);
            prefetch(self.slots.as_ptr().wrapping_add(pos));
        }
    }

    /// The index named by the first slot matching `hash`'s tag within
    /// [`HINT_SLOTS`] of its home, for the prefetching stages. It is only a
    /// guess: the lookup of the row decides.
    #[inline(always)]
    fn candidate(&self, hash: u64) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let tag = tag(hash);
        let home = hash as usize & mask;
        (0..HINT_SLOTS)
            .map(|step| self.slots[(home + step) & mask])
            .take_while(|slot| *slot != EMPTY)
            .find(|slot| slot & !INDEX_MASK == tag)
            .map(index_of)
    }

    /// Staged probe, second stage: prefetch the view and payload that the
    /// slot matching `hash`'s tag names.
    #[inline(always)]
    fn prefetch_entry(&self, hash: u64) {
        if let Some(index) = self.candidate(hash) {
            prefetch(self.views.as_ptr().wrapping_add(index));
            if size_of::<V>() != 0 {
                prefetch(self.payloads.as_ptr().wrapping_add(index));
            }
        }
    }

    /// Staged probe, third stage: when the candidate for an input key longer
    /// than 12 bytes has its length and prefix, prefetch the stored bytes the
    /// lookup will compare: the first and the last line they occupy.
    #[inline(always)]
    fn prefetch_key(&self, hash: u64, view: u128) {
        let len = view as u32;
        if len <= 12 {
            return;
        }
        let Some(index) = self.candidate(hash) else {
            return;
        };
        let stored = self.views[index];
        if stored as u32 != len || (stored >> 32) as u32 != (view >> 32) as u32 {
            return;
        }
        let byte_view = ByteView::from(stored);
        let base = match self.completed.get(byte_view.buffer_index as usize) {
            Some(buffer) => buffer.as_ptr(),
            None => self.in_progress.as_ptr(),
        };
        let start = base.wrapping_add(byte_view.offset as usize);
        prefetch(start);
        prefetch(start.wrapping_add(len as usize - 1));
    }

    /// Returns the bytes for the key at `index`, irrespective of nullness.
    fn key(&self, index: usize) -> &[u8] {
        let view = &self.views[index];
        let byte_view = ByteView::from(*view);
        let length = byte_view.length as usize;
        if length <= 12 {
            // SAFETY: `view` is a valid inline view with `length` bytes.
            unsafe { BinaryViewArray::inline_value(view, length) }
        } else {
            self.stored_value(*view)
        }
    }

    /// Converts this set into a `StringViewArray`, or `BinaryViewArray`,
    /// containing each distinct value
    /// that was inserted. This is done without copying the values.
    ///
    /// The values are guaranteed to be returned in the same order in which
    /// they were first seen.
    pub fn into_state(mut self) -> ArrayRef {
        // Flush any remaining in-progress buffer
        if !self.in_progress.is_empty() {
            let flushed = mem::take(&mut self.in_progress);
            self.completed.push(Buffer::from_vec(flushed));
        }

        // Build null buffer if we have any nulls
        let null_buffer = match self.null {
            Some((_payload, index)) => Some(single_null_buffer(self.views.len(), index)),
            None => None,
        };

        let views = ScalarBuffer::from(self.views);
        let array = unsafe {
            BinaryViewArray::new_unchecked(views, self.completed.into(), null_buffer)
        };

        match self.output_type {
            OutputType::BinaryView => Arc::new(array),
            OutputType::Utf8View => {
                // SAFETY: all input was valid utf8
                let array = unsafe { array.to_string_view_unchecked() };
                Arc::new(array)
            }
            _ => unreachable!("Utf8/Binary should use `ArrowBytesMap`"),
        }
    }

    /// Copies keys at `indices` into an array without changing this map.
    ///
    /// Keys are returned in index order, and duplicate indices are supported.
    pub fn keys<I>(&self, indices: I) -> Result<ArrayRef>
    where
        I: IntoIterator<Item = usize>,
    {
        let indices = indices.into_iter();
        let capacity = indices.size_hint().0;
        let num_keys = self.views.len();
        let null_index = self.null.map(|(_, index)| index);

        macro_rules! build_keys {
            ($builder:ty, $value:expr) => {{
                let mut builder = <$builder>::with_capacity(capacity);
                for index in indices {
                    if index >= num_keys {
                        return exec_err!(
                            "Key index {index} is out of bounds for {num_keys} keys"
                        );
                    }
                    if null_index == Some(index) {
                        builder.append_null();
                    } else {
                        builder.append_value($value(self.key(index)));
                    }
                }
                Arc::new(builder.finish()) as ArrayRef
            }};
        }

        Ok(match self.output_type {
            OutputType::BinaryView => build_keys!(BinaryViewBuilder, |value| value),
            OutputType::Utf8View => build_keys!(StringViewBuilder, |value| unsafe {
                // Inputs to an Utf8View map are validated UTF-8.
                std::str::from_utf8_unchecked(value)
            }),
            _ => unreachable!("Utf8/Binary should use `ArrowBytesMap`"),
        })
    }

    /// Append an already-computed inline view (len <= 12) directly, bypassing
    /// buffer allocation.
    ///
    /// Returns the view that was stored (identical to the argument).
    ///
    /// # Safety
    ///
    /// `view` must be a valid inline `ByteView`: the length field in the low
    /// 32 bits must be <= 12, and the remaining 12 bytes must hold the
    /// value's bytes (zero-padded if shorter). Calling with a non-inline view
    /// would store a value that downstream `views` consumers interpret as
    /// `[buffer_index, offset]` into the `completed`/`in_progress` buffers,
    /// which is unsound for any view that didn't originate from a real
    /// allocation in those buffers.
    unsafe fn append_inline_view(&mut self, view: u128) -> u128 {
        self.views.push(view);
        view
    }

    /// Append a value to our buffers and return the view pointing to it
    fn append_value(&mut self, value: &[u8]) -> u128 {
        let len = value.len();
        let view = if len <= 12 {
            make_view(value, 0, 0)
        } else {
            let block_limit = if self.block_size == 0 {
                BYTE_VIEW_MAX_BLOCK_SIZE
            } else {
                self.in_progress.capacity()
            };
            if self.in_progress.len() + len > block_limit {
                if !self.in_progress.is_empty() {
                    let flushed = mem::take(&mut self.in_progress);
                    self.completed.push(Buffer::from_vec(flushed));
                }
                let capacity = if self.block_size == 0 {
                    BYTE_VIEW_MAX_BLOCK_SIZE
                } else {
                    self.block_size = (self.block_size * 2).min(BYTE_VIEW_MAX_BLOCK_SIZE);
                    len.max(self.block_size)
                };
                self.in_progress = Vec::with_capacity(capacity);
            }

            let buffer_index = self.completed.len() as u32;
            let offset = self.in_progress.len() as u32;
            self.in_progress.extend_from_slice(value);

            make_view(value, buffer_index, offset)
        };

        self.views.push(view);
        view
    }

    /// Total number of entries (including null, if present)
    pub fn len(&self) -> usize {
        self.non_null_len() + self.null.map(|_| 1).unwrap_or(0)
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.items == 0 && self.null.is_none()
    }

    /// Number of non null entries
    pub fn non_null_len(&self) -> usize {
        self.items
    }

    /// Return the total size, in bytes, of memory used to store the data in
    /// this set, not including `self` or input-array buffers.
    pub fn size(&self) -> usize {
        // All fields below own their allocations. Count retained capacity rather
        // than used length because this value drives memory accounting.
        let views_size = self.views.allocated_size()
            + self.hashes.allocated_size()
            + self.payloads.allocated_size();
        let in_progress_size = self.in_progress.allocated_size();
        let completed_size = self.completed.allocated_size()
            + self.completed.iter().map(Buffer::capacity).sum::<usize>();

        self.slots.allocated_size()
            + views_size
            + in_progress_size
            + completed_size
            + self.hashes_buffer.allocated_size()
    }
}

impl<V> Debug for ArrowBytesViewMap<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowBytesMap")
            .field("slots", &self.slots.len())
            .field("items", &self.items)
            .field("views_len", &self.views.len())
            .field("completed_buffers", &self.completed.len())
            .field("random_state", &self.random_state)
            .field("hashes_buffer", &self.hashes_buffer)
            .finish()
    }
}

/// An empty slot. An occupied slot holds `index + 1` in its low bits, so it is
/// never zero.
const EMPTY: u64 = 0;

/// The low 48 bits of a slot: one more than the index of its key.
const INDEX_MASK: u64 = (1 << 48) - 1;

/// Slots a prefetching stage reads past a key's home slot while looking for
/// its candidate: one cache line. Probes are short at three quarters load, and
/// a stage that gives up only leaves that row unprefetched.
const HINT_SLOTS: usize = 8;

/// The top 16 bits of `hash`, in place: the part of a slot that tells keys
/// apart before their views are read.
#[inline(always)]
fn tag(hash: u64) -> u64 {
    hash & !INDEX_MASK
}

/// The slot for the key at `index` with `hash`.
#[inline(always)]
fn slot_of(hash: u64, index: usize) -> u64 {
    tag(hash) | (index as u64 + 1)
}

/// The index an occupied slot names.
#[inline(always)]
fn index_of(slot: u64) -> usize {
    ((slot & INDEX_MASK) - 1) as usize
}

/// The most keys `slots` slots hold: three quarters, so every probe meets an
/// empty slot.
#[inline(always)]
fn max_items(slots: usize) -> usize {
    slots - slots / 4
}

/// The smallest table, a power of two of at least 8 slots, that holds
/// `entries` keys; none for no keys.
fn slots_for(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }
    entries
        .saturating_add(entries / 3)
        .saturating_add(1)
        .max(8)
        .checked_next_power_of_two()
        .unwrap_or(1 << (usize::BITS - 1))
}

#[cfg(test)]
mod tests {
    use arrow::array::{GenericByteViewArray, StringViewArray};
    use datafusion_common::HashMap;
    use std::mem::size_of;

    use super::*;

    // asserts that the set contains the expected strings, in the same order
    fn assert_set(set: ArrowBytesViewSet, expected: &[Option<&str>]) {
        let strings = set.into_state();
        let strings = strings.as_string_view();
        let state = strings.into_iter().collect::<Vec<_>>();
        assert_eq!(state, expected);
    }

    #[test]
    fn string_view_set_empty() {
        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let array: ArrayRef = Arc::new(StringViewArray::new_null(0));
        set.insert(&array);
        assert_eq!(set.len(), 0);
        assert_eq!(set.non_null_len(), 0);
        assert_set(set, &[]);
    }

    #[test]
    fn string_view_set_one_null() {
        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let array: ArrayRef = Arc::new(StringViewArray::new_null(1));
        set.insert(&array);
        assert_eq!(set.len(), 1);
        assert_eq!(set.non_null_len(), 0);
        assert_set(set, &[None]);
    }

    #[test]
    fn string_view_set_many_null() {
        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let array: ArrayRef = Arc::new(StringViewArray::new_null(11));
        set.insert(&array);
        assert_eq!(set.len(), 1);
        assert_eq!(set.non_null_len(), 0);
        assert_set(set, &[None]);
    }

    #[test]
    fn test_string_view_set_basic() {
        // basic test for mixed small and large string values
        let values = GenericByteViewArray::from(vec![
            Some("a"),
            Some("b"),
            Some("CXCCCCCCCCAABB"), // 14 bytes
            Some(""),
            Some("cbcxx"), // 5 bytes
            None,
            Some("AAAAAAAA"),     // 8 bytes
            Some("BBBBBQBBBAAA"), // 12 bytes
            Some("a"),
            Some("cbcxx"),
            Some("b"),
            Some("cbcxx"),
            Some(""),
            None,
            Some("BBBBBQBBBAAA"),
            Some("BBBBBQBBBAAA"),
            Some("AAAAAAAA"),
            Some("CXCCCCCCCCAABB"),
        ]);

        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let array: ArrayRef = Arc::new(values);
        set.insert(&array);
        // values mut appear be in the order they were inserted
        assert_set(
            set,
            &[
                Some("a"),
                Some("b"),
                Some("CXCCCCCCCCAABB"),
                Some(""),
                Some("cbcxx"),
                None,
                Some("AAAAAAAA"),
                Some("BBBBBQBBBAAA"),
            ],
        );
    }

    #[test]
    fn test_string_set_non_utf8() {
        // basic test for mixed small and large string values
        let values = GenericByteViewArray::from(vec![
            Some("a"),
            Some("✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥"),
            Some("🔥"),
            Some("✨✨✨"),
            Some("foobarbaz"),
            Some("🔥"),
            Some("✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥"),
        ]);

        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let array: ArrayRef = Arc::new(values);
        set.insert(&array);
        // strings mut appear be in the order they were inserted
        assert_set(
            set,
            &[
                Some("a"),
                Some("✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥✨🔥"),
                Some("🔥"),
                Some("✨✨✨"),
                Some("foobarbaz"),
            ],
        );
    }

    // Test use of binary output type
    #[test]
    fn test_binary_set() {
        let v: Vec<Option<&[u8]>> = vec![
            Some(b"a"),
            Some(b"CXCCCCCCCCCCCCC"),
            None,
            Some(b"CXCCCCCCCCCCCCC"),
        ];
        let values: ArrayRef = Arc::new(BinaryViewArray::from(v));

        let expected: Vec<Option<&[u8]>> =
            vec![Some(b"a"), Some(b"CXCCCCCCCCCCCCC"), None];
        let expected: ArrayRef = Arc::new(GenericByteViewArray::from(expected));

        let mut set = ArrowBytesViewSet::new(OutputType::BinaryView);
        set.insert(&values);
        assert_eq!(&set.into_state(), &expected);
    }

    // inserting strings into the set does not increase reported memory
    #[test]
    fn test_string_set_memory_usage() {
        let strings1 = StringViewArray::from(vec![
            Some("a"),
            Some("b"),
            Some("CXCCCCCCCCCCC"), // 13 bytes
            Some("AAAAAAAA"),      // 8 bytes
            Some("BBBBBQBBB"),     // 9 bytes
        ]);
        let total_strings1_len = strings1
            .iter()
            .map(|s| s.map(|s| s.len()).unwrap_or(0))
            .sum::<usize>();
        let values1: ArrayRef = Arc::new(StringViewArray::from(strings1));

        // Much larger strings in strings2
        let strings2 = StringViewArray::from(vec![
            "FOO".repeat(1000),
            "BAR larger than 12 bytes.".repeat(100_000),
            "more unique.".repeat(1000),
            "more unique2.".repeat(1000),
            "FOO".repeat(3000),
        ]);
        let total_strings2_len = strings2
            .iter()
            .map(|s| s.map(|s| s.len()).unwrap_or(0))
            .sum::<usize>();
        let values2: ArrayRef = Arc::new(StringViewArray::from(strings2));

        let mut set = ArrowBytesViewSet::new(OutputType::Utf8View);
        let size_empty = set.size();

        set.insert(&values1);
        let size_after_values1 = set.size();
        assert!(size_empty < size_after_values1);
        assert!(
            size_after_values1 > total_strings1_len,
            "expect {size_after_values1} to be more than {total_strings1_len}"
        );
        assert!(size_after_values1 < total_strings1_len + total_strings2_len);

        // inserting the same strings should not affect the size
        set.insert(&values1);
        assert_eq!(set.size(), size_after_values1);
        assert_eq!(set.len(), 5);

        // inserting the large strings should increase the reported size
        set.insert(&values2);
        let size_after_values2 = set.size();
        assert!(size_after_values2 > size_after_values1);

        assert_eq!(set.len(), 10);
    }

    #[test]
    fn map_new_does_not_allocate() {
        let map = ArrowBytesViewMap::<()>::new(OutputType::Utf8View);

        assert_eq!(map.slots.capacity(), 0);
        assert_eq!(map.size(), 0);
    }

    #[test]
    fn test_size_counts_initial_hash_table_capacity() {
        let map = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );

        // Room for the capacity at no more than three quarters load, in a
        // power of two of slots.
        let slots = map.slots.len();
        assert!(slots.is_power_of_two());
        assert!(slots * 3 >= INITIAL_MAP_CAPACITY * 4);
        assert!(slots < INITIAL_MAP_CAPACITY * 4);
        assert_eq!(map.size(), slots * size_of::<u64>());
    }

    #[test]
    fn take_preserves_the_capacity_the_map_was_built_with() {
        let mut preallocated = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        let capacity = preallocated.slots.len();
        preallocated.take();
        assert_eq!(preallocated.slots.len(), capacity);

        let mut lazy = ArrowBytesViewMap::<()>::new(OutputType::Utf8View);
        lazy.take();
        assert_eq!(lazy.slots.len(), 0);
    }

    #[test]
    fn clear_and_release_frees_the_preallocation_that_take_keeps() {
        let mut map = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..1_000).map(|i| format!("distinct value number {i}")),
        ));
        map.insert_if_new(&values, |_| (), |_| ());

        assert!(map.slots.len() * 3 >= 1_000 * 4);

        // `take` deliberately keeps the map warm, so it does not release the
        // configured capacity.
        map.take();
        let taken_size = map.size();
        assert!(
            taken_size >= INITIAL_MAP_CAPACITY * size_of::<u64>(),
            "expected take to retain the warm up allocation, got {taken_size}"
        );

        map.clear_and_release();
        assert_eq!(map.slots.capacity(), 0);
        let released_size = map.size();
        assert!(
            released_size < 128,
            "expected the released map to report approximately zero bytes, got {released_size}"
        );

        // The configured capacity survives, so the map warms back up when it is
        // emitted from again.
        map.take();
        assert!(map.slots.len() * 3 >= INITIAL_MAP_CAPACITY * 4);
    }

    #[test]
    fn test_size_counts_retained_buffer_capacities() {
        let first = "a".repeat(BYTE_VIEW_MAX_BLOCK_SIZE / 2 + 1);
        let second = "b".repeat(BYTE_VIEW_MAX_BLOCK_SIZE / 2 + 1);
        let third = "c".repeat(BYTE_VIEW_MAX_BLOCK_SIZE / 2 + 1);
        let values: ArrayRef = Arc::new(StringViewArray::from(vec![
            first.as_str(),
            second.as_str(),
            third.as_str(),
        ]));

        let mut map = ArrowBytesViewMap::new(OutputType::Utf8View);
        map.insert_if_new(&values, |_| (), |_| {});

        // Make unused vector capacity explicit; the completed buffers were created
        // by the map's flush path.
        map.views.shrink_to_fit();
        map.views.reserve_exact(1);
        map.completed.shrink_to_fit();
        map.completed.reserve_exact(1);
        let completed = &mut map.completed[0];
        let mut padded = Vec::with_capacity(completed.len() + 16);
        padded.extend_from_slice(completed);
        *completed = Buffer::from_vec(padded);

        // The map owns these allocations; `values` and its Arrow buffers remain external.
        assert!(map.views.capacity() > map.views.len());
        assert!(map.completed.capacity() > map.completed.len());
        assert!(
            map.completed
                .iter()
                .any(|buffer| buffer.capacity() > buffer.len())
        );

        let expected_size = map.slots.allocated_size()
            + map.views.allocated_size()
            + map.hashes.allocated_size()
            + map.payloads.allocated_size()
            + map.in_progress.allocated_size()
            + map.completed.allocated_size()
            + map.completed.iter().map(Buffer::capacity).sum::<usize>()
            + map.hashes_buffer.allocated_size();
        assert_eq!(map.size(), expected_size);

        // Verify the retained-capacity delta independently from the production formula.
        let legacy_size = map.slots.len() * size_of::<u64>()
            + map.hashes.capacity() * size_of::<u64>()
            + map.views.len() * size_of::<u128>()
            + map.in_progress.capacity()
            + map.completed.iter().map(Buffer::len).sum::<usize>()
            + map.hashes_buffer.allocated_size();
        let retained_capacity_delta = (map.views.capacity() - map.views.len())
            * size_of::<u128>()
            + map.completed.capacity() * size_of::<Buffer>()
            + map
                .completed
                .iter()
                .map(|buffer| buffer.capacity() - buffer.len())
                .sum::<usize>();
        assert_eq!(map.size() - legacy_size, retained_capacity_delta);

        let size_after_insert = map.size();
        map.insert_if_new(&values, |_| (), |_| {});
        assert_eq!(map.size(), size_after_insert);
    }

    #[derive(Debug, PartialEq, Eq, Default, Clone, Copy)]
    struct TestPayload {
        // store the string value to check against input
        index: usize, // store the index of the string (each new string gets the next sequential input)
    }

    /// Wraps an [`ArrowBytesViewMap`], validating its invariants
    struct TestMap {
        map: ArrowBytesViewMap<TestPayload>,
        // stores distinct strings seen, in order
        strings: Vec<Option<String>>,
        // map strings to index in strings
        indexes: HashMap<Option<String>, usize>,
    }

    impl TestMap {
        /// creates a map with TestPayloads for the given strings and then
        /// validates the payloads
        fn new() -> Self {
            Self {
                map: ArrowBytesViewMap::new(OutputType::Utf8View),
                strings: vec![],
                indexes: HashMap::new(),
            }
        }

        /// Inserts strings into the map
        fn insert(&mut self, strings: &[Option<&str>]) {
            let string_array = StringViewArray::from(strings.to_vec());
            let arr: ArrayRef = Arc::new(string_array);

            let mut next_index = self.indexes.len();
            let mut actual_new_strings = vec![];
            let mut actual_seen_indexes = vec![];
            // update self with new values, keeping track of newly added values
            for str in strings {
                let str = str.map(|s| s.to_string());
                let index = self.indexes.get(&str).copied().unwrap_or_else(|| {
                    actual_new_strings.push(str.clone());
                    let index = self.strings.len();
                    self.strings.push(str.clone());
                    self.indexes.insert(str, index);
                    index
                });
                actual_seen_indexes.push(index);
            }

            // insert the values into the map, recording what we did
            let mut seen_new_strings = vec![];
            let mut seen_indexes = vec![];
            self.map.insert_if_new(
                &arr,
                |s| {
                    let value = s
                        .map(|s| String::from_utf8(s.to_vec()).expect("Non utf8 string"));
                    let index = next_index;
                    next_index += 1;
                    seen_new_strings.push(value);
                    TestPayload { index }
                },
                |payload| {
                    seen_indexes.push(payload.index);
                },
            );

            assert_eq!(actual_seen_indexes, seen_indexes);
            assert_eq!(actual_new_strings, seen_new_strings);
        }

        /// Call `self.map.into_array()` validating that the strings are in the same
        /// order as they were inserted
        fn into_array(self) -> ArrayRef {
            let Self {
                map,
                strings,
                indexes: _,
            } = self;

            let arr = map.into_state();
            let expected: ArrayRef = Arc::new(StringViewArray::from(strings));
            assert_eq!(&arr, &expected);
            arr
        }
    }

    #[test]
    fn test_map() {
        let input = vec![
            // Note mix of short/long strings
            Some("A"),
            Some("bcdefghijklmnop1234567"),
            Some("X"),
            Some("Y"),
            None,
            Some("qrstuvqxyzhjwya"),
            Some("✨🔥"),
            Some("🔥"),
            Some("🔥🔥🔥🔥🔥🔥"),
        ];

        let mut test_map = TestMap::new();
        test_map.insert(&input);
        test_map.insert(&input); // put it in twice
        let expected_output: ArrayRef = Arc::new(StringViewArray::from(input));
        assert_eq!(&test_map.into_array(), &expected_output);
    }

    /// A long value whose hash collides with an inline value sharing its
    /// 4-byte prefix must not read the inline bytes as a buffer reference.
    #[cfg(feature = "force_hash_collisions")]
    #[test]
    fn test_long_value_colliding_with_inline_value() {
        let mut test_map = TestMap::new();
        test_map.insert(&[Some("bcdeXXXXYYYY")]);
        test_map.insert(&[Some("bcdefghijklmnop1234567")]);
        let expected_output: ArrayRef = Arc::new(StringViewArray::from(vec![
            "bcdeXXXXYYYY",
            "bcdefghijklmnop1234567",
        ]));
        assert_eq!(&test_map.into_array(), &expected_output);
    }

    /// Deterministic keys for the staged probe tests: NULLs, inline keys,
    /// keys of exactly 12 bytes and long keys sharing a long prefix, each
    /// repeated about twice.
    fn mixed_keys(rows: usize, seed: u64) -> ArrayRef {
        let mut state = seed;
        let keys = (0..rows).map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let pick = state >> 33;
            let key = pick % (rows as u64 / 2 + 1);
            match pick % 7 {
                0 => None,
                1 | 2 => Some(format!("{key}")),
                3 => Some(format!("{key:012}")),
                _ => Some(format!("https://example.com/a/long/shared/prefix/{key}")),
            }
        });
        Arc::new(StringViewArray::from_iter(keys))
    }

    /// Each row's payload, the values payloads were made for, the state and
    /// the reported size after inserting `batches` with the probe forced to
    /// `staged`.
    type Run = (Vec<usize>, Vec<Option<Vec<u8>>>, ArrayRef, usize);

    fn run(batches: &[ArrayRef], staged: bool) -> Run {
        let mut map = ArrowBytesViewMap::<usize>::new(OutputType::Utf8View);
        let mut seen = vec![];
        let mut made = vec![];
        for batch in batches {
            let mut next = map.len();
            map.insert(
                batch,
                Some(staged),
                |value| {
                    made.push(value.map(<[u8]>::to_vec));
                    next += 1;
                    next - 1
                },
                |payload| seen.push(payload),
            );
        }
        let size = map.size();
        (seen, made, map.into_state(), size)
    }

    #[test]
    fn staged_probe_matches_plain_probe() {
        // A batch shorter than the look-ahead, then batches that grow the
        // table many times from empty while they are probed.
        let batches = [
            mixed_keys(5, 1),
            mixed_keys(2048, 2),
            mixed_keys(2048, 3),
            mixed_keys(3 * DISTANCE + 1, 4),
        ];
        let plain = run(&batches, false);
        let staged = run(&batches, true);
        assert_eq!(plain.0, staged.0);
        assert_eq!(plain.1, staged.1);
        assert_eq!(&plain.2, &staged.2);
        assert_eq!(plain.3, staged.3);

        // And both are right: a row's payload is the number of distinct keys
        // seen before its key first appeared, and the state lists the
        // distinct keys in that order.
        let mut first = HashMap::<Option<String>, usize>::new();
        let mut order = vec![];
        let expected = batches
            .iter()
            .flat_map(|batch| batch.as_string_view().iter().collect::<Vec<_>>())
            .map(|key| {
                let key = key.map(str::to_string);
                let next = first.len();
                *first.entry(key.clone()).or_insert_with(|| {
                    order.push(key);
                    next
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(staged.0, expected);
        let state: ArrayRef = Arc::new(StringViewArray::from(order));
        assert_eq!(&staged.2, &state);
    }

    #[test]
    fn take_with_capacity_empties_and_keeps_the_table() {
        let values = mixed_keys(1000, 5);
        let mut map = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        map.insert(&values, Some(true), |_| (), |_| ());
        let len = map.len();
        let slots = map.slots.len();

        let state = map.take_with_capacity(64);
        assert_eq!(state.len(), len);
        assert!(map.is_empty());
        assert_eq!(map.slots.len(), slots);
        assert!(map.slots.iter().all(|slot| *slot == EMPTY));

        // Every key is new again, and the emitted keys are the same keys.
        let mut made = 0;
        map.insert(&values, Some(true), |_| made += 1, |_| ());
        assert_eq!(made, len);
        assert_eq!(&map.take_with_capacity(0), &state);
    }

    /// Quadratic when every hash collides, so skipped under that feature, as
    /// the count distinct tests are.
    #[cfg(not(feature = "force_hash_collisions"))]
    #[test]
    fn growth_keeps_every_key_findable() {
        let mut map = ArrowBytesViewMap::<usize>::new(OutputType::BinaryView);
        let keys = (0..20_000u32)
            .map(|key| format!("{key:0>width$}", width = 1 + key as usize % 40))
            .collect::<Vec<_>>();
        let values: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            keys.iter().map(String::as_bytes),
        ));
        let mut next = 0;
        map.insert_if_new(
            &values,
            |_| {
                next += 1;
                next - 1
            },
            |_| (),
        );
        assert_eq!(map.non_null_len(), keys.len());
        assert!(map.items <= max_items(map.slots.len()));

        let mut seen = vec![];
        map.insert(&values, Some(true), |_| 0, |payload| seen.push(payload));
        assert_eq!(seen, (0..keys.len()).collect::<Vec<_>>());
    }

    #[test]
    fn staged_probe_pays_only_beyond_l2() {
        let map = ArrowBytesViewMap::<usize>::new(OutputType::Utf8View);
        assert!(!map.stage_pays(8192));

        let big = ArrowBytesViewMap::<usize>::with_capacity(
            OutputType::Utf8View,
            l2_cache_bytes() / size_of::<u64>(),
        );
        assert!(big.stage_pays(8192));
        assert!(!big.stage_pays(3 * DISTANCE));
    }

    #[test]
    fn slots_for_keeps_load_under_three_quarters() {
        assert_eq!(slots_for(0), 0);
        for entries in (1..5000).chain([1 << 20, (1 << 20) + 1]) {
            let slots = slots_for(entries);
            assert!(slots.is_power_of_two());
            assert!(max_items(slots) >= entries);
            assert!(max_items(slots) < slots);
        }
    }
}
