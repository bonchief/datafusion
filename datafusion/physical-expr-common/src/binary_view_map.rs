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
use hashbrown::hash_table::HashTable;
use std::fmt::Debug;
use std::mem::{self, MaybeUninit};
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
/// # Lookup tables
///
/// A map starts in hashbrown's table, probed one row at a time. A map whose
/// table outgrows one core's share of L2 ([`l2_cache_bytes`]) moves,
/// once, at the start of the next batch or when a reservation alone outgrows
/// it, to [`Table`]: a SwissTable-style
/// table of our own whose probe can prefetch the memory a later row's probe
/// reads, which hashbrown cannot since it exposes no addresses. It stays there
/// until [`Self::take`] or [`Self::clear_and_release`];
/// [`Self::take_with_capacity`] keeps whichever table the map is in.
///
/// Key bytes do not count towards the gate: a probe reads one line of them
/// only for an entry whose hash already matched, and counting them would move
/// a partial aggregate's 8,192 long keys, whose table stays cache-hot between
/// flushes, to the table that only pays beyond the cache.
///
/// # Staged probe
///
/// Probing one row at a time takes up to three dependent cache misses per row
/// once the table is larger than the cache: the control bytes, the entry and
/// the stored key bytes. In [`Table`], a batch longer than `2 * DISTANCE` rows
/// is probed in stages: while row `i` is looked up, the control bytes and the
/// entry at the home bucket of row `i + 2D` are prefetched, and for row
/// `i + D` the first entry its tag selects is prefetched and, when its hash,
/// length and prefix already match, so are the stored key bytes. The stages
/// only prefetch. Every lookup, insert and payload call still happens for row
/// `i` in row order, in either table, so results, payloads and group order do
/// not depend on which table holds the keys or how a batch was probed.
pub struct ArrowBytesViewMap<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// Should the output be StringView or BinaryView?
    output_type: OutputType,
    /// Underlying hash set for each distinct value, while the map is below
    /// the gate; empty, holding no allocation, once `table` is set
    map: HashTable<Entry<V>>,
    /// The prefetchable table, once the map has outgrown L2
    table: Option<Box<Table<V>>>,
    /// Hash table capacity to re-create the map with in [`Self::take`], so a
    /// map built with [`Self::with_capacity`] keeps its pre-allocation when it
    /// is emptied and reused
    initial_map_capacity: usize,

    /// Views for all stored values (in insertion order)
    views: Vec<u128>,
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
            map: HashTable::with_capacity(map_capacity),
            table: None,
            initial_map_capacity: map_capacity,
            views: Vec::new(),
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
        // A reservation that alone outgrows the gate goes to the prefetched
        // table now, rather than sizing hashbrown's table only to copy it.
        if self.table.is_none()
            && buckets_for(capacity) * (size_of::<Entry<V>>() + 1) > l2_cache_bytes()
        {
            self.table = Some(Box::new(Table::from_hashbrown(&mem::take(&mut self.map))));
        }
        match &mut self.table {
            Some(table) => table.reserve(capacity),
            None => self
                .map
                .reserve(capacity.saturating_sub(self.map.len()), |entry| entry.hash),
        }
        self.views
            .reserve_exact(capacity.saturating_sub(self.views.len()));
        self.hashes_buffer
            .reserve_exact(capacity.saturating_sub(self.hashes_buffer.len()));
    }

    /// Emits keys, retains the hash table and scratch buffer, and reserves fixed
    /// view capacity. Payload blocks restart their exponential allocation growth.
    pub fn take_with_capacity(&mut self, capacity: usize) -> ArrayRef {
        let mut outgoing = Self::new(self.output_type);
        mem::swap(self, &mut outgoing);
        mem::swap(&mut self.map, &mut outgoing.map);
        self.map.clear();
        mem::swap(&mut self.table, &mut outgoing.table);
        if let Some(table) = &mut self.table {
            table.clear();
        }
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

    /// [`Self::insert_if_new`]. `staged` forces the table the batch is probed
    /// with: `Some(true)` moves it to the prefetched table and probes in
    /// stages, `Some(false)` probes one row at a time in whichever table it
    /// is in, and `None` decides as the type docs say.
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
        // A map in hashbrown's table below the gate runs the loop below,
        // which is the one this map has always run.
        let moves = self.table.is_some()
            || staged.unwrap_or_else(|| self.map.allocation_size() > l2_cache_bytes());
        if moves {
            return self.insert_prefetched::<MP, OP, B>(
                values,
                staged,
                make_payload_fn,
                observe_payload_fn,
            );
        }

        // step 1: compute hashes
        let batch_hashes = &mut self.hashes_buffer;
        batch_hashes.clear();
        batch_hashes.resize(values.len(), 0);
        create_hashes([values], &self.random_state, batch_hashes)
            // hash is supported for all types and create_hashes only
            // returns errors for unsupported types
            .unwrap();

        // step 2: insert each value into the set, if not already present
        let values = values.as_byte_view::<B>();

        // Get raw views buffer for direct comparison
        let input_views = values.views();

        // Ensure lengths are equivalent
        assert_eq!(values.len(), self.hashes_buffer.len());

        for i in 0..values.len() {
            let view_u128 = input_views[i];
            let hash = self.hashes_buffer[i];

            // handle null value via validity bitmap check
            if values.is_null(i) {
                let payload = if let Some(&(payload, _offset)) = self.null.as_ref() {
                    payload
                } else {
                    let payload = make_payload_fn(None);
                    let null_index = self.views.len();
                    self.views.push(0);
                    self.null = Some((payload, null_index));
                    payload
                };
                observe_payload_fn(payload);
                continue;
            }

            // Extract length from the view (first 4 bytes of u128 in little-endian)
            let len = view_u128 as u32;

            // Check if value already exists
            let maybe_payload = {
                // Borrow completed and in_progress for comparison
                let completed = &self.completed;
                let in_progress = &self.in_progress;

                self.map
                    .find(hash, |header| {
                        if header.hash != hash || header.view as u32 != len {
                            return false;
                        }

                        // Fast path: inline strings can be compared directly
                        if len <= 12 {
                            return header.view == view_u128;
                        }

                        // For larger strings: first compare the 4-byte prefix
                        let stored_prefix = (header.view >> 32) as u32;
                        let input_prefix = (view_u128 >> 32) as u32;
                        if stored_prefix != input_prefix {
                            return false;
                        }

                        // Prefix matched - compare full bytes
                        let byte_view = ByteView::from(header.view);
                        let stored_len = byte_view.length as usize;
                        let buffer_index = byte_view.buffer_index as usize;
                        let offset = byte_view.offset as usize;

                        let stored_value = if buffer_index < completed.len() {
                            &completed[buffer_index].as_slice()
                                [offset..offset + stored_len]
                        } else {
                            &in_progress[offset..offset + stored_len]
                        };
                        let input_value: &[u8] = values.value(i).as_ref();
                        stored_value == input_value
                    })
                    .map(|entry| entry.payload)
            };

            let payload = if let Some(payload) = maybe_payload {
                payload
            } else {
                // no existing value, make a new one
                let (new_view, payload) = if len <= 12 {
                    // Inline path: bytes are already packed in view_u128.
                    // The inline ByteView format is [len:u32 LE][data:12 bytes zero-padded],
                    // so extracting bytes from the u128 avoids a round-trip through
                    // values.value(i) (which reads the views buffer and returns the same slice).
                    let view_bytes = view_u128.to_le_bytes();
                    let value = &view_bytes[4..4 + len as usize];
                    let payload = make_payload_fn(Some(value));
                    // For inline strings, the stored view is identical to the input view:
                    // make_view(value, 0, 0) produces the same u128 as view_u128.
                    //
                    // SAFETY: view_u128 was a valid view, and the enclosing `len <= 12`
                    // ensures it is inline
                    let new_view = unsafe { self.append_inline_view(view_u128) };
                    (new_view, payload)
                } else {
                    let value: &[u8] = values.value(i).as_ref();
                    let payload = make_payload_fn(Some(value));
                    let new_view = self.append_value(value);
                    (new_view, payload)
                };

                let new_header = Entry {
                    view: new_view,
                    hash,
                    payload,
                };

                self.map.insert_unique(hash, new_header, |h| h.hash);
                payload
            };
            observe_payload_fn(payload);
        }
    }

    /// [`Self::insert_if_new_inner`] in the prefetched table, moving the map
    /// to it first if it is not there yet. Kept out of line so the loop over
    /// hashbrown's table compiles as it did before this table existed.
    #[inline(never)]
    fn insert_prefetched<MP, OP, B>(
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
        let mut table = match self.table.take() {
            Some(table) => table,
            None => Box::new(Table::from_hashbrown(&mem::take(&mut self.map))),
        };

        let mut hashes = mem::take(&mut self.hashes_buffer);
        hashes.clear();
        hashes.resize(values.len(), 0);
        create_hashes([values], &self.random_state, &mut hashes)
            // hash is supported for all types and create_hashes only
            // returns errors for unsupported types
            .unwrap();
        let values = values.as_byte_view::<B>();
        assert_eq!(values.len(), hashes.len());
        let views = values.views();

        if staged.unwrap_or_else(|| values.len() > 2 * DISTANCE) {
            for (i, &hash) in hashes.iter().enumerate() {
                if let Some(&ahead) = hashes.get(i + 2 * DISTANCE) {
                    table.prefetch_home(ahead);
                }
                if let Some(&ahead) = hashes.get(i + DISTANCE) {
                    self.prefetch_key(&table, ahead, views[i + DISTANCE]);
                }
                self.insert_row(
                    &mut table,
                    values,
                    i,
                    views[i],
                    hash,
                    &mut make_payload_fn,
                    &mut observe_payload_fn,
                );
            }
        } else {
            for (i, &hash) in hashes.iter().enumerate() {
                self.insert_row(
                    &mut table,
                    values,
                    i,
                    views[i],
                    hash,
                    &mut make_payload_fn,
                    &mut observe_payload_fn,
                );
            }
        }
        self.table = Some(table);
        self.hashes_buffer = hashes;
    }

    /// Looks up row `i` of `values`, whose view is `view`, in the prefetched
    /// table and inserts it if it is new: the one step both the plain and the
    /// staged probe of that table take for every row, in row order.
    #[expect(clippy::too_many_arguments)]
    #[inline(always)]
    fn insert_row<MP, OP, B>(
        &mut self,
        table: &mut Table<V>,
        values: &GenericByteViewArray<B>,
        i: usize,
        view: u128,
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
                self.null = Some((payload, null_index));
                payload
            };
            observe_payload_fn(payload);
            return;
        }

        // Extract length from the view (first 4 bytes of u128 in little-endian)
        let len = view as u32;
        let value: &[u8] = if len <= 12 {
            &[]
        } else {
            values.value(i).as_ref()
        };

        let found = table.find(hash, |entry| {
            entry.hash == hash && self.matches(entry.view, view, value)
        });
        let payload = match found {
            Ok(payload) => payload,
            Err(vacant) => {
                let bucket = table.room(hash, vacant);
                // no existing value, make a new one
                let (new_view, payload) = if len <= 12 {
                    // See `insert_plain`: the stored view of an inline value
                    // is the input view.
                    let view_bytes = view.to_le_bytes();
                    let payload = make_payload_fn(Some(&view_bytes[4..4 + len as usize]));
                    // SAFETY: view was a valid view, and the enclosing `len <= 12`
                    // ensures it is inline
                    (unsafe { self.append_inline_view(view) }, payload)
                } else {
                    let payload = make_payload_fn(Some(value));
                    (self.append_value(value), payload)
                };
                table.put(
                    bucket,
                    Entry {
                        view: new_view,
                        hash,
                        payload,
                    },
                );
                payload
            }
        };
        observe_payload_fn(payload);
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

    /// Staged probe, second stage: prefetch the entry `hash`'s tag selects in
    /// its home group and, for an input key longer than 12 bytes whose hash,
    /// length and prefix that entry matches, the first and the last line of
    /// the stored bytes. It is only a guess; the lookup of the row decides.
    #[inline(always)]
    fn prefetch_key(&self, table: &Table<V>, hash: u64, view: u128) {
        let Some(entry) = table.candidate(hash) else {
            return;
        };
        let len = view as u32;
        if len <= 12
            || entry.hash != hash
            || entry.view as u32 != len
            || (entry.view >> 32) as u32 != (view >> 32) as u32
        {
            return;
        }
        let byte_view = ByteView::from(entry.view);
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
            let buffer_index = byte_view.buffer_index as usize;
            let offset = byte_view.offset as usize;
            if buffer_index < self.completed.len() {
                &self.completed[buffer_index][offset..offset + length]
            } else {
                &self.in_progress[offset..offset + length]
            }
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

    /// Bytes the hash tables hold.
    ///
    /// `HashTable::allocation_size` reports the whole hashbrown allocation,
    /// which is the entry array plus the control bytes plus the trailing
    /// group, so it is larger than `capacity() * size_of::<Entry<V>>()`. It
    /// is a constant time layout calculation, not a walk of the table.
    fn table_size(&self) -> usize {
        self.map.allocation_size()
            + self.table.as_ref().map_or(0, |table| {
                size_of::<Table<V>>()
                    + table.ctrl.allocated_size()
                    + table.entries.len() * size_of::<Entry<V>>()
            })
    }

    /// Total number of entries (including null, if present)
    pub fn len(&self) -> usize {
        self.non_null_len() + self.null.map(|_| 1).unwrap_or(0)
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.non_null_len() == 0 && self.null.is_none()
    }

    /// Number of non null entries
    pub fn non_null_len(&self) -> usize {
        match &self.table {
            Some(table) => table.items,
            None => self.map.len(),
        }
    }

    /// Return the total size, in bytes, of memory used to store the data in
    /// this set, not including `self` or input-array buffers.
    pub fn size(&self) -> usize {
        // All fields below own their allocations. Count retained capacity rather
        // than used length because this value drives memory accounting.
        let map_size = self.table_size();
        let views_size = self.views.allocated_size();
        let in_progress_size = self.in_progress.allocated_size();
        let completed_size = self.completed.allocated_size()
            + self.completed.iter().map(Buffer::capacity).sum::<usize>();

        map_size
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
            .field("map", &"<map>")
            .field("map_allocation_size", &self.table_size())
            .field("prefetched", &self.table.is_some())
            .field("views_len", &self.views.len())
            .field("completed_buffers", &self.completed.len())
            .field("random_state", &self.random_state)
            .field("hashes_buffer", &self.hashes_buffer)
            .finish()
    }
}

/// Entry in the hash table -- see [`ArrowBytesViewMap`] for more details
///
/// Stores the view pointing to our internal buffers, eliminating the need
/// for a separate builder index. For inline strings (<=12 bytes), the view
/// contains the entire value. For out-of-line strings, the view contains
/// buffer_index and offset pointing directly to our storage.
#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
struct Entry<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// The u128 view pointing to our internal buffers. For inline strings,
    /// this contains the complete value. For larger strings, this contains
    /// the buffer_index/offset into our completed/in_progress buffers.
    view: u128,

    hash: u64,

    /// value stored by the entry
    payload: V,
}

/// A SwissTable-style hash table whose memory a probe can prefetch.
///
/// It has a power of two of buckets, at least [`GROUP`]. `entries` holds one [`Entry`]
/// per bucket, and `ctrl` one control byte per bucket followed by a copy of
/// the first [`GROUP`] bytes, so a group of control bytes can be read at any
/// bucket without wrapping. A control byte is [`EMPTY`], or the top 7 bits of
/// the hash of the key whose entry is in that bucket. A key's probe reads
/// groups of [`GROUP`] control bytes from `hash & (buckets - 1)` onwards and
/// stops at the first group with an empty bucket; a new key goes into the
/// first empty bucket its probe meets. At most seven eighths of the buckets
/// are full, nothing is removed but by [`Self::clear`], and `items` counts the
/// full buckets. An entry is written exactly when its control byte becomes
/// full, and read only where the control byte is full, so a new table, like
/// hashbrown's, writes one byte per bucket rather than a whole entry. Growth
/// places each entry again from the hash it holds, without reading or hashing
/// a key.
struct Table<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// One control byte per bucket and a copy of the first [`GROUP`]
    ctrl: Vec<u8>,
    /// One entry per bucket, initialised where `ctrl` is full
    entries: Box<[MaybeUninit<Entry<V>>]>,
    /// Buckets less one
    mask: usize,
    /// Full buckets: the distinct non-null keys
    items: usize,
}

impl<V> Table<V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// An empty table of `buckets` buckets, a power of two of at least
    /// [`GROUP`].
    fn with_buckets(buckets: usize) -> Self {
        Self {
            ctrl: vec![EMPTY; buckets + GROUP],
            entries: Box::new_uninit_slice(buckets),
            mask: buckets - 1,
            items: 0,
        }
    }

    /// The entries of `map` in a table with at least its capacity.
    fn from_hashbrown(map: &HashTable<Entry<V>>) -> Self {
        let mut table =
            Self::with_buckets(buckets_for(map.capacity().max(map.len()).max(1)));
        for entry in map {
            let bucket = table.vacant(entry.hash);
            table.put(bucket, *entry);
        }
        table
    }

    /// The control bytes of the group of buckets starting at `bucket`.
    #[inline(always)]
    fn group(&self, bucket: usize) -> u64 {
        debug_assert!(bucket <= self.mask);
        // SAFETY: `bucket` is at most `mask` and `ctrl` holds `mask + 1 +
        // GROUP` bytes, so the eight bytes from `bucket` are inside it; the
        // read is unaligned.
        let bytes = unsafe {
            self.ctrl
                .as_ptr()
                .add(bucket)
                .cast::<u64>()
                .read_unaligned()
        };
        u64::from_le(bytes)
    }

    /// The entry of the full `bucket`.
    #[inline(always)]
    fn entry(&self, bucket: usize) -> &Entry<V> {
        debug_assert!(bucket <= self.mask && self.ctrl[bucket] & EMPTY == 0);
        // SAFETY: `bucket` is at most `mask`, below the length of `entries`,
        // and callers pass only a bucket whose control byte is full, whose
        // entry `put` or `resize` wrote.
        unsafe { self.entries.get_unchecked(bucket).assume_init_ref() }
    }

    /// Sets the control byte of `bucket`, and its copy after the last bucket.
    #[inline(always)]
    fn set_ctrl(&mut self, bucket: usize, byte: u8) {
        self.ctrl[bucket] = byte;
        if bucket < GROUP {
            self.ctrl[self.mask + 1 + bucket] = byte;
        }
    }

    /// The payload of the entry with `hash` that `eq` accepts; or, if there
    /// is none, the first empty bucket of its probe.
    #[inline(always)]
    fn find(&self, hash: u64, mut eq: impl FnMut(&Entry<V>) -> bool) -> Result<V, usize> {
        let tag = tag(hash);
        let mut pos = hash as usize & self.mask;
        loop {
            let group = self.group(pos);
            let mut hits = match_tag(group, tag);
            while hits != 0 {
                let bucket = (pos + (hits.trailing_zeros() / 8) as usize) & self.mask;
                let entry = self.entry(bucket);
                if eq(entry) {
                    return Ok(entry.payload);
                }
                hits &= hits - 1;
            }
            let empty = group & HIGH_BITS;
            if empty != 0 {
                return Err((pos + (empty.trailing_zeros() / 8) as usize) & self.mask);
            }
            pos = (pos + GROUP) & self.mask;
        }
    }

    /// The bucket a new key with `hash` goes into, given `vacant` from
    /// [`Self::find`]: that bucket, unless the table is full and grows first.
    #[inline(always)]
    fn room(&mut self, hash: u64, vacant: usize) -> usize {
        if self.items < max_items(self.mask + 1) {
            return vacant;
        }
        self.resize(buckets_for(self.items + 1));
        self.vacant(hash)
    }

    /// Fills the empty `bucket` with `entry`.
    #[inline(always)]
    fn put(&mut self, bucket: usize, entry: Entry<V>) {
        self.set_ctrl(bucket, tag(entry.hash));
        self.entries[bucket].write(entry);
        self.items += 1;
    }

    /// The first empty bucket of `hash`'s probe. The table has room.
    fn vacant(&self, hash: u64) -> usize {
        let mut pos = hash as usize & self.mask;
        loop {
            let empty = self.group(pos) & HIGH_BITS;
            if empty != 0 {
                return (pos + (empty.trailing_zeros() / 8) as usize) & self.mask;
            }
            pos = (pos + GROUP) & self.mask;
        }
    }

    /// Replaces the table with `buckets` buckets, a power of two of at least
    /// [`GROUP`] that holds every key, and places each entry again from the
    /// hash it holds.
    fn resize(&mut self, buckets: usize) {
        let old = mem::replace(self, Self::with_buckets(buckets));
        for (bucket, byte) in old.ctrl[..=old.mask].iter().enumerate() {
            if byte & EMPTY == 0 {
                let entry = *old.entry(bucket);
                let bucket = self.vacant(entry.hash);
                self.entries[bucket].write(entry);
                self.set_ctrl(bucket, *byte);
            }
        }
        self.items = old.items;
    }

    /// Grows the table to hold `capacity` keys without growing again.
    fn reserve(&mut self, capacity: usize) {
        let buckets = buckets_for(capacity);
        if buckets > self.mask + 1 {
            self.resize(buckets);
        }
    }

    /// Empties every bucket, keeping the allocation.
    fn clear(&mut self) {
        self.ctrl.fill(EMPTY);
        self.items = 0;
    }

    /// Staged probe, first stage: prefetch the control bytes and the entry
    /// of `hash`'s home bucket.
    #[inline(always)]
    fn prefetch_home(&self, hash: u64) {
        let home = hash as usize & self.mask;
        prefetch(self.ctrl.as_ptr().wrapping_add(home));
        prefetch(self.entries.as_ptr().wrapping_add(home));
    }

    /// Staged probe, second stage: the entry of the first bucket in the home
    /// group of `hash` whose control byte may match, after prefetching it.
    #[inline(always)]
    fn candidate(&self, hash: u64) -> Option<&Entry<V>> {
        let home = hash as usize & self.mask;
        let hits = match_tag(self.group(home), tag(hash));
        if hits == 0 {
            return None;
        }
        let bucket = (home + (hits.trailing_zeros() / 8) as usize) & self.mask;
        prefetch(self.entries.as_ptr().wrapping_add(bucket));
        Some(self.entry(bucket))
    }
}

/// Control bytes read at once: one `u64`.
const GROUP: usize = 8;

/// The control byte of an empty bucket. A full bucket's byte is a 7-bit tag,
/// so only an empty one has its high bit set.
const EMPTY: u8 = 0x80;

/// The low bit of every byte of a group.
const LOW_BITS: u64 = 0x0101_0101_0101_0101;

/// The high bit of every byte of a group: set in a group's bytes exactly
/// where its buckets are empty.
const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

/// The control byte of a full bucket holding a key with `hash`: its top 7
/// bits.
#[inline(always)]
fn tag(hash: u64) -> u8 {
    (hash >> 57) as u8
}

/// The high bit of the bytes of `group` that may equal `tag`: every byte
/// that does, and possibly full bytes above one that does, which the
/// caller's comparison of the entry rejects. Never an empty byte, and the
/// lowest hit is always a true one.
///
/// This is hashbrown's portable match. A byte of `diff` is zero where the
/// group's byte equals the tag, and subtracting one from it borrows into its
/// high bit; a borrow starts only at a zero byte, so a false hit sits above a
/// true one. `!diff` keeps only bytes whose
/// `diff` is below 0x80, which for a tag below 0x80 means a byte below 0x80:
/// a full bucket.
#[inline(always)]
fn match_tag(group: u64, tag: u8) -> u64 {
    let diff = group ^ (LOW_BITS * u64::from(tag));
    diff.wrapping_sub(LOW_BITS) & !diff & HIGH_BITS
}

/// The most keys `buckets` buckets hold: seven eighths, so every probe meets
/// an empty bucket.
#[inline(always)]
fn max_items(buckets: usize) -> usize {
    buckets - buckets / 8
}

/// The smallest table, a power of two of at least [`GROUP`] buckets, that
/// holds `entries` keys; none for no keys.
fn buckets_for(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }
    entries
        .saturating_add(entries / 7)
        .saturating_add(1)
        .max(GROUP)
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

    /// A lower bound on the bytes a hashbrown table holding `entries` entries
    /// of type `T` must allocate: one entry slot and one control byte each.
    /// Derived independently of the production accounting so it can bracket it.
    fn min_table_bytes<T>(entries: usize) -> usize {
        entries * (size_of::<T>() + 1)
    }

    #[test]
    fn map_new_does_not_allocate() {
        let map = ArrowBytesViewMap::<()>::new(OutputType::Utf8View);

        assert_eq!(capacity(&map), 0);
        assert_eq!(map.table_size(), 0);
        assert_eq!(map.size(), 0);
    }

    #[test]
    fn test_size_counts_initial_hash_table_capacity() {
        let map = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );

        assert!(capacity(&map) >= INITIAL_MAP_CAPACITY);
        assert_eq!(map.size(), map.table_size());

        // Before this accounting was corrected the map reported exactly
        // `capacity() * size_of::<Entry<V>>()`, which leaves out the control
        // bytes and undercounts the real allocation by roughly half.
        let lower_bound = min_table_bytes::<Entry<()>>(capacity(&map));
        assert!(
            map.size() >= lower_bound,
            "expected {} to be at least {lower_bound}",
            map.size()
        );
        assert!(
            map.size() <= 2 * lower_bound + 64,
            "expected {} to be within a small factor of {lower_bound}",
            map.size()
        );
        assert!(
            map.size() > capacity(&map) * size_of::<Entry<()>>(),
            "expected {} to exceed {}",
            map.size(),
            capacity(&map) * size_of::<Entry<()>>()
        );
    }

    #[test]
    fn take_preserves_the_capacity_the_map_was_built_with() {
        let mut preallocated = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        let built = capacity(&preallocated);
        preallocated.take();
        assert_eq!(capacity(&preallocated), built);

        let mut lazy = ArrowBytesViewMap::<()>::new(OutputType::Utf8View);
        lazy.take();
        assert_eq!(capacity(&lazy), 0);
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

        let warm_size = map.table_size();
        assert!(warm_size > 0);

        // `take` deliberately keeps the map warm, so it does not release the
        // configured capacity.
        map.take();
        let taken_size = map.size();
        assert!(
            taken_size >= min_table_bytes::<Entry<()>>(INITIAL_MAP_CAPACITY),
            "expected take to retain the warm up allocation, got {taken_size}"
        );

        map.clear_and_release();
        assert_eq!(map.table_size(), 0);
        let released_size = map.size();
        assert!(
            released_size < 128,
            "expected the released map to report approximately zero bytes, got {released_size}"
        );

        // The configured capacity survives, so the map warms back up when it is
        // emitted from again.
        map.take();
        assert!(capacity(&map) >= INITIAL_MAP_CAPACITY);
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

        let expected_size = map.table_size()
            + map.views.allocated_size()
            + map.in_progress.allocated_size()
            + map.completed.allocated_size()
            + map.completed.iter().map(Buffer::capacity).sum::<usize>()
            + map.hashes_buffer.allocated_size();
        assert_eq!(map.size(), expected_size);

        // Verify the retained-capacity delta independently from the production formula.
        let legacy_size = map.table_size()
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

    /// Keys the map's table holds without growing.
    fn capacity<V>(map: &ArrowBytesViewMap<V>) -> usize
    where
        V: Debug + PartialEq + Eq + Clone + Copy + Default,
    {
        match &map.table {
            Some(table) => max_items(table.entries.len()),
            None => map.map.capacity(),
        }
    }

    /// The prefetched table of a map that has moved to it.
    fn table<V>(map: &ArrowBytesViewMap<V>) -> &Table<V>
    where
        V: Debug + PartialEq + Eq + Clone + Copy + Default,
    {
        match &map.table {
            Some(table) => table,
            None => panic!("the map is in hashbrown's table"),
        }
    }

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

    /// Each row's payload, the values payloads were made for and the state
    /// after inserting `batches`, the first in hashbrown's table and the rest
    /// with the probe forced to `staged`.
    type Run = (Vec<usize>, Vec<Option<Vec<u8>>>, ArrayRef);

    fn run(batches: &[ArrayRef], staged: bool) -> Run {
        let mut map = ArrowBytesViewMap::<usize>::new(OutputType::Utf8View);
        let mut seen = vec![];
        let mut made = vec![];
        for (index, batch) in batches.iter().enumerate() {
            let mut next = map.len();
            map.insert(
                batch,
                Some(staged && index > 0),
                |value| {
                    made.push(value.map(<[u8]>::to_vec));
                    next += 1;
                    next - 1
                },
                |payload| seen.push(payload),
            );
        }
        assert_eq!(map.table.is_some(), staged);
        (seen, made, map.into_state())
    }

    #[test]
    fn staged_probe_matches_plain_probe() {
        // Keys in hashbrown's table, then a batch shorter than the look-ahead
        // that moves them to the prefetched table, then batches that grow it
        // many times while they are probed.
        let batches = [
            mixed_keys(2048, 2),
            mixed_keys(5, 1),
            mixed_keys(2048, 3),
            mixed_keys(4096, 6),
            mixed_keys(2 * DISTANCE + 1, 4),
        ];
        let plain = run(&batches, false);
        let staged = run(&batches, true);
        assert_eq!(plain.0, staged.0);
        assert_eq!(plain.1, staged.1);
        assert_eq!(&plain.2, &staged.2);

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
        let buckets = table(&map).entries.len();

        let state = map.take_with_capacity(64);
        assert_eq!(state.len(), len);
        assert!(map.is_empty());
        assert_eq!(table(&map).entries.len(), buckets);
        assert!(table(&map).ctrl.iter().all(|byte| *byte == EMPTY));

        // Every key is new again, and the emitted keys are the same keys.
        let mut made = 0;
        map.insert(&values, Some(true), |_| made += 1, |_| ());
        assert_eq!(made, len);
        assert_eq!(&map.take_with_capacity(0), &state);

        // Taking the map, or releasing it, goes back to hashbrown's table.
        map.insert(&values, Some(true), |_| (), |_| ());
        assert!(map.take().table.is_some());
        assert!(map.table.is_none());
        map.insert(&values, Some(true), |_| (), |_| ());
        map.clear_and_release();
        assert!(map.table.is_none());
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
        map.insert(
            &values,
            Some(true),
            |_| {
                next += 1;
                next - 1
            },
            |_| (),
        );
        assert_eq!(map.non_null_len(), keys.len());
        let grown = table(&map);
        assert!(grown.items <= max_items(grown.entries.len()));
        let buckets = grown.entries.len();
        assert_eq!(grown.ctrl.len(), buckets + GROUP);
        assert_eq!(grown.ctrl[..GROUP], grown.ctrl[buckets..]);
        assert_eq!(
            grown.ctrl[..buckets]
                .iter()
                .filter(|byte| **byte != EMPTY)
                .count(),
            keys.len()
        );

        let mut seen = vec![];
        map.insert(&values, Some(true), |_| 0, |payload| seen.push(payload));
        assert_eq!(seen, (0..keys.len()).collect::<Vec<_>>());
    }

    #[test]
    fn moves_to_the_prefetched_table_only_beyond_l2() {
        let values = mixed_keys(8192, 7);
        let mut small = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        small.insert_if_new(&values, |_| (), |_| ());
        small.insert_if_new(&values, |_| (), |_| ());
        assert!(small.table.is_none());

        let mut big = ArrowBytesViewMap::<()>::with_capacity(
            OutputType::Utf8View,
            l2_cache_bytes() / size_of::<Entry<()>>(),
        );
        big.insert_if_new(&values, |_| (), |_| ());
        assert!(big.table.is_some());
        assert!(capacity(&big) >= l2_cache_bytes() / size_of::<Entry<()>>());
        assert_eq!(&big.into_state(), &small.into_state());
    }

    #[test]
    fn buckets_for_keeps_load_under_seven_eighths() {
        assert_eq!(buckets_for(0), 0);
        for entries in (1..5000).chain([1 << 20, (1 << 20) + 1]) {
            let buckets = buckets_for(entries);
            assert!(buckets.is_power_of_two());
            assert!(buckets >= GROUP);
            assert!(max_items(buckets) >= entries);
            assert!(max_items(buckets) < buckets);
        }
    }

    #[test]
    fn match_tag_marks_every_equal_byte_and_no_empty_one() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Bytes from a few tags and EMPTY, so equal neighbours are common.
            let bytes: [u8; GROUP] =
                std::array::from_fn(|at| match (state >> (8 * at)) as u8 % 5 {
                    0 => EMPTY,
                    pick => pick - 1,
                });
            let group = u64::from_le_bytes(bytes);
            for tag in 0..5u8 {
                let hits = match_tag(group, tag);
                let marked = |at: usize| hits & (0x80 << (8 * at)) != 0;
                for (at, byte) in bytes.iter().enumerate() {
                    if *byte == tag {
                        assert!(marked(at), "{bytes:?} tag {tag} byte {at}");
                    }
                    if marked(at) {
                        assert_ne!(*byte, EMPTY, "{bytes:?} tag {tag} byte {at}");
                        // A false hit sits above a true one, so the lowest
                        // hit is always true.
                        assert!(
                            *byte == tag || bytes[..at].contains(&tag),
                            "{bytes:?} tag {tag} byte {at}"
                        );
                    }
                }
            }
            // The high bits mark exactly the empty bytes.
            for (at, byte) in bytes.iter().enumerate() {
                assert_eq!(group & (0x80 << (8 * at)) != 0, *byte == EMPTY);
            }
        }
    }
}
