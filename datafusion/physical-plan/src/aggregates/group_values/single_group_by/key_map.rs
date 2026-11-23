// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`KeyMap`]: group ids for fixed-width keys, the core of
//! [`GroupValuesPrimitive`] and of the packed multi-column group values.
//!
//! [`GroupValuesPrimitive`]: super::primitive::GroupValuesPrimitive

use std::mem;

use arrow::array::ArrowNativeTypeOp;
use arrow::buffer::NullBuffer;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use datafusion_physical_expr_common::prefetch::{DISTANCE, l2_cache_bytes};
use datafusion_physical_expr_common::swiss_table::{Table, table_bytes};
use hashbrown::hash_table::HashTable;

use super::primitive::HashValue;

/// A fixed-width group key: a primitive value, or several packed into one.
pub(crate) trait GroupKey: Copy + Default + Send + HashValue {
    /// Whether two keys are the same group. Keys are compared after
    /// [`HashValue::normalize_key`], so this may compare bits.
    fn same(self, other: Self) -> bool;
}

impl<T: ArrowNativeTypeOp + HashValue> GroupKey for T {
    #[inline(always)]
    fn same(self, other: Self) -> bool {
        self.is_eq(other)
    }
}

/// An entry of either table: the group id and its key.
type Slot<K> = (usize, K);

/// Group ids for keys of type `K`, assigned in the order keys first appear.
///
/// # Tables
///
/// Each table entry holds `(group, key)` under the key's hash, so a probe
/// compares the key in the entry it has already loaded and never reads
/// `values` (apache/datafusion#25247). Growth hashes the stored keys again,
/// which for a fixed-width key is cheaper than a wider entry with the hash.
///
/// The map starts in hashbrown's table, probed one row at a time. Once that
/// table outgrows one core's share of L2 ([`l2_cache_bytes`]) it moves, once,
/// at the start of the next batch or when a reservation alone outgrows the
/// gate, to a prefetchable [`Table`]. A batch there is probed in stages:
/// every row's hash is computed first, then while row `i` is looked up the
/// control bytes and entry at the home bucket of row `i + DISTANCE` are
/// prefetched. The key is in the entry, so there is no further miss to hide.
/// Lookups and inserts still happen row by row in row order in either table,
/// so group ids do not depend on which table holds the keys.
///
/// # Hash independence
///
/// The table hash is `HashValue::hash` with the aggregation's seed
/// (`AGGREGATION_HASH_SEED`), applied to the whole key. The rows a
/// partition or a bucket receives were chosen by `create_hashes` over the
/// key's columns with `RepartitionExec`'s seed, or with the bucket router's
/// seed (`BUCKET_HASH_SEED`). A different seed makes the table hash
/// independent of those, so the keys of one partition or bucket, which share
/// a residue of the routing hash, still spread over every bucket of the
/// table.
pub(crate) struct KeyMap<K: GroupKey> {
    /// The table below the gate; empty, holding no allocation, once `table`
    /// is set
    pub(super) map: HashTable<Slot<K>>,
    /// The prefetchable table, once the map has outgrown L2
    pub(super) table: Option<Box<Table<Slot<K>>>>,
    /// The group id of the null key, if any
    pub(super) null_group: Option<usize>,
    /// The key of each group id
    pub(super) values: Vec<K>,
    /// Hashes of the batch probed in stages
    pub(super) hashes: Vec<u64>,
    random_state: RandomState,
}

impl<K: GroupKey> KeyMap<K> {
    pub(crate) fn new(capacity: usize, random_state: RandomState) -> Self {
        Self {
            map: HashTable::with_capacity(capacity),
            table: None,
            null_group: None,
            values: Vec::with_capacity(capacity),
            hashes: Vec::new(),
            random_state,
        }
    }

    /// The hash of a stored entry's key.
    #[inline(always)]
    fn hash_of(state: &RandomState) -> impl Fn(&Slot<K>) -> u64 + '_ {
        move |slot| slot.1.hash(state)
    }

    /// Moves the keys to the prefetchable table, with room for `capacity`.
    fn move_to_table(&mut self, capacity: usize) {
        let map = mem::take(&mut self.map);
        self.table = Some(Box::new(Table::from_entries(
            &map,
            capacity.max(map.capacity()).max(map.len()),
            Self::hash_of(&self.random_state),
        )));
    }

    /// Writes the group id of each key of `keys` to `groups`, adding a group
    /// for each key not seen before. A row that `nulls` marks null is the
    /// null group, whatever its key.
    pub(crate) fn intern(
        &mut self,
        keys: &[K],
        nulls: Option<&NullBuffer>,
        groups: &mut Vec<usize>,
    ) {
        groups.clear();
        // A map in hashbrown's table below the gate runs this loop, one row
        // at a time, as the map always has.
        if self.table.is_some() || self.map.allocation_size() > l2_cache_bytes() {
            return self.intern_staged(keys, nulls, groups);
        }
        let state = &self.random_state;
        for (row, key) in keys.iter().enumerate() {
            let group = if nulls.is_some_and(|nulls| nulls.is_null(row)) {
                *self.null_group.get_or_insert_with(|| {
                    let group = self.values.len();
                    self.values.push(K::default());
                    group
                })
            } else {
                // Fold equivalence-class duplicates (e.g. `-0.0` → `+0.0`)
                // so the bit-equal comparison matches and the stored value
                // is the canonical representative.
                let key = key.normalize_key();
                let hash = key.hash(state);
                let entry =
                    self.map
                        .entry(hash, |&(_, k)| k.same(key), |&(_, k)| k.hash(state));
                match entry {
                    hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                    hashbrown::hash_table::Entry::Vacant(v) => {
                        let group = self.values.len();
                        v.insert((group, key));
                        self.values.push(key);
                        group
                    }
                }
            };
            groups.push(group);
        }
    }

    /// [`Self::intern`] in the prefetchable table, moving the keys to it
    /// first if they are not there yet. Kept out of line so the loop over
    /// hashbrown's table compiles as it does without this table.
    #[inline(never)]
    fn intern_staged(
        &mut self,
        keys: &[K],
        nulls: Option<&NullBuffer>,
        groups: &mut Vec<usize>,
    ) {
        if self.table.is_none() {
            self.move_to_table(0);
        }
        let Some(mut table) = self.table.take() else {
            unreachable!("moved to the table above")
        };
        let state = &self.random_state;
        let mut hashes = mem::take(&mut self.hashes);
        hashes.clear();
        // A null row's key is hashed too; its hash is only prefetched.
        hashes.extend(keys.iter().map(|key| key.normalize_key().hash(state)));
        for (row, (key, &hash)) in keys.iter().zip(&hashes).enumerate() {
            if let Some(&ahead) = hashes.get(row + DISTANCE) {
                table.prefetch_home(ahead);
            }
            let group = if nulls.is_some_and(|nulls| nulls.is_null(row)) {
                *self.null_group.get_or_insert_with(|| {
                    let group = self.values.len();
                    self.values.push(K::default());
                    group
                })
            } else {
                let key = key.normalize_key();
                match table.find(hash, |&(_, k)| k.same(key)) {
                    Ok(bucket) => table.entry(bucket).0,
                    Err(vacant) => {
                        let group = self.values.len();
                        let bucket = table.room(hash, vacant, Self::hash_of(state));
                        table.put(bucket, hash, (group, key));
                        self.values.push(key);
                        group
                    }
                }
            };
            groups.push(group);
        }
        self.hashes = hashes;
        self.table = Some(table);
    }

    /// Reserves room for `capacity` groups. A reservation that alone outgrows
    /// the gate moves the keys to the prefetchable table now, rather than
    /// sizing hashbrown's table only to copy it.
    pub(crate) fn reserve_groups(&mut self, capacity: usize) {
        if self.table.is_none() && table_bytes::<Slot<K>>(capacity) > l2_cache_bytes() {
            self.move_to_table(capacity);
        }
        match &mut self.table {
            Some(table) => table.reserve(capacity, Self::hash_of(&self.random_state)),
            None => self.map.reserve(
                capacity.saturating_sub(self.map.len()),
                Self::hash_of(&self.random_state),
            ),
        }
        self.values
            .reserve_exact(capacity.saturating_sub(self.values.len()));
    }

    /// Bytes held by the tables, the keys and the scratch hashes.
    pub(crate) fn size(&self) -> usize {
        self.map.allocation_size()
            + self.table.as_ref().map_or(0, |table| {
                size_of::<Table<Slot<K>>>() + table.allocation_size()
            })
            + self.values.allocated_size()
            + self.hashes.allocated_size()
    }

    /// Number of groups, the null group included.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// The key of each group, in group id order; the null group's is the
    /// default key.
    pub(crate) fn values(&self) -> &[K] {
        &self.values
    }

    /// The group id of the null key, if any.
    pub(crate) fn null_group(&self) -> Option<usize> {
        self.null_group
    }

    /// Removes the groups `emit_to` names and returns their keys and the
    /// position of the null group among them, if it is one of them. The
    /// remaining groups' ids go down by the number removed.
    pub(crate) fn emit(&mut self, emit_to: EmitTo) -> (Vec<K>, Option<usize>) {
        match emit_to {
            EmitTo::All => {
                self.map.clear();
                if let Some(table) = &mut self.table {
                    table.clear();
                }
                (mem::take(&mut self.values), self.null_group.take())
            }
            EmitTo::First(n) => {
                // Decrement group index by n, removing those below it
                let shift = |entry: &mut Slot<K>| match entry.0.checked_sub(n) {
                    Some(sub) => {
                        entry.0 = sub;
                        true
                    }
                    None => false,
                };
                match &mut self.table {
                    Some(table) => table.retain(shift, Self::hash_of(&self.random_state)),
                    None => self.map.retain(shift),
                }
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };
                (split_vec_min_alloc(&mut self.values, n), null_group)
            }
        }
    }

    /// Releases capacity the stored groups do not need. Keys that fit below
    /// the gate again go back to hashbrown's table.
    pub(crate) fn shrink_to_fit(&mut self) {
        let state = &self.random_state;
        match self.table.take() {
            Some(table) if table_bytes::<Slot<K>>(table.len()) > l2_cache_bytes() => {
                self.table = Some(Box::new(Table::from_entries(
                    table.iter(),
                    table.len(),
                    Self::hash_of(state),
                )));
            }
            Some(table) => {
                let mut map = HashTable::with_capacity(table.len());
                for slot in table.iter() {
                    map.insert_unique(slot.1.hash(state), *slot, Self::hash_of(state));
                }
                self.map = map;
            }
            None => self.map.shrink_to_fit(Self::hash_of(state)),
        }
        self.values.shrink_to_fit();
        self.hashes = Vec::new();
    }

    /// Removes every group, keeping room for `capacity`: in the prefetchable
    /// table if that is still past the gate, as `take_with_capacity` does for
    /// byte keys, and in hashbrown's table otherwise.
    pub(crate) fn clear_shrink(&mut self, capacity: usize) {
        self.values.clear();
        self.values.shrink_to(capacity);
        self.null_group = None;
        self.map.clear();
        match self.table.take() {
            Some(mut table) if table_bytes::<Slot<K>>(capacity) > l2_cache_bytes() => {
                if table.capacity() / 2 >= capacity.max(1) {
                    table = Box::new(Table::with_capacity(capacity));
                } else {
                    table.clear();
                }
                self.table = Some(table);
            }
            _ => self.map.shrink_to(capacity, |_| 0), // the map is empty
        }
        self.hashes.clear();
        self.hashes.shrink_to(capacity);
    }

    /// Whether the keys are in the prefetchable table.
    #[cfg(test)]
    pub(crate) fn is_staged(&self) -> bool {
        self.table.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::HashMap;

    fn keys(rows: usize, seed: u64) -> (Vec<i64>, NullBuffer) {
        let mut state = seed;
        let mut valid = vec![];
        let keys = (0..rows)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                valid.push(state >> 60 != 0);
                ((state >> 20) % (rows as u64 / 2 + 1)) as i64 - 50
            })
            .collect();
        (keys, NullBuffer::from(valid))
    }

    /// Interns `batches`, the first in hashbrown's table and the rest after
    /// moving to the prefetchable table when `staged`, and checks each
    /// row's group against a plain map of first appearances.
    fn run(batches: &[(Vec<i64>, NullBuffer)], staged: bool) -> (Vec<usize>, Vec<i64>) {
        let mut map = KeyMap::<i64>::new(0, RandomState::with_seed(7));
        let mut first = HashMap::<Option<i64>, usize>::new();
        let mut all = vec![];
        let mut groups = vec![];
        for (index, (keys, nulls)) in batches.iter().enumerate() {
            if staged && index == 1 {
                map.reserve_groups(map.len());
                map.move_to_table(0);
            }
            map.intern(keys, Some(nulls), &mut groups);
            for (row, group) in groups.iter().enumerate() {
                let key = nulls.is_valid(row).then_some(keys[row]);
                let next = first.len();
                assert_eq!(*first.entry(key).or_insert(next), *group);
            }
            all.extend_from_slice(&groups);
        }
        assert_eq!(map.is_staged(), staged);
        (all, map.values().to_vec())
    }

    #[test]
    fn staged_intern_matches_plain_intern() {
        let batches = [
            keys(2048, 1),
            keys(5, 2),
            keys(4096, 3),
            keys(8192, 4),
            keys(DISTANCE + 1, 5),
        ];
        assert_eq!(run(&batches, false), run(&batches, true));
    }

    #[test]
    fn emit_first_and_shrink_keep_ids() {
        for staged in [false, true] {
            let (keys, nulls) = keys(5000, 9);
            let mut map = KeyMap::<i64>::new(0, RandomState::with_seed(7));
            if staged {
                map.move_to_table(0);
            }
            let mut groups = vec![];
            map.intern(&keys, Some(&nulls), &mut groups);
            let before = groups.clone();
            let len = map.len();
            let (emitted, _) = map.emit(EmitTo::First(100));
            assert_eq!(emitted.len(), 100);
            map.shrink_to_fit();
            map.intern(&keys, Some(&nulls), &mut groups);
            // Groups that were not emitted keep their key, 100 ids lower;
            // emitted ones come back as new groups after them.
            for (row, group) in groups.iter().enumerate() {
                if before[row] >= 100 {
                    assert_eq!(*group, before[row] - 100, "row {row}");
                } else {
                    assert!(*group >= len - 100, "row {row}");
                }
            }
            map.clear_shrink(0);
            assert_eq!(map.len(), 0);
            assert!(!map.is_staged());
        }
    }

    #[cfg(not(feature = "force_hash_collisions"))]
    #[test]
    fn moves_to_the_table_only_beyond_l2() {
        let keys = (0..8192i64).collect::<Vec<_>>();
        let mut groups = vec![];
        let mut small = KeyMap::<i64>::new(128, RandomState::with_seed(7));
        small.intern(&keys, None, &mut groups);
        small.intern(&keys, None, &mut groups);
        assert!(!small.is_staged());

        let mut big = KeyMap::<i64>::new(128, RandomState::with_seed(7));
        big.reserve_groups(l2_cache_bytes() / size_of::<Slot<i64>>());
        assert!(big.is_staged());
        big.intern(&keys, None, &mut groups);
        assert_eq!(groups, (0..8192).collect::<Vec<_>>());
        assert_eq!(big.values(), small.values());
    }
}
