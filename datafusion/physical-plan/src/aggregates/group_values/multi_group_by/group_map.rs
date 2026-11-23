// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`GroupMap`]: the hash table of [`GroupValuesColumn`], in hashbrown's
//! table below the core's L2 cache and in a prefetchable [`Table`] beyond it.
//!
//! [`GroupValuesColumn`]: super::GroupValuesColumn

use datafusion_execution::memory_pool::proxy::HashTableAllocExt;
use datafusion_physical_expr_common::prefetch::l2_cache_bytes;
use datafusion_physical_expr_common::swiss_table::{Table, table_bytes};
use hashbrown::hash_table::HashTable;

use super::GroupIndexView;

/// An entry: a hash and the group, or the list of groups, with that hash.
pub(super) type Slot = (u64, GroupIndexView);

/// The hash of a stored entry.
fn hash_of(slot: &Slot) -> u64 {
    slot.0
}

/// The hash table of [`GroupValuesColumn`]: the same entries and the same
/// operations in either of two tables.
///
/// It starts in hashbrown's table. [`Self::stage`] moves it, once, to a
/// prefetchable [`Table`] when hashbrown's table has outgrown one core's
/// share of L2 ([`l2_cache_bytes`]), and it stays there until it is shrunk
/// below the gate. Entries, and what every operation does with them, do not
/// depend on the table.
///
/// [`GroupValuesColumn`]: super::GroupValuesColumn
pub(super) enum GroupMap {
    Hash(HashTable<Slot>),
    Staged(Box<Table<Slot>>),
}

impl Default for GroupMap {
    fn default() -> Self {
        Self::Hash(HashTable::new())
    }
}

impl GroupMap {
    /// Moves the entries to the prefetchable table if hashbrown's table has
    /// outgrown the gate. True if they are in the prefetchable table.
    pub(super) fn stage(&mut self) -> bool {
        if let Self::Hash(map) = self {
            if map.allocation_size() <= l2_cache_bytes() {
                return false;
            }
            *self = Self::Staged(Box::new(Table::from_entries(
                &*map,
                map.capacity().max(map.len()),
                hash_of,
            )));
        }
        true
    }

    /// The prefetchable table, if the entries are in it.
    pub(super) fn staged(&self) -> Option<&Table<Slot>> {
        match self {
            Self::Hash(_) => None,
            Self::Staged(table) => Some(table),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Hash(map) => map.len(),
            Self::Staged(table) => table.len(),
        }
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entries the table holds without growing.
    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Hash(map) => map.capacity(),
            Self::Staged(table) => table.capacity(),
        }
    }

    /// Bytes the prefetchable table allocates, if the entries are in it.
    pub(super) fn staged_size(&self) -> Option<usize> {
        self.staged()
            .map(|table| size_of::<Table<Slot>>() + table.allocation_size())
    }

    pub(super) fn find(&self, hash: u64, eq: impl FnMut(&Slot) -> bool) -> Option<&Slot> {
        match self {
            Self::Hash(map) => map.find(hash, eq),
            Self::Staged(table) => table.find(hash, eq).ok().map(|b| table.entry(b)),
        }
    }

    pub(super) fn find_mut(
        &mut self,
        hash: u64,
        eq: impl FnMut(&Slot) -> bool,
    ) -> Option<&mut Slot> {
        match self {
            Self::Hash(map) => map.find_mut(hash, eq),
            Self::Staged(table) => match table.find(hash, eq) {
                Ok(bucket) => Some(table.entry_mut(bucket)),
                Err(_) => None,
            },
        }
    }

    /// Inserts `slot`, which is not in the table, adding to `accounting` the
    /// bytes hashbrown's table grows by, as `insert_accounted` does.
    pub(super) fn insert_accounted(
        &mut self,
        slot: Slot,
        hasher: impl Fn(&Slot) -> u64,
        accounting: &mut usize,
    ) {
        match self {
            Self::Hash(map) => map.insert_accounted(slot, hasher, accounting),
            Self::Staged(table) => {
                let hash = hasher(&slot);
                let Err(vacant) = table.find(hash, |entry| entry.0 == hash) else {
                    unreachable!("an inserted hash is not in the table")
                };
                let bucket = table.room(hash, vacant, hash_of);
                table.put(bucket, hash, slot);
            }
        }
    }

    pub(super) fn clear(&mut self) {
        match self {
            Self::Hash(map) => map.clear(),
            Self::Staged(table) => table.clear(),
        }
    }

    pub(super) fn reserve(&mut self, additional: usize, hasher: impl Fn(&Slot) -> u64) {
        let capacity = self.len() + additional;
        if let Self::Hash(map) = self
            && table_bytes::<Slot>(capacity) > l2_cache_bytes()
        {
            // A reservation that alone outgrows the gate goes to the
            // prefetchable table now, rather than sizing hashbrown's table
            // only to copy it.
            *self = Self::Staged(Box::new(Table::from_entries(&*map, capacity, hash_of)));
        }
        match self {
            Self::Hash(map) => map.reserve(additional, hasher),
            Self::Staged(table) => table.reserve(capacity, hash_of),
        }
    }

    pub(super) fn retain(&mut self, keep: impl FnMut(&mut Slot) -> bool) {
        match self {
            Self::Hash(map) => map.retain(keep),
            Self::Staged(table) => table.retain(keep, hash_of),
        }
    }

    /// Releases capacity the entries do not need; entries that fit below
    /// the gate again go back to hashbrown's table.
    pub(super) fn shrink_to_fit(&mut self, hasher: impl Fn(&Slot) -> u64) {
        match self {
            Self::Hash(map) => map.shrink_to_fit(hasher),
            Self::Staged(table) => {
                *self = if table_bytes::<Slot>(table.len()) > l2_cache_bytes() {
                    Self::Staged(Box::new(Table::from_entries(
                        table.iter(),
                        table.len(),
                        hash_of,
                    )))
                } else {
                    let mut map = HashTable::with_capacity(table.len());
                    for slot in table.iter() {
                        map.insert_unique(slot.0, *slot, hash_of);
                    }
                    Self::Hash(map)
                };
            }
        }
    }

    /// Shrinks the empty table to hold `capacity` entries: in the
    /// prefetchable table if that is still past the gate, and in hashbrown's
    /// table otherwise.
    pub(super) fn shrink_to(&mut self, capacity: usize, hasher: impl Fn(&Slot) -> u64) {
        debug_assert_eq!(self.len(), 0);
        match self {
            Self::Hash(map) => map.shrink_to(capacity, hasher),
            Self::Staged(table) => {
                if table_bytes::<Slot>(capacity) <= l2_cache_bytes() {
                    *self = Self::Hash(HashTable::with_capacity(capacity));
                } else if table.capacity() / 2 >= capacity {
                    **table = Table::with_capacity(capacity);
                }
            }
        }
    }
}
