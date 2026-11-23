// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`Table`]: a SwissTable-style hash table whose memory a probe can
//! prefetch, for group maps that have outgrown the core's cache.
//!
//! Hashbrown's table exposes no addresses, so a caller cannot prefetch the
//! slot a later row will probe. This table can: [`Table::prefetch_home`]
//! hints the control bytes and the entry at a hash's home bucket, and
//! [`Table::candidate`] returns the first entry a hash's tag selects there,
//! so a caller can prefetch what that entry points to. A staged probe (see
//! [`crate::prefetch`]) issues these for a row `DISTANCE` rows ahead of the
//! one it looks up, so the misses of different rows overlap.
//!
//! The table only pays beyond the cache; callers keep hashbrown's table
//! below [`crate::prefetch::l2_cache_bytes`] and move to this one past it.

use std::mem::{self, MaybeUninit};

use crate::prefetch::prefetch;

/// A SwissTable-style hash table of `E` entries whose memory a probe can
/// prefetch.
///
/// It has a power of two of buckets, at least [`GROUP`]. `entries` holds one
/// `E` per bucket, and `ctrl` one control byte per bucket followed by a copy
/// of the first [`GROUP`] bytes, so a group of control bytes can be read at
/// any bucket without wrapping. A control byte is [`EMPTY`], or the top 7
/// bits of the hash of the key whose entry is in that bucket. A key's probe
/// reads groups of [`GROUP`] control bytes from `hash & (buckets - 1)`
/// onwards and stops at the first group with an empty bucket; a new key goes
/// into the first empty bucket its probe meets. At most seven eighths of the
/// buckets are full, nothing is removed but by [`Self::clear`], and `items`
/// counts the full buckets. An entry is written exactly when its control
/// byte becomes full, and read only where the control byte is full, so a new
/// table, like hashbrown's, writes one byte per bucket rather than a whole
/// entry.
///
/// The table does not know how to hash an entry: growth and
/// [`Self::reserve`] take `hash_of`, which gives the hash an entry was put
/// with. An entry that stores its hash returns it; one that stores a
/// primitive key may hash the key again.
pub struct Table<E: Copy> {
    /// One control byte per bucket and a copy of the first [`GROUP`]
    pub(crate) ctrl: Vec<u8>,
    /// One entry per bucket, initialised where `ctrl` is full
    pub(crate) entries: Box<[MaybeUninit<E>]>,
    /// Buckets less one
    pub(crate) mask: usize,
    /// Full buckets
    pub(crate) items: usize,
}

impl<E: Copy> Table<E> {
    /// An empty table of `buckets` buckets, a power of two of at least
    /// [`GROUP`].
    pub fn with_buckets(buckets: usize) -> Self {
        debug_assert!(buckets.is_power_of_two() && buckets >= GROUP);
        Self {
            ctrl: vec![EMPTY; buckets + GROUP],
            entries: Box::new_uninit_slice(buckets),
            mask: buckets - 1,
            items: 0,
        }
    }

    /// An empty table that holds `capacity` entries without growing.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_buckets(buckets_for(capacity.max(1)))
    }

    /// A table with at least room for `capacity` entries holding `entries`,
    /// each put with the hash `hash_of` gives. The entries must be distinct
    /// keys.
    pub fn from_entries<'a>(
        entries: impl IntoIterator<Item = &'a E>,
        capacity: usize,
        hash_of: impl Fn(&E) -> u64,
    ) -> Self
    where
        E: 'a,
    {
        let mut table = Self::with_capacity(capacity);
        for entry in entries {
            let hash = hash_of(entry);
            let bucket = table.room(hash, table.vacant(hash), &hash_of);
            table.put(bucket, hash, *entry);
        }
        table
    }

    /// Number of full buckets: the entries the table holds.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.items
    }

    /// True if the table holds no entry.
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }

    /// Number of buckets.
    pub fn buckets(&self) -> usize {
        self.mask + 1
    }

    /// The entries the table holds without growing.
    pub fn capacity(&self) -> usize {
        max_items(self.buckets())
    }

    /// Bytes the table allocates: the entries and the control bytes.
    pub fn allocation_size(&self) -> usize {
        self.ctrl.capacity() + self.entries.len() * size_of::<E>()
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

    /// The entry of the full `bucket`, as [`Self::find`] returns it.
    ///
    /// # Panics
    ///
    /// In debug builds, if `bucket` is not a full bucket of this table.
    #[inline(always)]
    pub fn entry(&self, bucket: usize) -> &E {
        assert!(bucket <= self.mask);
        debug_assert_eq!(self.ctrl[bucket] & EMPTY, 0);
        // SAFETY: `bucket` is at most `mask`, below the length of `entries`,
        // and callers pass only a bucket whose control byte is full, whose
        // entry `put` or `resize` wrote. A bucket that is not full holds an
        // entry written before the last `clear`, or none; `E: Copy` has no
        // invalid bit patterns callers rely on beyond what `put` wrote, so
        // callers must not pass one, which the debug assertion checks.
        unsafe { self.entries.get_unchecked(bucket).assume_init_ref() }
    }

    /// The entry of the full `bucket`, to change in place. A change must
    /// keep the hash `hash_of` gives for it.
    #[inline(always)]
    pub fn entry_mut(&mut self, bucket: usize) -> &mut E {
        assert!(bucket <= self.mask);
        debug_assert_eq!(self.ctrl[bucket] & EMPTY, 0);
        // SAFETY: as in `entry`.
        unsafe { self.entries.get_unchecked_mut(bucket).assume_init_mut() }
    }

    /// Sets the control byte of `bucket`, and its copy after the last bucket.
    #[inline(always)]
    fn set_ctrl(&mut self, bucket: usize, byte: u8) {
        self.ctrl[bucket] = byte;
        if bucket < GROUP {
            self.ctrl[self.mask + 1 + bucket] = byte;
        }
    }

    /// The bucket of the entry with `hash` that `eq` accepts; or, if there
    /// is none, the first empty bucket of its probe. `eq` sees only entries
    /// whose tag matches `hash`'s.
    #[inline(always)]
    pub fn find(
        &self,
        hash: u64,
        mut eq: impl FnMut(&E) -> bool,
    ) -> Result<usize, usize> {
        let tag = tag(hash);
        let mut pos = hash as usize & self.mask;
        loop {
            let group = self.group(pos);
            let mut hits = match_tag(group, tag);
            while hits != 0 {
                let bucket = (pos + (hits.trailing_zeros() / 8) as usize) & self.mask;
                if eq(self.entry(bucket)) {
                    return Ok(bucket);
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

    /// The bucket a new entry with `hash` goes into, given `vacant` from
    /// [`Self::find`]: that bucket, unless the table is full and grows first.
    #[inline(always)]
    pub fn room(
        &mut self,
        hash: u64,
        vacant: usize,
        hash_of: impl Fn(&E) -> u64,
    ) -> usize {
        if self.items < max_items(self.mask + 1) {
            return vacant;
        }
        self.resize(buckets_for(self.items + 1), hash_of);
        self.vacant(hash)
    }

    /// Fills the empty `bucket`, from [`Self::room`], with `entry`, whose
    /// hash is `hash`.
    #[inline(always)]
    pub fn put(&mut self, bucket: usize, hash: u64, entry: E) {
        debug_assert!(self.ctrl[bucket] & EMPTY != 0);
        self.set_ctrl(bucket, tag(hash));
        self.entries[bucket].write(entry);
        self.items += 1;
    }

    /// The first empty bucket of `hash`'s probe, for [`Self::room`]: where
    /// an entry goes without looking for an equal one first, as for a hash
    /// that may already be in the table under another key.
    pub fn vacant(&self, hash: u64) -> usize {
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
    /// [`GROUP`] that holds every entry, and places each entry again from the
    /// hash `hash_of` gives.
    fn resize(&mut self, buckets: usize, hash_of: impl Fn(&E) -> u64) {
        let old = mem::replace(self, Self::with_buckets(buckets));
        for (bucket, byte) in old.ctrl[..=old.mask].iter().enumerate() {
            if byte & EMPTY == 0 {
                let entry = *old.entry(bucket);
                let bucket = self.vacant(hash_of(&entry));
                self.entries[bucket].write(entry);
                self.set_ctrl(bucket, *byte);
            }
        }
        self.items = old.items;
    }

    /// Grows the table to hold `capacity` entries without growing again.
    pub fn reserve(&mut self, capacity: usize, hash_of: impl Fn(&E) -> u64) {
        let buckets = buckets_for(capacity);
        if buckets > self.mask + 1 {
            self.resize(buckets, hash_of);
        }
    }

    /// Empties every bucket, keeping the allocation.
    pub fn clear(&mut self) {
        self.ctrl.fill(EMPTY);
        self.items = 0;
    }

    /// The entries the table holds, in bucket order.
    pub fn iter(&self) -> impl Iterator<Item = &E> + '_ {
        self.ctrl[..=self.mask]
            .iter()
            .enumerate()
            .filter(|(_, byte)| *byte & EMPTY == 0)
            .map(|(bucket, _)| self.entry(bucket))
    }

    /// Keeps the entries `keep` returns true for, after it has changed them
    /// as it likes but for their hash, in a table as large as this one.
    pub fn retain(
        &mut self,
        mut keep: impl FnMut(&mut E) -> bool,
        hash_of: impl Fn(&E) -> u64,
    ) {
        let old = mem::replace(self, Self::with_buckets(self.mask + 1));
        for (bucket, byte) in old.ctrl[..=old.mask].iter().enumerate() {
            if byte & EMPTY == 0 {
                let mut entry = *old.entry(bucket);
                if keep(&mut entry) {
                    let bucket = self.vacant(hash_of(&entry));
                    self.entries[bucket].write(entry);
                    self.set_ctrl(bucket, *byte);
                    self.items += 1;
                }
            }
        }
    }

    /// Staged probe, first stage: prefetch the control bytes and the entry
    /// of `hash`'s home bucket.
    #[inline(always)]
    pub fn prefetch_home(&self, hash: u64) {
        let home = hash as usize & self.mask;
        prefetch(self.ctrl.as_ptr().wrapping_add(home));
        prefetch(self.entries.as_ptr().wrapping_add(home));
    }

    /// Staged probe, second stage: the entry of the first bucket in the home
    /// group of `hash` whose control byte may match, after prefetching it.
    #[inline(always)]
    pub fn candidate(&self, hash: u64) -> Option<&E> {
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
pub(crate) const GROUP: usize = 8;

/// The control byte of an empty bucket. A full bucket's byte is a 7-bit tag,
/// so only an empty one has its high bit set.
pub(crate) const EMPTY: u8 = 0x80;

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

/// The most entries `buckets` buckets hold: seven eighths, so every probe
/// meets an empty bucket.
#[inline(always)]
pub(crate) fn max_items(buckets: usize) -> usize {
    buckets - buckets / 8
}

/// The smallest table, a power of two of at least [`GROUP`] buckets, that
/// holds `entries` entries; none for no entries.
///
/// A table of `B` buckets holds `B - B / 8` entries, so it needs `entries +
/// ceil(entries / 7)` buckets: exactly `B` for hashbrown's capacity of `B`
/// buckets, which is the same `B - B / 8`. A map moved from hashbrown's
/// table at its capacity keeps its bucket count.
pub fn buckets_for(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }
    entries
        .saturating_add(entries.div_ceil(7))
        .max(GROUP)
        .checked_next_power_of_two()
        .unwrap_or(1 << (usize::BITS - 1))
}

/// Bytes a [`Table`] of `E` that holds `entries` entries allocates.
pub fn table_bytes<E>(entries: usize) -> usize {
    let buckets = buckets_for(entries);
    if buckets == 0 {
        0
    } else {
        buckets * (size_of::<E>() + 1) + GROUP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // A full table, and hashbrown's capacity for as many buckets,
        // need no more buckets than that.
        for buckets in (3..30).map(|bits| 1usize << bits) {
            assert_eq!(buckets_for(max_items(buckets)), buckets);
            assert_eq!(buckets_for(max_items(buckets) + 1), 2 * buckets);
        }
        for buckets in (3..18).map(|bits| 1usize << bits) {
            let map = hashbrown::HashTable::<u64>::with_capacity(max_items(buckets));
            assert_eq!(map.capacity(), max_items(buckets));
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

    /// A hash with few distinct values, so probes collide and run past their
    /// home group.
    fn crowded(key: u64) -> u64 {
        (key % 61).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Growth, `retain`, `iter`, `reserve` and `clear` keep every entry
    /// findable and the control bytes consistent, against a plain map.
    #[test]
    fn table_matches_a_plain_map() {
        let mut table = Table::<(u64, usize)>::with_capacity(1);
        let mut plain = std::collections::HashMap::new();
        let hash_of = |entry: &(u64, usize)| crowded(entry.0);
        let mut state = 7u64;
        for step in 0..30_000usize {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let key = (state >> 40) % 5000;
            let hash = crowded(key);
            let found = table.find(hash, |entry| entry.0 == key);
            match (found, plain.get(&key)) {
                (Ok(bucket), Some(value)) => assert_eq!(table.entry(bucket).1, *value),
                (Err(vacant), None) => {
                    let bucket = table.room(hash, vacant, hash_of);
                    table.put(bucket, hash, (key, step));
                    plain.insert(key, step);
                }
                (found, expected) => panic!("{key}: {found:?} against {expected:?}"),
            }
            if step % 7000 == 6999 {
                table.retain(
                    |entry| {
                        entry.1 += 1;
                        entry.0 % 3 != 0
                    },
                    hash_of,
                );
                plain.retain(|key, _| key % 3 != 0);
                for value in plain.values_mut() {
                    *value += 1;
                }
                table.reserve(table.len() * 3, hash_of);
            }
            assert_eq!(table.len(), plain.len());
        }
        let mut held = table.iter().copied().collect::<Vec<_>>();
        held.sort_unstable();
        let mut expected = plain.into_iter().collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(held, expected);
        let buckets = table.buckets();
        assert_eq!(table.ctrl.len(), buckets + GROUP);
        assert_eq!(table.ctrl[..GROUP], table.ctrl[buckets..]);
        assert_eq!(
            table.ctrl[..buckets]
                .iter()
                .filter(|b| **b != EMPTY)
                .count(),
            table.len()
        );
        assert!(table.len() <= table.capacity());

        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.buckets(), buckets);
        assert!(table.iter().next().is_none());
    }

    #[test]
    fn from_entries_and_sizes() {
        let entries = (0..100u64)
            .map(|key| (key, key as usize))
            .collect::<Vec<_>>();
        let table =
            Table::from_entries(&entries, 10, |entry: &(u64, usize)| crowded(entry.0));
        assert_eq!(table.len(), 100);
        assert!(table.capacity() >= 100);
        assert_eq!(
            table.allocation_size(),
            table.buckets() * (size_of::<(u64, usize)>() + 1) + GROUP
        );
        assert_eq!(table_bytes::<(u64, usize)>(100), table.allocation_size());
        assert_eq!(table_bytes::<(u64, usize)>(0), 0);
    }
}
