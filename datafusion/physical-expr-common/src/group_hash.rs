// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`GroupHash`], the hash of a row's group keys that a two-stage aggregation
//! computes once and carries from its partial to its final stage.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, Schema, UInt64Type};
use arrow::record_batch::RecordBatch;
use datafusion_common::hash_utils::{RandomState, create_hashes};

/// The hash of a row's group keys, computed once and carried between the
/// operators of a two-stage aggregation instead of being recomputed by each.
///
/// It is the hash `RepartitionExec` routes rows by: [`create_hashes`] of the
/// group key arrays with [`GroupHash::STATE`], seed 0. A row that carries it
/// goes to the same partition, `hash % n`, as one whose hash the exchange
/// computes itself, so hash joins and everything else that reproduces the
/// exchange's routing keep agreeing with it.
///
/// # Bit layout
///
/// The consumers of one carried hash `h` read disjoint bits of it:
///
/// | Bits of `h` | Read by |
/// |---|---|
/// | `h % n`: bits 0..8 for a power-of-two `n` up to 256, `h mod 2` for an even `n` | the exchange, to pick the partition |
/// | 8..24 | the final aggregation, to pick a bucket ([`Self::bucket`]) |
/// | 25..32 | a hash table's tag: its top 7 bits after [`Self::table`] |
/// | 32..64 | a hash table's home slot: its low bits after [`Self::table`] |
///
/// Hash tables probe with [`Self::table`], `h` rotated left by 32 bits. Rows
/// of one partition all share `h % n`, which fixes the low bits of `h` (all
/// of `0..log2(n)` when `n` is a power of two); rotated, those bits sit
/// where neither the home slot of a table below 2^32 slots nor its tag
/// reads, so the rows of one partition still spread over every slot and tag
/// of their table and of each of its buckets. Given the partition, the other
/// bits stay close to uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GroupHash(u64);

impl GroupHash {
    /// The state the carried hash is computed with. It equals the state of
    /// `RepartitionExec`'s routing hash, `REPARTITION_RANDOM_STATE`.
    pub const STATE: RandomState = RandomState::with_seed(0);

    /// First bit of `h` that picks a bucket, see [`Self::bucket`].
    pub const BUCKET_SHIFT: u32 = 8;

    /// Number of bits of `h` reserved for picking buckets, at every level.
    pub const BUCKET_BITS: u32 = 16;

    /// Wraps a hash computed with [`Self::STATE`].
    pub const fn new(carried: u64) -> Self {
        Self(carried)
    }

    /// Unwraps the hash, as the exchange routes by it and as it is carried.
    pub const fn carried(self) -> u64 {
        self.0
    }

    /// The hash a hash table probes with: see the bit layout.
    pub const fn table(self) -> u64 {
        self.0.rotate_left(32)
    }

    /// The carried hash whose [`Self::table`] hash is `table`.
    pub const fn from_table(table: u64) -> Self {
        Self(table.rotate_right(32))
    }

    /// The bucket, among `1 << bits`, that the `bits` bucket bits from
    /// `offset` on pick: bits `BUCKET_SHIFT + offset ..` of `h`. A bucket
    /// split again reads the bits after the ones that picked it. `offset +
    /// bits` must not exceed [`Self::BUCKET_BITS`].
    pub const fn bucket(self, offset: u32, bits: u32) -> usize {
        let shift = Self::BUCKET_SHIFT + offset;
        ((self.0 >> shift) & ((1 << bits) - 1)) as usize
    }
}

/// Name of the column a partial aggregation carries its [`GroupHash`]es in,
/// after its group keys and aggregate state.
pub const GROUP_HASH_COLUMN: &str = "__group_hash";

/// Metadata key of the [`GROUP_HASH_COLUMN`] field. Its value lists the
/// indices of the columns hashed, in order, separated by commas.
pub const GROUP_HASH_METADATA: &str = "datafusion.group_hash";

/// The field of a column that carries the [`GroupHash`] of the columns
/// `keys` of its batch.
pub fn group_hash_field(keys: &[usize]) -> Field {
    let keys = keys
        .iter()
        .map(|key| key.to_string())
        .collect::<Vec<_>>()
        .join(",");
    Field::new(GROUP_HASH_COLUMN, DataType::UInt64, false)
        .with_metadata(HashMap::from([(GROUP_HASH_METADATA.to_string(), keys)]))
}

/// The indices of the columns whose [`GroupHash`] `field` carries, if it is
/// a [`group_hash_field`].
pub fn group_hash_keys(field: &Field) -> Option<Vec<usize>> {
    if field.name() != GROUP_HASH_COLUMN
        || field.data_type() != &DataType::UInt64
        || field.is_nullable()
    {
        return None;
    }
    let keys = field.metadata().get(GROUP_HASH_METADATA)?;
    keys.split(',').map(|key| key.parse().ok()).collect()
}

/// True if the last field of `schema` carries the [`GroupHash`] of exactly
/// its first `num_keys` fields, in order, and is not one of them.
///
/// A plan decides whether an operator reads that column (see
/// `AggregateExec::with_read_group_hash` and
/// `RepartitionExec::with_carried_group_hash`): this only checks, when such a
/// flag is set, that the column it names is the one it expects.
pub fn carries_first_keys(schema: &Schema, num_keys: usize) -> bool {
    let fields = schema.fields();
    num_keys > 0
        && fields.len() > num_keys
        && fields
            .last()
            .and_then(|field| group_hash_keys(field))
            .is_some_and(|keys| keys.iter().copied().eq(0..num_keys))
}

/// The [`GroupHash`] of every row of `batch`, if its last column carries
/// them for exactly the columns `keys`, in that order.
pub fn carried_group_hashes<'a>(
    batch: &'a RecordBatch,
    keys: &[usize],
) -> Option<&'a [u64]> {
    let schema = batch.schema_ref();
    let last = schema.fields().len().checked_sub(1)?;
    if group_hash_keys(schema.field(last))? != keys {
        return None;
    }
    let column = batch.column(last).as_primitive_opt::<UInt64Type>()?;
    (column.null_count() == 0).then(|| column.values().as_ref())
}

/// Fills `out` with the hash a byte map probes its table with for each value
/// of `values`: [`GroupHash::table`] of the `carried` hashes when given, the
/// map's own hash with `random_state` otherwise.
pub(crate) fn table_hashes(
    values: &ArrayRef,
    carried: Option<&[u64]>,
    random_state: &RandomState,
    out: &mut Vec<u64>,
) {
    out.clear();
    match carried {
        Some(carried) => {
            out.extend(carried.iter().map(|&hash| GroupHash::new(hash).table()))
        }
        None => {
            out.resize(values.len(), 0);
            create_hashes([values], random_state, out)
                // hash is supported for all types and create_hashes only
                // returns errors for unsupported types
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_hash_round_trips() {
        for hash in [0, 1, u64::MAX, 0x0123_4567_89ab_cdef] {
            let group = GroupHash::new(hash);
            assert_eq!(GroupHash::from_table(group.table()), group);
            assert_eq!(group.carried(), hash);
        }
    }

    #[test]
    fn partition_bits_stay_out_of_slot_tag_and_bucket() {
        // Hashes of one partition of 16 share their low 4 bits. After the
        // rotate, the low 8 slot bits, the tag and the buckets of a level
        // still take every value.
        let partition = 5u64;
        let hashes = (0..1u64 << 16).map(|i| {
            let mixed = i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            (mixed & !0xf) | partition
        });
        let mut slots = std::collections::HashSet::new();
        let mut tags = std::collections::HashSet::new();
        let mut buckets = std::collections::HashSet::new();
        for hash in hashes {
            let group = GroupHash::new(hash);
            slots.insert(group.table() & 0xff);
            tags.insert(group.table() >> 57);
            buckets.insert(group.bucket(0, 6));
        }
        assert_eq!(slots.len(), 256);
        assert_eq!(tags.len(), 128);
        assert_eq!(buckets.len(), 64);
    }

    #[test]
    fn levels_read_disjoint_bits() {
        let hash = GroupHash::new(0b10_1011 << GroupHash::BUCKET_SHIFT);
        assert_eq!(hash.bucket(0, 6), 0b10_1011);
        assert_eq!(hash.bucket(6, 6), 0);
        let hash = GroupHash::new(0b11_0001 << (GroupHash::BUCKET_SHIFT + 6));
        assert_eq!(hash.bucket(0, 6), 0);
        assert_eq!(hash.bucket(6, 6), 0b11_0001);
        assert_eq!(hash.bucket(6, 4), 0b0001);
    }

    #[test]
    fn carried_column_round_trips_its_keys() {
        use arrow::array::{StringArray, UInt64Array};
        use arrow::datatypes::Schema;
        use std::sync::Arc;

        let field = group_hash_field(&[0, 2]);
        assert_eq!(group_hash_keys(&field), Some(vec![0, 2]));
        assert_eq!(
            group_hash_keys(&Field::new("x", DataType::UInt64, false)),
            None
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Utf8, true),
            field,
        ]));
        let strings: ArrayRef = Arc::new(StringArray::from(vec!["x", "y"]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::clone(&strings),
                Arc::clone(&strings),
                strings,
                Arc::new(UInt64Array::from(vec![7, 9])),
            ],
        )
        .unwrap();
        assert_eq!(carried_group_hashes(&batch, &[0, 2]), Some(&[7, 9][..]));
        // Hashes of other keys are not these keys' hashes
        assert_eq!(carried_group_hashes(&batch, &[0, 1]), None);
        assert_eq!(carried_group_hashes(&batch, &[0]), None);
    }
}
