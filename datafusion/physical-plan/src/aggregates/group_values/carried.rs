// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CarriedHashes`]: the hashes a [`GroupValues`] was interned with.
//!
//! [`GroupValues`]: super::GroupValues

use arrow::array::ArrayRef;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use datafusion_physical_expr_common::group_hash::GroupHash;

/// The [`GroupHash`] of every group of a [`GroupValues`] that was interned
/// with [`GroupValues::intern_hashed`], in group order, so that the hashes
/// can leave with the groups when they are emitted.
///
/// A [`GroupValues`] probes its table with hashes of one kind only: its own,
/// or the [`GroupHash::table`] hashes of carried ones. That is enforced here:
/// - carried hashes are admitted into an implementation that holds no
///   groups, or that already carries them, and refused otherwise
///   ([`Self::admit`]);
/// - once hashes are carried, rows that come without them are hashed with
///   [`GroupHash::STATE`] ([`Self::hash_rows`]), which is how the carried
///   hashes were computed, so they probe the same table.
///
/// [`GroupValues`]: super::GroupValues
/// [`GroupValues::intern_hashed`]: super::GroupValues::intern_hashed
#[derive(Debug, Default)]
pub(crate) struct CarriedHashes {
    /// `None` until hashes are first admitted, then the hash of every group
    groups: Option<Vec<u64>>,
    /// The hashes of the groups the last emit removed, until they are taken
    emitted: Option<Vec<u64>>,
    /// Reused to hash the rows that come without hashes
    buffer: Vec<u64>,
}

impl CarriedHashes {
    /// True once hashes are carried.
    pub(crate) fn is_carried(&self) -> bool {
        self.groups.is_some()
    }

    /// Admits carried hashes into an implementation holding `num_groups`
    /// groups. Refuses them if those groups were interned with hashes of the
    /// implementation's own.
    pub(crate) fn admit(&mut self, num_groups: usize) -> Result<()> {
        match self.groups {
            Some(_) => Ok(()),
            None if num_groups == 0 => {
                self.groups = Some(Vec::new());
                Ok(())
            }
            None => internal_err!(
                "Group values holding {num_groups} groups interned with their own hashes cannot intern carried hashes"
            ),
        }
    }

    /// The carried hash of every row of `cols`, for rows that come without
    /// one once hashes are carried. Hand the vector back with
    /// [`Self::reuse`].
    pub(crate) fn hash_rows(&mut self, cols: &[ArrayRef]) -> Result<Vec<u64>> {
        let mut hashes = std::mem::take(&mut self.buffer);
        hashes.clear();
        hashes.resize(cols.first().map_or(0, |col| col.len()), 0);
        create_hashes(cols, &GroupHash::STATE, &mut hashes)?;
        Ok(hashes)
    }

    /// Takes back the vector [`Self::hash_rows`] returned.
    pub(crate) fn reuse(&mut self, hashes: Vec<u64>) {
        self.buffer = hashes;
    }

    /// Records the hashes of the groups one intern created: the groups from
    /// `first_new` up to `num_groups`, where row `i`, with carried hash
    /// `hashes[i]`, went to group `groups[i]`. Rows of one group share their
    /// keys and therefore their hash.
    pub(crate) fn record(
        &mut self,
        first_new: usize,
        num_groups: usize,
        groups: &[usize],
        hashes: &[u64],
    ) {
        let Some(kept) = &mut self.groups else {
            return;
        };
        debug_assert_eq!(kept.len(), first_new);
        if num_groups == first_new {
            return;
        }
        kept.resize(num_groups, 0);
        for (&group, &hash) in groups.iter().zip(hashes) {
            if group >= first_new {
                kept[group] = hash;
            }
        }
    }

    /// Removes the hashes of the groups `emit_to` removes, to be taken with
    /// [`Self::take_emitted`].
    pub(crate) fn emit(&mut self, emit_to: EmitTo) {
        if let Some(kept) = &mut self.groups {
            self.emitted = Some(emit_to.take_needed(kept));
        }
    }

    /// Forgets the hashes of the groups that are left, which the caller is
    /// about to intern again.
    pub(crate) fn restart(&mut self) {
        if let Some(kept) = &mut self.groups {
            kept.clear();
        }
    }

    /// The hashes the last emit removed, once.
    pub(crate) fn take_emitted(&mut self) -> Option<Vec<u64>> {
        self.emitted.take()
    }

    /// Forgets every group's hash and whether hashes are carried, for an
    /// implementation that dropped its groups. Hashes already emitted stay
    /// until they are taken.
    pub(crate) fn clear(&mut self) {
        self.groups = None;
        self.buffer = Vec::new();
    }

    /// Memory held, in bytes
    pub(crate) fn size(&self) -> usize {
        self.groups.as_ref().map_or(0, |kept| kept.allocated_size())
            + self.emitted.as_ref().map_or(0, |kept| kept.allocated_size())
            + self.buffer.allocated_size()
    }
}

/// Fills `out` with the hash a table probes with for each row of `cols`:
/// [`GroupHash::table`] of the `carried` hashes when given, otherwise the
/// hash of `cols` with `random_state`.
pub(crate) fn table_hashes(
    cols: &[ArrayRef],
    carried: Option<&[u64]>,
    random_state: &RandomState,
    out: &mut Vec<u64>,
) -> Result<()> {
    out.clear();
    match carried {
        Some(carried) => {
            out.extend(carried.iter().map(|&hash| GroupHash::new(hash).table()));
        }
        None => {
            out.resize(cols.first().map_or(0, |col| col.len()), 0);
            create_hashes(cols, random_state, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_carried_hashes_after_own_ones() {
        let mut carried = CarriedHashes::default();
        assert!(carried.admit(3).is_err());
        assert!(!carried.is_carried());
        carried.admit(0).unwrap();
        assert!(carried.is_carried());
        // Groups interned since then carry their hashes
        carried.admit(3).unwrap();
    }

    #[test]
    fn keeps_hashes_in_group_order_through_emits() {
        let mut carried = CarriedHashes::default();
        carried.admit(0).unwrap();
        // Rows 0..5 create groups 0, 1, 0, 2, 1 in that order
        carried.record(0, 3, &[0, 1, 0, 2, 1], &[10, 11, 10, 12, 11]);
        // A later batch finds group 2 and creates groups 4 and 3, out of row order
        carried.record(3, 5, &[2, 4, 3], &[12, 14, 13]);

        carried.emit(EmitTo::First(2));
        assert_eq!(carried.take_emitted(), Some(vec![10, 11]));
        assert_eq!(carried.take_emitted(), None);
        carried.emit(EmitTo::All);
        assert_eq!(carried.take_emitted(), Some(vec![12, 13, 14]));

        // Without carried hashes nothing is kept
        let mut own = CarriedHashes::default();
        own.record(0, 2, &[0, 1], &[1, 2]);
        own.emit(EmitTo::All);
        assert_eq!(own.take_emitted(), None);
    }

    mod group_values {
        use std::sync::Arc;

        use arrow::array::{Int64Array, StringArray, StringViewArray};
        use arrow::datatypes::{DataType, Field, Int64Type, Schema};
        use datafusion_physical_expr::binary_map::OutputType;

        use super::*;
        use crate::aggregates::group_values::GroupValues;
        use crate::aggregates::group_values::multi_group_by::GroupValuesColumn;
        use crate::aggregates::group_values::single_group_by::{
            bytes::GroupValuesBytes, bytes_view::GroupValuesBytesView,
            primitive::GroupValuesPrimitive,
        };

        fn hash(cols: &[ArrayRef]) -> Vec<u64> {
            let mut hashes = vec![0; cols[0].len()];
            create_hashes(cols, &GroupHash::STATE, &mut hashes).unwrap();
            hashes
        }

        fn keys(rows: usize) -> impl Iterator<Item = Option<String>> {
            (0..rows).map(|row| {
                (row % 9 != 4).then(|| format!("a group key past the inline length {}", row % 13))
            })
        }

        /// Interning with carried hashes groups the rows of `cols` as
        /// interning with the implementation's own hashes does, rows that
        /// come without hashes afterwards find the same groups, the hashes
        /// leave with their groups, and an implementation holding groups of
        /// its own hashes refuses carried ones.
        fn check(make: fn() -> Box<dyn GroupValues>, cols: Vec<ArrayRef>) -> Result<()> {
            let carried = hash(&cols);
            let mut own = make();
            let mut values = make();
            let (mut expected, mut groups) = (vec![], vec![]);
            own.intern(&cols, &mut expected)?;
            values.intern_hashed(&cols, &carried, &mut groups)?;
            assert_eq!(groups, expected);
            values.intern(&cols, &mut groups)?;
            assert_eq!(groups, expected);
            assert_eq!(values.len(), own.len());

            let first = values.emit(EmitTo::First(2))?;
            assert_eq!(values.emit_hashes(), Some(hash(&first)));
            assert_eq!(values.emit_hashes(), None);
            // The groups left are found again under their hashes
            values.intern_hashed(&cols, &carried, &mut groups)?;
            let rest = values.emit(EmitTo::All)?;
            assert_eq!(rest[0].len(), own.len());
            assert_eq!(values.emit_hashes(), Some(hash(&rest)));

            assert!(own.intern_hashed(&cols, &carried, &mut groups).is_err());
            assert_eq!(own.emit_hashes(), None);
            Ok(())
        }

        #[test]
        fn carried_hashes_in_every_group_values() -> Result<()> {
            let rows = 200;
            let view: ArrayRef = Arc::new(StringViewArray::from_iter(keys(rows)));
            check(
                || Box::new(GroupValuesBytesView::new(OutputType::Utf8View)),
                vec![Arc::clone(&view)],
            )?;
            check(
                || Box::new(GroupValuesBytes::<i32>::new(OutputType::Utf8)),
                vec![Arc::new(StringArray::from_iter(keys(rows)))],
            )?;
            let ints: ArrayRef = Arc::new(Int64Array::from_iter(
                (0..rows as i64).map(|row| (row % 11 != 5).then_some(row % 17)),
            ));
            check(
                || Box::new(GroupValuesPrimitive::<Int64Type>::new(DataType::Int64)),
                vec![Arc::clone(&ints)],
            )?;
            fn schema() -> Arc<Schema> {
                Arc::new(Schema::new(vec![
                    Field::new("n", DataType::Int64, true),
                    Field::new("s", DataType::Utf8View, true),
                ]))
            }
            check(
                || Box::new(GroupValuesColumn::<false>::try_new(schema()).unwrap()),
                vec![Arc::clone(&ints), Arc::clone(&view)],
            )?;
            check(
                || Box::new(GroupValuesColumn::<true>::try_new(schema()).unwrap()),
                vec![ints, view],
            )
        }
    }
}
