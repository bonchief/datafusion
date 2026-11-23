// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::view_dedupe::ViewDedupe;
use crate::aggregates::group_values::GroupValues;
use arrow::array::{Array, ArrayRef};
use datafusion_expr::{EmitTo, GroupSelection};
use datafusion_physical_expr::binary_map::OutputType;
use datafusion_physical_expr_common::binary_view_map::{
    ArrowBytesViewMap, INITIAL_MAP_CAPACITY,
};
use std::mem::size_of;

/// A [`GroupValues`] storing single column of Utf8View/BinaryView values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
///
/// Rows of a batch that repeat a view are looked up once per distinct view
/// ([`ViewDedupe`]), when enough of them do.
pub struct GroupValuesBytesView {
    /// Map string/binary values to group index
    map: ArrowBytesViewMap<usize>,
    /// The total number of groups so far (used to assign group_index)
    num_groups: usize,
    /// The per-batch memo from view to first row
    dedupe: ViewDedupe,
    /// The group of each distinct view of the batch being interned
    distinct_groups: Vec<usize>,
}

impl GroupValuesBytesView {
    pub fn new(output_type: OutputType) -> Self {
        Self {
            // One map holds every group value for the whole query, so it is
            // worth pre-allocating the hash table.
            map: ArrowBytesViewMap::with_capacity(output_type, INITIAL_MAP_CAPACITY),
            num_groups: 0,
            dedupe: ViewDedupe::default(),
            distinct_groups: Vec::new(),
        }
    }

    /// Looks up every row of `values` in the map, in row order, assigning the
    /// next group index to each new value.
    fn intern_rows(&mut self, values: &ArrayRef, groups: &mut Vec<usize>) {
        let num_groups = &mut self.num_groups;
        self.map.insert_if_new(
            values,
            // called for each new group
            |_value| {
                // assign new group index on each insert
                let group_idx = *num_groups;
                *num_groups += 1;
                group_idx
            },
            // called for each group
            |group_idx| {
                groups.push(group_idx);
            },
        );
    }
}

impl GroupValues for GroupValuesBytesView {
    fn intern(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
    ) -> datafusion_common::Result<()> {
        assert_eq!(cols.len(), 1);

        // look up / add entries in the table
        let arr = &cols[0];

        groups.clear();
        match self.dedupe.distinct(arr)? {
            // Distinct views come in order of first appearance, so new
            // groups are numbered exactly as a row by row lookup numbers them.
            Some(distinct) => {
                let mut distinct_groups = std::mem::take(&mut self.distinct_groups);
                distinct_groups.clear();
                self.intern_rows(&distinct, &mut distinct_groups);
                groups.extend(
                    self.dedupe
                        .rows()
                        .iter()
                        .map(|&first| distinct_groups[first as usize]),
                );
                self.distinct_groups = distinct_groups;
            }
            None => self.intern_rows(arr, groups),
        }

        // ensure we assigned a group to for each row
        assert_eq!(groups.len(), arr.len());
        Ok(())
    }

    fn reserve_groups(&mut self, capacity: usize) {
        self.map.reserve_groups(capacity);
    }

    fn emit_with_capacity(
        &mut self,
        capacity: usize,
    ) -> datafusion_common::Result<Vec<ArrayRef>> {
        self.num_groups = 0;
        Ok(vec![self.map.take_with_capacity(capacity)])
    }

    fn size(&self) -> usize {
        self.map.size()
            + self.dedupe.size()
            + self.distinct_groups.capacity() * size_of::<usize>()
            + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.num_groups == 0
    }

    fn len(&self) -> usize {
        self.num_groups
    }

    fn emit(&mut self, emit_to: EmitTo) -> datafusion_common::Result<Vec<ArrayRef>> {
        // Reset the map to default, and convert it into a single array
        let map_contents = self.map.take().into_state();

        let group_values = match emit_to {
            EmitTo::All => {
                self.num_groups -= map_contents.len();
                map_contents
            }
            EmitTo::First(n) if n == self.len() => {
                self.num_groups -= map_contents.len();
                map_contents
            }
            EmitTo::First(n) => {
                // if we only wanted to take the first n, insert the rest back
                // into the map we could potentially avoid this reallocation, at
                // the expense of much more complex code.
                // see https://github.com/apache/datafusion/issues/9195
                let emit_group_values = map_contents.slice(0, n);
                let remaining_group_values =
                    map_contents.slice(n, map_contents.len() - n);

                self.num_groups = 0;
                // The map's own values are distinct: no memo.
                let mut group_indexes = Vec::with_capacity(remaining_group_values.len());
                self.intern_rows(&remaining_group_values, &mut group_indexes);

                // Verify that the group indexes were assigned in the correct order
                assert_eq!(0, group_indexes[0]);

                emit_group_values
            }
        };

        Ok(vec![group_values])
    }

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> datafusion_common::Result<Vec<ArrayRef>> {
        selection.validate_num_groups(self.len())?;
        Ok(vec![self.map.keys(selection.iter())?])
    }

    fn supports_values_preserving(&self) -> bool {
        true
    }

    fn clear_shrink(&mut self, _num_rows: usize) {
        // Callers use this to hand memory back before spilling or sorting, so
        // release the map's allocations rather than restoring the warm up
        // capacity that `take` keeps for the emit path.
        self.map.clear_and_release();
        self.dedupe.release();
        self.distinct_groups = Vec::new();
        self.num_groups = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{
        AsArray, BinaryViewArray, BinaryViewBuilder, StringViewArray, UInt32Array,
    };
    use arrow::compute::{concat, take};
    use datafusion_common::HashMap;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Group ids by first appearance, from a plain hash map of the values:
    /// the reference every lookup path must agree with.
    #[derive(Default)]
    struct Reference {
        groups: HashMap<Option<Vec<u8>>, usize>,
    }

    impl Reference {
        fn intern(&mut self, values: &ArrayRef) -> Vec<usize> {
            let values = values.as_binary_view();
            (0..values.len())
                .map(|row| {
                    let key = values.is_valid(row).then(|| values.value(row).to_vec());
                    let next = self.groups.len();
                    *self.groups.entry(key).or_insert(next)
                })
                .collect()
        }
    }

    /// A long value (out of line in its view) or a short one (inlined).
    fn value(index: usize, long: bool) -> Vec<u8> {
        if long {
            format!("https://example.com/a/long/path/{index}").into_bytes()
        } else {
            format!("v{index}").into_bytes()
        }
    }

    /// `rows` rows drawn from `distinct` values that share one view per
    /// value, as a dictionary decoded to views gives them; every `null_every`
    /// row is null (0 for none).
    fn shared_views(
        rng: &mut StdRng,
        rows: usize,
        distinct: usize,
        null_every: usize,
    ) -> ArrayRef {
        let dictionary: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            (0..distinct).map(|index| value(index, index % 3 != 0)),
        ));
        let keys = UInt32Array::from_iter((0..rows).map(|row| {
            (null_every == 0 || row % null_every != 0)
                .then(|| rng.random_range(0..distinct as u32))
        }));
        take(dictionary.as_ref(), &keys, None).unwrap()
    }

    /// `rows` rows of the same `distinct` values, each row's bytes written
    /// again, so equal values never share a view.
    fn copied_views(rng: &mut StdRng, rows: usize, distinct: usize) -> ArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for _ in 0..rows {
            let index = rng.random_range(0..distinct);
            builder.append_value(value(index, index % 3 != 0));
        }
        Arc::new(builder.finish())
    }

    /// Interns each batch with the memo, row by row in the map alone, and in
    /// the reference, and checks all three give the same group ids; returns
    /// how many batches the memo deduplicated.
    fn check(batches: &[ArrayRef]) -> usize {
        let mut memo = GroupValuesBytesView::new(OutputType::BinaryView);
        let mut plain = GroupValuesBytesView::new(OutputType::BinaryView);
        let mut reference = Reference::default();
        let (mut memo_groups, mut plain_groups) = (vec![], vec![]);
        for batch in batches {
            memo.intern(std::slice::from_ref(batch), &mut memo_groups)
                .unwrap();
            plain_groups.clear();
            plain.intern_rows(batch, &mut plain_groups);
            let expected = reference.intern(batch);
            assert_eq!(memo_groups, expected);
            assert_eq!(plain_groups, expected);
            assert_eq!(memo.len(), reference.groups.len());
        }
        let memo_values = memo.emit(EmitTo::All).unwrap();
        let plain_values = plain.emit(EmitTo::All).unwrap();
        assert_eq!(memo_values[0].as_ref(), plain_values[0].as_ref());
        memo.dedupe.used()
    }

    #[test]
    fn view_dedupe_matches_row_by_row_lookup() {
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let mut batches = vec![
            // all repeats, without and with nulls
            shared_views(&mut rng, 8192, 1, 0),
            shared_views(&mut rng, 8192, 1, 7),
            // a few hundred values, about two thirds of them out of line
            shared_views(&mut rng, 8192, 300, 0),
            shared_views(&mut rng, 8192, 3000, 11),
            // every row null
            shared_views(&mut rng, 1024, 5, 1),
        ];
        // a slice starting inside a batch, with its nulls offset too
        batches.push(shared_views(&mut rng, 9000, 200, 5).slice(123, 8192));
        // equal bytes in two buffers: views differ, values do not
        let left = shared_views(&mut rng, 4096, 100, 0);
        let right = copied_views(&mut rng, 4096, 100);
        batches.push(concat(&[left.as_ref(), right.as_ref()]).unwrap());
        // shared views mixed with copies of the same values, interleaved
        let mixed =
            concat(&[shared_views(&mut rng, 4096, 50, 0).as_ref(), right.as_ref()])
                .unwrap();
        let order = UInt32Array::from_iter_values(
            (0..8192u32).map(|row| (row % 2) * 4096 + row / 2),
        );
        batches.push(take(mixed.as_ref(), &order, None).unwrap());
        // short batches pass straight through
        batches.push(shared_views(&mut rng, 100, 3, 0));
        let used = check(&batches);
        assert!(used >= 6, "the memo deduplicated only {used} batches");
    }

    #[test]
    fn view_dedupe_passes_through_batches_without_repeats() {
        let mut rng = StdRng::seed_from_u64(7);
        // unique values, then equal values that never share a view
        let unique: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            (0..8192).map(|index| value(index, true)),
        ));
        let batches = vec![
            Arc::clone(&unique),
            copied_views(&mut rng, 8192, 20_000),
            copied_views(&mut rng, 8192, 20_000),
        ];
        assert_eq!(check(&batches), 0);
    }

    #[test]
    fn view_dedupe_backs_off_and_comes_back() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut batches = vec![];
        // a run without shared views backs the memo off ...
        batches.extend((0..20).map(|_| copied_views(&mut rng, 8192, 20_000)));
        // ... and a run of shared views brings it back
        batches.extend((0..140).map(|_| shared_views(&mut rng, 8192, 500, 13)));
        let used = check(&batches);
        assert!(
            used > 70,
            "the memo deduplicated {used} of 140 batches it should take"
        );
    }

    #[test]
    fn view_dedupe_keeps_group_ids_across_emits() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut memo = GroupValuesBytesView::new(OutputType::BinaryView);
        let mut reference = GroupValuesBytesView::new(OutputType::BinaryView);
        let mut groups = vec![];
        let mut expected = vec![];
        for round in 0..6 {
            let batch = shared_views(&mut rng, 8192, 400, 9);
            memo.intern(std::slice::from_ref(&batch), &mut groups)
                .unwrap();
            expected.clear();
            reference.intern_rows(&batch, &mut expected);
            assert_eq!(groups, expected);
            let emit = if round % 2 == 0 {
                EmitTo::First(memo.len() / 2)
            } else {
                EmitTo::All
            };
            let emitted = memo.emit(emit).unwrap();
            assert_eq!(
                emitted[0].as_ref(),
                reference.emit(emit).unwrap()[0].as_ref()
            );
        }
        assert!(memo.dedupe.used() > 0);
    }

    #[test]
    fn view_dedupe_on_string_views() {
        let mut rng = StdRng::seed_from_u64(5);
        let dictionary: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..64).map(|index| format!("a string value that is long {index}")),
        ));
        let keys =
            UInt32Array::from_iter_values((0..8192).map(|_| rng.random_range(0..64u32)));
        let batch = take(dictionary.as_ref(), &keys, None).unwrap();
        let mut memo = GroupValuesBytesView::new(OutputType::Utf8View);
        let mut plain = GroupValuesBytesView::new(OutputType::Utf8View);
        let (mut groups, mut expected) = (vec![], vec![]);
        memo.intern(std::slice::from_ref(&batch), &mut groups)
            .unwrap();
        plain.intern_rows(&batch, &mut expected);
        assert_eq!(groups, expected);
        assert_eq!(memo.dedupe.used(), 1);
    }

    /// `clear_shrink` is how the aggregate stream hands memory back before it
    /// spills and before the spilled batch is sorted, so the memory it releases
    /// has to actually show up in the size it reports afterwards.
    #[test]
    fn clear_shrink_releases_the_reported_memory() {
        let mut group_values = GroupValuesBytesView::new(OutputType::Utf8View);
        let empty = size_of::<GroupValuesBytesView>();

        // The hash table is pre-allocated at construction, so the map is
        // already well above its own struct size before a single row is
        // interned.
        let warm_size = group_values.size();
        assert!(
            warm_size > empty + INITIAL_MAP_CAPACITY,
            "expected the pre-allocated map to report more than {} bytes, got {warm_size}",
            empty + INITIAL_MAP_CAPACITY
        );

        let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..1_000).map(|i| format!("group value number {i}")),
        ));
        let mut groups = vec![];
        group_values
            .intern(&[Arc::clone(&values)], &mut groups)
            .unwrap();
        let populated_size = group_values.size();
        assert!(populated_size > warm_size);

        group_values.clear_shrink(0);

        // Everything the map held is gone: what remains is the struct itself.
        let released_size = group_values.size();
        assert!(
            released_size < empty + 128,
            "expected clear_shrink to release the map, got {released_size} with a struct size of {empty}"
        );
        assert!(
            released_size * 10 < populated_size,
            "expected {released_size} to be far below {populated_size}"
        );

        // The map still works, and warms back up on the next emit.
        group_values.intern(&[values], &mut groups).unwrap();
        assert!(group_values.size() > released_size);
        group_values.emit(EmitTo::All).unwrap();
        assert!(group_values.size() > empty + INITIAL_MAP_CAPACITY);
    }

    #[test]
    fn clear_shrink_should_reset_len() {
        let mut group_values = GroupValuesBytesView::new(OutputType::Utf8View);
        let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..10).map(|i| format!("group value number {i}")),
        ));
        let mut groups = vec![];
        group_values
            .intern(&[Arc::clone(&values)], &mut groups)
            .unwrap();
        assert_eq!(group_values.len(), 10);

        group_values.clear_shrink(0);
        assert_eq!(group_values.len(), 0);

        // The first new distinct value after clearing should get group index 0
        let values: ArrayRef = Arc::new(StringViewArray::from_iter_values([
            "new value",
            "another value",
        ]));
        group_values.intern(&[values], &mut groups).unwrap();
        assert_eq!(groups, vec![0, 1]);
        assert_eq!(group_values.len(), 2);
    }
}
