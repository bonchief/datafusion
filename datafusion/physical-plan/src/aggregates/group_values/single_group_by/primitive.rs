// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::key_map::KeyMap;
use crate::aggregates::group_values::GroupValues;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray, cast::AsArray,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::float_semantics::normalize_float;
use datafusion_common::hash_utils::RandomState;
use datafusion_expr::{EmitTo, GroupSelection};
use half::f16;
#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::sync::Arc;

/// A trait to allow hashing of floating point numbers
pub trait HashValue {
    fn hash(&self, state: &RandomState) -> u64;

    /// Return a canonical representative whose bit pattern is identical for
    /// all values that should be grouped together. Default is the identity;
    /// floats override this with [`normalize_float`] so the bit-equal
    /// `is_eq` check used during insertion follows SQL equality.
    #[inline]
    fn normalize_key(self) -> Self
    where
        Self: Sized,
    {
        self
    }
}

macro_rules! hash_integer {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self)
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }
        })+
    };
}
hash_integer!(i8, i16, i32, i64, i128, i256);
hash_integer!(u8, u16, u32, u64);
hash_integer!(IntervalDayTime, IntervalMonthDayNano);

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(datafusion_common::float_semantics::hash_input(*self).to_bits())
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }

            #[inline]
            fn normalize_key(self) -> Self {
                normalize_float(self)
            }
        })+
    };
}

hash_float!(f16, f32, f64);

/// A [`GroupValues`] storing a single column of primitive values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format. Group ids come from a [`KeyMap`], which keeps each
/// key in its table entry and probes in stages once the table outgrows L2.
pub struct GroupValuesPrimitive<T: ArrowPrimitiveType>
where
    T::Native: HashValue,
{
    /// The data type of the output array
    data_type: DataType,
    /// The group id of each value, and the value of each group id
    keys: KeyMap<T::Native>,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            keys: KeyMap::new(128, crate::aggregates::AGGREGATION_HASH_SEED),
        }
    }
}

/// The values of `keys` as an array, null at `null_idx`.
fn build_primitive<T: ArrowPrimitiveType>(
    values: Vec<T::Native>,
    null_idx: Option<usize>,
) -> PrimitiveArray<T> {
    let nulls = null_idx.map(|null_idx| {
        let mut buffer = NullBufferBuilder::new(values.len());
        buffer.append_n_non_nulls(null_idx);
        buffer.append_null();
        buffer.append_n_non_nulls(values.len() - null_idx - 1);
        // NOTE: The inner builder must be constructed as there is at least one null
        buffer.finish().unwrap()
    });
    PrimitiveArray::<T>::new(values.into(), nulls)
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        let array = cols[0].as_primitive::<T>();
        self.keys.intern(array.values(), array.nulls(), groups);
        Ok(())
    }

    fn reserve_groups(&mut self, capacity: usize) {
        self.keys.reserve_groups(capacity);
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.keys.size()
    }

    fn is_empty(&self) -> bool {
        self.keys.len() == 0
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let (values, null_group) = self.keys.emit(emit_to);
        let array = build_primitive::<T>(values, null_group);
        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        let stored = self.keys.values();
        selection.validate_num_groups(stored.len())?;
        let values: Vec<T::Native> =
            selection.iter().map(|index| stored[index]).collect();
        let nulls = if let Some(null_group) = self.keys.null_group() {
            let mut nulls = NullBufferBuilder::new(values.len());
            for index in selection.iter() {
                if index == null_group {
                    nulls.append_null();
                } else {
                    nulls.append_non_null();
                }
            }
            nulls.finish()
        } else {
            None
        };
        let array = PrimitiveArray::<T>::new(values.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(vec![Arc::new(array)])
    }

    fn supports_values_preserving(&self) -> bool {
        true
    }

    fn shrink_to_fit(&mut self) {
        self.keys.shrink_to_fit();
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.keys.clear_shrink(num_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::Int32Type;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::DataType;
    use datafusion_common::utils::proxy::VecAllocExt;
    use datafusion_expr::EmitTo;
    use std::mem::size_of;
    use std::sync::Arc;

    /// Mirror of the `EmitTo::take_needed` regression test, applied to the
    /// concrete `GroupValuesPrimitive` accumulator.
    ///
    /// When `n` is small, the old `split_off(n) + swap` pattern used inside
    /// `emit(EmitTo::First(n))` left `self.values` with a small fresh allocation
    /// and returned the emitted prefix carrying the original large backing.
    ///
    /// With `split_vec_min_alloc` and `n * 2 <= len`, the drain branch is taken:
    /// the emitted prefix gets a compact allocation and `self.values` retains the
    /// original large one.
    #[test]
    fn emit_first_small_n_allocates_minimally() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);

        // Intern 20 distinct values; `new()` pre-allocates capacity 128 for `values`.
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..20i32));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;
        let capacity_before = gv.keys.values.capacity(); // 128

        // n=4, n*2=8 <= len=20 -> drain branch
        let emitted = gv.emit(EmitTo::First(4))?;

        assert_eq!(emitted[0].len(), 4);

        // `self.values` must retain its original large allocation.
        // Old split_off+swap left it with a fresh small allocation (~16).
        assert_eq!(
            gv.keys.values.capacity(),
            capacity_before,
            "self.values capacity {} should equal original {} after small First(n) emit",
            gv.keys.values.capacity(),
            capacity_before,
        );

        Ok(())
    }

    #[test]
    fn test_exact_hash_table_allocation_accounting() {
        let mut group_values = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);
        let self_size = size_of::<GroupValuesPrimitive<Int32Type>>();

        // 1. Initial state: the constructor calls HashTable::with_capacity(128),
        //    so the map already has a nonzero retained allocation. Capture it
        //    and verify that size() accounts for it correctly.
        let initial_map_bytes = group_values.keys.map.allocation_size();
        assert!(group_values.keys.map.capacity() >= 128);
        assert!(initial_map_bytes > 0);
        assert_eq!(
            group_values.size(),
            self_size + group_values.keys.values.allocated_size() + initial_map_bytes,
        );

        // 2. Insert keys and observe table allocation growth
        let num_distinct = 500_i32;
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..num_distinct));
        let mut groups = vec![];
        group_values.intern(&[arr], &mut groups).unwrap();
        assert_eq!(groups.len(), num_distinct as usize);

        let table_bytes = group_values.keys.map.allocation_size();
        assert!(group_values.keys.map.capacity() >= num_distinct as usize);
        assert!(table_bytes > 0);

        // Hashbrown's layout contains entry slots, control bytes, and trailing group bytes.
        // It must strictly exceed the naive entry-only size
        let entry_only_bytes =
            group_values.keys.map.capacity() * size_of::<(usize, i32)>();
        assert!(table_bytes > entry_only_bytes);

        assert_eq!(
            group_values.size(),
            self_size + group_values.keys.values.allocated_size() + table_bytes,
        );

        // 3. Partial emit (EmitTo::First)
        let emit_count = 200;
        let emitted = group_values.emit(EmitTo::First(emit_count)).unwrap();
        assert_eq!(emitted[0].len(), emit_count);
        assert_eq!(group_values.len(), (num_distinct as usize) - emit_count);

        // Retained capacity is still charged until the table releases it
        assert_eq!(group_values.keys.map.allocation_size(), table_bytes);
        assert_eq!(
            group_values.size(),
            self_size + group_values.keys.values.allocated_size() + table_bytes,
        );

        // 4. Emit all (EmitTo::All)
        let remaining_count = group_values.len();
        let emitted_all = group_values.emit(EmitTo::All).unwrap();
        assert_eq!(emitted_all[0].len(), remaining_count);
        assert_eq!(group_values.len(), 0);

        // Clear preserves table capacity
        assert_eq!(group_values.keys.map.allocation_size(), table_bytes);

        // 5. clear_shrink: releases/shrinks map capacity
        group_values.clear_shrink(16);
        let shrunken_table_bytes = group_values.keys.map.allocation_size();
        assert!(shrunken_table_bytes < table_bytes);
        assert_eq!(
            group_values.size(),
            self_size + group_values.keys.values.allocated_size() + shrunken_table_bytes,
        );

        // 6. Reinsert new distinct values into the shrunken table
        let new_arr: ArrayRef = Arc::new(Int32Array::from_iter_values(
            num_distinct..(num_distinct * 2),
        ));
        group_values.intern(&[new_arr], &mut groups).unwrap();
        assert_eq!(groups.len(), num_distinct as usize);
        assert!(group_values.keys.map.allocation_size() > shrunken_table_bytes);
        assert_eq!(
            group_values.size(),
            self_size
                + group_values.keys.values.allocated_size()
                + group_values.keys.map.allocation_size(),
        );
    }

    #[test]
    fn clear_shrink_reset_null_group() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);

        // Intern some values including a null
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(2)]));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;

        assert_eq!(groups.len(), 3);

        let null_group = groups[1];
        assert_eq!(null_group, 1);

        // Clear and shrink
        gv.clear_shrink(0);

        let arr: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;
        assert_eq!(groups.len(), 1);

        let new_null_group = groups[0];
        assert_eq!(new_null_group, 0);

        Ok(())
    }
}
