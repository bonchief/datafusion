// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`GroupValuesPacked`]: several fixed-width group keys packed into one
//! integer, as ClickHouse's `keys64`/`keys128` methods do.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, BooleanBufferBuilder,
    PrimitiveArray,
};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::ArrowNativeType;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Float16Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, Schema, SchemaRef, Time32MillisecondType,
    Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimeUnit,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::{EmitTo, GroupSelection};
use half::f16;

use super::GroupValues;
use super::single_group_by::key_map::{GroupKey, KeyMap};
use super::single_group_by::primitive::HashValue;

/// The group values of two or more fixed-width columns whose values and
/// null bits fit together in 64 or 128 bits, packed into one `u64` or
/// [`Key128`] per row and interned as a single key by a [`KeyMap`].
///
/// One hash and one comparison of the packed key per row replace a hash and
/// a comparison per column, and the key lives in the table entry, so a probe
/// reads no column. Beyond L2 the [`KeyMap`] probes in stages.
///
/// # Layout
///
/// Column `i` takes `width(i)` bits at offset `shift(i)`, in schema order
/// from bit 0, and a nullable column takes one more bit, after all values,
/// that is set where its value is null. A null value's bits are zero, so
/// all nulls of a column pack alike whatever their slot holds. Signed
/// values are packed as their two's complement in their own width and
/// sign-extended back. Floats are packed as the bits of
/// [`HashValue::normalize_key`], which folds `-0.0` into `0.0` and keeps
/// every NaN payload, so NaNs with different payloads are different groups.
/// They group and emit exactly as [`GroupValuesColumn`]'s float columns,
/// which store the same normalized values.
///
/// # Semantics
///
/// Two rows pack alike exactly when every column is equal or null in both,
/// which is [`GroupValuesColumn`]'s grouping. A column whose field is not
/// nullable takes no null bit and its validity is ignored, as
/// [`GroupValuesColumn`]'s non-nullable builders ignore it. Group ids are
/// assigned in the order rows first appear, which is the order of
/// `GroupValuesColumn<true>` and, but for 64-bit hash collisions, of the
/// vectorized `GroupValuesColumn<false>`.
///
/// [`GroupValuesColumn`]: super::multi_group_by::GroupValuesColumn
pub struct GroupValuesPacked<K: PackedKey> {
    schema: SchemaRef,
    columns: Vec<Packer>,
    keys: KeyMap<K>,
    /// The packed keys of the batch being interned
    packed: Vec<K>,
}

/// A packed key: [`u64`] or [`Key128`].
pub trait PackedKey: GroupKey {
    /// Bits a key holds.
    const BITS: u32;

    /// Sets `bits`, which fit in 64 bits above `shift`, from bit `shift`.
    fn set(&mut self, bits: u64, shift: u32);

    /// The 64 bits from bit `shift`.
    fn get(self, shift: u32) -> u64;
}

impl PackedKey for u64 {
    const BITS: u32 = 64;

    #[inline(always)]
    fn set(&mut self, bits: u64, shift: u32) {
        *self |= bits << shift;
    }

    #[inline(always)]
    fn get(self, shift: u32) -> u64 {
        self >> shift
    }
}

/// A 128-bit packed key, kept as two words so the table entry it is part of
/// stays 8-byte aligned: `(usize, Key128)` is 24 bytes, `(usize, u128)` 32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Key128([u64; 2]);

impl Key128 {
    fn wide(self) -> u128 {
        (u128::from(self.0[1]) << 64) | u128::from(self.0[0])
    }
}

impl PackedKey for Key128 {
    const BITS: u32 = 128;

    #[inline(always)]
    fn set(&mut self, bits: u64, shift: u32) {
        let wide = self.wide() | (u128::from(bits) << shift);
        self.0 = [wide as u64, (wide >> 64) as u64];
    }

    #[inline(always)]
    fn get(self, shift: u32) -> u64 {
        (self.wide() >> shift) as u64
    }
}

impl HashValue for Key128 {
    /// Hashes both words of the key, as one `u128`, with the table's seed.
    #[cfg(not(feature = "force_hash_collisions"))]
    #[inline(always)]
    fn hash(&self, state: &RandomState) -> u64 {
        use std::hash::BuildHasher;
        state.hash_one(self.wide())
    }

    #[cfg(feature = "force_hash_collisions")]
    fn hash(&self, _state: &RandomState) -> u64 {
        0
    }
}

impl GroupKey for Key128 {
    #[inline(always)]
    fn same(self, other: Self) -> bool {
        self == other
    }
}

/// How one column is packed: its kind, where its value bits go and where its
/// null bit, if it is nullable, goes.
struct Packer {
    data_type: DataType,
    kind: Kind,
    shift: u32,
    null_bit: Option<u32>,
}

/// The fixed-width types that pack, by how their values become bits.
#[derive(Clone, Copy)]
enum Kind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    Bool,
}

impl Kind {
    fn of(data_type: &DataType) -> Option<Self> {
        Some(match data_type {
            DataType::Int8 => Self::I8,
            DataType::Int16 => Self::I16,
            DataType::Int32
            | DataType::Date32
            | DataType::Time32(TimeUnit::Second | TimeUnit::Millisecond) => Self::I32,
            DataType::Int64
            | DataType::Date64
            | DataType::Time64(TimeUnit::Microsecond | TimeUnit::Nanosecond)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_) => Self::I64,
            DataType::UInt8 => Self::U8,
            DataType::UInt16 => Self::U16,
            DataType::UInt32 => Self::U32,
            DataType::UInt64 => Self::U64,
            DataType::Float16 => Self::F16,
            DataType::Float32 => Self::F32,
            DataType::Float64 => Self::F64,
            DataType::Boolean => Self::Bool,
            _ => return None,
        })
    }

    fn width(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }
}

/// A primitive native type's value as the low bits of a `u64`, and back.
trait Bits: Copy + Default + HashValue {
    /// The value's bits, zero-extended from its own width.
    fn to_bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;
}

macro_rules! int_bits {
    ($($t:ty => $u:ty),+) => {
        $(impl Bits for $t {
            #[inline(always)]
            fn to_bits(self) -> u64 {
                self as $u as u64
            }
            #[inline(always)]
            fn from_bits(bits: u64) -> Self {
                bits as $u as $t
            }
        })+
    };
}
int_bits!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, u8 => u8, u16 => u16, u32 => u32, u64 => u64);

macro_rules! float_bits {
    ($($t:ty => $u:ty),+) => {
        $(impl Bits for $t {
            #[inline(always)]
            fn to_bits(self) -> u64 {
                self.normalize_key().to_bits() as u64
            }
            #[inline(always)]
            fn from_bits(bits: u64) -> Self {
                <$t>::from_bits(bits as $u)
            }
        })+
    };
}
float_bits!(f16 => u16, f32 => u32, f64 => u64);

impl Packer {
    /// ORs this column's bits of each row of `array` into `keys`: its value
    /// bits, zero for a null row, and its null bit, set for a null row.
    fn pack<K: PackedKey>(&self, array: &ArrayRef, keys: &mut [K]) {
        let nulls = match self.null_bit {
            Some(_) => array.logical_nulls().filter(|nulls| nulls.null_count() > 0),
            None => None,
        };
        match self.kind {
            Kind::I8 => self.pack_values::<i8, K>(array, nulls, keys),
            Kind::I16 => self.pack_values::<i16, K>(array, nulls, keys),
            Kind::I32 => self.pack_values::<i32, K>(array, nulls, keys),
            Kind::I64 => self.pack_values::<i64, K>(array, nulls, keys),
            Kind::U8 => self.pack_values::<u8, K>(array, nulls, keys),
            Kind::U16 => self.pack_values::<u16, K>(array, nulls, keys),
            Kind::U32 => self.pack_values::<u32, K>(array, nulls, keys),
            Kind::U64 => self.pack_values::<u64, K>(array, nulls, keys),
            Kind::F16 => self.pack_values::<f16, K>(array, nulls, keys),
            Kind::F32 => self.pack_values::<f32, K>(array, nulls, keys),
            Kind::F64 => self.pack_values::<f64, K>(array, nulls, keys),
            Kind::Bool => {
                let values = array.as_boolean().values();
                self.pack_bits(values.iter().map(u64::from), nulls, keys);
            }
        }
    }

    /// The bits of a primitive array whose native type is `N`, whatever its
    /// logical type, so a timestamp column reads as `i64`.
    fn pack_values<N: Bits + ArrowNativeType, K: PackedKey>(
        &self,
        array: &ArrayRef,
        nulls: Option<NullBuffer>,
        keys: &mut [K],
    ) {
        let data = array.to_data();
        let values =
            ScalarBuffer::<N>::new(data.buffers()[0].clone(), data.offset(), data.len());
        self.pack_bits(values.iter().map(|value| value.to_bits()), nulls, keys);
    }

    /// ORs `bits`, one per row, into `keys`, zeroed with the null bit set
    /// where `nulls` marks a row null.
    #[inline(always)]
    fn pack_bits<K: PackedKey>(
        &self,
        bits: impl Iterator<Item = u64>,
        nulls: Option<NullBuffer>,
        keys: &mut [K],
    ) {
        match (nulls, self.null_bit) {
            (Some(nulls), Some(null_bit)) => {
                for ((key, bits), valid) in keys.iter_mut().zip(bits).zip(nulls.iter()) {
                    // All ones for a valid row, zero for a null one.
                    let keep = u64::from(valid).wrapping_neg();
                    key.set(bits & keep, self.shift);
                    key.set(u64::from(!valid), null_bit);
                }
            }
            _ => {
                for (key, bits) in keys.iter_mut().zip(bits) {
                    key.set(bits, self.shift);
                }
            }
        }
    }

    /// This column of `keys`, as an array.
    fn unpack<K: PackedKey>(&self, keys: &[K]) -> ArrayRef {
        let nulls = self.null_bit.map(|bit| {
            NullBuffer::from_iter(keys.iter().map(|key| key.get(bit) & 1 == 0))
        });
        let nulls = nulls.filter(|nulls| nulls.null_count() > 0);
        match self.kind {
            Kind::I8 => self.unpack_values::<Int8Type, K>(keys, nulls),
            Kind::I16 => self.unpack_values::<Int16Type, K>(keys, nulls),
            Kind::I32 => match self.data_type {
                DataType::Date32 => self.unpack_values::<Date32Type, K>(keys, nulls),
                DataType::Time32(TimeUnit::Second) => {
                    self.unpack_values::<Time32SecondType, K>(keys, nulls)
                }
                DataType::Time32(_) => {
                    self.unpack_values::<Time32MillisecondType, K>(keys, nulls)
                }
                _ => self.unpack_values::<Int32Type, K>(keys, nulls),
            },
            Kind::I64 => match self.data_type {
                DataType::Date64 => self.unpack_values::<Date64Type, K>(keys, nulls),
                DataType::Time64(TimeUnit::Microsecond) => {
                    self.unpack_values::<Time64MicrosecondType, K>(keys, nulls)
                }
                DataType::Time64(_) => {
                    self.unpack_values::<Time64NanosecondType, K>(keys, nulls)
                }
                DataType::Timestamp(TimeUnit::Second, _) => {
                    self.unpack_values::<TimestampSecondType, K>(keys, nulls)
                }
                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    self.unpack_values::<TimestampMillisecondType, K>(keys, nulls)
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    self.unpack_values::<TimestampMicrosecondType, K>(keys, nulls)
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    self.unpack_values::<TimestampNanosecondType, K>(keys, nulls)
                }
                DataType::Duration(TimeUnit::Second) => {
                    self.unpack_values::<DurationSecondType, K>(keys, nulls)
                }
                DataType::Duration(TimeUnit::Millisecond) => {
                    self.unpack_values::<DurationMillisecondType, K>(keys, nulls)
                }
                DataType::Duration(TimeUnit::Microsecond) => {
                    self.unpack_values::<DurationMicrosecondType, K>(keys, nulls)
                }
                DataType::Duration(TimeUnit::Nanosecond) => {
                    self.unpack_values::<DurationNanosecondType, K>(keys, nulls)
                }
                _ => self.unpack_values::<Int64Type, K>(keys, nulls),
            },
            Kind::U8 => self.unpack_values::<UInt8Type, K>(keys, nulls),
            Kind::U16 => self.unpack_values::<UInt16Type, K>(keys, nulls),
            Kind::U32 => self.unpack_values::<UInt32Type, K>(keys, nulls),
            Kind::U64 => self.unpack_values::<UInt64Type, K>(keys, nulls),
            Kind::F16 => self.unpack_values::<Float16Type, K>(keys, nulls),
            Kind::F32 => self.unpack_values::<Float32Type, K>(keys, nulls),
            Kind::F64 => self.unpack_values::<Float64Type, K>(keys, nulls),
            Kind::Bool => {
                let mut values = BooleanBufferBuilder::new(keys.len());
                for key in keys {
                    values.append(key.get(self.shift) & 1 == 1);
                }
                Arc::new(BooleanArray::new(values.finish(), nulls))
            }
        }
    }

    /// The values of type `T` at this column's bits of `keys`.
    fn unpack_values<T: ArrowPrimitiveType, K: PackedKey>(
        &self,
        keys: &[K],
        nulls: Option<NullBuffer>,
    ) -> ArrayRef
    where
        T::Native: Bits,
    {
        let values = keys
            .iter()
            .map(|key| T::Native::from_bits(key.get(self.shift)))
            .collect::<ScalarBuffer<T::Native>>();
        Arc::new(
            PrimitiveArray::<T>::new(values, nulls)
                .with_data_type(self.data_type.clone()),
        )
    }
}

/// The packers for `schema` and the bits they take, or `None` if a column
/// does not pack.
fn layout(schema: &Schema) -> Option<(Vec<Packer>, u32)> {
    let kinds = schema
        .fields()
        .iter()
        .map(|field| Kind::of(field.data_type()))
        .collect::<Option<Vec<_>>>()?;
    let values = kinds.iter().map(|kind| kind.width()).sum::<u32>();
    let mut shift = 0;
    let mut null_bit = values;
    let columns = schema
        .fields()
        .iter()
        .zip(kinds)
        .map(|(field, kind)| {
            let packer = Packer {
                data_type: field.data_type().clone(),
                kind,
                shift,
                null_bit: field.is_nullable().then(|| {
                    null_bit += 1;
                    null_bit - 1
                }),
            };
            shift += kind.width();
            packer
        })
        .collect();
    Some((columns, null_bit))
}

/// [`GroupValuesPacked`] for `schema`, if it has two or more columns that
/// all pack and fit, with their null bits, in 128 bits.
pub(crate) fn try_new_packed(schema: &SchemaRef) -> Option<Box<dyn GroupValues>> {
    if schema.fields().len() < 2 {
        return None;
    }
    let (columns, bits) = layout(schema)?;
    if bits <= u64::BITS {
        Some(Box::new(GroupValuesPacked::<u64>::new(
            Arc::clone(schema),
            columns,
        )))
    } else if bits <= Key128::BITS {
        Some(Box::new(GroupValuesPacked::<Key128>::new(
            Arc::clone(schema),
            columns,
        )))
    } else {
        None
    }
}

impl<K: PackedKey> GroupValuesPacked<K> {
    fn new(schema: SchemaRef, columns: Vec<Packer>) -> Self {
        Self {
            schema,
            columns,
            keys: KeyMap::new(0, crate::aggregates::AGGREGATION_HASH_SEED),
            packed: Vec::new(),
        }
    }

    fn unpack(&self, keys: &[K]) -> Vec<ArrayRef> {
        self.columns
            .iter()
            .map(|column| column.unpack(keys))
            .collect()
    }
}

impl<K: PackedKey> GroupValues for GroupValuesPacked<K> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        debug_assert_eq!(cols.len(), self.columns.len());
        self.packed.clear();
        self.packed.resize(cols[0].len(), K::default());
        for (column, array) in self.columns.iter().zip(cols) {
            column.pack(array, &mut self.packed);
        }
        self.keys.intern(&self.packed, None, groups);
        Ok(())
    }

    fn reserve_groups(&mut self, capacity: usize) {
        self.keys.reserve_groups(capacity);
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.keys.size() + self.packed.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.keys.len() == 0
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let (keys, _) = self.keys.emit(emit_to);
        Ok(self.unpack(&keys))
    }

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        let stored = self.keys.values();
        selection.validate_num_groups(stored.len())?;
        let keys = selection
            .iter()
            .map(|index| stored[index])
            .collect::<Vec<_>>();
        Ok(self.unpack(&keys))
    }

    fn supports_values_preserving(&self) -> bool {
        true
    }

    fn shrink_to_fit(&mut self) {
        self.keys.shrink_to_fit();
        self.packed = Vec::new();
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.keys.clear_shrink(num_rows);
        self.packed.clear();
        self.packed.shrink_to(num_rows);
    }
}

impl<K: PackedKey> std::fmt::Debug for GroupValuesPacked<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupValuesPacked")
            .field("schema", &self.schema)
            .field("bits", &K::BITS)
            .field("groups", &self.keys.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::group_values::multi_group_by::GroupValuesColumn;
    use arrow::array::{
        BooleanArray, Date32Array, DurationMicrosecondArray, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, TimestampNanosecondArray,
        UInt16Array, UInt64Array, new_null_array,
    };
    use arrow::datatypes::Field;
    use datafusion_common::HashMap;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        /// One of `choices`, or a fresh value from `fresh`, so values repeat.
        fn pick<T: Copy>(&mut self, choices: &[T], fresh: impl Fn(u64) -> T) -> T {
            let n = self.next();
            if n.is_multiple_of(3) {
                fresh(n)
            } else {
                choices[(n / 3) as usize % choices.len()]
            }
        }
    }

    /// A column of `rows` values of `data_type`, a fifth of them null when
    /// `nullable`, drawn from the type's edge values and a few fresh ones.
    fn column(
        data_type: &DataType,
        nullable: bool,
        rows: usize,
        rng: &mut Rng,
    ) -> ArrayRef {
        let valid = (0..rows)
            .map(|_| !nullable || !rng.next().is_multiple_of(5))
            .collect::<Vec<_>>();
        let nulls = Some(NullBuffer::from(valid));
        macro_rules! values {
            ($array:ty, $edges:expr, $fresh:expr) => {{
                let edges = $edges;
                let values = (0..rows)
                    .map(|_| rng.pick(&edges, $fresh))
                    .collect::<Vec<_>>();
                Arc::new(<$array>::new(values.into(), nulls)) as ArrayRef
            }};
        }
        let array = match data_type {
            DataType::Int8 => {
                values!(Int8Array, [i8::MIN, -1, 0, 1, i8::MAX], |n| n as i8)
            }
            DataType::Int16 => {
                values!(Int16Array, [i16::MIN, -1, 0, 1, i16::MAX], |n| (n % 7)
                    as i16)
            }
            DataType::Int32 => {
                values!(Int32Array, [i32::MIN, -1, 0, 1, i32::MAX], |n| (n % 50)
                    as i32
                    - 25)
            }
            DataType::Int64 => {
                values!(Int64Array, [i64::MIN, -1, 0, 1, i64::MAX], |n| (n % 50)
                    as i64)
            }
            DataType::UInt16 => {
                values!(UInt16Array, [0, 1, u16::MAX], |n| (n % 9) as u16)
            }
            DataType::UInt64 => values!(UInt64Array, [0, 1, u64::MAX], |n| n % 40),
            DataType::Float32 => values!(
                Float32Array,
                [
                    -0.0,
                    0.0,
                    f32::NAN,
                    -f32::NAN,
                    f32::from_bits(0x7fc0_0001),
                    f32::MIN,
                    f32::MAX
                ],
                |n| (n % 5) as f32
            ),
            DataType::Float64 => values!(
                Float64Array,
                [
                    -0.0,
                    0.0,
                    f64::NAN,
                    f64::from_bits(0xfff8_0000_0000_0123),
                    f64::INFINITY
                ],
                |n| (n % 5) as f64 - 2.0
            ),
            DataType::Boolean => Arc::new(BooleanArray::new(
                (0..rows).map(|_| rng.next().is_multiple_of(2)).collect(),
                nulls,
            )),
            DataType::Date32 => {
                values!(Date32Array, [i32::MIN, 0, 19_000], |n| (n % 30) as i32)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let array: TimestampNanosecondArray =
                    values!(TimestampNanosecondArray, [i64::MIN, 0, i64::MAX], |n| n
                        as i64
                        % 9)
                    .as_primitive::<TimestampNanosecondType>()
                    .clone();
                Arc::new(array.with_data_type(data_type.clone()))
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                values!(
                    DurationMicrosecondArray,
                    [i64::MIN, 0, i64::MAX],
                    |n| (n % 9) as i64
                )
            }
            other => panic!("no test column for {other}"),
        };
        // Slice so the column has an offset, as an input batch may.
        array.slice(1, rows - 1)
    }

    /// Schemas that pack, each with every column nullable and not.
    fn schemas() -> Vec<SchemaRef> {
        let tz = DataType::Timestamp(TimeUnit::Nanosecond, Some("+01:00".into()));
        let shapes = vec![
            vec![DataType::Int16, DataType::Int32],
            vec![DataType::Int64, DataType::Int32],
            vec![DataType::Int8, DataType::Boolean, DataType::UInt16],
            vec![DataType::Float32, DataType::Int32],
            vec![DataType::Float64, DataType::Boolean],
            vec![DataType::Int64, DataType::Float64],
            vec![DataType::Date32, tz, DataType::Boolean],
            vec![DataType::UInt64, DataType::Duration(TimeUnit::Microsecond)],
        ];
        // Two 64-bit columns fill 128 bits without nulls; with their null
        // bits they do not pack, which `packs_only_what_fits` checks.
        let schemas = shapes
            .into_iter()
            .flat_map(|types| {
                [true, false].map(|nullable| {
                    Arc::new(Schema::new(
                        types
                            .iter()
                            .enumerate()
                            .map(|(i, t)| {
                                Field::new(format!("c{i}"), t.clone(), nullable)
                            })
                            .collect::<Vec<_>>(),
                    ))
                })
            })
            .filter(|schema| try_new_packed(schema).is_some())
            .collect::<Vec<_>>();
        assert_eq!(schemas.len(), 14);
        schemas
    }

    fn batch(schema: &SchemaRef, rows: usize, rng: &mut Rng) -> Vec<ArrayRef> {
        schema
            .fields()
            .iter()
            .map(|field| column(field.data_type(), field.is_nullable(), rows + 1, rng))
            .collect()
    }

    /// Packed group values give the groups, the group ids and the emitted
    /// values of `GroupValuesColumn<true>`, which also assigns ids in order of
    /// first appearance, through `EmitTo::First`, `EmitTo::All` and
    /// `values_preserving`, and the grouping of `GroupValuesColumn<false>`.
    #[test]
    fn packed_matches_group_values_column() -> Result<()> {
        let mut rng = Rng(42);
        for schema in schemas() {
            let mut packed = try_new_packed(&schema).expect("schema packs");
            let mut streaming = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let mut vectorized =
                GroupValuesColumn::<false>::try_new(Arc::clone(&schema))?;
            let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
            for round in 0..6 {
                let cols = batch(&schema, 300 + 97 * round, &mut rng);
                packed.intern(&cols, &mut a)?;
                streaming.intern(&cols, &mut b)?;
                vectorized.intern(&cols, &mut c)?;
                assert_eq!(a, b, "{schema:?} round {round}");
                // Same partition of the rows into groups.
                let mut pairs = HashMap::new();
                for (x, y) in a.iter().zip(&c) {
                    assert_eq!(*pairs.entry(*x).or_insert(*y), *y, "{schema:?}");
                }
                assert_eq!(packed.len(), streaming.len());
                assert_eq!(packed.len(), vectorized.len());

                let all = GroupSelection::all(packed.len());
                assert_eq!(
                    packed.values_preserving(all)?,
                    streaming.values_preserving(all)?
                );
                if round == 2 {
                    let n = packed.len() / 3;
                    assert_eq!(
                        packed.emit(EmitTo::First(n))?,
                        streaming.emit(EmitTo::First(n))?,
                        "{schema:?}"
                    );
                    vectorized =
                        GroupValuesColumn::<false>::try_new(Arc::clone(&schema))?;
                    // The vectorized table restarts; replay the rows kept.
                    let kept = streaming
                        .values_preserving(GroupSelection::all(streaming.len()))?;
                    vectorized.intern(&kept, &mut c)?;
                }
            }
            let packed_out = packed.emit(EmitTo::All)?;
            assert_eq!(packed_out, streaming.emit(EmitTo::All)?);
            for (column, field) in packed_out.iter().zip(schema.fields()) {
                assert_eq!(column.data_type(), field.data_type());
            }
            assert!(packed.is_empty());
        }
        Ok(())
    }

    #[test]
    fn packs_only_what_fits() {
        let schema = |types: &[(DataType, bool)]| {
            Arc::new(Schema::new(
                types
                    .iter()
                    .enumerate()
                    .map(|(i, (t, nullable))| {
                        Field::new(format!("c{i}"), t.clone(), *nullable)
                    })
                    .collect::<Vec<_>>(),
            ))
        };
        // The bits a schema packs into, if it packs at all.
        let bits = |types: &[(DataType, bool)]| {
            let schema = schema(types);
            try_new_packed(&schema).and_then(|_| layout(&schema).map(|(_, bits)| bits))
        };
        use DataType::*;
        assert_eq!(bits(&[(Int64, true)]), None);
        assert_eq!(bits(&[(Int16, true), (Int32, true)]), Some(50));
        assert_eq!(bits(&[(Int16, false), (Boolean, true)]), Some(18));
        assert_eq!(bits(&[(Int64, true), (Int32, true)]), Some(98));
        assert_eq!(bits(&[(Int64, false), (Int64, false)]), Some(128));
        assert_eq!(bits(&[(Int64, true), (Int64, false)]), None);
        assert_eq!(bits(&[(Int64, true), (Int64, true), (Int8, true)]), None);
        assert_eq!(bits(&[(Int64, true), (Utf8View, true)]), None);
        assert_eq!(bits(&[(Int32, true), (Decimal128(10, 2), true)]), None);
    }

    /// The table hash spreads the keys of one repartition partition, or of
    /// one bucket of the final aggregate's router, over the table: keys that
    /// all share the low bits of those routing hashes still take nearly as
    /// many distinct home buckets as random keys would.
    #[cfg(not(feature = "force_hash_collisions"))]
    #[test]
    fn table_hash_is_independent_of_routing_hashes() -> Result<()> {
        use crate::aggregates::AGGREGATION_HASH_SEED;
        use crate::aggregates::final_buckets::BUCKET_HASH_SEED;
        use crate::repartition::REPARTITION_RANDOM_STATE;
        use datafusion_common::hash_utils::create_hashes;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int32, true),
        ]));
        let rows = 1 << 16;
        let a = Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef;
        let b = Arc::new(Int32Array::from_iter_values(
            (0..rows as i32).map(|v| v % 1000),
        )) as ArrayRef;
        let cols = [a, b];
        for routing in [REPARTITION_RANDOM_STATE.random_state(), &BUCKET_HASH_SEED] {
            let mut hashes = vec![0; rows];
            create_hashes(&cols, routing, &mut hashes)?;
            // The rows of one partition out of 16, by the routing hash's low
            // bits, and of one bucket out of 64, by its high bits.
            for keep in [
                Box::new(|hash: u64| hash % 16 == 3) as Box<dyn Fn(u64) -> bool>,
                Box::new(|hash: u64| hash >> 58 == 5),
            ] {
                let picked = hashes
                    .iter()
                    .enumerate()
                    .filter(|(_, hash)| keep(**hash))
                    .map(|(row, _)| row)
                    .collect::<Vec<_>>();
                let take = |array: &ArrayRef| {
                    arrow::compute::take(
                        array,
                        &arrow::array::UInt32Array::from_iter_values(
                            picked.iter().map(|row| *row as u32),
                        ),
                        None,
                    )
                };
                let subset = [take(&cols[0])?, take(&cols[1])?];
                let mut packed = GroupValuesPacked::<Key128>::new(
                    Arc::clone(&schema),
                    layout(&schema)
                        .map(|(columns, _)| columns)
                        .unwrap_or_default(),
                );
                let mut groups = vec![];
                packed.intern(&subset, &mut groups)?;
                let keys = packed.keys.values();
                let buckets = (2 * keys.len()).next_power_of_two();
                let homes = keys
                    .iter()
                    .map(|key| key.hash(&AGGREGATION_HASH_SEED) as usize & (buckets - 1))
                    .collect::<std::collections::HashSet<_>>();
                // Random keys take 1 - e^-load of the buckets, at least 0.39
                // of them here, so about 0.78 of the keys' count; keys that
                // shared the routing bits would take at most 1/16 or 1/64.
                assert!(
                    homes.len() * 10 >= keys.len() * 7,
                    "{} homes for {} keys",
                    homes.len(),
                    keys.len()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn all_null_columns_pack() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Float64, true),
        ]));
        let mut packed = try_new_packed(&schema).expect("schema packs");
        let cols = vec![
            new_null_array(&DataType::Int32, 4),
            Arc::new(Float64Array::from(vec![Some(-0.0), Some(0.0), None, None]))
                as ArrayRef,
        ];
        let mut groups = vec![];
        packed.intern(&cols, &mut groups)?;
        assert_eq!(groups, vec![0, 0, 1, 1]);
        let out = packed.emit(EmitTo::All)?;
        assert_eq!(out[0].null_count(), 2);
        assert_eq!(
            out[1].as_primitive::<Float64Type>().value(0).to_bits(),
            0.0f64.to_bits()
        );
        Ok(())
    }
}
