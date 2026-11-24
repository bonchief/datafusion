// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A hash repartition producer's own builders, one per output, that each
//! routed row is copied into directly.
//!
//! Without them a producer reorders each input batch with one `take` (a
//! copy of every fixed-width value and view) and then pushes each output's
//! slice into the output's shared coalescer (a second copy, under the
//! output's lock). A scatter writes each fixed-width value and view once, and
//! a long string's bytes once, into a builder no other producer touches.

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayData, ArrayRef, AsArray, BinaryViewArray, BooleanArray,
    BooleanBufferBuilder, NullBufferBuilder, RecordBatch, RecordBatchOptions,
    StringViewArray, make_array,
};
use arrow::buffer::{BooleanBuffer, Buffer, MutableBuffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowNativeType, DataType, SchemaRef, i256};
use datafusion_common::instant::Instant;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;

use crate::metrics;

/// How long a builder may hold rows before the producer emits it, checked
/// each time it routes a batch.
pub(super) const AGE: Duration = Duration::from_millis(10);

/// Bytes of string payload a builder's view column holds per row it can hold.
const VIEW_BYTES: usize = 64;

/// Bytes a view takes.
const VIEW: usize = size_of::<u128>();

/// Rounding of each buffer an allocation may add.
const ROUNDING: usize = 64;

/// How a column of a supported type is copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A primitive type of this many bytes per value: integers, floats,
    /// dates, times, timestamps, durations, intervals and decimals.
    Fixed(usize),
    /// One bit per value.
    Bool,
    /// `Utf8View` or `BinaryView`: a 16-byte view per value, and the bytes of
    /// each value longer than 12 bytes.
    View,
}

impl Kind {
    fn of(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Boolean => Some(Self::Bool),
            DataType::Utf8View | DataType::BinaryView => Some(Self::View),
            other => other.primitive_width().map(Self::Fixed),
        }
    }
}

/// The shape every producer's scatter of one repartition shares.
///
/// **A repartition scatters only when every column has a supported type**
/// ([`Kind`]); any other column (`Utf8`, dictionaries, lists, structs, ...)
/// keeps the take and shared coalescers for the whole repartition, so one
/// output never mixes batches built two ways for one column.
#[derive(Debug)]
pub(super) struct Layout {
    schema: SchemaRef,
    kinds: Vec<Kind>,
    /// Rows a builder holds before it becomes a batch: a quarter of the
    /// target batch size.
    rows: usize,
    /// Bytes of long string values a builder's view column holds before it
    /// becomes a batch.
    block: usize,
    outputs: usize,
}

impl Layout {
    /// The layout of a scatter of `schema` to `outputs` outputs, or `None`
    /// when a column's type is not supported.
    pub(super) fn try_new(
        schema: SchemaRef,
        batch_size: usize,
        outputs: usize,
    ) -> Option<Self> {
        let kinds = schema
            .fields()
            .iter()
            .map(|field| Kind::of(field.data_type()))
            .collect::<Option<Vec<_>>>()?;
        let rows = batch_size.div_ceil(4).max(1);
        Some(Self {
            schema,
            kinds,
            rows,
            block: rows * VIEW_BYTES,
            outputs,
        })
    }

    /// **The most bytes one producer's builders hold at once.** Each output's
    /// builder holds at most [`Self::rows`] rows: per column, its values (a
    /// bit each for booleans, a view each for views) and a validity bit, and
    /// per view column at most [`Self::block`] bytes of long values. The one
    /// exception is a single value longer than the block, which a builder
    /// takes as its first row and is emitted with as soon as it is
    /// appended. Each buffer may round up by 64 bytes.
    pub(super) fn bound(&self) -> usize {
        let bits = self.rows.div_ceil(8) + ROUNDING;
        let row: usize = self
            .kinds
            .iter()
            .map(|kind| {
                bits + match kind {
                    Kind::Fixed(width) => self.rows * width + ROUNDING,
                    Kind::Bool => bits,
                    Kind::View => self.rows * VIEW + ROUNDING + self.block,
                }
            })
            .sum();
        row * self.outputs
    }
}

/// One input column, as the scatter reads it.
enum Source {
    /// The column's values, exactly `len * width` bytes from its first row.
    Fixed(Buffer),
    Bool(BooleanBuffer),
    View {
        views: ScalarBuffer<u128>,
        buffers: Vec<Buffer>,
    },
}

/// One input column and its validity.
struct Column {
    source: Source,
    nulls: Option<NullBuffer>,
}

impl Column {
    fn new(kind: Kind, array: &ArrayRef) -> Self {
        let source = match kind {
            Kind::Fixed(width) => {
                let data = array.to_data();
                Source::Fixed(
                    data.buffers()[0]
                        .slice_with_length(data.offset() * width, data.len() * width),
                )
            }
            Kind::Bool => Source::Bool(array.as_boolean().values().clone()),
            Kind::View => match array.data_type() {
                DataType::Utf8View => {
                    let array = array.as_string_view();
                    Source::View {
                        views: array.views().clone(),
                        buffers: array.data_buffers().to_vec(),
                    }
                }
                _ => {
                    let array = array.as_binary_view();
                    Source::View {
                        views: array.views().clone(),
                        buffers: array.data_buffers().to_vec(),
                    }
                }
            },
        };
        Self {
            source,
            nulls: array.nulls().cloned(),
        }
    }

    /// The bytes of a value too long for its view to hold inline, or 0.
    fn long(view: u128) -> usize {
        match view as u32 {
            len @ 13.. => len as usize,
            _ => 0,
        }
    }
}

/// The bytes a scatter wrote, for the repartition's metrics.
#[derive(Debug, Default)]
struct Written {
    rows: usize,
    /// Fixed-width values, views, and booleans at a byte per eight rows.
    fixed: usize,
    /// Bytes of long string values.
    strings: usize,
}

/// One output column being built.
enum Builder {
    Fixed {
        width: usize,
        values: MutableBuffer,
        nulls: NullBufferBuilder,
    },
    Bool {
        values: BooleanBufferBuilder,
        nulls: NullBufferBuilder,
    },
    View {
        views: Vec<u128>,
        data: Vec<u8>,
        nulls: NullBufferBuilder,
    },
}

/// Appends `values[row]` for each of `rows` when `bytes` are aligned `T`s,
/// and only `T`s have been appended to `dst`.
fn gather<T: ArrowNativeType>(
    dst: &mut MutableBuffer,
    bytes: &[u8],
    rows: &[u32],
) -> bool {
    // SAFETY: `T` is a fixed-width native type for which every bit pattern
    // is a value, and `align_to` only hands out the aligned middle.
    let (head, values, tail) = unsafe { bytes.align_to::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        return false;
    }
    let len = dst.len();
    dst.reserve(rows.len() * size_of::<T>());
    // SAFETY: the buffer is 64-byte aligned and holds only `T`s, so its end
    // is aligned for `T`; `reserve` made room for `rows.len()` more.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            dst.as_mut_ptr().add(len).cast::<MaybeUninit<T>>(),
            rows.len(),
        )
    };
    for (slot, &row) in out.iter_mut().zip(rows) {
        slot.write(values[row as usize]);
    }
    // SAFETY: every slot up to the new length was written above.
    unsafe { dst.set_len(len + rows.len() * size_of::<T>()) };
    true
}

/// Appends the view `view_of(row)` gives for each of `rows`, copying each
/// long value into `data` at the view's new offset in buffer 0, until a long
/// value would take `data` past `block` bytes; returns how many rows it
/// took. The first row is taken however long its value when `alone`.
fn copy_views(
    view_of: impl Fn(usize) -> u128,
    rows: &[u32],
    views: &mut Vec<u128>,
    data: &mut Vec<u8>,
    buffers: &[Buffer],
    block: usize,
    alone: bool,
) -> usize {
    views.reserve_exact(rows.len());
    let base = views.len();
    let out = &mut views.spare_capacity_mut()[..rows.len()];
    let mut taken = rows.len();
    for (i, (slot, &row)) in out.iter_mut().zip(rows).enumerate() {
        let view = view_of(row as usize);
        let len = Column::long(view);
        slot.write(if len == 0 {
            view
        } else {
            if data.len() + len > block && !(alone && i == 0) {
                taken = i;
                break;
            }
            let buffer = (view >> 64) as u32 as usize;
            let offset = (view >> 96) as u32 as usize;
            let at = data.len() as u128;
            data.extend_from_slice(&buffers[buffer].as_slice()[offset..offset + len]);
            // Length and prefix kept; buffer 0 at its new offset.
            (view & u128::from(u64::MAX)) | (at << 96)
        });
    }
    // SAFETY: the first `taken` slots past `base` were written above.
    unsafe { views.set_len(base + taken) };
    taken
}

fn append_nulls(dst: &mut NullBufferBuilder, src: Option<&NullBuffer>, rows: &[u32]) {
    match src {
        None => dst.append_n_non_nulls(rows.len()),
        Some(nulls) => rows
            .iter()
            .for_each(|&row| dst.append(nulls.is_valid(row as usize))),
    }
}

impl Builder {
    fn new(kind: Kind, rows: usize) -> Self {
        match kind {
            Kind::Fixed(width) => Self::Fixed {
                width,
                values: MutableBuffer::new(0),
                nulls: NullBufferBuilder::new(rows),
            },
            Kind::Bool => Self::Bool {
                values: BooleanBufferBuilder::new(rows),
                nulls: NullBufferBuilder::new(rows),
            },
            Kind::View => Self::View {
                views: Vec::new(),
                data: Vec::new(),
                nulls: NullBufferBuilder::new(rows),
            },
        }
    }

    /// Bytes of long values this builder holds.
    fn strings(&self) -> usize {
        match self {
            Self::View { data, .. } => data.len(),
            _ => 0,
        }
    }

    /// Appends `rows` of a fixed-width or boolean `column`, sizing a fresh
    /// builder for `capacity` rows.
    fn append(
        &mut self,
        column: &Column,
        rows: &[u32],
        capacity: usize,
        written: &mut Written,
    ) {
        match (self, &column.source) {
            (
                Self::Fixed {
                    width,
                    values,
                    nulls,
                },
                Source::Fixed(src),
            ) => {
                let width = *width;
                if values.capacity() == 0 {
                    values.reserve(capacity * width);
                }
                let bytes = src.as_slice();
                let typed = match width {
                    1 => gather::<u8>(values, bytes, rows),
                    2 => gather::<u16>(values, bytes, rows),
                    4 => gather::<u32>(values, bytes, rows),
                    8 => gather::<u64>(values, bytes, rows),
                    16 => gather::<i128>(values, bytes, rows),
                    32 => gather::<i256>(values, bytes, rows),
                    _ => false,
                };
                if !typed {
                    for &row in rows {
                        let at = row as usize * width;
                        values.extend_from_slice(&bytes[at..at + width]);
                    }
                }
                append_nulls(nulls, column.nulls.as_ref(), rows);
                written.fixed += rows.len() * width;
            }
            (Self::Bool { values, nulls }, Source::Bool(src)) => {
                for &row in rows {
                    values.append(src.value(row as usize));
                }
                append_nulls(nulls, column.nulls.as_ref(), rows);
                written.fixed += rows.len().div_ceil(8);
            }
            (Self::View { .. }, _) => {
                unreachable!("view columns are appended with append_views")
            }
            _ => unreachable!("a builder and its column share a kind"),
        }
    }

    /// Appends `rows` of a view `column` until the next long value would take
    /// this builder's long values past `block` bytes, and returns how many
    /// rows it took: all of them, or fewer, but at least one when `empty`,
    /// so a value longer than the block is taken alone.
    fn append_views(
        &mut self,
        column: &Column,
        rows: &[u32],
        capacity: usize,
        block: usize,
        empty: bool,
    ) -> usize {
        let (
            Self::View { views, data, nulls },
            Source::View {
                views: src,
                buffers,
            },
        ) = (self, &column.source)
        else {
            unreachable!("a builder and its column share a kind")
        };
        if views.capacity() == 0 {
            views.reserve_exact(capacity);
        }
        if data.capacity() == 0 {
            data.reserve_exact(block);
        }
        let alone = empty && views.is_empty();
        let taken = match &column.nulls {
            None => copy_views(|row| src[row], rows, views, data, buffers, block, alone),
            // A null's view may point anywhere; it is written as an empty value.
            Some(nulls) => copy_views(
                |row| if nulls.is_valid(row) { src[row] } else { 0 },
                rows,
                views,
                data,
                buffers,
                block,
                alone,
            ),
        };
        append_nulls(nulls, column.nulls.as_ref(), &rows[..taken]);
        taken
    }

    /// Drops the views a view builder took past its first `rows`, and the
    /// long values only they referenced.
    fn truncate_views(&mut self, rows: usize) {
        let Self::View { views, data, nulls } = self else {
            unreachable!("only view columns are cut")
        };
        views.truncate(rows);
        let end = views
            .iter()
            .rev()
            .find_map(|view| {
                let len = Column::long(*view);
                (len > 0).then(|| (view >> 96) as u32 as usize + len)
            })
            .unwrap_or(0);
        data.truncate(end);
        nulls.truncate(rows);
    }

    /// The column built so far, leaving this builder empty.
    ///
    /// A buffer more than half unused (a batch emitted early) is shrunk to
    /// what it holds first, so a short batch never keeps a full builder's
    /// allocation alive in the output's queue.
    fn finish(&mut self, data_type: &DataType, rows: usize) -> Result<ArrayRef> {
        Ok(match self {
            Self::Fixed { values, nulls, .. } => {
                let mut values = std::mem::replace(values, MutableBuffer::new(0));
                if values.len() * 2 < values.capacity() {
                    values.shrink_to_fit();
                }
                let data = ArrayData::builder(data_type.clone())
                    .len(rows)
                    .add_buffer(values.into())
                    .nulls(nulls.finish())
                    .build()?;
                make_array(data)
            }
            Self::Bool { values, nulls } => {
                Arc::new(BooleanArray::new(values.finish(), nulls.finish()))
            }
            Self::View { views, data, nulls } => {
                let views = ScalarBuffer::from(std::mem::take(views));
                let mut data = std::mem::take(data);
                if data.len() * 2 < data.capacity() {
                    data.shrink_to_fit();
                }
                let buffers: Arc<[Buffer]> = match data.is_empty() {
                    true => Arc::new([]),
                    false => Arc::new([Buffer::from_vec(data)]),
                };
                let nulls = nulls.finish();
                match data_type {
                    // SAFETY: each view was copied from a valid array of this
                    // type: a short value's view as it was, a long value's
                    // length and prefix with its bytes copied to the offset
                    // the view now names in buffer 0; a null's view is empty.
                    DataType::Utf8View => Arc::new(unsafe {
                        StringViewArray::new_unchecked(views, buffers, nulls)
                    }),
                    // SAFETY: as above.
                    _ => Arc::new(unsafe {
                        BinaryViewArray::new_unchecked(views, buffers, nulls)
                    }),
                }
            }
        })
    }
}

/// One output's builders.
struct Output {
    columns: Vec<Builder>,
    rows: usize,
    /// When the oldest held row was appended.
    since: Option<Instant>,
}

/// One producer's builders, one per output, and the memory they hold.
///
/// Invariants:
/// - **Each output receives this producer's rows in input order**: rows are
///   appended in input order, and an output's builder is emitted whole
///   before it takes more rows.
/// - **The builders hold at most [`Layout::bound`] bytes**, reserved once
///   when the producer starts; a producer whose reservation the pool refuses
///   does not scatter. The batches a producer takes from its builders while
///   it copies one input batch wait for their sends together, charged to
///   their outputs only as each is sent, as a take's slices do: at most that
///   input batch's rows and the rows the builders held before it.
/// - **A builder holds rows only while its producer is routing**: the
///   producer emits every builder before it waits for input, at the end of
///   its input, and any builder whose oldest row is [`AGE`] old as it routes
///   its next batch.
pub(super) struct Scatter {
    layout: Arc<Layout>,
    outputs: Vec<Output>,
    columns: Vec<Column>,
    written: Written,
    /// Rows held across all outputs.
    held: usize,
    _reservation: MemoryReservation,
}

impl Scatter {
    /// A scatter for one producer, or `None` when the pool refuses its
    /// builders' bound.
    pub(super) fn try_new(
        layout: Arc<Layout>,
        reservation: MemoryReservation,
    ) -> Option<Self> {
        reservation.try_grow(layout.bound()).ok()?;
        let outputs = (0..layout.outputs)
            .map(|_| Output {
                columns: layout
                    .kinds
                    .iter()
                    .map(|&kind| Builder::new(kind, layout.rows))
                    .collect(),
                rows: 0,
                since: None,
            })
            .collect();
        Some(Self {
            layout,
            outputs,
            columns: vec![],
            written: Written::default(),
            held: 0,
            _reservation: reservation,
        })
    }

    /// Reads `batch`'s columns for the appends that follow.
    pub(super) fn load(&mut self, batch: &RecordBatch) -> Result<()> {
        self.columns.clear();
        for ((array, kind), field) in batch
            .columns()
            .iter()
            .zip(&self.layout.kinds)
            .zip(self.layout.schema.fields())
        {
            if array.data_type() != field.data_type() {
                return internal_err!(
                    "repartition scatter expected {} for column {}, got {}",
                    field.data_type(),
                    field.name(),
                    array.data_type()
                );
            }
            self.columns.push(Column::new(*kind, array));
        }
        Ok(())
    }

    /// Lets go of the loaded batch's buffers.
    pub(super) fn unload(&mut self) {
        self.columns.clear();
    }

    /// Appends as many of the loaded batch's `rows` to `output` as fit, and
    /// returns how many it took and whether the builder is now full and must
    /// be emitted before it takes more.
    pub(super) fn append(
        &mut self,
        output: usize,
        rows: &[u32],
        now: Instant,
    ) -> (usize, bool) {
        let layout = &self.layout;
        let out = &mut self.outputs[output];
        let empty = out.rows == 0;
        let asked = rows.len();
        let mut take = asked.min(layout.rows - out.rows);
        // View columns first: each takes rows until its string block is
        // full, and a cut cuts the view columns already appended.
        let strings = out.columns.iter().map(Builder::strings).sum::<usize>();
        let mut cut = false;
        for (c, kind) in layout.kinds.iter().enumerate() {
            if *kind != Kind::View {
                continue;
            }
            let taken = out.columns[c].append_views(
                &self.columns[c],
                &rows[..take],
                layout.rows,
                layout.block,
                empty,
            );
            if taken < take {
                take = taken;
                cut = true;
            }
        }
        if cut {
            for (builder, kind) in out.columns.iter_mut().zip(&layout.kinds) {
                if *kind == Kind::View {
                    builder.truncate_views(out.rows + take);
                }
            }
        }
        self.written.strings +=
            out.columns.iter().map(Builder::strings).sum::<usize>() - strings;
        let rows = &rows[..take];
        for ((builder, column), kind) in
            out.columns.iter_mut().zip(&self.columns).zip(&layout.kinds)
        {
            match kind {
                Kind::View => self.written.fixed += rows.len() * VIEW,
                _ => builder.append(column, rows, layout.rows, &mut self.written),
            }
        }
        if empty && take > 0 {
            out.since = Some(now);
        }
        out.rows += take;
        self.held += take;
        self.written.rows += take;
        // A builder whose one value took its block past the bound is full.
        let over = out
            .columns
            .iter()
            .any(|builder| builder.strings() > layout.block);
        (take, out.rows == layout.rows || take < asked || over)
    }

    /// Whether any builder holds rows.
    pub(super) fn holds(&self) -> bool {
        self.held > 0
    }

    /// Whether `output`'s oldest held row was appended [`AGE`] or more
    /// before `now`.
    pub(super) fn stale(&self, output: usize, now: Instant) -> bool {
        self.outputs[output]
            .since
            .is_some_and(|since| now.duration_since(since) >= AGE)
    }

    /// The batch `output`'s builder holds, if any, leaving it empty.
    pub(super) fn emit(&mut self, output: usize) -> Result<Option<RecordBatch>> {
        let out = &mut self.outputs[output];
        if out.rows == 0 {
            return Ok(None);
        }
        let rows = out.rows;
        let columns = out
            .columns
            .iter_mut()
            .zip(self.layout.schema.fields())
            .map(|(builder, field)| builder.finish(field.data_type(), rows))
            .collect::<Result<Vec<_>>>()?;
        out.rows = 0;
        out.since = None;
        self.held -= rows;
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        Ok(Some(RecordBatch::try_new_with_options(
            Arc::clone(&self.layout.schema),
            columns,
            &options,
        )?))
    }

    /// Adds what was written since the last call to the producer's metrics.
    pub(super) fn record(
        &mut self,
        rows: &metrics::Count,
        fixed: &metrics::Count,
        strings: &metrics::Count,
    ) {
        let written = std::mem::take(&mut self.written);
        rows.add(written.rows);
        fixed.add(written.fixed);
        strings.add(written.strings);
    }
}
