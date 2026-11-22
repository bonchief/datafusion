// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Regex expressions
use memchr::{memchr, memchr2};

use crate::utils::transform_leaf_type_preserving_encoding;
use arrow::array::ArrayDataBuilder;
use arrow::array::BufferBuilder;
use arrow::array::GenericStringArray;
use arrow::array::make_view;
use arrow::array::{Array, ArrayRef, OffsetSizeTrait};
use arrow::array::{ArrayAccessor, StringViewArray};
use arrow::array::{ArrayIter, AsArray, new_null_array};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef};
use arrow_select::dictionary::garbage_collect_any_dictionary;
use datafusion_common::ScalarValue;
use datafusion_common::cast::{
    as_large_string_array, as_string_array, as_string_view_array,
};
use datafusion_common::exec_err;
use datafusion_common::plan_err;
use datafusion_common::utils::offset_span_len;
use datafusion_common::{Result, cast::as_generic_string_array, internal_err};
use datafusion_expr::ColumnarValue;
use datafusion_expr::function::Hint;
use datafusion_expr::{
    Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use regex::{CaptureLocations, Regex};
use regex_syntax::hir::{Class, ClassUnicode, Hir, HirKind, Look};

use super::compile_regex;
use crate::strings::{MAX_INLINE_LEN, sub_view};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Replaces substrings in a string that match a [regular expression](https://docs.rs/regex/latest/regex/#syntax).",
    syntax_example = "regexp_replace(str, regexp, replacement[, flags])",
    sql_example = r#"```sql
> select regexp_replace('foobarbaz', 'b(..)', 'X\\1Y', 'g');
+------------------------------------------------------------------------+
| regexp_replace(Utf8("foobarbaz"),Utf8("b(..)"),Utf8("X\1Y"),Utf8("g")) |
+------------------------------------------------------------------------+
| fooXarYXazY                                                            |
+------------------------------------------------------------------------+
SELECT regexp_replace('aBc', '(b|d)', 'Ab\\1a', 'i');
+-------------------------------------------------------------------+
| regexp_replace(Utf8("aBc"),Utf8("(b|d)"),Utf8("Ab\1a"),Utf8("i")) |
+-------------------------------------------------------------------+
| aAbBac                                                            |
+-------------------------------------------------------------------+
```
Additional examples can be found [here](https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/builtin_functions/regexp.rs)
"#,
    standard_argument(name = "str", prefix = "String"),
    argument(
        name = "regexp",
        description = "Regular expression to match against.
  Can be a constant, column, or function."
    ),
    argument(
        name = "replacement",
        description = "Replacement string expression to operate on. Can be a constant, column, or function, and any combination of operators."
    ),
    argument(
        name = "flags",
        description = r#"Optional regular expression flags that control the behavior of the regular expression. Refer to the flags reference above for supported flags."#
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RegexpReplaceFunc {
    signature: Signature,
}
impl Default for RegexpReplaceFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexpReplaceFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for RegexpReplaceFunc {
    fn name(&self) -> &str {
        "regexp_replace"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        transform_leaf_type_preserving_encoding(&arg_types[0], &regexp_replace_leaf_type)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = &args.arg_fields[0];
        let others_scalar = args.scalar_arguments.iter().skip(1).all(Option::is_some);
        let output_type = match input.data_type() {
            DataType::Dictionary(_, value_type) if !others_scalar => {
                regexp_replace_leaf_type(value_type.as_ref())?
            }
            other => self.return_type(std::slice::from_ref(other))?,
        };
        Ok(Arc::new(Field::new(self.name(), output_type, true)))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if !(3..=4).contains(&arg_types.len()) {
            return plan_err!(
                "regexp_replace expects 3 or 4 arguments, got {}",
                arg_types.len()
            );
        }
        let leaf = match &arg_types[0] {
            DataType::Dictionary(_, v) => v.as_ref(),
            other => other,
        };
        let is_string_leaf = matches!(
            leaf,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
        );
        let target = if is_string_leaf {
            regexp_replace_leaf_type(leaf)?
        } else {
            DataType::Utf8View
        };
        let first = match &arg_types[0] {
            DataType::Dictionary(k, _) if is_string_leaf => {
                DataType::Dictionary(k.clone(), Box::new(target.clone()))
            }
            _ => target.clone(),
        };
        let mut out = Vec::with_capacity(arg_types.len());
        out.push(first);
        for _ in 1..arg_types.len() {
            out.push(target.clone());
        }
        Ok(out)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let preserve_dictionary =
            matches!(args.return_field.data_type(), DataType::Dictionary(_, _));
        let mut args = args.args;

        let scalar_dict_key = if let Some(ColumnarValue::Scalar(
            ScalarValue::Dictionary(key_type, value),
        )) = args.first()
        {
            let key = key_type.clone();
            args[0] = ColumnarValue::Scalar(value.as_ref().clone());
            preserve_dictionary.then_some(key)
        } else {
            None
        };

        if let ColumnarValue::Array(array) = &args[0]
            && let DataType::Dictionary(_, value_type) = array.data_type().clone()
        {
            if preserve_dictionary {
                let compacted;
                let mut dict = array.as_any_dictionary();
                if dict.values().len() > dict.keys().len() {
                    // Filtered batches and shared Parquet dictionaries can have more
                    // values than rows, so drop unreferenced values first.
                    compacted = garbage_collect_any_dictionary(dict)?;
                    dict = compacted.as_any_dictionary();
                }
                let mut inner: Vec<ColumnarValue> = Vec::with_capacity(args.len());
                inner.push(ColumnarValue::Array(Arc::clone(dict.values())));
                inner.extend_from_slice(&args[1..]);
                let values_out = regexp_replace_func(&inner)?;
                return Ok(ColumnarValue::Array(dict.with_values(values_out)));
            }
            args[0] = ColumnarValue::Array(arrow::compute::cast(array, &value_type)?);
        }

        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        let is_scalar = len.is_none();
        let result = regexp_replace_func(&args)?;
        let out = if is_scalar {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&result, 0)?)
        } else {
            ColumnarValue::Array(result)
        };

        if let (Some(key_type), ColumnarValue::Scalar(scalar)) = (scalar_dict_key, &out) {
            return Ok(ColumnarValue::Scalar(ScalarValue::Dictionary(
                key_type,
                Box::new(scalar.clone()),
            )));
        }
        Ok(out)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn regexp_replace_leaf_type(dt: &DataType) -> Result<DataType> {
    use DataType::*;
    Ok(match dt {
        LargeUtf8 | LargeBinary => LargeUtf8,
        Utf8 | Binary => Utf8,
        Utf8View | BinaryView => Utf8View,
        Null => Null,
        other => {
            return plan_err!(
                "the regexp_replace can only accept strings but got {other:?}"
            );
        }
    })
}

fn regexp_replace_func(args: &[ColumnarValue]) -> Result<ArrayRef> {
    match args[0].data_type() {
        DataType::Utf8 => specialize_regexp_replace::<i32>(args),
        DataType::LargeUtf8 => specialize_regexp_replace::<i64>(args),
        DataType::Utf8View => specialize_regexp_replace::<i32>(args),
        other => {
            internal_err!("Unsupported data type {other:?} for function regexp_replace")
        }
    }
}

/// replace POSIX capture groups (like \1 or \\1) with Rust Regex group (like ${1})
/// used by regexp_replace
/// Handles both single backslash (\1) and double backslash (\\1) which can occur
/// when SQL strings with escaped backslashes are passed through
///
/// Note: \0 is converted to ${0}, which in Rust's regex replacement syntax
/// substitutes the entire match. This is consistent with POSIX behavior where
/// \0 (or &) refers to the entire matched string.
fn regex_replace_posix_groups(replacement: &str) -> String {
    static CAPTURE_GROUPS_RE_LOCK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\{1,2}(\d+)").unwrap());
    CAPTURE_GROUPS_RE_LOCK
        .replace_all(replacement, "$${$1}")
        .into_owned()
}

/// A replacement that is one capture group alone (`${N}`, `$N` or a group's
/// name) of a pattern every match of which starts at the start of the value
/// and ends at its end (`^...$` without the multi-line flag).
///
/// Such a pattern matches a value whole or not at all, and matches it at most
/// once, so with or without the global flag `regexp_replace` returns the
/// group's span of a matching value (empty when the group does not take part
/// in the match) and an unmatched value unchanged. The result is a substring
/// of the input, which a `Utf8View` result points at instead of copying.
///
/// Public so that a caller running `regexp_replace` over values of its own,
/// such as binary values it has checked are UTF-8, answers as this does.
#[derive(Clone, Debug)]
pub struct Extract {
    re: Regex,
    locs: CaptureLocations,
    group: usize,
    short: Option<Short>,
    prefix: Option<Prefix>,
}

/// A pattern `^<prefix>([^X]+)X<tail>$` whose prefix is one of a few
/// literals, as `^https?://(?:www\.)?([^/]+)/.*$`, matched without a regex:
/// try each literal in the order the regex prefers them, and the first that
/// starts the value, is followed by at least one byte before the next `X`,
/// and leaves a rest [`Tail`] admits, decides the match.
///
/// The group cannot hold `X` and must be followed by it, so it ends at the
/// first `X` after the literal; were the rest refused there, no shorter
/// group could end at an `X` either, so the regex moves on to its next
/// literal, as this does. No literal leading to a match means no match.
///
/// Ported from rust-lang/regex#1350 (`LiteralPrefixCapture`, open), which
/// makes the `regex` crate's own engine do this, with the group and the
/// tail taken from the pattern instead of fixed to group 1 and `.*`.
#[derive(Clone, Debug)]
struct Prefix {
    /// Distinct literals in the regex's order of preference.
    literals: Vec<Vec<u8>>,
    stop: u8,
    tail: Tail,
}

/// Each `(?:...)?` doubles the literals and each alternation adds to them,
/// so a pattern with more takes the regex.
const MAX_LITERALS: usize = 32;

/// The pattern without a trailing repetition of a class before its end
/// anchor, as `^https?://([^/]+)/` for `^https?://([^/]+)/.*$`.
///
/// The pattern matches where the shorter one matches and the class matches
/// the rest of the value, and with the same groups: the repetition captures
/// nothing and must reach the end, so the leftmost-first choice of the
/// shorter pattern decides the match. It spares searching the rest of the
/// value when [`Tail`] checks it with `memchr` instead. Group 0, the whole
/// match, differs between the two, so a replacement by it never uses this.
#[derive(Clone, Debug)]
struct Short {
    re: Regex,
    locs: CaptureLocations,
    tail: Tail,
}

/// The bytes the stripped repetition's class leaves out: none, or some of
/// `\n` and `\r`, as `.` does with and without the `s` and `R` flags.
#[derive(Clone, Copy, Debug)]
enum Tail {
    Any,
    One(u8),
    Two(u8, u8),
}

impl Extract {
    /// Decides whether `pattern`, as `re` was compiled from it, and
    /// `replacement`, in the `regex` crate's syntax, qualify; see [`Extract`].
    pub fn new(re: &Regex, pattern: &str, replacement: &str) -> Option<Self> {
        let group = group(re, replacement)?;
        // `Regex::new` parses with the parser's default options.
        let hir = regex_syntax::Parser::new().parse(pattern).ok()?;
        let props = hir.properties();
        if !props.look_set_prefix().contains(Look::Start)
            || !props.look_set_suffix().contains(Look::End)
        {
            return None;
        }
        let prefix = Prefix::new(&hir, group);
        let short = if group == 0 || prefix.is_some() {
            None
        } else {
            Short::new(&hir, re.captures_len())
        };
        Some(Self {
            re: re.clone(),
            locs: re.capture_locations(),
            group,
            short,
            prefix,
        })
    }

    /// The group's span in `val`, empty when the pattern matches without the
    /// group, or `None` when it does not match.
    pub fn find(&mut self, val: &str) -> Option<(usize, usize)> {
        if let Some(prefix) = &self.prefix {
            return prefix.find(val.as_bytes());
        }
        if let Some(short) = self.short.as_mut() {
            // The pattern cannot match where the shorter one does not.
            let end = short.re.captures_read(&mut short.locs, val)?.end();
            if short.tail.admits(&val.as_bytes()[end..]) {
                return Some(short.locs.get(self.group).unwrap_or_default());
            }
        }
        self.re.captures_read(&mut self.locs, val)?;
        Some(self.locs.get(self.group).unwrap_or_default())
    }
}

impl Short {
    fn new(hir: &Hir, groups: usize) -> Option<Self> {
        let HirKind::Concat(items) = hir.kind() else {
            return None;
        };
        let [head @ .., rest, end] = items.as_slice() else {
            return None;
        };
        if !matches!(end.kind(), HirKind::Look(Look::End)) {
            return None;
        }
        let HirKind::Repetition(rep) = rest.kind() else {
            return None;
        };
        if rep.min != 0 || rep.max.is_some() {
            return None;
        }
        let HirKind::Class(Class::Unicode(class)) = rep.sub.kind() else {
            return None;
        };
        let tail = Tail::new(class)?;
        let re = Regex::new(&Hir::concat(head.to_vec()).to_string()).ok()?;
        (re.captures_len() == groups).then(|| Self {
            locs: re.capture_locations(),
            re,
            tail,
        })
    }
}

impl Prefix {
    fn new(hir: &Hir, group: usize) -> Option<Self> {
        let HirKind::Concat(items) = hir.kind() else {
            return None;
        };
        let (first, items) = items.split_first()?;
        if !matches!(first.kind(), HirKind::Look(Look::Start)) {
            return None;
        }
        let at = items
            .iter()
            .position(|item| matches!(item.kind(), HirKind::Capture(_)))?;
        let mut literals = vec![Vec::new()];
        for item in &items[..at] {
            literals = concat(&literals, &variants(item)?)?;
        }
        let HirKind::Capture(capture) = items[at].kind() else {
            return None;
        };
        if capture.index as usize != group {
            return None;
        }
        let stop = stop(&capture.sub)?;
        let [next, rest, end] = &items[at + 1..] else {
            return None;
        };
        if !matches!(next.kind(), HirKind::Literal(lit) if lit.0.as_ref() == [stop])
            || !matches!(end.kind(), HirKind::Look(Look::End))
        {
            return None;
        }
        let HirKind::Repetition(rep) = rest.kind() else {
            return None;
        };
        let HirKind::Class(Class::Unicode(class)) = rep.sub.kind() else {
            return None;
        };
        if rep.min != 0 || rep.max.is_some() {
            return None;
        }
        let tail = Tail::new(class)?;
        let mut distinct: Vec<Vec<u8>> = Vec::with_capacity(literals.len());
        for literal in literals {
            if !distinct.contains(&literal) {
                distinct.push(literal);
            }
        }
        Some(Self {
            literals: distinct,
            stop,
            tail,
        })
    }

    /// The group's span, or `None` when the pattern does not match.
    fn find(&self, val: &[u8]) -> Option<(usize, usize)> {
        self.literals.iter().find_map(|literal| {
            let start = literal.len();
            let rest = val.strip_prefix(literal.as_slice())?;
            let end = start + memchr(self.stop, rest).filter(|&len| len > 0)?;
            self.tail.admits(&val[end + 1..]).then_some((start, end))
        })
    }
}

/// The literals `hir` matches, in the regex's order of preference, when it
/// is built of literals, the empty pattern, concatenations, alternations and
/// `?` alone.
fn variants(hir: &Hir) -> Option<Vec<Vec<u8>>> {
    match hir.kind() {
        HirKind::Empty => Some(vec![Vec::new()]),
        HirKind::Literal(lit) => Some(vec![lit.0.to_vec()]),
        HirKind::Concat(items) => items
            .iter()
            .try_fold(vec![Vec::new()], |acc, item| concat(&acc, &variants(item)?)),
        HirKind::Repetition(rep) if rep.min == 0 && rep.max == Some(1) => {
            let mut out = variants(&rep.sub)?;
            if out.len() >= MAX_LITERALS {
                return None;
            }
            if rep.greedy {
                out.push(Vec::new());
            } else {
                out.insert(0, Vec::new());
            }
            Some(out)
        }
        HirKind::Alternation(branches) => {
            let mut out = Vec::new();
            for branch in branches {
                out.extend(variants(branch)?);
                if out.len() > MAX_LITERALS {
                    return None;
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Every literal of `heads` followed by every literal of `tails`: a regex
/// exhausts the later part's choices before it backtracks into the earlier.
fn concat(heads: &[Vec<u8>], tails: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
    if heads.len().checked_mul(tails.len())? > MAX_LITERALS {
        return None;
    }
    Some(
        heads
            .iter()
            .flat_map(|head| tails.iter().map(move |tail| [&head[..], tail].concat()))
            .collect(),
    )
}

/// The byte `X` of a greedy `[^X]+`, a class of every char but one ASCII.
fn stop(hir: &Hir) -> Option<u8> {
    let HirKind::Repetition(rep) = hir.kind() else {
        return None;
    };
    if rep.min != 1 || rep.max.is_some() || !rep.greedy {
        return None;
    }
    let HirKind::Class(Class::Unicode(class)) = rep.sub.kind() else {
        return None;
    };
    let mut left = class.clone();
    left.negate();
    match left.ranges() {
        [range] if range.start() == range.end() && range.start().is_ascii() => {
            Some(range.start() as u8)
        }
        _ => None,
    }
}

impl Tail {
    fn new(class: &ClassUnicode) -> Option<Self> {
        let mut left = class.clone();
        left.negate();
        let mut bytes = Vec::with_capacity(2);
        for range in left.iter() {
            if u32::from(range.end()) - u32::from(range.start()) > 3 {
                return None;
            }
            for c in range.start()..=range.end() {
                if c != '\n' && c != '\r' {
                    return None;
                }
                bytes.push(c as u8);
            }
        }
        match bytes.as_slice() {
            [] => Some(Self::Any),
            [a] => Some(Self::One(*a)),
            [a, b] => Some(Self::Two(*a, *b)),
            _ => None,
        }
    }

    fn admits(&self, rest: &[u8]) -> bool {
        match self {
            Self::Any => true,
            Self::One(a) => memchr(*a, rest).is_none(),
            Self::Two(a, b) => memchr2(*a, *b, rest).is_none(),
        }
    }
}

/// The capture group `replacement` consists of alone, read as
/// `Regex::replacen` reads it: `$name` takes the longest run of
/// `[0-9A-Za-z_]`, `${name}` anything up to `}`, and a name that parses as a
/// number is that group. A group the pattern lacks does not qualify.
fn group(re: &Regex, replacement: &str) -> Option<usize> {
    let name = replacement.strip_prefix('$')?;
    let name = match name.strip_prefix('{') {
        Some(name) => name.strip_suffix('}').filter(|n| !n.contains('}'))?,
        None => Some(name)
            .filter(|n| n.bytes().all(|b| b == b'_' || b.is_ascii_alphanumeric()))?,
    };
    if name.is_empty() {
        return None;
    }
    let index = name
        .parse::<usize>()
        .ok()
        .or_else(|| re.capture_names().position(|n| n == Some(name)))?;
    (index < re.captures_len()).then_some(index)
}

/// The views of a `Utf8View` result. A value that is an input value or a
/// substring of one points into the input's own buffers; any other value of
/// more than 12 bytes is copied into new buffers after them.
struct Views<'a> {
    input: &'a StringViewArray,
    views: Vec<u128>,
    /// New buffers, filled before `data`.
    blocks: Vec<Buffer>,
    data: Vec<u8>,
}

impl<'a> Views<'a> {
    fn new(input: &'a StringViewArray) -> Self {
        Self {
            input,
            views: Vec::with_capacity(input.len()),
            blocks: Vec::new(),
            data: Vec::new(),
        }
    }

    fn view(&mut self, view: u128) {
        self.views.push(view);
    }

    fn push(&mut self, value: &[u8]) -> Result<()> {
        if value.len() <= MAX_INLINE_LEN {
            self.views.push(make_view(value, 0, 0));
            return Ok(());
        }
        if self.data.len() + value.len() > u32::MAX as usize {
            self.blocks
                .push(Buffer::from_vec(std::mem::take(&mut self.data)));
        }
        let index = self.input.data_buffers().len() + self.blocks.len();
        let Ok(block) = u32::try_from(index) else {
            return exec_err!("regexp_replace result has too many buffers");
        };
        // At most `u32::MAX` after the check above.
        let offset = self.data.len() as u32;
        self.views.push(make_view(value, block, offset));
        self.data.extend_from_slice(value);
        Ok(())
    }

    fn finish(mut self) -> ArrayRef {
        if !self.data.is_empty() {
            self.blocks.push(Buffer::from_vec(self.data));
        }
        let shared = self.input.data_buffers();
        let buffers = if self.blocks.is_empty() {
            Arc::clone(shared)
        } else {
            shared.iter().cloned().chain(self.blocks).collect()
        };
        // SAFETY: every view is inline, an input view, a view `sub_view` made
        // of a span of an input value in the same buffer, or `make_view` of
        // bytes appended at its offset in the buffer it names. Every value is
        // a whole `&str` or a span a regex matched, which starts and ends on
        // char boundaries, so each view addresses valid UTF-8 in bounds.
        let array = unsafe {
            StringViewArray::new_unchecked(
                ScalarBuffer::from(self.views),
                buffers,
                self.input.nulls().cloned(),
            )
        };
        Arc::new(array)
    }
}

/// Replaces substring(s) matching a PCRE-like regular expression.
///
/// The full list of supported features and syntax can be found at
/// <https://docs.rs/regex/latest/regex/#syntax>
///
/// Supported flags with the addition of 'g' can be found at
/// <https://docs.rs/regex/latest/regex/#grouping-and-flags>
///
/// # Examples
///
/// ```ignore
/// # use datafusion::prelude::*;
/// # use datafusion::error::Result;
/// # #[tokio::main]
/// # async fn main() -> Result<()> {
/// let ctx = SessionContext::new();
/// let df = ctx.read_csv("tests/data/regex.csv", CsvReadOptions::new()).await?;
///
/// // use the regexp_replace function to replace substring(s) without flags
/// let df = df.with_column(
///     "a",
///     regexp_replace(vec![col("values"), col("patterns"), col("replacement")])
/// )?;
/// // use the regexp_replace function to replace substring(s) with flags
/// let df = df.with_column(
///     "b",
///     regexp_replace(vec![col("values"), col("patterns"), col("replacement"), col("flags")]),
/// )?;
///
/// // literals can be used as well
/// let df = df.with_column(
///     "c",
///     regexp_replace(vec![lit("foobarbequebaz"), lit("(bar)(beque)"), lit(r"\2")]),
/// )?;
///
/// df.show().await?;
///
/// # Ok(())
/// # }
/// ```
pub fn regexp_replace<'a, T: OffsetSizeTrait, U>(
    string_array: U,
    pattern_array: U,
    replacement_array: U,
    flags_array: Option<U>,
) -> Result<ArrayRef>
where
    U: ArrayAccessor<Item = &'a str>,
{
    // Default implementation for regexp_replace, assumes all args are arrays
    // and args is a sequence of 3 or 4 elements.

    // creating Regex is expensive so create hashmap for memoization
    let mut patterns: HashMap<String, Regex> = HashMap::new();

    let datatype = string_array.data_type().to_owned();

    let string_array_iter = ArrayIter::new(string_array);
    let pattern_array_iter = ArrayIter::new(pattern_array);
    let replacement_array_iter = ArrayIter::new(replacement_array);

    match flags_array {
        None => {
            let result_iter = string_array_iter
                .zip(pattern_array_iter)
                .zip(replacement_array_iter)
                .map(|((string, pattern), replacement)| {
                    match (string, pattern, replacement) {
                        (Some(string), Some(pattern), Some(replacement)) => {
                            let replacement = regex_replace_posix_groups(replacement);
                            // if patterns hashmap already has regexp then use else create and return
                            let re = match patterns.get(pattern) {
                                Some(re) => Ok(re),
                                None => {
                                    match compile_regex("regexp_replace", pattern, None) {
                                        Ok(re) => {
                                            patterns.insert(pattern.to_string(), re);
                                            Ok(patterns.get(pattern).unwrap())
                                        }
                                        Err(err) => Err(err),
                                    }
                                }
                            };

                            Some(re.map(|re| re.replace(string, replacement.as_str())))
                                .transpose()
                        }
                        _ => Ok(None),
                    }
                });

            match datatype {
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let result =
                        result_iter.collect::<Result<GenericStringArray<T>>>()?;
                    Ok(Arc::new(result) as ArrayRef)
                }
                DataType::Utf8View => {
                    let result = result_iter.collect::<Result<StringViewArray>>()?;
                    Ok(Arc::new(result) as ArrayRef)
                }
                other => {
                    exec_err!(
                        "Unsupported data type {other:?} for function regex_replace"
                    )
                }
            }
        }
        Some(flags_array) => {
            let flags_array_iter = ArrayIter::new(flags_array);
            // Reused across rows to build the pattern cache key
            let mut pattern_buf = String::new();

            let result_iter = string_array_iter
                .zip(pattern_array_iter)
                .zip(replacement_array_iter)
                .zip(flags_array_iter)
                .map(|(((string, pattern), replacement), flags)| {
                    match (string, pattern, replacement, flags) {
                        (Some(string), Some(pattern), Some(replacement), Some(flags)) => {
                            let replacement = regex_replace_posix_groups(replacement);

                            // format flags into rust pattern
                            let replace_all = flags.contains('g');
                            let flags = flags.replace('g', "");
                            pattern_buf.clear();
                            if !flags.is_empty() {
                                pattern_buf.push_str("(?");
                                pattern_buf.push_str(&flags);
                                pattern_buf.push(')');
                            }
                            pattern_buf.push_str(pattern);
                            let pattern = pattern_buf.as_str();

                            // if patterns hashmap already has regexp then use else create and return
                            let re = match patterns.get(pattern) {
                                Some(re) => Ok(re),
                                None => {
                                    match compile_regex("regexp_replace", pattern, None) {
                                        Ok(re) => {
                                            patterns.insert(pattern.to_string(), re);
                                            Ok(patterns.get(pattern).unwrap())
                                        }
                                        Err(err) => Err(err),
                                    }
                                }
                            };

                            Some(re.map(|re| {
                                if replace_all {
                                    re.replace_all(string, replacement.as_str())
                                } else {
                                    re.replace(string, replacement.as_str())
                                }
                            }))
                            .transpose()
                        }
                        _ => Ok(None),
                    }
                });

            match datatype {
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let result =
                        result_iter.collect::<Result<GenericStringArray<T>>>()?;
                    Ok(Arc::new(result) as ArrayRef)
                }
                DataType::Utf8View => {
                    let result = result_iter.collect::<Result<StringViewArray>>()?;
                    Ok(Arc::new(result) as ArrayRef)
                }
                other => {
                    exec_err!(
                        "Unsupported data type {other:?} for function regex_replace"
                    )
                }
            }
        }
    }
}

/// Get the first argument from the given string array.
///
/// Note: If the array is empty or the first argument is null,
/// then aborts early.
macro_rules! fetch_string_arg {
    ($ARG:expr, $NAME:expr, $ARRAY_SIZE:expr) => {{
        let string_array_type = ($ARG).data_type();
        match string_array_type {
            dt if $ARG.len() == 0 || $ARG.is_null(0) => {
                // Mimicking the existing behavior of regexp_replace, if any of the scalar arguments
                // are actually null, then the result will be an array of the same size as the first argument with all nulls.
                //
                // Also acts like an early abort mechanism when the input array is empty.
                return Ok(new_null_array(dt, $ARRAY_SIZE));
            }
            DataType::Utf8 => {
                let array = as_string_array($ARG)?;
                array.value(0)
            }
            DataType::LargeUtf8 => {
                let array = as_large_string_array($ARG)?;
                array.value(0)
            }
            DataType::Utf8View => {
                let array = as_string_view_array($ARG)?;
                array.value(0)
            }
            _ => unreachable!(
                "Invalid data type for regexp_replace: {}",
                string_array_type
            ),
        }
    }};
}
/// Special cased regex_replace implementation for the scenario where
/// the pattern, replacement and flags are static (arrays that are derived
/// from scalars). This means we can skip regex caching system and basically
/// hold a single Regex object for the replace operation. This also speeds
/// up the pre-processing time of the replacement string, since it only
/// needs to processed once.
fn regexp_replace_static_pattern_replace<T: OffsetSizeTrait>(
    args: &[ArrayRef],
) -> Result<ArrayRef> {
    let array_size = args[0].len();
    let pattern = fetch_string_arg!(&args[1], "pattern", array_size);
    let replacement = fetch_string_arg!(&args[2], "replacement", array_size);
    let flags = match args.len() {
        3 => None,
        4 => Some(fetch_string_arg!(&args[3], "flags", array_size)),
        other => {
            return exec_err!(
                "regexp_replace was called with {other} arguments. It requires at least 3 and at most 4."
            );
        }
    };

    // Embed the flag (if it exists) into the pattern. Limit will determine
    // whether this is a global match (as in replace all) or just a single
    // replace operation.
    let (pattern, limit) = match flags {
        Some(flags) => {
            let limit = !flags.contains('g') as usize;
            let flags = flags.replace('g', "");
            let pattern = if flags.is_empty() {
                pattern.to_string()
            } else {
                format!("(?{flags}){pattern}")
            };
            (pattern, limit)
        }
        None => (pattern.to_string(), 1),
    };

    let re = compile_regex("regexp_replace", &pattern, None)?;

    // Replaces the posix groups in the replacement string
    // with rust ones.
    let replacement = regex_replace_posix_groups(replacement);

    let mut extract = Extract::new(&re, &pattern, &replacement);

    let string_array_type = args[0].data_type();
    match string_array_type {
        DataType::Utf8 | DataType::LargeUtf8 => {
            let string_array = as_generic_string_array::<T>(&args[0])?;

            // We are going to create the underlying string buffer from its parts
            // to be able to re-use the existing null buffer for sparse arrays.
            let mut vals =
                BufferBuilder::<u8>::new(offset_span_len(string_array.offsets()));
            let mut new_offsets = BufferBuilder::<T>::new(string_array.len() + 1);
            new_offsets.append(T::zero());

            for val in string_array.iter() {
                if let Some(val) = val {
                    match extract.as_mut() {
                        Some(extract) => match extract.find(val) {
                            Some((start, end)) => {
                                vals.append_slice(&val.as_bytes()[start..end])
                            }
                            None => vals.append_slice(val.as_bytes()),
                        },
                        None => {
                            let result = re.replacen(val, limit, replacement.as_str());
                            vals.append_slice(result.as_bytes());
                        }
                    }
                }
                let Some(offset) = T::from_usize(vals.len()) else {
                    return exec_err!("regexp_replace result is too large");
                };
                new_offsets.append(offset);
            }

            let data = ArrayDataBuilder::new(GenericStringArray::<T>::DATA_TYPE)
                .len(string_array.len())
                .nulls(string_array.nulls().cloned())
                .buffers(vec![new_offsets.finish(), vals.finish()])
                .build()?;
            let result_array = GenericStringArray::<T>::from(data);
            Ok(Arc::new(result_array) as ArrayRef)
        }
        DataType::Utf8View => {
            let array = as_string_view_array(&args[0])?;
            let mut out = Views::new(array);
            for (i, &view) in array.views().iter().enumerate() {
                if array.is_null(i) {
                    out.view(0);
                    continue;
                }
                let val = array.value(i);
                match extract.as_mut() {
                    Some(extract) => match extract.find(val) {
                        Some((start, end)) => {
                            out.view(sub_view(view, val.as_bytes(), start..end))
                        }
                        None => out.view(view),
                    },
                    None => match re.replacen(val, limit, replacement.as_str()) {
                        // `replacen` borrows the value only when nothing matched.
                        Cow::Borrowed(result) if std::ptr::eq(result, val) => {
                            out.view(view)
                        }
                        result => out.push(result.as_bytes())?,
                    },
                }
            }
            Ok(out.finish())
        }
        _ => unreachable!(
            "Invalid data type for regexp_replace: {}",
            string_array_type
        ),
    }
}

/// Determine which implementation of the regexp_replace to use based
/// on the given set of arguments.
fn specialize_regexp_replace<T: OffsetSizeTrait>(
    args: &[ColumnarValue],
) -> Result<ArrayRef> {
    // This will serve as a dispatch table where we can
    // leverage it in order to determine whether the scalarity
    // of the given set of arguments fits a better specialized
    // function.
    let (is_source_scalar, is_pattern_scalar, is_replacement_scalar, is_flags_scalar) = (
        matches!(args[0], ColumnarValue::Scalar(_)),
        matches!(args[1], ColumnarValue::Scalar(_)),
        matches!(args[2], ColumnarValue::Scalar(_)),
        // The forth argument (flags) is optional; so in the event that
        // it is not available, we'll claim that it is scalar.
        matches!(args.get(3), Some(ColumnarValue::Scalar(_)) | None),
    );
    let len = args
        .iter()
        .fold(Option::<usize>::None, |acc, arg| match arg {
            ColumnarValue::Scalar(_) => acc,
            ColumnarValue::Array(a) => Some(a.len()),
        });
    let inferred_length = len.unwrap_or(1);
    match (
        is_source_scalar,
        is_pattern_scalar,
        is_replacement_scalar,
        is_flags_scalar,
    ) {
        // This represents a very hot path for the case where the there is
        // a single pattern that is being matched against and a single replacement.
        // This is extremely important to specialize on since it removes the overhead
        // of DF's in-house regex pattern cache (since there will be at most a single
        // pattern) and the pre-processing of the same replacement pattern at each
        // query.
        //
        // The flags needs to be a scalar as well since each pattern is actually
        // constructed with the flags embedded into the pattern itself. This means
        // even if the pattern itself is scalar, if the flags are an array then
        // we will create many regexes and it is best to use the implementation
        // that caches it. If there are no flags, we can simply ignore it here,
        // and let the specialized function handle it.
        (_, true, true, true) => {
            let hints = [
                Hint::Pad,
                Hint::AcceptsSingular,
                Hint::AcceptsSingular,
                Hint::AcceptsSingular,
            ];
            let args = args
                .iter()
                .zip(hints.iter().chain(std::iter::repeat(&Hint::Pad)))
                .map(|(arg, hint)| {
                    // Decide on the length to expand this scalar to depending
                    // on the given hints.
                    let expansion_len = match hint {
                        Hint::AcceptsSingular => 1,
                        Hint::Pad => inferred_length,
                    };
                    arg.to_array(expansion_len)
                })
                .collect::<Result<Vec<_>>>()?;
            regexp_replace_static_pattern_replace::<T>(&args)
        }

        // If there are no specialized implementations, we'll fall back to the
        // generic implementation.
        (_, _, _, _) => {
            let args = args
                .iter()
                .map(|arg| arg.to_array(inferred_length))
                .collect::<Result<Vec<_>>>()?;

            match (
                args[0].data_type(),
                args[1].data_type(),
                args[2].data_type(),
                args.get(3).map(|a| a.data_type()),
            ) {
                (
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    Some(DataType::Utf8) | None,
                ) => {
                    let string_array = args[0].as_string::<i32>();
                    let pattern_array = args[1].as_string::<i32>();
                    let replacement_array = args[2].as_string::<i32>();
                    let flags_array = args.get(3).map(|a| a.as_string::<i32>());
                    regexp_replace::<i32, _>(
                        string_array,
                        pattern_array,
                        replacement_array,
                        flags_array,
                    )
                }
                (
                    DataType::Utf8View,
                    DataType::Utf8View,
                    DataType::Utf8View,
                    Some(DataType::Utf8View) | None,
                ) => {
                    let string_array = args[0].as_string_view();
                    let pattern_array = args[1].as_string_view();
                    let replacement_array = args[2].as_string_view();
                    let flags_array = args.get(3).map(|a| a.as_string_view());
                    regexp_replace::<i32, _>(
                        string_array,
                        pattern_array,
                        replacement_array,
                        flags_array,
                    )
                }
                (
                    DataType::LargeUtf8,
                    DataType::LargeUtf8,
                    DataType::LargeUtf8,
                    Some(DataType::LargeUtf8) | None,
                ) => {
                    let string_array = args[0].as_string::<i64>();
                    let pattern_array = args[1].as_string::<i64>();
                    let replacement_array = args[2].as_string::<i64>();
                    let flags_array = args.get(3).map(|a| a.as_string::<i64>());
                    regexp_replace::<i64, _>(
                        string_array,
                        pattern_array,
                        replacement_array,
                        flags_array,
                    )
                }
                other => {
                    exec_err!(
                        "Unsupported data type {other:?} for function regex_replace"
                    )
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use arrow::array::*;

    use super::*;

    #[test]
    fn test_regex_replace_posix_groups() {
        // Test that \1, \2, etc. are replaced with ${1}, ${2}, etc.
        assert_eq!(regex_replace_posix_groups(r"\1"), "${1}");
        assert_eq!(regex_replace_posix_groups(r"\12"), "${12}");
        assert_eq!(regex_replace_posix_groups(r"X\1Y"), "X${1}Y");
        assert_eq!(regex_replace_posix_groups(r"\1\2"), "${1}${2}");

        // Test double backslash (from SQL escaped strings like '\\1')
        assert_eq!(regex_replace_posix_groups(r"\\1"), "${1}");
        assert_eq!(regex_replace_posix_groups(r"X\\1Y"), "X${1}Y");
        assert_eq!(regex_replace_posix_groups(r"\\1\\2"), "${1}${2}");

        // Test 3 or 4 backslashes before digits to document expected behavior
        assert_eq!(regex_replace_posix_groups(r"\\\1"), r"\${1}");
        assert_eq!(regex_replace_posix_groups(r"\\\\1"), r"\\${1}");
        assert_eq!(regex_replace_posix_groups(r"\\\1\\\\2"), r"\${1}\\${2}");

        // Test that a lone backslash is NOT replaced (requires at least one digit)
        assert_eq!(regex_replace_posix_groups(r"\"), r"\");
        assert_eq!(regex_replace_posix_groups(r"foo\bar"), r"foo\bar");

        // Test that backslash followed by non-digit is preserved
        assert_eq!(regex_replace_posix_groups(r"\n"), r"\n");
        assert_eq!(regex_replace_posix_groups(r"\t"), r"\t");

        // Test \0 behavior: \0 is converted to ${0}, which in Rust's regex
        // replacement syntax substitutes the entire match. This is consistent
        // with POSIX behavior where \0 (or &) refers to the entire matched string.
        assert_eq!(regex_replace_posix_groups(r"\0"), "${0}");
        assert_eq!(
            regex_replace_posix_groups(r"prefix\0suffix"),
            "prefix${0}suffix"
        );
    }

    macro_rules! static_pattern_regexp_replace {
        ($name:ident, $T:ty, $O:ty) => {
            #[test]
            fn $name() {
                let values = vec!["abc", "acd", "abcd1234567890123", "123456789012abc"];
                let patterns = vec!["b"; 4];
                let replacement = vec!["foo"; 4];
                let expected =
                    vec!["afooc", "acd", "afoocd1234567890123", "123456789012afooc"];

                let values = <$T>::from(values);
                let patterns = <$T>::from(patterns);
                let replacements = <$T>::from(replacement);
                let expected = <$T>::from(expected);

                let re = regexp_replace_static_pattern_replace::<$O>(&[
                    Arc::new(values),
                    Arc::new(patterns),
                    Arc::new(replacements),
                ])
                .unwrap();

                assert_eq!(re.as_ref(), &expected);
            }
        };
    }

    static_pattern_regexp_replace!(string_array, StringArray, i32);
    static_pattern_regexp_replace!(string_view_array, StringViewArray, i32);
    static_pattern_regexp_replace!(large_string_array, LargeStringArray, i64);

    macro_rules! static_pattern_regexp_replace_with_flags {
        ($name:ident, $T:ty, $O: ty) => {
            #[test]
            fn $name() {
                let values = vec![
                    "abc",
                    "aBc",
                    "acd",
                    "abcd1234567890123",
                    "aBcd1234567890123",
                    "123456789012abc",
                    "123456789012aBc",
                ];
                let expected = vec![
                    "afooc",
                    "afooc",
                    "acd",
                    "afoocd1234567890123",
                    "afoocd1234567890123",
                    "123456789012afooc",
                    "123456789012afooc",
                ];

                let values = <$T>::from(values);
                let patterns = StringArray::from(vec!["b"; 7]);
                let replacements = StringArray::from(vec!["foo"; 7]);
                let flags = StringArray::from(vec!["i"; 5]);
                let expected = <$T>::from(expected);

                let re = regexp_replace_static_pattern_replace::<$O>(&[
                    Arc::new(values),
                    Arc::new(patterns),
                    Arc::new(replacements),
                    Arc::new(flags),
                ])
                .unwrap();

                assert_eq!(re.as_ref(), &expected);
            }
        };
    }

    static_pattern_regexp_replace_with_flags!(string_array_with_flags, StringArray, i32);
    static_pattern_regexp_replace_with_flags!(
        string_view_array_with_flags,
        StringViewArray,
        i32
    );
    static_pattern_regexp_replace_with_flags!(
        large_string_array_with_flags,
        LargeStringArray,
        i64
    );

    #[test]
    fn test_static_pattern_regexp_replace_early_abort() {
        let values = StringArray::from(vec!["abc"; 5]);
        let patterns = StringArray::from(vec![None::<&str>; 5]);
        let replacements = StringArray::from(vec!["foo"; 5]);
        let expected = StringArray::from(vec![None::<&str>; 5]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
        ])
        .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_static_pattern_regexp_replace_early_abort_when_empty() {
        let values = StringArray::from(Vec::<Option<&str>>::new());
        let patterns = StringArray::from(Vec::<Option<&str>>::new());
        let replacements = StringArray::from(Vec::<Option<&str>>::new());
        let expected = StringArray::from(Vec::<Option<&str>>::new());

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
        ])
        .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_static_pattern_regexp_replace_early_abort_flags() {
        let values = StringArray::from(vec!["abc"; 5]);
        let patterns = StringArray::from(vec!["a"; 5]);
        let replacements = StringArray::from(vec!["foo"; 5]);
        let flags = StringArray::from(vec![None::<&str>; 5]);
        let expected = StringArray::from(vec![None::<&str>; 5]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
            Arc::new(flags),
        ])
        .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_static_pattern_regexp_replace_empty_flags() {
        // An empty flags string must behave like no flags at all, matching
        // `compile_regex` in this module, which is used by the other regexp
        // functions.
        let values = StringArray::from(vec!["abc"; 3]);
        let patterns = StringArray::from(vec!["b"; 3]);
        let replacements = StringArray::from(vec!["X"; 3]);
        // `gg` collapses to an empty flags string once `g` is stripped.
        let flags = StringArray::from(vec![Some(""), Some("g"), Some("gg")]);
        let expected = StringArray::from(vec!["aXc"; 3]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
            Arc::new(flags),
        ])
        .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_regexp_replace_empty_flags() {
        let values = StringArray::from(vec!["abc"; 3]);
        let patterns = StringArray::from(vec!["b"; 3]);
        let replacements = StringArray::from(vec!["X"; 3]);
        let flags = StringArray::from(vec![Some(""), Some("g"), Some("gg")]);
        let expected = StringArray::from(vec!["aXc"; 3]);

        let re =
            regexp_replace::<i32, _>(&values, &patterns, &replacements, Some(&flags))
                .unwrap();

        assert_eq!(re.as_ref(), &expected);
    }

    #[test]
    fn test_static_pattern_regexp_replace_pattern_error() {
        let values = StringArray::from(vec!["abc"; 5]);
        // Deliberately using an invalid pattern to see how the single pattern
        // error is propagated on regexp_replace.
        let patterns = StringArray::from(vec!["["; 5]);
        let replacements = StringArray::from(vec!["foo"; 5]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
        ]);
        let pattern_err = re.expect_err("broken pattern should have failed");
        assert_eq!(
            pattern_err.strip_backtrace(),
            "Execution error: Regular expression did not compile: regex parse error:\n    [\n    ^\nerror: unclosed character class"
        );
    }

    #[test]
    fn test_static_pattern_regexp_replace_with_null_buffers() {
        let values = StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            None,
            Some("a"),
            None,
            None,
            Some("c"),
        ]);
        let patterns = StringArray::from(vec!["a"; 1]);
        let replacements = StringArray::from(vec!["foo"; 1]);
        let expected = StringArray::from(vec![
            Some("foo"),
            None,
            Some("b"),
            None,
            Some("foo"),
            None,
            None,
            Some("c"),
        ]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
        ])
        .unwrap();

        assert_eq!(re.as_ref(), &expected);
        assert_eq!(re.null_count(), 4);
    }

    #[test]
    fn test_static_pattern_regexp_replace_with_sliced_null_buffer() {
        let values = StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            None,
            Some("a"),
            None,
            None,
            Some("c"),
        ]);
        let values = values.slice(2, 5);
        let patterns = StringArray::from(vec!["a"; 1]);
        let replacements = StringArray::from(vec!["foo"; 1]);
        let expected = StringArray::from(vec![Some("b"), None, Some("foo"), None, None]);

        let re = regexp_replace_static_pattern_replace::<i32>(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(replacements),
        ])
        .unwrap();
        assert_eq!(re.as_ref(), &expected);
        assert_eq!(re.null_count(), 3);
    }

    fn static_replace(
        values: ArrayRef,
        pattern: &str,
        rep: &str,
        flags: Option<&str>,
    ) -> ArrayRef {
        let large = values.data_type() == &DataType::LargeUtf8;
        let scalar = |s: &str| -> ArrayRef {
            if large {
                Arc::new(LargeStringArray::from(vec![s]))
            } else {
                Arc::new(StringArray::from(vec![s]))
            }
        };
        let mut args = vec![values, scalar(pattern), scalar(rep)];
        args.extend(flags.map(scalar));
        if large {
            regexp_replace_static_pattern_replace::<i64>(&args).unwrap()
        } else {
            regexp_replace_static_pattern_replace::<i32>(&args).unwrap()
        }
    }

    fn strings(array: &ArrayRef) -> Vec<Option<String>> {
        let own = |v: Option<&str>| v.map(String::from);
        match array.data_type() {
            DataType::Utf8View => array.as_string_view().iter().map(own).collect(),
            DataType::LargeUtf8 => array.as_string::<i64>().iter().map(own).collect(),
            _ => array.as_string::<i32>().iter().map(own).collect(),
        }
    }

    /// What `regex` itself returns for `regexp_replace(values, pattern, rep, flags)`.
    fn oracle(
        values: &[Option<String>],
        pattern: &str,
        rep: &str,
        flags: Option<&str>,
    ) -> Vec<Option<String>> {
        let inner = flags.map_or(String::new(), |f| f.replace('g', ""));
        let full = if inner.is_empty() {
            pattern.to_string()
        } else {
            format!("(?{inner}){pattern}")
        };
        let re = Regex::new(&full).unwrap();
        let rep = regex_replace_posix_groups(rep);
        let all = flags.is_some_and(|f| f.contains('g'));
        values
            .iter()
            .map(|v| {
                v.as_deref().map(|v| {
                    if all {
                        re.replace_all(v, rep.as_str()).into_owned()
                    } else {
                        re.replace(v, rep.as_str()).into_owned()
                    }
                })
            })
            .collect()
    }

    /// The same values as `Utf8View` inline and in one buffer, in buffers of
    /// 16 bytes, sliced, as `Utf8` and as `LargeUtf8`.
    fn layouts(values: &[Option<String>]) -> Vec<(&'static str, ArrayRef)> {
        let refs: Vec<Option<&str>> = values.iter().map(|v| v.as_deref()).collect();
        let mut blocks = StringViewBuilder::new().with_fixed_block_size(16);
        for v in &refs {
            blocks.append_option(*v);
        }
        let mut out: Vec<(&'static str, ArrayRef)> = vec![
            ("view", Arc::new(StringViewArray::from(refs.clone()))),
            ("view-blocks", Arc::new(blocks.finish())),
            ("utf8", Arc::new(StringArray::from(refs.clone()))),
            ("large", Arc::new(LargeStringArray::from(refs.clone()))),
        ];
        if refs.len() > 2 {
            let view = StringViewArray::from(refs.clone());
            out.push(("view-sliced", Arc::new(view.slice(1, refs.len() - 2))));
        }
        out
    }

    fn check(values: &[Option<String>], pattern: &str, rep: &str, flags: Option<&str>) {
        for (layout, input) in layouts(values) {
            let want = oracle(&strings(&input), pattern, rep, flags);
            let result = static_replace(Arc::clone(&input), pattern, rep, flags);
            result.to_data().validate_full().unwrap();
            assert_eq!(result.data_type(), input.data_type());
            let got = strings(&result);
            for (row, (got, want)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    got,
                    want,
                    "pattern {pattern:?} replacement {rep:?} flags {flags:?} \
                     layout {layout} row {row} value {:?}",
                    strings(&input)[row]
                );
            }
        }
    }

    /// Every path the pattern and replacement select (a group of a pattern
    /// anchored at both ends, with a literal prefix or a stripped tail or
    /// neither, and the general replacement) returns what the regex crate's
    /// own replacement does, on each string type.
    #[test]
    fn test_static_pattern_regexp_replace_matches_regex_crate() {
        let values: Vec<Option<String>> = [
            None,
            Some(""),
            Some("http://a.b/"),
            Some("https://www.example.com/path?q=1"),
            Some("https://www.x.com/path\nmore"),
            Some("http://www./x"),
            Some("http://example.com"),
            Some("https://www.longer-than-twelve-bytes.example/a/b/c"),
            Some("http://a.b/c\rd"),
            Some("aaa"),
            Some("xbz"),
            Some("abc/x"),
            Some("a/x"),
            Some("abx/y"),
            Some("ab/b\ncd/e"),
            Some("a\nb/x"),
        ]
        .into_iter()
        .map(|v| v.map(String::from))
        .collect();
        let cases: &[(&str, &str, Option<&str>)] = &[
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", None),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", Some("g")),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", Some("i")),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", Some("m")),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", Some("s")),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", Some("R")),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\0", None),
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"x\1", None),
            (r"^https?://([^/?]+)[^?]*(?:\?.*)?$", r"\1", None),
            (r"^(?P<h>[a-z]+)://.*$", r"${h}", None),
            (r"^(x)?(.*)$", r"\1", None),
            // `.*$` stripped from the pattern's text would cut the
            // alternation's second branch short.
            (r"^(a)|b.*$", r"\1", None),
            (r"^(a*)$", r"\1", Some("g")),
            (r"^(?:a|ab)([^/]+)/.*$", r"\1", None),
            (r"^a??([^/]+)/.*$", r"\1", None),
            (r"^a?([^/]+)/.*$", r"\1", None),
            (r"^a?(?:ab)?([^/]+)/.*$", r"\1", None),
            ("^(?:a|ab/b\nc)([^/]+)/.*$", r"\1", None),
            (r"^a([^/]+)/.*$", r"\1", None),
            (r"://([^/]+)", r"\1", None),
            (r"^https?://([^/]+)/([^?]*)", r"\2-\1", None),
            (r"[a-z]+", "N", Some("g")),
        ];
        for (pattern, rep, flags) in cases {
            check(&values, pattern, rep, *flags);
        }
    }

    /// Patterns in and just outside the shapes `Prefix` and `Short` accept,
    /// on values built to reach their edges, return byte for byte what the
    /// regex crate's own replacement does. `REGEXP_REPLACE_DIFF_SEED` and
    /// `REGEXP_REPLACE_DIFF_PATTERNS` replay or widen a run.
    #[test]
    fn test_static_pattern_regexp_replace_differential() {
        use rand::rngs::StdRng;
        use rand::seq::IndexedRandom;
        use rand::{Rng, SeedableRng};

        const LITERALS: &[&str] = &[
            "", "a", "ab", "abc", "b", "h", "http", "https", "s", "://", "www.", "é",
            "日", "x", "a.b",
        ];
        const STOPS: &[char] = &['/', ',', '.', ':', 'a', '\n', 'é'];
        const TOKENS: &[&str] = &[
            "a", "ab", "b", "h", "http", "https", "://", "www.", "/", ",", ".", ":",
            "\n", "\r", "\r\n", "é", "日本", "x", "y", "1", "A", "W", "s",
        ];

        fn literal(rng: &mut StdRng) -> String {
            regex_syntax::escape(LITERALS.choose(rng).unwrap())
        }
        fn alternation(rng: &mut StdRng) -> String {
            let n = rng.random_range(2..=3);
            let branches: Vec<String> = (0..n).map(|_| literal(rng)).collect();
            format!("(?:{})", branches.join("|"))
        }
        /// A literal, an alternation of literals, or either made optional.
        fn piece(rng: &mut StdRng) -> String {
            match rng.random_range(0..6) {
                0 | 1 => literal(rng),
                2 => alternation(rng),
                3 => format!("(?:{})?", literal(rng)),
                4 => format!("(?:{})??", literal(rng)),
                _ => format!("{}?", alternation(rng)),
            }
        }
        /// One of `inside` (a choice the fast paths accept) four times in
        /// five, else one of `outside` (a near miss).
        fn pick(
            rng: &mut StdRng,
            inside: &[&'static str],
            outside: &[&'static str],
        ) -> &'static str {
            if rng.random_bool(0.8) {
                inside.choose(rng).unwrap()
            } else {
                outside.choose(rng).unwrap()
            }
        }
        fn tail(rng: &mut StdRng) -> &'static str {
            pick(
                rng,
                &[
                    ".*",
                    r"[^\r]*",
                    r"[^\r\n]*",
                    r"[^\n]*",
                    "(?s:.*)",
                    r"[\s\S]*",
                    ".*?",
                ],
                &[".+", r"\S*", "[^a]*", "", ".*.*", r"[^\r\n\t]*"],
            )
        }
        fn start(rng: &mut StdRng) -> &'static str {
            pick(
                rng,
                &["^", "^", "^", r"\A"],
                &["", "(?:^|x)", "(?m:^)", "x?^"],
            )
        }
        fn end(rng: &mut StdRng) -> &'static str {
            pick(
                rng,
                &["$", "$", "$", r"\z"],
                &["", r"\n?$", "(?m:$)", "$x?"],
            )
        }
        fn open(rng: &mut StdRng) -> &'static str {
            if rng.random_bool(0.2) { "(?P<g>" } else { "(" }
        }
        /// `^<pieces>([^X]+)X<tail>$` and its near misses.
        fn prefix_shape(rng: &mut StdRng) -> String {
            let stop = *STOPS.choose(rng).unwrap();
            let x = regex_syntax::escape(&stop.to_string());
            let body = if rng.random_bool(0.8) {
                format!("[^{x}]+")
            } else {
                [
                    format!("[^{x}]*"),
                    format!("[^{x}]+?"),
                    format!("[^{x}y]+"),
                    r"\w+".to_string(),
                    format!("[^{x}]{{1,3}}"),
                ]
                .choose(rng)
                .unwrap()
                .clone()
            };
            let after = if rng.random_bool(0.85) {
                x.clone()
            } else {
                [format!("{x}{x}"), String::new(), "y".to_string()]
                    .choose(rng)
                    .unwrap()
                    .clone()
            };
            let pieces: String =
                (0..rng.random_range(0..=3)).map(|_| piece(rng)).collect();
            format!(
                "{}{pieces}{}{body}){after}{}{}",
                start(rng),
                open(rng),
                tail(rng),
                end(rng)
            )
        }
        /// `^<head><class>*$` with groups in the head, and near misses.
        fn short_shape(rng: &mut StdRng) -> String {
            const HEADS: &[&str] = &[
                "(a|ab)",
                r"(\w*)",
                r"(?:www\.)?",
                "([^/]+)/",
                "(é|e)?",
                r"(\d*)",
                "([a-z]+)",
                "(.)",
                "(?:x(y))?",
                "(a*)",
                "(?:(a)|(b))",
                "()",
                "(a)|(b)",
                "(.*)",
                "(a+?)",
            ];
            let head: String = (0..rng.random_range(1..=3))
                .map(|_| {
                    let h = HEADS.choose(rng).unwrap();
                    if rng.random_bool(0.15) {
                        h.replacen('(', "(?P<g>", 1).replace("(?P<g>?:", "(?:")
                    } else {
                        h.to_string()
                    }
                })
                .collect();
            format!("{}{head}{}{}", start(rng), tail(rng), end(rng))
        }
        fn unanchored(rng: &mut StdRng) -> String {
            [
                r"([a-z]+)",
                r"(\d)",
                "(a)(b)?",
                r"://([^/]+)",
                "(é)",
                "",
                "(x*)",
            ]
            .choose(rng)
            .unwrap()
            .to_string()
        }
        /// A reference to `group` in each spelling four times in five, else a
        /// replacement that is not one group alone.
        fn reference(rng: &mut StdRng, group: usize, named: bool) -> String {
            if rng.random_bool(0.8) {
                match rng.random_range(0..6) {
                    0 => format!(r"\{group}"),
                    1 => format!(r"\\{group}"),
                    2 => format!("${{{group}}}"),
                    3 => format!("${group}"),
                    4 if named => "${g}".to_string(),
                    5 if named => "$g".to_string(),
                    _ => format!(r"\{group}"),
                }
            } else {
                match rng.random_range(0..6) {
                    0 => format!(r"x\{group}"),
                    1 => format!(r"\{group}\{group}"),
                    2 => format!("${group}x"),
                    3 => "$$1".to_string(),
                    4 => "${9}".to_string(),
                    _ => "${}".to_string(),
                }
            }
        }
        fn value(rng: &mut StdRng) -> String {
            let mut s: String = (0..rng.random_range(0..8))
                .map(|_| *TOKENS.choose(rng).unwrap())
                .collect();
            match rng.random_range(0..10) {
                // 12 bytes is the most a view holds inline, 13 the least
                // it keeps in a buffer.
                0 => pad(&mut s, 12),
                1 => pad(&mut s, 13),
                2 => pad(&mut s, 4096),
                _ => {}
            }
            s
        }
        fn pad(s: &mut String, len: usize) {
            while s.len() < len {
                s.push(if s.len().is_multiple_of(7) { '/' } else { 'b' });
            }
            while s.len() > len {
                s.pop();
            }
        }

        let seed: u64 = std::env::var("REGEXP_REPLACE_DIFF_SEED")
            .map_or(0x5EED_2026, |s| s.parse().unwrap());
        let patterns: u64 = std::env::var("REGEXP_REPLACE_DIFF_PATTERNS")
            .map_or(400, |s| s.parse().unwrap());
        let flags = [
            None,
            Some(""),
            Some("i"),
            Some("g"),
            Some("m"),
            Some("s"),
            Some("gi"),
            Some("gm"),
            Some("gs"),
            Some("ims"),
            Some("gims"),
            Some("R"),
            Some("U"),
        ];
        let mut paths = [0usize; 4];
        let mut cases = 0usize;
        for i in 0..patterns {
            let rng = &mut StdRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9));
            let pattern = match rng.random_range(0..20) {
                0..=10 => prefix_shape(rng),
                11..=17 => short_shape(rng),
                _ => unanchored(rng),
            };
            // Flags that leave the fast paths open, three times in five.
            let flag = if rng.random_bool(0.6) {
                *[None, Some(""), Some("g"), Some("s"), Some("gs"), Some("R")]
                    .choose(rng)
                    .unwrap()
            } else {
                *flags.choose(rng).unwrap()
            };
            let inner = flag.map_or(String::new(), |f| f.replace('g', ""));
            let full = if inner.is_empty() {
                pattern.clone()
            } else {
                format!("(?{inner}){pattern}")
            };
            let Ok(re) = Regex::new(&full) else {
                continue;
            };
            let named = re.capture_names().any(|n| n == Some("g"));
            // A group of the pattern four times in five, else the whole
            // match or one it lacks.
            let groups = re.captures_len();
            let group = if groups > 1 && rng.random_bool(0.8) {
                rng.random_range(1..groups)
            } else {
                rng.random_range(0..=groups)
            };
            let rep = reference(rng, group, named);
            paths[match Extract::new(&re, &full, &regex_replace_posix_groups(&rep)) {
                None => 0,
                Some(e) if e.prefix.is_some() => 1,
                Some(e) if e.short.is_some() => 2,
                Some(_) => 3,
            }] += 1;
            let mut values: Vec<Option<String>> =
                (0..16).map(|_| Some(value(rng))).collect();
            values.push(None);
            values.push(Some(String::new()));
            let at = rng.random_range(0..values.len());
            values.insert(at, None);
            cases += values.len() * 5;
            let run = std::panic::catch_unwind(|| check(&values, &pattern, &rep, flag));
            assert!(
                run.is_ok(),
                "mismatch: REGEXP_REPLACE_DIFF_SEED={seed} pattern number {i}: \
                 {pattern:?} {rep:?} {flag:?}"
            );
        }
        println!(
            "seed {seed}: {cases} cases; patterns by path: fallback {}, prefix {}, \
             short {}, whole regex {}",
            paths[0], paths[1], paths[2], paths[3]
        );
    }

    /// A `Utf8View` result of a group of a pattern anchored at both ends, or
    /// of a replacement that leaves values unchanged, holds no bytes of its
    /// own: its views point into the input's buffers.
    #[test]
    fn test_static_pattern_regexp_replace_view_shares_input() {
        let values = StringViewArray::from(vec![
            Some("https://www.longer-than-twelve-bytes.example/a"),
            None,
            Some("https://short.ex/"),
            Some("not a url at all, but long"),
        ]);
        let input: ArrayRef = Arc::new(values.clone());
        for (pattern, rep) in [
            (r"^https?://(?:www\.)?([^/]+)/.*$", r"\1"),
            (r"^(?i)https?://(?:www\.)?([^/]+)/.*$", r"\1"),
            (r"xyzzy(\d)", r"\1"),
        ] {
            let result = static_replace(Arc::clone(&input), pattern, rep, None);
            let result = result.as_string_view();
            assert!(Arc::ptr_eq(result.data_buffers(), values.data_buffers()));
            assert_eq!(result.null_count(), 1);
        }
        let result =
            static_replace(input, r"^https?://(?:www\.)?([^/]+)/.*$", r"\1", None);
        let expected = StringViewArray::from(vec![
            Some("longer-than-twelve-bytes.example"),
            None,
            Some("short.ex"),
            Some("not a url at all, but long"),
        ]);
        assert_eq!(result.as_string_view(), &expected);
    }
}
