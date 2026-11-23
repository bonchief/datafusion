// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Group lookup in the [`GroupValues`] that `new_group_values` picks, in
//! nanoseconds per input row, across table sizes on both sides of the core's
//! L2 cache, and the partial aggregate's loop of filling a table to 8,192
//! groups and taking it. The sibling of `physical-expr-common`'s
//! `binary_view_map_probe`, for fixed-width and mixed keys.
//!
//! A final aggregate on a many-core node probes one large table per
//! partition, all at once, so the cost of a lookup is set by memory latency
//! under load. This runs each case with one table, and with several tables
//! probed at the same time on separate threads, each with its own keys.
//!
//! It uses only the public `new_group_values`, so the same file measures any
//! revision.
//!
//! ```text
//! cargo bench -p datafusion-physical-plan --bench group_values_probe
//! ```
//!
//! Environment:
//! - `PROBE_KEYS`: comma separated key shapes (`I64`, `I64I32`, `I16I32`,
//!   `I64Str`), default `I64`
//! - `PROBE_THREADS`: comma separated table counts, default `1,8`
//! - `PROBE_MAX_BYTES`: largest table, default 256 MiB
//! - `PROBE_MAX_TOTAL`: skip a case whose tables and inputs together would
//!   exceed this many bytes, default 24 GiB
//! - `PROBE_ROWS`: rows timed per case and table, at least, default 4 Mi
//! - `PROBE_ONLY`: comma separated workloads to run (`Flush`, `New`, `Grow`,
//!   `Half`, `Skewed`), default all

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use arrow::array::{ArrayRef, Int16Array, Int32Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::instant::Instant;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;

const BATCH: usize = 8192;
const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;

/// The partial aggregate's fixed group capacity, and the rows it reads
/// between checks of it.
const FLUSH_GROUPS: usize = 8192;
const FLUSH_BATCH: usize = 2048;

#[derive(Clone, Copy, Debug)]
enum Workload {
    /// Every row is a key the table has not seen, into a table reserved for
    /// them: a miss and an insert per row.
    New,
    /// Every row is a key the table has not seen, into a table that starts
    /// empty and grows, as a final aggregate's table fills: the time
    /// includes each growth.
    Grow,
    /// Half the rows are keys the table holds, drawn uniformly, and half are
    /// new keys.
    Half,
    /// Every row is a key the table holds, drawn log-uniformly, so a few keys
    /// take most rows.
    Skewed,
}

/// The group keys of a case, after ClickBench's heavy group-bys.
#[derive(Clone, Copy, Debug)]
enum Keys {
    /// `UserID`: one `Int64`
    I64,
    /// `WatchID, ClientIP`: `Int64, Int32`
    I64I32,
    /// `SearchEngineID, ClientIP`: `Int16, Int32`
    I16I32,
    /// `UserID, SearchPhrase`: `Int64, Utf8View` of up to 30 bytes
    I64Str,
}

impl Keys {
    fn parse(name: &str) -> Option<Self> {
        match name.trim() {
            "I64" => Some(Self::I64),
            "I64I32" => Some(Self::I64I32),
            "I16I32" => Some(Self::I16I32),
            "I64Str" => Some(Self::I64Str),
            _ => None,
        }
    }

    fn schema(self) -> SchemaRef {
        let field = |name: &str, data_type| Field::new(name, data_type, true);
        Arc::new(Schema::new(match self {
            Self::I64 => vec![field("a", DataType::Int64)],
            Self::I64I32 => {
                vec![field("a", DataType::Int64), field("b", DataType::Int32)]
            }
            Self::I16I32 => {
                vec![field("a", DataType::Int16), field("b", DataType::Int32)]
            }
            Self::I64Str => {
                vec![field("a", DataType::Int64), field("b", DataType::Utf8View)]
            }
        }))
    }

    /// Roughly what one group costs in the table and its stored key, so a
    /// case's target is the size its table reaches.
    fn per_group(self) -> usize {
        match self {
            Self::I64 => 40,
            Self::I64I32 | Self::I16I32 => 48,
            Self::I64Str => 72,
        }
    }

    /// The columns for keys `ids`: distinct ids give distinct keys.
    fn columns(self, ids: &[u64]) -> Vec<ArrayRef> {
        let mixed = || ids.iter().map(|id| mix(*id) as i64);
        match self {
            Self::I64 => vec![Arc::new(Int64Array::from_iter_values(mixed()))],
            Self::I64I32 => vec![
                Arc::new(Int64Array::from_iter_values(mixed())),
                Arc::new(Int32Array::from_iter_values(
                    ids.iter().map(|id| (mix(*id) >> 32) as i32),
                )),
            ],
            // Few search engines, many addresses, as in ClickBench.
            Self::I16I32 => vec![
                Arc::new(Int16Array::from_iter_values(
                    ids.iter().map(|id| (id % 40) as i16),
                )),
                Arc::new(Int32Array::from_iter_values(
                    ids.iter().map(|id| mix(id / 40) as i32),
                )),
            ],
            // Many phrases per user; a phrase is its id's hex digits after a
            // shared stem, 15 to 30 bytes.
            Self::I64Str => vec![
                Arc::new(Int64Array::from_iter_values(
                    ids.iter().map(|id| mix(id / 8) as i64),
                )),
                Arc::new(StringViewArray::from_iter_values(ids.iter().map(|id| {
                    let stem = "search phrase ";
                    let width = 1 + (mix(*id) % 16) as usize;
                    format!("{stem}{:0width$x}", id % 8)
                }))),
            ],
        }
    }
}

/// A bijective mix of a 64-bit id, so keys are spread like real ids.
fn mix(id: u64) -> u64 {
    let mut z = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn batches(keys: Keys, ids: &[u64], rows: usize) -> Vec<Vec<ArrayRef>> {
    ids.chunks(rows).map(|chunk| keys.columns(chunk)).collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn table(keys: Keys) -> Box<dyn GroupValues> {
    match new_group_values(keys.schema(), &GroupClustering::None) {
        Ok(values) => values,
        Err(error) => panic!("{keys:?}: {error}"),
    }
}

fn intern(values: &mut dyn GroupValues, batch: &[ArrayRef], groups: &mut Vec<usize>) {
    if let Err(error) = values.intern(batch, groups) {
        panic!("intern: {error}");
    }
    black_box(&groups);
}

/// One table's case: the time spent in timed lookups, the rows they took
/// and the size the table reported at the end.
fn case(
    keys: Keys,
    groups: usize,
    workload: Workload,
    seed: u64,
) -> (Duration, usize, usize) {
    let mut rng = Rng(seed | 1);
    let target = env("PROBE_ROWS", 4 << 20);
    let mut spent = Duration::ZERO;
    let mut rows = 0;
    let mut size = 0;
    let mut ids = vec![];
    // Keys of one repetition are distinct from every other's, so a
    // repetition never finds a key an earlier one inserted.
    let mut base = 0u64;
    while rows < target {
        let held = (0..groups as u64).map(|id| base + id).collect::<Vec<_>>();
        let (prefill, probe) = match workload {
            Workload::New | Workload::Grow => (vec![], held),
            Workload::Half => {
                let probe = (0..groups as u64)
                    .map(|n| {
                        if n % 2 == 0 {
                            base + rng.next() % groups as u64
                        } else {
                            base + groups as u64 + n
                        }
                    })
                    .collect();
                (held, probe)
            }
            Workload::Skewed => {
                let probe = (0..groups)
                    .map(|_| {
                        let rank = (groups as f64).powf(rng.unit()) as u64;
                        base + rank.saturating_sub(1).min(groups as u64 - 1)
                    })
                    .collect();
                (held, probe)
            }
        };
        base += 2 * groups as u64 + 1;

        let mut values = table(keys);
        for batch in batches(keys, &prefill, BATCH) {
            intern(values.as_mut(), &batch, &mut ids);
        }
        // Room for every key the timed rows add, so no case measures growth.
        let added = match workload {
            Workload::New => groups,
            Workload::Half => groups / 2 + 1,
            Workload::Skewed | Workload::Grow => 0,
        };
        values.reserve_groups(values.len() + added);
        let probe = batches(keys, &probe, BATCH);

        let start = Instant::now();
        for batch in &probe {
            intern(values.as_mut(), batch, &mut ids);
        }
        spent += start.elapsed();
        rows += groups;
        size = values.size();
        black_box(values);
    }
    (spent, rows, size)
}

/// The partial aggregate's loop: rows of keys drawn from many more than
/// 8,192, interned until the table holds 8,192 groups, which are then
/// emitted and the table kept for the next fill. The time includes each
/// emit.
fn flush_case(keys: Keys, seed: u64) -> (Duration, usize, usize) {
    let mut rng = Rng(seed | 1);
    let rows = env("PROBE_ROWS", 4 << 20);
    let ids = (0..rows)
        .map(|_| rng.next() % (1 << 24))
        .collect::<Vec<_>>();
    let input = batches(keys, &ids, FLUSH_BATCH);
    let mut values = table(keys);
    values.reserve_groups(FLUSH_GROUPS);
    let mut groups = vec![];
    let mut size = 0;
    let start = Instant::now();
    for batch in &input {
        intern(values.as_mut(), batch, &mut groups);
        if values.len() >= FLUSH_GROUPS {
            size = size.max(values.size());
            match values.emit_with_capacity(FLUSH_GROUPS) {
                Ok(emitted) => black_box(emitted),
                Err(error) => panic!("emit: {error}"),
            };
        }
    }
    (start.elapsed(), rows, size)
}

/// Runs `run` on `tables` threads at once and prints the mean ns per row.
fn report(
    label: &str,
    tables: usize,
    run: impl Fn(u64) -> (Duration, usize, usize) + Sync,
) {
    let barrier = Barrier::new(tables);
    let results = std::thread::scope(|scope| {
        let barrier = &barrier;
        let run = &run;
        (0..tables)
            .map(|table| {
                scope.spawn(move || {
                    barrier.wait();
                    run(0x5EED + table as u64)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .collect::<Vec<_>>()
    });
    let nanos = results
        .iter()
        .map(|(spent, rows, _)| spent.as_nanos() as f64 / *rows as f64)
        .sum::<f64>()
        / results.len().max(1) as f64;
    let bytes = results.iter().map(|(_, _, size)| *size).max().unwrap_or(0);
    println!("{label},{tables},{bytes},{nanos:.1}");
}

fn main() {
    let threads = std::env::var("PROBE_THREADS")
        .unwrap_or_else(|_| "1,8".to_string())
        .split(',')
        .filter_map(|count| count.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    let shapes = std::env::var("PROBE_KEYS")
        .unwrap_or_else(|_| "I64".to_string())
        .split(',')
        .filter_map(Keys::parse)
        .collect::<Vec<_>>();
    let max_bytes = env("PROBE_MAX_BYTES", 256 * MIB);
    let max_total = env("PROBE_MAX_TOTAL", 24 * GIB);

    let mut sizes = std::iter::successors(Some(64 * KIB), |size| Some(size * 4))
        .take_while(|size| *size <= max_bytes)
        .chain(std::iter::once(max_bytes))
        .collect::<Vec<_>>();
    sizes.dedup();
    // Only the workloads named here (Flush, New, Grow, Half, Skewed), if set.
    let only = std::env::var("PROBE_ONLY").ok();
    let wanted = |name: &str| {
        only.as_ref()
            .is_none_or(|only| only.split(',').any(|item| item.trim() == name))
    };

    println!("keys,table_target,workload,groups,tables,table_bytes,ns_per_row");
    for keys in shapes {
        for &tables in threads.iter().filter(|_| wanted("Flush")) {
            report(
                &format!("{keys:?},{FLUSH_GROUPS},Flush,{FLUSH_GROUPS}"),
                tables,
                |seed| flush_case(keys, seed),
            );
        }
        for &size in &sizes {
            let groups = (size / keys.per_group()).max(64);
            for workload in [
                Workload::New,
                Workload::Grow,
                Workload::Half,
                Workload::Skewed,
            ]
            .into_iter()
            .filter(|workload| wanted(&format!("{workload:?}")))
            {
                for &tables in &threads {
                    let label = format!("{keys:?},{size},{workload:?},{groups}");
                    // Table and input keys, at about twice the target for
                    // growth and for the input.
                    let total = tables * (2 * size + groups * 64 * 2);
                    if total > max_total {
                        println!("{label},{tables},skipped,skipped");
                        continue;
                    }
                    report(&label, tables, |seed| case(keys, groups, workload, seed));
                }
            }
        }
    }
}
