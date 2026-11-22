// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Group lookup in [`ArrowBytesViewMap`] alone, in nanoseconds per input row,
//! across table sizes on both sides of the core's L2 cache, and the partial
//! aggregate's loop of filling a table to 8,192 groups and taking it.
//!
//! A final aggregate on a many-core node probes one large table per partition,
//! all at once, so the cost of a lookup is set by memory latency under load.
//! This runs each case with one table, and with several tables probed at the
//! same time on separate threads, each with its own keys.
//!
//! It uses only the map's public API, so the same file measures any revision.
//!
//! ```text
//! cargo bench -p datafusion-physical-expr-common --bench binary_view_map_probe
//! ```
//!
//! Environment:
//! - `PROBE_THREADS`: comma separated table counts, default `1,8`
//! - `PROBE_MAX_BYTES`: largest table, default 2 GiB
//! - `PROBE_MAX_TOTAL`: skip a case whose tables and inputs together would
//!   exceed this many bytes, default 24 GiB
//! - `PROBE_ROWS`: rows timed per case and table, at least, default 4 Mi

use arrow::array::{ArrayRef, StringViewArray};
use datafusion_physical_expr_common::binary_map::OutputType;
use datafusion_physical_expr_common::binary_view_map::{
    ArrowBytesViewMap, INITIAL_MAP_CAPACITY,
};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const BATCH: usize = 8192;
const KIB: usize = 1024;
const GIB: usize = 1024 * 1024 * 1024;

/// The partial aggregate's fixed group capacity, and the rows it reads
/// between checks of it.
const FLUSH_GROUPS: usize = 8192;
const FLUSH_BATCH: usize = 2048;

#[derive(Clone, Copy, Debug)]
enum Workload {
    /// Every row is a key the table has not seen, into a table reserved for
    /// them: a miss and an insert per row.
    New,
    /// Half the rows are keys the table holds, drawn uniformly, and half are
    /// new keys.
    Half,
    /// Every row is a key the table holds, drawn log-uniformly, so a few keys
    /// take most rows.
    Skewed,
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// A distinct key for `id`, `len` bytes long: hex digits for an inline key,
/// or a URL-like key with a shared prefix and the id near its start.
fn key(id: u64, len: usize) -> String {
    if len <= 12 {
        return format!("{id:0len$x}");
    }
    let mut key = format!("https://example.com/path/{id:016x}/");
    let mut fill = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    while key.len() < len {
        key.push(char::from(b'a' + (fill % 26) as u8));
        fill = fill.rotate_left(5) ^ 0x2545_F491;
    }
    key.truncate(len);
    key
}

fn batches(ids: &[u64], len: usize, rows: usize) -> Vec<ArrayRef> {
    ids.chunks(rows)
        .map(|chunk| {
            Arc::new(StringViewArray::from_iter_values(
                chunk.iter().map(|id| key(*id, len)),
            )) as ArrayRef
        })
        .collect()
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

fn insert(map: &mut ArrowBytesViewMap<usize>, batch: &ArrayRef) {
    let mut next = map.len();
    map.insert_if_new(
        batch,
        |_| {
            next += 1;
            next - 1
        },
        |payload| {
            black_box(payload);
        },
    );
}

/// One table's case: the time spent in timed lookups, the rows they took
/// and the size the table reported at the end.
fn case(
    groups: usize,
    len: usize,
    workload: Workload,
    seed: u64,
) -> (Duration, usize, usize) {
    let mut rng = Rng(seed | 1);
    let target = env("PROBE_ROWS", 4 << 20);
    let mut spent = Duration::ZERO;
    let mut rows = 0;
    let mut size = 0;
    // Keys of one repetition are distinct from every other's, so a
    // repetition never finds a key an earlier one inserted.
    let mut base = 0u64;
    while rows < target {
        let held = (0..groups as u64).map(|id| base + id).collect::<Vec<_>>();
        let (prefill, probe) = match workload {
            Workload::New => (vec![], held),
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

        let mut map = ArrowBytesViewMap::<usize>::with_capacity(
            OutputType::Utf8View,
            INITIAL_MAP_CAPACITY,
        );
        for batch in batches(&prefill, len, BATCH) {
            insert(&mut map, &batch);
        }
        // Room for every key the timed rows add, so no case measures growth.
        let added = match workload {
            Workload::New => groups,
            Workload::Half => groups / 2 + 1,
            Workload::Skewed => 0,
        };
        map.reserve_groups(map.len() + added);
        let probe = batches(&probe, len, BATCH);

        let start = Instant::now();
        for batch in &probe {
            insert(&mut map, batch);
        }
        spent += start.elapsed();
        rows += groups;
        size = map.size();
        black_box(map);
    }
    (spent, rows, size)
}

/// The partial aggregate's loop: rows of keys drawn from many more than
/// 8,192, inserted until the table holds 8,192 groups, which are then taken
/// and the table kept for the next fill. The time includes each take.
fn flush_case(len: usize, seed: u64) -> (Duration, usize, usize) {
    let mut rng = Rng(seed | 1);
    let rows = env("PROBE_ROWS", 4 << 20);
    let ids = (0..rows)
        .map(|_| rng.next() % (1 << 24))
        .collect::<Vec<_>>();
    let input = batches(&ids, len, FLUSH_BATCH);
    let mut map = ArrowBytesViewMap::<usize>::with_capacity(
        OutputType::Utf8View,
        INITIAL_MAP_CAPACITY,
    );
    map.reserve_groups(FLUSH_GROUPS);
    let mut size = 0;
    let start = Instant::now();
    for batch in &input {
        insert(&mut map, batch);
        if map.len() >= FLUSH_GROUPS {
            size = size.max(map.size());
            black_box(map.take_with_capacity(FLUSH_GROUPS));
        }
    }
    (start.elapsed(), rows, size)
}

/// Runs `run` on `tables` threads at once and prints the mean ns per row.
fn report(
    label: String,
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
    let max_bytes = env("PROBE_MAX_BYTES", 2 * GIB);
    let max_total = env("PROBE_MAX_TOTAL", 24 * GIB);

    let mut sizes = std::iter::successors(Some(64 * KIB), |size| Some(size * 4))
        .take_while(|size| *size <= max_bytes)
        .chain(std::iter::once(max_bytes))
        .collect::<Vec<_>>();
    sizes.dedup();

    println!("key_bytes,table_target,workload,groups,tables,table_bytes,ns_per_row");
    for len in [80, 8] {
        for &tables in &threads {
            report(
                format!("{len},{FLUSH_GROUPS},Flush,{FLUSH_GROUPS}"),
                tables,
                |seed| flush_case(len, seed),
            );
        }
        // Roughly what one group costs in the table, its views and its key
        // bytes, so the target is the size the table reaches.
        let per_group = 48 + if len > 12 { len } else { 0 };
        for &size in &sizes {
            let groups = (size / per_group).max(64);
            for workload in [Workload::New, Workload::Half, Workload::Skewed] {
                for &tables in &threads {
                    let label = format!("{len},{size},{workload:?},{groups}");
                    // Table, input keys and their views, at about twice the
                    // target for growth and for the input.
                    let total = tables * (2 * size + groups * (len + 16) * 2);
                    if total > max_total {
                        println!("{label},{tables},skipped,skipped");
                        continue;
                    }
                    report(label, tables, |seed| case(groups, len, workload, seed));
                }
            }
        }
    }
}
