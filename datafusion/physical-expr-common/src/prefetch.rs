// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Software prefetch and the cache size that decides when it pays.
//!
//! A hash table probed one row at a time takes its cache misses one after
//! another: the slot, then the entry the slot names, then the stored key. When
//! the table is larger than the core's L2 cache each of those is a trip to
//! memory, and the core waits for each before it can issue the next. Probing a
//! batch in stages, each stage prefetching what a later stage of a row further
//! ahead will read, lets the misses of different rows overlap.
//!
//! Prefetching a table that fits in L2 only adds instructions, so callers take
//! the staged path only when their table is larger than one core's share of
//! L2, [`l2_cache_bytes`], as ClickHouse does with L2 (`getMinBytesForPrefetch`).

use std::sync::OnceLock;

/// Rows between two stages of a staged probe.
///
/// A stage prefetches what the next stage reads `DISTANCE` rows later, so the
/// distance should cover one memory latency under load (about 100–150 ns with
/// many cores missing at once) with the work the loop does for `DISTANCE`
/// rows whose data is already in cache (about 15–40 ns a row for a byte key).
/// That gives 4 to 10; Velox uses 10 (`kPrefetchDistance`). Too short and the
/// line has not arrived; too long and it may be evicted before use, and each
/// batch's last rows go unprefetched.
pub const DISTANCE: usize = 8;

/// Per-core L2 size assumed where the platform does not report one.
///
/// 1 MiB is the per-core L2 of Graviton 3 and of most current x86 server
/// cores; Graviton 4 has 2 MiB. Too small a value turns the staged probe on
/// for tables that still fit in cache, which costs a few nanoseconds a row;
/// too large a value leaves it off for tables that do not.
const DEFAULT_L2: usize = 1024 * 1024;

/// Hints that the cache line holding `ptr` will be read soon.
///
/// The address may be anything, including dangling or null: a prefetch is a
/// hint that never faults and never changes memory.
#[inline(always)]
pub fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` only hints the cache. PREFETCHT0 does not fault
    // on any address, valid or not, and writes no register or memory. SSE is
    // part of the x86_64 baseline.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            ptr.cast::<i8>(),
        );
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM PLDL1KEEP only hints the cache. It does not fault on any
    // address, valid or not, reads no memory into a register and writes none,
    // so the asm touches no state the compiler tracks beyond its input.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr.cast::<u8>(),
            options(nostack, preserves_flags, readonly)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// One core's share of its L2 cache, in bytes: the L2 size divided by the
/// cores that share it.
///
/// A table that fits in this share stays in L2 while every core probes a
/// table of its own, as a final aggregate does with one partition per core;
/// one larger than it misses. Graviton's L2 is per core (1 MiB on Graviton 3,
/// 2 MiB on Graviton 4), so the share is the whole cache. Apple's is shared by
/// a cluster (16 MiB by 5 performance cores on an M5 Pro), so the share is
/// 3.2 MiB, not 16.
///
/// Read once: on Linux from sysfs (`/sys/devices/system/cpu/cpu0/cache`, the
/// level 2 data or unified cache's `size` and `shared_cpu_list`), on macOS
/// from `sysctl` (`hw.perflevel0.l2cachesize` and `hw.perflevel0.cpusperl2`,
/// the performance cores). Where neither says, 1 MiB.
pub fn l2_cache_bytes() -> usize {
    static L2: OnceLock<usize> = OnceLock::new();
    *L2.get_or_init(|| platform_l2().unwrap_or(DEFAULT_L2))
}

#[cfg(target_os = "linux")]
fn platform_l2() -> Option<usize> {
    let root = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
    std::fs::read_dir(root).ok()?.find_map(|entry| {
        let dir = entry.ok()?.path();
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
        let level = read("level")?;
        let kind = read("type")?;
        let kind = kind.trim();
        if level.trim() != "2" || !(kind == "Unified" || kind == "Data") {
            return None;
        }
        let cores = read("shared_cpu_list")
            .and_then(|list| count_cpus(&list))
            .unwrap_or(1);
        share(parse_size(&read("size")?)?, cores)
    })
}

/// Reads two `sysctl` values through the `sysctl` tool, which needs no
/// `unsafe` call into libc; it runs once per process.
#[cfg(target_os = "macos")]
fn platform_l2() -> Option<usize> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.perflevel0.l2cachesize", "hw.perflevel0.cpusperl2"])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let mut values = text.lines().map(|line| line.trim().parse::<usize>().ok());
    let size = values.next()??;
    let cores = values.next().flatten().unwrap_or(1);
    share(size, cores)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn platform_l2() -> Option<usize> {
    None
}

/// One of `cores`' share of a cache of `size` bytes. Zero is no answer.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", test)),
    expect(dead_code)
)]
fn share(size: usize, cores: usize) -> Option<usize> {
    Some(size / cores.max(1)).filter(|share| *share > 0)
}

/// The number of CPUs in a sysfs CPU list such as `0-3,8`.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
fn count_cpus(list: &str) -> Option<usize> {
    list.trim()
        .split(',')
        .map(|range| match range.split_once('-') {
            Some((first, last)) => {
                let first = first.parse::<usize>().ok()?;
                last.parse::<usize>()
                    .ok()?
                    .checked_sub(first)
                    .map(|n| n + 1)
            }
            None => range.parse::<usize>().ok().map(|_| 1),
        })
        .sum::<Option<usize>>()
        .filter(|count| *count > 0)
}

/// Parses a sysfs cache size such as `2048K` or `1M`. Zero is no answer.
#[cfg_attr(not(any(target_os = "linux", test)), expect(dead_code))]
fn parse_size(text: &str) -> Option<usize> {
    let text = text.trim();
    let (digits, unit) = match text.char_indices().find(|(_, c)| !c.is_ascii_digit()) {
        Some((at, _)) => text.split_at(at),
        None => (text, ""),
    };
    let scale = match unit {
        "" => 1,
        "K" => 1 << 10,
        "M" => 1 << 20,
        "G" => 1 << 30,
        _ => return None,
    };
    digits
        .parse::<usize>()
        .ok()?
        .checked_mul(scale)
        .filter(|size| *size > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sysfs_sizes() {
        assert_eq!(parse_size("2048K\n"), Some(2 << 20));
        assert_eq!(parse_size("1M"), Some(1 << 20));
        assert_eq!(parse_size("65536"), Some(65536));
        assert_eq!(parse_size("0K"), None);
        assert_eq!(parse_size("12Q"), None);
        assert_eq!(parse_size(""), None);
    }

    #[test]
    fn counts_sysfs_cpu_lists() {
        assert_eq!(count_cpus("0\n"), Some(1));
        assert_eq!(count_cpus("0-3,8"), Some(5));
        assert_eq!(count_cpus("4-5,10-11"), Some(4));
        assert_eq!(count_cpus("3-1"), None);
        assert_eq!(count_cpus(""), None);
        assert_eq!(share(16 << 20, 5), Some((16 << 20) / 5));
        assert_eq!(share(1 << 20, 0), Some(1 << 20));
        assert_eq!(share(0, 1), None);
    }

    #[test]
    fn l2_is_positive() {
        assert!(l2_cache_bytes() > 0);
    }

    #[test]
    fn prefetch_accepts_any_address() {
        prefetch(std::ptr::null::<u8>());
        prefetch(std::ptr::without_provenance::<u64>(usize::MAX));
        let value = 7u64;
        prefetch(std::ptr::from_ref(&value));
    }
}
