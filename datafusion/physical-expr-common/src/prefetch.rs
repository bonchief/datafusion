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
//! the staged path only when their table is larger than [`l2_cache_bytes`], as
//! ClickHouse does (`getMinBytesForPrefetch`).

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

/// The L2 data cache of one core, in bytes.
///
/// Read once from Linux sysfs (`/sys/devices/system/cpu/cpu0/cache`); on other
/// platforms, or when sysfs does not say, 1 MiB.
pub fn l2_cache_bytes() -> usize {
    static L2: OnceLock<usize> = OnceLock::new();
    *L2.get_or_init(|| sysfs_l2().unwrap_or(DEFAULT_L2))
}

#[cfg(target_os = "linux")]
fn sysfs_l2() -> Option<usize> {
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
        parse_size(&read("size")?)
    })
}

#[cfg(not(target_os = "linux"))]
fn sysfs_l2() -> Option<usize> {
    None
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
    fn l2_is_positive() {
        assert!(l2_cache_bytes() > 0);
    }

    #[test]
    fn prefetch_accepts_any_address() {
        prefetch(std::ptr::null::<u8>());
        prefetch(std::ptr::without_provenance::<u64>(usize::MAX));
        let value = 7u64;
        prefetch(&value);
    }
}
