// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Group-id cost of a single `BinaryView` key, per row, over batches read
//! from Arrow IPC files, as a partial aggregate sees them: `intern` each
//! batch, and emit all groups once the table holds 8,192 of them. Only
//! `intern` is timed.
//!
//! usage: `view_dedupe <reps> <file.arrow>...`; prints, per file, the share of
//! rows repeating a view in their batch and the ns per row of every rep.

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use datafusion_expr::EmitTo;
use datafusion_physical_plan::aggregates::group_values::new_group_values;
use datafusion_physical_plan::aggregates::order::GroupClustering;
use std::collections::HashSet;
use std::fs::File;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const FLUSH: usize = 8192;

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let reps: usize = args[0].parse().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::BinaryView,
        true,
    )]));
    for path in &args[1..] {
        let batches: Vec<ArrayRef> = FileReader::try_new(File::open(path).unwrap(), None)
            .unwrap()
            .map(|batch| Arc::clone(batch.unwrap().column(0)))
            .collect();
        let rows: usize = batches.iter().map(|b| b.len()).sum();
        let repeats: usize = batches
            .iter()
            .map(|b| {
                let views = b.as_binary_view().views();
                let mut seen = HashSet::with_capacity(views.len());
                views.iter().filter(|v| !seen.insert(**v)).count()
            })
            .sum();
        let mut times = vec![];
        for _ in 0..reps {
            let mut values =
                new_group_values(Arc::clone(&schema), &GroupClustering::None).unwrap();
            let mut groups = Vec::with_capacity(FLUSH);
            let mut nanos = 0u128;
            for batch in &batches {
                let start = Instant::now();
                values
                    .intern(std::slice::from_ref(batch), &mut groups)
                    .unwrap();
                nanos += start.elapsed().as_nanos();
                black_box(&groups);
                if values.len() >= FLUSH {
                    black_box(values.emit(EmitTo::All).unwrap());
                }
            }
            times.push(nanos as f64 / rows as f64);
        }
        let name = path.rsplit('/').next().unwrap();
        let list: Vec<String> = times.iter().map(|t| format!("{t:.2}")).collect();
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "{name:<32} rows {rows:>9} view repeats {:>5.1}% | ns/row median {:.2} min {:.2} | {}",
            100.0 * repeats as f64 / rows as f64,
            sorted[sorted.len() / 2],
            sorted[0],
            list.join(" ")
        );
    }
}
