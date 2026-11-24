// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::record_batch::RecordBatch;

use crate::metrics;

/// Rows a stream skips before its first re-test, in windows of
/// `probe_rows_threshold`. DuckDB retries lookups after 128K to 1M skipped
/// rows (`RadixHTAdaptivity::MIN_LOOKUP_RETRY_ROWS`), Trino re-enables partial
/// aggregation after 200 times its window (`PartialAggregationController`).
const FIRST_RETRY_WINDOWS: usize = 10;

/// Upper bound on the skipped rows between two re-tests, in windows: each
/// re-test that skips again doubles the interval up to this, so input that
/// never repeats spends at most one window in this many aggregating.
const MAX_RETRY_WINDOWS: usize = 160;

/// Windows of aggregated rows the decayed history holds before it is halved.
///
/// A single window is noisy: on ClickBench's URL key a stream that reduces to
/// 0.3-0.4 of its rows overall still meets 100K-row runs above 0.5, so a
/// history as short as one window keeps switching such a stream to skipping.
/// Ten windows hold about a million rows, enough to average over such runs
/// while still following a key pattern that changes for good.
const HISTORY_WINDOWS: usize = 10;

/// Decides, window by window, whether a partial aggregation reduces its input
/// enough to be worth its hash table, or should convert rows to partial states
/// one for one instead.
///
/// The reduction is measured as groups created per input row over a window of
/// `probe_rows_threshold` aggregated rows. A key that returns after its table
/// was flushed, at a fixed group capacity or under memory pressure, counts
/// again, since its group goes downstream again: this is the share of rows the
/// stream hands on. A stream skips when both the last window and a decayed
/// history of earlier windows exceed `probe_ratio_threshold`, so one odd
/// window neither starts nor stops skipping on its own. Before the stream has
/// aggregated [`HISTORY_WINDOWS`] windows it skips only keys that barely
/// repeat, with a ratio more than halfway from the threshold to 1: a key that
/// reduces to near the threshold overall meets windows on both sides of it,
/// and the first of them would otherwise decide for the stream. DuckDB
/// likewise never skips lookups before a million input rows
/// (`MIN_INPUT_ROWS_BEFORE_LOOKUP_SKIP`).
///
/// The decision is never final. A skipping stream aggregates again after an
/// interval of skipped rows and re-tests on a fresh window. Each re-test that
/// skips again doubles the interval, up to [`MAX_RETRY_WINDOWS`]; a window that
/// keeps aggregating resets it. Input whose order changes how much repeats
/// nearby, such as files written in different orders, so settles on what the
/// rows show over the stream rather than on its first rows, as DuckDB's
/// `RadixHTAdaptivity` does with its decayed lookup history and retries.
///
/// Used by [`crate::aggregates::hash_stream::PartialHashAggregateStream`],
/// which re-tests, and by the legacy `GroupedHashAggregateStream`, which keeps
/// skipping once it starts.
pub(super) struct SkipAggregationProbe {
    /// Aggregated rows in a window (from `SessionConfig`).
    probe_rows_threshold: usize,
    /// Groups per input row above which aggregation is skipped (from
    /// `SessionConfig`).
    probe_ratio_threshold: f64,

    /// Input rows and created groups of the current window.
    window_rows: usize,
    window_groups: usize,
    /// Input rows and created groups of earlier windows, halved whenever they
    /// reach [`HISTORY_WINDOWS`] windows' worth, so recent windows weigh most.
    recent_rows: usize,
    recent_groups: usize,

    /// Rows aggregated over all windows.
    aggregated_rows: usize,

    /// Whether the stream currently converts rows instead of aggregating them.
    should_skip: bool,
    /// Rows skipped since skipping began.
    skipped_rows: usize,
    /// Skipped rows after which the stream re-tests.
    retry_rows: usize,

    /// Number of rows output as partial states without aggregation.
    skipped_aggregation_rows: metrics::Count,
}

impl SkipAggregationProbe {
    pub(super) fn new(
        probe_rows_threshold: usize,
        probe_ratio_threshold: f64,
        skipped_aggregation_rows: metrics::Count,
    ) -> Self {
        Self {
            probe_rows_threshold,
            probe_ratio_threshold,
            window_rows: 0,
            window_groups: 0,
            recent_rows: 0,
            recent_groups: 0,
            aggregated_rows: 0,
            should_skip: false,
            skipped_rows: 0,
            retry_rows: Self::first_retry(probe_rows_threshold),
            skipped_aggregation_rows,
        }
    }

    fn first_retry(window: usize) -> usize {
        window.max(1).saturating_mul(FIRST_RETRY_WINDOWS)
    }

    /// Adds the input rows of an aggregated batch and the groups it created,
    /// and decides at the end of each window whether to start skipping.
    pub(super) fn update_state(&mut self, input_rows: usize, new_groups: usize) {
        if self.should_skip {
            return;
        }
        self.window_rows += input_rows;
        self.window_groups += new_groups;
        if self.window_rows < self.probe_rows_threshold {
            return;
        }
        if self.recent_rows >= self.probe_rows_threshold.saturating_mul(HISTORY_WINDOWS) {
            self.recent_rows /= 2;
            self.recent_groups /= 2;
        }
        self.recent_rows += self.window_rows;
        self.recent_groups += self.window_groups;
        self.aggregated_rows = self.aggregated_rows.saturating_add(self.window_rows);
        let ratio = |groups: usize, rows: usize| groups as f64 / rows as f64;
        let threshold = self.probe_ratio_threshold;
        let history = ratio(self.recent_groups, self.recent_rows);
        // Until the history spans HISTORY_WINDOWS, only a ratio halfway from
        // the threshold to 1 (keys that barely repeat) skips.
        let settled = self.aggregated_rows
            >= self.probe_rows_threshold.saturating_mul(HISTORY_WINDOWS);
        self.should_skip = ratio(self.window_groups, self.window_rows) > threshold
            && history > threshold
            && (settled || history > (1.0 + threshold) / 2.0);
        self.window_rows = 0;
        self.window_groups = 0;
        if self.should_skip {
            self.skipped_rows = 0;
        } else {
            self.retry_rows = Self::first_retry(self.probe_rows_threshold);
        }
    }

    pub(super) fn should_skip(&self) -> bool {
        self.should_skip
    }

    /// Record the number of rows that were output directly without aggregation
    pub(super) fn record_skipped(&mut self, batch: &RecordBatch) {
        self.skipped_aggregation_rows.add(batch.num_rows());
        self.skipped_rows += batch.num_rows();
    }

    /// Whether a skipping stream has skipped enough rows to aggregate again
    /// and re-test. Returns true once per interval, and doubles the next.
    pub(super) fn should_resume(&mut self) -> bool {
        if !self.should_skip || self.skipped_rows < self.retry_rows {
            return false;
        }
        self.should_skip = false;
        self.retry_rows = self.retry_rows.saturating_mul(2).min(
            self.probe_rows_threshold
                .max(1)
                .saturating_mul(MAX_RETRY_WINDOWS),
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::grouped_hash_stream::GroupedHashAggregateStream;
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use crate::execution_plan::ExecutionPlan;
    use crate::test::TestMemoryExec;

    use std::sync::Arc;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Result;
    use datafusion_execution::TaskContext;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;
    use futures::StreamExt;

    // Migrated to PartialHashAggregateStream coverage in hash_stream.rs;
    // kept here for the legacy GroupedHashAggregateStream implementation.
    #[tokio::test]
    async fn test_skip_aggregation_probe_not_locked_until_skip() -> Result<()> {
        // Test that the probe is not locked until we actually decide to skip.
        // This allows us to continue evaluating the skip condition across multiple batches.
        //
        // Scenario:
        // - Batch 1: Hits rows threshold but NOT ratio threshold (low cardinality) -> don't skip
        // - Batch 2: Now hits ratio threshold (high cardinality) -> skip
        //
        // Without the fix, the probe would be locked after batch 1, preventing the skip
        // decision from being made on batch 2.

        let schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("value_col", DataType::Int32, false),
        ]));

        // Configure thresholds:
        // - probe_rows_threshold: 100 rows
        // - probe_ratio_threshold: 0.8 (80%)
        let probe_rows_threshold = 100;
        let probe_ratio_threshold = 0.8;

        // Batch 1: 100 rows with only 10 unique groups
        // Ratio: 10/100 = 0.1 (10%) < 0.8 -> should NOT skip
        // This will hit the rows threshold but not the ratio threshold
        let batch1_rows = 100;
        let batch1_groups = 10;
        let mut group_ids_batch1 = Vec::new();
        for i in 0..batch1_rows {
            group_ids_batch1.push((i % batch1_groups) as i32);
        }
        let values_batch1: Vec<i32> = vec![1; batch1_rows];

        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch1)),
                Arc::new(Int32Array::from(values_batch1)),
            ],
        )?;

        // Batch 2: 1000 rows with 1000 unique NEW groups (starting from group 10)
        // After batch 2, total: 1100 rows, 1010 groups, ten windows of history
        // Ratio: 1010/1100 is about 0.918 > 0.8 -> SHOULD decide to skip
        let batch2_rows = 1000;
        let batch2_groups = 1000;
        let group_ids_batch2: Vec<i32> = (batch1_groups..(batch1_groups + batch2_groups))
            .map(|x| x as i32)
            .collect();
        let values_batch2: Vec<i32> = vec![1; batch2_rows];

        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch2)),
                Arc::new(Int32Array::from(values_batch2)),
            ],
        )?;

        // Batch 3: This batch should be skipped since we decided to skip after batch 2
        // 100 rows with 100 unique groups (continuing from where batch 2 left off)
        let batch3_rows = 100;
        let batch3_groups = 100;
        let batch3_start_group = batch1_groups + batch2_groups;
        let group_ids_batch3: Vec<i32> = (batch3_start_group
            ..(batch3_start_group + batch3_groups))
            .map(|x| x as i32)
            .collect();
        let values_batch3: Vec<i32> = vec![1; batch3_rows];

        let batch3 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch3)),
                Arc::new(Int32Array::from(values_batch3)),
            ],
        )?;

        let input_partitions = vec![vec![batch1, batch2, batch3]];

        let runtime = RuntimeEnvBuilder::default().build_arc()?;
        let mut task_ctx = TaskContext::default().with_runtime(runtime);

        // Configure skip aggregation settings
        let mut session_config = task_ctx.session_config().clone();
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
            &datafusion_common::ScalarValue::UInt64(Some(probe_rows_threshold)),
        );
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
            &datafusion_common::ScalarValue::Float64(Some(probe_ratio_threshold)),
        );
        task_ctx = task_ctx.with_session_config(session_config);
        let task_ctx = Arc::new(task_ctx);

        // Create aggregate: COUNT(*) GROUP BY group_col
        let group_expr = vec![(col("group_col", &schema)?, "group_col".to_string())];
        let aggr_expr = vec![Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("value_col", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("count_value")
                .build()?,
        )];

        let exec = TestMemoryExec::try_new(&input_partitions, Arc::clone(&schema), None)?;
        let exec = Arc::new(TestMemoryExec::update_cache(&Arc::new(exec)));

        // Use Partial mode
        let aggregate_exec = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(group_expr),
            aggr_expr,
            vec![None],
            exec,
            Arc::clone(&schema),
        )?;

        // Execute and collect results
        let mut stream =
            GroupedHashAggregateStream::new(&aggregate_exec, &Arc::clone(&task_ctx), 0)?;
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            let batch = result?;
            results.push(batch);
        }

        // Check that skip aggregation actually happened.
        // The key metric is skipped_aggregation_rows.
        let metrics = aggregate_exec.metrics().unwrap();
        let skipped_rows = metrics
            .sum_by_name("skipped_aggregation_rows")
            .map(|m| m.as_usize())
            .unwrap_or(0);

        // We expect batch 3's rows to be skipped (100 rows)
        assert_eq!(
            skipped_rows, batch3_rows,
            "Expected batch 3's rows ({batch3_rows}) to be skipped",
        );

        Ok(())
    }

    #[test]
    fn test_skip_aggregation_probe_retests_and_resumes() {
        let mut probe = SkipAggregationProbe::new(10, 0.5, metrics::Count::new());
        let skip = |probe: &mut SkipAggregationProbe, rows: usize| {
            let mut resumed = false;
            for _ in 0..rows {
                assert!(probe.should_skip());
                probe.skipped_rows += 1;
                resumed = probe.should_resume();
                if resumed {
                    break;
                }
            }
            resumed
        };

        // A first window of unique keys starts skipping.
        probe.update_state(10, 10);
        assert!(probe.should_skip());
        // It re-tests after ten windows of skipped rows, not before.
        assert!(!skip(&mut probe, 99));
        assert!(skip(&mut probe, 1));
        assert!(!probe.should_skip());

        // A re-test that still finds unique keys skips again, and the next
        // interval is twice as long.
        probe.update_state(10, 10);
        assert!(probe.should_skip());
        assert!(!skip(&mut probe, 199));
        assert!(skip(&mut probe, 1));

        // Keys that now repeat: the window is good, so aggregation continues
        // although the history of unique keys is still above the ratio.
        probe.update_state(10, 1);
        assert!(!probe.should_skip());
        // A window that keeps aggregating resets the interval to ten windows.
        probe.update_state(10, 10);
        assert!(probe.should_skip());
        assert!(skip(&mut probe, 100));
    }

    #[test]
    fn test_skip_aggregation_probe_waits_for_history_near_the_threshold() {
        let mut probe = SkipAggregationProbe::new(10, 0.5, metrics::Count::new());
        // 0.6 is above the threshold but below 0.75, halfway to 1: no
        // decision until ten windows of history.
        for _ in 0..9 {
            probe.update_state(10, 6);
            assert!(!probe.should_skip());
        }
        probe.update_state(10, 6);
        assert!(probe.should_skip());
    }

    #[test]
    fn test_skip_aggregation_probe_history_outlasts_a_run_of_bad_windows() {
        let mut probe = SkipAggregationProbe::new(10, 0.5, metrics::Count::new());
        for _ in 0..4 {
            probe.update_state(10, 1);
        }
        // Windows of 0.8 after a history of 0.1: the history keeps the stream
        // aggregating through five of them.
        for _ in 0..5 {
            probe.update_state(10, 8);
            assert!(!probe.should_skip());
        }
        // The sixth moves the history to 52 groups in 100 rows.
        probe.update_state(10, 8);
        assert!(probe.should_skip());
    }

    #[test]
    fn test_skip_aggregation_probe_equality_does_not_skip() {
        // When num_groups / input_rows == probe_ratio_threshold, the `>` boundary
        // means we must NOT skip: equality is not sufficient to trigger skip.
        let threshold_ratio = 0.5_f64;
        let threshold_rows = 10_usize;
        let mut probe = SkipAggregationProbe::new(
            threshold_rows,
            threshold_ratio,
            metrics::Count::new(),
        );

        // 10 rows, 5 groups: ratio = 5/10 = 0.5 exactly equals threshold
        probe.update_state(10, 5);

        assert!(
            !probe.should_skip(),
            "ratio == threshold should not trigger skip (boundary is exclusive)"
        );
    }
}
