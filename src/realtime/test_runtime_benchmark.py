from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.realtime.plot_runtime_benchmark import generate_plots
from src.realtime.predictor import Predictor
from src.realtime.runtime_benchmark import (
    CYCLE_FIELDS,
    RESOURCE_FIELDS,
    AsyncCsvWriter,
)
from src.realtime.summarize_runtime_benchmark import _overhead_group_summary, summarize

os.environ.setdefault("MPLBACKEND", "Agg")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_cycle(run_id: str, cycle_id: int, total_ms: float, interval_ms: float | None) -> dict[str, object]:
    return {
        "run_id": run_id,
        "comparison_id": "pair-1",
        "repetition_index": 1,
        "condition": "ebpf_plus_diagnosis",
        "workload_type": "synthetic_fixture",
        "intensity": "small",
        "workload_workers": 1,
        "cycle_id": cycle_id,
        "phase": "measurement",
        "cycle_start_ns": 1_000_000 * cycle_id,
        "cycle_end_ns": 1_000_000 * cycle_id + int(total_ms * 1_000_000),
        "map_read_ms": 0.1,
        "aggregation_ms": 0.2,
        "feature_ms": 0.3,
        "rf_inference_ms": 0.4,
        "mlp_inference_ms": "",
        "output_ms": 0.5,
        "detection_latency_ms": total_ms - 0.5,
        "total_cycle_ms": total_ms,
        "records_processed": 3,
        "high_predictions": 1,
        "configured_interval_ms": 1000,
        "actual_cycle_interval_ms": interval_ms,
        "deadline_missed": "",
        "cycle_status": "completed",
        "error": "",
    }


def _base_resource(run_id: str, condition: str, cpu: float, memory: int, monitor_rss: int | None) -> dict[str, object]:
    return {
        "run_id": run_id,
        "resource_run_id": run_id,
        "comparison_id": "pair-1",
        "repetition_index": 1,
        "condition": condition,
        "workload_type": "synthetic_fixture",
        "intensity": "small",
        "workload_workers": 1,
        "timestamp_ns": 1_000_000_000,
        "elapsed_s": 1,
        "phase": "measurement",
        "system_cpu_percent": cpu,
        "system_memory_used_bytes": memory,
        "system_memory_used_percent": 50,
        "workload_pid": 10,
        "workload_process_cpu_percent_one_core": 80,
        "workload_process_rss_bytes": 1000,
        "monitor_pid": 20 if monitor_rss else "",
        "monitor_process_cpu_percent_one_core": 5 if monitor_rss else "",
        "monitor_process_rss_bytes": monitor_rss or "",
    }


class _FakeReader:
    collect_timings = True
    model = object()
    last_stage_timings_ns = {
        "map_read_ns": 100_000,
        "aggregation_ns": 200_000,
        "feature_ns": 300_000,
        "rf_inference_ns": 400_000,
    }

    def predict(self) -> dict[str, object]:
        return {
            "rows": pd.DataFrame([{"pid": 123, "cpu": 0, "comm": "fixture"}]),
            "predictions": [1],
            "probabilities": [[0.1, 0.9]],
        }


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_async_writer_flushes_rows_and_keeps_unmeasured_stage_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cycle_metrics.csv"
            writer = AsyncCsvWriter(path, CYCLE_FIELDS)
            writer.submit({"run_id": "fixture", "mlp_inference_ms": None, "cycle_status": "completed"})
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "fixture")
            self.assertEqual(rows[0]["mlp_inference_ms"], "")

    def test_existing_loop_reports_measured_cycle_fields_without_deadline(self) -> None:
        predictor = Predictor.__new__(Predictor)
        predictor.reader = _FakeReader()
        predictor.config = SimpleNamespace(polling_interval_s=1.0)
        predictor.cleanup_maps_on_exit = False
        predictor.class_names = {0: "Not-High", 1: "High"}
        predictor.bottleneck_class = 1
        cycles: list[dict[str, object]] = []
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.run_benchmark(cycles.append, duration_s=0.002, warmup_s=0)
        self.assertGreaterEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertEqual(cycle["cycle_status"], "completed")
        self.assertEqual(cycle["records_processed"], 1)
        self.assertEqual(cycle["high_predictions"], 1)
        self.assertIsNone(cycle["mlp_inference_ms"])
        self.assertIsNone(cycle["deadline_missed"])
        self.assertIsNone(cycle["actual_cycle_interval_ms"])
        self.assertAlmostEqual(cycle["map_read_ms"], 0.1)
        self.assertGreaterEqual(cycle["total_cycle_ms"], cycle["detection_latency_ms"])
        self.assertGreater(cycle["output_ms"], 0)

    def test_actual_interval_is_start_to_start_between_cycles(self) -> None:
        predictor = Predictor.__new__(Predictor)
        predictor.reader = _FakeReader()
        predictor.config = SimpleNamespace(polling_interval_s=0.001)
        predictor.cleanup_maps_on_exit = False
        predictor.class_names = {0: "Not-High", 1: "High"}
        predictor.bottleneck_class = 1
        cycles: list[dict[str, object]] = []
        with contextlib.redirect_stdout(io.StringIO()):
            predictor.run_benchmark(cycles.append, duration_s=0.004, warmup_s=0)
        self.assertGreaterEqual(len(cycles), 2)
        self.assertIsNone(cycles[0]["actual_cycle_interval_ms"])
        self.assertGreater(cycles[1]["actual_cycle_interval_ms"], 0)

    def test_benchmark_cleanup_does_not_unpin_maps(self) -> None:
        predictor = Predictor.__new__(Predictor)
        predictor.cleanup_maps_on_exit = False
        with patch("src.realtime.predictor.unpin_maps") as unpin:
            with contextlib.redirect_stdout(io.StringIO()):
                predictor.cleanup()
        unpin.assert_not_called()

    def test_summary_preserves_failed_and_incomplete_runs_and_calculates_paired_overhead(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_rows = [
                {
                    "run_id": "monitored",
                    "comparison_id": "pair-1",
                    "repetition_index": 1,
                    "condition": "ebpf_plus_diagnosis",
                    "workload_type": "synthetic_fixture",
                    "intensity": "small",
                    "workload_workers": 1,
                    "status": "completed",
                    "configured_interval_ms": 1000,
                    "measurement_duration_s": 2,
                    "started_at": "start",
                    "ended_at": "end",
                },
                {
                    "run_id": "baseline",
                    "comparison_id": "pair-1",
                    "repetition_index": 1,
                    "condition": "workload_only",
                    "workload_type": "synthetic_fixture",
                    "intensity": "small",
                    "workload_workers": 1,
                    "status": "completed",
                    "configured_interval_ms": 1000,
                    "measurement_duration_s": 2,
                    "started_at": "start",
                    "ended_at": "end",
                },
                {
                    "run_id": "interrupted",
                    "workload_type": "synthetic_fixture",
                    "intensity": "small",
                    "configured_interval_ms": 1000,
                    "condition": "workload_only",
                    "status": "running",
                },
                {
                    "run_id": "failed",
                    "workload_type": "synthetic_fixture",
                    "intensity": "small",
                    "configured_interval_ms": 1000,
                    "condition": "workload_only",
                    "status": "failed",
                    "error": "sample failure",
                },
            ]
            for config in config_rows:
                (root / f"run_{config['run_id']}.json").write_text(json.dumps(config), encoding="utf-8")

            _write_csv(
                root / "cycle_metrics.csv",
                CYCLE_FIELDS,
                [
                    _base_cycle("monitored", 1, 2.0, None),
                    _base_cycle("monitored", 2, 4.0, 1001.0),
                ],
            )
            _write_csv(
                root / "system_resource_metrics.csv",
                RESOURCE_FIELDS,
                [
                    _base_resource("baseline", "workload_only", 40.0, 10000, None),
                    _base_resource("monitored", "ebpf_plus_diagnosis", 44.0, 11000, 5000),
                ],
            )

            runs, groups, overhead = summarize(root)
            by_id = runs.set_index("run_id")
            self.assertEqual(by_id.loc["interrupted", "status"], "incomplete")
            self.assertEqual(by_id.loc["failed", "status"], "failed")
            self.assertEqual(by_id.loc["monitored", "n_cycles"], 2)
            self.assertAlmostEqual(by_id.loc["monitored", "median_total_cycle_ms"], 3.0)
            self.assertAlmostEqual(by_id.loc["monitored", "p95_total_cycle_ms"], 3.9)
            self.assertEqual(len(groups), 2)
            self.assertEqual(len(overhead), 1)
            self.assertAlmostEqual(overhead.iloc[0]["system_cpu_delta_percentage_points"], 4.0)
            self.assertAlmostEqual(overhead.iloc[0]["system_memory_delta_bytes"], 1000.0)
            overhead_groups = _overhead_group_summary(overhead)
            self.assertEqual(overhead_groups.iloc[0]["n_matched_repetitions"], 1)
            self.assertAlmostEqual(overhead_groups.iloc[0]["mean_system_cpu_delta_percentage_points"], 4.0)
            report = (root / "analysis" / "benchmark_report.md").read_text(encoding="utf-8")
            self.assertIn("no independently verified bottleneck onset timestamp", report)
            self.assertIn("Matched repetitions: 1", report)
            self.assertIn("not measured / not measured ms", report)

    def test_plots_are_generated_only_from_fixture_csv_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            analysis.mkdir()
            _write_csv(
                root / "cycle_metrics.csv",
                CYCLE_FIELDS,
                [
                    _base_cycle("fixture-1", 1, 2.0, None),
                    _base_cycle("fixture-1", 2, 3.0, 1001.0),
                ],
            )
            _write_csv(
                analysis / "overhead_summary.csv",
                [
                    "comparison_id",
                    "workload_type",
                    "intensity",
                    "system_cpu_delta_percentage_points",
                    "monitor_process_rss_bytes",
                ],
                [{
                    "comparison_id": "fixture-pair",
                    "workload_type": "synthetic_fixture",
                    "intensity": "small",
                    "system_cpu_delta_percentage_points": 2,
                    "monitor_process_rss_bytes": 5000,
                }],
            )
            generated = generate_plots(root)
            self.assertEqual(len(generated), 6)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in generated))


if __name__ == "__main__":
    unittest.main()
