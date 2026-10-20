from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.backend import app as backend
from src.backend import live_reader


class BackendModelIntegrationTests(unittest.TestCase):
    def test_live_reader_pin_base_defaults_and_environment_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(live_reader.get_pin_dir(), Path("/sys/fs/bpf/pbd_test"))

        with patch.dict(
            os.environ,
            {"REALTIME_PIN_BASE": "/sys/fs/bpf/pbd_test_alt"},
            clear=True,
        ):
            self.assertEqual(live_reader.get_pin_dir(), Path("/sys/fs/bpf/pbd_test_alt"))

    @patch("src.backend.live_reader.subprocess.run")
    def test_live_reader_dump_uses_configured_pin_base(self, run) -> None:
        run.return_value = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        with patch.dict(
            os.environ,
            {"REALTIME_PIN_BASE": "/sys/fs/bpf/pbd_test_alt"},
            clear=True,
        ):
            self.assertEqual(live_reader.dump_map("syscall"), [])

        self.assertEqual(
            run.call_args.args[0][-1],
            "/sys/fs/bpf/pbd_test_alt/syscall_map",
        )

    def test_pinner_metadata_path_override_and_default_are_used_for_both_writes(self) -> None:
        pinner_source = (
            Path(__file__).resolve().parents[1] / "realtime" / "pinner.c"
        ).read_text(encoding="utf-8")

        self.assertIn('getenv("REALTIME_META_FILE")', pinner_source)
        self.assertIn("meta_file = META_FILE;", pinner_source)
        self.assertEqual(pinner_source.count("write_session_metadata(meta_file,"), 2)

    def test_attribution_feature_contract_is_fully_mapped(self) -> None:
        directly_observed = {
            feature
            for metrics in live_reader.METRIC_SOURCE_KEYS.values()
            for feature in metrics
        }
        derived = set(live_reader.DERIVED_METRIC_SOURCE_KEYS)
        feature_columns = json.loads(
            backend.ATTRIBUTION_FEATURES_PATH.read_text(encoding="utf-8")
        )

        self.assertFalse(set(feature_columns) - (directly_observed | derived))
        self.assertFalse(backend.CHAMPION_RAW_REQUIRED - (directly_observed | derived))

    def test_decode_supports_formatted_and_raw_bpftool_keys(self) -> None:
        self.assertEqual(
            live_reader.decode({"formatted": {
                "key": {"pid": 9, "cpu": 2}, "value": {"ctx_switches": 4}
            }}),
            ((9, 2), {"ctx_switches": 4}),
        )
        self.assertEqual(
            live_reader.decode({
                "key": ["0x09", "0x00", "0x00", "0x00", "0x02", "0x00", "0x00", "0x00"],
                "value": {"ctx_switches": 4},
            }),
            ((9, 2), {"ctx_switches": 4}),
        )
        self.assertIsNone(live_reader.decode({"key": ["bad"], "value": {}}))

    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_marks_missing_raw_metrics_without_marking_observed_zero(self, dump_map) -> None:
        dump_map.side_effect = [
            [{"formatted": {"key": {"pid": 7, "cpu": 0}, "value": {
                "ctx_switches": 0,
                "voluntary_switches": 0,
                "involuntary_switches": 0,
                "cpu_migrations": 0,
                "total_runtime_ns": 0,
            }}}],
            [],
            [],
            [],
        ]

        process = live_reader.read_live_processes()[0]

        self.assertEqual(process["ctx_switches"], 0)
        self.assertNotIn("ctx_switches", process["missing_metrics"])
        self.assertIn("syscall_count", process["missing_metrics"])
        self.assertNotIn("mutex_contentions", process)
        self.assertNotIn("max_mutex_wait_ns", process)
        self.assertNotIn("rwsem_read_contentions", process)
        self.assertIn("avg_mutex_wait_ns", process["missing_metrics"])

    @patch("src.backend.live_reader.dump_map")
    def test_lock_map_features_preserve_counts_and_only_derive_with_samples(self, dump_map) -> None:
        dump_map.side_effect = [
            [], [], [],
            [{"formatted": {"key": {"pid": 71, "cpu": 2}, "value": {
                "mutex_contentions": 0,
                "mutex_wait_sum_ns": 900,
                "mutex_wait_max_ns": 0,
                "rwsem_read_contentions": 2,
                "rwsem_read_wait_sum_ns": 120,
                "rwsem_write_contentions": 3,
            }}}],
        ]

        process = live_reader.read_live_processes()[0]

        self.assertEqual(process["mutex_contentions"], 0)
        self.assertNotIn("avg_mutex_wait_ns", process)
        self.assertIn("avg_mutex_wait_ns", process["missing_metrics"])
        self.assertEqual(process["max_mutex_wait_ns"], 0)
        self.assertEqual(process["rwsem_read_contentions"], 2)
        self.assertEqual(process["avg_rwsem_read_wait_ns"], 60)
        self.assertEqual(process["rwsem_write_contentions"], 3)

    @patch("src.backend.live_reader.dump_map")
    def test_live_lock_averages_match_training_integer_average(self, dump_map) -> None:
        from src.realtime.realtime_reader import RealtimeReader

        dump_map.side_effect = [
            [], [], [],
            [{"formatted": {"key": {"pid": 72, "cpu": 3}, "value": {
                "mutex_contentions": 2,
                "mutex_wait_sum_ns": 101,
                "mutex_wait_max_ns": 60,
                "rwsem_read_contentions": 3,
                "rwsem_read_wait_sum_ns": 122,
                "rwsem_write_contentions": 0,
            }}}],
        ]

        process = live_reader.read_live_processes()[0]
        training_average = RealtimeReader.__new__(RealtimeReader)._iavg

        self.assertEqual(
            process["avg_mutex_wait_ns"],
            training_average(101, 2),
        )
        self.assertEqual(
            process["avg_rwsem_read_wait_ns"],
            training_average(122, 3),
        )

    @patch("src.backend.live_reader.dump_map")
    def test_reader_does_not_observe_avg_stall_with_zero_latency_count(self, dump_map) -> None:
        dump_map.side_effect = [
            [{"formatted": {"key": {"pid": 17, "cpu": 0}, "value": {
                "stall_ns": 120,
                "latency_count": 0,
                "total_runtime_ns": 0,
            }}}],
            [],
            [],
            [],
        ]

        process = live_reader.read_live_processes()[0]

        self.assertEqual(process["stall_ns"], 120)
        self.assertEqual(process["latency_count"], 0)
        self.assertNotIn("avg_stall_ns", process)
        self.assertIn("avg_stall_ns", process["missing_metrics"])

    @patch("src.backend.live_reader.dump_map")
    def test_reader_preserves_training_pid_cpu_rows(self, dump_map) -> None:
        def entry(cpu: int, value: dict) -> dict:
            return {"formatted": {"key": {"pid": 23, "cpu": cpu}, "value": value}}

        dump_map.side_effect = [
            [
                entry(0, {"ctx_switches": 3, "voluntary_switches": 1,
                          "involuntary_switches": 2, "cpu_migrations": 1,
                          "total_runtime_ns": 100, "stall_ns": 30,
                          "latency_count": 2, "max_stall_ns": 20, "comm": "worker"}),
                entry(1, {"ctx_switches": 5, "voluntary_switches": 2,
                          "involuntary_switches": 3, "cpu_migrations": 2,
                          "total_runtime_ns": 300, "stall_ns": 70,
                          "latency_count": 1, "max_stall_ns": 60, "comm": "worker"}),
            ],
            [
                entry(0, {"minor_faults": 4, "major_faults": 1, "kmalloc_count": 2, "kfree_count": 6,
                          "total_alloc_bytes": 80}),
                entry(1, {"minor_faults": 2, "major_faults": 3, "kmalloc_count": 3, "kfree_count": 5,
                          "total_alloc_bytes": 120}),
            ],
            [
                entry(0, {"total_count": 2, "latency_sum_ns": 100,
                          "latency_max_ns": 70, "read_count": 1, "write_count": 1,
                          "read_bytes": 11, "write_bytes": 13, "mmap_count": 1,
                          "futex_count": 2, "futex_latency_sum_ns": 40,
                          "epoll_count": 1, "epoll_latency_sum_ns": 30,
                          "poll_count": 4, "error_count": 1}),
                entry(1, {"total_count": 3, "latency_sum_ns": 200,
                          "latency_max_ns": 110, "read_count": 2, "write_count": 1,
                          "read_bytes": 17, "write_bytes": 19, "mmap_count": 2,
                          "futex_count": 3, "futex_latency_sum_ns": 60,
                          "epoll_count": 2, "epoll_latency_sum_ns": 30,
                          "poll_count": 5, "error_count": 2}),
            ],
            [
                entry(0, {"mutex_contentions": 2, "mutex_wait_sum_ns": 100,
                          "mutex_wait_max_ns": 150, "rwsem_read_contentions": 4,
                          "rwsem_read_wait_sum_ns": 80, "rwsem_write_contentions": 1}),
                entry(1, {"mutex_contentions": 1, "mutex_wait_sum_ns": 200,
                          "mutex_wait_max_ns": 220, "rwsem_read_contentions": 6,
                          "rwsem_read_wait_sum_ns": 120, "rwsem_write_contentions": 2}),
            ],
        ]

        rows = live_reader.read_live_processes()
        self.assertEqual(len(rows), 2)
        by_cpu = {row["cpu"]: row for row in rows}
        self.assertEqual(by_cpu[0]["ctx_switches"], 3)
        self.assertEqual(by_cpu[1]["ctx_switches"], 5)
        self.assertEqual(by_cpu[0]["total_runtime_ns"], 100)
        self.assertEqual(by_cpu[1]["total_runtime_ns"], 300)
        self.assertEqual(by_cpu[0]["avg_syscall_latency_ns"], 50)
        self.assertEqual(by_cpu[1]["avg_syscall_latency_ns"], 200 // 3)
        self.assertEqual(by_cpu[0]["avg_mutex_wait_ns"], 50)
        self.assertEqual(by_cpu[1]["avg_mutex_wait_ns"], 200)
        self.assertEqual(by_cpu[0]["missing_metrics"], [])
        self.assertEqual(by_cpu[1]["missing_metrics"], [])

    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_decodes_and_averages_training_runq_samples(self, dump_map) -> None:
        dump_map.side_effect = [
            [{"formatted": {"key": {"pid": 44, "cpu": 3}, "value": {
                "runq_ratio_sum": 44, "runq_ratio_count": 3,
                "ctx_switches": 3, "last_update_ns": 1,
            }}}], [], [], [],
        ]

        row = live_reader.read_live_processes()[0]

        # The training reader uses integer division (sum // count).
        self.assertEqual(row["cpu"], 3)
        self.assertEqual(row["avg_runq_ratio"], 14)
        self.assertNotIn("avg_runq_ratio", row["missing_metrics"])
        self.assertNotIn("runq_ratio_sum", row)

    @patch("src.backend.live_reader.time.monotonic_ns", return_value=12_500_000_000)
    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_exposes_genuine_monotonic_sample_time_and_age(
        self, dump_map, _clock
    ) -> None:
        dump_map.side_effect = [
            [{"formatted": {"key": {"pid": 45, "cpu": 2}, "value": {
                "ctx_switches": 1, "last_update_ns": 10_000_000_000,
            }}}], [], [], [],
        ]

        row = live_reader.read_live_processes()[0]

        self.assertEqual(row["sample_timestamp_ns"], 10_000_000_000)
        self.assertEqual(row["sample_timestamp_clock"], "CLOCK_MONOTONIC")
        self.assertEqual(row["sample_timestamp_status"], "available")
        self.assertEqual(row["data_age_seconds"], 2.5)
        self.assertEqual(row["data_status"], "live")

    @patch("src.backend.live_reader.time.monotonic_ns", return_value=50_000_000_000)
    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_marks_old_sample_stale(self, dump_map, _clock) -> None:
        dump_map.side_effect = [
            [{"formatted": {"key": {"pid": 46, "cpu": 1}, "value": {
                "ctx_switches": 1, "last_update_ns": 1_000_000_000,
            }}}], [], [], [],
        ]

        row = live_reader.read_live_processes()[0]

        self.assertEqual(row["sample_timestamp_ns"], 1_000_000_000)
        self.assertEqual(row["data_age_seconds"], 49.0)
        self.assertEqual(row["data_status"], "stale")

    @patch("src.backend.live_reader.time.monotonic_ns", return_value=50_000_000_000)
    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_marks_missing_invalid_and_future_sample_times_unavailable_or_invalid(
        self, dump_map, _clock
    ) -> None:
        def entry(pid: int, value: dict) -> dict:
            return {"formatted": {"key": {"pid": pid, "cpu": 0}, "value": value}}

        dump_map.side_effect = [
            [entry(47, {"ctx_switches": 1}),
             entry(48, {"ctx_switches": 1, "last_update_ns": "not-a-number"}),
             entry(49, {"ctx_switches": 1, "last_update_ns": 60_000_000_000})],
            [], [], [],
        ]

        rows = {row["pid"]: row for row in live_reader.read_live_processes()}

        self.assertEqual(rows[47]["sample_timestamp_status"], "unavailable")
        self.assertIsNone(rows[47]["sample_timestamp_ns"])
        self.assertIsNone(rows[47]["data_age_seconds"])
        self.assertEqual(rows[47]["data_status"], "stale")
        for pid in (48, 49):
            self.assertEqual(rows[pid]["sample_timestamp_status"], "invalid")
            self.assertIsNone(rows[pid]["sample_timestamp_ns"])
            self.assertIsNone(rows[pid]["data_age_seconds"])
            self.assertEqual(rows[pid]["data_status"], "stale")

    def test_champion_live_inference_rejects_missing_runq_observation(self) -> None:
        raw = backend.ProcessMetrics().model_dump()
        raw.pop("avg_runq_ratio")

        result = backend.predict_champion_batch([raw])[0]

        self.assertFalse(result["available"])
        self.assertIn("avg_runq_ratio", result["missing_features"])

    def test_current_map_schema_reports_exact_unsupported_attribution_features(self) -> None:
        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                [{"formatted": {"key": {"pid": 7, "cpu": 0}, "value": {
                    "ctx_switches": 1, "voluntary_switches": 1,
                    "involuntary_switches": 0, "cpu_migrations": 0,
                    "total_runtime_ns": 100,
                }}}],
                [{"formatted": {"key": {"pid": 7, "cpu": 0}, "value": {
                    "minor_faults": 2, "major_faults": 0, "kfree_count": 3,
                    "total_alloc_bytes": 64,
                }}}],
                [{"formatted": {"key": {"pid": 7, "cpu": 0}, "value": {
                    "total_count": 2, "latency_sum_ns": 20, "latency_max_ns": 12,
                    "read_count": 1, "write_count": 1, "read_bytes": 4, "write_bytes": 5,
                }}}],
                [{"formatted": {"key": {"pid": 7, "cpu": 0}, "value": {
                    "mutex_contentions": 2, "mutex_wait_sum_ns": 40,
                    "rwsem_read_contentions": 1, "rwsem_write_contentions": 0,
                }}}],
            ]

            process = live_reader.read_live_processes()[0]

        self.assertEqual(process["avg_syscall_latency_ns"], 10)
        self.assertEqual(process["avg_mutex_wait_ns"], 20)
        self.assertEqual(set(process["missing_metrics"]), {
            "stall_ns", "avg_stall_ns", "max_stall_ns", "latency_count",
            "kmalloc_count", "mmap_count", "futex_count", "avg_futex_latency_ns",
            "epoll_count", "avg_epoll_latency_ns", "poll_count", "syscall_error_count",
            "max_mutex_wait_ns", "avg_rwsem_read_wait_ns",
        })

    def test_reader_marks_zero_denominator_per_syscall_averages_unavailable(self) -> None:
        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                [{"formatted": {"key": {"pid": 31, "cpu": 0}, "value": {
                    "total_runtime_ns": 0,
                }}}],
                [],
                [{"formatted": {"key": {"pid": 31, "cpu": 0}, "value": {
                    "total_count": 0, "latency_sum_ns": 900,
                    "futex_count": 0, "futex_latency_sum_ns": 800,
                    "epoll_count": 0, "epoll_latency_sum_ns": 700,
                }}}],
                [{"formatted": {"key": {"pid": 31, "cpu": 0}, "value": {
                    "mutex_contentions": 0, "mutex_wait_sum_ns": 600,
                    "rwsem_read_contentions": 0, "rwsem_read_wait_sum_ns": 500,
                }}}],
            ]

            process = live_reader.read_live_processes()[0]

        for metric in (
            "avg_syscall_latency_ns", "avg_futex_latency_ns",
            "avg_epoll_latency_ns", "avg_mutex_wait_ns",
            "avg_rwsem_read_wait_ns",
        ):
            self.assertNotIn(metric, process)
            self.assertIn(metric, process["missing_metrics"])

    def test_current_artifact_bundle_loads(self) -> None:
        self.assertEqual(len(backend.CHAMPION_FEATURE_COLS), 33)
        self.assertEqual(
            list(backend.champion_model.feature_names_in_),
            backend.CHAMPION_FEATURE_COLS,
        )
        self.assertEqual(backend.ATTRIBUTION_MODEL_PATH.name, "baseline_original_best.pt")
        self.assertIsNotNone(backend.attributor)
        scaler = joblib.load(backend.ATTRIBUTION_SCALER_PATH)
        feature_columns = json.loads(backend.ATTRIBUTION_FEATURES_PATH.read_text(encoding="utf-8"))
        target_columns = json.loads(backend.ATTRIBUTION_TARGETS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(feature_columns), 33)
        self.assertEqual(backend.attributor.feature_columns, feature_columns)
        self.assertEqual(backend.attributor.scaler.n_features_in_, len(feature_columns))
        self.assertEqual(scaler.n_features_in_, 33)
        self.assertEqual(target_columns, ["CPU", "Memory", "Cache", "IO", "Lock", "ContextSwitch"])

    def test_unavailable_attribution_has_no_prediction_fields(self) -> None:
        unavailable = [
            "stall_ns", "avg_stall_ns", "max_stall_ns", "latency_count",
            "kmalloc_count", "mmap_count", "futex_count", "avg_futex_latency_ns",
            "epoll_count", "avg_epoll_latency_ns", "poll_count", "syscall_error_count",
            "max_mutex_wait_ns", "avg_rwsem_read_wait_ns",
        ]
        result = backend.predict_mlp(
            backend.ProcessMetrics().model_dump(), missing_metrics=unavailable
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["labels"])
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["missing_features"], [
            "avg_epoll_latency_ns", "avg_futex_latency_ns", "avg_rwsem_read_wait_ns",
            "avg_stall_ns", "epoll_count", "futex_count", "kmalloc_count",
            "latency_count", "max_mutex_wait_ns", "max_stall_ns", "mmap_count",
            "poll_count", "stall_ns", "syscall_error_count",
        ])
        self.assertIn("Required live attribution features are unavailable", result["error"])

    def test_attribution_uses_exact_artifact_feature_order(self) -> None:
        seen: dict[str, object] = {}

        class Attributor:
            feature_columns = ["second", "first"]
            target_columns = ["CPU"]
            THRESHOLD = 0.5

            def predict(self, features):
                seen["columns"] = list(features.columns)
                seen["values"] = features.iloc[0].tolist()
                return np.array([[0.75]])

        with patch.object(backend, "attributor", Attributor()):
            result = backend.predict_mlp({"first": 11, "second": 22})

        self.assertEqual(seen["columns"], ["second", "first"])
        self.assertEqual(seen["values"], [22, 11])
        self.assertTrue(result["available"])
        self.assertEqual(result["labels"], ["CPU"])
        self.assertEqual(result["missing_features"], [])

    def test_champion_uses_saved_order_for_explicit_metric_values(self) -> None:
        seen: dict[str, object] = {}

        class Model:
            classes_ = np.array([0, 1])

            def predict_proba(self, features):
                seen["columns"] = list(features.columns)
                seen["values"] = features.iloc[0].to_dict()
                return np.array([[0.7, 0.3]])

        with patch.object(backend, "champion_model", Model()):
            raw = backend.ProcessMetrics(pid=123, ctx_switches=4).model_dump()
            result = backend.predict_champion(raw)

        self.assertEqual(seen["columns"], backend.CHAMPION_FEATURE_COLS)
        values = seen["values"]
        self.assertEqual(values["ctx_switches"], 4)
        self.assertEqual(values["avg_runq_ratio"], 0)
        self.assertTrue(result["available"])
        self.assertEqual(result["prediction"], "Not-High")
        self.assertEqual(result["high_probability"], 0.3)

    def test_submitted_prediction_does_not_treat_omitted_fields_as_observed_zeros(self) -> None:
        result = backend.predict(backend.ProcessMetrics(ctx_switches=10))

        self.assertFalse(result["severity_prediction"]["available"])
        self.assertIn("avg_runq_ratio", result["severity_prediction"]["missing_features"])

    @patch("src.backend.app.predict_mlp")
    @patch("src.backend.app.predict_champion_batch")
    @patch("src.backend.app.read_live_processes")
    def test_live_response_keeps_severity_when_attribution_is_unavailable(
        self, read, predict_champion_batch, predict_mlp
    ) -> None:
        read.return_value = [{"pid": 123, "cpu": 0, "comm": "fixture", "session_label": "live_ebpf", "data_status": "live", "sample_timestamp_ns": 123_456_789, "sample_timestamp_clock": "CLOCK_MONOTONIC", "sample_timestamp_status": "available", "data_age_seconds": 0.25, "ctx_switches": 10, "missing_metrics": ["avg_syscall_latency_ns"]}]
        predict_mlp.return_value = {
            "available": False, "status": "unavailable", "model": "baseline_original_best.pt",
            "probabilities": None, "labels": None, "threshold": None,
            "error": "Required live attribution features are unavailable: stall_ns",
            "missing_features": ["stall_ns"],
        }
        predict_champion_batch.return_value = [{
            "available": True,
            "status": "available",
            "pred_class": 1,
            "prediction": "High",
            "is_bottleneck": True,
            "high_probability": 0.7,
            "threshold": 0.5,
        }]

        payload = backend.live()

        self.assertEqual(payload["source"], "live_ebpf")
        self.assertEqual(payload["count"], 1)
        process = payload["processes"][0]
        self.assertFalse(process["attribution_prediction"]["available"])
        self.assertEqual(process["attribution_prediction"]["missing_features"], ["stall_ns"])
        self.assertEqual(process["severity_prediction"]["prediction"], "High")
        self.assertEqual(process["process"]["pid"], 123)
        self.assertEqual(process["sample_timestamp_ns"], "123456789")
        self.assertEqual(process["sample_timestamp_clock"], "CLOCK_MONOTONIC")
        self.assertEqual(process["sample_timestamp_status"], "available")
        self.assertEqual(process["data_age_seconds"], 0.25)
        self.assertNotIn("sample_timestamp_ns", process["metrics"])
        self.assertIn("models", payload)
        self.assertIn("avg_syscall_latency_ns", process["missing_metrics"])
        self.assertIn("stall_ns", process["missing_metrics"])

    @patch("src.backend.app.read_live_processes", return_value=[])
    def test_live_response_supports_empty_metrics(self, _read) -> None:
        response = backend.live()
        self.assertEqual(response["count"], 0)
        self.assertEqual(response["processes"], [])

    @patch("src.backend.app.read_live_processes", side_effect=RuntimeError("maps unavailable"))
    def test_unavailable_live_metrics_return_service_unavailable(self, _read) -> None:
        with self.assertRaises(HTTPException) as caught:
            backend.live()
        self.assertEqual(caught.exception.status_code, 503)

    def test_missing_and_incompatible_champion_artifacts_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features_path = root / "features.json"
            thresholds_path = root / "thresholds.json"
            features_path.write_text(json.dumps(backend.CHAMPION_FEATURE_COLS), encoding="utf-8")
            thresholds_path.write_text(
                json.dumps({"decision_probability_threshold": 0.5}), encoding="utf-8"
            )

            with self.assertRaises(FileNotFoundError):
                backend.load_champion_artifacts(
                    root / "missing.pkl", features_path, thresholds_path
                )

            bad_model_path = root / "bad.pkl"
            joblib.dump(SimpleNamespace(feature_names_in_=["wrong"], classes_=[0, 1]), bad_model_path)
            with self.assertRaisesRegex(ValueError, "training order"):
                backend.load_champion_artifacts(bad_model_path, features_path, thresholds_path)


if __name__ == "__main__":
    unittest.main()
