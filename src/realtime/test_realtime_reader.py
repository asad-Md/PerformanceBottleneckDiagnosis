from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.realtime.attribution_model import BottleneckAttributor
from src.realtime.realtime_reader import RealtimeReader, engineer_features_v5
from src.realtime.utils import run_bpftool_clear
from src.realtime.config import load_config


class BpfMapClearTests(unittest.TestCase):
    @patch("src.realtime.utils.os.path.exists", return_value=True)
    @patch("src.realtime.utils.subprocess.run")
    def test_deletes_captured_keys_with_one_batch(self, run, _exists) -> None:
        batch_text: list[str] = []

        def read_batch(command, **kwargs):
            self.assertEqual(command[:3], ["bpftool", "batch", "file"])
            batch_text.append(Path(command[3]).read_text(encoding="utf-8"))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        run.side_effect = read_batch
        run_bpftool_clear([
            ("/sys/fs/bpf/sched_map", [{"key": ["0x01", "0x00", "0x02"]}]),
            ("/sys/fs/bpf/mem_map", [{"key": [3, "0", 255]}]),
        ])

        run.assert_called_once()
        self.assertEqual(
            batch_text,
            [
                "map delete pinned /sys/fs/bpf/sched_map key hex 0x01 0x00 0x02\n"
                "map delete pinned /sys/fs/bpf/mem_map key hex 0x03 0x00 0xff\n"
            ],
        )

    @patch("src.realtime.utils.os.path.exists", return_value=True)
    @patch("src.realtime.utils.subprocess.run")
    def test_surfaces_bpftool_reset_failure(self, run, _exists) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["bpftool"], 1, stdout="", stderr="Operation not supported"
        )

        with self.assertRaisesRegex(RuntimeError, "Operation not supported"):
            run_bpftool_clear([("/sys/fs/bpf/array_map", [{"key": ["0x00"]}])])

    @patch("src.realtime.utils.os.path.exists", return_value=True)
    @patch("src.realtime.utils.subprocess.run")
    def test_rejects_malformed_keys_before_running_bpftool(self, run, _exists) -> None:
        with self.assertRaisesRegex(RuntimeError, "no raw byte-array key"):
            run_bpftool_clear([("/sys/fs/bpf/sched_map", [{"key": {"pid": 1}}])])
        run.assert_not_called()


class SnapshotResetTests(unittest.TestCase):
    def _reader(self, *, collect_timings: bool = False) -> RealtimeReader:
        reader = RealtimeReader.__new__(RealtimeReader)
        reader.config = SimpleNamespace(
            sched_map_path="/sys/fs/bpf/sched_map",
            mem_map_path="/sys/fs/bpf/mem_map",
            syscall_map_path="/sys/fs/bpf/syscall_map",
            lock_map_path="/sys/fs/bpf/lock_map",
        )
        reader.collect_timings = collect_timings
        reader.debug_enabled = False
        reader.model_features = ["feature"]
        return reader

    @patch("src.realtime.realtime_reader.engineer_features_v5")
    @patch("src.realtime.realtime_reader.run_bpftool_clear")
    @patch("src.realtime.realtime_reader.run_bpftool_dump")
    def test_reads_all_maps_before_clearing_and_processing(self, dump, clear, engineer) -> None:
        events: list[str] = []
        map_paths = [
            "/sys/fs/bpf/sched_map",
            "/sys/fs/bpf/mem_map",
            "/sys/fs/bpf/syscall_map",
            "/sys/fs/bpf/lock_map",
        ]

        def do_dump(path):
            events.append(f"dump:{path}")
            return [{"key": ["0x01"]}]

        def do_clear(map_entries):
            events.append("clear")
            self.assertEqual([path for path, _ in map_entries], map_paths)

        def do_aggregate(_snapshot):
            events.append("aggregate")
            return pd.DataFrame({"pid": [1]})

        def do_engineer(rows):
            events.append("feature")
            return rows.assign(feature=1)

        dump.side_effect = do_dump
        clear.side_effect = do_clear
        engineer.side_effect = do_engineer
        reader = self._reader()
        reader._rows_from_snapshot = do_aggregate
        reader._validate_reader_columns = lambda _rows: None

        result = reader.read_once()

        self.assertEqual(events, [*(f"dump:{path}" for path in map_paths), "clear", "aggregate", "feature"])
        self.assertEqual(result["rows"]["feature"].tolist(), [1])
        self.assertEqual(result["features"]["feature"].tolist(), [1])

    @patch("src.realtime.realtime_reader.time.perf_counter_ns", side_effect=[100, 180])
    @patch("src.realtime.realtime_reader.run_bpftool_clear", side_effect=RuntimeError("reset failed"))
    @patch("src.realtime.realtime_reader.run_bpftool_dump", return_value=[{"key": ["0x01"]}])
    def test_records_map_io_time_and_stops_on_reset_failure(self, _dump, _clear, _clock) -> None:
        reader = self._reader(collect_timings=True)
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            reader.read_once()
        self.assertEqual(reader.last_stage_timings_ns["map_read_ns"], 80)


class ShapFeaturePipelineTests(unittest.TestCase):
    def test_config_defaults_to_the_shap_retrained_artifacts(self) -> None:
        config = load_config()

        self.assertEqual(Path(config.model_path).name, "champion_model_v7_shap.pkl")
        self.assertEqual(Path(config.feature_cols_path).name, "feature_cols_v7_shap.json")
        self.assertEqual(Path(config.label_thresholds_path).name, "label_thresholds_v7_shap.json")
        self.assertEqual(Path(config.bpf_object_path).name, "realtime_monitor.bpf.o")
        self.assertEqual(Path(config.attribution_model_path).name, "baseline_original_best.pt")

    def test_engineering_produces_the_selected_classifier_features(self) -> None:
        rows = pd.DataFrame([{
            "ctx_switches": 4,
            "involuntary_switches": 2,
            "voluntary_switches": 1,
            "cpu_migrations": 3,
            "total_runtime_ns": 100,
            "minor_faults": 5,
            "major_faults": 6,
            "mutex_contentions": 7,
            "rwsem_read_contentions": 8,
            "rwsem_write_contentions": 9,
            "avg_mutex_wait_ns": 10,
            "syscall_count": 11,
            "kfree_count": 12,
            "total_alloc_bytes": 13,
            "read_bytes": 14,
            "write_bytes": 15,
            "max_syscall_latency_ns": 16,
            "avg_syscall_latency_ns": 17,
        }])

        featured = engineer_features_v5(rows)

        expected = {
            "runtime_per_switch",
            "cpu_migrations",
            "involuntary_ratio",
            "ctx_switches",
            "voluntary_switches",
            "voluntary_ratio",
            "log_total_runtime_ns",
            "minor_faults",
            "involuntary_switches",
            "mutex_contentions",
            "rwsem_read_contentions",
            "avg_mutex_wait_ns",
            "syscall_count",
            "lock_pressure",
            "kfree_count",
            "alloc_pressure",
            "read_write_ratio",
            "log_max_syscall_latency_ns",
            "log_avg_syscall_latency_ns",
            "major_faults",
        }
        self.assertTrue(expected.issubset(featured.columns))
        self.assertEqual(featured.loc[0, "runtime_per_switch"], 25)
        self.assertEqual(featured.loc[0, "lock_pressure"], 24)
        self.assertEqual(featured.loc[0, "read_write_ratio"], 14 / 16)

    def test_predict_uses_saved_binary_decision_threshold(self) -> None:
        class ThresholdModel:
            classes_ = [0, 1]

            def predict_proba(self, _features):
                return np.array([[0.65, 0.35], [0.60, 0.40]])

            def predict(self, _features):
                return [0, 0]

        reader = RealtimeReader.__new__(RealtimeReader)
        reader.collect_timings = False
        reader.model = ThresholdModel()
        reader.uses_scaled_input = False
        reader.scaler = None
        reader.decision_probability_threshold = 0.37
        reader.read_once = lambda: {"features": pd.DataFrame({"feature": [1, 2]})}

        result = reader.predict()

        self.assertEqual(result["predictions"].tolist(), [0, 1])


class AttributionScalerCompatibilityTests(unittest.TestCase):
    def test_transform_matches_scaler_training_input_format(self) -> None:
        class ArrayScaler:
            def transform(self, values):
                self.values = values
                return values

        class Model(torch.nn.Module):
            def forward(self, features):
                return torch.zeros((features.shape[0], 1))

        attributor = BottleneckAttributor.__new__(BottleneckAttributor)
        attributor.feature_columns = ["feature"]
        attributor.target_columns = ["target"]
        attributor.scaler = ArrayScaler()
        attributor.model = Model()

        result = attributor.predict(pd.DataFrame({"feature": [1.0]}))

        self.assertIsInstance(attributor.scaler.values, np.ndarray)
        self.assertEqual(result.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
