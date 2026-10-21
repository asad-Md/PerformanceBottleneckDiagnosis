from __future__ import annotations

import ctypes
import errno
import os
import struct
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
from src.realtime.utils import AtomicMapConsumeError, consume_map_snapshot, run_bpftool_clear
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
        reader._pending_consumed_snapshot = {}
        reader._defer_consumed_commit = False
        return reader

    @patch("src.realtime.realtime_reader.engineer_features_v5")
    @patch("src.realtime.realtime_reader.run_bpftool_dump")
    def test_reads_all_maps_then_atomically_consumes_before_processing(self, dump, engineer) -> None:
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

        def do_consume(candidates):
            events.append("consume")
            self.assertEqual(len(candidates), 4)
            return {name: entries for name, entries in zip(
                ("sched", "mem", "syscall", "lock"),
                ([{"key": [1], "value": {}}] for _ in map_paths),
            )}

        def do_aggregate(_snapshot):
            events.append("aggregate")
            return pd.DataFrame({"pid": [1]})

        def do_engineer(rows):
            events.append("feature")
            return rows.assign(feature=1)

        dump.side_effect = do_dump
        engineer.side_effect = do_engineer
        reader = self._reader()
        reader._consume_snapshot = do_consume
        reader._rows_from_snapshot = do_aggregate
        reader._validate_reader_columns = lambda _rows: None

        result = reader.read_once()

        self.assertEqual(events, [*(f"dump:{path}" for path in map_paths), "consume", "aggregate", "feature"])
        self.assertEqual(result["rows"]["feature"].tolist(), [1])
        self.assertEqual(result["features"]["feature"].tolist(), [1])

    @patch("src.realtime.realtime_reader.time.perf_counter_ns", side_effect=[100, 180])
    @patch("src.realtime.realtime_reader.run_bpftool_dump", return_value=[{"key": ["0x01"]}])
    def test_records_map_io_time_and_stops_on_atomic_consume_failure(self, _dump, _clock) -> None:
        reader = self._reader(collect_timings=True)
        reader._consume_snapshot = lambda _snapshot: (_ for _ in ()).throw(
            RuntimeError("atomic consume failed")
        )
        with self.assertRaisesRegex(RuntimeError, "atomic consume failed"):
            reader.read_once()
        self.assertEqual(reader.last_stage_timings_ns["map_read_ns"], 80)

    def test_partial_atomic_consumption_is_retained_for_next_interval(self) -> None:
        reader = self._reader()
        consumed_part = {name: [] for name in ("sched", "mem", "syscall", "lock")}
        consumed_part["sched"] = [{"key": {"pid": 3, "cpu": 1}, "value": {"ctx_switches": 5}}]
        consume = unittest.mock.Mock(side_effect=[
            AtomicMapConsumeError("mid-map failure", consumed_part),
            {name: [] for name in consumed_part},
        ])
        reader._consume_snapshot = RealtimeReader._consume_snapshot.__get__(reader)
        with patch("src.backend.live_reader.get_map_metadata", return_value={}), \
             patch("src.backend.live_reader._validate_map_metadata"), \
             patch("src.realtime.realtime_reader.consume_map_snapshot", consume):
            with self.assertRaisesRegex(AtomicMapConsumeError, "mid-map failure"):
                reader._consume_snapshot({name: [] for name in consumed_part})
            self.assertEqual(reader._pending_consumed_snapshot["sched"], consumed_part["sched"])
            recovered = reader._consume_snapshot({name: [] for name in consumed_part})

        self.assertEqual(recovered["sched"], consumed_part["sched"])
        self.assertEqual(reader._pending_consumed_snapshot["sched"], consumed_part["sched"])
        reader._commit_consumed_snapshot()
        self.assertEqual(reader._pending_consumed_snapshot, {})

    @patch("src.realtime.realtime_reader.engineer_features_v5", side_effect=RuntimeError("feature failure"))
    @patch("src.realtime.realtime_reader.run_bpftool_dump", return_value=[])
    def test_pending_consumed_values_survive_post_consume_processing_failure(self, _dump, _engineer) -> None:
        reader = self._reader()
        pending = [{"key": {"pid": 8, "cpu": 0}, "value": {"ctx_switches": 6}}]
        reader._pending_consumed_snapshot["sched"] = pending
        reader._rows_from_snapshot = lambda _snapshot: pd.DataFrame({"pid": [8]})
        reader._validate_reader_columns = lambda _rows: None
        reader._consume_snapshot = RealtimeReader._consume_snapshot.__get__(reader)
        consumed = {
            "sched": [{"key": {"pid": 9, "cpu": 0}, "value": {"ctx_switches": 1}}],
            "mem": [], "syscall": [], "lock": [],
        }

        with patch("src.backend.live_reader.get_map_metadata", return_value={}), \
             patch("src.backend.live_reader._validate_map_metadata"), \
             patch("src.realtime.realtime_reader.consume_map_snapshot", return_value=consumed):
            with self.assertRaisesRegex(RuntimeError, "feature failure"):
                reader.read_once()

        self.assertEqual(
            [row["key"]["pid"] for row in reader._pending_consumed_snapshot["sched"]],
            [8, 9],
        )

    @patch("src.realtime.realtime_reader.engineer_features_v5", side_effect=lambda rows: rows)
    def test_pending_consumed_values_survive_model_inference_failure(self, _engineer) -> None:
        reader = self._reader()
        pending = [{"key": {"pid": 8, "cpu": 0}, "value": {"ctx_switches": 6}}]
        reader._pending_consumed_snapshot["sched"] = pending
        reader._read_snapshot = lambda: {
            "sched": [{"key": {"pid": 9, "cpu": 0}}],
            "mem": [], "syscall": [], "lock": [],
        }
        reader._consume_snapshot = RealtimeReader._consume_snapshot.__get__(reader)
        consumed = {
            "sched": [{"key": {"pid": 9, "cpu": 0}, "value": {"ctx_switches": 1}}],
            "mem": [], "syscall": [], "lock": [],
        }
        reader._rows_from_snapshot = lambda _snapshot: pd.DataFrame({"pid": [9], "feature": [1]})
        reader._validate_reader_columns = lambda _rows: None
        reader.uses_scaled_input = False
        reader.scaler = None
        reader.model = SimpleNamespace(
            predict_proba=lambda _features: (_ for _ in ()).throw(RuntimeError("model failure")),
        )

        with patch("src.backend.live_reader.get_map_metadata", return_value={}), \
             patch("src.backend.live_reader._validate_map_metadata"), \
             patch("src.realtime.realtime_reader.consume_map_snapshot", return_value=consumed):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                reader.predict()

        self.assertEqual(
            [row["key"]["pid"] for row in reader._pending_consumed_snapshot["sched"]],
            [8, 9],
        )


class AtomicMapConsumptionTests(unittest.TestCase):
    class FakeLibbpf:
        def __init__(self, values, *, after_consume=None, failing_keys=None, bad_map_info=None):
            self.values = values
            self.after_consume = after_consume
            self.failing_keys = set(failing_keys or ())
            self.bad_map_info = bad_map_info
            self.fd_paths = {}
            self.calls = 0

        def bpf_obj_get(self, path):
            fd = os.open("/dev/null", os.O_RDONLY)
            self.fd_paths[fd] = os.fsdecode(path)
            return fd

        def bpf_obj_get_info_by_fd(self, fd, info_pointer, info_size_pointer):
            from src.realtime.utils import _BpfMapInfoPrefix, _MAP_VALUE_STRUCTS

            path = self.fd_paths[fd]
            info = ctypes.cast(info_pointer, ctypes.POINTER(_BpfMapInfoPrefix)).contents
            map_name = path.rsplit("/", 1)[-1].removesuffix("_map")
            info.type = 1
            info.id = 17
            info.key_size = 8
            info.value_size = _MAP_VALUE_STRUCTS[map_name][0].size
            if self.bad_map_info == map_name:
                info.value_size -= 8
            info.max_entries = 65536
            return 0

        def bpf_map_lookup_and_delete_elem(self, fd, key_buffer, value_buffer):
            path = self.fd_paths[fd]
            key = ctypes.string_at(key_buffer, 8)
            if key in self.failing_keys:
                return -errno.EIO
            value = self.values[path].pop(key, None)
            if value is None:
                return -errno.ENOENT
            ctypes.memmove(value_buffer, value, len(value))
            self.calls += 1
            if self.after_consume:
                self.after_consume(path, key, self.values[path])
            return 0

    @staticmethod
    def key_bytes(pid, cpu):
        return struct.pack("=II", pid, cpu)

    @staticmethod
    def raw_entry(pid, cpu):
        return {"key": list(AtomicMapConsumptionTests.key_bytes(pid, cpu))}

    @staticmethod
    def sched_value(ctx_switches):
        return struct.pack(
            "=11Q16s", ctx_switches, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, b"worker",
        )

    def test_updates_after_dump_are_captured_and_later_updates_survive(self) -> None:
        path = "/fake/sched_map"
        key = self.key_bytes(10, 2)
        map_values = {path: {key: self.sched_value(7)}}

        def update_after_atomic_consume(_path, consumed_key, values):
            values[consumed_key] = self.sched_value(3)

        api = self.FakeLibbpf(map_values, after_consume=update_after_atomic_consume)
        result = consume_map_snapshot(
            [("sched", path, [self.raw_entry(10, 2)])], api=api,
        )

        self.assertEqual(result["sched"][0]["value"]["ctx_switches"], 7)
        self.assertEqual(map_values[path][key], self.sched_value(3))

    def test_pid_reuse_before_consume_returns_new_value_not_stale_dump_value(self) -> None:
        path = "/fake/sched_map"
        key = self.key_bytes(42, 0)
        # The original process exited after the key dump; a new process reused
        # the PID and populated the same ABI key before atomic consumption.
        map_values = {path: {key: self.sched_value(11)}}
        api = self.FakeLibbpf(map_values)

        result = consume_map_snapshot(
            [("sched", path, [self.raw_entry(42, 0)])], api=api,
        )

        self.assertEqual(result["sched"][0]["value"]["ctx_switches"], 11)
        self.assertEqual(map_values[path], {})

    def test_process_exit_before_consume_is_a_benign_missing_key(self) -> None:
        path = "/fake/sched_map"
        api = self.FakeLibbpf({path: {}})

        result = consume_map_snapshot(
            [("sched", path, [self.raw_entry(70, 4)])], api=api,
        )

        self.assertEqual(result["sched"], [])

    def test_formatted_bpftool_candidate_key_is_supported(self) -> None:
        path = "/fake/sched_map"
        key = self.key_bytes(13, 2)
        api = self.FakeLibbpf({path: {key: self.sched_value(12)}})
        entry = {"formatted": {"key": {"pid": 13, "cpu": 2}}}

        result = consume_map_snapshot([("sched", path, [entry])], api=api)

        self.assertEqual(result["sched"][0]["value"]["ctx_switches"], 12)

    def test_partial_failure_reports_and_preserves_unconsumed_keys(self) -> None:
        path = "/fake/sched_map"
        first = self.key_bytes(1, 0)
        second = self.key_bytes(2, 1)
        map_values = {path: {
            first: self.sched_value(3),
            second: self.sched_value(4),
        }}
        api = self.FakeLibbpf(map_values, failing_keys={second})

        with self.assertRaises(AtomicMapConsumeError) as caught:
            consume_map_snapshot(
                [("sched", path, [self.raw_entry(1, 0), self.raw_entry(2, 1)])],
                api=api,
            )

        self.assertIn("after 1 consumed entries", str(caught.exception))
        self.assertEqual(caught.exception.partial_snapshot["sched"][0]["value"]["ctx_switches"], 3)
        self.assertNotIn(first, map_values[path])
        self.assertIn(second, map_values[path])

    def test_descriptor_abi_mismatch_fails_before_deleting_any_candidate(self) -> None:
        path = "/fake/sched_map"
        key = self.key_bytes(5, 0)
        map_values = {path: {key: self.sched_value(8)}}
        api = self.FakeLibbpf(map_values, bad_map_info="sched")

        with self.assertRaisesRegex(AtomicMapConsumeError, "incompatible ABI"):
            consume_map_snapshot(
                [("sched", path, [self.raw_entry(5, 0)])], api=api,
            )

        self.assertEqual(map_values[path][key], self.sched_value(8))
        self.assertEqual(api.calls, 0)


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
