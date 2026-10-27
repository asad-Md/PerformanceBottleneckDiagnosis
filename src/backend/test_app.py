from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.backend import app as backend
from src.backend import live_reader


def _map_metadata_fixture(name: str) -> dict:
    value_struct, value_size = live_reader.MAP_VALUE_STRUCTS[name]
    return {
        "id": 19,
        "type": "hash",
        "name": live_reader.MAPS[name],
        "bytes_key": live_reader.MAP_ABI[name][0],
        "bytes_value": live_reader.MAP_ABI[name][1],
        "btf_id": 207,
        "map_symbol": live_reader.MAPS[name],
        "key_struct": "agg_key",
        "key_fields": ["pid", "cpu"],
        "value_struct": value_struct,
        "value_struct_size": value_size,
        "value_fields": sorted(live_reader.MAP_VALUE_FIELDS[name]),
    }


def _raw_btf_fixture(
    map_name: str,
    *,
    value_struct: str | None = None,
    value_size: int | None = None,
    value_fields: list[str] | None = None,
    linked_value_struct: str | None = None,
) -> str:
    value_struct = value_struct or live_reader.MAP_VALUE_STRUCTS[map_name][0]
    value_size = value_size or live_reader.MAP_VALUE_STRUCTS[map_name][1]
    value_fields = value_fields or sorted(live_reader.MAP_VALUE_FIELDS[map_name])
    linked_value_struct = linked_value_struct or value_struct
    lines = [
        "[1] INT '__u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)",
        "[2] STRUCT 'agg_key' size=8 vlen=2",
        "    'pid' type_id=1 bits_offset=0",
        "    'cpu' type_id=1 bits_offset=32",
        f"[3] STRUCT '{value_struct}' size={value_size} vlen={len(value_fields)}",
    ]
    lines.extend(f"    '{field}' type_id=1 bits_offset={index * 64}"
                 for index, field in enumerate(value_fields))
    linked_type_id = 3
    if linked_value_struct != value_struct:
        linked_type_id = 5
        lines.append(
            f"[5] STRUCT '{linked_value_struct}' size={value_size} vlen={len(value_fields)}"
        )
        lines.extend(f"    '{field}' type_id=1 bits_offset={index * 64}"
                     for index, field in enumerate(value_fields))
    lines.extend([
        "[4] PTR '(anon)' type_id=2",
        f"[{6 if linked_type_id == 5 else 5}] PTR '(anon)' type_id={linked_type_id}",
        f"[{7 if linked_type_id == 5 else 6}] STRUCT '(anon)' size=16 vlen=2",
        "    'key' type_id=4 bits_offset=0",
        f"    'value' type_id={6 if linked_type_id == 5 else 5} bits_offset=64",
        f"[{8 if linked_type_id == 5 else 7}] VAR '{live_reader.MAPS[map_name]}' "
        f"type_id={7 if linked_type_id == 5 else 6} linkage=global-alloc",
        f"[{9 if linked_type_id == 5 else 8}] DATASEC '.maps' size=16 vlen=1",
        f"    type_id={8 if linked_type_id == 5 else 7} offset=0 size=16",
    ])
    return "\n".join(lines) + "\n"


def _read_fixture_processes():
    """Expand concise test values to the complete bpftool struct format."""
    dump = live_reader.dump_map
    side_effect = dump.side_effect
    if side_effect is not None and not callable(side_effect):
        outcomes = list(side_effect)
        for map_name, snapshot in zip(live_reader.MAPS, outcomes):
            if not isinstance(snapshot, list):
                continue
            for entry in snapshot:
                try:
                    _key, value = live_reader.decode(entry)
                except (TypeError, ValueError):
                    continue
                for field in live_reader.MAP_VALUE_FIELDS[map_name] - set(value):
                    value[field] = "" if field == "comm" else 0
        dump.side_effect = outcomes
    return live_reader.read_live_processes()


class BackendModelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._metadata_patcher = patch(
            "src.backend.live_reader.get_map_metadata",
            side_effect=_map_metadata_fixture,
        )
        self._metadata_patcher.start()
        self.addCleanup(self._metadata_patcher.stop)

    def test_live_response_direct_json_render_matches_fastapi_encoder(self) -> None:
        with patch("src.backend.app.read_live_processes", return_value=[]):
            direct = backend.live()

        legacy_payload = {
            "source": "live_ebpf",
            "source_status": "empty",
            "source_error": "The four pinned maps are readable but contain no process entries.",
            "count": 0,
            "models": backend.model_statuses(),
            "processes": [],
        }
        legacy = JSONResponse(content=jsonable_encoder(legacy_payload))
        self.assertIsInstance(direct, JSONResponse)
        self.assertEqual(direct.status_code, legacy.status_code)
        self.assertEqual(direct.media_type, legacy.media_type)
        self.assertEqual(json.loads(direct.body), json.loads(legacy.body))

    def test_direct_and_legacy_render_match_null_and_non_finite_semantics(self) -> None:
        for value in (None, float("nan"), float("inf"), float("-inf")):
            outcomes = []
            for content in (value, jsonable_encoder(value)):
                try:
                    outcomes.append(("body", JSONResponse(content=content).body))
                except Exception as exc:
                    outcomes.append((type(exc), None))
            self.assertEqual(outcomes[0], outcomes[1])

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

    def test_bpf_exit_cleanup_removes_every_cpu_key_from_all_aggregate_maps(self) -> None:
        bpf_source = (
            Path(__file__).resolve().parents[1] / "realtime" / "realtime_monitor.bpf.c"
        ).read_text(encoding="utf-8")

        self.assertIn("bpf_for_each_map_elem(map, delete_pid_map_entry", bpf_source)
        self.assertIn("if (agg_key->pid != cleanup->pid)", bpf_source)
        callback_source = bpf_source.split("static long delete_pid_map_entry", 1)[1].split(
            "static __always_inline void cleanup_pid_map", 1
        )[0]
        self.assertNotIn("agg_key->cpu", callback_source)
        self.assertIn("return 0;", callback_source)
        self.assertIn("cleanup->error = result;", callback_source)
        self.assertIn("result != -BPF_ENOENT", callback_source)
        self.assertIn("cleanup->deleted++", callback_source)
        self.assertIn("cleanup_pid_map((struct bpf_map *)&sched_map, tid, 0);", bpf_source)
        self.assertIn("if (group_dead)", bpf_source)
        for map_name, map_id in (("mem_map", 1), ("syscall_map", 2), ("lock_map", 3)):
            self.assertIn(
                f"cleanup_pid_map((struct bpf_map *)&{map_name}, tgid, {map_id});",
                bpf_source,
            )

        self.assertIn("BPF_CORE_READ(signal, live.counter) == 0", bpf_source)
        self.assertIn("BPF_CORE_READ(task, flags) & PF_EXITING", bpf_source)
        self.assertIn("if (prev_pid && !prev_exiting)", bpf_source)
        migrate_handler = bpf_source.split("int handle_migrate(", 1)[1].split(
            "static __always_inline int record_wakeup", 1
        )[0]
        self.assertIn(".cpu = ctx->dest_cpu", migrate_handler)
        self.assertIn("key.cpu = ctx->orig_cpu", migrate_handler)
        self.assertIn("bpf_map_lookup_elem(&sched_map, &key)", migrate_handler)
        self.assertNotIn("GET_OR_CREATE", migrate_handler)
        self.assertIn('bpf_printk("exit cleanup failed', bpf_source)
        self.assertIn("#define MAX_ENTRIES 65536", bpf_source)
        cleanup_function = bpf_source.split("static __always_inline void cleanup_pid_map", 1)[1].split(
            "#define GET_OR_CREATE", 1
        )[0]
        self.assertIn("if (traversed < 0 || cleanup.error)", cleanup_function)
        self.assertIn("cleanup.error ? cleanup.error : traversed", cleanup_function)

        exit_handler = bpf_source.split("int handle_exit(", 1)[1]
        self.assertLess(exit_handler.index("cleanup_pid_map("), exit_handler.index("return 0;"))
        self.assertNotIn("last_update_ns", callback_source)

        reader_source = Path(live_reader.__file__).read_text(encoding="utf-8")
        read_function = reader_source.split("def read_live_processes():", 1)[1]
        self.assertNotIn("bpf_map_delete_elem", read_function)
        self.assertNotIn("map delete", read_function.lower())

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
        with self.assertRaisesRegex(ValueError, "malformed bpftool map entry"):
            live_reader.decode({"key": ["bad"], "value": {}})

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

        process = _read_fixture_processes()[0]

        self.assertEqual(process["ctx_switches"], 0)
        self.assertNotIn("ctx_switches", process["missing_metrics"])
        self.assertEqual(process["syscall_count"], 0)
        self.assertEqual(process["futex_count"], 0)
        self.assertEqual(process["kmalloc_count"], 0)
        self.assertEqual(process["mutex_contentions"], 0)
        self.assertEqual(process["max_mutex_wait_ns"], 0)
        self.assertEqual(process["rwsem_read_contentions"], 0)
        self.assertEqual(process["avg_mutex_wait_ns"], 0)

    @patch("src.backend.live_reader.dump_map")
    def test_lock_map_features_preserve_counts_and_zero_denominator_average(self, dump_map) -> None:
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

        process = _read_fixture_processes()[0]

        self.assertEqual(process["mutex_contentions"], 0)
        self.assertEqual(process["avg_mutex_wait_ns"], 0)
        self.assertNotIn("avg_mutex_wait_ns", process["missing_metrics"])
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

        process = _read_fixture_processes()[0]
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
    def test_reader_matches_baseline_zero_average_with_zero_latency_count(self, dump_map) -> None:
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

        process = _read_fixture_processes()[0]

        self.assertEqual(process["stall_ns"], 120)
        self.assertEqual(process["latency_count"], 0)
        self.assertEqual(process["avg_stall_ns"], 0)
        self.assertNotIn("avg_stall_ns", process["missing_metrics"])

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

        rows = _read_fixture_processes()
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

        row = _read_fixture_processes()[0]

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

        row = _read_fixture_processes()[0]

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

        row = _read_fixture_processes()[0]

        self.assertEqual(row["sample_timestamp_ns"], 1_000_000_000)
        self.assertEqual(row["data_age_seconds"], 49.0)
        self.assertEqual(row["data_status"], "stale")

    @patch("src.backend.live_reader.time.monotonic_ns", return_value=50_000_000_000)
    @patch("src.backend.live_reader.dump_map")
    def test_live_reader_rejects_incomplete_timestamp_and_marks_invalid_or_future_times(
        self, dump_map, _clock
    ) -> None:
        def entry(pid: int, value: dict) -> dict:
            return {"formatted": {"key": {"pid": pid, "cpu": 0}, "value": value}}

        def full_sched(timestamp):
            value = {
                field: ("worker" if field == "comm" else 0)
                for field in live_reader.MAP_VALUE_FIELDS["sched"]
            }
            value["ctx_switches"] = 1
            value["last_update_ns"] = timestamp
            return value

        dump_map.side_effect = [
            [entry(48, full_sched("not-a-number")),
             entry(49, full_sched(60_000_000_000))],
            [], [], [],
        ]

        rows = {row["pid"]: row for row in _read_fixture_processes()}

        for pid in (48, 49):
            self.assertEqual(rows[pid]["sample_timestamp_status"], "invalid")
            self.assertIsNone(rows[pid]["sample_timestamp_ns"])
            self.assertIsNone(rows[pid]["data_age_seconds"])
            self.assertEqual(rows[pid]["data_status"], "stale")

        with patch("src.backend.live_reader.dump_map", side_effect=[
            [entry(47, {"ctx_switches": 1})], [], [], [],
        ]):
            with self.assertRaisesRegex(ValueError, "incomplete 'sched' map value"):
                live_reader.read_live_processes()

    def test_champion_live_inference_rejects_missing_runq_observation(self) -> None:
        raw = backend.ProcessMetrics().model_dump()
        raw.pop("avg_runq_ratio")

        result = backend.predict_champion_batch([raw])[0]

        self.assertFalse(result["available"])
        self.assertIn("avg_runq_ratio", result["missing_features"])

    def test_champion_batch_matches_scalar_and_isolates_invalid_rows(self) -> None:
        class Model:
            classes_ = np.array([0, 1])

            def __init__(self):
                self.batch_sizes = []

            def predict_proba(self, features):
                self.batch_sizes.append(len(features))
                high = features["ctx_switches"].to_numpy(dtype=float) / 100.0
                return np.column_stack((1.0 - high, high))

        def raw_row(pid: int, ctx_switches: float) -> dict:
            row = {feature: float(index + 1) for index, feature in enumerate(backend.CHAMPION_RAW_REQUIRED)}
            row.update({"pid": pid, "ctx_switches": ctx_switches})
            return row

        first = raw_row(101, 10)
        second = raw_row(202, 20)
        model = Model()
        with patch.object(backend, "champion_model", model):
            scalar = [backend.predict_champion(row) for row in (first, second)]
            self.assertEqual(model.batch_sizes, [1, 1])
            model.batch_sizes.clear()

            batched = backend.predict_champion_batch([first, second])

            self.assertEqual(model.batch_sizes, [2])
            self.assertEqual(
                [result["prediction"] for result in batched],
                [result["prediction"] for result in scalar],
            )
            np.testing.assert_allclose(
                [result["high_probability"] for result in batched],
                [result["high_probability"] for result in scalar],
                rtol=0,
                atol=1e-6,
            )

            missing = raw_row(303, 30)
            missing.pop("avg_runq_ratio")
            invalid = raw_row(404, 40)
            invalid["total_runtime_ns"] = -2.0  # makes training's log1p feature non-finite
            mixed = backend.predict_champion_batch([first, missing, invalid, second])

        self.assertTrue(mixed[0]["available"])
        self.assertFalse(mixed[1]["available"])
        self.assertIn("avg_runq_ratio", mixed[1]["missing_features"])
        self.assertFalse(mixed[2]["available"])
        self.assertIn("log_total_runtime_ns", mixed[2]["missing_features"])
        self.assertTrue(mixed[3]["available"])
        self.assertEqual(model.batch_sizes, [2, 2])

    def test_mlp_batch_matches_scalar_and_isolates_missing_or_invalid_rows(self) -> None:
        class Attributor:
            feature_columns = ["second", "first"]
            target_columns = ["CPU", "Lock"]
            THRESHOLD = 0.5

            def __init__(self):
                self.batch_sizes = []

            def predict(self, features):
                self.batch_sizes.append(len(features))
                first = features["first"].to_numpy(dtype=float)
                second = features["second"].to_numpy(dtype=float)
                cpu = first / (first + second)
                return np.column_stack((cpu, 1.0 - cpu))

        rows = [
            {"first": 3.0, "second": 1.0},
            {"first": 1.0, "second": 3.0},
        ]
        model = Attributor()
        with patch.object(backend, "attributor", model):
            scalar = [backend.predict_mlp(row) for row in rows]
            self.assertEqual(model.batch_sizes, [1, 1])
            model.batch_sizes.clear()

            batched = backend.predict_mlp_batch(rows)

            self.assertEqual(model.batch_sizes, [2])
            for batch_result, scalar_result in zip(batched, scalar):
                self.assertEqual(batch_result["labels"], scalar_result["labels"])
                np.testing.assert_allclose(
                    [batch_result["probabilities"][name] for name in model.target_columns],
                    [scalar_result["probabilities"][name] for name in model.target_columns],
                    rtol=0,
                    atol=1e-6,
                )

            missing = {"first": 2.0}
            invalid = {"first": float("nan"), "second": 2.0}
            mixed = backend.predict_mlp_batch(
                [rows[0], missing, invalid, rows[1]],
                [[], [], [], []],
            )

        self.assertTrue(mixed[0]["available"])
        self.assertFalse(mixed[1]["available"])
        self.assertEqual(mixed[1]["missing_features"], ["second"])
        self.assertEqual(mixed[2]["status"], "error")
        self.assertFalse(mixed[2]["available"])
        self.assertTrue(mixed[3]["available"])
        self.assertEqual(model.batch_sizes, [2, 2])

    def test_prediction_batch_helpers_accept_empty_batches(self) -> None:
        self.assertEqual(backend.predict_champion_batch([]), [])
        self.assertEqual(backend.predict_mlp_batch([]), [])

    def test_saved_models_batched_predictions_match_scalar_on_complete_fixture(self) -> None:
        self.assertIsNotNone(backend.champion_model)
        self.assertIsNotNone(backend.attributor)
        input_columns = sorted(
            set(backend.CHAMPION_RAW_REQUIRED) | set(backend.attributor.feature_columns)
        )

        def complete_fixture(offset: int) -> dict:
            return {
                feature: float(offset + index + 1)
                for index, feature in enumerate(input_columns)
            }

        rows = [complete_fixture(1), complete_fixture(3)]
        champion_scalar = [backend.predict_champion(row) for row in rows]
        champion_batch = backend.predict_champion_batch(rows)
        self.assertEqual(
            [result["prediction"] for result in champion_batch],
            [result["prediction"] for result in champion_scalar],
        )
        np.testing.assert_allclose(
            [result["high_probability"] for result in champion_batch],
            [result["high_probability"] for result in champion_scalar],
            rtol=0,
            atol=1e-6,
        )

        mlp_scalar = [backend.predict_mlp(row) for row in rows]
        mlp_batch = backend.predict_mlp_batch(rows)
        for batch_result, scalar_result in zip(mlp_batch, mlp_scalar):
            self.assertEqual(batch_result["labels"], scalar_result["labels"])
            np.testing.assert_allclose(
                [batch_result["probabilities"][name] for name in backend.attributor.target_columns],
                [scalar_result["probabilities"][name] for name in backend.attributor.target_columns],
                rtol=0,
                atol=1e-6,
            )

    def test_current_map_schema_complete_values_cover_attribution_features(self) -> None:
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

            process = _read_fixture_processes()[0]

        self.assertEqual(process["avg_syscall_latency_ns"], 10)
        self.assertEqual(process["avg_mutex_wait_ns"], 20)
        self.assertEqual(process["missing_metrics"], [])

    def test_reader_returns_zero_for_zero_denominator_averages(self) -> None:
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

            process = _read_fixture_processes()[0]

        for metric in (
            "avg_syscall_latency_ns", "avg_futex_latency_ns",
            "avg_epoll_latency_ns", "avg_mutex_wait_ns",
            "avg_rwsem_read_wait_ns",
        ):
            self.assertEqual(process[metric], 0)
            self.assertNotIn(metric, process["missing_metrics"])

    def test_reader_fills_disjoint_map_keys_with_family_specific_zeroes(self) -> None:
        def complete_value(map_name: str, overrides: dict) -> dict:
            fields = live_reader.MAP_VALUE_FIELDS[map_name] - {"last_update_ns", "comm"}
            value = {field: 0 for field in fields}
            value.update(overrides)
            value["last_update_ns"] = 1
            if map_name == "sched":
                value["comm"] = "worker"
            return value

        def entry(pid: int, cpu: int, value: dict) -> dict:
            return {"formatted": {"key": {"pid": pid, "cpu": cpu}, "value": value}}

        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                [entry(101, 0, complete_value("sched", {"ctx_switches": 7}))],
                [entry(102, 0, complete_value("mem", {"kmalloc_count": 4}))],
                [entry(103, 0, complete_value("syscall", {"futex_count": 3}))],
                [entry(104, 0, complete_value("lock", {"mutex_contentions": 2}))],
            ]
            rows = {row["pid"]: row for row in _read_fixture_processes()}

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[101]["kmalloc_count"], 0)
        self.assertEqual(rows[101]["futex_count"], 0)
        self.assertEqual(rows[102]["kmalloc_count"], 4)
        self.assertEqual(rows[102]["futex_count"], 0)
        self.assertEqual(rows[103]["futex_count"], 3)
        self.assertEqual(rows[103]["kmalloc_count"], 0)
        self.assertEqual(rows[104]["futex_count"], 0)
        self.assertEqual(rows[104]["kmalloc_count"], 0)
        self.assertTrue(all(row["missing_metrics"] == [] for row in rows.values()))

    def test_reader_zero_and_positive_denominator_average_semantics(self) -> None:
        def value(map_name: str, overrides: dict) -> dict:
            fields = live_reader.MAP_VALUE_FIELDS[map_name] - {"last_update_ns", "comm"}
            result = {field: 0 for field in fields}
            result.update(overrides)
            result["last_update_ns"] = 1
            if map_name == "sched":
                result["comm"] = "worker"
            return result

        entry = lambda map_name, values: [{"formatted": {
            "key": {"pid": 111, "cpu": 0}, "value": value(map_name, values),
        }}]
        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                entry("sched", {"stall_ns": 100, "latency_count": 0,
                                "runq_ratio_sum": 99, "runq_ratio_count": 0}),
                [],
                entry("syscall", {"total_count": 0, "latency_sum_ns": 90,
                                  "futex_count": 0, "futex_latency_sum_ns": 80,
                                  "epoll_count": 0, "epoll_latency_sum_ns": 70}),
                entry("lock", {"mutex_contentions": 0, "mutex_wait_sum_ns": 60,
                               "rwsem_read_contentions": 0,
                               "rwsem_read_wait_sum_ns": 50}),
            ]
            zero_row = _read_fixture_processes()[0]

        for feature in (
            "avg_runq_ratio", "avg_stall_ns", "avg_syscall_latency_ns",
            "avg_futex_latency_ns", "avg_epoll_latency_ns",
            "avg_mutex_wait_ns", "avg_rwsem_read_wait_ns",
        ):
            self.assertEqual(zero_row[feature], 0)
            self.assertNotIn(feature, zero_row["missing_metrics"])

        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                entry("sched", {"runq_ratio_sum": 17, "runq_ratio_count": 2,
                                "stall_ns": 101, "latency_count": 2}),
                [],
                entry("syscall", {"total_count": 2, "latency_sum_ns": 101,
                                  "futex_count": 2, "futex_latency_sum_ns": 99,
                                  "epoll_count": 2, "epoll_latency_sum_ns": 79}),
                entry("lock", {"mutex_contentions": 2, "mutex_wait_sum_ns": 103,
                               "rwsem_read_contentions": 2,
                               "rwsem_read_wait_sum_ns": 105}),
            ]
            positive_row = _read_fixture_processes()[0]

        expected = {
            "avg_runq_ratio": 8, "avg_stall_ns": 50,
            "avg_syscall_latency_ns": 50, "avg_futex_latency_ns": 49,
            "avg_epoll_latency_ns": 39, "avg_mutex_wait_ns": 51,
            "avg_rwsem_read_wait_ns": 52,
        }
        for feature, result in expected.items():
            self.assertEqual(positive_row[feature], result)

    def test_reader_rejects_malformed_or_incompatible_map_entries(self) -> None:
        malformed_cases = [
            [{"formatted": {"key": {"pid": 1, "cpu": 0}, "value": "bad"}}],
            [{"formatted": {"key": {"pid": 1, "cpu": 0}, "value": {
                "ctx_switches": 1.5,
            }}}],
            [{"formatted": {"key": {"pid": 1, "cpu": 0}, "value": {
                "future_counter": 9,
            }}}],
        ]
        for malformed in malformed_cases:
            with self.subTest(malformed=malformed), patch(
                "src.backend.live_reader.dump_map",
                side_effect=[malformed, [], [], []],
            ):
                with self.assertRaises(ValueError):
                    _read_fixture_processes()

        with self.assertRaisesRegex(ValueError, "exactly sched, mem, syscall, and lock"):
            live_reader._validate_map_snapshot(
                {"sched": [], "mem": [], "syscall": []}, {}
            )

    def test_reader_rejects_recognized_but_incomplete_entry_before_zero_filling(self) -> None:
        incomplete = [{"formatted": {
            "key": {"pid": 99, "cpu": 0}, "value": {"ctx_switches": 1},
        }}]
        # The target key is absent from sched, but that snapshot contains an
        # incompatible entry. It must invalidate the snapshot before defaults.
        complete_mem = {
            field: 0 for field in live_reader.MAP_VALUE_FIELDS["mem"]
            if field != "last_update_ns"
        }
        complete_mem["last_update_ns"] = 1
        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [
                incomplete,
                [{"formatted": {"key": {"pid": 100, "cpu": 0}, "value": complete_mem}}],
                [], [],
            ]
            with self.assertRaisesRegex(ValueError, "incomplete 'sched' map value"):
                live_reader.read_live_processes()

    def test_reader_requires_verifiable_identity_and_abi_for_empty_maps(self) -> None:
        sched_value = {
            field: ("worker" if field == "comm" else (1 if field == "last_update_ns" else 0))
            for field in live_reader.MAP_VALUE_FIELDS["sched"]
        }
        metadata = {name: _map_metadata_fixture(name) for name in live_reader.MAPS}
        with patch("src.backend.live_reader.get_map_metadata", side_effect=lambda name: metadata[name]), \
             patch("src.backend.live_reader.dump_map", side_effect=[
                 [{"formatted": {"key": {"pid": 109, "cpu": 0}, "value": sched_value}}],
                 [], [], [],
             ]):
            row = live_reader.read_live_processes()[0]
            self.assertEqual(row["kmalloc_count"], 0)
            self.assertEqual(row["futex_count"], 0)

        invalid_metadata = {name: _map_metadata_fixture(name) for name in live_reader.MAPS}
        invalid_metadata["sched"] = {"name": "sched_map", "type": "hash"}
        with patch("src.backend.live_reader.get_map_metadata", side_effect=lambda name: invalid_metadata[name]), \
             patch("src.backend.live_reader.dump_map", return_value=[]):
            with self.assertRaisesRegex(ValueError, "lacks verifiable ABI identity"):
                _read_fixture_processes()

    def test_raw_btf_resolves_the_map_associated_value_struct_for_all_maps(self) -> None:
        for name in live_reader.MAPS:
            with self.subTest(map_name=name):
                schema = live_reader._parse_raw_btf_schema(_raw_btf_fixture(name), name)
                self.assertEqual(schema["map_symbol"], live_reader.MAPS[name])
                self.assertEqual(schema["key_struct"], "agg_key")
                self.assertEqual(set(schema["key_fields"]), {"pid", "cpu"})
                expected_name, expected_size = live_reader.MAP_VALUE_STRUCTS[name]
                self.assertEqual(schema["value_struct"], expected_name)
                self.assertEqual(schema["value_struct_size"], expected_size)
                self.assertEqual(set(schema["value_fields"]), live_reader.MAP_VALUE_FIELDS[name])

    def test_raw_btf_follows_typedefs_and_qualifiers_on_the_value_member(self) -> None:
        raw_btf = _raw_btf_fixture("sched").replace(
            "[5] PTR '(anon)' type_id=3",
            "[10] CONST '(anon)' type_id=3\n"
            "[11] TYPEDEF 'sched_value_alias' type_id=10\n"
            "[5] PTR '(anon)' type_id=11",
        )
        schema = live_reader._parse_raw_btf_schema(raw_btf, "sched")
        self.assertEqual(schema["value_struct"], "sched_value")
        self.assertEqual(schema["value_struct_size"], 104)

    def test_raw_btf_accepts_trailing_commas_on_integer_tokens(self) -> None:
        raw_btf = _raw_btf_fixture("sched")
        raw_btf = raw_btf.replace(
            "[2] STRUCT 'agg_key' size=8 vlen=2",
            "[2] STRUCT 'agg_key' size=8, vlen=2,",
        ).replace(
            "[3] STRUCT 'sched_value' size=104 vlen=12",
            "[3] STRUCT 'sched_value' size=104, vlen=12,",
        ).replace(
            "[5] PTR '(anon)' type_id=3",
            "[5] PTR '(anon)' type_id=3,",
        )

        schema = live_reader._parse_raw_btf_schema(raw_btf, "sched")

        self.assertEqual(schema["value_struct"], "sched_value")
        self.assertEqual(schema["value_struct_size"], 104)
        self.assertEqual(live_reader._parse_btf_integer("23,", "size"), 23)

    def test_raw_btf_integer_parser_rejects_malformed_punctuation(self) -> None:
        for token in ("23,,", "23,x", ",23", "-23", ""):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "invalid BTF size"):
                    live_reader._parse_btf_integer(token, "size")

        malformed_btf = _raw_btf_fixture("sched").replace(
            "size=104", "size=23,,", 1,
        )
        with self.assertRaisesRegex(ValueError, "invalid BTF size"):
            live_reader._parse_raw_btf_schema(malformed_btf, "sched")

    def test_map_metadata_consumes_btf_id_and_text_dump_without_value_type_id(self) -> None:
        self._metadata_patcher.stop()
        for name in live_reader.MAPS:
            with self.subTest(map_name=name):
                map_payload = {
                    "id": 19,
                    "type": "hash",
                    "name": live_reader.MAPS[name],
                    "flags": 0,
                    "bytes_key": live_reader.MAP_ABI[name][0],
                    "bytes_value": live_reader.MAP_ABI[name][1],
                    "max_entries": 65536,
                    "btf_id": 207,
                }
                responses = [
                    SimpleNamespace(returncode=0, stdout=json.dumps(map_payload), stderr=""),
                    SimpleNamespace(returncode=0, stdout=_raw_btf_fixture(name), stderr=""),
                ]
                with patch("src.backend.live_reader.subprocess.run", side_effect=responses) as run:
                    metadata = live_reader.get_map_metadata(name)
                self.assertEqual(metadata["btf_id"], 207)
                self.assertEqual(metadata["value_struct"], live_reader.MAP_VALUE_STRUCTS[name][0])
                self.assertEqual(run.call_args_list[1].args[0][-6:], [
                    "btf", "dump", "id", "207", "format", "raw",
                ])
                self.assertNotIn("-j", run.call_args_list[1].args[0])
                self.assertNotIn("btf_value_type_id", map_payload)

    def test_btf_schema_rejects_missing_metadata_malformed_and_incompatible_layouts(self) -> None:
        self._metadata_patcher.stop()
        no_btf_id = {
            "id": 19, "type": "hash", "name": "sched_map",
            "bytes_key": 8, "bytes_value": 104,
        }
        with patch("src.backend.live_reader.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(no_btf_id), stderr=""
        )):
            with self.assertRaisesRegex(ValueError, "no verifiable BTF ID"):
                live_reader.get_map_metadata("sched")

        malformed_fixtures = [
            "not a bpftool BTF dump\n",
            _raw_btf_fixture("sched", linked_value_struct="mem_value"),
            _raw_btf_fixture("sched", value_size=96),
            _raw_btf_fixture(
                "sched",
                value_fields=sorted(live_reader.MAP_VALUE_FIELDS["sched"] - {"stall_ns"}),
            ),
            _raw_btf_fixture(
                "sched",
                value_fields=sorted(live_reader.MAP_VALUE_FIELDS["sched"] | {"extra_counter"}),
            ),
        ]
        for raw_btf in malformed_fixtures:
            with self.subTest(raw_btf=raw_btf[:100]):
                with self.assertRaises(ValueError):
                    live_reader._parse_raw_btf_schema(raw_btf, "sched")

    def test_metadata_validation_rejects_wrong_map_value_size_and_struct(self) -> None:
        for field, invalid_value in (
            ("name", "other_map"),
            ("type", "array"),
            ("bytes_key", 16),
            ("bytes_value", 64),
            ("btf_id", 0),
            ("map_symbol", "other_map"),
            ("key_struct", "wrong_key"),
            ("value_struct", "mem_value"),
            ("value_struct_size", 64),
            ("value_fields", sorted(live_reader.MAP_VALUE_FIELDS["sched"] - {"stall_ns"})),
            ("value_fields", sorted(live_reader.MAP_VALUE_FIELDS["sched"] | {"extra_counter"})),
        ):
            metadata = {name: _map_metadata_fixture(name) for name in live_reader.MAPS}
            metadata["sched"][field] = invalid_value
            with self.subTest(field=field, value=invalid_value):
                with self.assertRaisesRegex(ValueError, "pinned identity or ABI is incompatible"):
                    live_reader._validate_map_metadata(metadata)

    def test_reader_propagates_failed_map_reads_without_zero_defaults(self) -> None:
        with patch("src.backend.live_reader.dump_map") as dump_map:
            dump_map.side_effect = [[], [], RuntimeError("map read failed")]
            with self.assertRaisesRegex(RuntimeError, "map read failed"):
                _read_fixture_processes()

    def test_model_feature_schemas_keep_the_saved_exact_orders(self) -> None:
        champion_columns = json.loads(
            backend.CHAMPION_FEATURES_PATH.read_text(encoding="utf-8")
        )
        attribution_columns = json.loads(
            backend.ATTRIBUTION_FEATURES_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(len(champion_columns), 33)
        self.assertEqual(champion_columns, backend.CHAMPION_FEATURE_COLS)
        self.assertEqual(list(backend.champion_model.feature_names_in_), champion_columns)
        self.assertEqual(len(attribution_columns), 33)
        self.assertEqual(backend.attributor.feature_columns, attribution_columns)

    def test_current_artifact_bundle_loads(self) -> None:
        expected_artifacts_dir = Path(backend.__file__).resolve().parent / "artifacts"
        self.assertEqual(backend.ARTIFACTS_DIR, expected_artifacts_dir)
        self.assertTrue((expected_artifacts_dir / "realtime_pinner").is_file())
        self.assertTrue((expected_artifacts_dir / "realtime_monitor.bpf.o").is_file())
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

    @patch("src.backend.app.predict_mlp_batch")
    @patch("src.backend.app.predict_champion_batch")
    @patch("src.backend.app.read_live_processes")
    def test_live_response_keeps_severity_when_attribution_is_unavailable(
        self, read, predict_champion_batch, predict_mlp_batch
    ) -> None:
        read.return_value = [{"pid": 123, "cpu": 0, "comm": "fixture", "session_label": "live_ebpf", "data_status": "live", "sample_timestamp_ns": 123_456_789, "sample_timestamp_clock": "CLOCK_MONOTONIC", "sample_timestamp_status": "available", "data_age_seconds": 0.25, "ctx_switches": 10, "missing_metrics": ["avg_syscall_latency_ns"]}]
        predict_mlp_batch.return_value = [{
            "available": False, "status": "unavailable", "model": "baseline_original_best.pt",
            "probabilities": None, "labels": None, "threshold": None,
            "error": "Required live attribution features are unavailable: stall_ns",
            "missing_features": ["stall_ns"],
        }]
        predict_champion_batch.return_value = [{
            "available": True,
            "status": "available",
            "pred_class": 1,
            "prediction": "High",
            "is_bottleneck": True,
            "high_probability": 0.7,
            "threshold": 0.5,
        }]

        payload = backend._live_payload()

        self.assertEqual(payload["source"], "live_ebpf")
        self.assertEqual(payload["count"], 1)
        process = payload["processes"][0]
        self.assertFalse(process["attribution_prediction"]["available"])
        self.assertEqual(process["attribution_prediction"]["missing_features"], ["stall_ns"])
        self.assertEqual(process["severity_prediction"]["prediction"], "High")
        self.assertEqual(process["process"]["pid"], 123)
        self.assertEqual((process["pid"], process["cpu"], process["comm"]), (123, 0, "fixture"))
        self.assertEqual(process["sample_timestamp_ns"], "123456789")
        self.assertEqual(process["sample_timestamp_clock"], "CLOCK_MONOTONIC")
        self.assertEqual(process["sample_timestamp_status"], "available")
        self.assertEqual(process["data_age_seconds"], 0.25)
        self.assertNotIn("sample_timestamp_ns", process["metrics"])
        self.assertNotIn("data_status", process["metrics"])
        self.assertNotIn("missing_metrics", process["metrics"])
        self.assertEqual(process["metrics"]["ctx_switches"], 10)
        self.assertNotIn("ctx_switches", process)
        self.assertNotIn("avg_syscall_latency_ns", process["metrics"])
        self.assertIn("models", payload)
        self.assertIn("avg_syscall_latency_ns", process["missing_metrics"])
        self.assertIn("stall_ns", process["missing_metrics"])

    @patch("src.backend.app.predict_mlp_batch")
    @patch("src.backend.app.predict_champion_batch")
    @patch("src.backend.app.read_live_processes")
    def test_live_predictions_remain_attached_to_their_pid_cpu_rows(
        self, read, predict_champion_batch, predict_mlp_batch
    ) -> None:
        common = {
            "comm": "fixture", "session_label": "live_ebpf", "data_status": "live",
            "sample_timestamp_ns": 123_456_789, "sample_timestamp_clock": "CLOCK_MONOTONIC",
            "sample_timestamp_status": "available", "data_age_seconds": 0.25,
            "missing_metrics": [],
        }
        read.return_value = [
            {**common, "pid": 123, "cpu": 0, "ctx_switches": 10},
            {**common, "pid": 999, "cpu": 1, "data_status": "stale", "ctx_switches": 15},
            {**common, "pid": 456, "cpu": 3, "ctx_switches": 20},
        ]
        predict_champion_batch.side_effect = lambda rows: [
            {"available": True, "prediction": "High" if row["pid"] == 123 else "Not-High"}
            for row in rows
        ]
        predict_mlp_batch.side_effect = lambda rows, _missing: [
            {
                "available": True,
                "labels": ["CPU"] if row["pid"] == 123 else ["Lock"],
                "probabilities": {"CPU": 0.9} if row["pid"] == 123 else {"Lock": 0.8},
            }
            for row in rows
        ]

        payload = backend._live_payload()

        first, stale, second = payload["processes"]
        predict_champion_batch.assert_called_once()
        predict_mlp_batch.assert_called_once()
        self.assertEqual([row["pid"] for row in predict_champion_batch.call_args.args[0]], [123, 456])
        self.assertEqual([row["pid"] for row in predict_mlp_batch.call_args.args[0]], [123, 456])
        self.assertEqual((first["pid"], first["cpu"], first["metrics"]["ctx_switches"]), (123, 0, 10))
        self.assertEqual((first["severity_prediction"]["prediction"], first["attribution_prediction"]["labels"]), ("High", ["CPU"]))
        self.assertEqual((second["pid"], second["cpu"], second["metrics"]["ctx_switches"]), (456, 3, 20))
        self.assertEqual((second["severity_prediction"]["prediction"], second["attribution_prediction"]["labels"]), ("Not-High", ["Lock"]))
        self.assertEqual(stale["pid"], 999)
        self.assertEqual(stale["severity_prediction"]["status"], "unavailable")
        self.assertEqual(stale["attribution_prediction"]["status"], "unavailable")

    @patch("src.backend.app.read_live_processes", return_value=[])
    def test_live_response_supports_empty_metrics(self, _read) -> None:
        response = backend._live_payload()
        self.assertEqual(response["count"], 0)
        self.assertEqual(response["processes"], [])

    @patch("src.backend.app.read_live_processes", side_effect=RuntimeError("maps unavailable"))
    def test_unavailable_live_metrics_return_service_unavailable(self, _read) -> None:
        with self.assertRaises(HTTPException) as caught:
            backend._live_payload()
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
