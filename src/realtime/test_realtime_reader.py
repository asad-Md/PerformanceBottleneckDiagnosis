from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.realtime.realtime_reader import RealtimeReader
from src.realtime.utils import run_bpftool_clear


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
        self.assertEqual(result["features"]["feature"].tolist(), [1])

    @patch("src.realtime.realtime_reader.time.perf_counter_ns", side_effect=[100, 180])
    @patch("src.realtime.realtime_reader.run_bpftool_clear", side_effect=RuntimeError("reset failed"))
    @patch("src.realtime.realtime_reader.run_bpftool_dump", return_value=[{"key": ["0x01"]}])
    def test_records_map_io_time_and_stops_on_reset_failure(self, _dump, _clear, _clock) -> None:
        reader = self._reader(collect_timings=True)
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            reader.read_once()
        self.assertEqual(reader.last_stage_timings_ns["map_read_ns"], 80)


if __name__ == "__main__":
    unittest.main()
