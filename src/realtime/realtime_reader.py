from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features.engineering import engineer_features

from .config import RealtimeConfig, load_config
from .utils import run_bpftool_dump


class RealtimeReader:
    READER_COLUMNS = [
        "timestamp_ns",
        "pid",
        "cpu",
        "comm",
        "ctx_switches",
        "voluntary_switches",
        "involuntary_switches",
        "cpu_migrations",
        "total_runtime_ns",
        "stall_ns",
        "avg_stall_ns",
        "max_stall_ns",
        "latency_count",
        "avg_runq_ratio",
        "minor_faults",
        "major_faults",
        "kmalloc_count",
        "kfree_count",
        "total_alloc_bytes",
        "total_free_bytes",
        "large_page_allocs",
        "syscall_count",
        "avg_syscall_latency_ns",
        "max_syscall_latency_ns",
        "read_count",
        "write_count",
        "read_bytes",
        "write_bytes",
        "mmap_count",
        "futex_count",
        "avg_futex_latency_ns",
        "epoll_count",
        "avg_epoll_latency_ns",
        "poll_count",
        "syscall_error_count",
        "mutex_contentions",
        "avg_mutex_wait_ns",
        "max_mutex_wait_ns",
        "rwsem_read_contentions",
        "avg_rwsem_read_wait_ns",
        "rwsem_write_contentions",
        "avg_rwsem_write_wait_ns",
        "max_rwsem_write_wait_ns",
        "session_label",
    ]
    
    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or load_config()
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        # Load trained model
        self.model = joblib.load(self.config.model_path)

        # Exact feature list used during training
        self.model_features = list(self.model.feature_names_in_)

    def _log_debug(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[realtime][debug] {message}")

    def _read_snapshot(self) -> dict[str, Any]:
        return {
            "sched": run_bpftool_dump(self.config.sched_map_path),
            "mem": run_bpftool_dump(self.config.mem_map_path),
            "syscall": run_bpftool_dump(self.config.syscall_map_path),
            "lock": run_bpftool_dump(self.config.lock_map_path),
        }

    def _normalize_map_entry(self, entry: Any) -> tuple[dict[str, int], dict[str, Any]]:
        if not isinstance(entry, dict):
            return {}, {}

        # Prefer bpftool's decoded representation
        if "formatted" in entry:
            formatted = entry.get("formatted", {})
            key_payload = formatted.get("key", {})
            value_payload = formatted.get("value", {})

        elif "key" in entry:
            key_payload = entry.get("key", {})
            value_payload = entry.get("value", {})

        else:
            key_payload = {
                "pid": entry.get("pid"),
                "cpu": entry.get("cpu"),
            }
            value_payload = entry.get("value", entry)

        if isinstance(key_payload, dict):
            pid = int(key_payload.get("pid", 0) or 0)
            cpu = int(key_payload.get("cpu", 0) or 0)

        elif isinstance(key_payload, list):
            # Fallback for older JSON layouts
            try:
                pid = int(key_payload[0], 16) if len(key_payload) > 0 else 0
                cpu = int(key_payload[1], 16) if len(key_payload) > 1 else 0
            except Exception:
                return {}, {}

        else:
            return {}, {}

        if not isinstance(value_payload, dict):
            value_payload = {}

        return {"pid": pid, "cpu": cpu}, value_payload

    def _iavg(self, total: Any, count: Any) -> int:
        total_value = int(total or 0)
        count_value = int(count or 0)
        return total_value // count_value if count_value > 0 else 0

    def _rows_from_snapshot(self, snapshot: dict[str, Any]) -> pd.DataFrame:
        rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}

        for map_name, entries in snapshot.items():
            for entry in entries:
                key, value = self._normalize_map_entry(entry)
                if not key:
                    continue

                key_tuple = (key["pid"], key["cpu"])
                row = rows_by_key.setdefault(
                    key_tuple,
                    {
                        "timestamp_ns": 0,
                        "pid": key["pid"],
                        "cpu": key["cpu"],
                        "comm": "",
                        "ctx_switches": 0,
                        "voluntary_switches": 0,
                        "involuntary_switches": 0,
                        "cpu_migrations": 0,
                        "total_runtime_ns": 0,
                        "stall_ns": 0,
                        "max_stall_ns": 0,
                        "latency_count": 0,
                        "runq_ratio_sum": 0,
                        "runq_ratio_count": 0,
                        "minor_faults": 0,
                        "major_faults": 0,
                        "kmalloc_count": 0,
                        "kfree_count": 0,
                        "total_alloc_bytes": 0,
                        "total_free_bytes": 0,
                        "large_page_allocs": 0,
                        "syscall_count": 0,
                        "latency_sum_ns": 0,
                        "latency_max_ns": 0,
                        "read_count": 0,
                        "write_count": 0,
                        "read_bytes": 0,
                        "write_bytes": 0,
                        "mmap_count": 0,
                        "futex_count": 0,
                        "futex_latency_sum_ns": 0,
                        "epoll_count": 0,
                        "epoll_latency_sum_ns": 0,
                        "poll_count": 0,
                        "syscall_error_count": 0,
                        "mutex_contentions": 0,
                        "mutex_wait_sum_ns": 0,
                        "mutex_wait_max_ns": 0,
                        "rwsem_read_contentions": 0,
                        "rwsem_read_wait_sum_ns": 0,
                        "rwsem_write_contentions": 0,
                        "rwsem_write_wait_sum_ns": 0,
                        "rwsem_write_wait_max_ns": 0,
                        "session_label": "unknown",
                    },
                )

                last_update_ns = int(value.get("last_update_ns", 0) or 0)
                if last_update_ns > int(row.get("timestamp_ns", 0) or 0):
                    row["timestamp_ns"] = last_update_ns

                if map_name == "sched":
                    row["ctx_switches"] += int(value.get("ctx_switches", 0) or 0)
                    row["voluntary_switches"] += int(value.get("voluntary_switches", 0) or 0)
                    row["involuntary_switches"] += int(value.get("involuntary_switches", 0) or 0)
                    row["cpu_migrations"] += int(value.get("cpu_migrations", 0) or 0)
                    row["total_runtime_ns"] += int(value.get("total_runtime_ns", 0) or 0)
                    row["stall_ns"] += int(value.get("stall_ns", 0) or 0)
                    row["max_stall_ns"] = max(int(row.get("max_stall_ns", 0) or 0), int(value.get("max_stall_ns", 0) or 0))
                    row["latency_count"] += int(value.get("latency_count", 0) or 0)
                    row["runq_ratio_sum"] += int(value.get("runq_ratio_sum", 0) or 0)
                    row["runq_ratio_count"] += int(value.get("runq_ratio_count", 0) or 0)
                    if value.get("comm"):
                        row["comm"] = str(value.get("comm"))
                elif map_name == "mem":
                    row["minor_faults"] += int(value.get("minor_faults", 0) or 0)
                    row["major_faults"] += int(value.get("major_faults", 0) or 0)
                    row["kmalloc_count"] += int(value.get("kmalloc_count", 0) or 0)
                    row["kfree_count"] += int(value.get("kfree_count", 0) or 0)
                    row["total_alloc_bytes"] += int(value.get("total_alloc_bytes", 0) or 0)
                    row["total_free_bytes"] += int(value.get("total_free_bytes", 0) or 0)
                    row["large_page_allocs"] += int(value.get("large_page_allocs", 0) or 0)
                elif map_name == "syscall":
                    row["syscall_count"] += int(value.get("total_count", 0) or 0)
                    row["latency_sum_ns"] += int(value.get("latency_sum_ns", 0) or 0)
                    row["latency_max_ns"] = max(int(row.get("latency_max_ns", 0) or 0), int(value.get("latency_max_ns", 0) or 0))
                    row["read_count"] += int(value.get("read_count", 0) or 0)
                    row["write_count"] += int(value.get("write_count", 0) or 0)
                    row["read_bytes"] += int(value.get("read_bytes", 0) or 0)
                    row["write_bytes"] += int(value.get("write_bytes", 0) or 0)
                    row["mmap_count"] += int(value.get("mmap_count", 0) or 0)
                    row["futex_count"] += int(value.get("futex_count", 0) or 0)
                    row["futex_latency_sum_ns"] += int(value.get("futex_latency_sum_ns", 0) or 0)
                    row["epoll_count"] += int(value.get("epoll_count", 0) or 0)
                    row["epoll_latency_sum_ns"] += int(value.get("epoll_latency_sum_ns", 0) or 0)
                    row["poll_count"] += int(value.get("poll_count", 0) or 0)
                    row["syscall_error_count"] += int(value.get("error_count", 0) or 0)
                elif map_name == "lock":
                    row["mutex_contentions"] += int(value.get("mutex_contentions", 0) or 0)
                    row["mutex_wait_sum_ns"] += int(value.get("mutex_wait_sum_ns", 0) or 0)
                    row["mutex_wait_max_ns"] = max(int(row.get("mutex_wait_max_ns", 0) or 0), int(value.get("mutex_wait_max_ns", 0) or 0))
                    row["rwsem_read_contentions"] += int(value.get("rwsem_read_contentions", 0) or 0)
                    row["rwsem_read_wait_sum_ns"] += int(value.get("rwsem_read_wait_sum_ns", 0) or 0)
                    row["rwsem_write_contentions"] += int(value.get("rwsem_write_contentions", 0) or 0)
                    row["rwsem_write_wait_sum_ns"] += int(value.get("rwsem_write_wait_sum_ns", 0) or 0)
                    row["rwsem_write_wait_max_ns"] = max(int(row.get("rwsem_write_wait_max_ns", 0) or 0), int(value.get("rwsem_write_wait_max_ns", 0) or 0))

        final_rows = []
        for row in rows_by_key.values():
            row_out = {col: row.get(col, 0) for col in self.READER_COLUMNS if col in row}
            row_out.setdefault("timestamp_ns", 0)
            row_out.setdefault("pid", 0)
            row_out.setdefault("cpu", 0)
            row_out.setdefault("comm", "")
            row_out["avg_stall_ns"] = self._iavg(row.get("stall_ns", 0), row.get("latency_count", 0))
            row_out["avg_runq_ratio"] = self._iavg(row.get("runq_ratio_sum", 0), row.get("runq_ratio_count", 0))
            row_out["avg_syscall_latency_ns"] = self._iavg(row.get("latency_sum_ns", 0), row.get("syscall_count", 0))
            row_out["avg_futex_latency_ns"] = self._iavg(row.get("futex_latency_sum_ns", 0), row.get("futex_count", 0))
            row_out["avg_epoll_latency_ns"] = self._iavg(row.get("epoll_latency_sum_ns", 0), row.get("epoll_count", 0))
            row_out["avg_mutex_wait_ns"] = self._iavg(row.get("mutex_wait_sum_ns", 0), row.get("mutex_contentions", 0))
            row_out["avg_rwsem_read_wait_ns"] = self._iavg(row.get("rwsem_read_wait_sum_ns", 0), row.get("rwsem_read_contentions", 0))
            row_out["avg_rwsem_write_wait_ns"] = self._iavg(row.get("rwsem_write_wait_sum_ns", 0), row.get("rwsem_write_contentions", 0))
            row_out["max_syscall_latency_ns"] = row.get("latency_max_ns", 0)
            row_out["max_mutex_wait_ns"] = row.get("mutex_wait_max_ns", 0)
            row_out["max_rwsem_write_wait_ns"] = row.get("rwsem_write_wait_max_ns", 0)
            final_rows.append(row_out)

        df = pd.DataFrame(final_rows, columns=self.READER_COLUMNS)
        for column in [
            "timestamp_ns",
            "pid",
            "cpu",
            "ctx_switches",
            "voluntary_switches",
            "involuntary_switches",
            "cpu_migrations",
            "total_runtime_ns",
            "stall_ns",
            "avg_stall_ns",
            "max_stall_ns",
            "latency_count",
            "avg_runq_ratio",
            "minor_faults",
            "major_faults",
            "kmalloc_count",
            "kfree_count",
            "total_alloc_bytes",
            "total_free_bytes",
            "large_page_allocs",
            "syscall_count",
            "avg_syscall_latency_ns",
            "max_syscall_latency_ns",
            "read_count",
            "write_count",
            "read_bytes",
            "write_bytes",
            "mmap_count",
            "futex_count",
            "avg_futex_latency_ns",
            "epoll_count",
            "avg_epoll_latency_ns",
            "poll_count",
            "syscall_error_count",
            "mutex_contentions",
            "avg_mutex_wait_ns",
            "max_mutex_wait_ns",
            "rwsem_read_contentions",
            "avg_rwsem_read_wait_ns",
            "rwsem_write_contentions",
            "avg_rwsem_write_wait_ns",
            "max_rwsem_write_wait_ns",
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        df["comm"] = df["comm"].fillna("")
        df["session_label"] = df["session_label"].fillna("unknown")
        return df

    def _validate_reader_columns(self, rows_df: pd.DataFrame) -> None:
        missing_columns = [column for column in self.READER_COLUMNS if column not in rows_df.columns]
        if missing_columns:
            raise RuntimeError(
                "Reconstructed process-row DataFrame is missing reader.c columns: " + ", ".join(missing_columns)
            )

    def read_once(self) -> dict[str, Any]:
        snapshot = self._read_snapshot()
        map_entry_counts = {name: len(entries) for name, entries in snapshot.items()}
        self._log_debug("entries read from each map: " + json.dumps(map_entry_counts, sort_keys=True))

        if all(count == 0 for count in map_entry_counts.values()):
            raise RuntimeError(
                "Realtime inference aborted: empty snapshot from pinned maps "
                f"({', '.join(f'{name}={count}' for name, count in map_entry_counts.items())})"
            )

        rows_df = self._rows_from_snapshot(snapshot)
        self._validate_reader_columns(rows_df)
        self._log_debug(f"reconstructed row count: {len(rows_df)}")
        if not rows_df.empty:
            self._log_debug("first reconstructed row: " + json.dumps(rows_df.iloc[0].to_dict(), default=str))

        featured_df = engineer_features(rows_df, artifact_dir=None)

        # Verify every feature expected by the trained model exists
        missing_features = [
            column
            for column in self.model_features
            if column not in featured_df.columns
        ]

        if missing_features:
            raise RuntimeError(
                "Realtime feature matrix is missing model features: "
                + ", ".join(missing_features)
            )

        # Select exactly the columns the model was trained on
        feature_frame = featured_df.loc[:, self.model_features].copy()

        self._log_debug(f"featured dataframe columns: {len(featured_df.columns)}")
        self._log_debug(f"model expects {len(self.model_features)} features")
        self._log_debug(
            "first model feature row: "
            + json.dumps(feature_frame.iloc[0].to_dict(), default=str)
        )
        self._log_debug(f"feature matrix shape: {feature_frame.shape}")

        return {
            "rows": rows_df,
            "features": feature_frame,
        }

def main() -> None:
    reader = RealtimeReader()
    reader.read_once()

if __name__ == "__main__":
    main()
