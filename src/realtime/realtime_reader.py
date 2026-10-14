from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .config import RealtimeConfig, load_config
from .utils import run_bpftool_clear, run_bpftool_dump, load_feature_columns, load_label_thresholds

# Keeps the legacy multiclass labels available for older classifier artifacts.
#
# The deployed v7.2 model is binary and Predictor selects its labels from model.classes_.
CLASS_NAMES = {0: "Normal", 1: "Low", 2: "Medium", 3: "High"}
BOTTLENECK_CLASS = 3  # predictions == BOTTLENECK_CLASS are the actual bottleneck cases


def engineer_features_v5(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the v7.2 training notebooks' causal feature engineering.

    Keep these transformations in lockstep with the classifier and attribution notebooks.
    """
    df = df.copy()

    df["involuntary_ratio"] = df["involuntary_switches"] / df["ctx_switches"].clip(lower=1)
    df["voluntary_ratio"] = df["voluntary_switches"] / df["ctx_switches"].clip(lower=1)
    df["runtime_per_switch"] = df["total_runtime_ns"] / df["ctx_switches"].clip(lower=1)
    df["read_write_ratio"] = df["read_bytes"] / (df["write_bytes"] + 1)
    df["io_bytes_total"] = df["read_bytes"] + df["write_bytes"]
    df["alloc_pressure"] = df["total_alloc_bytes"] / df["ctx_switches"].clip(lower=1)
    df["lock_pressure"] = (
        df["mutex_contentions"] + df["rwsem_read_contentions"] + df["rwsem_write_contentions"]
    )

    log_cols = [
        "total_runtime_ns",
        "avg_syscall_latency_ns",
        "max_syscall_latency_ns",
        "io_bytes_total",
        "total_alloc_bytes",
    ]
    for col in log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


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

    def __init__(self, config: RealtimeConfig | None = None, *, collect_timings: bool = False) -> None:
        self.config = config or load_config()
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        self.collect_timings = collect_timings
        self.last_stage_timings_ns: dict[str, int | None] = {}

        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise RuntimeError(f"Missing champion model for realtime inference: {self.config.model_path}")

        # Load the SHAP-reduced v7.2 classifier.
        self.model = joblib.load(self.config.model_path)
        print(f"Loaded model: {self.config.model_path} ({type(self.model).__name__})")

        # The notebook-saved ordered list is the deployment contract for the SHAP-reduced model.
        json_features = None
        if self.config.feature_cols_path:
            if not os.path.isfile(self.config.feature_cols_path):
                raise RuntimeError(
                    f"Missing classifier feature columns file: {self.config.feature_cols_path}"
                )
            json_features = load_feature_columns(self.config.feature_cols_path)

        model_features = list(self.model.feature_names_in_) if hasattr(self.model, "feature_names_in_") else None

        if model_features is not None:
            self.model_features = model_features
        elif json_features is not None:
            self.model_features = json_features
        else:
            raise RuntimeError(
                "Could not determine model feature list: model has no feature_names_in_ and "
                f"no configured feature list was found at {self.config.feature_cols_path!r}"
            )

        if json_features is not None and json_features != self.model_features:
            raise RuntimeError(
                "Configured feature columns do not exactly match the model's training order: "
                f"configured={json_features}; model={self.model_features}"
            )

        print(f"Model expects {len(self.model_features)} features")

        # Only apply a scaler if the loaded classifier actually needs scaled inputs.
        self.scaler = None
        self.uses_scaled_input = not hasattr(self.model, "feature_importances_")
        if self.uses_scaled_input:
            if not self.config.scaler_path or not os.path.exists(self.config.scaler_path):
                raise RuntimeError(
                    f"Model {type(self.model).__name__} appears to require scaled input (no "
                    "feature_importances_ attribute, i.e. not a tree model) but "
                    f"Configured classifier scaler was not found at {self.config.scaler_path!r}"
                )
            self.scaler = joblib.load(self.config.scaler_path)
            self._log_debug("Loaded classifier scaler — will scale features before predict")
        else:
            self._log_debug("Champion is a tree model — skipping scaling (matches notebook Section 4)")

        # This file provides the deployment probability threshold and the training label cutoff.
        self.label_thresholds: dict[str, float] | None = None
        self.decision_probability_threshold: float | None = None
        if self.config.label_thresholds_path:
            if not os.path.isfile(self.config.label_thresholds_path):
                raise RuntimeError(
                    f"Missing classifier threshold metadata: {self.config.label_thresholds_path}"
                )
            self.label_thresholds = load_label_thresholds(self.config.label_thresholds_path)
            threshold = self.label_thresholds.get("decision_probability_threshold")
            if threshold is None:
                raise ValueError(
                    "Classifier threshold metadata must define decision_probability_threshold: "
                    f"{self.config.label_thresholds_path}"
                )
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "decision_probability_threshold in "
                    f"{self.config.label_thresholds_path} must be between 0 and 1"
                )
            self.decision_probability_threshold = threshold
            self._log_debug(f"Loaded classifier threshold metadata: {self.label_thresholds}")

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

    def _clear_snapshot_maps(self, snapshot: dict[str, Any]) -> None:
        map_entries = [
            (self.config.sched_map_path, snapshot["sched"]),
            (self.config.mem_map_path, snapshot["mem"]),
            (self.config.syscall_map_path, snapshot["syscall"]),
            (self.config.lock_map_path, snapshot["lock"]),
        ]
        run_bpftool_clear(map_entries)

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
        if self.collect_timings:
            self.last_stage_timings_ns = {
                "map_read_ns": None,
                "aggregation_ns": None,
                "feature_ns": None,
                "rf_inference_ns": None,
            }
            stage_start_ns = time.perf_counter_ns()

        try:
            snapshot = self._read_snapshot()
            self._clear_snapshot_maps(snapshot)
        finally:
            if self.collect_timings:
                self.last_stage_timings_ns["map_read_ns"] = time.perf_counter_ns() - stage_start_ns

        map_entry_counts = {name: len(entries) for name, entries in snapshot.items()}
        self._log_debug("entries read from each map: " + json.dumps(map_entry_counts, sort_keys=True))

        if all(count == 0 for count in map_entry_counts.values()):
            raise RuntimeError(
                "Realtime inference aborted: empty snapshot from pinned maps "
                f"({', '.join(f'{name}={count}' for name, count in map_entry_counts.items())})"
            )

        if self.collect_timings:
            stage_start_ns = time.perf_counter_ns()
        rows_df = self._rows_from_snapshot(snapshot)
        self._validate_reader_columns(rows_df)
        self._log_debug(f"reconstructed row count: {len(rows_df)}")
        if not rows_df.empty:
            self._log_debug("first reconstructed row: " + json.dumps(rows_df.iloc[0].to_dict(), default=str))
        if self.collect_timings:
            self.last_stage_timings_ns["aggregation_ns"] = time.perf_counter_ns() - stage_start_ns

        # Keep realtime transformations aligned with both training notebooks.
        if self.collect_timings:
            stage_start_ns = time.perf_counter_ns()
        featured_df = engineer_features_v5(rows_df)

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

        # Select exactly the columns the model was trained on, in the model's expected order
        feature_frame = featured_df.loc[:, self.model_features].copy()

        self._log_debug(f"featured dataframe columns: {len(featured_df.columns)}")
        self._log_debug(f"model expects {len(self.model_features)} features")
        if not feature_frame.empty:
            self._log_debug(
                "first model feature row: "
                + json.dumps(feature_frame.iloc[0].to_dict(), default=str)
            )
        self._log_debug(f"feature matrix shape: {feature_frame.shape}")
        if self.collect_timings:
            self.last_stage_timings_ns["feature_ns"] = time.perf_counter_ns() - stage_start_ns

        return {
            "rows": featured_df,
            "features": feature_frame,
        }

    def predict(self) -> dict[str, Any]:
        """Read one snapshot and infer, scaling inputs only for a non-tree classifier."""
        result = self.read_once()
        feature_frame = result["features"]

        if self.collect_timings:
            stage_start_ns = time.perf_counter_ns()
        model_input: Any = feature_frame
        if self.uses_scaled_input and self.scaler is not None:
            model_input = self.scaler.transform(feature_frame)

        probabilities = self.model.predict_proba(model_input) if hasattr(self.model, "predict_proba") else None
        if self.decision_probability_threshold is not None and probabilities is not None:
            classes = list(self.model.classes_) if hasattr(self.model, "classes_") else []
            if set(classes) != {0, 1}:
                raise RuntimeError(
                    "A configured binary decision threshold requires model classes {0, 1}; "
                    f"found {classes}"
                )
            positive_class_index = classes.index(1)
            predictions = (
                probabilities[:, positive_class_index] >= self.decision_probability_threshold
            ).astype(int)
        else:
            predictions = self.model.predict(model_input)
        if self.collect_timings:
            self.last_stage_timings_ns["rf_inference_ns"] = time.perf_counter_ns() - stage_start_ns

        result["predictions"] = predictions
        result["probabilities"] = probabilities
        return result


def main() -> None:
    reader = RealtimeReader()
    result = reader.predict()
    class_names = CLASS_NAMES
    bottleneck_class = BOTTLENECK_CLASS
    model_classes = set(int(value) for value in getattr(reader.model, "classes_", []))
    if model_classes == {0, 1}:
        class_names = {0: "Not-High", 1: "High"}
        bottleneck_class = 1
    for position, (_, row) in enumerate(result["rows"].iterrows()):
        pred_class = int(result["predictions"][position])
        label = class_names.get(pred_class, str(pred_class))
        flag = "  <-- BOTTLENECK" if pred_class == bottleneck_class else ""
        print(f"pid={row['pid']} cpu={row['cpu']} comm={row['comm']!r} -> {pred_class} ({label}){flag}")


if __name__ == "__main__":
    main()