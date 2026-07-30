"""Session-level feature engineering for Phase 4 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.features.engineering import _safe_div
from ml.models.phase4_common import (
    DEFAULT_INPUT,
    PHASE4_ARTIFACT_DIR,
    RANDOM_STATE,
    DERIVED_LEAKAGE_COLS,
    ensure_phase4_dir,
    evaluate_estimator_cv,
    load_phase4_dataset,
    markdown_table,
    phase4_feature_columns,
)


SESSION_BASE_COLUMNS = [
    "fault_rate",
    "lock_pressure",
    "io_cpu_ratio",
    "memory_pressure",
    "avg_runq_latency_ns",
    "avg_syscall_latency_ns",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def add_session_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add prior-row rolling and lag features within each session."""

    out = df.copy()
    if "memory_pressure" not in out.columns:
        out["memory_pressure"] = _num(out, "fault_rate") + _safe_div(_num(out, "total_alloc_bytes"), _num(out, "ctx_switches"))

    sort_cols = ["session_label"]
    if "timestamp_ns" in out.columns:
        sort_cols.append("timestamp_ns")
    out = out.sort_values(sort_cols).copy()

    for col in SESSION_BASE_COLUMNS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").fillna(0)
        previous = values.groupby(out["session_label"], sort=False).shift(1)
        rolling = previous.groupby(out["session_label"], sort=False).rolling(window=window, min_periods=1)
        out[f"{col}_rolling_mean"] = rolling.mean().reset_index(level=0, drop=True).fillna(0)
        out[f"{col}_rolling_std"] = rolling.std().reset_index(level=0, drop=True).fillna(0)
        out[f"{col}_rolling_min"] = rolling.min().reset_index(level=0, drop=True).fillna(0)
        out[f"{col}_rolling_max"] = rolling.max().reset_index(level=0, drop=True).fillna(0)
        out[f"{col}_previous_value"] = previous.fillna(0)
        out[f"{col}_delta"] = (values - previous).fillna(0)
        out[f"{col}_percentage_change"] = _safe_div(values - previous.fillna(0), previous.replace(0, np.nan)).fillna(0)

    ctx = _num(out, "ctx_switches")
    syscalls = _num(out, "syscall_count")
    out["stall_per_ctx_switch"] = _safe_div(_num(out, "stall_ns"), ctx)
    out["stall_per_syscall"] = _safe_div(_num(out, "stall_ns"), syscalls)
    out["runtime_per_fault"] = _safe_div(_num(out, "total_runtime_ns"), _num(out, "minor_faults") + _num(out, "major_faults"))
    out["lock_wait_per_switch"] = _safe_div(_num(out, "avg_mutex_wait_ns") * _num(out, "mutex_contentions"), ctx)
    out["memory_pressure_x_lock_pressure"] = _num(out, "memory_pressure") * _num(out, "lock_pressure")
    out["io_cpu_ratio_x_syscall_latency"] = _num(out, "io_cpu_ratio") * _num(out, "avg_syscall_latency_ns")

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.sort_index()


def run_session_feature_experiment(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = PHASE4_ARTIFACT_DIR,
) -> pd.DataFrame:
    out_dir = ensure_phase4_dir(artifact_dir)
    base_df = load_phase4_dataset(input_csv, session_features=False)
    session_df = load_phase4_dataset(input_csv, session_features=True)

    base_cols = phase4_feature_columns(base_df)
    session_cols = phase4_feature_columns(session_df)
    model_params = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "max_features": "sqrt",
    }
    factory = lambda: RandomForestClassifier(**model_params)
    rows = [
        evaluate_estimator_cv(base_df, factory, model_name="random_forest", experiment="base_features", feature_cols=base_cols),
        evaluate_estimator_cv(session_df, factory, model_name="random_forest", experiment="session_features", feature_cols=session_cols),
    ]
    comparison = pd.concat(rows, ignore_index=True)
    mean_comparison = comparison.groupby(["experiment", "model", "n_features"], as_index=False).mean(numeric_only=True)
    mean_comparison.to_csv(out_dir / "session_feature_comparison.csv", index=False)
    write_summary(mean_comparison, session_cols, out_dir)
    return mean_comparison


def write_summary(comparison: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    best = comparison.sort_values("macro_f1", ascending=False).iloc[0]
    leak_exclusions = ", ".join(f"`{col}`" for col in DERIVED_LEAKAGE_COLS)
    lines = [
        "# Phase 4 Session Feature Summary",
        "",
        f"- Best feature set: `{best['experiment']}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- MCC: `{best['mcc']:.4f}`",
        f"- Leakage-derived requested features excluded from training: {leak_exclusions}",
        f"- Leakage-free feature count with session features: `{len(feature_cols)}`",
        "",
        "## Mean Metrics",
        "",
        markdown_table(comparison),
    ]
    (out_dir / "session_feature_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 session feature experiment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=PHASE4_ARTIFACT_DIR)
    args = parser.parse_args()
    print(run_session_feature_experiment(args.input_csv, args.artifact_dir).to_string(index=False))


if __name__ == "__main__":
    main()
