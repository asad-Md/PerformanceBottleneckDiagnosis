"""Dataset cleaning utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .labels import derive_stall_class


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def clean_dataset(
    df: pd.DataFrame,
    artifact_dir: Optional[Path | str] = DEFAULT_ARTIFACT_DIR,
    label_col: str = "bottleneck_class",
) -> pd.DataFrame:
    """Filter low-confidence rows and write a cleaning report.

    Required filters:
    - pid > 0
    - ctx_switches >= 3
    - latency_count >= 3
    - total_runtime_ns > 0
    """

    required_cols = {"pid", "ctx_switches", "latency_count", "total_runtime_ns", "avg_stall_ns"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"clean_dataset missing required columns: {missing}")

    original_rows = int(len(df))
    mask = (
        (df["pid"] > 0)
        & (df["ctx_switches"] >= 3)
        & (df["latency_count"] >= 3)
        & (df["total_runtime_ns"] > 0)
    )
    cleaned = df.loc[mask].copy()

    if label_col not in cleaned.columns:
        cleaned[label_col] = derive_stall_class(cleaned["avg_stall_ns"])

    per_class_counts = {
        str(cls): int(count)
        for cls, count in cleaned[label_col].value_counts().sort_index().items()
    }
    report = {
        "original_rows": original_rows,
        "removed_rows": int(original_rows - len(cleaned)),
        "retained_rows": int(len(cleaned)),
        "per_class_counts": per_class_counts,
        "filters": {
            "pid": "> 0",
            "ctx_switches": ">= 3",
            "latency_count": ">= 3",
            "total_runtime_ns": "> 0",
        },
    }

    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cleaning_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    return cleaned

