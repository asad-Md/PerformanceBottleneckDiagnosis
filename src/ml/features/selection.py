"""Feature selection helpers with explicit leakage control."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


LEAKAGE_COLS = [
    "avg_stall_ns",
    "stall_ns",
    "max_stall_ns",
]

NON_FEATURE_COLS = [
    "timestamp_ns",
    "pid",
    "cpu",
    "comm",
    "session_label",
    "bottleneck_class",
    "y",
]


def numeric_feature_columns(
    df: pd.DataFrame,
    *,
    include_leakage: bool = False,
    include_latency_count: bool = True,
    extra_drop_cols: Iterable[str] = (),
) -> list[str]:
    """Return numeric feature columns for model training."""

    drop = set(NON_FEATURE_COLS) | set(extra_drop_cols)
    if not include_leakage:
        drop.update(LEAKAGE_COLS)
    if not include_latency_count:
        drop.add("latency_count")

    feature_cols = []
    for col in df.columns:
        if col in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols

