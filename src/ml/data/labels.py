"""Target label derivation from runqueue stall latency."""

from __future__ import annotations

import pandas as pd


CLASS_NAMES = {
    0: "normal",
    1: "low",
    2: "medium",
    3: "high",
}


def derive_stall_class(avg_stall_ns: pd.Series) -> pd.Series:
    """Map avg_stall_ns to the four ML classes.

    Class boundaries:
    - 0: < 2 ms
    - 1: 2 ms to < 10 ms
    - 2: 10 ms to < 50 ms
    - 3: >= 50 ms
    """

    bins = [float("-inf"), 2_000_000, 10_000_000, 50_000_000, float("inf")]
    return pd.cut(avg_stall_ns, bins=bins, labels=[0, 1, 2, 3], right=False).astype(int)

