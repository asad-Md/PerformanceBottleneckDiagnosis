"""Phase 1 hybrid label strategy comparison for bottleneck severity labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from ml.data.cleaning import clean_dataset
from ml.evaluation.metrics import classification_metrics
from ml.features.engineering import engineer_features
from ml.features.selection import numeric_feature_columns


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RANDOM_STATE = 42


@dataclass(frozen=True)
class HybridLabelStrategy:
    name: str
    description: str
    interpretability: str
    builder: Callable[[pd.Series], pd.Series]


def labels_strategy_a_thresholds(avg_stall_ns: pd.Series) -> pd.Series:
    """Fixed latency thresholds: 0-2ms, 2-8ms, 8-25ms, 25ms+."""

    bins = [float("-inf"), 2_000_000, 8_000_000, 25_000_000, float("inf")]
    return pd.cut(avg_stall_ns, bins=bins, labels=[0, 1, 2, 3], right=False).astype(int)


def labels_strategy_b_percentile_qcut(avg_stall_ns: pd.Series) -> pd.Series:
    """Four percentile buckets from pd.qcut."""

    return pd.qcut(avg_stall_ns.rank(method="first"), q=4, labels=[0, 1, 2, 3]).astype(int)


def labels_strategy_c_quantile_min_size(avg_stall_ns: pd.Series, min_class_size: int = 1000) -> pd.Series:
    """Quantile labels that back off to fewer bins until every class is large enough."""

    ranked = avg_stall_ns.rank(method="first")
    for q in range(4, 1, -1):
        labels = pd.qcut(ranked, q=q, labels=list(range(q))).astype(int)
        if labels.value_counts().min() >= min_class_size:
            return labels
    return pd.Series(0, index=avg_stall_ns.index, dtype=int)


def labels_strategy_d_hybrid_mid_qcut(avg_stall_ns: pd.Series) -> pd.Series:
    """Hybrid labels: normal/high thresholds with mid-range qcut for moderate classes."""

    low_cut = 2_000_000
    high_cut = 25_000_000
    labels = pd.Series(index=avg_stall_ns.index, dtype=int)
    labels.loc[avg_stall_ns < low_cut] = 0
    labels.loc[avg_stall_ns >= high_cut] = 3

    mid_mask = (avg_stall_ns >= low_cut) & (avg_stall_ns < high_cut)
    mid = avg_stall_ns[mid_mask]
    if len(mid) >= 2:
        try:
            mid_labels = pd.qcut(mid.rank(method="first"), q=2, labels=[1, 2]).astype(int)
        except ValueError:
            mid_labels = pd.cut(mid.rank(method="first"), bins=2, labels=[1, 2]).astype(int)
        labels.loc[mid.index] = mid_labels
    elif len(mid) == 1:
        labels.loc[mid.index] = 1

    if (labels.isna()).any():
        labels = labels.fillna(1).astype(int)

    return labels.astype(int)


LABEL_STRATEGIES = [
    HybridLabelStrategy(
        name="strategy_a_thresholds",
        description="Fixed latency thresholds: 0-2ms, 2-8ms, 8-25ms, 25ms+",
        interpretability="high: classes map directly to stall severity thresholds",
        builder=labels_strategy_a_thresholds,
    ),
    HybridLabelStrategy(
        name="strategy_b_percentile_qcut",
        description="Four equal-frequency percentile buckets using qcut",
        interpretability="medium: balanced class sizes, but thresholds depend on dataset distribution",
        builder=labels_strategy_b_percentile_qcut,
    ),
    HybridLabelStrategy(
        name="strategy_c_quantile_min_size",
        description="Quantile classes with minimum class-size fallback for rare bins",
        interpretability="medium: retains ordered severity while protecting against tiny classes",
        builder=labels_strategy_c_quantile_min_size,
    ),
    HybridLabelStrategy(
        name="strategy_d_hybrid_mid_qcut",
        description="Hybrid strategy: fixed normal/high thresholds, qcut splits the middle range",
        interpretability="high: preserves intuitive normal/high boundaries while balancing middle classes",
        builder=labels_strategy_d_hybrid_mid_qcut,
    ),
]


def load_featured_dataset(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | None = None,
    label_builder: Callable[[pd.Series], pd.Series] = labels_strategy_a_thresholds,
) -> pd.DataFrame:
    """Load raw input, derive labels, clean, and engineer features."""

    df = pd.read_csv(input_csv)
    df["bottleneck_class"] = label_builder(df["avg_stall_ns"])
    cleaned = clean_dataset(df, artifact_dir=artifact_dir)
    cleaned["bottleneck_class"] = label_builder(cleaned["avg_stall_ns"])
    return engineer_features(cleaned, artifact_dir=artifact_dir)


def class_distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall class counts and percentages."""

    counts = df["bottleneck_class"].value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame(
        {
            "class": counts.index.astype(int),
            "count": counts.values.astype(int),
            "percentage": (counts.values / total) * 100,
        }
    )


def build_classifier(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def evaluate_strategy(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate one label strategy using GroupKFold with session-aware groups."""

    feature_cols = numeric_feature_columns(df, include_leakage=False, include_latency_count=True)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    rows: list[dict[str, object]] = []

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = build_classifier(random_state=RANDOM_STATE + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        metrics = classification_metrics(y.iloc[test_idx], pred)
        rows.append(
            {
                "fold": fold,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def compare_hybrid_label_strategies(input_csv: Path | str, artifact_dir: Path) -> pd.DataFrame:
    """Run all hybrid label strategies and save comparison artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []
    distribution_frames: list[pd.DataFrame] = []

    for strategy in LABEL_STRATEGIES:
        df = load_featured_dataset(input_csv=input_csv, artifact_dir=artifact_dir, label_builder=strategy.builder)
        distribution = class_distribution_frame(df)
        distribution["label_strategy"] = strategy.name
        distribution_frames.append(distribution)

        strategy_metrics = evaluate_strategy(df)
        metrics = strategy_metrics.mean(numeric_only=True)
        comparison_rows.append(
            {
                "label_strategy": strategy.name,
                "description": strategy.description,
                "interpretability": strategy.interpretability,
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "mcc": float(metrics["mcc"]),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(artifact_dir / "hybrid_label_comparison.csv", index=False)

    distribution = pd.concat(distribution_frames, ignore_index=True)
    distribution.to_csv(artifact_dir / "hybrid_label_distribution.csv", index=False)

    write_recommendation(comparison, artifact_dir)
    return comparison


def write_recommendation(comparison: pd.DataFrame, artifact_dir: Path) -> None:
    """Write a short markdown recommendation based on the comparison metrics."""

    best = comparison.sort_values(["macro_f1", "balanced_accuracy", "mcc"], ascending=False).iloc[0]
    lines = [
        "# Hybrid Label Strategy Recommendation",
        "",
        "This report compares four label derivation methods for bottleneck severity classification.",
        "",
        "## Best strategy",
        "",
        f"- **Recommended strategy:** `{best['label_strategy']}`",
        f"- **Description:** {best['description']}",
        f"- **Interpretability:** {best['interpretability']}",
        "",
        "## Summary metrics",
        "",
        "| Strategy | Macro F1 | Balanced accuracy | MCC |",
        "|---|---|---|---|",
    ]

    for _, row in comparison.sort_values("macro_f1", ascending=False).iterrows():
        lines.append(
            f"| `{row['label_strategy']}` | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} | {row['mcc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `strategy_a_thresholds` preserves the original fixed latency bins used by the baseline pipeline.",
            "- `strategy_b_percentile_qcut` enforces balanced class sizes at the cost of dataset-relative thresholds.",
            "- `strategy_c_quantile_min_size` protects against tiny classes by falling back to fewer bins.",
            "- `strategy_d_hybrid_mid_qcut` keeps intuitive normal/high boundaries while balancing the middle severity range.",
        ]
    )

    Path(artifact_dir / "hybrid_label_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hybrid bottleneck label strategies.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    compare_hybrid_label_strategies(args.input_csv, args.artifact_dir)


if __name__ == "__main__":
    main()
