"""Phase 2 experiments for class imbalance and label strategy analysis."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from ml.data.cleaning import clean_dataset
from ml.evaluation.metrics import classification_metrics
from ml.features.engineering import engineer_features
from ml.features.selection import numeric_feature_columns

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RANDOM_STATE = 42


@dataclass(frozen=True)
class LabelStrategy:
    name: str
    description: str
    interpretability: str
    builder: Callable[[pd.Series], pd.Series]


def labels_strategy_a(avg_stall_ns: pd.Series) -> pd.Series:
    """Current threshold strategy requested for Phase 2: 0-2ms, 2-8ms, 8-25ms, 25ms+."""

    bins = [float("-inf"), 2_000_000, 8_000_000, 25_000_000, float("inf")]
    return pd.cut(avg_stall_ns, bins=bins, labels=[0, 1, 2, 3], right=False).astype(int)


def labels_strategy_b(avg_stall_ns: pd.Series) -> pd.Series:
    """Percentile labels using pd.qcut."""

    return pd.qcut(avg_stall_ns.rank(method="first"), q=4, labels=[0, 1, 2, 3]).astype(int)


def labels_strategy_c(avg_stall_ns: pd.Series, min_class_size: int = 1000) -> pd.Series:
    """Quantile labels that back off to fewer bins until every class is large enough."""

    ranked = avg_stall_ns.rank(method="first")
    for q in range(4, 1, -1):
        labels = pd.qcut(ranked, q=q, labels=list(range(q))).astype(int)
        if labels.value_counts().min() >= min_class_size:
            return labels
    return pd.Series(0, index=avg_stall_ns.index, dtype=int)


LABEL_STRATEGIES = [
    LabelStrategy(
        name="strategy_a_thresholds",
        description="Fixed latency thresholds: 0-2ms, 2-8ms, 8-25ms, 25ms+",
        interpretability="high: classes map directly to stall latency severity",
        builder=labels_strategy_a,
    ),
    LabelStrategy(
        name="strategy_b_percentile_qcut",
        description="Four percentile buckets from pd.qcut",
        interpretability="medium: balanced classes, but thresholds are dataset-relative",
        builder=labels_strategy_b,
    ),
    LabelStrategy(
        name="strategy_c_quantile_min_size",
        description="Quantile labels with minimum class-size guard",
        interpretability="medium: preserves class sizes while retaining ordered severity",
        builder=labels_strategy_c,
    ),
]


def load_featured_dataset(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Optional[Path | str] = None,
    label_builder: Callable[[pd.Series], pd.Series] = labels_strategy_a,
) -> pd.DataFrame:
    """Load, label, clean, and engineer features without touching Phase 1 pipeline code."""

    df = pd.read_csv(input_csv)
    df["bottleneck_class"] = label_builder(df["avg_stall_ns"])
    cleaned = clean_dataset(df, artifact_dir=artifact_dir)
    cleaned["bottleneck_class"] = label_builder(cleaned["avg_stall_ns"])
    return engineer_features(cleaned, artifact_dir=artifact_dir)


def class_distribution_frame(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """Return counts and percentages by class, optionally within a group column."""

    if group_col is None:
        counts = df["bottleneck_class"].value_counts().sort_index()
        total = counts.sum()
        return pd.DataFrame(
            {
                "scope": "overall",
                "class": counts.index.astype(int),
                "count": counts.values.astype(int),
                "percentage": (counts.values / total) * 100,
            }
        )

    grouped = df.groupby(group_col, dropna=False)["bottleneck_class"].value_counts().rename("count").reset_index()
    totals = grouped.groupby(group_col)["count"].transform("sum")
    grouped["percentage"] = (grouped["count"] / totals) * 100
    grouped = grouped.rename(columns={group_col: "scope", "bottleneck_class": "class"})
    grouped["class"] = grouped["class"].astype(int)
    return grouped.sort_values(["scope", "class"]).reset_index(drop=True)


def save_class_distribution_plots(raw: pd.DataFrame, cleaned: pd.DataFrame, artifact_dir: Path) -> None:
    """Save overall and per-session class distribution plots."""

    artifact_dir.mkdir(parents=True, exist_ok=True)

    dist = class_distribution_frame(cleaned)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist["class"].astype(str), dist["count"], color="#4c78a8")
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows after cleaning")
    ax.set_title("Class distribution after cleaning")
    fig.tight_layout()
    fig.savefig(artifact_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    session_counts = pd.crosstab(cleaned["session_label"], cleaned["bottleneck_class"])
    fig, ax = plt.subplots(figsize=(12, 8))
    session_counts.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_xlabel("Session label")
    ax.set_ylabel("Rows after cleaning")
    ax.set_title("Per-session class distribution")
    ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(artifact_dir / "session_class_distribution.png", dpi=150)
    plt.close(fig)

    before = class_distribution_frame(raw)
    after = class_distribution_frame(cleaned)
    for frame, filename, title in [
        (before, "class_distribution_before.png", "Class distribution before SMOTE"),
        (after, "class_distribution_after.png", "Class distribution after SMOTE"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(frame["class"].astype(str), frame["count"], color="#59a14f")
        ax.set_xlabel("Class")
        ax.set_ylabel("Rows")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(artifact_dir / filename, dpi=150)
        plt.close(fig)


def write_class_analysis(raw: pd.DataFrame, cleaned: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Write class counts, percentages, per-session distributions, and cleaned distributions."""

    raw_overall = class_distribution_frame(raw).assign(dataset="raw")
    cleaned_overall = class_distribution_frame(cleaned).assign(dataset="cleaned")
    per_session = class_distribution_frame(cleaned, group_col="session_label").assign(dataset="cleaned_by_session")
    analysis = pd.concat([raw_overall, cleaned_overall, per_session], ignore_index=True)
    analysis = analysis[["dataset", "scope", "class", "count", "percentage"]]
    analysis.to_csv(artifact_dir / "class_analysis.csv", index=False)
    return analysis


def class_weight_dict(y: pd.Series) -> dict[int, float]:
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def build_model(model_name: str, weighted: bool, y_train: pd.Series, random_state: int):
    """Build model with class-weight handling where the estimator supports it."""

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=60,
            class_weight="balanced" if weighted else None,
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=60,
            learning_rate=0.05,
            class_weight="balanced" if weighted else None,
            random_state=random_state,
            verbosity=-1,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def fit_model(model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, weighted: bool):
    """Fit with estimator-native or sample-weight class balancing."""

    if model_name == "xgboost" and weighted:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        return model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.fit(X_train, y_train)


def adaptive_smote(X_train: pd.DataFrame, y_train: pd.Series, kind: str = "smote", random_state: int = RANDOM_STATE):
    """Apply SMOTE to training data only, adapting k_neighbors to rare class counts."""

    counts = y_train.value_counts()
    min_count = int(counts.min())
    if min_count < 2 or len(counts) < 2:
        return X_train, y_train, "skipped_minority_class_too_small"

    k_neighbors = max(1, min(5, min_count - 1))
    sampler_cls = BorderlineSMOTE if kind == "borderline_smote" else SMOTE
    sampler = sampler_cls(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled, f"{kind}_k{k_neighbors}"


def aggregate_metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "mcc"]
    group_cols = ["model", "strategy", "class_weight", "smote", "sampler_status"]
    return df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()


def evaluate_strategy(
    df: pd.DataFrame,
    *,
    model_name: str = "random_forest",
    weighted: bool = False,
    smote_kind: Optional[str] = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate one imbalance strategy with GroupKFold, preserving group boundaries."""

    feature_cols = numeric_feature_columns(df, include_leakage=False, include_latency_count=True)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    rows: list[dict[str, object]] = []

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        sampler_status = "none"

        if smote_kind:
            X_train, y_train, sampler_status = adaptive_smote(X_train, y_train, kind=smote_kind, random_state=RANDOM_STATE + fold)

        model = build_model(model_name, weighted=weighted, y_train=y_train, random_state=RANDOM_STATE + fold)
        fit_model(model, model_name, X_train, y_train, weighted=weighted)
        pred = model.predict(X.iloc[test_idx])
        metrics = classification_metrics(y.iloc[test_idx], pred)
        rows.append(
            {
                "fold": fold,
                "model": model_name,
                "strategy": strategy_name(weighted, smote_kind),
                "class_weight": weighted,
                "smote": smote_kind or "none",
                "sampler_status": sampler_status,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def strategy_name(weighted: bool, smote_kind: Optional[str]) -> str:
    if weighted and smote_kind:
        return "class_weight_smote"
    if weighted:
        return "class_weight"
    if smote_kind:
        return smote_kind
    return "baseline"


def compare_imbalance_strategies(df: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Evaluate baseline, class weights, SMOTE, and class weights + SMOTE."""

    configs = [
        {"weighted": False, "smote_kind": None},
        {"weighted": True, "smote_kind": None},
        {"weighted": False, "smote_kind": "smote"},
        {"weighted": True, "smote_kind": "smote"},
        {"weighted": False, "smote_kind": "borderline_smote"},
    ]
    rows = []
    for cfg in configs:
        fold_results = evaluate_strategy(df, model_name="random_forest", **cfg)
        rows.extend(fold_results.to_dict("records"))

    comparison = aggregate_metrics(rows)
    comparison.to_csv(artifact_dir / "imbalance_comparison.csv", index=False)
    return comparison


def compare_class_weights_by_model(df: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Compare baseline vs class_weight for RandomForest, LightGBM, and XGBoost."""

    model_names = ["random_forest"]
    if LGBMClassifier is not None:
        model_names.append("lightgbm")
    if XGBClassifier is not None:
        model_names.append("xgboost")

    rows = []
    for model_name in model_names:
        for weighted in [False, True]:
            rows.extend(evaluate_strategy(df, model_name=model_name, weighted=weighted).to_dict("records"))

    comparison = aggregate_metrics(rows)
    comparison.to_csv(artifact_dir / "class_weight_model_comparison.csv", index=False)
    return comparison


def compare_label_strategies(input_csv: Path | str, artifact_dir: Path) -> pd.DataFrame:
    """Compare class balance, performance, and interpretability for label strategies."""

    rows: list[dict[str, object]] = []
    for strategy in LABEL_STRATEGIES:
        df = load_featured_dataset(input_csv=input_csv, artifact_dir=None, label_builder=strategy.builder)
        counts = df["bottleneck_class"].value_counts().sort_index()
        perf = aggregate_metrics(evaluate_strategy(df, model_name="random_forest", weighted=True).to_dict("records"))
        perf_row = perf.iloc[0].to_dict()
        for cls in range(int(counts.index.max()) + 1):
            rows.append(
                {
                    "label_strategy": strategy.name,
                    "description": strategy.description,
                    "interpretability": strategy.interpretability,
                    "class": cls,
                    "class_count": int(counts.get(cls, 0)),
                    "class_percentage": float(counts.get(cls, 0) / counts.sum() * 100),
                    "model": "random_forest",
                    "training_strategy": "class_weight",
                    "macro_precision": perf_row["macro_precision"],
                    "macro_recall": perf_row["macro_recall"],
                    "macro_f1": perf_row["macro_f1"],
                    "balanced_accuracy": perf_row["balanced_accuracy"],
                    "mcc": perf_row["mcc"],
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(artifact_dir / "label_strategy_comparison.csv", index=False)
    return comparison


def save_smote_distribution_plots(df: pd.DataFrame, artifact_dir: Path) -> None:
    """Generate before/after SMOTE distribution plots from the first grouped training fold."""

    feature_cols = numeric_feature_columns(df, include_leakage=False, include_latency_count=True)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    train_idx, _ = next(GroupKFold(n_splits=5).split(X, y, groups=groups))
    y_train = y.iloc[train_idx]
    X_resampled, y_resampled, _ = adaptive_smote(X.iloc[train_idx], y_train, kind="smote")

    for labels, filename, title in [
        (y_train, "class_distribution_before.png", "Training class distribution before SMOTE"),
        (pd.Series(y_resampled), "class_distribution_after.png", "Training class distribution after SMOTE"),
    ]:
        counts = labels.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#f28e2b")
        ax.set_xlabel("Class")
        ax.set_ylabel("Training rows")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(artifact_dir / filename, dpi=150)
        plt.close(fig)


def run_phase2a(input_csv: Path | str = DEFAULT_INPUT, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> dict[str, str]:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    raw["bottleneck_class"] = labels_strategy_a(raw["avg_stall_ns"])
    featured = load_featured_dataset(input_csv=input_csv, artifact_dir=artifact_path, label_builder=labels_strategy_a)

    write_class_analysis(raw, featured, artifact_path)
    save_class_distribution_plots(raw, featured, artifact_path)
    save_smote_distribution_plots(featured, artifact_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imbalance = compare_imbalance_strategies(featured, artifact_path)
        model_weights = compare_class_weights_by_model(featured, artifact_path)

    summary = {
        "class_analysis": str(artifact_path / "class_analysis.csv"),
        "imbalance_comparison": str(artifact_path / "imbalance_comparison.csv"),
        "class_weight_model_comparison": str(artifact_path / "class_weight_model_comparison.csv"),
        "best_imbalance_strategy": str(imbalance.sort_values("macro_f1", ascending=False).iloc[0]["strategy"]),
        "best_class_weight_model": str(model_weights.sort_values("macro_f1", ascending=False).iloc[0]["model"]),
    }
    (artifact_path / "phase2a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_phase2(input_csv: Path | str = DEFAULT_INPUT, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> dict[str, str]:
    """Backward-compatible entry point; intentionally runs Phase 2A only."""

    return run_phase2a(input_csv=input_csv, artifact_dir=artifact_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 imbalance and label strategy experiments.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    summary = run_phase2(args.input_csv, args.artifact_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
