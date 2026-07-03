"""Shared helpers for Phase 4 experiments.

Phase 4 keeps earlier experiment files untouched and centralizes the safeguards
needed by the new modules: strategy-C labels, leakage-free features, grouped
cross-validation, and ordinal-aware metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from ml.data.cleaning import clean_dataset
from ml.evaluation.ordinal_metrics import ordinal_metrics
from ml.features.engineering import engineer_features
from ml.features.selection import numeric_feature_columns
from ml.models.phase1_hybrid_labels import labels_strategy_c_quantile_min_size


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
PHASE4_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "phase4"
RANDOM_STATE = 42
N_SPLITS = 5
N_CLASSES = 4

DERIVED_LEAKAGE_COLS = [
    "stall_per_ctx_switch",
    "stall_per_syscall",
]


def ensure_phase4_dir(artifact_dir: Path | str = PHASE4_ARTIFACT_DIR) -> Path:
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_phase4_dataset(input_csv: Path | str = DEFAULT_INPUT, *, session_features: bool = False) -> pd.DataFrame:
    """Load, label with strategy C, clean, engineer, and optionally add session features."""

    df = pd.read_csv(input_csv)
    df["bottleneck_class"] = labels_strategy_c_quantile_min_size(df["avg_stall_ns"])
    cleaned = clean_dataset(df, artifact_dir=None)
    cleaned["bottleneck_class"] = labels_strategy_c_quantile_min_size(cleaned["avg_stall_ns"])
    featured = engineer_features(cleaned, artifact_dir=None)

    if session_features:
        from ml.features.session_features import add_session_features

        featured = add_session_features(featured)

    return featured


def phase4_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return leakage-free numeric features, including explicit derived-leakage drops."""

    return numeric_feature_columns(
        df,
        include_leakage=False,
        include_latency_count=True,
        extra_drop_cols=DERIVED_LEAKAGE_COLS,
    )


def grouped_cv() -> GroupKFold:
    return GroupKFold(n_splits=N_SPLITS)


def grouped_cv_splits(df: pd.DataFrame, feature_cols: Iterable[str]):
    X = df[list(feature_cols)]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    return grouped_cv().split(X, y, groups=groups)


def mean_metric_frame(rows: list[dict[str, object]], group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "mean_absolute_error",
        "mean_squared_error",
        "average_severity_distance",
        "adjacent_accuracy",
        "severe_error_rate",
        "quadratic_weighted_kappa",
    ]
    frame = pd.DataFrame(rows)
    return frame.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()


def evaluate_estimator_cv(
    df: pd.DataFrame,
    estimator_factory: Callable[[], BaseEstimator],
    *,
    model_name: str,
    experiment: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate an estimator with outer GroupKFold and ordinal-aware metrics."""

    cols = feature_cols or phase4_feature_columns(df)
    X = df[cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(grouped_cv().split(X, y, groups=groups), start=1):
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append(
            {
                "experiment": experiment,
                "model": model_name,
                "fold": fold,
                "n_features": len(cols),
                **ordinal_metrics(y.iloc[test_idx], pred),
            }
        )

    return pd.DataFrame(rows)


def write_json(path: Path | str, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render a compact markdown table without requiring optional tabulate."""

    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.4f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def params_to_jsonable(params: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


class CumulativeRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal cumulative binary RandomForest wrapper with predict_proba support."""

    def __init__(self, n_estimators: int = 80, max_depth: int | None = None, random_state: int = RANDOM_STATE):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.arange(N_CLASSES)
        self.threshold_models_ = []
        y_arr = np.asarray(y, dtype=int)
        for threshold in range(N_CLASSES - 1):
            binary_y = (y_arr > threshold).astype(int)
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state + threshold,
            )
            model.fit(X, binary_y)
            self.threshold_models_.append(model)
        return self

    def predict_proba(self, X):
        gt = []
        for model in self.threshold_models_:
            proba = model.predict_proba(X)
            if 1 in model.classes_:
                pos_idx = list(model.classes_).index(1)
                gt.append(proba[:, pos_idx])
            else:
                gt.append(np.zeros(len(X), dtype=float))

        gt_arr = np.vstack(gt).T
        gt_arr = np.minimum.accumulate(gt_arr, axis=1)
        class_proba = np.column_stack(
            [
                1.0 - gt_arr[:, 0],
                gt_arr[:, 0] - gt_arr[:, 1],
                gt_arr[:, 1] - gt_arr[:, 2],
                gt_arr[:, 2],
            ]
        )
        class_proba = np.clip(class_proba, 0, 1)
        sums = class_proba.sum(axis=1)
        sums[sums == 0] = 1.0
        return class_proba / sums[:, None]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(int)


def aligned_predict_proba(model, X) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES), dtype=float)
    classes = getattr(model, "classes_", np.arange(raw.shape[1]))
    for idx, cls in enumerate(classes):
        if 0 <= int(cls) < N_CLASSES:
            out[:, int(cls)] = raw[:, idx]
    sums = out.sum(axis=1)
    missing = sums == 0
    out[missing, 0] = 1.0
    sums = out.sum(axis=1)
    return out / sums[:, None]


def clone_with_params(estimator, params: dict[str, object]):
    model = clone(estimator)
    model.set_params(**params)
    return model
