"""Training, leakage comparison, and grouped cross-validation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ml.evaluation.metrics import classification_metrics, confusion_matrix_frame, save_confusion_matrix_plot
from ml.features.selection import numeric_feature_columns


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def build_classifier(random_state: int = 42) -> RandomForestClassifier:
    """Create the baseline classifier used across experiments."""

    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def train_group_shuffle(
    df: pd.DataFrame,
    *,
    include_leakage: bool = False,
    include_latency_count: bool = True,
    random_state: int = 42,
    artifact_dir: Optional[Path | str] = DEFAULT_ARTIFACT_DIR,
) -> dict[str, object]:
    """Train/evaluate a single GroupShuffleSplit holdout."""

    feature_cols = numeric_feature_columns(
        df,
        include_leakage=include_leakage,
        include_latency_count=include_latency_count,
    )
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = build_classifier(random_state=random_state)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    metrics = classification_metrics(y.iloc[test_idx], pred)
    matrix = confusion_matrix_frame(y.iloc[test_idx], pred, labels=[0, 1, 2, 3])

    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(out_dir / "group_shuffle_confusion_matrix.csv")
        save_confusion_matrix_plot(
            y.iloc[test_idx],
            pred,
            out_dir / "group_shuffle_confusion_matrix.png",
            labels=[0, 1, 2, 3],
        )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "confusion_matrix": matrix,
    }


def leakage_comparison(
    df: pd.DataFrame,
    artifact_dir: Optional[Path | str] = DEFAULT_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Compare with leakage, without leakage, and without latency_count."""

    scenarios = [
        ("with_leakage", True, True),
        ("without_leakage", False, True),
        ("without_latency_count", False, False),
    ]
    rows = []
    for name, include_leakage, include_latency in scenarios:
        result = train_group_shuffle(
            df,
            include_leakage=include_leakage,
            include_latency_count=include_latency,
            artifact_dir=None,
        )
        rows.append(
            {
                "scenario": name,
                "include_leakage": include_leakage,
                "include_latency_count": include_latency,
                "n_features": len(result["feature_cols"]),
                **result["metrics"],
            }
        )

    comparison = pd.DataFrame(rows)
    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_dir / "leakage_comparison.csv", index=False)
    return comparison


def groupkfold_validate(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    artifact_dir: Optional[Path | str] = DEFAULT_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Run GroupKFold using session_label groups."""

    feature_cols = numeric_feature_columns(df, include_leakage=False, include_latency_count=True)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(f"GroupKFold requires at least {n_splits} session_label groups; found {n_groups}")

    rows = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups), start=1):
        model = build_classifier(random_state=random_state + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                **classification_metrics(y.iloc[test_idx], pred),
            }
        )

    results = pd.DataFrame(rows)
    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "groupkfold_results.csv", index=False)
    return results

