"""Phase 4 ensemble experiments using tuned models and the Phase 3 ordinal model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.models.hyperparameter_tuning import candidate_models
from ml.models.phase4_common import (
    CumulativeRandomForestClassifier,
    DEFAULT_INPUT,
    PHASE4_ARTIFACT_DIR,
    RANDOM_STATE,
    aligned_predict_proba,
    ensure_phase4_dir,
    grouped_cv,
    load_phase4_dataset,
    markdown_table,
    mean_metric_frame,
    ordinal_metrics,
    phase4_feature_columns,
)


def _base_estimators(out_dir: Path) -> dict[str, object]:
    models = candidate_models()
    estimators: dict[str, object] = {}
    params_path = out_dir / "best_model_params.json"
    if params_path.exists():
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        for name, item in payload.items():
            if name in models:
                estimator = clone(models[name][0])
                estimator.set_params(**item["params"])
                estimators[name] = estimator

    if not estimators:
        estimators["random_forest"] = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    estimators["cumulative_random_forest"] = CumulativeRandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE)
    return estimators


def _weights_from_artifacts(out_dir: Path, names: list[str]) -> np.ndarray:
    comparison_path = out_dir / "hyperparameter_comparison.csv"
    weights = np.ones(len(names), dtype=float)
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        lookup = dict(zip(comparison["model"], comparison["macro_f1"]))
        for idx, name in enumerate(names):
            weights[idx] = max(float(lookup.get(name, weights[idx])), 0.001)
    return weights / weights.sum()


def run_ensemble_experiments(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = PHASE4_ARTIFACT_DIR,
) -> pd.DataFrame:
    out_dir = ensure_phase4_dir(artifact_dir)
    df = load_phase4_dataset(input_csv)
    feature_cols = phase4_feature_columns(df)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)
    estimators = _base_estimators(out_dir)
    names = list(estimators)
    weights = _weights_from_artifacts(out_dir, names)
    rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(grouped_cv().split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx]

        fitted = {}
        test_probas = []
        for name, estimator in estimators.items():
            model = clone(estimator)
            model.fit(X_train, y_train)
            fitted[name] = model
            test_probas.append(aligned_predict_proba(model, X_test))

        proba_stack = np.stack(test_probas, axis=0)
        soft_pred = np.argmax(proba_stack.mean(axis=0), axis=1)
        rows.append({"experiment": "ensemble", "model": "soft_voting", "fold": fold, **ordinal_metrics(y_test, soft_pred)})

        weighted_pred = np.argmax(np.average(proba_stack, axis=0, weights=weights), axis=1)
        rows.append({"experiment": "ensemble", "model": "weighted_probability", "fold": fold, **ordinal_metrics(y_test, weighted_pred)})

        meta_X = np.zeros((len(X_train), len(names) * 4), dtype=float)
        for inner_train_idx, inner_valid_idx in grouped_cv().split(X_train, y_train, groups=train_groups):
            inner_blocks = []
            for estimator in estimators.values():
                inner_model = clone(estimator)
                inner_model.fit(X_train.iloc[inner_train_idx], y_train.iloc[inner_train_idx])
                inner_blocks.append(aligned_predict_proba(inner_model, X_train.iloc[inner_valid_idx]))
            meta_X[inner_valid_idx] = np.hstack(inner_blocks)

        meta_model = LogisticRegression(max_iter=1000, multi_class="auto", random_state=RANDOM_STATE)
        meta_model.fit(meta_X, y_train)
        meta_test = np.hstack(test_probas)
        stack_pred = meta_model.predict(meta_test)
        rows.append({"experiment": "ensemble", "model": "stacking_classifier", "fold": fold, **ordinal_metrics(y_test, stack_pred)})

    comparison = mean_metric_frame(rows, ["experiment", "model"])
    comparison.to_csv(out_dir / "ensemble_comparison.csv", index=False)
    write_summary(comparison, names, weights, out_dir)
    return comparison


def write_summary(comparison: pd.DataFrame, names: list[str], weights: np.ndarray, out_dir: Path) -> None:
    best = comparison.sort_values("macro_f1", ascending=False).iloc[0]
    weight_text = ", ".join(f"{name}={weight:.3f}" for name, weight in zip(names, weights))
    lines = [
        "# Phase 4 Ensemble Summary",
        "",
        f"- Best ensemble: `{best['model']}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- MCC: `{best['mcc']:.4f}`",
        f"- Base estimators: `{', '.join(names)}`",
        f"- Weighted ensemble weights: `{weight_text}`",
        "",
        "## Mean Metrics",
        "",
        markdown_table(comparison),
    ]
    (out_dir / "ensemble_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 ensemble experiments.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=PHASE4_ARTIFACT_DIR)
    args = parser.parse_args()
    print(run_ensemble_experiments(args.input_csv, args.artifact_dir).to_string(index=False))


if __name__ == "__main__":
    main()
