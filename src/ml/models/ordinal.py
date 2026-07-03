"""Phase 3 ordinal classification experiments.

This module intentionally uses only strategy_c_quantile_min_size labels and
does not modify Phase 1 or Phase 2 experiment code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from ml.data.cleaning import clean_dataset
from ml.evaluation.ordinal_metrics import ordinal_metrics
from ml.features.engineering import engineer_features
from ml.features.selection import numeric_feature_columns
from ml.models.phase1_hybrid_labels import labels_strategy_c_quantile_min_size


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ORDINAL_DIRNAME = "ordinal"
RANDOM_STATE = 42
N_CLASSES = 4


def load_strategy_c_dataset(input_csv: Path | str = DEFAULT_INPUT) -> pd.DataFrame:
    """Load, label with strategy C, clean, and engineer features."""

    df = pd.read_csv(input_csv)
    df["bottleneck_class"] = labels_strategy_c_quantile_min_size(df["avg_stall_ns"])
    cleaned = clean_dataset(df, artifact_dir=None)
    cleaned["bottleneck_class"] = labels_strategy_c_quantile_min_size(cleaned["avg_stall_ns"])
    return engineer_features(cleaned, artifact_dir=None)


def rounded_clipped(pred: np.ndarray | pd.Series) -> np.ndarray:
    """Round continuous severity predictions and clip into [0, 3]."""

    return np.clip(np.rint(np.asarray(pred)), 0, N_CLASSES - 1).astype(int)


def expected_ordinal_cost_prediction(proba: np.ndarray) -> np.ndarray:
    """Choose class minimizing sum_k P(y=k) * abs(k-c)."""

    classes = np.arange(N_CLASSES)
    costs = np.zeros((proba.shape[0], N_CLASSES), dtype=float)
    for candidate in classes:
        costs[:, candidate] = np.sum(proba * np.abs(classes - candidate), axis=1)
    return np.argmin(costs, axis=1).astype(int)


def aligned_proba(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return predict_proba aligned to classes 0..3."""

    raw = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES), dtype=float)
    for idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, idx]
    row_sums = out.sum(axis=1)
    missing = row_sums == 0
    if missing.any():
        out[missing, 0] = 1.0
        row_sums = out.sum(axis=1)
    return out / row_sums[:, None]


def cumulative_probabilities(threshold_models: list[RandomForestClassifier], X: pd.DataFrame) -> np.ndarray:
    """Convert P(y > threshold) models into class probabilities."""

    gt = []
    for model in threshold_models:
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
    return np.clip(class_proba, 0, 1)


def record_metrics(
    rows: list[dict[str, object]],
    predictions: list[pd.DataFrame],
    *,
    model: str,
    approach: str,
    fold: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    metrics = ordinal_metrics(y_true, y_pred)
    rows.append({"model": model, "approach": approach, "fold": fold, **metrics})
    predictions.append(
        pd.DataFrame(
            {
                "model": model,
                "approach": approach,
                "fold": fold,
                "y_true": y_true.to_numpy(dtype=int),
                "y_pred": y_pred.astype(int),
                "error": y_pred.astype(int) - y_true.to_numpy(dtype=int),
            }
        )
    )


def run_ordinal_experiments(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Run Phase 3 ordinal experiments and write artifacts."""

    artifacts = Path(artifact_dir)
    ordinal_dir = artifacts / ORDINAL_DIRNAME
    ordinal_dir.mkdir(parents=True, exist_ok=True)

    df = load_strategy_c_dataset(input_csv)
    feature_cols = numeric_feature_columns(df, include_leakage=False)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        rf_reg = RandomForestRegressor(
            n_estimators=40,
            n_jobs=-1,
            random_state=RANDOM_STATE + fold,
        )
        rf_reg.fit(X_train, y_train)
        record_metrics(
            rows,
            prediction_frames,
            model="RandomForestRegressor",
            approach="regression_as_ordinal",
            fold=fold,
            y_true=y_test,
            y_pred=rounded_clipped(rf_reg.predict(X_test)),
        )

        hgb_reg = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.08,
            random_state=RANDOM_STATE + fold,
        )
        hgb_reg.fit(X_train, y_train)
        record_metrics(
            rows,
            prediction_frames,
            model="HistGradientBoostingRegressor",
            approach="regression_as_ordinal",
            fold=fold,
            y_true=y_test,
            y_pred=rounded_clipped(hgb_reg.predict(X_test)),
        )

        nominal = RandomForestClassifier(
            n_estimators=40,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE + fold,
        )
        nominal.fit(X_train, y_train)
        proba = aligned_proba(nominal, X_test)
        record_metrics(
            rows,
            prediction_frames,
            model="RandomForestClassifier",
            approach="nominal_argmax",
            fold=fold,
            y_true=y_test,
            y_pred=np.argmax(proba, axis=1).astype(int),
        )
        record_metrics(
            rows,
            prediction_frames,
            model="RandomForestClassifier",
            approach="expected_ordinal_distance",
            fold=fold,
            y_true=y_test,
            y_pred=expected_ordinal_cost_prediction(proba),
        )

        threshold_models: list[RandomForestClassifier] = []
        for threshold in range(N_CLASSES - 1):
            binary_y = (y_train > threshold).astype(int)
            clf = RandomForestClassifier(
                n_estimators=40,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE + fold + threshold,
            )
            clf.fit(X_train, binary_y)
            threshold_models.append(clf)

        cumulative_proba = cumulative_probabilities(threshold_models, X_test)
        record_metrics(
            rows,
            prediction_frames,
            model="CumulativeRandomForestClassifier",
            approach="cumulative_binary",
            fold=fold,
            y_true=y_test,
            y_pred=np.argmax(cumulative_proba, axis=1).astype(int),
        )

    comparison = pd.DataFrame(rows)
    output_cols = [
        "model",
        "approach",
        "fold",
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
    comparison = comparison[output_cols]
    comparison.to_csv(artifacts / "ordinal_comparison.csv", index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(ordinal_dir / "ordinal_predictions.csv", index=False)
    write_ordinal_artifacts(comparison, predictions, ordinal_dir)
    write_summary(comparison, artifacts)
    return comparison


def write_ordinal_artifacts(comparison: pd.DataFrame, predictions: pd.DataFrame, ordinal_dir: Path) -> None:
    """Write confusion matrix and error distribution for the best mean macro-F1 approach."""

    grouped = comparison.groupby(["model", "approach"], as_index=False).mean(numeric_only=True)
    best = grouped.sort_values(["macro_f1", "quadratic_weighted_kappa"], ascending=False).iloc[0]
    mask = (predictions["model"] == best["model"]) & (predictions["approach"] == best["approach"])
    best_predictions = predictions.loc[mask].copy()

    matrix = confusion_matrix(best_predictions["y_true"], best_predictions["y_pred"], labels=list(range(N_CLASSES)))
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{i}" for i in range(N_CLASSES)],
        columns=[f"pred_{i}" for i in range(N_CLASSES)],
    )
    matrix_df.to_csv(ordinal_dir / "ordinal_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(N_CLASSES), labels=list(range(N_CLASSES)))
    ax.set_yticks(np.arange(N_CLASSES), labels=list(range(N_CLASSES)))
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("Actual severity")
    ax.set_title(f"Ordinal confusion matrix: {best['approach']}")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(ordinal_dir / "ordinal_confusion_matrix.png", dpi=150)
    plt.close(fig)

    errors = best_predictions["error"]
    counts = errors.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    ax.set_xlabel("Prediction error (y_pred - y_true)")
    ax.set_ylabel("Rows")
    ax.set_title(f"Ordinal error distribution: {best['approach']}")
    fig.tight_layout()
    fig.savefig(ordinal_dir / "ordinal_error_distribution.png", dpi=150)
    plt.close(fig)


def write_summary(comparison: pd.DataFrame, artifact_dir: Path) -> None:
    """Write a concise markdown summary of ordinal results."""

    grouped = comparison.groupby(["model", "approach"], as_index=False).mean(numeric_only=True)
    best = grouped.sort_values(["macro_f1", "quadratic_weighted_kappa"], ascending=False).iloc[0]
    nominal = grouped[grouped["approach"] == "nominal_argmax"].sort_values("macro_f1", ascending=False).iloc[0]
    ordinal_only = grouped[grouped["approach"] != "nominal_argmax"].copy()
    best_ordinal = ordinal_only.sort_values(["macro_f1", "quadratic_weighted_kappa"], ascending=False).iloc[0]
    best_ordered_error = grouped.sort_values(["mean_absolute_error", "severe_error_rate"], ascending=True).iloc[0]

    lines = [
        "# Ordinal Classification Summary",
        "",
        "Labels: `strategy_c_quantile_min_size` only.",
        "",
        "## Best Overall Model",
        "",
        f"- Model: `{best['model']}`",
        f"- Approach: `{best['approach']}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- MCC: `{best['mcc']:.4f}`",
        f"- Mean absolute error: `{best['mean_absolute_error']:.4f}`",
        f"- Adjacent accuracy: `{best['adjacent_accuracy']:.4f}`",
        f"- Severe error rate: `{best['severe_error_rate']:.4f}`",
        f"- Quadratic weighted kappa: `{best['quadratic_weighted_kappa']:.4f}`",
        "",
        "## Best Ordinal Approach",
        "",
        f"- Model: `{best_ordinal['model']}`",
        f"- Approach: `{best_ordinal['approach']}`",
        f"- Macro F1: `{best_ordinal['macro_f1']:.4f}`",
        f"- Mean absolute error: `{best_ordinal['mean_absolute_error']:.4f}`",
        f"- Adjacent accuracy: `{best_ordinal['adjacent_accuracy']:.4f}`",
        f"- Severe error rate: `{best_ordinal['severe_error_rate']:.4f}`",
        f"- Quadratic weighted kappa: `{best_ordinal['quadratic_weighted_kappa']:.4f}`",
        "",
        "## Comparison Against Nominal Classification",
        "",
        f"- Nominal baseline: `{nominal['model']}` with `{nominal['approach']}`",
        f"- Nominal macro F1: `{nominal['macro_f1']:.4f}`",
        f"- Nominal mean absolute error: `{nominal['mean_absolute_error']:.4f}`",
        f"- Nominal severe error rate: `{nominal['severe_error_rate']:.4f}`",
        f"- Lowest mean absolute error: `{best_ordered_error['model']}` with `{best_ordered_error['approach']}` at `{best_ordered_error['mean_absolute_error']:.4f}`",
        "",
        "## Recommendations",
        "",
        "- Use nominal argmax if Macro F1 is the primary decision metric for `strategy_c_quantile_min_size` labels.",
        "- Use the lowest-error ordinal approach when adjacent severity mistakes are acceptable but large severity jumps are costly.",
        "- Track macro F1 together with mean absolute error, adjacent accuracy, severe error rate, and quadratic weighted kappa.",
        "- Keep Phase 1 and Phase 2 nominal experiments as baselines; this phase should remain a separate ordinal comparison layer.",
        "",
        "## Mean Metrics",
        "",
        "| Model | Approach | Macro F1 | MAE | Adjacent Accuracy | Severe Error Rate | QWK |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in grouped.sort_values("macro_f1", ascending=False).iterrows():
        lines.append(
            f"| `{row['model']}` | `{row['approach']}` | {row['macro_f1']:.4f} | "
            f"{row['mean_absolute_error']:.4f} | {row['adjacent_accuracy']:.4f} | "
            f"{row['severe_error_rate']:.4f} | {row['quadratic_weighted_kappa']:.4f} |"
        )

    (artifact_dir / "ordinal_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 ordinal classification experiments.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    comparison = run_ordinal_experiments(args.input_csv, args.artifact_dir)
    print(comparison.groupby(["model", "approach"]).mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
