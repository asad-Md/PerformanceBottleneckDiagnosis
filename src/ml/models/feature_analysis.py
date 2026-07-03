"""Phase 4 feature importance, permutation importance, and subset evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from ml.models.hyperparameter_tuning import candidate_models
from ml.models.phase4_common import (
    DEFAULT_INPUT,
    PHASE4_ARTIFACT_DIR,
    RANDOM_STATE,
    ensure_phase4_dir,
    evaluate_estimator_cv,
    load_phase4_dataset,
    markdown_table,
    phase4_feature_columns,
)


def _load_best_model(out_dir: Path):
    params_path = out_dir / "best_model_params.json"
    models = candidate_models()
    if params_path.exists():
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        best_name = max(payload, key=lambda name: payload[name]["best_cv_macro_f1"])
        estimator = models[best_name][0]
        estimator.set_params(**payload[best_name]["params"])
        return best_name, estimator, payload[best_name]["params"]

    return (
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {"n_estimators": 300, "class_weight": "balanced", "max_features": "sqrt"},
    )


def _tree_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        importance = np.zeros(len(feature_cols), dtype=float)
    return pd.DataFrame({"feature": feature_cols, "tree_importance": importance}).sort_values(
        "tree_importance", ascending=False
    )


def _safe_model_factory(estimator):
    from sklearn.base import clone

    return lambda: clone(estimator)


def run_feature_analysis(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = PHASE4_ARTIFACT_DIR,
) -> pd.DataFrame:
    out_dir = ensure_phase4_dir(artifact_dir)
    df = load_phase4_dataset(input_csv)
    feature_cols = phase4_feature_columns(df)
    X = df[feature_cols]
    y = df["bottleneck_class"].astype(int)
    groups = df["session_label"].astype(str)

    model_name, estimator, params = _load_best_model(out_dir)
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    train_idx, test_idx = next(GroupKFold(n_splits=5).split(X, y, groups=groups))
    fitted = clone(estimator)
    fitted.fit(X.iloc[train_idx], y.iloc[train_idx])

    feature_importance = _tree_importance(fitted, feature_cols)
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    perm = permutation_importance(
        fitted,
        X.iloc[test_idx],
        y.iloc[test_idx],
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    permutation.to_csv(out_dir / "permutation_importance.csv", index=False)

    ranking = feature_importance.merge(permutation, on="feature", how="outer").fillna(0)
    for col in ["tree_importance", "permutation_importance_mean"]:
        max_value = ranking[col].max()
        ranking[f"{col}_norm"] = ranking[col] / max_value if max_value else 0
    ranking["combined_importance"] = ranking["tree_importance_norm"] + ranking["permutation_importance_mean_norm"]
    top_features = ranking.sort_values("combined_importance", ascending=False)
    top_features.to_csv(out_dir / "top_features.csv", index=False)

    shap_status = _write_optional_shap(fitted, X.iloc[test_idx], feature_cols, out_dir)

    subset_frames = []
    for n_features in [10, 15, 20, 30, len(feature_cols)]:
        subset = top_features["feature"].head(min(n_features, len(feature_cols))).tolist()
        subset_frames.append(
            evaluate_estimator_cv(
                df,
                _safe_model_factory(estimator),
                model_name=model_name,
                experiment=f"top_{n_features if n_features != len(feature_cols) else 'all'}_features",
                feature_cols=subset,
            )
        )
    subset = pd.concat(subset_frames, ignore_index=True)
    subset_mean = subset.groupby(["experiment", "model", "n_features"], as_index=False).mean(numeric_only=True)
    subset_mean.to_csv(out_dir / "feature_subset_comparison.csv", index=False)
    write_summary(model_name, params, top_features, subset_mean, shap_status, out_dir)
    return subset_mean


def _write_optional_shap(model, X_eval: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> str:
    try:
        import shap  # type: ignore
    except Exception:
        return "skipped: shap is not installed"

    sample = X_eval.head(min(1000, len(X_eval)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    return "generated"


def smallest_within_one_percent(comparison: pd.DataFrame) -> pd.Series:
    best_f1 = comparison["macro_f1"].max()
    threshold = best_f1 * 0.99
    return comparison[comparison["macro_f1"] >= threshold].sort_values(["n_features", "macro_f1"]).iloc[0]


def write_summary(
    model_name: str,
    params: dict[str, object],
    top_features: pd.DataFrame,
    subset_comparison: pd.DataFrame,
    shap_status: str,
    out_dir: Path,
) -> None:
    best_subset = subset_comparison.sort_values("macro_f1", ascending=False).iloc[0]
    smallest = smallest_within_one_percent(subset_comparison)
    lines = [
        "# Phase 4 Feature Analysis Summary",
        "",
        f"- Model analyzed: `{model_name}`",
        f"- Parameters: `{params}`",
        f"- Best subset: `{best_subset['experiment']}` with Macro F1 `{best_subset['macro_f1']:.4f}`",
        f"- Smallest subset within 1% of best Macro F1: `{smallest['experiment']}` ({int(smallest['n_features'])} features)",
        f"- SHAP: {shap_status}",
        "",
        "## Top 20 Features",
        "",
        markdown_table(top_features.head(20)),
        "",
        "## Subset Metrics",
        "",
        markdown_table(subset_comparison),
    ]
    (out_dir / "feature_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 feature analysis.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=PHASE4_ARTIFACT_DIR)
    args = parser.parse_args()
    print(run_feature_analysis(args.input_csv, args.artifact_dir).to_string(index=False))


if __name__ == "__main__":
    main()
