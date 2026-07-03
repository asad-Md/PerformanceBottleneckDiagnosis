"""Phase 4 hyperparameter optimization for bottleneck severity classifiers."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV

from ml.models.phase4_common import (
    DEFAULT_INPUT,
    PHASE4_ARTIFACT_DIR,
    RANDOM_STATE,
    ensure_phase4_dir,
    evaluate_estimator_cv,
    grouped_cv,
    load_phase4_dataset,
    markdown_table,
    params_to_jsonable,
    phase4_feature_columns,
    write_json,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


def candidate_models() -> dict[str, tuple[object, dict[str, list[object]]]]:
    models: dict[str, tuple[object, dict[str, list[object]]]] = {
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [None, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        )
    }

    if XGBClassifier is not None:
        models["xgboost"] = (
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1, 2, 5],
            },
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = (
            LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "max_depth": [-1, 5, 10, 20],
                "min_child_samples": [10, 20, 50, 100],
            },
        )

    return models


def run_hyperparameter_tuning(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = PHASE4_ARTIFACT_DIR,
    n_iter: int = 20,
    tuning_max_rows: int = 10000,
) -> pd.DataFrame:
    """Run RandomizedSearchCV with GroupKFold and save Phase 4 tuning artifacts."""

    if not 20 <= n_iter <= 50:
        raise ValueError("n_iter must be between 20 and 50")

    out_dir = ensure_phase4_dir(artifact_dir)
    df = load_phase4_dataset(input_csv)
    tuning_df = sample_for_tuning(df, tuning_max_rows)
    feature_cols = phase4_feature_columns(df)
    X = tuning_df[feature_cols]
    y = tuning_df["bottleneck_class"].astype(int)
    groups = tuning_df["session_label"].astype(str)
    scoring = make_scorer(f1_score, average="macro", zero_division=0)

    summary_rows: list[pd.DataFrame] = []
    best_params: dict[str, object] = {}

    for model_name, (estimator, params) in candidate_models().items():
        search_estimator = clone(estimator)
        if "n_jobs" in search_estimator.get_params():
            search_estimator.set_params(n_jobs=1)
        search = RandomizedSearchCV(
            estimator=search_estimator,
            param_distributions=params,
            n_iter=n_iter,
            scoring=scoring,
            cv=grouped_cv(),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            pre_dispatch="2*n_jobs",
            refit=True,
            verbose=1,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y, groups=groups)

        model_params = params_to_jsonable(search.best_params_)
        best_params[model_name] = {
            "best_cv_macro_f1": float(search.best_score_),
            "tuning_rows": int(len(tuning_df)),
            "full_evaluation_rows": int(len(df)),
            "params": model_params,
        }

        evaluated = evaluate_estimator_cv(
            df,
            lambda estimator=estimator, model_params=model_params: clone(estimator).set_params(**model_params),
            model_name=model_name,
            experiment="hyperparameter_tuning",
            feature_cols=feature_cols,
        )
        summary_rows.append(evaluated)

    comparison = pd.concat(summary_rows, ignore_index=True)
    mean_comparison = comparison.groupby(["experiment", "model", "n_features"], as_index=False).mean(numeric_only=True)
    mean_comparison.to_csv(out_dir / "hyperparameter_comparison.csv", index=False)
    write_json(out_dir / "best_model_params.json", best_params)
    write_summary(mean_comparison, best_params, out_dir)
    return mean_comparison


def sample_for_tuning(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Deterministically downsample within session/class cells for faster search."""

    if max_rows <= 0 or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    sampled = (
        df.groupby(["session_label", "bottleneck_class"], group_keys=False, dropna=False)
        .sample(frac=frac, random_state=RANDOM_STATE)
        .sort_index()
    )
    return sampled


def write_summary(comparison: pd.DataFrame, best_params: dict[str, object], out_dir: Path) -> None:
    best = comparison.sort_values("macro_f1", ascending=False).iloc[0]
    lines = [
        "# Phase 4 Hyperparameter Tuning Summary",
        "",
        f"- Best model: `{best['model']}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- MCC: `{best['mcc']:.4f}`",
        f"- N features: `{int(best['n_features'])}`",
        f"- Tuning rows: `{next(iter(best_params.values()))['tuning_rows']}`",
        f"- Full evaluation rows: `{next(iter(best_params.values()))['full_evaluation_rows']}`",
        "",
        "## Tuned Parameters",
        "",
    ]
    for model_name, payload in best_params.items():
        lines.append(f"- `{model_name}`: `{payload['params']}`")
    lines.extend(["", "## Mean Metrics", "", markdown_table(comparison)])
    (out_dir / "hyperparameter_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 hyperparameter tuning.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=PHASE4_ARTIFACT_DIR)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--tuning-max-rows", type=int, default=10000)
    args = parser.parse_args()
    print(
        run_hyperparameter_tuning(
            args.input_csv,
            args.artifact_dir,
            args.n_iter,
            tuning_max_rows=args.tuning_max_rows,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
