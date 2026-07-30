"""Phase 4 final artifact aggregation and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.models.phase4_common import PHASE4_ARTIFACT_DIR, ensure_phase4_dir, markdown_table


METRIC_COLS = [
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


def _read_optional(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_artifact"] = source
    return df


def build_final_recommendation(artifact_dir: Path | str = PHASE4_ARTIFACT_DIR) -> pd.DataFrame:
    out_dir = ensure_phase4_dir(artifact_dir)
    frames = [
        _read_optional(out_dir / "hyperparameter_comparison.csv", "hyperparameter_tuning"),
        _read_optional(out_dir / "feature_subset_comparison.csv", "feature_selection"),
        _read_optional(out_dir / "session_feature_comparison.csv", "session_features"),
        _read_optional(out_dir / "ensemble_comparison.csv", "ensemble"),
    ]
    comparison = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if comparison.empty:
        raise FileNotFoundError("No Phase 4 comparison artifacts found")

    for col in METRIC_COLS:
        if col not in comparison.columns:
            comparison[col] = pd.NA
    if "experiment" not in comparison.columns:
        comparison["experiment"] = comparison["source_artifact"]
    if "model" not in comparison.columns:
        comparison["model"] = comparison["source_artifact"]
    if "n_features" not in comparison.columns:
        comparison["n_features"] = pd.NA

    output_cols = ["source_artifact", "experiment", "model", "n_features", *METRIC_COLS]
    comparison = comparison[output_cols].sort_values("macro_f1", ascending=False)
    comparison.to_csv(out_dir / "final_model_comparison.csv", index=False)
    write_recommendation(comparison, out_dir)
    return comparison


def _best_params(out_dir: Path, model_name: str) -> dict[str, object]:
    params_path = out_dir / "best_model_params.json"
    if not params_path.exists():
        return {}
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    return payload.get(model_name, {}).get("params", {})


def _best_feature_subset(out_dir: Path) -> str:
    path = out_dir / "feature_subset_comparison.csv"
    if not path.exists():
        return "not available"
    df = pd.read_csv(path)
    best_f1 = df["macro_f1"].max()
    threshold = best_f1 * 0.99
    smallest = df[df["macro_f1"] >= threshold].sort_values(["n_features", "macro_f1"]).iloc[0]
    return f"{smallest['experiment']} ({int(smallest['n_features'])} features)"


def _best_ensemble(out_dir: Path) -> str:
    path = out_dir / "ensemble_comparison.csv"
    if not path.exists():
        return "not available"
    df = pd.read_csv(path)
    best = df.sort_values("macro_f1", ascending=False).iloc[0]
    return f"{best['model']} (Macro F1 {best['macro_f1']:.4f})"


def write_recommendation(comparison: pd.DataFrame, out_dir: Path) -> None:
    best = comparison.iloc[0]
    params = _best_params(out_dir, str(best["model"]))
    best_subset = _best_feature_subset(out_dir)
    best_ensemble = _best_ensemble(out_dir)
    lines = [
        "# Phase 4 Final Recommendation",
        "",
        f"- Best model: `{best['model']}`",
        f"- Source: `{best['source_artifact']}` / `{best['experiment']}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- MCC: `{best['mcc']:.4f}`",
        f"- MAE: `{best['mean_absolute_error']:.4f}`",
        f"- Quadratic weighted kappa: `{best['quadratic_weighted_kappa']:.4f}`",
        f"- Best parameters: `{params}`",
        f"- Best feature subset: `{best_subset}`",
        f"- Best ensemble: `{best_ensemble}`",
        "",
        "## Expected Production Pipeline",
        "",
        "1. Load eBPF metric rows.",
        "2. Apply Phase 1 cleaning filters.",
        "3. Generate `strategy_c_quantile_min_size` labels for training only.",
        "4. Engineer base and selected Phase 4 session features.",
        "5. Select leakage-free numeric features only.",
        "6. Train the recommended Phase 4 model with grouped validation by `session_label`.",
        "7. Report nominal and ordinal metrics together.",
        "",
        "## Final Comparison",
        "",
        markdown_table(comparison),
    ]
    (out_dir / "final_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 4 final recommendation artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=PHASE4_ARTIFACT_DIR)
    args = parser.parse_args()
    print(build_final_recommendation(args.artifact_dir).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
