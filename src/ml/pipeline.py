"""End-to-end artifact generation for the ML pipeline foundation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ml.data.cleaning import clean_dataset
from ml.data.labels import derive_stall_class
from ml.features.engineering import engineer_features
from ml.models.train import (
    groupkfold_validate,
    leakage_comparison,
    train_group_shuffle,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def run_pipeline(
    input_csv: Path | str = DEFAULT_INPUT,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> None:
    """Run cleaning, feature engineering, training, evaluation, and artifact generation."""

    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_csv)

    # Generate training labels
    df["bottleneck_class"] = derive_stall_class(df["avg_stall_ns"])

    # Cleaning
    cleaned = clean_dataset(df, artifact_dir=artifacts)

    # Feature engineering
    featured = engineer_features(cleaned, artifact_dir=artifacts)

    # Train Random Forest
    result = train_group_shuffle(featured, artifact_dir=artifacts)

    # Save model for realtime inference.
    # NOTE:
    # feature_cols.json contains only engineered features.
    # The trained model stores the COMPLETE feature list
    # (raw + engineered) in model.feature_names_in_.
    model_path = artifacts / "rf_bottleneck_classifier.pkl"
    joblib.dump(result["model"], model_path)

    # Evaluation
    leakage_comparison(featured, artifact_dir=artifacts)
    groupkfold_validate(featured, artifact_dir=artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ML pipeline artifacts."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )

    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    main()