"""End-to-end artifact generation for the ML pipeline foundation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.data.cleaning import clean_dataset
from ml.data.labels import derive_stall_class
from ml.features.engineering import engineer_features
from ml.models.train import groupkfold_validate, leakage_comparison, train_group_shuffle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "approach3" / "perf_metrics.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def run_pipeline(input_csv: Path | str = DEFAULT_INPUT, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> None:
    """Run cleaning, feature engineering, leakage comparison, and CV artifacts."""

    artifacts = Path(artifact_dir)
    df = pd.read_csv(input_csv)
    df["bottleneck_class"] = derive_stall_class(df["avg_stall_ns"])
    cleaned = clean_dataset(df, artifact_dir=artifacts)
    featured = engineer_features(cleaned, artifact_dir=artifacts)

    train_group_shuffle(featured, artifact_dir=artifacts)
    leakage_comparison(featured, artifact_dir=artifacts)
    groupkfold_validate(featured, artifact_dir=artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML pipeline artifacts.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    run_pipeline(args.input_csv, args.artifact_dir)


if __name__ == "__main__":
    main()

