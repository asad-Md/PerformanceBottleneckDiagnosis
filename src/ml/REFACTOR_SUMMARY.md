# ML Pipeline Refactor Summary

## Scope

Created a new self-contained `src/ml/` package for the data pipeline foundation. No eBPF or C runtime files were modified.

## Package Layout

- `data/cleaning.py`: implements `clean_dataset(df)` with the required minimum filters and emits `artifacts/cleaning_report.json`.
- `data/labels.py`: derives the four-class ML target from `avg_stall_ns`.
- `features/engineering.py`: implements `engineer_features(df)` for CPU, memory, syscall, lock, and cross-domain signals and emits `artifacts/feature_cols.json`.
- `features/selection.py`: centralizes leakage removal through `LEAKAGE_COLS = ["avg_stall_ns", "stall_ns", "max_stall_ns"]` and supports evaluating `latency_count`.
- `evaluation/metrics.py`: reusable macro precision, macro recall, macro F1, balanced accuracy, MCC, and confusion matrix helpers.
- `models/train.py`: Random Forest baseline, `GroupShuffleSplit`, leakage comparison, and `GroupKFold(n_splits=5)` using `session_label`.
- `pipeline.py`: end-to-end artifact generation entry point.
- `notebooks/`: lightweight executable notebooks for pipeline execution and artifact inspection.

## Artifacts

Generated under `src/ml/artifacts/`:

- `cleaning_report.json`
- `feature_cols.json`
- `leakage_comparison.csv`
- `groupkfold_results.csv`

## Notes

- The target label is derived from `avg_stall_ns`, so `avg_stall_ns`, `stall_ns`, and `max_stall_ns` are excluded from normal training features.
- `latency_count` is kept in the standard no-leakage scenario and removed in a separate comparison scenario.
- Grouped validation uses `session_label` to avoid row-level leakage across train/test splits.

