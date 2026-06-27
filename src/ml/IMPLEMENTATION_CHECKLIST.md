# Implementation Checklist

## Task 1 - Dataset Cleaning

- [x] Added `clean_dataset(df)`.
- [x] Enforced `pid > 0`.
- [x] Enforced `ctx_switches >= 3`.
- [x] Enforced `latency_count >= 3`.
- [x] Enforced `total_runtime_ns > 0`.
- [x] Generated `src/ml/artifacts/cleaning_report.json`.

## Task 2 - Leakage Removal

- [x] Defined `LEAKAGE_COLS = ["avg_stall_ns", "stall_ns", "max_stall_ns"]`.
- [x] Excluded leakage columns from normal feature selection.
- [x] Added comparison modes for with leakage, without leakage, and without `latency_count`.
- [x] Generated `src/ml/artifacts/leakage_comparison.csv`.

## Task 3 - Feature Engineering

- [x] Added `engineer_features(df)`.
- [x] Added CPU features.
- [x] Added memory features.
- [x] Added syscall features.
- [x] Added lock features.
- [x] Added cross-domain features.
- [x] Generated `src/ml/artifacts/feature_cols.json`.

## Task 4 - Evaluation Framework

- [x] Added macro precision.
- [x] Added macro recall.
- [x] Added macro F1.
- [x] Added balanced accuracy.
- [x] Added MCC.
- [x] Added confusion matrix helpers.

## Task 5 - Cross Validation

- [x] Kept `GroupShuffleSplit`.
- [x] Added `GroupKFold(n_splits=5)`.
- [x] Used `session_label` as the grouping key.
- [x] Generated `src/ml/artifacts/groupkfold_results.csv`.

## Execution Status

- [x] Code files created under `src/ml/`.
- [x] Local Python 3.12.5 runtime installed under `.tools/python312`.
- [x] Required Python dependencies installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nbclient`, `nbformat`, `ipykernel`.
- [x] All `src/ml` Python modules imported successfully.
- [x] `python -m compileall src/ml` completed successfully.
- [x] Feature engineering executed successfully on real CSV rows.
- [x] Leakage exclusion verified: no `avg_stall_ns`, `stall_ns`, or `max_stall_ns` in the no-leakage feature set.
- [x] `latency_count` comparison verified: present in the standard no-leakage feature set and absent in the `without_latency_count` feature set.
- [x] `GroupKFold(n_splits=5)` verified in `groupkfold_validate` and executed through the pipeline.
- [x] End-to-end pipeline executed successfully with `python -m ml.pipeline`.
- [x] At least one explicit training run executed successfully with leakage-free features.
- [x] Artifacts regenerated from the available `src/approach3/perf_metrics.csv` dataset by the Python pipeline.
- [x] Notebooks executed successfully with the local `perfdiag-phase1` kernel.

## Runtime Evidence

- Python executable: `.tools/python312/python.exe`
- Pipeline command: `.tools/python312/python.exe -m ml.pipeline --input-csv src/approach3/perf_metrics.csv --artifact-dir src/ml/artifacts`
- Explicit training metrics: macro precision `0.335582`, macro recall `0.348422`, macro F1 `0.336715`, balanced accuracy `0.522632`, MCC `0.016641`
- Notebook outputs: `src/ml/notebooks/01_pipeline_foundation.executed.ipynb`, `src/ml/notebooks/02_artifact_review.executed.ipynb`

## Notes

- Installing the full `jupyter` meta-package initially failed because Windows long-path support blocked a JupyterLab asset path. The required notebook execution dependencies were installed separately, and notebooks executed successfully through `nbclient` and `ipykernel`.
- Scikit-learn emitted class-distribution warnings during validation because rare classes are absent from some validation splits while predictions may include them. The runs completed and artifacts were regenerated.
