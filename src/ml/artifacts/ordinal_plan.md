# Ordinal Classification Integration Plan

## Current Repository State

### 1. Current Label Generation Code

- Primary labels live in `src/ml/data/labels.py`.
- `derive_stall_class(avg_stall_ns)` maps `avg_stall_ns` into ordered severity classes:
  - class 0: `< 2 ms`
  - class 1: `2 ms to < 10 ms`
  - class 2: `10 ms to < 50 ms`
  - class 3: `>= 50 ms`
- Phase 2 alternative labels live in:
  - `src/ml/models/phase2_imbalance.py`
  - `src/ml/models/phase1_hybrid_labels.py`
- Existing alternative strategies include:
  - fixed thresholds: `0-2ms`, `2-8ms`, `8-25ms`, `25ms+`
  - percentile `pd.qcut`
  - quantile labels with minimum class-size guard
  - hybrid labels with fixed normal/high boundaries and qcut middle classes

### 2. Current Training Pipeline

- Main Phase 1 pipeline: `src/ml/pipeline.py`
  - loads `src/approach3/perf_metrics.csv`
  - derives `bottleneck_class`
  - applies `clean_dataset`
  - applies `engineer_features`
  - runs `train_group_shuffle`
  - runs `leakage_comparison`
  - runs `groupkfold_validate`
- Main training code: `src/ml/models/train.py`
  - `build_classifier`: RandomForestClassifier with `class_weight="balanced"`
  - `train_group_shuffle`: `GroupShuffleSplit`
  - `groupkfold_validate`: `GroupKFold(n_splits=5)` grouped by `session_label`
- Phase 2 imbalance code: `src/ml/models/phase2_imbalance.py`
  - Random Forest, LightGBM, XGBoost model builders
  - class weights
  - SMOTE and BorderlineSMOTE on training folds only
  - GroupKFold preserved
- Hybrid label comparison: `src/ml/models/phase1_hybrid_labels.py`
  - compares label strategies using Random Forest and GroupKFold

### 3. Existing Evaluation Metrics

- Shared metrics live in `src/ml/evaluation/metrics.py`.
- Current metrics:
  - macro precision
  - macro recall
  - macro F1
  - balanced accuracy
  - MCC
  - confusion matrix DataFrame
  - confusion matrix plot
- Current limitation:
  - These metrics treat severity classes as nominal categories.
  - They do not distinguish near misses from severe ordinal mistakes.
  - Example: predicting class 1 for class 2 is penalized the same way as predicting class 0 for class 3.

### 4. Existing Artifacts

Core Phase 1 artifacts:

- `src/ml/artifacts/cleaning_report.json`
- `src/ml/artifacts/feature_cols.json`
- `src/ml/artifacts/leakage_comparison.csv`
- `src/ml/artifacts/groupkfold_results.csv`
- `src/ml/artifacts/group_shuffle_confusion_matrix.csv`
- `src/ml/artifacts/group_shuffle_confusion_matrix.png`

Phase 2 imbalance artifacts:

- `src/ml/artifacts/class_analysis.csv`
- `src/ml/artifacts/class_distribution.png`
- `src/ml/artifacts/session_class_distribution.png`
- `src/ml/artifacts/class_distribution_before.png`
- `src/ml/artifacts/class_distribution_after.png`
- `src/ml/artifacts/imbalance_comparison.csv`
- `src/ml/artifacts/class_weight_model_comparison.csv`
- `src/ml/artifacts/phase2a_summary.json`
- `src/ml/artifacts/phase2_summary.json`

Hybrid label artifacts:

- `src/ml/artifacts/hybrid_label_comparison.csv`
- `src/ml/artifacts/hybrid_label_distribution.csv`
- `src/ml/artifacts/hybrid_label_recommendation.md`
- `src/ml/artifacts/label_strategy_comparison.csv`

Current evidence:

- Leakage columns produce perfect scores.
- Leakage-free Phase 1 macro F1 is about `0.337`.
- Cleaned fixed-threshold class counts are highly imbalanced:
  - class 0: `178936`
  - class 1: `855`
  - class 2: `81`
  - class 3: `20`
- Existing hybrid-label comparison shows `strategy_b_percentile_qcut` and `strategy_c_quantile_min_size` with macro F1 around `0.4945`.

## Where Ordinal Classification Fits With Minimal Disruption

The smallest safe integration point is a new experiment module under `src/ml/models/`, leaving Phase 1 code unchanged.

Recommended new module:

- `src/ml/models/ordinal.py`

Recommended support additions:

- `src/ml/evaluation/ordinal_metrics.py`
- optional: `src/ml/data/ordinal_labels.py` only if label utilities become too large for `data/labels.py`

Do not change:

- `src/ml/pipeline.py`
- `src/ml/models/train.py`
- existing notebooks
- eBPF code

The ordinal module should reuse:

- `clean_dataset`
- `engineer_features`
- `numeric_feature_columns`
- `GroupKFold(session_label)`
- existing label builders from `data/labels.py`, `phase2_imbalance.py`, or `phase1_hybrid_labels.py`

## Proposed Architecture

### A. Ordinal Metrics Layer

Create `src/ml/evaluation/ordinal_metrics.py`.

Add metrics that respect ordered severity:

- mean absolute class error: `mean(abs(y_true - y_pred))`
- mean squared class error: `mean((y_true - y_pred) ** 2)`
- adjacent accuracy: percentage where `abs(y_true - y_pred) <= 1`
- severe error rate: percentage where `abs(y_true - y_pred) >= 2`
- quadratic weighted kappa
- ordinal confusion matrix with distance-weighted error summary

Keep existing nominal metrics for comparison.

### B. Ordinal Model Strategies

Create `src/ml/models/ordinal.py`.

Implement three low-disruption baselines:

1. Regression-as-ordinal:
   - train `RandomForestRegressor`, `HistGradientBoostingRegressor`, or XGBoost regressor if available
   - predict continuous severity
   - round/clip to valid classes `0..3`

2. Cumulative binary classifiers:
   - train one classifier per threshold:
     - `y > 0`
     - `y > 1`
     - `y > 2`
   - combine probabilities into final ordinal class
   - uses existing classifiers and feature pipeline

3. Cost-sensitive nominal classifier:
   - keep RandomForest/LightGBM/XGBoost classification
   - evaluate with ordinal metrics
   - optionally choose predictions by minimizing expected ordinal distance from class probabilities

### C. Label Strategy Compatibility

Run ordinal models against existing label strategies:

- original `derive_stall_class`
- Phase 2 fixed thresholds
- percentile qcut
- quantile minimum-size labels
- hybrid middle qcut labels

This lets the next phase separate two questions:

- Are labels easier after balancing?
- Are ordered models better even when nominal macro F1 is similar?

### D. Grouped Validation

Preserve existing methodology:

- use `GroupKFold(n_splits=5)`
- group by `session_label`
- apply any resampling only on training folds
- never touch validation folds with SMOTE or label balancing

## Required Files

New files:

- `src/ml/evaluation/ordinal_metrics.py`
- `src/ml/models/ordinal.py`
- `src/ml/notebooks/03_ordinal_experiments.ipynb` if notebook evidence is required later

Optional files:

- `src/ml/data/ordinal_labels.py`
- `src/ml/artifacts/ordinal_model_summary.md`

Existing files to import from, not rewrite:

- `src/ml/data/labels.py`
- `src/ml/data/cleaning.py`
- `src/ml/features/engineering.py`
- `src/ml/features/selection.py`
- `src/ml/evaluation/metrics.py`
- `src/ml/models/phase1_hybrid_labels.py`
- `src/ml/models/phase2_imbalance.py`

## Expected Artifacts

Recommended artifacts for the ordinal phase:

- `src/ml/artifacts/ordinal_comparison.csv`
  - model
  - label_strategy
  - fold
  - macro precision
  - macro recall
  - macro F1
  - balanced accuracy
  - MCC
  - mean absolute class error
  - mean squared class error
  - adjacent accuracy
  - severe error rate
  - quadratic weighted kappa

- `src/ml/artifacts/ordinal_summary.csv`
  - mean metrics aggregated by model and label strategy

- `src/ml/artifacts/ordinal_confusion_matrix.csv`

- `src/ml/artifacts/ordinal_confusion_matrix.png`

- `src/ml/artifacts/ordinal_error_distribution.png`
  - distribution of `y_pred - y_true`

- `src/ml/artifacts/ordinal_recommendation.md`
  - recommendation based on both nominal and ordinal metrics

## Implementation Order

1. Add ordinal metrics only.
   - Implement `ordinal_metrics(y_true, y_pred)`.
   - Unit/smoke check on small fixed arrays.
   - Do not train models yet.

2. Add regression-as-ordinal baseline.
   - Use existing leakage-free `numeric_feature_columns`.
   - Use existing cleaning and feature engineering.
   - Use `GroupKFold(session_label)`.
   - Save fold-level metrics only.

3. Add cumulative binary classifier baseline.
   - Train three threshold classifiers.
   - Combine threshold probabilities into ordered class predictions.
   - Compare against regression-as-ordinal.

4. Add probability post-processing for existing classifiers.
   - Use predicted class probabilities.
   - Select class minimizing expected absolute or squared ordinal distance.
   - Compare with normal `argmax`.

5. Compare label strategies.
   - Start with original `derive_stall_class`.
   - Then reuse qcut and hybrid label builders.
   - Avoid modifying Phase 1 label behavior.

6. Generate artifacts.
   - `ordinal_comparison.csv`
   - `ordinal_summary.csv`
   - confusion matrix artifacts
   - error distribution plot
   - recommendation markdown

7. Add an optional notebook after code execution succeeds.
   - Keep it separate as `03_ordinal_experiments.ipynb`.
   - Do not edit existing notebooks.

## Minimal-Disruption Recommendation

Start with `src/ml/evaluation/ordinal_metrics.py` and `src/ml/models/ordinal.py`.

This keeps ordinal work as an experiment layer beside the current nominal classification pipeline. It avoids changing Phase 1 artifacts, preserves existing leakage checks, reuses GroupKFold, and lets ordinal metrics reveal whether the model is making mostly adjacent severity mistakes or truly severe class jumps.
