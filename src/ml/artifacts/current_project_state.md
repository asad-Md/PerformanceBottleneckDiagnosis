# Current Project State Report

## Overview
This document summarizes the current state of the `PerformanceBottleneckDiagnosis` project as of Phase 0. It is a read-only audit of repository structure, ML modules, notebooks, artifact evidence, findings from Phase 1 and Phase 2A, label strategy analysis, the best current pipeline, research gaps, and next priorities.

## Step 1 — Repository Inventory

### Repository structure
- Root contains: `README.md`, `architecture_diagrams/`, `eBPF_Programs/`, `report/`, and `src/`.
- ML work is concentrated under `src/ml/` with data cleaning, feature engineering, model training, evaluation, and artifacts.
- There are additional analysis notebooks and plans under `src/approach2/` and `src/approach3/`.

### `src/ml/` modules
- `src/ml/data/`: data cleaning and label derivation.
  - `cleaning.py`: filters rows by `pid > 0`, `ctx_switches >= 3`, `latency_count >= 3`, and `total_runtime_ns > 0`; writes `cleaning_report.json`.
  - `labels.py`: derives the 4-class ML target from `avg_stall_ns` using fixed latency thresholds.
- `src/ml/features/`: engineered feature generation and feature selection.
  - `engineering.py`: creates 22 engineered metrics from CPU, memory, syscall, and lock signals.
  - `selection.py`: centralizes leakage removal and numeric feature selection.
- `src/ml/models/`: model training and Phase 2 imbalance/label strategy experiments.
  - `train.py`: baseline RandomForest experiments, leakage comparison, GroupShuffleSplit, GroupKFold validation.
  - `phase2_imbalance.py`: label strategy definitions, SMOTE/BorderlineSMOTE experiments, class-weight comparisons, and model comparisons over RandomForest/LightGBM/XGBoost.
- `src/ml/evaluation/`: reusable evaluation metrics and confusion matrix helpers.
  - `metrics.py`: computes macro precision, macro recall, macro F1, balanced accuracy, MCC, and saves confusion matrix plots.
- `src/ml/artifacts/`: experiment outputs and summary tables.
- `src/ml/notebooks/`: two notebooks documenting pipeline foundation and artifact review.

### Existing notebooks
- `src/ml/notebooks/01_pipeline_foundation.ipynb`
- `src/ml/notebooks/02_artifact_review.ipynb`

### Existing artifacts
Key artifact files in `src/ml/artifacts/`:
- `cleaning_report.json`
- `feature_cols.json`
- `leakage_comparison.csv`
- `imbalance_comparison.csv`
- `class_weight_model_comparison.csv`
- `class_analysis.csv`
- `label_strategy_comparison.csv`
- `phase2a_summary.json`
- `phase2_summary.json`
- plots and confusion matrix outputs

### Available models
- Code defines `RandomForestClassifier` as the baseline model.
- Phase 2 code also supports optional `LightGBM` and `XGBoost`.
- No serialized model artifacts (`.pkl`, `.joblib`, or saved model files) are present under `src/ml/models/`.

### Evaluation pipeline
- Baseline evaluation uses grouped cross-validation to preserve `session_label` boundaries.
- `train_group_shuffle()` uses `GroupShuffleSplit` with 20% holdout.
- `groupkfold_validate()` uses `GroupKFold(n_splits=5)`.
- Metrics: macro precision, macro recall, macro F1, balanced accuracy, MCC.
- Leakage experiments compare three scenarios: with leakage, without leakage, and without `latency_count`.
- Phase 2 imbalance evaluation compares baseline, class-weight, SMOTE, class-weight+SMOTE, and BorderlineSMOTE.
- Label strategy comparisons evaluate fixed thresholds, percentile qcut, and quantile min-size strategies.

### Feature engineering pipeline
- Input source is `src/approach3/perf_metrics.csv` by default.
- Cleaning removes low-confidence rows and derives `bottleneck_class` from `avg_stall_ns`.
- `engineer_features()` computes derived ratios and pressures such as `runtime_per_switch`, `fault_rate`, `lock_pressure`, `io_cpu_ratio`, `memory_lock_ratio`, and others.
- `feature_cols.json` is written from the engineered feature list.
- Numeric feature selection excludes non-feature columns and optionally excludes leakage columns plus `latency_count`.

## Step 2 — Phase 1 Findings

### Original model approach
- The baseline model is a `RandomForestClassifier` with 300 trees and `class_weight="balanced"`.
- The baseline target label is derived from `avg_stall_ns` using fixed bins: 0-2ms, 2-10ms, 10-50ms, 50ms+.
- Features are numeric metrics derived from eBPF performance counters, excluding non-feature columns.

### Target leakage issue
- The target label is derived from `avg_stall_ns`.
- Leakage columns are explicitly defined in `src/ml/features/selection.py` as:
  - `avg_stall_ns`
  - `stall_ns`
  - `max_stall_ns`
- These columns must not be used as input features for models trained to predict `bottleneck_class`.

### Leakage comparison metrics
From `src/ml/artifacts/leakage_comparison.csv`:
- `with_leakage`: macro F1 = `1.0`, balanced accuracy = `1.0`, MCC = `1.0`
- `without_leakage`: macro F1 = `0.3367`, balanced accuracy = `0.5226`, MCC = `0.01664`
- `without_latency_count`: macro F1 = `0.3365`, balanced accuracy = `0.5225`, MCC = `0.01607`

### Conclusions
- Leakage is severe: `with_leakage` yields perfect scores, proving that `avg_stall_ns`, `stall_ns`, and/or `max_stall_ns` directly reveal the label.
- Removing leakage drops performance to realistic but low values, confirming the artifact removal logic is essential.
- `latency_count` has minimal effect when leakage is already removed.
- The dramatic improvement achieved by quantile-based labels suggests that label construction has a larger impact on performance than model choice or imbalance mitigation alone.

## Step 3 — Phase 2A Findings

### Class distributions
From `src/ml/artifacts/class_analysis.csv`:
- Raw data: class 0 = 352,314 rows (99.43%), class 1 = 1,657 rows (0.47%), class 2 = 267 rows (0.08%), class 3 = 106 rows (0.03%).
- Cleaned data: class 0 = 178,936 rows (99.47%), class 1 = 855 rows (0.48%), class 2 = 81 rows (0.045%), class 3 = 20 rows (0.011%).
- The cleaned dataset remains extremely imbalanced, with almost all samples in class 0 and only 20 examples in class 3.
- Per-session distributions confirm nearly every session is dominated by class 0.

### Imbalance severity
- The primary issue is extreme skew: class 0 dominates, while classes 1-3 are vanishingly rare after cleaning.
- This makes standard multiclass classification hard and increases the importance of sampling or weighted strategies.

### SMOTE experiments
From `src/ml/artifacts/imbalance_comparison.csv`:
- Baseline: macro F1 = `0.2871`, balanced accuracy = `0.2834`, MCC = `0.1058`.
- `smote`: macro F1 = `0.2869`, balanced accuracy = `0.3192`, MCC = `0.1364`.
- `borderline_smote`: macro F1 = `0.2988`, balanced accuracy = `0.3006`, MCC = `0.1260`.
- `class_weight_smote`: macro F1 = `0.2931`, balanced accuracy = `0.3293`, MCC = `0.1524`.

### Class weighting experiments
From `src/ml/artifacts/class_weight_model_comparison.csv`:
- RandomForest baseline: macro F1 = `0.2871`, balanced accuracy = `0.2834`, MCC = `0.1058`.
- RandomForest class_weight: macro F1 = `0.2934`, balanced accuracy = `0.3101`, MCC = `0.1519`.
- LightGBM baseline: macro F1 = `0.2873`, balanced accuracy = `0.3035`, MCC = `0.1058`.
- LightGBM class_weight: macro F1 = `0.2555`, balanced accuracy = `0.4080`, MCC = `0.1229`.
- XGBoost baseline: macro F1 = `0.2947`, balanced accuracy = `0.2909`, MCC = `0.1441`.
- XGBoost class_weight: macro F1 = `0.2467`, balanced accuracy = `0.5129`, MCC = `0.0836`.

### Borderline SMOTE results
- Borderline SMOTE is the best imbalance-only strategy by the available metric set.
- It yielded the highest macro F1 among resampling strategies: `0.2988`.
- It also improved MCC compared to baseline and provided a moderate gain over basic SMOTE.

### Model comparison results
- The highest macro F1 among models evaluated in `class_weight_model_comparison.csv` is XGBoost baseline at `0.2947`.
- RandomForest with `class_weight=True` is close with macro F1 = `0.2934`.
- LightGBM baseline and weighted variants do not exceed these two.
- The Phase 2A summary artifact reports `best_class_weight_model` as `xgboost` and `best_imbalance_strategy` as `borderline_smote`.

## Step 4 — Label Strategy Findings

### Original threshold labels
- `strategy_a_thresholds` uses fixed stall latency bins: 0-2ms, 2-8ms, 8-25ms, 25ms+.
- Class counts after cleaning are highly imbalanced in this strategy.
- Performance with this strategy: macro F1 = `0.3178`, balanced accuracy = `0.3455`, MCC = `0.2681`.

### Qcut labels
- `strategy_b_percentile_qcut` uses percentile-based bucketing via `pd.qcut`.
- This produces balanced class counts of 25.0% per class.
- Performance with this strategy: macro F1 = `0.5040`, balanced accuracy = `0.5057`, MCC = `0.3386`.

### Quantile labels
- `strategy_c_quantile_min_size` uses quantile labeling with a minimum class size fallback.
- In the available artifact, it produces the same balanced class counts and identical performance as the qcut strategy.
- Performance: macro F1 = `0.5040`, balanced accuracy = `0.5057`, MCC = `0.3386`.

### Performance differences and improvement
Comparing fixed thresholds to balanced quantile strategies:
- Macro F1 improved by 58.6%.
- Balanced Accuracy improved by 46.4%.
- MCC improved by 26.3%.

### Best label strategy
- The best-performing label strategy is the balanced quantile approach (`strategy_b_percentile_qcut` / `strategy_c_quantile_min_size`), because it produces equal class sizes and much higher multiclass metrics.
- Between the two, `strategy_c_quantile_min_size` has a stronger practical interpretation, since it preserves ordered severity while guarding against tiny classes. The artifact shows it tied with percentile qcut on measured performance.

### Interpretability tradeoffs
- `strategy_a_thresholds` is most interpretable: classes map directly to meaningful latency ranges.
- `strategy_b_percentile_qcut` is less interpretable because boundaries depend on the dataset distribution.
- `strategy_c_quantile_min_size` is a middle ground: it is still dataset-relative, but it preserves ordered severity and avoids classes that are too small to learn from.

## Step 5 — Current Best Pipeline

### Best label strategy
- Best current label strategy: balanced quantile labeling.
- Recommended practical choice: `strategy_c_quantile_min_size` for robustness, or `strategy_b_percentile_qcut` if strict quartiles are desired.

### Best imbalance strategy
- Best current imbalance strategy: `borderline_smote`.
- This strategy outperforms baseline and standard SMOTE on macro F1 in current artifacts.

### Best-performing model
- The current best-performing model by Macro F1 is XGBoost baseline under the evaluated Phase 2A settings. However, the performance gains are smaller than those obtained through label reformulation.
- RandomForest with class weights is a strong runner-up and has a more stable default implementation path.

### Current recommended pipeline
1. eBPF metrics capture from `src/approach3/perf_metrics.csv` or equivalent collection.
2. Cleaning and filtering with `src/ml/data/cleaning.py`.
3. Feature engineering with `src/ml/features/engineering.py`.
4. Label generation from `avg_stall_ns` using a quantile-based strategy (`strategy_c_quantile_min_size`).
5. Imbalance mitigation using `BorderlineSMOTE`.
6. Model training using a tree-based classifier, with `xgboost` or `random_forest + class_weight` as the most promising candidates.

## Step 6 — Open Research Questions

### Remaining bottlenecks
- The original threshold-based label formulation induces extreme class imbalance and is currently the largest bottleneck.
- The dataset is dominated by class 0, with very few examples for classes 2 and 3.
- Current performance remains modest on the intended multiclass task.

### Weaknesses
- The current ML system is not designed as an ordinal classifier, despite latency severity being ordered.
- There is no explicit explainability or feature attribution pipeline in the current artifacts.
- The pipeline is only evaluated on group-based splits; there is no documented temporal or deployment-style split.
- Root-cause diagnosis is not yet separated from class prediction; the model predicts severity rather than specific causal signals.

### Risks
- Leakage is a real risk if `avg_stall_ns`, `stall_ns`, or `max_stall_ns` are accidentally included as features.
- The fixed-threshold label strategy may misalign with actual class frequency and leads to poor performance.
- Optional dependencies such as LightGBM and XGBoost may not be installed in all environments.
- Current metrics are still low enough that production use would be risky without stronger validation.

### Missing experiments
- Ordinal classification methods (ordinal regression, monotonic scoring, target order-aware loss).
- Explainability: SHAP, feature importance ranking, and root-cause signal attribution.
- Root-cause diagnosis: mapping model outputs back to system-level causes, not just stall severity.
- Anomaly detection: detect unusual sessions or metric patterns separately from class prediction.
- Forecasting: predict future stall severity or trend drift from session histories.
- Alternative imbalance approaches: under-sampling, hybrid sampling, class-specific oversampling, or cost-sensitive multiclass loss.

## Step 7 — Recommendations

### Priority 1
- Stabilize label strategy and class definition.
- Implement `strategy_c_quantile_min_size` with clear documentation and ensure it is used consistently in training and evaluation.
- Add an ordinal-aware evaluation axis so severity order is captured beyond flat multiclass metrics.

### Priority 2
- Add explainability and model diagnostics.
- Introduce SHAP or similar feature-attribution reporting for the chosen model.
- Use session-level validation and root-cause signal correlation to verify predictions.

### Priority 3
- Improve imbalance handling and robust sampling.
- Validate `borderline_smote` more thoroughly and compare it with additional resampling or cost-sensitive methods.
- Explore anomaly detection and session-level forecasting as complementary workflows.

### Priority 4
- Harden the pipeline for deployment readiness.
- Ensure optional model dependencies are explicitly managed.
- Add artifact generation for model explainability, calibration, and performance monitoring.

## Key Research Findings

1. Target leakage can produce deceptively perfect performance.
2. Removing leakage reduces performance to realistic levels.
3. The original latency thresholds induce extreme class imbalance.
4. Imbalance mitigation alone provides modest gains.
5. Reformulating labels using quantile strategies produces the largest performance improvement observed so far.

## Conclusion
The project currently demonstrates three major findings:

1. Target leakage was responsible for the original perfect performance.
2. Fixed latency thresholds induce severe class imbalance that significantly limits model performance.
3. Reformulating bottleneck severity labels using balanced quantile strategies provides substantially larger improvements than model selection or imbalance mitigation alone.

The next phase of the project should therefore focus on:
- interpretable hybrid label strategies,
- ordinal classification methods,
- explainability and root-cause diagnosis.
