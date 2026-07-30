# Ordinal Classification Summary

Labels: `strategy_c_quantile_min_size` only.

## Best Overall Model

- Model: `RandomForestClassifier`
- Approach: `nominal_argmax`
- Macro F1: `0.4922`
- Balanced accuracy: `0.4995`
- MCC: `0.3281`
- Mean absolute error: `0.7530`
- Adjacent accuracy: `0.8093`
- Severe error rate: `0.1907`
- Quadratic weighted kappa: `0.4152`

## Best Ordinal Approach

- Model: `CumulativeRandomForestClassifier`
- Approach: `cumulative_binary`
- Macro F1: `0.4893`
- Mean absolute error: `0.7803`
- Adjacent accuracy: `0.7940`
- Severe error rate: `0.2060`
- Quadratic weighted kappa: `0.4273`

## Comparison Against Nominal Classification

- Nominal baseline: `RandomForestClassifier` with `nominal_argmax`
- Nominal macro F1: `0.4922`
- Nominal mean absolute error: `0.7530`
- Nominal severe error rate: `0.1907`
- Lowest mean absolute error: `HistGradientBoostingRegressor` with `regression_as_ordinal` at `0.6972`

## Recommendations

- Use nominal argmax if Macro F1 is the primary decision metric for `strategy_c_quantile_min_size` labels.
- Use the lowest-error ordinal approach when adjacent severity mistakes are acceptable but large severity jumps are costly.
- Track macro F1 together with mean absolute error, adjacent accuracy, severe error rate, and quadratic weighted kappa.
- Keep Phase 1 and Phase 2 nominal experiments as baselines; this phase should remain a separate ordinal comparison layer.

## Mean Metrics

| Model | Approach | Macro F1 | MAE | Adjacent Accuracy | Severe Error Rate | QWK |
|---|---|---:|---:|---:|---:|---:|
| `RandomForestClassifier` | `nominal_argmax` | 0.4922 | 0.7530 | 0.8093 | 0.1907 | 0.4152 |
| `CumulativeRandomForestClassifier` | `cumulative_binary` | 0.4893 | 0.7803 | 0.7940 | 0.2060 | 0.4273 |
| `RandomForestClassifier` | `expected_ordinal_distance` | 0.4852 | 0.7242 | 0.8312 | 0.1688 | 0.4266 |
| `RandomForestRegressor` | `regression_as_ordinal` | 0.4391 | 0.7003 | 0.8638 | 0.1362 | 0.4480 |
| `HistGradientBoostingRegressor` | `regression_as_ordinal` | 0.3527 | 0.6972 | 0.8962 | 0.1038 | 0.4351 |