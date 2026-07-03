# Phase 4 Hyperparameter Tuning Summary

- Best model: `random_forest`
- Macro F1: `0.5366`
- Balanced accuracy: `0.5433`
- MCC: `0.3903`
- N features: `59`
- Tuning rows: `10007`
- Full evaluation rows: `179892`

## Tuned Parameters

- `random_forest`: `{'n_estimators': 800, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 20}`
- `xgboost`: `{'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 100, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.6}`
- `lightgbm`: `{'num_leaves': 15, 'n_estimators': 300, 'min_child_samples': 20, 'max_depth': -1, 'learning_rate': 0.03}`

## Mean Metrics

| experiment | model | n_features | fold | accuracy | macro_precision | macro_recall | macro_f1 | balanced_accuracy | mcc | mean_absolute_error | mean_squared_error | average_severity_distance | adjacent_accuracy | severe_error_rate | quadratic_weighted_kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hyperparameter_tuning | lightgbm | 59 | 3.0000 | 0.5329 | 0.5189 | 0.5228 | 0.5155 | 0.5228 | 0.3648 | 0.7175 | 1.3571 | 0.7175 | 0.8189 | 0.1811 | 0.4600 |
| hyperparameter_tuning | random_forest | 59 | 3.0000 | 0.5521 | 0.5386 | 0.5433 | 0.5366 | 0.5433 | 0.3903 | 0.6822 | 1.2800 | 0.6822 | 0.8302 | 0.1698 | 0.4837 |
| hyperparameter_tuning | xgboost | 59 | 3.0000 | 0.5466 | 0.5362 | 0.5437 | 0.5336 | 0.5437 | 0.3878 | 0.6950 | 1.3101 | 0.6950 | 0.8242 | 0.1758 | 0.4780 |