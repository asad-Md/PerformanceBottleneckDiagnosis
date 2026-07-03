# Phase 4 Feature Analysis Summary

- Model analyzed: `lightgbm`
- Parameters: `{'num_leaves': 15, 'n_estimators': 300, 'min_child_samples': 20, 'max_depth': -1, 'learning_rate': 0.03}`
- Best subset: `top_30_features` with Macro F1 `0.5163`
- Smallest subset within 1% of best Macro F1: `top_15_features` (15 features)
- SHAP: skipped: shap is not installed

## Top 20 Features

| feature | tree_importance | permutation_importance_mean | permutation_importance_std | tree_importance_norm | permutation_importance_mean_norm | combined_importance |
| --- | --- | --- | --- | --- | --- | --- |
| involuntary_switches | 996 | 0.0793 | 0.0019 | 0.7202 | 1.0000 | 1.7202 |
| migration_ratio | 1383 | 0.0288 | 0.0014 | 1.0000 | 0.3638 | 1.3638 |
| runtime_per_switch | 1326 | 0.0131 | 0.0003 | 0.9588 | 0.1653 | 1.1241 |
| voluntary_ratio | 904 | 0.0308 | 0.0018 | 0.6537 | 0.3891 | 1.0427 |
| cpu_migrations | 1040 | 0.0073 | 0.0006 | 0.7520 | 0.0920 | 0.8440 |
| switches_per_runtime | 816 | 0.0140 | 0.0005 | 0.5900 | 0.1763 | 0.7663 |
| involuntary_ratio | 715 | 0.0128 | 0.0008 | 0.5170 | 0.1621 | 0.6790 |
| voluntary_switches | 694 | 0.0129 | 0.0006 | 0.5018 | 0.1625 | 0.6644 |
| avg_mutex_wait_ns | 650 | 0.0129 | 0.0009 | 0.4700 | 0.1633 | 0.6333 |
| avg_futex_latency_ns | 649 | 0.0095 | 0.0006 | 0.4693 | 0.1195 | 0.5888 |
| latency_count | 546 | 0.0151 | 0.0004 | 0.3948 | 0.1904 | 0.5852 |
| total_runtime_ns | 749 | 0.0023 | 0.0007 | 0.5416 | 0.0287 | 0.5702 |
| ctx_switches | 482 | 0.0075 | 0.0005 | 0.3485 | 0.0944 | 0.4429 |
| avg_rwsem_read_wait_ns | 452 | 0.0008 | 0.0004 | 0.3268 | 0.0098 | 0.3366 |
| futex_ratio | 314 | 0.0067 | 0.0007 | 0.2270 | 0.0851 | 0.3121 |
| rwsem_read_contentions | 358 | 0.0034 | 0.0005 | 0.2589 | 0.0431 | 0.3020 |
| max_syscall_latency_ns | 370 | 0.0015 | 0.0004 | 0.2675 | 0.0186 | 0.2861 |
| mutex_contentions | 286 | 0.0055 | 0.0004 | 0.2068 | 0.0695 | 0.2763 |
| kfree_count | 331 | 0.0019 | 0.0006 | 0.2393 | 0.0237 | 0.2630 |
| avg_syscall_latency_ns | 284 | 0.0020 | 0.0002 | 0.2054 | 0.0253 | 0.2307 |

## Subset Metrics

| experiment | model | n_features | fold | accuracy | macro_precision | macro_recall | macro_f1 | balanced_accuracy | mcc | mean_absolute_error | mean_squared_error | average_severity_distance | adjacent_accuracy | severe_error_rate | quadratic_weighted_kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_10_features | lightgbm | 10 | 3.0000 | 0.5275 | 0.5143 | 0.5165 | 0.5097 | 0.5165 | 0.3574 | 0.7331 | 1.3990 | 0.7331 | 0.8119 | 0.1881 | 0.4459 |
| top_15_features | lightgbm | 15 | 3.0000 | 0.5310 | 0.5182 | 0.5204 | 0.5135 | 0.5204 | 0.3621 | 0.7247 | 1.3781 | 0.7247 | 0.8153 | 0.1847 | 0.4537 |
| top_20_features | lightgbm | 20 | 3.0000 | 0.5327 | 0.5187 | 0.5218 | 0.5150 | 0.5218 | 0.3639 | 0.7188 | 1.3620 | 0.7188 | 0.8187 | 0.1813 | 0.4574 |
| top_30_features | lightgbm | 30 | 3.0000 | 0.5337 | 0.5197 | 0.5236 | 0.5163 | 0.5236 | 0.3657 | 0.7165 | 1.3553 | 0.7165 | 0.8189 | 0.1811 | 0.4602 |
| top_all_features | lightgbm | 59 | 3.0000 | 0.5329 | 0.5189 | 0.5228 | 0.5155 | 0.5228 | 0.3648 | 0.7175 | 1.3571 | 0.7175 | 0.8189 | 0.1811 | 0.4600 |