# Hybrid Label Strategy Recommendation

This report compares four label derivation methods for bottleneck severity classification.

## Best strategy

- **Recommended strategy:** `strategy_b_percentile_qcut`
- **Description:** Four equal-frequency percentile buckets using qcut
- **Interpretability:** medium: balanced class sizes, but thresholds depend on dataset distribution

## Summary metrics

| Strategy | Macro F1 | Balanced accuracy | MCC |
|---|---|---|---|
| `strategy_b_percentile_qcut` | 0.4945 | 0.5021 | 0.3316 |
| `strategy_c_quantile_min_size` | 0.4945 | 0.5021 | 0.3316 |
| `strategy_a_thresholds` | 0.3035 | 0.3186 | 0.1553 |
| `strategy_d_hybrid_mid_qcut` | 0.2940 | 0.3188 | 0.1214 |

## Notes

- `strategy_a_thresholds` preserves the original fixed latency bins used by the baseline pipeline.
- `strategy_b_percentile_qcut` enforces balanced class sizes at the cost of dataset-relative thresholds.
- `strategy_c_quantile_min_size` protects against tiny classes by falling back to fewer bins.
- `strategy_d_hybrid_mid_qcut` keeps intuitive normal/high boundaries while balancing the middle severity range.