"""Ordinal-aware metrics for ordered bottleneck severity classes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def ordinal_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    """Compute nominal and ordinal metrics for ordered severity labels."""

    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    distance = np.abs(pred - true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "mean_absolute_error": float(np.mean(distance)),
        "mean_squared_error": float(np.mean((pred - true) ** 2)),
        "average_severity_distance": float(np.mean(distance)),
        "adjacent_accuracy": float(np.mean(distance <= 1)),
        "severe_error_rate": float(np.mean(distance >= 2)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true, pred, weights="quadratic")),
    }

