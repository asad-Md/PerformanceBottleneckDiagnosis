from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import RealtimeConfig


class BottleneckMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class BottleneckAttributor:
    THRESHOLD = 0.50

    def __init__(self, config: RealtimeConfig) -> None:
        self.model_path = Path(config.attribution_model_path)
        self.scaler_path = Path(config.attribution_scaler_path)
        self.feature_cols_path = Path(config.attribution_feature_cols_path)
        self.target_cols_path = Path(config.attribution_target_cols_path)

        for label, path in (
            ("model checkpoint", self.model_path),
            ("feature scaler", self.scaler_path),
            ("feature columns", self.feature_cols_path),
            ("target columns", self.target_cols_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing attribution {label}: {path}")

        self.feature_columns = self._load_columns(self.feature_cols_path, "feature")
        self.target_columns = self._load_columns(self.target_cols_path, "target")
        self.scaler = joblib.load(self.scaler_path)

        scaler_feature_count = getattr(self.scaler, "n_features_in_", len(self.feature_columns))
        if scaler_feature_count != len(self.feature_columns):
            raise ValueError(
                f"Attribution scaler expects {scaler_feature_count} features, but "
                f"{self.feature_cols_path} lists {len(self.feature_columns)}"
            )

        self.model = BottleneckMLP(len(self.feature_columns), len(self.target_columns))
        try:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "This PyTorch version does not support safe weights-only checkpoint loading; "
                "upgrade PyTorch before loading the attribution model."
            ) from exc
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _load_columns(path: Path, column_type: str) -> list[str]:
        try:
            columns: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read attribution {column_type} columns from {path}: {exc}") from exc
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(f"Attribution {column_type} columns in {path} must be unique non-empty strings")
        return columns

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        missing_columns = [column for column in self.feature_columns if column not in rows.columns]
        if missing_columns:
            raise RuntimeError(
                "Realtime rows are missing attribution model features: " + ", ".join(missing_columns)
            )

        feature_frame = rows.loc[:, self.feature_columns]
        try:
            feature_values = feature_frame.to_numpy(dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Realtime attribution features must be numeric") from exc
        if not np.isfinite(feature_values).all():
            raise RuntimeError("Realtime attribution features contain NaN or infinite values")

        scaled_values = self.scaler.transform(feature_frame)
        features = torch.as_tensor(np.asarray(scaled_values, dtype=np.float32))
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(features)).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Attribution MLP produced NaN or infinite probabilities")
        return probabilities

    def predict_labels(self, probabilities: np.ndarray) -> np.ndarray:
        return probabilities >= self.THRESHOLD
