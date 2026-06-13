from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_trading.models.ml import ML_FEATURE_COLUMNS
from agent_trading.features.technical import add_technical_features


@dataclass(frozen=True)
class DeepLearningPrediction:
    probability_up: float
    model_name: str
    train_samples: int
    validation_accuracy: float | None
    enabled: bool
    message: str


class GRUSequencePredictor:
    """Optional PyTorch GRU predictor for sequence-based forecasting."""

    def __init__(self, horizon_days: int = 5, lookback: int = 60, epochs: int = 12) -> None:
        self.horizon_days = horizon_days
        self.lookback = lookback
        self.epochs = epochs

    def predict(self, history: pd.DataFrame) -> DeepLearningPrediction:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            return DeepLearningPrediction(
                probability_up=0.5,
                model_name="GRU",
                train_samples=0,
                validation_accuracy=None,
                enabled=False,
                message="未安装 PyTorch，深度学习模型未启用。可执行：python -m pip install torch",
            )

        x, y = self._sequence_dataset(history)
        if len(x) < 180 or len(np.unique(y)) < 2:
            return DeepLearningPrediction(
                probability_up=0.5,
                model_name="GRU",
                train_samples=len(x),
                validation_accuracy=None,
                enabled=False,
                message="可用序列样本不足，深度学习模型未启用。",
            )

        split = int(len(x) * 0.8)
        x_train, x_valid = x[:split], x[split:]
        y_train, y_valid = y[:split], y[split:]
        mean = x_train.mean(axis=(0, 1), keepdims=True)
        std = x_train.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1
        x_train = (x_train - mean) / std
        x_valid = (x_valid - mean) / std

        torch.manual_seed(42)
        model = _GRUClassifier(input_size=x_train.shape[-1], hidden_size=24)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        loss_fn = nn.BCELoss()
        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            ),
            batch_size=64,
            shuffle=False,
        )
        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                probability = model(batch_x)
                loss = loss_fn(probability, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_probability = model(torch.tensor(x_valid, dtype=torch.float32)).numpy().reshape(-1)
            valid_prediction = (valid_probability >= 0.5).astype(int)
            validation_accuracy = float((valid_prediction == y_valid).mean())
            latest = ((x[-1:] - mean) / std).astype(np.float32)
            probability_up = float(model(torch.tensor(latest, dtype=torch.float32)).item())

        return DeepLearningPrediction(
            probability_up=probability_up,
            model_name="GRU",
            train_samples=int(len(x_train)),
            validation_accuracy=validation_accuracy,
            enabled=True,
            message="GRU 序列模型已基于历史因子序列完成快速训练。",
        )

    def _sequence_dataset(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = add_technical_features(history)
        forward_return = features["close"].shift(-self.horizon_days) / features["close"] - 1
        data = features.loc[:, ML_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median()).fillna(0)
        values = data.to_numpy(dtype=np.float32)
        labels = (forward_return > 0).astype(int).to_numpy()
        valid_count = len(values) - self.horizon_days
        x_items = []
        y_items = []
        for index in range(self.lookback, valid_count):
            x_items.append(values[index - self.lookback : index])
            y_items.append(labels[index])
        return np.asarray(x_items, dtype=np.float32), np.asarray(y_items, dtype=np.int64)


class _GRUClassifier:
    def __new__(cls, input_size: int, hidden_size: int):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, hidden = self.gru(x)
                return self.head(hidden[-1])

        return Model()
