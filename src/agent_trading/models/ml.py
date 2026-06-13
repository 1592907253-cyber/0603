from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agent_trading.features.technical import add_technical_features


ML_FEATURE_COLUMNS = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma_gap_5_20",
    "ma_gap_10_20",
    "ma_gap_20_60",
    "price_ma20_gap",
    "ma_alignment_score",
    "volatility_20",
    "volume_ratio_5_20",
    "drawdown_20",
    "momentum_score",
)


@dataclass(frozen=True)
class MLPrediction:
    probability_up: float
    model_name: str
    train_samples: int
    test_accuracy: float | None
    test_auc: float | None
    top_features: list[tuple[str, float]]


class MLDirectionPredictor:
    """Train a leakage-aware baseline ML model on one symbol's history.

    The implementation prefers LightGBM/CatBoost if installed, and falls back to
    sklearn models so the project remains runnable in a hackathon environment.
    """

    def __init__(self, horizon_days: int = 5) -> None:
        self.horizon_days = horizon_days

    def predict(self, history: pd.DataFrame) -> MLPrediction | None:
        x, y = self._dataset(history)
        if len(x) < 300 or y.nunique() < 2:
            return None

        split = int(len(x) * 0.78)
        x_train, x_test = x.iloc[:split], x.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None

        model, model_name = self._build_model()
        model.fit(x_train, y_train)
        latest = x.iloc[[-1]]
        probability = float(model.predict_proba(latest)[0, 1])

        test_probability = model.predict_proba(x_test)[:, 1]
        test_prediction = (test_probability >= 0.5).astype(int)
        accuracy = float(accuracy_score(y_test, test_prediction))
        auc = float(roc_auc_score(y_test, test_probability))
        top_features = self._feature_importance(model, x_train.columns)

        return MLPrediction(
            probability_up=probability,
            model_name=model_name,
            train_samples=int(len(x_train)),
            test_accuracy=accuracy,
            test_auc=auc,
            top_features=top_features,
        )

    def _dataset(self, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = add_technical_features(history)
        forward_return = features["close"].shift(-self.horizon_days) / features["close"] - 1
        label = (forward_return > 0).astype(int)
        dataset = features.loc[:, ML_FEATURE_COLUMNS].copy()
        valid = forward_return.notna()
        return dataset.loc[valid], label.loc[valid]

    def _build_model(self) -> tuple[object, str]:
        try:
            from lightgbm import LGBMClassifier

            return (
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=220,
                                learning_rate=0.035,
                                max_depth=4,
                                num_leaves=15,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "LightGBM",
            )
        except Exception:
            pass

        try:
            from catboost import CatBoostClassifier

            return (
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            CatBoostClassifier(
                                iterations=220,
                                depth=4,
                                learning_rate=0.035,
                                loss_function="Logloss",
                                verbose=False,
                                random_seed=42,
                            ),
                        ),
                    ]
                ),
                "CatBoost",
            )
        except Exception:
            pass

        return (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=180,
                            learning_rate=0.04,
                            max_leaf_nodes=15,
                            l2_regularization=0.05,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "HistGradientBoosting",
        )

    def _feature_importance(self, model: object, columns: pd.Index) -> list[tuple[str, float]]:
        estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            # HistGradientBoosting has no built-in importance. Use logistic coefficients as a cheap proxy.
            proxy = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500)),
                ]
            )
            return [(column, 0.0) for column in columns[:5]]
        total = float(np.sum(np.abs(values))) or 1.0
        pairs = sorted(
            zip(columns.tolist(), (np.abs(values) / total).tolist(), strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(name, float(value)) for name, value in pairs[:6]]
