from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    roc_auc: float | None
    n_samples: int


def build_baseline_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_time_series_classifier(x: pd.DataFrame, y: pd.Series) -> TrainingResult:
    splitter = TimeSeriesSplit(n_splits=3)
    predictions: list[float] = []
    probabilities: list[float] = []
    actuals: list[int] = []

    for train_idx, test_idx in splitter.split(x):
        model = build_baseline_classifier()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        proba = model.predict_proba(x.iloc[test_idx])[:, 1]
        predictions.extend(pred.tolist())
        probabilities.extend(proba.tolist())
        actuals.extend(y.iloc[test_idx].astype(int).tolist())

    accuracy = accuracy_score(actuals, predictions)
    roc_auc = None
    if len(set(actuals)) > 1:
        roc_auc = roc_auc_score(actuals, probabilities)
    return TrainingResult(accuracy=float(accuracy), roc_auc=roc_auc, n_samples=len(y))
