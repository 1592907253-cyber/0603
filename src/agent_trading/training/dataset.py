from dataclasses import dataclass

import pandas as pd

from agent_trading.features.technical import add_technical_features
from agent_trading.training.labels import make_excess_return_labels, make_forward_return_labels


@dataclass(frozen=True)
class DatasetSpec:
    target: str
    feature_columns: tuple[str, ...]


MARKET_DIRECTION_SPEC = DatasetSpec(
    target="up_5d",
    feature_columns=(
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ma_gap_5_20",
        "volatility_20",
        "volume_ratio_5_20",
        "drawdown_20",
        "momentum_score",
    ),
)

STOCK_OUTPERFORM_SPEC = DatasetSpec(
    target="outperform_5d",
    feature_columns=MARKET_DIRECTION_SPEC.feature_columns,
)


def build_market_direction_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = add_technical_features(history)
    labeled = make_forward_return_labels(features)
    dataset = labeled.dropna(subset=[MARKET_DIRECTION_SPEC.target])
    x = dataset.loc[:, MARKET_DIRECTION_SPEC.feature_columns]
    y = dataset.loc[:, MARKET_DIRECTION_SPEC.target]
    return x, y


def build_stock_outperform_dataset(
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    stock_features = add_technical_features(stock_history)
    benchmark_features = add_technical_features(benchmark_history)
    labeled = make_excess_return_labels(stock_features, benchmark_features)
    dataset = labeled.dropna(subset=[STOCK_OUTPERFORM_SPEC.target])
    x = dataset.loc[:, STOCK_OUTPERFORM_SPEC.feature_columns]
    y = dataset.loc[:, STOCK_OUTPERFORM_SPEC.target]
    return x, y
