from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_trading.features.technical import add_technical_features


FEATURE_COLUMNS = (
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
)


@dataclass(frozen=True)
class AnalogForecast:
    horizon_days: int
    expected_return: float
    up_probability: float
    sample_size: int
    median_return: float
    downside_probability: float


class HistoricalAnalogForecaster:
    """Calibrate a forecast from historically similar factor states.

    This is not a magic predictor. It makes the baseline more empirical by
    asking: when the market previously looked similar across trend, moving
    average, volatility, volume, and drawdown factors, what happened next?
    """

    def __init__(self, n_neighbors: int = 80, min_samples: int = 25) -> None:
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples

    def forecast(
        self,
        history: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 5, 20),
    ) -> dict[int, AnalogForecast]:
        features = add_technical_features(history)
        if len(features) < 120:
            return {}

        latest = features.iloc[-1]
        results: dict[int, AnalogForecast] = {}
        for horizon in horizons:
            dataset = features.copy()
            dataset[f"fwd_ret_{horizon}d"] = dataset["close"].shift(-horizon) / dataset["close"] - 1
            dataset = dataset.dropna(subset=[*FEATURE_COLUMNS, f"fwd_ret_{horizon}d"])
            # Do not compare against the most recent rows whose labels overlap current time.
            dataset = dataset.iloc[: -horizon] if len(dataset) > horizon else dataset.iloc[:0]
            if len(dataset) < self.min_samples:
                continue

            x = dataset.loc[:, FEATURE_COLUMNS].astype(float)
            current = latest.loc[list(FEATURE_COLUMNS)].astype(float)
            mean = x.mean()
            std = x.std().replace(0, 1).fillna(1)
            z = (x - mean) / std
            current_z = (current - mean) / std
            distance = np.sqrt(((z - current_z) ** 2).sum(axis=1))
            neighbors = dataset.loc[distance.nsmallest(self.n_neighbors).index]
            returns = neighbors[f"fwd_ret_{horizon}d"].astype(float)
            if len(returns) < self.min_samples:
                continue
            results[horizon] = AnalogForecast(
                horizon_days=horizon,
                expected_return=float(returns.mean()),
                up_probability=float((returns > 0).mean()),
                sample_size=int(len(returns)),
                median_return=float(returns.median()),
                downside_probability=float((returns < -0.03).mean()),
            )
        return results
