from dataclasses import dataclass

import pandas as pd

from agent_trading.features.technical import add_technical_features
from agent_trading.schemas import ExplanationItem, ForecastHorizon, MarketForecast, StockForecast


@dataclass(frozen=True)
class ForecastConfig:
    horizons: tuple[int, ...] = (1, 5, 20)


class HeuristicForecastModel:
    """Baseline model.

    This is intentionally simple and deterministic. Replace it with trained LightGBM/CatBoost
    models once historical A-share datasets and labels are prepared.
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    def market_forecast(self, symbol: str, history: pd.DataFrame) -> MarketForecast:
        features = add_technical_features(history)
        latest = features.iloc[-1]

        trend = float(latest["ma_gap_5_20"])
        momentum = float(latest["ret_20d"])
        volatility = float(latest["volatility_20"])

        if volatility > 0.025:
            regime = "high_volatility"
            position = 0.3
        elif trend > 0.01 and momentum > 0:
            regime = "bull"
            position = 0.7
        elif trend < -0.01 and momentum < 0:
            regime = "bear"
            position = 0.2
        else:
            regime = "neutral"
            position = 0.5

        forecasts = [
            self._horizon_forecast(horizon, momentum=momentum, volatility=volatility, trend=trend)
            for horizon in self.config.horizons
        ]
        explanations = [
            ExplanationItem(
                factor="trend",
                impact="positive" if trend > 0 else "negative",
                detail=f"MA5/MA20 gap is {trend:.2%}.",
            ),
            ExplanationItem(
                factor="momentum",
                impact="positive" if momentum > 0 else "negative",
                detail=f"20-day return is {momentum:.2%}.",
            ),
            ExplanationItem(
                factor="volatility",
                impact="negative" if volatility > 0.02 else "neutral",
                detail=f"20-day volatility is {volatility:.2%}.",
            ),
        ]
        risks = []
        if volatility > 0.02:
            risks.append("Market volatility is elevated; reduce position sizing.")
        if momentum < 0:
            risks.append("Medium-term momentum is weak.")

        return MarketForecast(
            symbol=symbol,
            regime=regime,
            suggested_position=position,
            forecasts=forecasts,
            explanations=explanations,
            risks=risks or ["No major baseline risk detected."],
        )

    def stock_forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
        benchmark_history: pd.DataFrame | None = None,
    ) -> StockForecast:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        momentum = float(latest["ret_20d"])
        trend = float(latest["ma_gap_5_20"])
        drawdown = float(latest["drawdown_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])

        benchmark_momentum = 0.0
        if benchmark_history is not None:
            benchmark_features = add_technical_features(benchmark_history)
            benchmark_momentum = float(benchmark_features.iloc[-1]["ret_20d"])

        excess_signal = momentum - benchmark_momentum
        outperform_probability = min(max(0.5 + excess_signal * 3 + trend * 2, 0.05), 0.95)
        drawdown_risk = min(max(abs(drawdown) * 4 + max(0.0, 0.9 - volume_ratio) * 0.2, 0.05), 0.95)

        if outperform_probability > 0.62 and drawdown_risk < 0.45:
            action = "buy"
        elif drawdown_risk > 0.65 or outperform_probability < 0.45:
            action = "avoid"
        else:
            action = "hold"

        forecasts = [
            self._horizon_forecast(horizon, momentum=momentum, volatility=0.015, trend=trend)
            for horizon in self.config.horizons
        ]
        explanations = [
            ExplanationItem(
                factor="relative_momentum",
                impact="positive" if excess_signal > 0 else "negative",
                detail=f"20-day excess return signal is {excess_signal:.2%}.",
            ),
            ExplanationItem(
                factor="trend",
                impact="positive" if trend > 0 else "negative",
                detail=f"MA5/MA20 gap is {trend:.2%}.",
            ),
            ExplanationItem(
                factor="liquidity",
                impact="positive" if volume_ratio > 1 else "neutral",
                detail=f"5/20 day volume ratio is {volume_ratio:.2f}.",
            ),
        ]
        risks = []
        if drawdown < -0.08:
            risks.append("Recent drawdown is deep; wait for stabilization.")
        if volume_ratio < 0.8:
            risks.append("Liquidity is weakening compared with the 20-day baseline.")

        return StockForecast(
            symbol=symbol,
            action=action,
            outperform_probability=outperform_probability,
            drawdown_risk=drawdown_risk,
            forecasts=forecasts,
            explanations=explanations,
            risks=risks or ["No major baseline risk detected."],
        )

    def _horizon_forecast(
        self,
        horizon: int,
        momentum: float,
        volatility: float,
        trend: float,
    ) -> ForecastHorizon:
        expected_return = momentum * min(horizon / 20, 1.0) + trend * 0.4
        if expected_return > volatility * 0.25:
            direction = "up"
        elif expected_return < -volatility * 0.25:
            direction = "down"
        else:
            direction = "flat"
        confidence = min(max(0.5 + abs(expected_return) * 4 - volatility * 2, 0.05), 0.9)
        return ForecastHorizon(
            horizon_days=horizon,
            direction=direction,
            expected_return=expected_return,
            confidence=confidence,
        )
