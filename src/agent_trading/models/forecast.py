from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from agent_trading.agents.sentiment import SentimentAgent, SentimentResult
from agent_trading.features.technical import add_technical_features
from agent_trading.models.analog import HistoricalAnalogForecaster
from agent_trading.models.deep_learning import GRUSequencePredictor
from agent_trading.models.ml import MLDirectionPredictor
from agent_trading.models.signal_engine import MultiFactorSignalEngine
from agent_trading.schemas import (
    AnalysisSection,
    FactorContribution,
    ExplanationItem,
    ForecastHorizon,
    IndicatorPoint,
    KlinePoint,
    MarketForecast,
    ModelConsensus,
    ModelDiagnostic,
    PredictionChart,
    StockForecast,
    TradeSignalPoint,
    ValidationMetric,
)


SYMBOL_NAMES = {
    "000300.SH": "沪深300",
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "600519.SH": "贵州茅台",
    "000001.SZ": "平安银行",
    "300750.SZ": "宁德时代",
    "601318.SH": "中国平安",
    "600036.SH": "招商银行",
    "600030.SH": "中信证券",
}


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
        self.signal_engine = MultiFactorSignalEngine()
        self.analog_forecaster = HistoricalAnalogForecaster()
        self.ml_predictor = MLDirectionPredictor(horizon_days=5)
        self.deep_predictor = GRUSequencePredictor(horizon_days=5)

    def market_forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
        sentiment: SentimentResult | None = None,
    ) -> MarketForecast:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        signal = self.signal_engine.market_score(history, sentiment)
        analogs = self.analog_forecaster.forecast(history, self.config.horizons)

        trend = float(latest["ma_gap_5_20"])
        momentum = float(latest["ret_20d"])
        volatility = float(latest["volatility_20"])

        if volatility > 0.025:
            regime = "high_volatility"
            position = 0.3
        elif signal.score > 0.22:
            regime = "bull"
            position = min(max(0.55 + signal.score * 0.35, 0.55), 0.8)
        elif signal.score < -0.18:
            regime = "bear"
            position = max(0.15, 0.35 + signal.score * 0.35)
        else:
            regime = "neutral"
            position = 0.5

        forecasts = [
            self._horizon_forecast(
                horizon,
                momentum=momentum + signal.score * 0.06,
                volatility=volatility,
                trend=trend,
                analog=analogs.get(horizon),
            )
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
        sentiment: SentimentResult | None = None,
    ) -> StockForecast:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        signal = self.signal_engine.stock_score(history, benchmark_history, sentiment)
        analogs = self.analog_forecaster.forecast(history, self.config.horizons)
        momentum = float(latest["ret_20d"])
        trend = float(latest["ma_gap_5_20"])
        drawdown = float(latest["drawdown_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])

        benchmark_momentum = 0.0
        if benchmark_history is not None:
            benchmark_features = add_technical_features(benchmark_history)
            benchmark_momentum = float(benchmark_features.iloc[-1]["ret_20d"])

        excess_signal = momentum - benchmark_momentum
        analog_5d = analogs.get(5)
        analog_probability = analog_5d.up_probability if analog_5d else 0.5
        outperform_probability = min(
            max(0.5 + signal.score * 0.32 + excess_signal * 0.9 + (analog_probability - 0.5) * 0.35, 0.05),
            0.95,
        )
        drawdown_risk = min(max(abs(drawdown) * 4 + max(0.0, 0.9 - volume_ratio) * 0.2, 0.05), 0.95)

        if outperform_probability > 0.62 and drawdown_risk < 0.45:
            action = "buy"
        elif drawdown_risk > 0.65 or outperform_probability < 0.45:
            action = "avoid"
        else:
            action = "hold"

        forecasts = [
            self._horizon_forecast(
                horizon,
                momentum=momentum + signal.score * 0.04,
                volatility=0.015,
                trend=trend,
                analog=analogs.get(horizon),
            )
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

    def prediction_chart(
        self,
        symbol: str,
        history: pd.DataFrame,
        benchmark_history: pd.DataFrame | None = None,
        future_days: int = 12,
        provider_name: str = "mock",
        model_mode: str = "ensemble",
    ) -> PredictionChart:
        is_index = symbol.endswith(".SH") and symbol.startswith(("000", "399"))
        name = SYMBOL_NAMES.get(symbol, symbol)
        history_tail = history.tail(1250)
        sentiment = SentimentAgent().analyze(symbol, provider_name=provider_name)
        forecast = (
            self.market_forecast(symbol, history, sentiment)
            if is_index
            else self.stock_forecast(symbol, history, benchmark_history, sentiment)
        )
        validation_metrics = self._validation_metrics(history)
        model_diagnostics = self._model_diagnostics(history)
        model_diagnostics.extend(self._deep_diagnostics(history))
        model_consensus = self._model_consensus(validation_metrics, model_diagnostics)
        adjustment = self._model_adjustment(model_diagnostics, model_mode)
        if abs(adjustment) > 0:
            forecast.forecasts[1].expected_return = forecast.forecasts[1].expected_return + adjustment
        future = self._project_future_klines(history_tail, forecast.forecasts, future_days)
        indicators = self._indicator_points(history_tail, future)
        trade_signals = self._trade_signal_points(history_tail, future)

        if isinstance(forecast, MarketForecast):
            analysis = self._market_analysis_cn(name, forecast)
            trend_summary = f"{name}当前处于{self._regime_cn(forecast.regime)}，建议仓位约{forecast.suggested_position:.0%}。"
            risk_summary = "；".join(forecast.risks)
            analysis_sections = self._market_sections(name, history_tail, forecast)
            strategy_description = self._market_strategy_description()
            factor_contributions = self.signal_engine.market_score(history_tail, sentiment).contributions
            explanations = forecast.explanations
        else:
            analysis = self._stock_analysis_cn(name, forecast)
            trend_summary = (
                f"{name}未来跑赢基准概率约{forecast.outperform_probability:.0%}，"
                f"模型建议为{self._action_cn(forecast.action)}。"
            )
            risk_summary = f"未来回撤风险约{forecast.drawdown_risk:.0%}。" + "；".join(forecast.risks)
            analysis_sections = self._stock_sections(name, history_tail, forecast, benchmark_history)
            strategy_description = self._stock_strategy_description()
            factor_contributions = self.signal_engine.stock_score(
                history_tail,
                benchmark_history,
                sentiment,
            ).contributions
            explanations = forecast.explanations

        return PredictionChart(
            symbol=symbol,
            name=name,
            kind="index" if is_index else "stock",
            analysis=analysis,
            trend_summary=trend_summary,
            risk_summary=risk_summary,
            analysis_sections=analysis_sections,
            strategy_description=strategy_description,
            factor_contributions=factor_contributions,
            validation_metrics=validation_metrics,
            model_diagnostics=model_diagnostics,
            model_consensus=model_consensus,
            selected_model_mode=model_mode,
            history=self._to_kline_points(history_tail, predicted=False),
            future=future,
            indicators=indicators,
            trade_signals=trade_signals,
            explanations=explanations,
        )

    def _horizon_forecast(
        self,
        horizon: int,
        momentum: float,
        volatility: float,
        trend: float,
        analog: object | None = None,
    ) -> ForecastHorizon:
        factor_return = momentum * min(horizon / 20, 1.0) + trend * 0.4
        analog_return = getattr(analog, "expected_return", None)
        expected_return = (
            factor_return * 0.55 + float(analog_return) * 0.45
            if analog_return is not None
            else factor_return
        )
        if expected_return > volatility * 0.25:
            direction = "up"
        elif expected_return < -volatility * 0.25:
            direction = "down"
        else:
            direction = "flat"
        analog_probability = getattr(analog, "up_probability", None)
        analog_sample = getattr(analog, "sample_size", 0)
        base_confidence = 0.5 + abs(expected_return) * 4 - volatility * 2
        if analog_probability is not None:
            base_confidence = base_confidence * 0.55 + abs(float(analog_probability) - 0.5) * 1.6 * 0.45
            base_confidence += min(float(analog_sample) / 200, 0.12)
        confidence = min(max(base_confidence, 0.05), 0.9)
        return ForecastHorizon(
            horizon_days=horizon,
            direction=direction,
            expected_return=expected_return,
            confidence=confidence,
        )

    def _project_future_klines(
        self,
        history: pd.DataFrame,
        forecasts: list[ForecastHorizon],
        future_days: int,
    ) -> list[KlinePoint]:
        last = history.iloc[-1]
        last_date = history.index[-1]
        last_close = float(last["close"])
        target_return = next(
            (item.expected_return for item in forecasts if item.horizon_days == 20),
            forecasts[-1].expected_return,
        )
        daily_drift = target_return / max(future_days, 1)
        volatility = float(history["close"].pct_change().tail(20).std() or 0.012)
        rng = np.random.default_rng(int(last_close * 1000) % (2**32))

        points: list[KlinePoint] = []
        prev_close = last_close
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=future_days)
        for current_date in future_dates:
            noise = rng.normal(0, volatility * 0.35)
            close = prev_close * (1 + daily_drift + noise)
            open_ = prev_close * (1 + rng.normal(0, volatility * 0.18))
            high = max(open_, close) * (1 + abs(rng.normal(volatility * 0.35, volatility * 0.12)))
            low = min(open_, close) * (1 - abs(rng.normal(volatility * 0.35, volatility * 0.12)))
            volume = float(last["volume"]) * (1 + rng.normal(0, 0.08))
            points.append(
                KlinePoint(
                    date=current_date.strftime("%Y-%m-%d"),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=max(float(volume), 0.0),
                    predicted=True,
                )
            )
            prev_close = close

        return points

    def _to_kline_points(self, frame: pd.DataFrame, predicted: bool) -> list[KlinePoint]:
        points = []
        for date_index, row in frame.iterrows():
            points.append(
                KlinePoint(
                    date=date_index.strftime("%Y-%m-%d"),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    predicted=predicted,
                )
            )
        return points

    def _indicator_points(
        self,
        history: pd.DataFrame,
        future: list[KlinePoint],
    ) -> list[IndicatorPoint]:
        future_frame = pd.DataFrame(
            {
                "open": [item.open for item in future],
                "high": [item.high for item in future],
                "low": [item.low for item in future],
                "close": [item.close for item in future],
                "volume": [item.volume for item in future],
                "amount": [item.volume * item.close for item in future],
            },
            index=pd.to_datetime([item.date for item in future]),
        )
        combined = pd.concat([history, future_frame]).sort_index()
        data = combined.copy()
        data["ma5"] = data["close"].rolling(5).mean()
        data["ma10"] = data["close"].rolling(10).mean()
        data["ma20"] = data["close"].rolling(20).mean()
        data["ma60"] = data["close"].rolling(60).mean()
        data["ma120"] = data["close"].rolling(120).mean()
        data["ma250"] = data["close"].rolling(250).mean()

        points: list[IndicatorPoint] = []
        for date_index, row in data.iterrows():
            points.append(
                IndicatorPoint(
                    date=date_index.strftime("%Y-%m-%d"),
                    ma5=self._nullable_float(row["ma5"]),
                    ma10=self._nullable_float(row["ma10"]),
                    ma20=self._nullable_float(row["ma20"]),
                    ma60=self._nullable_float(row["ma60"]),
                    ma120=self._nullable_float(row["ma120"]),
                    ma250=self._nullable_float(row["ma250"]),
                )
            )
        return points

    def _trade_signal_points(
        self,
        history: pd.DataFrame,
        future: list[KlinePoint],
    ) -> list[TradeSignalPoint]:
        future_frame = pd.DataFrame(
            {
                "open": [item.open for item in future],
                "high": [item.high for item in future],
                "low": [item.low for item in future],
                "close": [item.close for item in future],
                "volume": [item.volume for item in future],
                "amount": [item.volume * item.close for item in future],
                "predicted": [item.predicted for item in future],
            },
            index=pd.to_datetime([item.date for item in future]),
        )
        history_frame = history.copy()
        history_frame["predicted"] = False
        combined = pd.concat([history_frame, future_frame]).sort_index()
        if len(combined) < 80:
            return []

        close = pd.to_numeric(combined["close"], errors="coerce")
        high = pd.to_numeric(combined["high"], errors="coerce")
        low = pd.to_numeric(combined["low"], errors="coerce")
        volume = pd.to_numeric(combined["volume"], errors="coerce").replace(0, pd.NA)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = dif - dea
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, pd.NA)
        rsi = 100 - 100 / (1 + gain / loss)
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
        atr_pct = atr / close
        volume_ratio = volume.rolling(5).mean() / volume.rolling(20).mean()
        price_volume_macd = macd_hist * (1 + (volume_ratio - 1).clip(-0.4, 0.8))
        price_volume_macd = price_volume_macd / (1 + (atr_pct * 8).clip(0, 1.2))
        donchian_high = high.rolling(20).max().shift(1)
        donchian_low = low.rolling(10).min().shift(1)
        close_high_60 = close.rolling(60).max().shift(1)
        ret_1d = close.pct_change()
        ret_5d = close.pct_change(5)
        ret_20d = close.pct_change(20)
        ma_gap = close / ema20 - 1
        trend_gap = ema20 / ema60 - 1
        swing_low = (
            (low.shift(2) < low.shift(4))
            & (low.shift(2) < low.shift(3))
            & (low.shift(2) < low.shift(1))
            & (low.shift(2) < low)
        )
        swing_high = (
            (high.shift(2) > high.shift(4))
            & (high.shift(2) > high.shift(3))
            & (high.shift(2) > high.shift(1))
            & (high.shift(2) > high)
        )

        def buy_setup(idx: int) -> tuple[bool, list[str], float]:
            if idx < 65 or pd.isna(close.iloc[idx]) or pd.isna(atr.iloc[idx]):
                return False, [], 0.0
            trend_ok = bool(close.iloc[idx] > ema20.iloc[idx] > ema60.iloc[idx])
            breakout = bool(close.iloc[idx] > donchian_high.iloc[idx])
            early_breakout = bool(close.iloc[idx] > close_high_60.iloc[idx] and volume_ratio.iloc[idx] > 1.05)
            pullback_turn = bool(
                swing_low.iloc[idx]
                and close.iloc[idx] > high.iloc[idx - 2]
                and close.iloc[idx] > ema20.iloc[idx]
            )
            momentum_ok = bool(
                price_volume_macd.iloc[idx] > price_volume_macd.iloc[idx - 1]
                and price_volume_macd.iloc[idx] > price_volume_macd.iloc[idx - 3]
            )
            rsi_value = float(rsi.iloc[idx]) if pd.notna(rsi.iloc[idx]) else 50.0
            rsi_ok = 38 <= rsi_value <= 72
            volatility_ok = bool(0.008 <= atr_pct.iloc[idx] <= 0.065) if pd.notna(atr_pct.iloc[idx]) else True
            volume_ok = bool(volume_ratio.iloc[idx] > 0.9) if pd.notna(volume_ratio.iloc[idx]) else False
            reasons: list[str] = []
            if breakout:
                reasons.append("突破20日唐奇安上轨")
            if early_breakout:
                reasons.append("创60日收盘新高且量能确认")
            if pullback_turn:
                reasons.append("上升趋势中分形低点确认后转强")
            if trend_ok:
                reasons.append("价格位于EMA20/EMA60上方")
            if momentum_ok:
                reasons.append("量价调整MACD改善")
            if rsi_ok:
                reasons.append(f"RSI {rsi_value:.0f}，处于可交易区间")
            score = (
                (0.28 if breakout else 0.0)
                + (0.22 if early_breakout else 0.0)
                + (0.2 if pullback_turn else 0.0)
                + (0.16 if trend_ok else 0.0)
                + (0.1 if momentum_ok else 0.0)
                + (0.04 if rsi_ok and volume_ok and volatility_ok else 0.0)
            )
            return score >= 0.5 and bool(breakout or early_breakout or pullback_turn), reasons, score

        def historical_success_rate(idx: int) -> tuple[float, int]:
            start = max(65, idx - 320)
            end = min(idx - 12, len(history_frame) - 12)
            if end <= start:
                return 0.5, 0
            wins = 0
            samples = 0
            for past_idx in range(start, end):
                is_setup, _, _ = buy_setup(past_idx)
                if not is_setup or pd.isna(atr.iloc[past_idx]):
                    continue
                future_slice = close.iloc[past_idx + 1 : past_idx + 11]
                low_slice = low.iloc[past_idx + 1 : past_idx + 11]
                if future_slice.empty:
                    continue
                entry = close.iloc[past_idx]
                take_profit = entry + atr.iloc[past_idx] * 1.6
                stop_loss = entry - atr.iloc[past_idx] * 1.1
                hit_profit = bool((future_slice >= take_profit).any())
                hit_stop = bool((low_slice <= stop_loss).any())
                final_gain = future_slice.iloc[-1] > entry * 1.012
                wins += int((hit_profit and not hit_stop) or final_gain)
                samples += 1
            if samples < 6:
                return 0.5, samples
            return wins / samples, samples

        feature_frame = pd.DataFrame(
            {
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "ma_gap": ma_gap,
                "trend_gap": trend_gap,
                "rsi": rsi,
                "macd": price_volume_macd,
                "atr_pct": atr_pct,
                "volume_ratio": volume_ratio,
                "distance_high": close / donchian_high - 1,
                "distance_low": close / donchian_low - 1,
            }
        ).replace([np.inf, -np.inf], np.nan)

        ml_probability = pd.Series(0.5, index=combined.index)
        ml_samples = 0
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            labels: list[int] = []
            train_rows: list[pd.Series] = []
            history_limit = len(history_frame)
            for idx in range(80, max(80, history_limit - 6)):
                if feature_frame.iloc[idx].isna().all() or pd.isna(atr.iloc[idx]):
                    continue
                entry = close.iloc[idx]
                upper_barrier = entry + atr.iloc[idx] * 1.4
                lower_barrier = entry - atr.iloc[idx] * 1.1
                future_high = high.iloc[idx + 1 : idx + 6]
                future_low = low.iloc[idx + 1 : idx + 6]
                future_close = close.iloc[idx + 5]
                if future_high.empty or future_low.empty or pd.isna(future_close):
                    continue
                hit_up_positions = np.where(future_high.to_numpy() >= upper_barrier)[0]
                hit_down_positions = np.where(future_low.to_numpy() <= lower_barrier)[0]
                first_up = int(hit_up_positions[0]) if len(hit_up_positions) else 99
                first_down = int(hit_down_positions[0]) if len(hit_down_positions) else 99
                if first_up < first_down:
                    label = 1
                elif first_down < first_up:
                    label = 0
                else:
                    label = int(future_close > entry * 1.006)
                train_rows.append(feature_frame.iloc[idx])
                labels.append(label)
            if len(set(labels)) == 2 and len(labels) >= 90:
                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            HistGradientBoostingClassifier(
                                max_iter=80,
                                max_leaf_nodes=15,
                                learning_rate=0.06,
                                l2_regularization=0.08,
                                random_state=7,
                            ),
                        ),
                    ]
                )
                x_train = pd.DataFrame(train_rows)
                model.fit(x_train, labels)
                probabilities = model.predict_proba(feature_frame)
                positive_index = list(model.classes_).index(1)
                ml_probability = pd.Series(probabilities[:, positive_index], index=combined.index)
                ml_samples = len(labels)
        except Exception:
            ml_probability = pd.Series(0.5, index=combined.index)
            ml_samples = 0

        signals: list[TradeSignalPoint] = []
        start_index = max(80, len(combined) - 320)
        predicted_dates = set(future_frame.index)
        holding = bool(close.iloc[start_index - 1] > ema20.iloc[start_index - 1] > ema60.iloc[start_index - 1])
        entry_price = float(close.iloc[start_index - 1]) if holding else 0.0
        highest_close = entry_price
        last_signal_index = -999

        for idx in range(start_index, len(combined)):
            if idx - last_signal_index < 5:
                continue
            date_index = combined.index[idx]
            price = float(close.iloc[idx])
            if not np.isfinite(price):
                continue

            is_buy_setup, buy_reasons, setup_score = buy_setup(idx)
            success_rate, sample_size = historical_success_rate(idx)
            probability = float(ml_probability.iloc[idx]) if pd.notna(ml_probability.iloc[idx]) else 0.5
            trend_bonus = 0.05 if close.iloc[idx] > ema20.iloc[idx] else 0.0
            pullback_bonus = 0.06 if close.iloc[idx] > ema20.iloc[idx] and close.iloc[idx - 1] < ema20.iloc[idx - 1] else 0.0
            calibrated_score = (
                probability * 0.55
                + setup_score * 0.25
                + max(success_rate - 0.42, 0) * 0.28
                + trend_bonus
                + pullback_bonus
            )
            atr_stop = highest_close - float(atr.iloc[idx]) * 2.2 if holding and pd.notna(atr.iloc[idx]) else 0.0
            stop_hit = bool(holding and atr_stop > 0 and close.iloc[idx] < atr_stop)
            channel_exit = bool(close.iloc[idx] < donchian_low.iloc[idx])
            fractal_exit = bool(swing_high.iloc[idx] and close.iloc[idx] < low.iloc[idx - 2])
            trend_exit = bool(close.iloc[idx] < ema20.iloc[idx] and ema20.iloc[idx] < ema60.iloc[idx])
            momentum_exit = bool(price_volume_macd.iloc[idx] < 0 and price_volume_macd.iloc[idx] < price_volume_macd.iloc[idx - 1])
            rsi_value = float(rsi.iloc[idx]) if pd.notna(rsi.iloc[idx]) else 50.0
            probability_exit = bool(probability < 0.43 and price < ema20.iloc[idx])
            sell_score = (
                (0.35 if stop_hit else 0.0)
                + (0.25 if channel_exit else 0.0)
                + (0.18 if fractal_exit else 0.0)
                + (0.14 if trend_exit else 0.0)
                + (0.08 if momentum_exit else 0.0)
                + (0.14 if probability_exit else 0.0)
                + (0.06 if rsi_value > 78 and price < close.iloc[idx - 1] else 0.0)
            )

            if holding:
                highest_close = max(highest_close, price)

            probability_buy = probability >= 0.58 and close.iloc[idx] > ema20.iloc[idx]
            if not holding and (is_buy_setup or probability_buy) and calibrated_score >= 0.55:
                buy_reasons.append(f"ML barrier上涨概率约{probability:.0%}" + (f"，训练样本{ml_samples}个" if ml_samples else ""))
                if sample_size >= 6:
                    buy_reasons.append(f"近似历史候选10日胜率约{success_rate:.0%}，样本{sample_size}个")
                else:
                    buy_reasons.append("历史相似样本不足，按形态与风险过滤触发")
                if probability_buy and not is_buy_setup:
                    buy_reasons.append("概率模型触发趋势内低位介入")
                signals.append(
                    TradeSignalPoint(
                        date=date_index.strftime("%Y-%m-%d"),
                        action="buy",
                        price=price,
                        strength=float(min(calibrated_score, 1.0)),
                        reason="；".join(buy_reasons),
                        predicted=date_index in predicted_dates,
                    )
                )
                holding = True
                entry_price = price
                highest_close = price
                last_signal_index = idx
            elif holding and sell_score >= 0.45:
                reasons = []
                if stop_hit:
                    reasons.append("跌破ATR动态止损线")
                if channel_exit:
                    reasons.append("跌破10日唐奇安下轨")
                if fractal_exit:
                    reasons.append("分形高点确认后跌破其低点")
                if trend_exit:
                    reasons.append("EMA20跌破EMA60且价格处于弱势")
                if momentum_exit:
                    reasons.append("量价调整MACD转弱")
                if probability_exit:
                    reasons.append(f"ML barrier上涨概率降至{probability:.0%}且跌破EMA20")
                if rsi_value > 78:
                    reasons.append(f"RSI {rsi_value:.0f}，存在短线过热")
                if entry_price and price > entry_price:
                    reasons.append(f"相对入场价浮盈约{price / entry_price - 1:.2%}，触发保护性退出")
                signals.append(
                    TradeSignalPoint(
                        date=date_index.strftime("%Y-%m-%d"),
                        action="sell",
                        price=price,
                        strength=float(min(sell_score, 1.0)),
                        reason="；".join(reasons),
                        predicted=date_index in predicted_dates,
                    )
                )
                holding = False
                entry_price = 0.0
                highest_close = 0.0
                last_signal_index = idx

        return signals[-12:]

    def _nullable_float(self, value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    def _validation_metrics(self, history: pd.DataFrame) -> list[ValidationMetric]:
        analogs = self.analog_forecaster.forecast(history, self.config.horizons)
        if not analogs:
            return [
                ValidationMetric(
                    name="历史校准",
                    value="样本不足",
                    detail="历史数据不足，当前预测主要依赖多因子规则评分。",
                )
            ]
        metrics: list[ValidationMetric] = []
        for horizon, analog in analogs.items():
            metrics.append(
                ValidationMetric(
                    name=f"相似行情 {horizon}日",
                    value=f"上涨概率 {analog.up_probability:.0%}",
                    detail=(
                        f"找到 {analog.sample_size} 个历史相似窗口；"
                        f"之后平均收益 {analog.expected_return:.2%}，"
                        f"中位数收益 {analog.median_return:.2%}，"
                        f"下跌超过3%概率 {analog.downside_probability:.0%}。"
                    ),
                )
            )
        return metrics

    def _model_diagnostics(self, history: pd.DataFrame) -> list[ModelDiagnostic]:
        prediction = self.ml_predictor.predict(history)
        if prediction is None:
            return [
                ModelDiagnostic(
                    model_name="ML模型",
                    probability_up=0.5,
                    train_samples=0,
                    test_accuracy=None,
                    test_auc=None,
                    top_features=[
                        FactorContribution(
                            name="样本不足",
                            value="N/A",
                            impact="neutral",
                            weight=0.0,
                            explanation="历史样本不足或标签单一，暂未启用机器学习校准。",
                        )
                    ],
                )
            ]
        return [
            ModelDiagnostic(
                model_name=prediction.model_name,
                probability_up=prediction.probability_up,
                train_samples=prediction.train_samples,
                test_accuracy=prediction.test_accuracy,
                test_auc=prediction.test_auc,
                top_features=[
                    FactorContribution(
                        name=name,
                        value=f"{importance:.2%}",
                        impact="neutral",
                        weight=float(importance),
                        explanation="该特征在当前机器学习模型中贡献较高。",
                    )
                    for name, importance in prediction.top_features
                ],
            )
        ]

    def _deep_diagnostics(self, history: pd.DataFrame) -> list[ModelDiagnostic]:
        prediction = self.deep_predictor.predict(history)
        return [
            ModelDiagnostic(
                model_name=prediction.model_name if prediction.enabled else "GRU未启用",
                probability_up=prediction.probability_up,
                train_samples=prediction.train_samples,
                test_accuracy=prediction.validation_accuracy,
                test_auc=None,
                top_features=[
                    FactorContribution(
                        name="深度学习序列模型",
                        value="启用" if prediction.enabled else "未启用",
                        impact="neutral",
                        weight=0.0,
                        explanation=prediction.message,
                    )
                ],
            )
        ]

    def _model_consensus(
        self,
        validation_metrics: list[ValidationMetric],
        model_diagnostics: list[ModelDiagnostic],
    ) -> ModelConsensus:
        probability_items: list[tuple[str, float, float]] = []
        for metric in validation_metrics:
            if "上涨概率" in metric.value:
                try:
                    pct = float(metric.value.split("上涨概率", 1)[1].strip().rstrip("%")) / 100
                    weight = 0.25 if "5日" in metric.name else 0.12
                    probability_items.append((metric.name, pct, weight))
                except Exception:
                    continue
        for diagnostic in model_diagnostics:
            if diagnostic.train_samples <= 0:
                continue
            reliability = diagnostic.test_auc or diagnostic.test_accuracy or 0.5
            weight = 0.25 + max(reliability - 0.5, 0) * 0.5
            if diagnostic.model_name.lower().startswith("gru"):
                weight *= 0.7
            probability_items.append((diagnostic.model_name, diagnostic.probability_up, weight))

        if not probability_items:
            return ModelConsensus(
                consensus_probability=0.5,
                disagreement_level="样本不足",
                summary="暂无足够模型输出形成共识。",
                details=["当前结果主要依赖多因子规则和图形解释。"],
            )

        total_weight = sum(item[2] for item in probability_items) or 1
        consensus = sum(probability * weight for _, probability, weight in probability_items) / total_weight
        probabilities = [item[1] for item in probability_items]
        spread = max(probabilities) - min(probabilities)
        if spread >= 0.25:
            level = "高分歧"
            summary = "不同模型结论差异较大，建议降低仓位或等待更多确认信号。"
        elif spread >= 0.12:
            level = "中等分歧"
            summary = "模型存在一定分歧，建议结合均线、量能和风险提示综合判断。"
        else:
            level = "低分歧"
            summary = "模型输出较一致，预测可信度相对更高。"
        details = [
            f"{name}: {probability:.0%}，权重 {weight / total_weight:.0%}"
            for name, probability, weight in probability_items
        ]
        return ModelConsensus(
            consensus_probability=float(consensus),
            disagreement_level=level,
            summary=summary,
            details=details,
        )

    def _model_adjustment(self, diagnostics: list[ModelDiagnostic], model_mode: str) -> float:
        if model_mode == "factor":
            return 0.0
        adjustments = []
        for diagnostic in diagnostics:
            if diagnostic.train_samples <= 0:
                continue
            confidence = diagnostic.test_auc or diagnostic.test_accuracy or 0.5
            reliability = max(min((confidence - 0.5) * 2, 0.35), 0.0)
            if diagnostic.model_name.lower().startswith("gru"):
                weight = 0.45 if model_mode == "deep" else 0.18
            else:
                weight = 0.45 if model_mode == "ml" else 0.25
            if model_mode == "deep" and not diagnostic.model_name.lower().startswith("gru"):
                continue
            if model_mode == "ml" and diagnostic.model_name.lower().startswith("gru"):
                continue
            adjustments.append((diagnostic.probability_up - 0.5) * reliability * 0.04 * weight)
        return float(sum(adjustments))

    def _ml_adjustment(self, diagnostics: list[ModelDiagnostic]) -> float:
        if not diagnostics:
            return 0.0
        diagnostic = diagnostics[0]
        if diagnostic.train_samples <= 0:
            return 0.0
        confidence = diagnostic.test_auc or diagnostic.test_accuracy or 0.5
        reliability = max(min((confidence - 0.5) * 2, 0.35), 0.0)
        return (diagnostic.probability_up - 0.5) * reliability * 0.04

    def _market_analysis_cn(self, name: str, forecast: MarketForecast) -> str:
        main = forecast.forecasts[1] if len(forecast.forecasts) > 1 else forecast.forecasts[0]
        direction = self._direction_cn(main.direction)
        return (
            f"{name}未来{main.horizon_days}个交易日预计{direction}，"
            f"预期收益约{main.expected_return:.2%}，置信度{main.confidence:.0%}。"
            f"当前市场状态为{self._regime_cn(forecast.regime)}，"
            f"适合采用{self._position_style(forecast.suggested_position)}。"
        )

    def _stock_analysis_cn(self, name: str, forecast: StockForecast) -> str:
        main = forecast.forecasts[1] if len(forecast.forecasts) > 1 else forecast.forecasts[0]
        return (
            f"{name}未来{main.horizon_days}个交易日走势判断为{self._direction_cn(main.direction)}，"
            f"预期收益约{main.expected_return:.2%}，置信度{main.confidence:.0%}。"
            f"跑赢基准概率{forecast.outperform_probability:.0%}，"
            f"回撤风险{forecast.drawdown_risk:.0%}，综合建议：{self._action_cn(forecast.action)}。"
        )

    def _direction_cn(self, direction: str) -> str:
        return {"up": "偏强上行", "flat": "震荡整理", "down": "偏弱下行"}[direction]

    def _regime_cn(self, regime: str) -> str:
        return {
            "bull": "趋势上行",
            "neutral": "震荡均衡",
            "bear": "弱势下行",
            "high_volatility": "高波动风险",
        }[regime]

    def _action_cn(self, action: str) -> str:
        return {"buy": "关注买入", "hold": "继续观察", "avoid": "暂时规避"}[action]

    def _position_style(self, position: float) -> str:
        if position >= 0.65:
            return "偏积极仓位，优先选择强趋势和高流动性标的"
        if position <= 0.35:
            return "防守仓位，控制回撤并减少追高"
        return "中性仓位，等待趋势进一步确认"

    def _market_sections(
        self,
        name: str,
        history: pd.DataFrame,
        forecast: MarketForecast,
    ) -> list[AnalysisSection]:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        ret_5d = float(latest["ret_5d"])
        ret_20d = float(latest["ret_20d"])
        ma_gap = float(latest["ma_gap_5_20"])
        volatility = float(latest["volatility_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])
        drawdown = float(latest["drawdown_20"])
        main = forecast.forecasts[1] if len(forecast.forecasts) > 1 else forecast.forecasts[0]
        return [
            AnalysisSection(
                title="趋势结构",
                conclusion=f"{name}短期趋势{self._direction_cn(main.direction)}。",
                details=[
                    f"5日收益率为{ret_5d:.2%}，20日收益率为{ret_20d:.2%}。",
                    f"MA5/MA20 偏离为{ma_gap:.2%}，用于判断短期趋势是否强于中期均线。",
                ],
            ),
            AnalysisSection(
                title="量能与情绪",
                conclusion="成交量能" + ("有所放大。" if volume_ratio > 1 else "尚未明显放大。"),
                details=[
                    f"5日/20日成交量比值为{volume_ratio:.2f}。",
                    "量能放大通常支持趋势延续，量能不足时更容易进入震荡。",
                ],
            ),
            AnalysisSection(
                title="波动与回撤",
                conclusion="风险水平" + ("偏高。" if volatility > 0.02 or drawdown < -0.08 else "可控。"),
                details=[
                    f"20日波动率为{volatility:.2%}，20日区间回撤为{drawdown:.2%}。",
                    f"当前仓位建议为{forecast.suggested_position:.0%}，用于匹配市场风险状态。",
                ],
            ),
            AnalysisSection(
                title="策略含义",
                conclusion=self._position_style(forecast.suggested_position) + "。",
                details=[
                    "大盘强势时优先考虑趋势跟随和行业轮动。",
                    "大盘弱势或高波动时优先控制仓位、减少追高和低流动性标的。",
                ],
            ),
        ]

    def _stock_sections(
        self,
        name: str,
        history: pd.DataFrame,
        forecast: StockForecast,
        benchmark_history: pd.DataFrame | None,
    ) -> list[AnalysisSection]:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        ret_5d = float(latest["ret_5d"])
        ret_20d = float(latest["ret_20d"])
        ma_gap = float(latest["ma_gap_5_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])
        drawdown = float(latest["drawdown_20"])
        volatility = float(latest["volatility_20"])
        benchmark_ret = 0.0
        if benchmark_history is not None:
            benchmark_features = add_technical_features(benchmark_history)
            benchmark_ret = float(benchmark_features.iloc[-1]["ret_20d"])
        excess = ret_20d - benchmark_ret
        main = forecast.forecasts[1] if len(forecast.forecasts) > 1 else forecast.forecasts[0]
        return [
            AnalysisSection(
                title="价格趋势",
                conclusion=f"{name}短线走势判断为{self._direction_cn(main.direction)}。",
                details=[
                    f"5日收益率{ret_5d:.2%}，20日收益率{ret_20d:.2%}。",
                    f"MA5/MA20 偏离为{ma_gap:.2%}，反映短期趋势相对中期趋势的位置。",
                ],
            ),
            AnalysisSection(
                title="相对强弱",
                conclusion="相对基准" + ("占优。" if excess > 0 else "偏弱。"),
                details=[
                    f"20日相对基准收益为{excess:.2%}。",
                    f"模型估计跑赢基准概率为{forecast.outperform_probability:.0%}。",
                ],
            ),
            AnalysisSection(
                title="量能确认",
                conclusion="资金交易活跃度" + ("改善。" if volume_ratio > 1 else "一般。"),
                details=[
                    f"5日/20日成交量比值为{volume_ratio:.2f}。",
                    "趋势上涨但量能不足时，需要警惕突破失败或冲高回落。",
                ],
            ),
            AnalysisSection(
                title="风险控制",
                conclusion="风险复核结果为" + self._action_cn(forecast.action) + "。",
                details=[
                    f"20日区间回撤为{drawdown:.2%}，20日波动率为{volatility:.2%}。",
                    f"模型估计未来回撤风险为{forecast.drawdown_risk:.0%}。",
                ],
            ),
        ]

    def _market_strategy_description(self) -> str:
        return (
            "当前版本采用透明多因子预测策略：综合趋势动量、短期延续、均线结构、"
            "波动惩罚、量能确认、回撤惩罚和新闻情绪，形成市场状态评分。"
            "图上的买卖点采用ML barrier标签模型和meta-labeling风格过滤："
            "先用未来收益/风险障碍训练上涨概率，再结合唐奇安突破、分形转强、"
            "ATR动态止损和量价动量生成入场/退出点。"
            "该设计参考了多因子量化研究中的动量、风险、流动性和情绪维度；"
            "未来K线根据综合评分、近期波动率和预测收益生成，用于展示情景路径，"
            "不代表确定价格。"
        )

    def _stock_strategy_description(self) -> str:
        return (
            "当前版本采用透明多因子选股预测策略：综合相对强弱、绝对动量、"
            "短期延续、均线结构、量能确认、低波动质量、回撤控制和新闻情绪。"
            "图上的买卖点采用ML barrier标签模型和meta-labeling风格过滤："
            "先用未来收益/风险障碍训练上涨概率，再结合唐奇安突破、分形转强、"
            "ATR动态止损和量价动量生成入场/退出点。"
            "跑赢概率由综合因子评分和相对基准收益共同决定，风险复核由回撤、"
            "波动和量能走弱共同约束。最终建议分为关注买入、继续观察和暂时规避。"
        )
