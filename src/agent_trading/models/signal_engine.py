from dataclasses import dataclass

import pandas as pd

from agent_trading.agents.sentiment import SentimentResult
from agent_trading.features.technical import add_technical_features
from agent_trading.models.fear import FearIndexModel
from agent_trading.schemas import FactorContribution


@dataclass(frozen=True)
class SignalScore:
    score: float
    contributions: list[FactorContribution]


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def impact_from_score(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


class MultiFactorSignalEngine:
    """Transparent multi-factor score engine inspired by common quant practice."""

    def __init__(self) -> None:
        self.fear_model = FearIndexModel()

    def market_score(self, history: pd.DataFrame, sentiment: SentimentResult | None = None) -> SignalScore:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        fear = self.fear_model.calculate(history)
        ret_20d = float(latest["ret_20d"])
        ret_5d = float(latest["ret_5d"])
        ma_gap = float(latest["ma_gap_5_20"])
        ma_alignment = float(latest["ma_alignment_score"])
        price_ma20_gap = float(latest["price_ma20_gap"])
        volatility = float(latest["volatility_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])
        drawdown = float(latest["drawdown_20"])
        sentiment_score = sentiment.score if sentiment else 0.0

        raw = [
            self._factor("趋势动量", ret_20d / 0.08, 0.24, f"20日收益率 {ret_20d:.2%}，用于衡量中期趋势。"),
            self._factor("短期确认", ret_5d / 0.035, 0.12, f"5日收益率 {ret_5d:.2%}，用于观察趋势是否延续。"),
            self._factor("均线结构", ma_gap / 0.035, 0.18, f"MA5/MA20 偏离 {ma_gap:.2%}，反映短线相对中线强弱。"),
            self._factor("均线排列", (ma_alignment - 0.5) * 2, 0.1, f"MA5/MA10/MA20/MA60 排列得分 {ma_alignment:.2f}，多头排列更支持趋势延续。"),
            self._factor("MA20支撑", price_ma20_gap / 0.06, 0.08, f"价格相对 MA20 偏离 {price_ma20_gap:.2%}，站上MA20通常代表中短期支撑较强。"),
            self._factor("波动惩罚", -(volatility - 0.015) / 0.02, 0.16, f"20日波动率 {volatility:.2%}，高波动会降低预测可信度。"),
            self._factor("量能确认", (volume_ratio - 1) / 0.45, 0.12, f"5/20日量能比 {volume_ratio:.2f}，放量支持趋势延续。"),
            self._factor("回撤惩罚", drawdown / 0.12, 0.1, f"20日回撤 {drawdown:.2%}，回撤越深越偏防守。"),
            self._factor("恐慌指数", -(fear.score - 35) / 40, 0.12, f"{fear.summary} 恐慌上行时降低仓位和趋势置信度。"),
            self._factor("新闻情绪", sentiment_score, 0.08, sentiment.summary if sentiment else "未接入新闻情绪，按中性处理。"),
        ]
        return SignalScore(score=sum(item.weight * self._signed_value(item.value) for item in raw), contributions=raw)

    def stock_score(
        self,
        history: pd.DataFrame,
        benchmark_history: pd.DataFrame | None,
        sentiment: SentimentResult | None = None,
    ) -> SignalScore:
        features = add_technical_features(history)
        latest = features.iloc[-1]
        ret_20d = float(latest["ret_20d"])
        ret_5d = float(latest["ret_5d"])
        ma_gap = float(latest["ma_gap_5_20"])
        ma_alignment = float(latest["ma_alignment_score"])
        price_ma20_gap = float(latest["price_ma20_gap"])
        volatility = float(latest["volatility_20"])
        volume_ratio = float(latest["volume_ratio_5_20"])
        drawdown = float(latest["drawdown_20"])
        benchmark_ret = 0.0
        if benchmark_history is not None:
            benchmark_features = add_technical_features(benchmark_history)
            benchmark_ret = float(benchmark_features.iloc[-1]["ret_20d"])
        excess = ret_20d - benchmark_ret
        sentiment_score = sentiment.score if sentiment else 0.0

        raw = [
            self._factor("相对强弱", excess / 0.08, 0.22, f"20日相对基准收益 {excess:.2%}，衡量是否跑赢大盘。"),
            self._factor("绝对动量", ret_20d / 0.1, 0.14, f"个股20日收益率 {ret_20d:.2%}，用于判断自身趋势。"),
            self._factor("短期延续", ret_5d / 0.04, 0.1, f"个股5日收益率 {ret_5d:.2%}，观察短线延续性。"),
            self._factor("均线结构", ma_gap / 0.04, 0.14, f"MA5/MA20 偏离 {ma_gap:.2%}，反映趋势结构。"),
            self._factor("均线排列", (ma_alignment - 0.5) * 2, 0.1, f"MA5/MA10/MA20/MA60 排列得分 {ma_alignment:.2f}，多头排列更有利于趋势延续。"),
            self._factor("MA20支撑", price_ma20_gap / 0.07, 0.08, f"价格相对 MA20 偏离 {price_ma20_gap:.2%}，跌破MA20会削弱短中期趋势。"),
            self._factor("量能确认", (volume_ratio - 1) / 0.5, 0.1, f"5/20日量能比 {volume_ratio:.2f}，放量突破更可靠。"),
            self._factor("低波动质量", -(volatility - 0.018) / 0.025, 0.08, f"20日波动率 {volatility:.2%}，高波动会降低稳定性。"),
            self._factor("回撤控制", drawdown / 0.15, 0.12, f"20日回撤 {drawdown:.2%}，回撤越深风险越高。"),
            self._factor("新闻情绪", sentiment_score, 0.1, sentiment.summary if sentiment else "未接入新闻情绪，按中性处理。"),
        ]
        return SignalScore(score=sum(item.weight * self._signed_value(item.value) for item in raw), contributions=raw)

    def _factor(self, name: str, score: float, weight: float, explanation: str) -> FactorContribution:
        bounded = clamp(score)
        return FactorContribution(
            name=name,
            value=f"{bounded:+.2f}",
            impact=impact_from_score(bounded),
            weight=weight,
            explanation=explanation,
        )

    def _signed_value(self, value: str) -> float:
        return float(value)
