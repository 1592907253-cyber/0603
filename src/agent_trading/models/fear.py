from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FearIndex:
    score: float
    level: str
    summary: str
    details: list[str]


class FearIndexModel:
    """A-share fear proxy.

    VIX-like option-implied volatility is not always available for a portable
    demo, so this proxy uses realized market stress components that are present
    in OHLCV data.
    """

    def calculate(self, history: pd.DataFrame) -> FearIndex:
        data = history.copy().tail(90)
        close = pd.to_numeric(data["close"], errors="coerce")
        open_ = pd.to_numeric(data["open"], errors="coerce")
        volume = pd.to_numeric(data["volume"], errors="coerce")
        returns = close.pct_change().dropna()

        realized_vol = float(returns.tail(20).std() or 0.0)
        annualized_vol = realized_vol * (252**0.5)
        drawdown_20 = float(close.iloc[-1] / close.tail(20).max() - 1)
        down_ratio = float((returns.tail(20) < 0).mean())
        large_down_ratio = float((returns.tail(20) < -0.02).mean())
        gap_down_ratio = float(((open_ / close.shift(1) - 1).tail(20) < -0.01).mean())
        volume_ratio = float(volume.tail(5).mean() / volume.tail(20).mean()) if volume.tail(20).mean() else 1.0
        ma20 = close.rolling(20).mean().iloc[-1]
        price_ma20_gap = float(close.iloc[-1] / ma20 - 1) if ma20 else 0.0

        vol_score = self._clamp((annualized_vol - 0.16) / 0.28)
        drawdown_score = self._clamp(abs(min(drawdown_20, 0)) / 0.12)
        breadth_score = self._clamp((down_ratio - 0.45) / 0.35)
        crash_score = self._clamp(large_down_ratio / 0.25 + gap_down_ratio / 0.2)
        volume_score = self._clamp((volume_ratio - 1.0) / 1.2)
        trend_break_score = self._clamp(abs(min(price_ma20_gap, 0)) / 0.08)

        score = 100 * (
            vol_score * 0.28
            + drawdown_score * 0.22
            + breadth_score * 0.16
            + crash_score * 0.14
            + volume_score * 0.1
            + trend_break_score * 0.1
        )
        score = round(self._clamp(score / 100, 0, 1) * 100, 1)
        level = self._level(score)
        details = [
            f"20日年化实现波动约 {annualized_vol:.1%}。",
            f"20日区间回撤 {drawdown_20:.2%}。",
            f"近20日下跌日占比 {down_ratio:.0%}，大跌日占比 {large_down_ratio:.0%}。",
            f"近5日/20日成交量比 {volume_ratio:.2f}。",
            f"价格相对MA20偏离 {price_ma20_gap:.2%}。",
        ]
        summary = f"A股恐慌指数 {score:.1f}/100，处于{level}区间。"
        return FearIndex(score=score, level=level, summary=summary, details=details)

    def _level(self, score: float) -> str:
        if score >= 75:
            return "极端恐慌"
        if score >= 55:
            return "高恐慌"
        if score >= 35:
            return "中性偏谨慎"
        if score >= 20:
            return "风险偏好正常"
        return "低恐慌"

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return min(max(float(value), lower), upper)
