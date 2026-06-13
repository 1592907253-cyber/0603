from typing import Literal

from pydantic import BaseModel, Field


MarketRegime = Literal["bull", "neutral", "bear", "high_volatility"]
SignalAction = Literal["buy", "hold", "avoid"]


class ForecastHorizon(BaseModel):
    horizon_days: int
    direction: Literal["up", "flat", "down"]
    expected_return: float
    confidence: float = Field(ge=0.0, le=1.0)


class ExplanationItem(BaseModel):
    factor: str
    impact: Literal["positive", "neutral", "negative"]
    detail: str


class MarketForecast(BaseModel):
    symbol: str
    regime: MarketRegime
    suggested_position: float = Field(ge=0.0, le=1.0)
    forecasts: list[ForecastHorizon]
    explanations: list[ExplanationItem]
    risks: list[str]


class StockForecast(BaseModel):
    symbol: str
    action: SignalAction
    outperform_probability: float = Field(ge=0.0, le=1.0)
    drawdown_risk: float = Field(ge=0.0, le=1.0)
    forecasts: list[ForecastHorizon]
    explanations: list[ExplanationItem]
    risks: list[str]


class StrategyCandidate(BaseModel):
    symbol: str
    score: float
    action: SignalAction
    reason: str


class AgentReport(BaseModel):
    market: MarketForecast
    candidates: list[StrategyCandidate]
    summary: str
    risk_review: str
