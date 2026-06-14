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


class FearIndexSnapshot(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    level: str
    summary: str
    details: list[str]


class MarketForecast(BaseModel):
    symbol: str
    regime: MarketRegime
    suggested_position: float = Field(ge=0.0, le=1.0)
    forecasts: list[ForecastHorizon]
    explanations: list[ExplanationItem]
    risks: list[str]
    fear_index: FearIndexSnapshot | None = None


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


class KlinePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    predicted: bool = False


class IndicatorPoint(BaseModel):
    date: str
    ma5: float | None = None
    ma10: float | None = None
    ma20: float | None = None
    ma60: float | None = None
    ma120: float | None = None
    ma250: float | None = None


class TradeSignalPoint(BaseModel):
    date: str
    action: Literal["buy", "sell"]
    price: float
    strength: float = Field(ge=0.0, le=1.0)
    reason: str
    predicted: bool = False


class AnalysisSection(BaseModel):
    title: str
    conclusion: str
    details: list[str]


class FactorContribution(BaseModel):
    name: str
    value: str
    impact: Literal["positive", "neutral", "negative"]
    weight: float
    explanation: str


class ValidationMetric(BaseModel):
    name: str
    value: str
    detail: str


class ModelDiagnostic(BaseModel):
    model_name: str
    probability_up: float
    train_samples: int
    test_accuracy: float | None = None
    test_auc: float | None = None
    top_features: list[FactorContribution]


class ModelConsensus(BaseModel):
    consensus_probability: float
    disagreement_level: str
    summary: str
    details: list[str]


class PredictionModelMode(BaseModel):
    mode: str
    description: str


class PredictionChart(BaseModel):
    symbol: str
    name: str
    kind: Literal["index", "stock"]
    analysis: str
    trend_summary: str
    risk_summary: str
    analysis_sections: list[AnalysisSection]
    strategy_description: str
    factor_contributions: list[FactorContribution]
    validation_metrics: list[ValidationMetric]
    model_diagnostics: list[ModelDiagnostic]
    model_consensus: ModelConsensus
    selected_model_mode: str
    fear_index: FearIndexSnapshot | None = None
    history: list[KlinePoint]
    future: list[KlinePoint]
    indicators: list[IndicatorPoint]
    trade_signals: list[TradeSignalPoint] = Field(default_factory=list)
    explanations: list[ExplanationItem]


class SymbolInfo(BaseModel):
    symbol: str
    name: str
    kind: Literal["index", "stock"]


class SectorTrendPoint(BaseModel):
    date: str
    close: float
    ma20: float | None = None
    ma60: float | None = None
    source: str = "real"


class StockTrendPoint(BaseModel):
    date: str
    close: float
    ma20: float | None = None
    ma60: float | None = None
    source: str = "real"


class SectorOpportunity(BaseModel):
    name: str
    score: float
    rating: str
    change_pct: float | None = None
    turnover_rate: float | None = None
    leader: str | None = None
    leader_change_pct: float | None = None
    trend_summary: str | None = None
    trend_points: list[SectorTrendPoint] = Field(default_factory=list)
    breadth_summary: str | None = None
    news_summary: str | None = None
    signal_breakdown: list[FactorContribution]
    reasons: list[str]
    risks: list[str]
    watch_points: list[str]


class SectorOpportunityReport(BaseModel):
    summary: str
    methodology: str
    sectors: list[SectorOpportunity]


class StockOpportunity(BaseModel):
    symbol: str
    name: str
    score: float
    rating: str
    sector: str | None = None
    change_pct: float | None = None
    volume_ratio: float | None = None
    turnover_rate: float | None = None
    amount: float | None = None
    trend_summary: str | None = None
    trend_points: list[StockTrendPoint] = Field(default_factory=list)
    fund_flow_summary: str | None = None
    signal_breakdown: list[FactorContribution]
    reasons: list[str]
    risks: list[str]
    watch_points: list[str]


class StockOpportunityReport(BaseModel):
    summary: str
    methodology: str
    stocks: list[StockOpportunity]


class SectorStockGroup(BaseModel):
    sector: str
    sector_score: float
    sector_reasons: list[str]
    stocks: list[StockOpportunity]


class SectorStockOpportunityReport(BaseModel):
    summary: str
    methodology: str
    groups: list[SectorStockGroup]


class QlibStatus(BaseModel):
    installed: bool
    data_dir: str
    data_ready: bool
    message: str


class QlibTrainingRequest(BaseModel):
    market: str = "csi300"
    benchmark: str = "SH000300"
    start_time: str = "2018-01-01"
    end_time: str = "2025-12-31"
    train_end_time: str = "2021-12-31"
    valid_start_time: str = "2022-01-01"
    valid_end_time: str = "2023-12-31"
    test_start_time: str = "2024-01-01"
    test_end_time: str = "2025-12-31"
    topk: int = 50
    n_drop: int = 5
    run: bool = False


class QlibTrainingResponse(BaseModel):
    status: str
    message: str
    config_path: str
    command: str
    stdout_tail: str | None = None
    stderr_tail: str | None = None
