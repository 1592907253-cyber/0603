from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, build_data_provider
from agent_trading.models.forecast import HeuristicForecastModel
from agent_trading.schemas import AgentReport, StrategyCandidate
from agent_trading.settings import get_settings


class ResearchWorkflow:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = build_data_provider(settings.data_provider)
        self.model = HeuristicForecastModel()

    def run_market_and_stock_scan(
        self,
        benchmark: str = "000300.SH",
        stock_pool: list[str] | None = None,
    ) -> AgentReport:
        stock_pool = stock_pool or ["600519.SH", "000001.SZ", "300750.SZ"]
        end = date.today()
        start = end - timedelta(days=260)

        benchmark_history = self.provider.history(MarketDataRequest(benchmark, start, end))
        market = self.model.market_forecast(benchmark, benchmark_history)

        candidates: list[StrategyCandidate] = []
        risk_notes: list[str] = []
        for symbol in stock_pool:
            history = self.provider.history(MarketDataRequest(symbol, start, end))
            forecast = self.model.stock_forecast(symbol, history, benchmark_history)
            score = forecast.outperform_probability - forecast.drawdown_risk * 0.35
            candidates.append(
                StrategyCandidate(
                    symbol=symbol,
                    score=round(score, 4),
                    action=forecast.action,
                    reason=forecast.explanations[0].detail,
                )
            )
            if forecast.action == "avoid":
                risk_notes.append(f"{symbol}: {', '.join(forecast.risks)}")

        candidates.sort(key=lambda item: item.score, reverse=True)
        fear_text = f"，恐慌指数为 {market.fear_index.score:.1f}（{market.fear_index.level}）" if market.fear_index else ""
        summary = (
            f"当前大盘状态为 {market.regime}，建议仓位约 {market.suggested_position:.0%}{fear_text}。"
            f"综合评分最高的候选标的是 {candidates[0].symbol}。"
        )
        risk_review = (
            "风险复核：暂未发现需要一票否决的风险。"
            if not risk_notes
            else "风险复核提示：" + " | ".join(risk_notes)
        )
        return AgentReport(
            market=market,
            candidates=candidates,
            summary=summary,
            risk_review=risk_review,
        )
