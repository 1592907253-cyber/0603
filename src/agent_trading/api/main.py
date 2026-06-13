from datetime import date, timedelta

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from agent_trading.agents.workflow import ResearchWorkflow
from agent_trading.data.providers import MarketDataRequest, build_data_provider
from agent_trading.models.forecast import HeuristicForecastModel
from agent_trading.schemas import AgentReport, MarketForecast, StockForecast
from agent_trading.settings import get_settings


app = FastAPI(title="AgentTrading API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/forecast/market/{symbol}", response_model=MarketForecast)
def forecast_market(symbol: str = "000300.SH") -> MarketForecast:
    settings = get_settings()
    provider = build_data_provider(settings.data_provider)
    end = date.today()
    start = end - timedelta(days=260)
    history = provider.history(MarketDataRequest(symbol, start, end))
    return HeuristicForecastModel().market_forecast(symbol, history)


@app.get("/forecast/stock/{symbol}", response_model=StockForecast)
def forecast_stock(
    symbol: str,
    benchmark: str = Query("000300.SH", description="Benchmark index symbol."),
) -> StockForecast:
    settings = get_settings()
    provider = build_data_provider(settings.data_provider)
    end = date.today()
    start = end - timedelta(days=260)
    stock_history = provider.history(MarketDataRequest(symbol, start, end))
    benchmark_history = provider.history(MarketDataRequest(benchmark, start, end))
    return HeuristicForecastModel().stock_forecast(symbol, stock_history, benchmark_history)


@app.get("/agents/research", response_model=AgentReport)
def research_scan(
    benchmark: str = Query("000300.SH"),
    symbols: str = Query("600519.SH,000001.SZ,300750.SZ"),
) -> AgentReport:
    stock_pool = [item.strip() for item in symbols.split(",") if item.strip()]
    return ResearchWorkflow().run_market_and_stock_scan(benchmark=benchmark, stock_pool=stock_pool)
