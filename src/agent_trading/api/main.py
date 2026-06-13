from datetime import date, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from agent_trading.agents.workflow import ResearchWorkflow
from agent_trading.agents.opportunity import OpportunityAgent
from agent_trading.data.providers import MarketDataRequest, build_data_provider
from agent_trading.data.symbols import search_symbols
from agent_trading.models.forecast import HeuristicForecastModel
from agent_trading.qlib_integration.service import QlibService
from agent_trading.schemas import AgentReport, MarketForecast, PredictionChart, StockForecast, SymbolInfo
from agent_trading.schemas import (
    QlibStatus,
    QlibTrainingRequest,
    QlibTrainingResponse,
    SectorOpportunityReport,
    StockOpportunityReport,
)
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
    settings = get_settings()
    return {
        "status": "ok",
        "data_provider": settings.data_provider,
        "akshare_disable_proxy": str(settings.akshare_disable_proxy).lower(),
    }


@app.get("/qlib/status", response_model=QlibStatus)
def qlib_status() -> QlibStatus:
    return QlibService().status()


@app.post("/qlib/train", response_model=QlibTrainingResponse)
def qlib_train(request: QlibTrainingRequest) -> QlibTrainingResponse:
    return QlibService().create_or_run_workflow(request)


@app.get("/symbols/search", response_model=list[SymbolInfo])
def symbols_search(q: str = Query("", description="Stock or index keyword.")) -> list[SymbolInfo]:
    settings = get_settings()
    return search_symbols(q, provider_name=settings.data_provider)


@app.get("/opportunities/sectors", response_model=SectorOpportunityReport)
def sector_opportunities(limit: int = Query(8, ge=1, le=30)) -> SectorOpportunityReport:
    try:
        settings = get_settings()
        return OpportunityAgent(settings.data_provider).sector_opportunities(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"板块机会扫描失败：{exc}") from exc


@app.get("/opportunities/stocks", response_model=StockOpportunityReport)
def stock_opportunities(limit: int = Query(30, ge=1, le=100)) -> StockOpportunityReport:
    try:
        settings = get_settings()
        return OpportunityAgent(settings.data_provider).stock_opportunities(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"全A股机会扫描失败：{exc}") from exc


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


@app.get("/chart/prediction/{symbol}", response_model=PredictionChart)
def prediction_chart(
    symbol: str,
    benchmark: str = Query("000300.SH", description="Benchmark index symbol."),
    model_mode: str = Query("ensemble", description="factor, ml, deep, or ensemble."),
) -> PredictionChart:
    try:
        settings = get_settings()
        provider = build_data_provider(settings.data_provider)
        end = date.today()
        start = end - timedelta(days=365 * 5)
        history = provider.history(MarketDataRequest(symbol, start, end))
        benchmark_history = provider.history(MarketDataRequest(benchmark, start, end))
        return HeuristicForecastModel().prediction_chart(
            symbol,
            history,
            benchmark_history,
            provider_name=settings.data_provider,
            model_mode=model_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"标的数据获取或分析失败：{exc}") from exc


@app.get("/agents/research", response_model=AgentReport)
def research_scan(
    benchmark: str = Query("000300.SH"),
    symbols: str = Query("600519.SH,000001.SZ,300750.SZ"),
) -> AgentReport:
    stock_pool = [item.strip() for item in symbols.split(",") if item.strip()]
    return ResearchWorkflow().run_market_and_stock_scan(benchmark=benchmark, stock_pool=stock_pool)
