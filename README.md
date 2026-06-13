# AgentTrading

A hackathon-ready agentic quant research scaffold for China A-share market
analysis, index/stock forecasting, strategy validation, and explainable reports.

## What This Project Provides

- A-share data provider interface with mock data now and AKShare/TuShare adapters planned.
- Feature engineering for market index and stock OHLCV data.
- Forecasting services for market regime, index direction, stock alpha, and drawdown risk.
- Agent workflow skeleton for market analysis, stock selection, risk review, and report generation.
- FastAPI service exposing forecast and analysis endpoints.
- Lightweight browser dashboard for demo use.
- Simple backtest utilities for validating prediction-driven strategies.

## Suggested Architecture

```text
data providers -> feature pipeline -> forecast models -> agents -> api/dashboard
                                             |
                                      backtest/evaluation
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn agent_trading.api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
web/index.html
```

The default app uses deterministic mock data so the demo can run without data
vendor credentials.

## Useful Commands

Run the agent scan:

```powershell
agent-trading
```

Run a demo time-series training evaluation on mock沪深300 data:

```powershell
agent-trading train-demo
```

Run tests:

```powershell
pytest
```

API examples:

```text
GET /forecast/market/000300.SH
GET /forecast/stock/600519.SH?benchmark=000300.SH
GET /agents/research?benchmark=000300.SH&symbols=600519.SH,000001.SZ,300750.SZ
```

## Roadmap

1. Replace mock data with AKShare/TuShare adapters.
2. Train LightGBM/CatBoost models for:
   - index direction and market regime
   - stock excess return probability
   - stock drawdown risk
3. Add SHAP or feature-importance explanations.
4. Add Qlib-based research workflow and backtest reports.
5. Add LangGraph orchestration for multi-agent debate and risk review.

## Disclaimer

This project is for research and hackathon demonstration only. It is not
investment advice.
