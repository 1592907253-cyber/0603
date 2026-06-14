# AgentTrading - yes茄豆

第40组「yes茄豆」黑客松参赛项目。

AgentTrading is a hackathon-ready agentic quant research scaffold for China
A-share market analysis, index/stock forecasting, strategy validation, and
explainable reports.

## Team Information

- Group: 第40组
- Team Name: yes茄豆
- Project Track: Agent + Quantitative Trading
- Target Market: 中国 A 股
- Repository: https://github.com/1592907253-cyber/0603

## What This Project Provides

- A-share data provider interface with deterministic mock data and optional AKShare real data.
- Feature engineering for market index and stock OHLCV data.
- Forecasting services for market regime, index direction, stock alpha, and drawdown risk.
- Agent workflow skeleton for market analysis, stock selection, risk review, and report generation.
- FastAPI service exposing forecast and analysis endpoints.
- Lightweight browser dashboard with all A-share search, ECharts K-line chart, and model diagnostics.
- Simple backtest utilities for validating prediction-driven strategies.

## Problem Statement

China A-share investors face noisy market data, rapidly rotating sectors, and
strong market-specific constraints such as price limits, trading suspensions,
ST risk, and T+1 trading. A normal chatbot can explain market news, but it
usually cannot connect data, forecasts, strategy validation, and risk review
into one auditable workflow.

This project builds a multi-agent research assistant that turns market data
into explainable forecasts and strategy candidates. The goal is not to provide
investment advice, but to demonstrate a verifiable research workflow for
forecasting and risk-aware decision support.

## Core Features

- Market Forecast Agent
  - Predicts broad index direction and market regime.
  - Produces suggested position level for defensive or aggressive strategy use.
- Stock Forecast Agent
  - Estimates stock outperformance probability against a benchmark index.
  - Flags drawdown and liquidity risk.
- Risk Review Agent
  - Reviews weak momentum, elevated volatility, and high drawdown risk.
  - Outputs "buy / hold / avoid" style research actions.
- Strategy Candidate Ranking
  - Scores stock candidates by alpha probability and risk penalty.
- Explainable Output
  - Gives factor-level explanations such as trend, momentum, volatility, and liquidity.
  - Shows prediction strategy, factor values, directional impact, weights, and reasoning.
- Predicted K-Line Chart
  - Generates historical candlesticks plus future projected candlesticks for indices or stocks.
  - Supports market index and individual stock selection from the web dashboard.
  - Overlays MA5, MA10, MA20, MA60, MA120, and MA250 moving-average curves for short,
    medium, and long-term trend confirmation.
  - Marks historical and predicted buy/sell points with model-generated reasons.
- Sector Opportunity Agent
  - Scans industry boards and ranks potentially strong sectors.
  - Explains sector strength using sector momentum, trading activity, leader stock anchor,
    leader strength, overheating penalty, risk notes, and watch points.
  - Falls back to real-time all-A-share thematic aggregation when board APIs are slow or unavailable.
- Full A-Share Opportunity Scan
  - Uses AKShare real-time A-share snapshot to rank broader stock candidates.
  - Scores candidates by price momentum, volume ratio, turnover participation,
    liquidity quality, valuation constraint, and chasing-risk penalty.
  - Enriches top candidates with K-line trend confirmation and recent fund-flow checks.
- Full A-share Search
  - Searches A-share symbols through AKShare when enabled.
  - Selecting a result automatically adds it to the pool and generates K-line forecast.
- Offline Demo Mode
  - Uses deterministic mock A-share data so judges can run the project without data tokens.

## Suggested Architecture

```text
data providers -> feature pipeline -> forecast models -> agents -> api/dashboard
                                             |
                                      backtest/evaluation
```

## Quick Start

### 1. Create Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
```

If Windows cannot find `python`, use the Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
```

### 2. Run API

```powershell
uvicorn agent_trading.api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
web/index.html
```

The default app uses deterministic mock data so the demo can run without data
vendor credentials.

On Windows, the safest way to avoid using the wrong Python interpreter is:

```powershell
.\scripts\start_api.ps1
```

## Enable Real AKShare Data

The project defaults to offline mock data. To use real A-share market data:

```powershell
python -m pip install -e ".[dev,data]"
Copy-Item .env.example .env
```

Edit `.env`:

```text
DATA_PROVIDER=akshare
```

Then restart the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn agent_trading.api.main:app --reload --host 127.0.0.1 --port 8000
```

The dashboard will use AKShare for historical K-line data, all A-share search,
sector opportunity scanning, and stock opportunity scanning. Some public data
interfaces are unstable; the opportunity agent therefore uses fast real-time
snapshots first and only performs deeper K-line/fund-flow enrichment for the top
candidates. If a board API is unavailable, it falls back to real all-A-share
thematic aggregation rather than showing fake results.

Quick API health check:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/health -UseBasicParsing
```

## Optional Model Dependencies

The runnable default stack is:

```powershell
python -m pip install -e ".[dev,data]"
```

Optional deep-learning model:

```powershell
python -m pip install torch
```

Optional Qlib interface:

```powershell
python -m pip install pyqlib
```

Both PyTorch and Qlib are optional. If they are not installed, the dashboard will
show that the corresponding model is unavailable and continue with the main
AKShare + multi-factor + ML baseline workflow.

## Current Forecast Strategy

The current implementation uses an explainable multi-factor baseline rather than
a pretrained black-box model. It combines momentum, relative strength,
volatility, liquidity/volume confirmation, drawdown risk, and sentiment signals
before a final risk-aware forecast is produced.

Market forecast uses:

- trend momentum
- short-term continuation
- moving-average structure
- moving-average alignment
- MA20 support/resistance
- volatility penalty
- volume confirmation
- drawdown penalty
- news sentiment

Stock forecast uses:

- benchmark-relative strength
- absolute momentum
- short-term continuation
- moving-average structure
- moving-average alignment
- MA20 support/resistance
- volume confirmation
- low-volatility quality
- drawdown control
- news sentiment

The web dashboard displays each factor's standardized score, impact direction,
weight, and reason. Future K-line candles are generated from expected return,
recent volatility, and controlled random disturbance, so they should be
interpreted as a scenario path rather than a guaranteed price forecast.

To improve scientific validity, the forecast also uses historical analog
calibration. For each prediction horizon, the model searches past windows with
similar factor states and uses their realized forward returns to calibrate
expected return, up probability, downside probability, and confidence. This
reduces reliance on hand-written rules and makes the forecast more auditable.

The forecast layer also includes a train-on-history machine learning diagnostic.
It prefers LightGBM, then CatBoost, and falls back to sklearn
HistGradientBoosting if optional ML libraries are not installed. The model is
trained on historical factor rows and tested on the latest chronological holdout
set, then the dashboard displays probability, sample count, holdout accuracy,
AUC, and top features.

An optional GRU deep-learning sequence model is also available. It uses recent
factor sequences to estimate the next 5-day direction when PyTorch is installed.
The web dashboard lets users choose among factor-only, ML-only, GRU-only, and
ensemble modes. If PyTorch is unavailable, the GRU panel reports that it is not
enabled and the rest of the system keeps running.

K-line buy/sell markers use an ML barrier + meta-labeling style signal layer.
The model builds historical features such as returns, EMA trend gap, RSI,
volume-adjusted MACD, ATR volatility, volume ratio, and Donchian distances. It
then labels historical rows using future return/risk barriers and trains a
lightweight sklearn `HistGradientBoostingClassifier`. Signals are filtered by
trend, ATR risk control, Donchian breakout/downside exits, and fund/volume
confirmation. Predicted candles can also receive "预测买/预测卖" markers when
the future scenario path satisfies the same signal rules.

Install PyTorch separately if you want to try the deep model:

```powershell
python -m pip install torch
```

## Why Add A Sentiment Agent

A-share prices are sensitive to news, policy expectations, earnings previews,
shareholder changes, penalties, and public opinion. A pure price-volume model can
miss those event-driven signals. The `SentimentAgent` reads recent financial news
when AKShare is enabled and converts positive/negative headline signals into a
transparent sentiment score. The current keyword model is simple and auditable;
it can later be replaced by Chinese FinBERT or another financial text classifier.

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

Or use the repository check script:

```powershell
.\scripts\check_project.ps1
```

Generate a Qlib Alpha158 + LightGBM workflow config:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/qlib/train -Method Post -ContentType "application/json" -Body '{"run":false}'
```

Qlib integration is optional. Install it with:

```powershell
python -m pip install pyqlib
```

Qlib also requires local market data. Set `QLIB_DATA_DIR` if your data is not in
the default `~/.qlib/qlib_data/cn_data`.

API examples:

```text
GET /forecast/market/000300.SH
GET /forecast/stock/600519.SH?benchmark=000300.SH
GET /symbols/search?q=茅台
GET /chart/prediction/000300.SH?benchmark=000300.SH
GET /opportunities/sectors?limit=8
GET /opportunities/stocks?limit=30
GET /opportunities/stocks/grouped?limit=60
GET /qlib/status
POST /qlib/train
GET /agents/research?benchmark=000300.SH&symbols=600519.SH,000001.SZ,300750.SZ
```

## Project Structure

```text
.
├── config/                  # YAML configuration
├── docs/                    # Architecture and modeling documents
├── src/agent_trading/
│   ├── agents/              # Agent workflow
│   ├── api/                 # FastAPI application
│   ├── backtesting/         # Strategy validation utilities
│   ├── data/                # Data provider interfaces
│   ├── features/            # Feature engineering
│   ├── models/              # Forecasting services
│   ├── pipelines/           # Runnable experiment pipelines
│   └── training/            # Labels, datasets, baseline training
├── tests/                   # Unit tests
└── web/                     # Static demo dashboard
```

## Hackathon Submission Checklist

- [x] Public GitHub repository prepared
- [x] Source code included
- [x] README includes project introduction
- [x] README includes installation guide
- [x] README includes running guide
- [x] Agent core logic included
- [x] Forecasting module included
- [ ] Demo video, 3-5 minutes
- [ ] Project document PDF with background, architecture, pain points, and future plan
- [ ] Multiple meaningful commits during the competition period

## Roadmap

1. Add local cache for AKShare responses to reduce public-interface latency.
2. Train LightGBM/CatBoost models for:
   - index direction and market regime
   - stock excess return probability
   - stock drawdown risk
3. Add SHAP or feature-importance explanations.
4. Add Qlib-based research workflow and backtest reports.
5. Add LangGraph orchestration for multi-agent debate and risk review.

## Future Plan

- Integrate TuShare with local cache as a second real-data source.
- Add A-share filters for ST, suspension, limit-up/limit-down, and low-liquidity names.
- Replace heuristic forecasts with LightGBM/CatBoost models.
- Add SHAP explanations for model decisions.
- Add Qlib-based factor research and backtesting.
- Add LangGraph multi-agent debate workflow.
- Add PDF report export for competition and investor-style research output.

## Disclaimer

This project is for research and hackathon demonstration only. It is not
investment advice.
