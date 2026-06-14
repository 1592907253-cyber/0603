# Hackathon Submission

## Basic Information

- Group: 第40组
- Team Name: yes茄豆
- Project Name: AgentTrading
- Repository: https://github.com/1592907253-cyber/0603
- Project Type: Agent + Quantitative Trading + A-share Forecasting

## One-Sentence Introduction

AgentTrading is a multi-agent A-share research assistant that combines market
forecasting, stock outperformance prediction, strategy candidate ranking, risk
review, and explainable reports.

## Problem

A-share market analysis often requires combining market regime, sector rotation,
individual stock signals, liquidity, risk, and backtest validation. These steps
are usually scattered across different tools. This project demonstrates an
agentic workflow that connects these steps into one auditable system.

## Solution

The project uses a layered architecture:

```text
A-share data -> feature engineering -> forecasting model -> research agents -> API/dashboard
```

Current implementation supports:

- mock A-share data for offline judge review
- market index forecast
- stock outperformance forecast
- drawdown risk estimation
- predicted K-line chart for index and stock future movement
- searchable stock pool in the web dashboard
- Chinese research-style analysis output
- factor-level prediction reasons and strategy explanation
- sentiment analysis agent for news-driven signals
- optional AKShare real A-share data provider
- ECharts professional candlestick chart
- sector opportunity scan
- full A-share stock opportunity scan
- all A-share search through AKShare
- ML barrier style buy/sell point markers on historical and predicted K-lines
- real-time fund-flow fallback through Tonghuashun when Eastmoney is unstable
- candidate ranking
- FastAPI endpoints
- static dashboard
- training dataset construction
- frontend/backend separated startup
- professional tabbed dashboard
- A-share Fear Index for market stress and position sizing
- real sector and stock trend charts inside opportunity scan cards
- select-any-stock Agent analysis from search results
- cached all-A-share symbol search with timeout fallback

## How To Run

```powershell
cd E:\AgentTrading
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
.\scripts\start_api.ps1
```

Open a second terminal:

```powershell
.\scripts\start_web.ps1
```

Open:

```text
http://127.0.0.1:5173
```

The dashboard and API are separated. The frontend runs on port `5173`; the
FastAPI backend runs on port `8000`.

## Demo Script

1. Start backend and frontend with `scripts/start_api.ps1` and `scripts/start_web.ps1`.
2. Open `http://127.0.0.1:5173`.
3. Search an A-share stock such as `洛阳钼业` or `603993`.
4. Select it as the analysis target and click "运行 Agent 分析".
5. Show the overview tab:
   - market regime
   - suggested position
   - A-share Fear Index
   - candidate ranking
   - Chinese risk review
6. Open the K-line prediction tab:
   - ECharts candlestick chart
   - MA5/10/20/60/120/250 curves
   - historical and predicted buy/sell markers
   - model consensus and factor explanation
7. Open the opportunity scan tab:
   - scan potential sectors
   - scan all-A-share candidates grouped by sector
   - show real trend charts with high/low/current markers when data is available
8. Open API docs at `http://127.0.0.1:8000/docs`.
9. Show endpoints:
   - `/forecast/market/{symbol}`
   - `/forecast/stock/{symbol}`
   - `/chart/prediction/{symbol}`
   - `/opportunities/sectors`
   - `/opportunities/stocks/grouped`
   - `/symbols/search`
   - `/agents/research`

## This Version's Main Improvements

- More professional frontend design with tabs instead of a long single page.
- Real K-line/price charts in sector and stock opportunity cards.
- A-share Fear Index added to market forecast, risk control, and dashboard.
- Faster and clearer stock search with local fallback.
- Any searched stock can become the current Agent analysis target.
- Frontend/backend separated startup for a more standard project structure.
- Submission documents and changelog updated for GitHub review.

## Commit Requirement Reminder

According to the competition guide, the repository should be public and contain
multiple meaningful commits during the competition period. Avoid uploading the
entire project only once near the deadline. Recommended commit topics:

- project scaffold
- forecasting module
- data provider integration
- dashboard improvement
- README and submission document
- tests and bug fixes

## Materials To Prepare

- GitHub repository link
- 3-5 minute demo video
- PDF project document including:
  - project background
  - pain point
  - technical architecture
  - core Agent workflow
  - future plan
