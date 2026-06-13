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
- candidate ranking
- FastAPI endpoints
- static dashboard
- training dataset construction

## How To Run

```powershell
cd E:\AgentTrading
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn agent_trading.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
web/index.html
```

## Demo Script

1. Open the dashboard.
2. Keep benchmark as `000300.SH`.
3. Enter stock pool: `600519.SH,000001.SZ,300750.SZ`.
4. Click "运行 Agent 分析".
5. Show market regime, suggested position, candidate ranking, summary, and risk review.
6. Open API docs at `http://127.0.0.1:8000/docs`.
7. Show endpoints:
   - `/forecast/market/{symbol}`
   - `/forecast/stock/{symbol}`
   - `/agents/research`

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
