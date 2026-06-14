# Changelog

## 2026-06-14 - Hackathon Enhanced Version

This version upgrades AgentTrading from a basic research scaffold into a more
complete A-share intelligent research terminal.

### Added

- Frontend/backend separated startup:
  - Backend API: `scripts/start_api.ps1`
  - Frontend web app: `scripts/start_web.ps1`
- Professional tabbed dashboard:
  - Overview
  - K-line prediction
  - Opportunity scan
  - Model diagnostics
- A-share Fear Index:
  - VIX-inspired 0-100 market stress proxy
  - Uses realized volatility, drawdown, downside pressure, volume anomaly, and MA20 breakdown
  - Feeds into market regime, position sizing, risk review, and factor explanations
- Real chart-based opportunity display:
  - Sector cards prefer real board/industry index trend lines with high/low/current markers
  - Stock cards prefer real historical stock trend lines with MA20/MA60 and high/low/current markers
  - Proxy strength charts are used only when real K-line data is unavailable
- All-A-share search performance improvements:
  - AKShare symbol table cache
  - Fast timeout fallback
  - Immediate local search feedback
- Analysis target selector:
  - Search or add any A-share stock, then run Agent analysis for the selected stock
- Opportunity scan explanations:
  - Sector K-line trend
  - Thematic aggregation fallback
  - Individual stock K-line confirmation
  - Fund-flow fallback and risk notes

### Improved

- Dashboard UI redesigned as a professional A-share research terminal.
- Candidate scan grouped by sector and displayed with charts instead of long tables.
- Prediction chart retains ECharts candlesticks, MA5/10/20/60/120/250, and buy/sell markers.
- Forecast explanations include fear index and more complete risk context.
- Check script validates frontend entry and startup script before running tests.

### Verified

- `python -m pytest` passes with 12 tests.
- Frontend static server returns `http://127.0.0.1:5173`.
