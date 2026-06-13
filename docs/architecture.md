# Architecture

## Layers

1. Data providers
   - `MockAshareDataProvider` for offline demos.
   - `AkshareDataProvider` and `TushareDataProvider` placeholders for real A-share data.
2. Feature engineering
   - Technical features today.
   - Add capital flow, fundamentals, industry, and sentiment features later.
3. Forecast models
   - Baseline heuristic model today.
   - LightGBM/CatBoost models planned for index direction, market regime, stock alpha, and drawdown risk.
4. Agent workflow
   - Market forecast agent.
   - Stock forecast agent.
   - Risk review agent.
   - Report synthesis agent.
5. API and dashboard
   - FastAPI endpoints for external integration.
   - Static dashboard for hackathon demo.

## Recommended Hackathon Story

The system does not directly claim to predict exact prices. It predicts tradable
states:

- market regime
- index direction
- probability of stock outperformance
- probability of drawdown risk

Each prediction is converted into portfolio guidance only after risk review and
backtest validation.
