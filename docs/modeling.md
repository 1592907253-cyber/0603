# Forecast Modeling Plan

## Targets

Market index:

- `up_1d`, `up_5d`, `up_20d`
- `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_20d`
- regime labels: bull, neutral, bear, high-volatility

Stock:

- `outperform_5d`: future 5-day return beats benchmark
- `outperform_20d`: future 20-day return beats benchmark
- `drawdown_risk_5d`: future 5-day drawdown breaches threshold

## Baseline Models

Start with tree-based models:

- LightGBM or CatBoost for direction and outperformance.
- Logistic regression or LightGBM for drawdown risk.
- Time-series cross validation only; avoid random splits.

## A-Share Specific Filters

Before training or generating signals, filter or flag:

- ST and delisting-risk names
- suspended stocks
- limit-up/limit-down bars
- newly listed stocks with insufficient history
- very low liquidity names

## Demo Output

Each prediction should include:

- probability or expected return
- confidence score
- top positive and negative factors
- risk review
- historical validation metrics
