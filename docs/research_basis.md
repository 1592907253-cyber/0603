# Research Basis

This project should not rely on one or two technical indicators. The forecasting
layer is organized as a transparent multi-factor baseline.

## Design Principles

1. Multi-factor instead of single-indicator prediction.
2. Forecast states and scenario paths, not guaranteed exact prices.
3. Make every forecast risk-aware.
4. Keep factor scores, weights, and explanations visible in the UI.

## Factor Groups

Market index:

- trend momentum
- short-term continuation
- moving-average structure
- moving-average alignment
- MA20 support/resistance
- volatility penalty
- volume confirmation
- drawdown penalty
- news sentiment

Individual stock:

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

## Sentiment Agent

The sentiment agent is necessary because A-share stocks can react sharply to
policy news, earnings previews, shareholder changes, investigations, penalties,
and public opinion. Price-volume data often reflects these events with a lag.

Current version:

- uses AKShare news when available
- applies transparent Chinese financial keyword scoring
- returns score, label, confidence, summary, and headline samples

Future version:

- replace keyword scoring with a Chinese financial sentiment model
- classify announcements, policy news, and social media separately
- add event-risk labels such as penalty, investigation, earnings warning, and buyback

## References To Discuss In Presentation

- Fama-French style multi-factor thinking: market, value, size, profitability,
  and investment factors.
- Carhart-style momentum extension.
- Common quantitative risk dimensions: volatility, drawdown, liquidity, and exposure control.
- Qlib-style workflow: data, alpha factors, model training, backtesting, risk modeling,
  and portfolio construction.
