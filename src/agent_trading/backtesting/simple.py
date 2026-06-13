import pandas as pd


def long_only_signal_backtest(
    prices: pd.Series,
    signal: pd.Series,
    fee_rate: float = 0.0005,
) -> pd.DataFrame:
    data = pd.DataFrame({"price": prices, "signal": signal}).dropna()
    data["position"] = data["signal"].clip(0, 1).shift(1).fillna(0)
    data["return"] = data["price"].pct_change().fillna(0)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"])
    data["strategy_return"] = data["position"] * data["return"] - data["turnover"] * fee_rate
    data["equity"] = (1 + data["strategy_return"]).cumprod()
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1
    return data


def summarize_backtest(result: pd.DataFrame) -> dict[str, float]:
    if result.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    total_return = float(result["equity"].iloc[-1] - 1)
    max_drawdown = float(result["drawdown"].min())
    win_rate = float((result["strategy_return"] > 0).mean())
    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
