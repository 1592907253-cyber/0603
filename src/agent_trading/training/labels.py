import pandas as pd


def make_forward_return_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    data = frame.copy()
    for horizon in horizons:
        forward_return = data["close"].shift(-horizon) / data["close"] - 1
        data[f"fwd_ret_{horizon}d"] = forward_return
        data[f"up_{horizon}d"] = (forward_return > 0).astype(int)
    return data


def make_excess_return_labels(
    stock_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20),
) -> pd.DataFrame:
    stock = stock_frame.copy()
    benchmark = benchmark_frame[["close"]].rename(columns={"close": "benchmark_close"})
    data = stock.join(benchmark, how="inner")
    for horizon in horizons:
        stock_return = data["close"].shift(-horizon) / data["close"] - 1
        benchmark_return = data["benchmark_close"].shift(-horizon) / data["benchmark_close"] - 1
        excess_return = stock_return - benchmark_return
        data[f"excess_ret_{horizon}d"] = excess_return
        data[f"outperform_{horizon}d"] = (excess_return > 0).astype(int)
    return data


def make_drawdown_labels(
    frame: pd.DataFrame,
    horizon: int = 5,
    threshold: float = -0.05,
) -> pd.DataFrame:
    data = frame.copy()
    future_low = pd.concat(
        [data["low"].shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    ).min(axis=1)
    future_drawdown = future_low / data["close"] - 1
    data[f"fwd_drawdown_{horizon}d"] = future_drawdown
    data[f"drawdown_risk_{horizon}d"] = (future_drawdown < threshold).astype(int)
    return data
