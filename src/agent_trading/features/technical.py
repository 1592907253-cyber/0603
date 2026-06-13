import pandas as pd


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]

    data["ret_1d"] = close.pct_change()
    data["ret_5d"] = close.pct_change(5)
    data["ret_20d"] = close.pct_change(20)
    data["ma_5"] = close.rolling(5).mean()
    data["ma_20"] = close.rolling(20).mean()
    data["ma_60"] = close.rolling(60).mean()
    data["ma_gap_5_20"] = data["ma_5"] / data["ma_20"] - 1
    data["volatility_20"] = data["ret_1d"].rolling(20).std()
    data["volume_ratio_5_20"] = data["volume"].rolling(5).mean() / data["volume"].rolling(20).mean()
    data["drawdown_20"] = close / close.rolling(20).max() - 1
    data["momentum_score"] = data["ret_20d"] - data["volatility_20"].fillna(0)
    return data.dropna()


def latest_feature_snapshot(frame: pd.DataFrame) -> dict[str, float]:
    features = add_technical_features(frame)
    if features.empty:
        return {}
    row = features.iloc[-1]
    return {key: float(value) for key, value in row.items() if pd.notna(value)}
