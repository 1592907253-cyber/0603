import pandas as pd


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]

    data["ret_1d"] = close.pct_change()
    data["ret_5d"] = close.pct_change(5)
    data["ret_20d"] = close.pct_change(20)
    data["ma_5"] = close.rolling(5).mean()
    data["ma_10"] = close.rolling(10).mean()
    data["ma_20"] = close.rolling(20).mean()
    data["ma_30"] = close.rolling(30).mean()
    data["ma_60"] = close.rolling(60).mean()
    data["ma_120"] = close.rolling(120).mean()
    data["ma_250"] = close.rolling(250).mean()
    for column in ("ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_120", "ma_250"):
        if data[column].isna().all():
            data[column] = close.expanding().mean()
        data[column] = data[column].bfill().ffill()
    data["ma_gap_5_20"] = data["ma_5"] / data["ma_20"] - 1
    data["ma_gap_10_20"] = data["ma_10"] / data["ma_20"] - 1
    data["ma_gap_20_60"] = data["ma_20"] / data["ma_60"] - 1
    data["price_ma20_gap"] = close / data["ma_20"] - 1
    data["ma_alignment_score"] = (
        (data["ma_5"] > data["ma_10"]).astype(int)
        + (data["ma_10"] > data["ma_20"]).astype(int)
        + (data["ma_20"] > data["ma_60"]).astype(int)
    ) / 3
    data["volatility_20"] = data["ret_1d"].rolling(20).std()
    data["volume_ratio_5_20"] = data["volume"].rolling(5).mean() / data["volume"].rolling(20).mean()
    data["drawdown_20"] = close / close.rolling(20).max() - 1
    data["momentum_score"] = data["ret_20d"] - data["volatility_20"].fillna(0)
    feature_columns = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "volatility_20",
        "volume_ratio_5_20",
        "drawdown_20",
        "momentum_score",
    ]
    data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
    data[feature_columns] = data[feature_columns].bfill().ffill()
    return data.dropna(subset=["ret_1d", "ret_5d", "ret_20d", "volatility_20"])


def latest_feature_snapshot(frame: pd.DataFrame) -> dict[str, float]:
    features = add_technical_features(frame)
    if features.empty:
        return {}
    row = features.iloc[-1]
    return {key: float(value) for key, value in row.items() if pd.notna(value)}
