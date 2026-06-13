from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str
    start: date
    end: date


class DataProvider(ABC):
    @abstractmethod
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        """Return OHLCV history indexed by trading date."""


class MockAshareDataProvider(DataProvider):
    """Deterministic demo data provider used before vendor integrations are configured."""

    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        days = max((request.end - request.start).days, 80)
        dates = pd.bdate_range(request.end - timedelta(days=days), request.end)
        seed = int(hashlib.sha256(request.symbol.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        drift = 0.0004 if request.symbol.endswith(".SH") else 0.0002
        returns = rng.normal(drift, 0.014, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.random(len(dates)) * 0.012)
        low = np.minimum(open_, close) * (1 - rng.random(len(dates)) * 0.012)
        volume = rng.integers(8_000_000, 50_000_000, len(dates))

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": volume * close,
            },
            index=dates,
        )
        frame.index.name = "date"
        return frame


class AkshareDataProvider(DataProvider):
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        raise NotImplementedError("AKShare adapter placeholder: implement symbol mapping and caching.")


class TushareDataProvider(DataProvider):
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        raise NotImplementedError("TuShare adapter placeholder: implement token auth and caching.")


def build_data_provider(name: str) -> DataProvider:
    normalized = name.lower()
    if normalized == "mock":
        return MockAshareDataProvider()
    if normalized == "akshare":
        return AkshareDataProvider()
    if normalized == "tushare":
        return TushareDataProvider()
    raise ValueError(f"Unsupported data provider: {name}")
