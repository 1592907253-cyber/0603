from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, MockAshareDataProvider
from agent_trading.training.dataset import build_market_direction_dataset


def test_market_direction_dataset_has_samples() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    history = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=900), end))

    x, y = build_market_direction_dataset(history)

    assert len(x) == len(y)
    assert len(x) > 100
