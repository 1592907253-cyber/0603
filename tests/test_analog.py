from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, MockAshareDataProvider
from agent_trading.models.analog import HistoricalAnalogForecaster


def test_historical_analog_forecaster_returns_calibration() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    history = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=1200), end))

    result = HistoricalAnalogForecaster().forecast(history)

    assert result
    assert 5 in result
    assert 0 <= result[5].up_probability <= 1
