from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, MockAshareDataProvider
from agent_trading.models.forecast import HeuristicForecastModel


def test_market_forecast_returns_valid_payload() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    history = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=260), end))

    result = HeuristicForecastModel().market_forecast("000300.SH", history)

    assert result.symbol == "000300.SH"
    assert 0 <= result.suggested_position <= 1
    assert result.forecasts


def test_stock_forecast_returns_valid_action() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    stock = provider.history(MarketDataRequest("600519.SH", end - timedelta(days=260), end))
    benchmark = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=260), end))

    result = HeuristicForecastModel().stock_forecast("600519.SH", stock, benchmark)

    assert result.action in {"buy", "hold", "avoid"}
    assert 0 <= result.outperform_probability <= 1
    assert 0 <= result.drawdown_risk <= 1
