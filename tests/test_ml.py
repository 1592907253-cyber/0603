from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, MockAshareDataProvider
from agent_trading.models.ml import MLDirectionPredictor


def test_ml_direction_predictor_runs_on_mock_history() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    history = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=1800), end))

    result = MLDirectionPredictor().predict(history)

    assert result is None or 0 <= result.probability_up <= 1
