from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, MockAshareDataProvider
from agent_trading.models.deep_learning import GRUSequencePredictor


def test_gru_predictor_degrades_gracefully() -> None:
    provider = MockAshareDataProvider()
    end = date(2026, 6, 13)
    history = provider.history(MarketDataRequest("000300.SH", end - timedelta(days=900), end))

    result = GRUSequencePredictor(epochs=1).predict(history)

    assert 0 <= result.probability_up <= 1
    assert result.model_name in {"GRU"}
