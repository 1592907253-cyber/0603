from datetime import date, timedelta

from agent_trading.data.providers import MarketDataRequest, build_data_provider
from agent_trading.settings import get_settings
from agent_trading.training.dataset import build_market_direction_dataset
from agent_trading.training.train_baseline import evaluate_time_series_classifier


def run_demo_training(symbol: str = "000300.SH") -> dict[str, float | int | None]:
    settings = get_settings()
    provider = build_data_provider(settings.data_provider)
    end = date.today()
    start = end - timedelta(days=900)
    history = provider.history(MarketDataRequest(symbol, start, end))
    x, y = build_market_direction_dataset(history)
    result = evaluate_time_series_classifier(x, y)
    return {
        "accuracy": result.accuracy,
        "roc_auc": result.roc_auc,
        "n_samples": result.n_samples,
    }
