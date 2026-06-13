from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    data_provider: str = "mock"
    tushare_token: str = ""
    model_dir: Path = Path("artifacts/models")
    cache_dir: Path = Path("artifacts/cache")
    akshare_disable_proxy: bool = True
    qlib_data_dir: Path = Path("~/.qlib/qlib_data/cn_data").expanduser()
    qlib_artifact_dir: Path = Path("artifacts/qlib")
    qlib_experiment_name: str = "agent_trading_qlib_alpha158_lgb"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
