from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    data_provider: str = "mock"
    tushare_token: str = ""
    model_dir: Path = Path("artifacts/models")
    cache_dir: Path = Path("artifacts/cache")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
