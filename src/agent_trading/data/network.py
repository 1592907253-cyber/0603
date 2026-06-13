from collections.abc import Iterator
from contextlib import contextmanager
import os
from typing import Any

from agent_trading.settings import get_settings


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextmanager
def akshare_network_context() -> Iterator[None]:
    """Temporarily bypass process-level proxies for AKShare data requests.

    Some local proxy tools work for GitHub but break Eastmoney endpoints used by
    AKShare. This keeps the behavior configurable through .env.
    """

    settings = get_settings()
    if not settings.akshare_disable_proxy:
        yield
        return

    previous = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    previous_no_proxy = os.environ.get("NO_PROXY")
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"

    try:
        import requests.sessions

        original_merge_environment_settings = requests.sessions.Session.merge_environment_settings

        def merge_without_proxy(
            self: requests.sessions.Session,
            url: str,
            proxies: dict[str, Any] | None,
            stream: bool | None,
            verify: bool | str | None,
            cert: Any,
        ) -> dict[str, Any]:
            settings = original_merge_environment_settings(self, url, {}, stream, verify, cert)
            settings["proxies"] = {}
            return settings

        requests.sessions.Session.merge_environment_settings = merge_without_proxy
    except Exception:
        original_merge_environment_settings = None

    try:
        yield
    finally:
        if original_merge_environment_settings is not None:
            requests.sessions.Session.merge_environment_settings = original_merge_environment_settings
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = previous_no_proxy
