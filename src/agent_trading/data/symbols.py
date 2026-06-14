from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import monotonic

from agent_trading.data.providers import AkshareDataProvider
from agent_trading.schemas import SymbolInfo


DEFAULT_SYMBOLS = [
    SymbolInfo(symbol="000300.SH", name="沪深300", kind="index"),
    SymbolInfo(symbol="000001.SH", name="上证指数", kind="index"),
    SymbolInfo(symbol="399001.SZ", name="深证成指", kind="index"),
    SymbolInfo(symbol="399006.SZ", name="创业板指", kind="index"),
    SymbolInfo(symbol="600519.SH", name="贵州茅台", kind="stock"),
    SymbolInfo(symbol="000001.SZ", name="平安银行", kind="stock"),
    SymbolInfo(symbol="300750.SZ", name="宁德时代", kind="stock"),
    SymbolInfo(symbol="601318.SH", name="中国平安", kind="stock"),
    SymbolInfo(symbol="600036.SH", name="招商银行", kind="stock"),
    SymbolInfo(symbol="600030.SH", name="中信证券", kind="stock"),
    SymbolInfo(symbol="603993.SH", name="洛阳钼业", kind="stock"),
    SymbolInfo(symbol="002594.SZ", name="比亚迪", kind="stock"),
    SymbolInfo(symbol="000858.SZ", name="五粮液", kind="stock"),
    SymbolInfo(symbol="601899.SH", name="紫金矿业", kind="stock"),
    SymbolInfo(symbol="002475.SZ", name="立讯精密", kind="stock"),
    SymbolInfo(symbol="000625.SZ", name="长安汽车", kind="stock"),
]

_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_AKSHARE_SYMBOL_CACHE: tuple[float, list[SymbolInfo]] | None = None
_CACHE_TTL_SECONDS = 60 * 60 * 6
_SEARCH_TIMEOUT_SECONDS = 2.5


def search_symbols(keyword: str = "", provider_name: str = "mock") -> list[SymbolInfo]:
    fallback = _filter_default_symbols(keyword)
    if provider_name.lower() == "akshare":
        cached = _search_cached_symbols(keyword)
        if cached:
            return _merge_symbol_results(cached, fallback)
        future = _SEARCH_EXECUTOR.submit(_load_akshare_symbols)
        try:
            items = future.result(timeout=_SEARCH_TIMEOUT_SECONDS)
            return _merge_symbol_results(_filter_symbols(items, keyword), fallback)
        except TimeoutError:
            return fallback
        except Exception:
            return fallback

    return fallback


def _filter_default_symbols(keyword: str = "") -> list[SymbolInfo]:
    value = keyword.strip().lower()
    if not value:
        return DEFAULT_SYMBOLS
    return [
        item
        for item in DEFAULT_SYMBOLS
        if value in item.symbol.lower() or value in item.name.lower()
    ]


def _load_akshare_symbols() -> list[SymbolInfo]:
    global _AKSHARE_SYMBOL_CACHE

    now = monotonic()
    if _AKSHARE_SYMBOL_CACHE and now - _AKSHARE_SYMBOL_CACHE[0] < _CACHE_TTL_SECONDS:
        return _AKSHARE_SYMBOL_CACHE[1]

    try:
        import akshare as ak
    except ImportError:
        return DEFAULT_SYMBOLS

    provider = AkshareDataProvider()
    table = provider._load_symbol_table(ak)
    symbols = [
        SymbolInfo(
            symbol=provider._format_stock_symbol(str(row["code"])),
            name=str(row["name"]),
            kind="stock",
        )
        for _, row in table.iterrows()
    ]
    symbols = _merge_symbol_results(DEFAULT_SYMBOLS, symbols, limit=None)
    _AKSHARE_SYMBOL_CACHE = (now, symbols)
    return symbols


def _search_cached_symbols(keyword: str = "") -> list[SymbolInfo]:
    if not _AKSHARE_SYMBOL_CACHE:
        return []
    if monotonic() - _AKSHARE_SYMBOL_CACHE[0] >= _CACHE_TTL_SECONDS:
        return []
    return _filter_symbols(_AKSHARE_SYMBOL_CACHE[1], keyword)


def _filter_symbols(items: list[SymbolInfo], keyword: str = "") -> list[SymbolInfo]:
    value = keyword.strip().lower()
    if not value:
        return items[:50]
    return [
        item
        for item in items
        if value in item.symbol.lower() or value in item.name.lower()
    ][:50]


def _merge_symbol_results(
    primary: list[SymbolInfo],
    fallback: list[SymbolInfo],
    limit: int | None = 50,
) -> list[SymbolInfo]:
    merged: list[SymbolInfo] = []
    seen: set[str] = set()
    for item in [*primary, *fallback]:
        if item.symbol in seen:
            continue
        seen.add(item.symbol)
        merged.append(item)
    return merged if limit is None else merged[:limit]
