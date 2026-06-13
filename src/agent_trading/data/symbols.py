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


def search_symbols(keyword: str = "", provider_name: str = "mock") -> list[SymbolInfo]:
    if provider_name.lower() == "akshare":
        try:
            items = AkshareDataProvider().symbol_search(keyword)
            return [SymbolInfo(**item) for item in items]
        except Exception:
            pass

    value = keyword.strip().lower()
    if not value:
        return DEFAULT_SYMBOLS
    return [
        item
        for item in DEFAULT_SYMBOLS
        if value in item.symbol.lower() or value in item.name.lower()
    ]
