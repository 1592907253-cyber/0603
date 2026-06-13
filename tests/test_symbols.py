from agent_trading.data.symbols import search_symbols


def test_search_symbols_falls_back_to_default_universe() -> None:
    results = search_symbols("茅台", provider_name="mock")

    assert results
    assert results[0].symbol == "600519.SH"
