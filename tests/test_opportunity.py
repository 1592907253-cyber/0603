from agent_trading.agents.opportunity import OpportunityAgent


def test_mock_sector_opportunities() -> None:
    report = OpportunityAgent("mock").sector_opportunities(limit=2)

    assert report.sectors
    assert report.summary


def test_mock_stock_opportunities() -> None:
    report = OpportunityAgent("mock").stock_opportunities(limit=2)

    assert report.stocks
    assert report.summary
