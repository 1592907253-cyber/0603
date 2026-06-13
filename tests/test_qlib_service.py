from agent_trading.qlib_integration.service import QlibService
from agent_trading.schemas import QlibTrainingRequest


def test_qlib_service_generates_config() -> None:
    response = QlibService().create_or_run_workflow(QlibTrainingRequest(run=False))

    assert response.status == "created"
    assert response.config_path.endswith(".yaml")
    assert "qrun" in response.command
