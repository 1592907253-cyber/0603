import json
import sys

from agent_trading.agents.workflow import ResearchWorkflow
from agent_trading.pipelines.demo_training import run_demo_training


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if command == "train-demo":
        print(json.dumps(run_demo_training(), ensure_ascii=False, indent=2))
        return

    report = ResearchWorkflow().run_market_and_stock_scan()
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
