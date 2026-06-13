from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import yaml

from agent_trading.schemas import QlibStatus, QlibTrainingRequest, QlibTrainingResponse
from agent_trading.settings import get_settings


class QlibService:
    def status(self) -> QlibStatus:
        settings = get_settings()
        installed = importlib.util.find_spec("qlib") is not None
        data_dir = settings.qlib_data_dir
        data_ready = self._data_ready(data_dir)
        if not installed:
            message = "未安装 Qlib。可执行：python -m pip install pyqlib"
        elif not data_ready:
            message = (
                "Qlib 已安装，但未发现可用 cn_data。请先准备 Qlib 中国市场数据，"
                "或设置 QLIB_DATA_DIR 指向已有数据目录。"
            )
        else:
            message = "Qlib 已安装，且检测到本地数据目录。"
        return QlibStatus(
            installed=installed,
            data_dir=str(data_dir),
            data_ready=data_ready,
            message=message,
        )

    def create_or_run_workflow(self, request: QlibTrainingRequest) -> QlibTrainingResponse:
        settings = get_settings()
        settings.qlib_artifact_dir.mkdir(parents=True, exist_ok=True)
        config_path = settings.qlib_artifact_dir / "workflow_alpha158_lightgbm.yaml"
        config = self._workflow_config(request)
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

        command = f"qrun {config_path}"
        status = self.status()
        if not request.run:
            return QlibTrainingResponse(
                status="created",
                message="已生成 Qlib workflow 配置。确认 Qlib 和数据准备完成后，可设置 run=true 运行。",
                config_path=str(config_path),
                command=command,
            )
        if not status.installed:
            return QlibTrainingResponse(
                status="blocked",
                message=status.message,
                config_path=str(config_path),
                command=command,
            )
        if not status.data_ready:
            return QlibTrainingResponse(
                status="blocked",
                message=status.message,
                config_path=str(config_path),
                command=command,
            )

        completed = subprocess.run(
            ["qrun", str(config_path)],
            capture_output=True,
            text=True,
            timeout=60 * 30,
            check=False,
        )
        return QlibTrainingResponse(
            status="completed" if completed.returncode == 0 else "failed",
            message=f"qrun exited with code {completed.returncode}",
            config_path=str(config_path),
            command=command,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
        )

    def _data_ready(self, data_dir: Path) -> bool:
        return data_dir.exists() and any(data_dir.iterdir())

    def _workflow_config(self, request: QlibTrainingRequest) -> dict:
        settings = get_settings()
        data_handler_config = {
            "start_time": request.start_time,
            "end_time": request.end_time,
            "fit_start_time": request.start_time,
            "fit_end_time": request.train_end_time,
            "instruments": request.market,
        }
        port_analysis_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "signal": None,
                    "topk": request.topk,
                    "n_drop": request.n_drop,
                },
            },
            "backtest": {
                "start_time": request.test_start_time,
                "end_time": request.test_end_time,
                "account": 100000000,
                "benchmark": request.benchmark,
                "exchange_kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }
        return {
            "qlib_init": {
                "provider_uri": str(settings.qlib_data_dir),
                "region": "cn",
            },
            "market": request.market,
            "benchmark": request.benchmark,
            "data_handler_config": data_handler_config,
            "port_analysis_config": port_analysis_config,
            "task": {
                "model": {
                    "class": "LGBModel",
                    "module_path": "qlib.contrib.model.gbdt",
                    "kwargs": {
                        "loss": "mse",
                        "colsample_bytree": 0.8879,
                        "learning_rate": 0.05,
                        "subsample": 0.8789,
                        "lambda_l1": 205.6999,
                        "lambda_l2": 580.9768,
                        "max_depth": 8,
                        "num_leaves": 210,
                        "num_threads": 8,
                    },
                },
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158",
                            "module_path": "qlib.contrib.data.handler",
                            "kwargs": data_handler_config,
                        },
                        "segments": {
                            "train": [request.start_time, request.train_end_time],
                            "valid": [request.valid_start_time, request.valid_end_time],
                            "test": [request.test_start_time, request.test_end_time],
                        },
                    },
                },
                "record": [
                    {
                        "class": "SignalRecord",
                        "module_path": "qlib.workflow.record_temp",
                        "kwargs": {"model": None, "dataset": None},
                    },
                    {
                        "class": "SigAnaRecord",
                        "module_path": "qlib.workflow.record_temp",
                        "kwargs": {"ana_long_short": False, "ann_scaler": 252},
                    },
                    {
                        "class": "PortAnaRecord",
                        "module_path": "qlib.workflow.record_temp",
                        "kwargs": {"config": port_analysis_config},
                    },
                ],
            },
        }
