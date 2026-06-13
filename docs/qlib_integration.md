# Qlib Integration

This project includes an optional Qlib training interface.

## What It Adds

- Alpha158 feature handler
- LightGBM model workflow
- TopK + Dropout strategy backtest
- Signal analysis and portfolio analysis records

## Why Qlib

Qlib is an AI-oriented quantitative investment platform from Microsoft. Its
workflow covers data processing, model training, backtesting, risk modeling,
portfolio optimization, and execution-oriented research.

## Requirements

```powershell
python -m pip install pyqlib
```

Qlib also needs local market data. The default path is:

```text
~/.qlib/qlib_data/cn_data
```

You can override it:

```text
QLIB_DATA_DIR=E:\qlib_data\cn_data
```

## API

Check status:

```text
GET /qlib/status
```

Generate workflow config:

```text
POST /qlib/train
```

Body:

```json
{
  "market": "csi300",
  "benchmark": "SH000300",
  "run": false
}
```

Set `run=true` only after Qlib and local data are ready.
