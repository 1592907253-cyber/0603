from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib

import numpy as np
import pandas as pd

from agent_trading.data.network import akshare_network_context


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str
    start: date
    end: date


class DataProvider(ABC):
    @abstractmethod
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        """Return OHLCV history indexed by trading date."""


class MockAshareDataProvider(DataProvider):
    """Deterministic demo data provider used before vendor integrations are configured."""

    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        days = max((request.end - request.start).days, 80)
        dates = pd.bdate_range(request.end - timedelta(days=days), request.end)
        seed = int(hashlib.sha256(request.symbol.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        drift = 0.0004 if request.symbol.endswith(".SH") else 0.0002
        returns = rng.normal(drift, 0.014, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.random(len(dates)) * 0.012)
        low = np.minimum(open_, close) * (1 - rng.random(len(dates)) * 0.012)
        volume = rng.integers(8_000_000, 50_000_000, len(dates))

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": volume * close,
            },
            index=dates,
        )
        frame.index.name = "date"
        return frame


class AkshareDataProvider(DataProvider):
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "AKShare is not installed. Run: python -m pip install -e \".[data]\""
            ) from exc

        start = request.start.strftime("%Y%m%d")
        end = request.end.strftime("%Y%m%d")
        raw_symbol = request.symbol.split(".")[0]
        with akshare_network_context():
            if self._is_index(request.symbol):
                frame = self._index_history(ak, raw_symbol, start, end)
            else:
                frame = self._stock_history(ak, raw_symbol, start, end)
        return self._normalize_akshare_history(frame)

    def symbol_search(self, keyword: str = "") -> list[dict[str, str]]:
        base_indices = [
            {"symbol": "000300.SH", "name": "沪深300", "kind": "index"},
            {"symbol": "000001.SH", "name": "上证指数", "kind": "index"},
            {"symbol": "399001.SZ", "name": "深证成指", "kind": "index"},
            {"symbol": "399006.SZ", "name": "创业板指", "kind": "index"},
        ]
        try:
            import akshare as ak
        except ImportError:
            return base_indices

        stocks = self._load_symbol_table(ak)
        if keyword:
            value = keyword.lower()
            stocks = stocks[
                stocks["code"].astype(str).str.contains(value, case=False, na=False)
                | stocks["name"].astype(str).str.lower().str.contains(value, na=False)
            ]
        results = base_indices + [
            {
                "symbol": self._format_stock_symbol(str(row["code"])),
                "name": str(row["name"]),
                "kind": "stock",
            }
            for _, row in stocks.head(40).iterrows()
        ]
        return results[:50]

    def _load_symbol_table(self, ak: object) -> pd.DataFrame:
        errors: list[str] = []
        for function_name in ("stock_info_a_code_name", "stock_zh_a_spot_em", "stock_zh_a_spot"):
            function = getattr(ak, function_name, None)
            if function is None:
                continue
            try:
                with akshare_network_context():
                    frame = function()
                table = self._normalize_symbol_table(frame)
                if not table.empty:
                    return table
            except Exception as exc:
                errors.append(f"{function_name}: {exc}")
        raise RuntimeError("AKShare stock symbol search failed: " + " | ".join(errors))

    def _normalize_symbol_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        code_candidates = ("code", "代码", "证券代码", "股票代码", "symbol")
        name_candidates = ("name", "名称", "证券简称", "股票简称", "简称")
        code_column = next((column for column in code_candidates if column in frame.columns), None)
        name_column = next((column for column in name_candidates if column in frame.columns), None)
        if code_column is None or name_column is None:
            raise ValueError(f"Cannot find code/name columns in: {list(frame.columns)}")
        table = frame[[code_column, name_column]].rename(
            columns={code_column: "code", name_column: "name"}
        )
        table["code"] = table["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        table["name"] = table["name"].astype(str).str.strip()
        table = table.dropna(subset=["code", "name"])
        table = table[table["code"].str.len() == 6]
        return table.drop_duplicates(subset=["code"])

    def _index_history(self, ak: object, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            return ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
            )
        except Exception:
            # Some AKShare versions expose index data through this older interface.
            frame = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if symbol.startswith("0") else f"sz{symbol}")
            frame = frame.reset_index().rename(
                columns={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                }
            )
            frame["日期"] = pd.to_datetime(frame["日期"])
            mask = (frame["日期"] >= pd.to_datetime(start)) & (frame["日期"] <= pd.to_datetime(end))
            return frame.loc[mask]

    def _stock_history(self, ak: object, symbol: str, start: str, end: str) -> pd.DataFrame:
        errors: list[str] = []
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
        except Exception as exc:
            errors.append(f"stock_zh_a_hist 失败：{exc}")

        try:
            prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
            frame = ak.stock_zh_a_daily(
                symbol=f"{prefix}{symbol}",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            frame = frame.reset_index().rename(
                columns={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                    "amount": "成交额",
                }
            )
            return frame
        except Exception as exc:
            errors.append(f"stock_zh_a_daily 失败：{exc}")

        raise RuntimeError("；".join(errors))

    def _normalize_akshare_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("AKShare returned empty history.")
        rename_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        data = frame.rename(columns=rename_map)
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"AKShare history missing columns: {missing}")
        if "amount" not in data.columns:
            data["amount"] = data["volume"] * data["close"]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date").sort_index()
        return data[["open", "high", "low", "close", "volume", "amount"]].astype(float)

    def _is_index(self, symbol: str) -> bool:
        return symbol in {"000300.SH", "000001.SH", "399001.SZ", "399006.SZ"}

    def _format_stock_symbol(self, code: str) -> str:
        suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{suffix}"


class TushareDataProvider(DataProvider):
    def history(self, request: MarketDataRequest) -> pd.DataFrame:
        raise NotImplementedError("TuShare adapter placeholder: implement token auth and caching.")


def build_data_provider(name: str) -> DataProvider:
    normalized = name.lower()
    if normalized == "mock":
        return MockAshareDataProvider()
    if normalized == "akshare":
        return AkshareDataProvider()
    if normalized == "tushare":
        return TushareDataProvider()
    raise ValueError(f"Unsupported data provider: {name}")
