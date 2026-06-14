from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agent_trading.data.providers import AkshareDataProvider, MarketDataRequest
from agent_trading.data.network import akshare_network_context
from agent_trading.schemas import (
    FactorContribution,
    SectorOpportunity,
    SectorOpportunityReport,
    SectorTrendPoint,
    SectorStockGroup,
    SectorStockOpportunityReport,
    StockTrendPoint,
    StockOpportunity,
    StockOpportunityReport,
)
from datetime import date, timedelta


HOT_SECTOR_KEYWORDS = (
    "通信",
    "算力",
    "半导体",
    "芯片",
    "人工智能",
    "机器人",
    "证券",
    "有色",
    "稀土",
    "新能源",
    "汽车",
)

SECTOR_SOURCE_ALIASES = {
    "通信": ("通信设备", "通信服务"),
    "算力": ("通信设备", "计算机设备", "软件开发", "IT服务"),
    "半导体": ("半导体", "元件", "消费电子"),
    "芯片": ("半导体", "元件"),
    "人工智能": ("软件开发", "IT服务", "计算机设备"),
    "机器人": ("自动化设备", "通用设备", "专用设备"),
    "证券": ("证券",),
    "有色金属": ("工业金属", "小金属", "贵金属", "能源金属"),
    "有色": ("工业金属", "小金属", "贵金属", "能源金属"),
    "稀土": ("小金属",),
    "新能源": ("电池", "光伏设备", "风电设备", "能源金属"),
    "汽车": ("汽车整车", "汽车零部件", "乘用车"),
    "银行": ("银行",),
}


def _number(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        text = str(value).replace("%", "").replace(",", "").strip()
        return float(text)
    except Exception:
        return None


def _first_existing(row: pd.Series, names: tuple[str, ...]) -> object:
    for name in names:
        if name in row.index:
            return row[name]
    return None


def _impact(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _bounded_score(value: float) -> float:
    return min(max(value, -1), 1)


def _sector_news_terms(sector_name: str) -> tuple[str, ...]:
    aliases = {
        "通信": ("通信", "5G", "6G", "光模块", "CPO", "卫星互联网"),
        "算力": ("算力", "数据中心", "服务器", "液冷", "AI算力", "英伟达"),
        "半导体": ("半导体", "芯片", "晶圆", "封测", "存储"),
        "芯片": ("芯片", "半导体", "晶圆", "封测", "存储"),
        "人工智能": ("人工智能", "AI", "大模型", "机器人", "智能体"),
        "机器人": ("机器人", "人形机器人", "减速器", "伺服"),
        "证券": ("证券", "券商", "资本市场", "并购重组"),
        "有色": ("有色", "铜", "铝", "锂", "钴", "稀土"),
        "稀土": ("稀土", "永磁", "镨钕", "磁材"),
        "新能源": ("新能源", "光伏", "储能", "锂电", "风电"),
        "汽车": ("汽车", "新能源车", "智能驾驶", "车路云"),
    }
    terms = {sector_name}
    for key, values in aliases.items():
        if key in sector_name or sector_name in key:
            terms.update(values)
    compact = sector_name.replace("板块", "").replace("行业", "").replace("概念", "")
    if compact and compact != sector_name:
        terms.add(compact)
    return tuple(term for term in terms if term)


def _rating(score: float) -> str:
    if score >= 0.65:
        return "高潜力"
    if score >= 0.45:
        return "中等偏强"
    if score >= 0.25:
        return "观察"
    return "谨慎"


@dataclass
class OpportunityAgent:
    provider_name: str = "mock"
    _fund_flow_snapshot: pd.DataFrame | None = None
    _industry_flow_snapshot: pd.DataFrame | None = None

    def sector_opportunities(self, limit: int = 8) -> SectorOpportunityReport:
        if self.provider_name.lower() != "akshare":
            return self._mock_sectors(limit)

        errors: list[str] = []
        frame = self._load_sector_frame(errors)
        if frame is None or frame.empty:
            fallback = self._sector_opportunities_from_stock_snapshot(limit=limit, errors=errors)
            if fallback.sectors:
                return fallback
            return SectorOpportunityReport(
                summary="AKShare 行业板块数据获取失败，未使用离线演示数据。错误信息：" + " | ".join(errors),
                methodology="行业板块主接口和备用接口均不可用，因此无法生成真实板块扫描结果。",
                sectors=[],
            )

        sectors = [self._sector_from_row(row) for _, row in frame.iterrows()]
        sectors = [item for item in sectors if item.name]
        sectors.sort(key=lambda item: item.score, reverse=True)
        sectors = [self._fast_enrich_sector(item) for item in sectors[: max(limit, 12)]]
        sectors = self._ensure_hot_sector_visibility(sectors)
        sectors.sort(key=lambda item: item.score, reverse=True)
        sectors = [
            self._attach_sector_trend_points(item) if index < min(limit, 8) else item
            for index, item in enumerate(sectors)
        ]
        return SectorOpportunityReport(
            summary="基于板块动量、交易活跃度、龙头锚点、龙头强度、主题热度和过热风险筛选潜力板块。",
            methodology="评分不是单看涨幅，而是同时考察资金是否参与、龙头是否明确、强度是否扩散，以及是否已经过热。",
            sectors=sectors[:limit],
        )

    def _attach_sector_trend_points(self, sector: SectorOpportunity) -> SectorOpportunity:
        if sector.trend_points:
            return sector
        _, trend_summary, trend_points, trend_reasons, trend_risks = self._sector_trend_analysis(sector.name)
        if not trend_points:
            return sector
        return sector.model_copy(
            update={
                "trend_summary": trend_summary,
                "trend_points": trend_points,
                "reasons": [*sector.reasons, *trend_reasons],
                "risks": [*sector.risks, *trend_risks],
            }
        )

    def stock_opportunities(self, limit: int = 30) -> StockOpportunityReport:
        if self.provider_name.lower() != "akshare":
            return self._mock_stocks(limit)

        errors: list[str] = []
        frame = self._load_stock_frame(errors)
        if frame is None or frame.empty:
            return StockOpportunityReport(
                summary="AKShare 全 A 股实时行情获取失败，未使用离线演示数据。错误信息：" + " | ".join(errors),
                methodology="全A实时行情主接口和备用接口均不可用，因此无法生成真实股票扫描结果。",
                stocks=[],
            )

        stocks: list[StockOpportunity] = []
        for _, row in frame.iterrows():
            stock = self._stock_from_row(row)
            if stock is not None:
                stocks.append(stock)
        stocks.sort(key=lambda item: item.score, reverse=True)
        candidates = [self._fast_enrich_stock(stock) for stock in stocks[: max(limit, 30)]]
        candidates.sort(key=lambda item: item.score, reverse=True)
        deep_symbols = {stock.symbol for stock in candidates[: min(limit, 8)]}
        stocks = [
            self._enrich_stock(stock) if stock.symbol in deep_symbols else stock
            for stock in candidates
        ]
        stocks.sort(key=lambda item: item.score, reverse=True)
        return StockOpportunityReport(
            summary="基于全 A 股实时行情、K线趋势、板块主题、资金流、流动性和风险约束筛选候选股票。",
            methodology="先用全市场实时快照做粗筛，再对候选股补充历史K线趋势和资金流验证；结果用于生成观察池，不等同于直接买入建议。",
            stocks=stocks[:limit],
        )

    def grouped_stock_opportunities(self, limit: int = 60) -> SectorStockOpportunityReport:
        stock_report = self.stock_opportunities(limit=limit)
        if not stock_report.stocks:
            return SectorStockOpportunityReport(
                summary=stock_report.summary,
                methodology=stock_report.methodology,
                groups=[],
            )

        groups: dict[str, list[StockOpportunity]] = {}
        for stock in stock_report.stocks:
            groups.setdefault(self._infer_sector(stock.name), []).append(stock)

        grouped = [
            SectorStockGroup(
                sector=sector,
                sector_score=round(sum(stock.score for stock in items) / max(len(items), 1), 4),
                sector_reasons=[
                    f"该组有 {len(items)} 只候选股进入观察池。",
                    "分组优先使用股票名称和主题关键词，避免板块行情接口阻塞全A扫描。",
                    "后续可结合板块指数K线、新闻和资金流做深度复核。",
                ],
                stocks=sorted(items, key=lambda item: item.score, reverse=True)[:8],
            )
            for sector, items in groups.items()
        ]
        grouped.sort(key=lambda item: (item.sector_score, max(stock.score for stock in item.stocks)), reverse=True)
        return SectorStockOpportunityReport(
            summary="将全 A 股候选股票按板块/主题归类展示，便于观察同一方向内的个股扩散。",
            methodology="优先使用板块机会评分排序；若实时行情缺少行业字段，则用股票名称和主题关键词进行兜底归类。",
            groups=grouped[:12],
        )

    def _sector_opportunities_from_stock_snapshot(
        self,
        limit: int,
        errors: list[str],
    ) -> SectorOpportunityReport:
        stock_frame = self._load_stock_frame(errors)
        if stock_frame is None or stock_frame.empty:
            return SectorOpportunityReport(
                summary="行业板块接口不可用，且全A实时行情也无法用于板块兜底聚合。",
                methodology="未使用离线演示数据；请稍后重试 AKShare 或检查网络连接。",
                sectors=[],
            )
        buckets: dict[str, list[StockOpportunity]] = {}
        for _, row in stock_frame.iterrows():
            stock = self._stock_from_row(row)
            if stock is None:
                continue
            buckets.setdefault(self._infer_sector(stock.name), []).append(stock)

        sectors: list[SectorOpportunity] = []
        for sector, stocks in buckets.items():
            if sector == "其他" or not stocks:
                continue
            top = sorted(stocks, key=lambda item: item.score, reverse=True)[:8]
            avg_score = sum(item.score for item in top) / len(top)
            avg_change = self._average([item.change_pct for item in top])
            avg_turnover = self._average([item.turnover_rate for item in top])
            leader = top[0]
            trend_summary, trend_points, trend_reasons, trend_risks = self._sector_trend_from_ths_alias(sector)
            flow_summary, flow_score, flow_reasons, flow_risks = self._sector_flow_from_ths_alias(sector)
            if flow_score is not None:
                avg_score += flow_score * 0.12
            reasons = [
                f"东方财富行业板块主接口不可用，改用全A实时行情中 {len(stocks)} 只同主题股票聚合判断。",
                f"组内高分代表为 {leader.name}，当前评分 {leader.score:.3f}。",
                *trend_reasons,
                *flow_reasons,
            ]
            if avg_change is not None:
                reasons.append(f"组内候选平均涨跌幅约 {avg_change:.2f}%。")
            sectors.append(
                SectorOpportunity(
                    name=sector,
                    score=round(avg_score, 4),
                    rating=_rating(avg_score),
                    change_pct=avg_change,
                    turnover_rate=avg_turnover,
                    leader=leader.name,
                    leader_change_pct=leader.change_pct,
                    trend_summary=trend_summary,
                    trend_points=trend_points,
                    breadth_summary=f"兜底聚合：全A实时行情中识别到 {len(stocks)} 只同主题股票。",
                    news_summary=flow_summary,
                    signal_breakdown=[
                        self._factor("同主题个股强度", avg_score, 0.5, "用同主题候选股平均评分代理板块强度。"),
                        self._factor("龙头带动", leader.score, 0.3, f"组内代表股票为 {leader.name}。"),
                        self._factor("行业资金流", flow_score or 0.0, 0.12, flow_summary),
                    ],
                    reasons=reasons,
                    risks=[
                        *trend_risks,
                        *flow_risks,
                        "这是东方财富板块接口失败后的多源兜底结果，仍需结合后续行情继续确认。",
                    ],
                    watch_points=[
                        "观察同主题候选股是否继续集体上榜。",
                        "观察组内代表股票是否持续放量并维持相对强势。",
                    ],
                )
            )
        sectors.sort(key=lambda item: item.score, reverse=True)
        return SectorOpportunityReport(
            summary="东方财富行业板块接口暂不可用，已改用全A实时行情、同花顺行业指数和行业资金流生成板块机会。",
            methodology="先扫描真实全A实时行情，再按主题聚合；若可匹配同花顺行业名，则补充行业指数K线和行业资金流。",
            sectors=sectors[:limit],
        )

    def _sector_aliases(self, sector: str) -> tuple[str, ...]:
        aliases = {sector}
        for key, values in SECTOR_SOURCE_ALIASES.items():
            if key in sector or sector in key:
                aliases.update(values)
        return tuple(aliases)

    def _sector_trend_from_ths_alias(self, sector: str) -> tuple[str, list[SectorTrendPoint], list[str], list[str]]:
        try:
            import akshare as ak

            end = date.today().strftime("%Y%m%d")
            start = (date.today() - timedelta(days=520)).strftime("%Y%m%d")
            errors: list[str] = []
            for alias in self._sector_aliases(sector):
                try:
                    with akshare_network_context():
                        frame = ak.stock_board_industry_index_ths(symbol=alias, start_date=start, end_date=end)
                    if frame is None or frame.empty:
                        continue
                    data = frame.rename(columns={"收盘价": "close", "成交量": "volume"})
                    close = pd.to_numeric(data["close"], errors="coerce").dropna()
                    volume = pd.to_numeric(data["volume"], errors="coerce").dropna()
                    if len(close) < 30:
                        continue
                    ma20 = close.rolling(20).mean().iloc[-1]
                    ret20 = close.iloc[-1] / close.iloc[-20] - 1
                    vol_ratio = volume.tail(5).mean() / volume.tail(20).mean() if len(volume) >= 20 else 1.0
                    summary = f"同花顺行业指数K线({alias})：20日收益 {ret20:.2%}，价格相对MA20 {'偏强' if close.iloc[-1] > ma20 else '偏弱'}，量能比 {vol_ratio:.2f}。"
                    points = self._sector_trend_points(data, source=alias)
                    reasons = [summary] if close.iloc[-1] > ma20 or ret20 > 0 else []
                    risks = [] if reasons else [summary]
                    return summary, points, reasons, risks
                except Exception as exc:
                    errors.append(f"{alias}: {exc}")
            return (
                "板块指数K线：同花顺行业指数未匹配成功，使用同主题候选股实时表现代理趋势。",
                [],
                [],
                ["未能匹配到可用同花顺行业指数K线。"],
            )
        except Exception as exc:
            return f"板块指数K线暂不可用：{exc}", [], [], ["未能拉取板块指数K线。"]

    def _sector_flow_from_ths_alias(self, sector: str) -> tuple[str, float | None, list[str], list[str]]:
        try:
            import akshare as ak

            if self._industry_flow_snapshot is None:
                with akshare_network_context():
                    self._industry_flow_snapshot = ak.stock_fund_flow_industry(symbol="即时")
            frame = self._industry_flow_snapshot
            industry_col = self._find_column(frame, ("行业", "板块", "名称"))
            net_col = self._find_column(frame, ("净额", "主力净额"))
            change_col = self._find_column(frame, ("行业-涨跌幅", "涨跌幅"))
            leader_col = self._find_column(frame, ("领涨股",))
            if industry_col is None or net_col is None:
                return "行业资金流：同花顺资金流字段不完整。", None, [], ["行业资金流字段不完整。"]
            aliases = self._sector_aliases(sector)
            data = frame.copy()
            matched = data[data[industry_col].astype(str).apply(lambda value: any(alias in value or value in alias for alias in aliases))]
            if matched.empty:
                return "行业资金流：未匹配到同花顺行业资金流，建议结合新闻和成分股二次确认。", None, [], []
            row = matched.iloc[0]
            net = _number(row.get(net_col))
            change = _number(row.get(change_col)) if change_col else None
            leader = str(row.get(leader_col)) if leader_col else "暂无"
            score = _bounded_score((net or 0) / 40)
            summary = f"同花顺行业资金流({row[industry_col]})：净额 {net:.2f}亿元" if net is not None else f"同花顺行业资金流({row[industry_col]})：净额暂无"
            if change is not None:
                summary += f"，行业涨跌幅 {change:.2f}%"
            summary += f"，领涨股 {leader}。"
            reasons = [summary] if net is not None and net > 0 else []
            risks = [summary] if net is not None and net < 0 else []
            return summary, score, reasons, risks
        except Exception as exc:
            return f"行业资金流暂不可用：{exc}", None, [], ["未能拉取同花顺行业资金流。"]

    def _load_sector_frame(self, errors: list[str]) -> pd.DataFrame | None:
        try:
            import akshare as ak

            loaders = [
                ("stock_board_industry_name_em", getattr(ak, "stock_board_industry_name_em", None)),
            ]
            for name, function in loaders:
                if function is None:
                    errors.append(f"AKShare 未提供 {name}")
                    continue
                try:
                    with akshare_network_context():
                        raw = function()
                    normalized = self._normalize_sector_frame(raw)
                    if not normalized.empty:
                        return normalized.drop_duplicates(subset=["sector"])
                except Exception as exc:
                    errors.append(f"{name} 失败：{exc}")
        except Exception as exc:
            errors.append(f"导入 AKShare 失败：{exc}")
        return None

    def _normalize_sector_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        sector_col = self._find_column(data, ("板块名称", "板块", "行业", "概念名称", "名称", "name"))
        if sector_col is None:
            raise ValueError(f"未找到板块名称列：{list(data.columns)}")
        normalized = pd.DataFrame({"sector": data[sector_col].astype(str)})
        normalized["change_pct"] = self._optional_column(data, ("涨跌幅", "涨幅", "涨跌幅%", "change_pct"))
        normalized["turnover_rate"] = self._optional_column(data, ("换手率", "换手", "turnover_rate"))
        normalized["leader"] = self._optional_column(data, ("领涨股票", "领涨股", "龙头股", "股票", "leader"))
        normalized["leader_change_pct"] = self._optional_column(data, ("领涨股票-涨跌幅", "领涨股涨跌幅", "领涨股-涨跌幅"))
        normalized = normalized[normalized["sector"].str.len() > 0]
        return normalized

    def _load_stock_frame(self, errors: list[str]) -> pd.DataFrame | None:
        try:
            import akshare as ak

            for name in ("stock_zh_a_spot_em", "stock_zh_a_spot"):
                function = getattr(ak, name, None)
                if function is None:
                    errors.append(f"AKShare 未提供 {name}")
                    continue
                try:
                    with akshare_network_context():
                        raw = function()
                    normalized = self._normalize_stock_frame(raw)
                    if not normalized.empty:
                        return normalized
                except Exception as exc:
                    errors.append(f"{name} 失败：{exc}")
        except Exception as exc:
            errors.append(f"导入 AKShare 失败：{exc}")
        return None

    def _normalize_stock_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        code_col = self._find_column(data, ("代码", "code", "symbol"))
        name_col = self._find_column(data, ("名称", "name"))
        if code_col is None or name_col is None:
            raise ValueError(f"未找到股票代码或名称列：{list(data.columns)}")
        normalized = pd.DataFrame(
            {
                "code": data[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
                "name": data[name_col].astype(str),
            }
        )
        normalized["change_pct"] = self._optional_column(data, ("涨跌幅", "changepercent", "change_pct"))
        normalized["volume_ratio"] = self._optional_column(data, ("量比", "volume_ratio"))
        normalized["turnover_rate"] = self._optional_column(data, ("换手率", "turnoverratio", "turnover_rate"))
        normalized["amount"] = self._optional_column(data, ("成交额", "amount"))
        normalized["pe"] = self._optional_column(data, ("市盈率-动态", "动态市盈率", "市盈率", "pe"))
        return normalized.dropna(subset=["code", "name"])

    def _sector_from_row(self, row: pd.Series) -> SectorOpportunity:
        name = str(row["sector"])
        change = _number(row.get("change_pct"))
        turnover = _number(row.get("turnover_rate"))
        leader = row.get("leader")
        leader_change = _number(row.get("leader_change_pct"))

        score = 0.0
        breakdown: list[FactorContribution] = []
        reasons: list[str] = []
        risks: list[str] = []
        if any(keyword in name for keyword in HOT_SECTOR_KEYWORDS):
            score += 0.16
            reasons.append("该方向属于近期高关注主题，纳入主题热度加分。")
        if change is not None:
            factor_score = _bounded_score(change / 6)
            score += factor_score * 0.34
            breakdown.append(self._factor("板块动量", factor_score, 0.34, f"板块当日涨跌幅为 {change:.2f}%。"))
            if 0.5 <= change <= 6:
                reasons.append(f"板块当日上涨 {change:.2f}%，说明短线资金已经开始形成关注。")
            elif change > 6:
                risks.append("板块涨幅过高，短期可能存在追高风险。")
            elif change < 0:
                risks.append(f"板块当日下跌 {change:.2f}%，趋势仍需确认。")
        if turnover is not None:
            factor_score = _bounded_score((turnover - 1.5) / 8)
            score += factor_score * 0.2
            breakdown.append(self._factor("交易活跃度", factor_score, 0.2, f"板块换手率为 {turnover:.2f}%。"))
            if turnover > 2:
                reasons.append(f"板块换手率 {turnover:.2f}%，交易活跃度较高。")
        if leader and str(leader) != "nan":
            score += 0.45 * 0.12
            breakdown.append(self._factor("龙头锚点", 0.45, 0.12, f"领涨股为 {leader}。"))
            reasons.append(f"领涨股为 {leader}，可作为板块持续性的观察锚点。")
        if leader_change is not None:
            factor_score = _bounded_score(leader_change / 10)
            score += factor_score * 0.18
            breakdown.append(self._factor("龙头强度", factor_score, 0.18, f"领涨股涨幅为 {leader_change:.2f}%。"))
        risk_penalty = -0.3 if change is not None and change > 7 else 0.05
        score += risk_penalty * 0.16
        breakdown.append(self._factor("过热风险", risk_penalty, 0.16, "涨幅过高时降低评分，避免只追逐短期情绪高点。"))

        return SectorOpportunity(
            name=name,
            score=round(score, 4),
            rating=_rating(score),
            change_pct=change,
            turnover_rate=turnover,
            leader=str(leader) if leader and str(leader) != "nan" else None,
            leader_change_pct=leader_change,
            signal_breakdown=breakdown,
            reasons=reasons or ["板块数据不完整，当前仅作为观察项。"],
            risks=risks or ["需结合大盘状态和板块持续性验证。"],
            watch_points=[
                "观察板块能否连续2-3个交易日维持相对强势。",
                "观察领涨股是否继续放量并带动更多成分股上涨。",
                "若大盘进入高波动状态，应降低追涨权重。",
            ],
        )

    def _enrich_sector(self, sector: SectorOpportunity) -> SectorOpportunity:
        trend_score, trend_summary, trend_points, trend_reasons, trend_risks = self._sector_trend_analysis(sector.name)
        breadth_score, breadth_summary, breadth_reasons, breadth_risks = self._sector_breadth_analysis(sector.name)
        news_score, news_summary, news_reasons, news_risks = self._sector_news_analysis(sector.name)
        score = sector.score + trend_score * 0.24 + breadth_score * 0.18 + news_score * 0.12
        signal_breakdown = [
            *sector.signal_breakdown,
            self._factor("板块K线趋势", trend_score, 0.24, trend_summary),
            self._factor("成分股扩散", breadth_score, 0.18, breadth_summary),
            self._factor("板块新闻情绪", news_score, 0.12, news_summary),
        ]
        return sector.model_copy(
            update={
                "score": round(score, 4),
                "rating": _rating(score),
                "trend_summary": trend_summary,
                "trend_points": trend_points,
                "breadth_summary": breadth_summary,
                "news_summary": news_summary,
                "signal_breakdown": signal_breakdown,
                "reasons": [*sector.reasons, *trend_reasons, *breadth_reasons, *news_reasons],
                "risks": [*sector.risks, *trend_risks, *breadth_risks, *news_risks],
            }
        )

    def _fast_enrich_sector(self, sector: SectorOpportunity) -> SectorOpportunity:
        trend_summary = "快速扫描：已使用实时板块涨跌幅、换手率、领涨股和龙头强度做初筛；详细K线/新闻建议进入单板块复核。"
        breadth_summary = "快速扫描：成分股扩散暂不逐板块拉取，避免AKShare子接口长时间阻塞。"
        news_summary = "快速扫描：新闻情绪暂不阻塞主扫描，可结合热点主题和新闻面二次确认。"
        signal_breakdown = [
            *sector.signal_breakdown,
            self._factor("快速趋势代理", 0.05, 0.08, trend_summary),
            self._factor("快速扩散代理", 0.0, 0.04, breadth_summary),
            self._factor("新闻复核提示", 0.0, 0.04, news_summary),
        ]
        return sector.model_copy(
            update={
                "trend_summary": trend_summary,
                "breadth_summary": breadth_summary,
                "news_summary": news_summary,
                "signal_breakdown": signal_breakdown,
                "watch_points": [
                    *sector.watch_points,
                    "快速扫描优先保证实时出结果；若要做深度复核，再查看板块K线、新闻和成分股扩散。",
                ],
            }
        )

    def _sector_trend_analysis(self, sector_name: str) -> tuple[float, str, list[SectorTrendPoint], list[str], list[str]]:
        errors: list[str] = []
        try:
            import akshare as ak

            for function_name in ("stock_board_industry_hist_em", "stock_board_concept_hist_em"):
                function = getattr(ak, function_name, None)
                if function is None:
                    continue
                try:
                    with akshare_network_context():
                        frame = function(symbol=sector_name, period="日k", adjust="")
                    if frame is not None and not frame.empty:
                        data = frame.rename(columns={"收盘": "close", "成交量": "volume"})
                        close = pd.to_numeric(data["close"], errors="coerce").dropna()
                        volume = pd.to_numeric(data["volume"], errors="coerce").dropna()
                        if len(close) < 30:
                            continue
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        ret20 = close.iloc[-1] / close.iloc[-20] - 1
                        vol_ratio = volume.tail(5).mean() / volume.tail(20).mean() if len(volume) >= 20 else 1
                        score = 0.0
                        reasons: list[str] = []
                        risks: list[str] = []
                        if close.iloc[-1] > ma20:
                            score += 0.25
                            reasons.append("板块指数站上MA20，趋势结构较好。")
                        else:
                            score -= 0.15
                            risks.append("板块指数仍在MA20下方，趋势确认不足。")
                        if ma5 > ma20:
                            score += 0.18
                            reasons.append("板块MA5高于MA20，短线强于中期均线。")
                        if ret20 > 0:
                            score += min(ret20 / 0.12, 0.22)
                            reasons.append(f"板块近20日收益 {ret20:.2%}，中期动量为正。")
                        if vol_ratio > 1.15:
                            score += 0.1
                            reasons.append(f"板块近5日量能相对20日放大 {vol_ratio:.2f} 倍。")
                        summary = f"板块K线：20日收益 {ret20:.2%}，价格相对MA20 {'偏强' if close.iloc[-1] > ma20 else '偏弱'}，量能比 {vol_ratio:.2f}。"
                        points = self._sector_trend_points(data, source=sector_name)
                        return _bounded_score(score), summary, points, reasons, risks
                except Exception as exc:
                    errors.append(f"{function_name}失败：{exc}")
        except Exception as exc:
            errors.append(f"导入AKShare失败：{exc}")
        return 0.0, "板块K线数据暂不可用：" + "；".join(errors[:2]), [], [], ["未能拉取板块历史K线，趋势确认缺失。"]

    def _sector_trend_points(self, frame: pd.DataFrame, source: str) -> list[SectorTrendPoint]:
        date_col = self._find_column(frame, ("日期", "date", "时间"))
        close_col = self._find_column(frame, ("close", "收盘", "收盘价"))
        if date_col is None or close_col is None:
            return []
        data = frame[[date_col, close_col]].copy()
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data[close_col] = pd.to_numeric(data[close_col], errors="coerce")
        data = data.dropna(subset=[date_col, close_col]).sort_values(date_col).tail(120)
        if data.empty:
            return []
        data["ma20"] = data[close_col].rolling(20).mean()
        data["ma60"] = data[close_col].rolling(60).mean()
        return [
            SectorTrendPoint(
                date=row[date_col].strftime("%Y-%m-%d"),
                close=round(float(row[close_col]), 4),
                ma20=None if pd.isna(row["ma20"]) else round(float(row["ma20"]), 4),
                ma60=None if pd.isna(row["ma60"]) else round(float(row["ma60"]), 4),
                source=source,
            )
            for _, row in data.iterrows()
        ]

    def _sector_breadth_analysis(self, sector_name: str) -> tuple[float, str, list[str], list[str]]:
        errors: list[str] = []
        try:
            import akshare as ak

            for function_name in ("stock_board_industry_cons_em", "stock_board_concept_cons_em"):
                function = getattr(ak, function_name, None)
                if function is None:
                    continue
                try:
                    with akshare_network_context():
                        frame = function(symbol=sector_name)
                    if frame is not None and not frame.empty:
                        change_col = self._find_column(frame, ("涨跌幅", "涨幅", "change_pct"))
                        if change_col is None:
                            continue
                        changes = pd.to_numeric(frame[change_col].astype(str).str.replace("%", ""), errors="coerce").dropna()
                        if changes.empty:
                            continue
                        up_ratio = float((changes > 0).mean())
                        strong_ratio = float((changes > 3).mean())
                        score = (up_ratio - 0.5) * 0.8 + strong_ratio * 0.4
                        reasons = []
                        risks = []
                        if up_ratio > 0.6:
                            reasons.append(f"成分股上涨占比 {up_ratio:.0%}，板块扩散较好。")
                        else:
                            risks.append(f"成分股上涨占比 {up_ratio:.0%}，板块扩散不足。")
                        if strong_ratio > 0.15:
                            reasons.append(f"涨幅超过3%的成分股占比 {strong_ratio:.0%}，强势股数量较多。")
                        summary = f"成分股扩散：上涨占比 {up_ratio:.0%}，强势股占比 {strong_ratio:.0%}。"
                        return _bounded_score(score), summary, reasons, risks
                except Exception as exc:
                    errors.append(f"{function_name}失败：{exc}")
        except Exception as exc:
            errors.append(f"导入AKShare失败：{exc}")
        return 0.0, "成分股扩散数据暂不可用：" + "；".join(errors[:2]), [], ["未能拉取板块成分股，扩散确认缺失。"]

    def _sector_news_analysis(self, sector_name: str) -> tuple[float, str, list[str], list[str]]:
        positive_keywords = ("利好", "增长", "突破", "政策", "订单", "景气", "创新高", "回购")
        negative_keywords = ("利空", "下滑", "亏损", "减持", "处罚", "调查", "风险")
        try:
            import akshare as ak

            headlines: list[str] = []
            terms = _sector_news_terms(sector_name)
            for function_name in ("stock_info_global_cls", "stock_news_em"):
                function = getattr(ak, function_name, None)
                if function is None:
                    continue
                try:
                    with akshare_network_context():
                        try:
                            frame = function()
                        except TypeError:
                            frame = function(symbol=sector_name)
                    title_col = self._find_column(frame, ("标题", "新闻标题", "title", "内容"))
                    if title_col is None:
                        continue
                    titles = [str(item) for item in frame[title_col].dropna().head(80)]
                    headlines.extend([title for title in titles if any(term in title for term in terms)])
                except Exception:
                    continue
            headlines = list(dict.fromkeys(headlines))
            if not headlines:
                return 0.0, f"近期新闻：未匹配到 {sector_name} 及相关主题新闻，情绪按中性处理。", [], []
            positive = sum(any(word in title for word in positive_keywords) for title in headlines)
            negative = sum(any(word in title for word in negative_keywords) for title in headlines)
            score = (positive - negative) / max(positive + negative, 1)
            sample = "；".join(headlines[:2])
            summary = f"近期新闻：匹配 {len(headlines)} 条，正面 {positive} 条，负面 {negative} 条；样例：{sample}。"
            reasons = [summary] if positive >= negative and positive > 0 else []
            risks = [summary] if negative > positive else []
            return _bounded_score(score), summary, reasons, risks
        except Exception as exc:
            return 0.0, f"板块新闻数据暂不可用：{exc}", [], ["未能拉取板块新闻，情绪确认缺失。"]

    def _stock_from_row(self, row: pd.Series) -> StockOpportunity | None:
        code = str(row["code"])
        name = str(row["name"])
        if not code or "ST" in name.upper():
            return None
        change = _number(row.get("change_pct"))
        volume_ratio = _number(row.get("volume_ratio"))
        turnover = _number(row.get("turnover_rate"))
        amount = _number(row.get("amount"))
        pe = _number(row.get("pe"))

        score = 0.0
        breakdown: list[FactorContribution] = []
        reasons: list[str] = []
        risks: list[str] = []
        if change is not None:
            factor_score = _bounded_score(change / 7)
            score += factor_score * 0.24
            breakdown.append(self._factor("价格动量", factor_score, 0.24, f"当日涨跌幅 {change:.2f}%。"))
            if 0.5 <= change <= 6:
                reasons.append(f"当日上涨 {change:.2f}%，短线动量较好且未明显过热。")
            elif change > 7:
                risks.append("当日涨幅较高，需警惕追高。")
        if volume_ratio is not None:
            factor_score = _bounded_score((volume_ratio - 1) / 3)
            score += factor_score * 0.22
            breakdown.append(self._factor("量比确认", factor_score, 0.22, f"量比 {volume_ratio:.2f}。"))
            if volume_ratio > 1.2:
                reasons.append(f"量比 {volume_ratio:.2f}，成交活跃度改善。")
        if turnover is not None:
            factor_score = _bounded_score((turnover - 1) / 8)
            score += factor_score * 0.14
            breakdown.append(self._factor("换手参与度", factor_score, 0.14, f"换手率 {turnover:.2f}%。"))
        if amount is not None:
            factor_score = _bounded_score(amount / 2_000_000_000)
            score += factor_score * 0.16
            breakdown.append(self._factor("流动性质量", factor_score, 0.16, "成交额越高，策略进出场可执行性通常越好。"))
            if amount < 500_000_000:
                risks.append("成交额偏低，可能存在流动性不足。")
        if pe is not None and pe > 0:
            factor_score = 0.35 if pe < 80 else -0.35
            score += factor_score * 0.12
            breakdown.append(self._factor("估值约束", factor_score, 0.12, f"动态市盈率 {pe:.2f}。"))
        heat_penalty = -0.35 if change is not None and change > 8 else 0.05
        score += heat_penalty * 0.12
        breakdown.append(self._factor("追高惩罚", heat_penalty, 0.12, "涨幅过高时降低评分，避免把高波动冲顶误判为低风险机会。"))

        suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return StockOpportunity(
            symbol=f"{code}.{suffix}",
            name=name,
            score=round(score, 4),
            rating=_rating(score),
            sector=self._infer_sector(name),
            change_pct=change,
            volume_ratio=volume_ratio,
            turnover_rate=turnover,
            amount=amount,
            trend_summary=None,
            fund_flow_summary=None,
            signal_breakdown=breakdown,
            reasons=reasons or ["实时行情信号完整，但优势尚不突出，适合加入观察池。"],
            risks=risks or ["仍需结合行业、财务和回测进一步确认。"],
            watch_points=[
                "观察次日是否继续放量且价格不跌破当日关键区间。",
                "结合所属板块是否同步走强，避免孤立个股信号。",
                "进入股票池后建议再生成K线预测和多因子解释。",
            ],
        )

    def _enrich_stock(self, stock: StockOpportunity) -> StockOpportunity:
        trend_score, trend_summary, trend_points, trend_reasons, trend_risks = self._stock_trend_analysis(stock)
        flow_score, flow_summary, flow_reasons, flow_risks = self._fund_flow_analysis(stock)
        sector_score = 0.08 if stock.sector and stock.sector != "其他" else 0.0
        score = stock.score + trend_score * 0.22 + flow_score * 0.2 + sector_score
        signal_breakdown = [
            *stock.signal_breakdown,
            self._factor("K线趋势确认", trend_score, 0.22, trend_summary),
            self._factor("资金流确认", flow_score, 0.2, flow_summary),
        ]
        reasons = [*stock.reasons, *trend_reasons, *flow_reasons]
        risks = [*stock.risks, *trend_risks, *flow_risks]
        return stock.model_copy(
            update={
                "score": round(score, 4),
                "rating": _rating(score),
                "trend_summary": trend_summary,
                "trend_points": trend_points,
                "fund_flow_summary": flow_summary,
                "signal_breakdown": signal_breakdown,
                "reasons": reasons,
                "risks": risks,
            }
        )

    def _fast_enrich_stock(self, stock: StockOpportunity) -> StockOpportunity:
        trend_score, trend_summary, trend_reasons, trend_risks = self._proxy_trend_analysis(stock)
        flow_score, flow_summary, flow_reasons, flow_risks = self._proxy_fund_flow_analysis(stock)
        sector_score = 0.08 if stock.sector and stock.sector != "其他" else 0.0
        score = stock.score + trend_score * 0.18 + flow_score * 0.16 + sector_score
        signal_breakdown = [
            *stock.signal_breakdown,
            self._factor("快速趋势代理", trend_score, 0.18, trend_summary),
            self._factor("快速资金代理", flow_score, 0.16, flow_summary),
        ]
        return stock.model_copy(
            update={
                "score": round(score, 4),
                "rating": _rating(score),
                "trend_summary": trend_summary,
                "fund_flow_summary": flow_summary,
                "signal_breakdown": signal_breakdown,
                "reasons": [*stock.reasons, *trend_reasons, *flow_reasons],
                "risks": [*stock.risks, *trend_risks, *flow_risks],
            }
        )

    def _stock_trend_analysis(self, stock: StockOpportunity) -> tuple[float, str, list[StockTrendPoint], list[str], list[str]]:
        try:
            end = date.today()
            start = end - timedelta(days=260)
            data = AkshareDataProvider().history(MarketDataRequest(stock.symbol, start, end))
            close = pd.to_numeric(data["close"], errors="coerce")
            volume = pd.to_numeric(data["volume"], errors="coerce")
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            ret20 = close.iloc[-1] / close.iloc[-20] - 1
            vol_ratio = volume.tail(5).mean() / volume.tail(20).mean()
            score = 0.0
            reasons: list[str] = []
            risks: list[str] = []
            if close.iloc[-1] > ma20:
                score += 0.25
                reasons.append("价格站上MA20，中短期趋势结构较好。")
            else:
                score -= 0.18
                risks.append("价格仍在MA20下方，趋势确认不足。")
            if ma5 > ma20 > ma60:
                score += 0.28
                reasons.append("MA5 > MA20 > MA60，均线多头排列。")
            if ret20 > 0:
                score += min(ret20 / 0.12, 0.25)
                reasons.append(f"近20日收益 {ret20:.2%}，中期动量为正。")
            if vol_ratio > 1.15:
                score += 0.12
                reasons.append(f"近5日量能相对20日放大 {vol_ratio:.2f} 倍。")
            summary = f"K线趋势：收盘价相对MA20 {'偏强' if close.iloc[-1] > ma20 else '偏弱'}，20日收益 {ret20:.2%}，量能比 {vol_ratio:.2f}。"
            points = self._stock_trend_points(data, source=stock.symbol)
            return _bounded_score(score), summary, points, reasons, risks
        except Exception as exc:
            score, summary, reasons, risks = self._proxy_trend_analysis(stock)
            summary += f" 上游原因：{self._short_error(exc)}"
            risks = [*risks, "未能拉取单股历史K线，趋势确认降级为实时行情代理。"]
            return score, summary, [], reasons, risks

    def _stock_trend_points(self, frame: pd.DataFrame, source: str) -> list[StockTrendPoint]:
        if frame.empty or "close" not in frame.columns:
            return []
        data = frame[["close"]].copy()
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data = data.dropna(subset=["close"]).sort_index().tail(120)
        if data.empty:
            return []
        data["ma20"] = data["close"].rolling(20).mean()
        data["ma60"] = data["close"].rolling(60).mean()
        return [
            StockTrendPoint(
                date=index.strftime("%Y-%m-%d"),
                close=round(float(row["close"]), 4),
                ma20=None if pd.isna(row["ma20"]) else round(float(row["ma20"]), 4),
                ma60=None if pd.isna(row["ma60"]) else round(float(row["ma60"]), 4),
                source=source,
            )
            for index, row in data.iterrows()
        ]

    def _fund_flow_analysis(self, stock: StockOpportunity) -> tuple[float, str, list[str], list[str]]:
        code = stock.symbol.split(".")[0]
        ths_result = self._fund_flow_from_ths_snapshot(code, stock.name)
        if ths_result is not None:
            return ths_result
        try:
            import akshare as ak

            symbol = stock.symbol
            with akshare_network_context():
                frame = ak.stock_individual_fund_flow(stock=code, market="sh" if symbol.endswith(".SH") else "sz")
            if frame.empty:
                return 0.0, "资金流数据为空。", [], ["资金流数据为空，无法确认资金方向。"]
            latest = frame.tail(5)
            main_col = self._find_column(latest, ("主力净流入-净额", "主力净流入净额", "主力净流入"))
            pct_col = self._find_column(latest, ("主力净流入-净占比", "主力净流入净占比"))
            score = 0.0
            reasons: list[str] = []
            risks: list[str] = []
            if main_col is not None:
                values = pd.to_numeric(latest[main_col], errors="coerce").fillna(0)
                positive_days = int((values > 0).sum())
                total_flow = float(values.sum())
                score += (positive_days - 2.5) / 5
                if total_flow > 0:
                    reasons.append(f"近5日主力资金合计净流入，正流入天数 {positive_days} 天。")
                else:
                    risks.append(f"近5日主力资金合计净流出，正流入天数 {positive_days} 天。")
            if pct_col is not None:
                pct = pd.to_numeric(latest[pct_col], errors="coerce").tail(3).mean()
                if pd.notna(pct):
                    score += _bounded_score(float(pct) / 10) * 0.3
            summary = "资金流：已纳入近5日主力资金净流入方向和净占比变化。"
            return _bounded_score(score), summary, reasons, risks
        except Exception as exc:
            score, summary, reasons, risks = self._proxy_fund_flow_analysis(stock)
            summary += f" 上游原因：{self._short_error(exc)}"
            risks = [*risks, "未能拉取单股主力资金流，资金确认降级为成交额、量比和换手率代理。"]
            return score, summary, reasons, risks

    def _fund_flow_from_ths_snapshot(self, code: str, stock_name: str | None = None) -> tuple[float, str, list[str], list[str]] | None:
        try:
            import akshare as ak

            if self._fund_flow_snapshot is None:
                with akshare_network_context():
                    self._fund_flow_snapshot = ak.stock_fund_flow_individual(symbol="即时")
            frame = self._fund_flow_snapshot
            code_col = self._find_column(frame, ("股票代码", "代码", "code"))
            name_col = self._find_column(frame, ("股票简称", "名称", "name"))
            net_col = self._find_column(frame, ("净额", "主力净额", "资金净额"))
            amount_col = self._find_column(frame, ("成交额", "amount"))
            if code_col is None or net_col is None:
                return None
            data = frame.copy()
            data[code_col] = self._normalize_stock_codes(data[code_col])
            matched = data[data[code_col] == code.zfill(6)]
            if matched.empty and stock_name and name_col is not None:
                names = data[name_col].astype(str).str.strip()
                target_name = stock_name.strip()
                matched = data[names == target_name]
                if matched.empty:
                    matched = data[names.str.contains(target_name, regex=False, na=False)]
            if matched.empty:
                return None
            row = matched.iloc[0]
            net = self._money_to_float(row.get(net_col))
            amount = self._money_to_float(row.get(amount_col)) if amount_col else None
            ratio = net / amount if net is not None and amount not in (None, 0) else None
            score = _bounded_score((ratio or 0) * 8)
            name = str(row.get(name_col)) if name_col else code
            summary = f"资金流：同花顺实时资金流显示 {name} 净额 {self._money_text(net)}"
            if amount is not None:
                summary += f"，成交额 {self._money_text(amount)}"
            if ratio is not None:
                summary += f"，净额/成交额约 {ratio:.2%}"
            summary += "。"
            reasons = [summary] if net is not None and net > 0 else []
            risks = [summary] if net is not None and net < 0 else []
            return score, summary, reasons, risks
        except Exception:
            return None

    def _proxy_trend_analysis(self, stock: StockOpportunity) -> tuple[float, str, list[str], list[str]]:
        score = 0.0
        reasons: list[str] = []
        risks: list[str] = []
        if stock.change_pct is not None:
            if 0.5 <= stock.change_pct <= 6:
                score += 0.18
                reasons.append("单股历史K线接口暂不可用，使用实时涨跌幅作为短线趋势代理。")
            elif stock.change_pct > 7:
                score -= 0.2
                risks.append("单日涨幅过高，实时趋势可能已经过热。")
            elif stock.change_pct < 0:
                score -= 0.12
                risks.append("实时价格走弱，短线趋势代理偏弱。")
        if stock.volume_ratio is not None and stock.volume_ratio > 1.2:
            score += 0.12
            reasons.append("量比放大，说明当前交易活跃度较前期均值改善。")
        fields = []
        if stock.change_pct is not None:
            fields.append(f"涨跌幅 {stock.change_pct:.2f}%")
        if stock.volume_ratio is not None:
            fields.append(f"量比 {stock.volume_ratio:.2f}")
        if stock.turnover_rate is not None:
            fields.append(f"换手率 {stock.turnover_rate:.2f}%")
        summary = "K线趋势代理：单股历史K线接口暂不可用，改用实时行情快照做弱替代"
        if fields:
            summary += "；" + "，".join(fields) + "。"
        else:
            summary += "；当前快照缺少可用趋势代理字段。"
        return _bounded_score(score), summary, reasons, risks

    def _proxy_fund_flow_analysis(self, stock: StockOpportunity) -> tuple[float, str, list[str], list[str]]:
        score = 0.0
        reasons: list[str] = []
        risks: list[str] = []
        if stock.amount is not None and stock.amount > 500_000_000:
            score += 0.12
            reasons.append("单股资金流接口暂不可用，使用成交额作为资金参与度代理。")
        elif stock.amount is not None:
            risks.append("成交额偏低，资金参与度代理不足。")
        if stock.volume_ratio is not None and stock.volume_ratio > 1.5:
            score += 0.12
            reasons.append("量比明显放大，可作为短线资金活跃的代理信号。")
        fields = []
        if stock.amount is not None:
            fields.append(f"成交额 {stock.amount / 100000000:.2f}亿元")
        if stock.volume_ratio is not None:
            fields.append(f"量比 {stock.volume_ratio:.2f}")
        if stock.turnover_rate is not None:
            fields.append(f"换手率 {stock.turnover_rate:.2f}%")
        summary = "资金流代理：主力资金流接口暂不可用，改用成交额、量比和换手率评估资金参与度"
        if fields:
            summary += "；" + "，".join(fields) + "。"
        else:
            summary += "；当前快照缺少可用资金代理字段。"
        return _bounded_score(score), summary, reasons, risks

    def _factor(self, name: str, score: float, weight: float, explanation: str) -> FactorContribution:
        return FactorContribution(
            name=name,
            value=f"{score:+.2f}",
            impact=_impact(score),
            weight=weight,
            explanation=explanation,
        )

    def _find_column(self, frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
        return next((name for name in names if name in frame.columns), None)

    def _optional_column(self, frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
        column = self._find_column(frame, names)
        if column is None:
            return None
        return frame[column]

    def _average(self, values: list[float | None]) -> float | None:
        numbers = [float(value) for value in values if value is not None and pd.notna(value)]
        if not numbers:
            return None
        return sum(numbers) / len(numbers)

    def _money_to_float(self, value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).replace(",", "").strip()
        multiplier = 1.0
        if text.endswith("亿"):
            multiplier = 100_000_000
            text = text[:-1]
        elif text.endswith("万"):
            multiplier = 10_000
            text = text[:-1]
        try:
            return float(text) * multiplier
        except Exception:
            return None

    def _money_text(self, value: float | None) -> str:
        if value is None:
            return "暂无"
        if abs(value) >= 100_000_000:
            return f"{value / 100_000_000:.2f}亿元"
        if abs(value) >= 10_000:
            return f"{value / 10_000:.2f}万元"
        return f"{value:.2f}元"

    def _normalize_stock_codes(self, values: pd.Series) -> pd.Series:
        codes = values.astype(str).str.extract(r"(\d{1,6})", expand=False)
        return codes.where(codes.isna(), codes.str.zfill(6))

    def _short_error(self, exc: Exception) -> str:
        text = str(exc).replace("\n", " ").strip()
        if not text:
            return exc.__class__.__name__
        return text[:120] + ("..." if len(text) > 120 else "")

    def _ensure_hot_sector_visibility(self, sectors: list[SectorOpportunity]) -> list[SectorOpportunity]:
        existing = {sector.name for sector in sectors}
        for keyword in HOT_SECTOR_KEYWORDS:
            if any(keyword in name for name in existing):
                continue
            sectors.append(
                SectorOpportunity(
                    name=keyword,
                    score=0.28,
                    rating="观察",
                    change_pct=None,
                    turnover_rate=None,
                    leader=None,
                    leader_change_pct=None,
                    trend_summary=None,
                    breadth_summary=None,
                    news_summary=None,
                    signal_breakdown=[self._factor("主题关注", 0.2, 0.16, "该方向属于近期高关注主题，但当前数据源未返回完整板块快照。")],
                    reasons=["热门主题观察项：当前数据源未给出完整板块行情，建议结合新闻、行业指数和成分股扩散继续验证。"],
                    risks=["缺少实时板块涨跌幅和换手率，不能仅凭主题热度判断机会。"],
                    watch_points=["观察是否有龙头股持续放量。", "观察全A股扫描中是否出现同主题个股集体上榜。"],
                )
            )
        return sectors

    def _infer_sector(self, stock_name: str) -> str:
        rules = {
            "通信": ("通信", "中兴", "烽火", "光迅", "亨通", "中际", "新易盛", "天孚"),
            "半导体": ("芯", "半导体", "电子", "韦尔", "兆易", "北方华创", "中微"),
            "有色金属": ("铜", "铝", "锂", "钼", "矿业", "洛阳钼业", "紫金", "赣锋", "天齐"),
            "汽车": ("汽车", "比亚迪", "长安", "赛力斯", "江淮"),
            "证券": ("证券", "中信证券", "东方财富", "华泰"),
            "新能源": ("宁德", "阳光", "隆基", "通威", "亿纬"),
            "白酒消费": ("茅台", "五粮液", "泸州", "汾酒"),
            "银行": ("银行", "招商银行", "平安银行", "兴业"),
        }
        for sector, keywords in rules.items():
            if any(keyword in stock_name for keyword in keywords):
                return sector
        return "其他"

    def _mock_sectors(self, limit: int) -> SectorOpportunityReport:
        sectors = [
            SectorOpportunity(
                name="半导体",
                score=0.78,
                rating="高潜力",
                change_pct=2.4,
                turnover_rate=3.8,
                leader="样例龙头",
                leader_change_pct=6.2,
                signal_breakdown=[],
                reasons=["行业景气度预期改善。", "成交活跃度较高。", "领涨股带动明显。"],
                risks=["短期涨幅较快，需要观察持续性。"],
                watch_points=["观察板块是否连续放量。"],
            )
        ]
        return SectorOpportunityReport(
            summary="离线演示：启用 AKShare 后将扫描真实行业板块。",
            methodology="离线模式只展示输出结构。",
            sectors=sectors[:limit],
        )

    def _mock_stocks(self, limit: int) -> StockOpportunityReport:
        stocks = [
            StockOpportunity(
                symbol="300750.SZ",
                name="宁德时代",
                score=0.68,
                rating="高潜力",
                change_pct=1.8,
                volume_ratio=1.4,
                turnover_rate=2.1,
                amount=1_800_000_000,
                signal_breakdown=[],
                reasons=["离线演示：量能改善。", "离线演示：相对强弱较好。"],
                risks=["需启用 AKShare 获取真实全 A 股扫描。"],
                watch_points=["启用真实数据后再验证。"],
            )
        ]
        return StockOpportunityReport(
            summary="离线演示：启用 AKShare 后将扫描全 A 股。",
            methodology="离线模式只展示输出结构。",
            stocks=stocks[:limit],
        )
