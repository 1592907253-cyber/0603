from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agent_trading.data.network import akshare_network_context
from agent_trading.schemas import (
    FactorContribution,
    SectorOpportunity,
    SectorOpportunityReport,
    StockOpportunity,
    StockOpportunityReport,
)


def _number(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
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

    def sector_opportunities(self, limit: int = 8) -> SectorOpportunityReport:
        if self.provider_name.lower() != "akshare":
            return self._mock_sectors(limit)

        errors: list[str] = []
        try:
            import akshare as ak

            with akshare_network_context():
                frame = ak.stock_board_industry_name_em()
        except Exception as exc:
            errors.append(f"东方财富行业板块接口失败：{exc}")
            frame = self._try_ths_sector_frame(errors)

        if frame is None or frame.empty:
            return SectorOpportunityReport(
                summary=(
                    "AKShare 行业板块数据获取失败，未使用离线演示数据。"
                    f"错误信息：{' | '.join(errors)}"
                ),
                sectors=[],
            )

        sectors: list[SectorOpportunity] = []
        for _, row in frame.iterrows():
            name = str(_first_existing(row, ("板块名称", "名称", "name")) or "")
            change = _number(_first_existing(row, ("涨跌幅", "涨跌幅%", "change_pct")))
            turnover = _number(_first_existing(row, ("换手率", "turnover_rate")))
            leader = _first_existing(row, ("领涨股票", "领涨股", "leader"))
            leader_change = _number(
                _first_existing(row, ("领涨股票-涨跌幅", "领涨股涨跌幅", "领涨股-涨跌幅"))
            )

            score = 0.0
            breakdown: list[FactorContribution] = []
            reasons: list[str] = []
            risks: list[str] = []
            if change is not None:
                factor_score = _bounded_score(change / 6)
                score += factor_score * 0.34
                breakdown.append(
                    FactorContribution(
                        name="板块动量",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.34,
                        explanation=f"板块当日涨跌幅为 {change:.2f}%，反映资金短线进攻方向。",
                    )
                )
                if 0.5 <= change <= 6:
                    reasons.append(f"板块当日上涨 {change:.2f}%，说明短线资金已经开始形成关注。")
                elif change > 6:
                    risks.append("板块涨幅过高，短期可能存在追高风险。")
                elif change < 0:
                    risks.append(f"板块当日下跌 {change:.2f}%，趋势仍需确认。")
            if turnover is not None:
                factor_score = _bounded_score((turnover - 1.5) / 8)
                score += factor_score * 0.2
                breakdown.append(
                    FactorContribution(
                        name="交易活跃度",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.2,
                        explanation=f"板块换手率为 {turnover:.2f}%，用于判断资金参与深度。",
                    )
                )
                if turnover > 2:
                    reasons.append(f"板块换手率 {turnover:.2f}%，交易活跃度较高，具备扩散基础。")
            if leader and str(leader) != "nan":
                factor_score = 0.45
                score += factor_score * 0.12
                breakdown.append(
                    FactorContribution(
                        name="龙头锚点",
                        value=f"{factor_score:+.2f}",
                        impact="positive",
                        weight=0.12,
                        explanation=f"领涨股为 {leader}，板块有明确强度观察对象。",
                    )
                )
                reasons.append(f"领涨股为 {leader}，可作为板块持续性的观察锚点。")
            if leader_change is not None:
                factor_score = _bounded_score(leader_change / 10)
                score += factor_score * 0.18
                breakdown.append(
                    FactorContribution(
                        name="龙头强度",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.18,
                        explanation=f"领涨股涨幅为 {leader_change:.2f}%，用于判断龙头带动是否足够强。",
                    )
                )
                if leader_change > 5:
                    reasons.append(f"领涨股涨幅 {leader_change:.2f}%，板块龙头带动较明显。")
            risk_penalty = -0.3 if change is not None and change > 7 else 0.05
            score += risk_penalty * 0.16
            breakdown.append(
                FactorContribution(
                    name="过热风险",
                    value=f"{risk_penalty:+.2f}",
                    impact=_impact(risk_penalty),
                    weight=0.16,
                    explanation="涨幅过高时降低评分，避免只追逐短期情绪高点。",
                )
            )

            if name:
                sectors.append(
                    SectorOpportunity(
                        name=name,
                        score=round(score, 4),
                        rating=_rating(score),
                        change_pct=change,
                        turnover_rate=turnover,
                        leader=str(leader) if leader and str(leader) != "nan" else None,
                        leader_change_pct=leader_change,
                        signal_breakdown=breakdown,
                        reasons=reasons
                        or [
                            "板块信号完整但优势不突出，适合作为观察对象而非立即追涨。",
                            "需要等待连续放量、涨幅扩散或龙头持续走强来确认。",
                        ],
                        risks=risks or ["需结合大盘状态和板块持续性验证。"],
                        watch_points=[
                            "观察板块能否连续2-3个交易日维持相对强势。",
                            "观察领涨股是否继续放量并带动更多成分股上涨。",
                            "若大盘进入高波动状态，应降低追涨权重。",
                        ],
                    )
                )

        sectors.sort(key=lambda item: item.score, reverse=True)
        return SectorOpportunityReport(
            summary="基于板块动量、交易活跃度、龙头锚点、龙头强度和过热风险筛选潜力板块。",
            methodology="评分不是单看涨幅，而是同时考察资金是否参与、龙头是否明确、强度是否扩散，以及是否已经过热。",
            sectors=sectors[:limit],
        )

    def stock_opportunities(self, limit: int = 30) -> StockOpportunityReport:
        if self.provider_name.lower() != "akshare":
            return self._mock_stocks(limit)

        errors: list[str] = []
        try:
            import akshare as ak

            with akshare_network_context():
                frame = ak.stock_zh_a_spot_em()
        except Exception as exc:
            errors.append(f"东方财富全A实时行情接口失败：{exc}")
            frame = self._try_legacy_stock_spot_frame(errors)

        if frame is None or frame.empty:
            return StockOpportunityReport(
                summary=(
                    "AKShare 全 A 股实时行情获取失败，未使用离线演示数据。"
                    f"错误信息：{' | '.join(errors)}"
                ),
                stocks=[],
            )

        stocks: list[StockOpportunity] = []
        for _, row in frame.iterrows():
            code = str(_first_existing(row, ("代码", "code")) or "")
            name = str(_first_existing(row, ("名称", "name")) or "")
            if not code or "ST" in name.upper():
                continue

            change = _number(_first_existing(row, ("涨跌幅", "change_pct")))
            volume_ratio = _number(_first_existing(row, ("量比", "volume_ratio")))
            turnover = _number(_first_existing(row, ("换手率", "turnover_rate")))
            amount = _number(_first_existing(row, ("成交额", "amount")))
            pe = _number(_first_existing(row, ("市盈率-动态", "动态市盈率", "市盈率", "pe")))

            score = 0.0
            breakdown: list[FactorContribution] = []
            reasons: list[str] = []
            risks: list[str] = []
            if change is not None:
                factor_score = _bounded_score(change / 7)
                score += factor_score * 0.24
                breakdown.append(
                    FactorContribution(
                        name="价格动量",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.24,
                        explanation=f"当日涨跌幅 {change:.2f}%，用于捕捉短线资金方向。",
                    )
                )
                if 0.5 <= change <= 6:
                    reasons.append(f"当日上涨 {change:.2f}%，短线动量较好且未明显过热。")
                elif change > 7:
                    risks.append("当日涨幅较高，需警惕追高。")
            if volume_ratio is not None:
                factor_score = _bounded_score((volume_ratio - 1) / 3)
                score += factor_score * 0.22
                breakdown.append(
                    FactorContribution(
                        name="量比确认",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.22,
                        explanation=f"量比 {volume_ratio:.2f}，用于判断上涨是否有成交配合。",
                    )
                )
                if volume_ratio > 1.2:
                    reasons.append(f"量比 {volume_ratio:.2f}，成交活跃度改善。")
            if turnover is not None:
                factor_score = _bounded_score((turnover - 1) / 8)
                score += factor_score * 0.14
                breakdown.append(
                    FactorContribution(
                        name="换手参与度",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.14,
                        explanation=f"换手率 {turnover:.2f}%，用于判断资金参与是否充分。",
                    )
                )
                if turnover > 2:
                    reasons.append(f"换手率 {turnover:.2f}%，资金参与度较高。")
            if amount is not None:
                factor_score = _bounded_score(amount / 2_000_000_000)
                score += factor_score * 0.16
                breakdown.append(
                    FactorContribution(
                        name="流动性质量",
                        value=f"{factor_score:+.2f}",
                        impact=_impact(factor_score),
                        weight=0.16,
                        explanation="成交额越高，策略进出场可执行性通常越好。",
                    )
                )
                if amount > 500_000_000:
                    reasons.append("成交额较高，流动性较好。")
                else:
                    risks.append("成交额偏低，可能存在流动性不足。")
            if pe is not None and pe > 0:
                if pe < 80:
                    factor_score = 0.35
                    score += factor_score * 0.12
                    breakdown.append(
                        FactorContribution(
                            name="估值约束",
                            value=f"{factor_score:+.2f}",
                            impact="positive",
                            weight=0.12,
                            explanation=f"动态市盈率 {pe:.2f}，未处于极端高估区间。",
                        )
                    )
                    reasons.append("动态市盈率未处于极端高位。")
                else:
                    factor_score = -0.35
                    score += factor_score * 0.12
                    breakdown.append(
                        FactorContribution(
                            name="估值约束",
                            value=f"{factor_score:+.2f}",
                            impact="negative",
                            weight=0.12,
                            explanation=f"动态市盈率 {pe:.2f}，估值弹性与风险需要同时关注。",
                        )
                    )
                    risks.append("动态市盈率偏高，估值风险需关注。")
            heat_penalty = -0.35 if change is not None and change > 8 else 0.05
            score += heat_penalty * 0.12
            breakdown.append(
                FactorContribution(
                    name="追高惩罚",
                    value=f"{heat_penalty:+.2f}",
                    impact=_impact(heat_penalty),
                    weight=0.12,
                    explanation="涨幅过高时降低评分，避免把高波动冲顶误判为低风险机会。",
                )
            )

            suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
            stocks.append(
                StockOpportunity(
                    symbol=f"{code}.{suffix}",
                    name=name,
                    score=round(score, 4),
                    rating=_rating(score),
                    change_pct=change,
                    volume_ratio=volume_ratio,
                    turnover_rate=turnover,
                    amount=amount,
                    signal_breakdown=breakdown,
                    reasons=reasons
                    or [
                        "实时行情信号完整，但优势尚不突出，适合加入观察池。",
                        "需要结合所属板块强度、历史K线和新闻情绪继续验证。",
                    ],
                    risks=risks or ["仍需结合行业、财务和回测进一步确认。"],
                    watch_points=[
                        "观察次日是否继续放量且价格不跌破当日关键区间。",
                        "结合所属板块是否同步走强，避免孤立个股信号。",
                        "进入股票池后建议再生成K线预测和多因子解释。",
                    ],
                )
            )

        stocks.sort(key=lambda item: item.score, reverse=True)
        return StockOpportunityReport(
            summary="基于全 A 股实时行情，从价格动量、量比确认、换手参与度、流动性质量、估值约束和追高惩罚筛选候选股票。",
            methodology="评分优先寻找有资金参与、流动性可执行、涨幅不过热的候选标的；结果用于生成观察池，不等同于直接买入建议。",
            stocks=stocks[:limit],
        )

    def _try_ths_sector_frame(self, errors: list[str]) -> pd.DataFrame | None:
        try:
            import akshare as ak

            candidates = (
                "stock_board_industry_name_ths",
                "stock_board_concept_name_ths",
            )
            for function_name in candidates:
                function = getattr(ak, function_name, None)
                if function is None:
                    errors.append(f"AKShare 未提供 {function_name}")
                    continue
                try:
                    with akshare_network_context():
                        frame = function()
                    if frame is not None and not frame.empty:
                        return self._normalize_ths_sector_frame(frame)
                except Exception as exc:
                    errors.append(f"{function_name} 失败：{exc}")
        except Exception as exc:
            errors.append(f"导入 AKShare 失败：{exc}")
        return None

    def _normalize_ths_sector_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        rename_map = {
            "板块": "板块名称",
            "行业": "板块名称",
            "概念名称": "板块名称",
            "名称": "板块名称",
            "涨幅": "涨跌幅",
            "涨跌幅": "涨跌幅",
            "换手": "换手率",
        }
        for source, target in rename_map.items():
            if source in data.columns and target not in data.columns:
                data[target] = data[source]
        if "领涨股票" not in data.columns:
            for name in ("领涨股", "龙头股", "股票"):
                if name in data.columns:
                    data["领涨股票"] = data[name]
                    break
        return data

    def _try_legacy_stock_spot_frame(self, errors: list[str]) -> pd.DataFrame | None:
        try:
            import akshare as ak

            candidates = ("stock_zh_a_spot", "stock_zh_a_spot_em")
            for function_name in candidates:
                function = getattr(ak, function_name, None)
                if function is None:
                    errors.append(f"AKShare 未提供 {function_name}")
                    continue
                try:
                    with akshare_network_context():
                        frame = function()
                    if frame is not None and not frame.empty:
                        return self._normalize_stock_spot_frame(frame)
                except Exception as exc:
                    errors.append(f"{function_name} 失败：{exc}")
        except Exception as exc:
            errors.append(f"导入 AKShare 失败：{exc}")
        return None

    def _normalize_stock_spot_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        rename_map = {
            "code": "代码",
            "symbol": "代码",
            "name": "名称",
            "trade": "最新价",
            "price": "最新价",
            "changepercent": "涨跌幅",
            "change_pct": "涨跌幅",
            "turnoverratio": "换手率",
            "turnover_rate": "换手率",
            "volume_ratio": "量比",
            "amount": "成交额",
        }
        for source, target in rename_map.items():
            if source in data.columns and target not in data.columns:
                data[target] = data[source]
        return data

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
            ),
            SectorOpportunity(
                name="机器人",
                score=0.72,
                rating="高潜力",
                change_pct=1.9,
                turnover_rate=3.1,
                leader="样例龙头",
                leader_change_pct=5.4,
                signal_breakdown=[],
                reasons=["主题关注度较高。", "板块内个股扩散较好。"],
                risks=["主题交易波动较大。"],
                watch_points=["观察龙头股是否持续走强。"],
            ),
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
