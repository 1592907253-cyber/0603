from dataclasses import dataclass

from agent_trading.data.network import akshare_network_context


POSITIVE_WORDS = {
    "增长",
    "上涨",
    "突破",
    "增持",
    "盈利",
    "利好",
    "改善",
    "创新高",
    "回购",
    "中标",
    "扩产",
}

NEGATIVE_WORDS = {
    "下跌",
    "亏损",
    "减持",
    "利空",
    "处罚",
    "调查",
    "风险",
    "暴雷",
    "下滑",
    "违约",
    "退市",
}


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    confidence: float
    summary: str
    headlines: tuple[str, ...]


class SentimentAgent:
    """Lightweight Chinese financial-news sentiment agent.

    The agent uses AKShare when available and falls back to neutral sentiment.
    The keyword model is intentionally transparent for hackathon demo use; it can
    later be replaced by FinBERT/Chinese financial sentiment models.
    """

    def analyze(self, symbol: str, provider_name: str = "mock") -> SentimentResult:
        if provider_name.lower() != "akshare":
            return self._neutral("当前未启用 AKShare，情绪因子按中性处理。")

        try:
            import akshare as ak

            code = symbol.split(".")[0]
            with akshare_network_context():
                news = ak.stock_news_em(symbol=code)
            title_column = "新闻标题" if "新闻标题" in news.columns else "title"
            headlines = tuple(str(item) for item in news[title_column].dropna().head(12))
        except Exception:
            return self._neutral("新闻数据暂时不可用，情绪因子按中性处理。")

        if not headlines:
            return self._neutral("未获取到近期新闻，情绪因子按中性处理。")

        positive_hits = sum(any(word in title for word in POSITIVE_WORDS) for title in headlines)
        negative_hits = sum(any(word in title for word in NEGATIVE_WORDS) for title in headlines)
        total_hits = positive_hits + negative_hits
        if total_hits == 0:
            return SentimentResult(
                score=0.0,
                label="中性",
                confidence=0.45,
                summary="近期新闻未出现明显正负面关键词，情绪因子保持中性。",
                headlines=headlines[:5],
            )

        score = (positive_hits - negative_hits) / max(total_hits, 1)
        label = "偏正面" if score > 0.2 else "偏负面" if score < -0.2 else "中性"
        confidence = min(0.45 + total_hits / max(len(headlines), 1) * 0.5, 0.9)
        summary = (
            f"近期新闻中正面信号 {positive_hits} 条，负面信号 {negative_hits} 条，"
            f"情绪判断为{label}。"
        )
        return SentimentResult(
            score=float(score),
            label=label,
            confidence=float(confidence),
            summary=summary,
            headlines=headlines[:5],
        )

    def _neutral(self, summary: str) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            label="中性",
            confidence=0.35,
            summary=summary,
            headlines=(),
        )
