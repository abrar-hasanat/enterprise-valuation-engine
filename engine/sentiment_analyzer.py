"""Market sentiment analysis and executive intelligence synthesis.

The module combines lightweight market-news sentiment, DCF valuation outputs, and
financial-ratio diagnostics into a structured equity research memo. It is built
to run with zero external API cost by default, while optionally using OpenAI for
memo drafting when an API key is supplied.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class ExecutiveIntelligenceSynthesizer:
    """Synthesize valuation, ratio diagnostics, sentiment, and executive memos.

    Args:
        custom_headlines: Optional proprietary/private-company headlines used
            when market-data news is unavailable or the target is not public.
        positive_keywords: Optional override for bullish financial sentiment
            vocabulary.
        negative_keywords: Optional override for bearish financial sentiment
            vocabulary.
    """

    custom_headlines: list[str] | None = None
    positive_keywords: set[str] | None = None
    negative_keywords: set[str] | None = None
    _latest_articles: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)
    _latest_sentiment: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _latest_memo: str = field(default="", init=False, repr=False)
    _latest_payload_inputs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    DEFAULT_POSITIVE_KEYWORDS: frozenset[str] = frozenset(
        {
            "accelerate",
            "beat",
            "beats",
            "bullish",
            "buyback",
            "catalyst",
            "expansion",
            "growth",
            "guidance raise",
            "margin expansion",
            "momentum",
            "outperform",
            "record revenue",
            "upgrade",
            "upside",
            "strong demand",
            "profitability improvement",
            "cash flow growth",
        }
    )
    DEFAULT_NEGATIVE_KEYWORDS: frozenset[str] = frozenset(
        {
            "bearish",
            "contraction",
            "cut guidance",
            "downgrade",
            "headwinds",
            "investigation",
            "lawsuit",
            "litigation",
            "margin pressure",
            "miss",
            "restructuring",
            "slowdown",
            "underperform",
            "weak demand",
            "impairment",
            "cash burn",
            "debt concerns",
        }
    )

    def fetch_company_news(self, ticker_or_name: str, max_articles: int = 8) -> list[dict[str, str]]:
        """Fetch recent company news or return deterministic fallback updates.

        The method first attempts ``yfinance.Ticker(ticker).news``. If yfinance
        is unavailable, the ticker is private/custom, or no articles are returned,
        it falls back to user-provided ``custom_headlines`` and then to generic
        corporate-update headlines.
        """
        if max_articles < 1:
            raise ValueError("max_articles must be at least 1.")

        articles = self._fetch_yfinance_news(ticker_or_name, max_articles)
        if not articles:
            headlines = self.custom_headlines or self._fallback_headlines(ticker_or_name)
            now = datetime.now(timezone.utc).isoformat()
            articles = [
                {"headline": headline, "publisher": "Internal Research Fallback", "timestamp": now}
                for headline in headlines[:max_articles]
            ]
        self._latest_articles = articles[:max_articles]
        return self._latest_articles

    def analyze_news_sentiment(self, articles: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Aggregate article-level sentiment into a market sentiment profile."""
        selected_articles = articles if articles is not None else self._latest_articles
        if not selected_articles:
            selected_articles = self.fetch_company_news("Target Company")

        scored_articles: list[dict[str, Any]] = []
        total_score = 0.0
        themes: dict[str, int] = {}
        for article in selected_articles:
            headline = article.get("headline", "")
            score = self._analyze_sentiment(headline)
            total_score += score
            for theme in self._extract_themes(headline):
                themes[theme] = themes.get(theme, 0) + 1
            scored_articles.append({**article, "sentiment_score": round(score, 4)})

        aggregate_score = total_score / len(scored_articles) if scored_articles else 0.0
        sentiment = {
            "sentiment_score": round(aggregate_score, 4),
            "sentiment_classification": self._classify_sentiment(aggregate_score),
            "key_themes": [theme for theme, _ in sorted(themes.items(), key=lambda item: (-item[1], item[0]))[:5]],
            "article_count": len(scored_articles),
            "articles": scored_articles,
        }
        self._latest_sentiment = sentiment
        return sentiment

    def generate_executive_memo(
        self,
        company_name: str,
        valuation_summary: dict[str, Any],
        ratios_summary: pd.DataFrame,
        health_scorecard: dict[str, Any],
        news_sentiment: dict[str, Any],
        api_key: str | None = None,
    ) -> str:
        """Generate a one-page institutional-grade equity research memo.

        If ``api_key`` or ``OPENAI_API_KEY`` exists, the method attempts to use
        OpenAI for drafting. Any SDK/runtime failure falls back to deterministic
        Markdown templating so the platform remains fully operational offline.
        """
        self._latest_payload_inputs = {
            "company_name": company_name,
            "valuation_summary": valuation_summary,
            "ratios_summary": ratios_summary,
            "health_scorecard": health_scorecard,
            "news_sentiment": news_sentiment,
        }
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_api_key:
            memo = self._generate_openai_memo(
                company_name, valuation_summary, ratios_summary, health_scorecard, news_sentiment, resolved_api_key
            )
            if memo:
                self._latest_memo = memo
                return memo

        memo = self._generate_rule_based_memo(company_name, valuation_summary, ratios_summary, health_scorecard, news_sentiment)
        self._latest_memo = memo
        return memo

    def export_intelligence_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable intelligence payload from the latest run."""
        payload = {
            "news_articles": self._jsonify(self._latest_articles),
            "news_sentiment": self._jsonify(self._latest_sentiment),
            "executive_memo": self._latest_memo,
        }
        payload.update({key: self._jsonify(value) for key, value in self._latest_payload_inputs.items()})
        return payload

    def _fetch_yfinance_news(self, ticker_or_name: str, max_articles: int) -> list[dict[str, str]]:
        """Best-effort yfinance news retrieval without hard dependency failures."""
        try:
            yfinance = importlib.import_module("yfinance")
            raw_news = yfinance.Ticker(ticker_or_name).news or []
        except Exception:
            return []

        articles: list[dict[str, str]] = []
        for item in raw_news[:max_articles]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            headline = content.get("title") or item.get("title") or item.get("headline") if isinstance(item, dict) else ""
            if not headline:
                continue
            publisher = content.get("provider", {}) if isinstance(content, dict) else {}
            publisher_name = publisher.get("displayName") if isinstance(publisher, dict) else None
            timestamp = content.get("pubDate") or item.get("providerPublishTime") or item.get("pubDate")
            articles.append(
                {
                    "headline": str(headline),
                    "publisher": str(publisher_name or item.get("publisher", "Yahoo Finance")),
                    "timestamp": self._format_timestamp(timestamp),
                }
            )
        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Score financial sentiment polarity from -1.0 to +1.0."""
        normalized = self._normalize_text(text)
        positive_terms = self.positive_keywords or set(self.DEFAULT_POSITIVE_KEYWORDS)
        negative_terms = self.negative_keywords or set(self.DEFAULT_NEGATIVE_KEYWORDS)
        positive_hits = self._keyword_hits(normalized, positive_terms)
        negative_hits = self._keyword_hits(normalized, negative_terms)
        raw_score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
        return max(min(raw_score, 1.0), -1.0)

    def _generate_openai_memo(
        self,
        company_name: str,
        valuation_summary: dict[str, Any],
        ratios_summary: pd.DataFrame,
        health_scorecard: dict[str, Any],
        news_sentiment: dict[str, Any],
        api_key: str,
    ) -> str | None:
        """Attempt OpenAI memo generation, returning ``None`` on any failure."""
        try:
            openai = importlib.import_module("openai")
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior equity research analyst and strategy consultant. "
                            "Draft concise, institutional-grade markdown with explicit numbers, balanced risk language, "
                            "and no unsupported claims."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._memo_prompt(
                            company_name, valuation_summary, ratios_summary, health_scorecard, news_sentiment
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=1_200,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def _generate_rule_based_memo(
        self,
        company_name: str,
        valuation_summary: dict[str, Any],
        ratios_summary: pd.DataFrame,
        health_scorecard: dict[str, Any],
        news_sentiment: dict[str, Any],
    ) -> str:
        """Build a deterministic Markdown memo populated with calculated values."""
        target_price = self._format_currency(valuation_summary.get("intrinsic_share_price"), precision=2)
        enterprise_value = self._format_currency(valuation_summary.get("enterprise_value"), precision=1)
        equity_value = self._format_currency(valuation_summary.get("equity_value"), precision=1)
        upside = self._format_percent(valuation_summary.get("upside_downside"))
        wacc = self._format_percent(valuation_summary.get("wacc") or valuation_summary.get("weighted_average_cost_of_capital"))
        terminal_growth = self._format_percent(
            valuation_summary.get("perpetual_growth_rate") or valuation_summary.get("terminal_growth_rate")
        )
        rating = str(health_scorecard.get("rating", "N/A"))
        sentiment_classification = str(news_sentiment.get("sentiment_classification", "NEUTRAL"))
        sentiment_score = news_sentiment.get("sentiment_score", 0.0)
        themes = news_sentiment.get("key_themes", []) or ["company-specific execution", "valuation discipline"]
        flags = health_scorecard.get("flags", []) or ["• No major ratio-based risk triggers identified."]

        return f"""# Equity Research & Strategic Memo: {company_name}

## 1. Executive Summary & Investment Verdict
- **Investment verdict:** {self._investment_verdict(upside, rating, sentiment_classification)}
- **Intrinsic target price:** {target_price}; **implied upside/downside:** {upside}.
- **Enterprise value:** {enterprise_value}; **equity value:** {equity_value}.
- **Financial health badge:** {rating}; **market sentiment:** {sentiment_classification} ({float(sentiment_score):+.2f}).

## 2. Fundamental Financial Health & Ratio Diagnostics
{self._ratio_snapshot(ratios_summary)}

**Risk flags**
{os.linesep.join(str(flag) for flag in flags)}

## 3. Valuation Methodology & DCF Drivers
- The DCF output indicates an enterprise value of **{enterprise_value}** and equity value of **{equity_value}** after net debt/cash adjustments.
- Core model drivers include **WACC of {wacc}** and **terminal growth of {terminal_growth}**; sensitivity should focus on discount-rate durability, terminal margin quality, and reinvestment intensity.
- The target price should be interpreted as a fundamentals-based intrinsic value estimate rather than a short-term trading forecast.

## 4. Market Sentiment, Catalysts & Strategic Risk Factors
- Aggregated headline sentiment is **{sentiment_classification}** with a score of **{float(sentiment_score):+.2f}**.
- Dominant market themes: {', '.join(str(theme) for theme in themes)}.
- Key catalysts include execution against growth expectations, margin delivery, free-cash-flow conversion, and balance-sheet discipline.
- Key risks include macro headwinds, cost inflation, competitive pressure, financing costs, and any company-specific adverse news reflected in the sentiment feed.
"""

    def _memo_prompt(
        self,
        company_name: str,
        valuation_summary: dict[str, Any],
        ratios_summary: pd.DataFrame,
        health_scorecard: dict[str, Any],
        news_sentiment: dict[str, Any],
    ) -> str:
        """Assemble a strict prompt for optional OpenAI memo drafting."""
        return (
            f"Company: {company_name}\n"
            f"Valuation summary: {self._jsonify(valuation_summary)}\n"
            f"Ratio summary table: {self._jsonify(ratios_summary)}\n"
            f"Health scorecard: {self._jsonify(health_scorecard)}\n"
            f"News sentiment: {self._jsonify(news_sentiment)}\n\n"
            "Write a one-page markdown memo with four sections: Executive Summary & Investment Verdict; "
            "Fundamental Financial Health & Ratio Diagnostics; Valuation Methodology & DCF Drivers; "
            "Market Sentiment, Catalysts & Strategic Risk Factors. Use exact supplied values."
        )

    def _ratio_snapshot(self, ratios_summary: pd.DataFrame) -> str:
        """Extract a concise latest-period ratio snapshot from a formatted table."""
        if ratios_summary.empty:
            return "- Ratio summary was unavailable."
        latest_period = ratios_summary.columns[-1]
        desired_rows = [
            "Operating (EBIT) Margin",
            "Net Profit Margin",
            "Return on Equity",
            "Return on Invested Capital",
            "Net Debt-to-EBITDA",
            "Interest Coverage",
            "Free Cash Flow Conversion",
        ]
        lines = [f"- Latest reporting period analyzed: **{latest_period}**."]
        for row in desired_rows:
            if row in ratios_summary.index:
                lines.append(f"- **{row}:** {ratios_summary.loc[row, latest_period]}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_headlines(ticker_or_name: str) -> list[str]:
        """Generate neutral fallback headlines for private/custom companies."""
        return [
            f"{ticker_or_name} reports continued focus on profitable growth and cash flow discipline",
            f"{ticker_or_name} management highlights margin expansion initiatives amid macro headwinds",
            f"{ticker_or_name} reviews capital allocation priorities and balance sheet flexibility",
            f"Analysts monitor {ticker_or_name} demand trends, competition, and execution catalysts",
        ]

    def _extract_themes(self, text: str) -> list[str]:
        """Map headline text into high-level market themes."""
        normalized = self._normalize_text(text)
        theme_keywords = {
            "growth": ("growth", "demand", "revenue", "expansion"),
            "margins": ("margin", "cost", "profitability", "pricing"),
            "balance sheet": ("debt", "cash", "liquidity", "capital allocation"),
            "market sentiment": ("upgrade", "downgrade", "outperform", "underperform"),
            "legal/regulatory": ("litigation", "lawsuit", "investigation", "regulatory"),
            "execution risk": ("headwinds", "miss", "slowdown", "competition"),
        }
        return [theme for theme, terms in theme_keywords.items() if any(term in normalized for term in terms)]

    @staticmethod
    def _classify_sentiment(score: float) -> str:
        """Classify aggregate polarity score into bullish/neutral/bearish."""
        if score >= 0.2:
            return "BULLISH"
        if score <= -0.2:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _keyword_hits(text: str, keywords: set[str] | frozenset[str]) -> int:
        """Count exact word/phrase keyword hits in normalized text."""
        hits = 0
        for keyword in keywords:
            normalized_keyword = ExecutiveIntelligenceSynthesizer._normalize_text(keyword)
            if " " in normalized_keyword:
                hits += int(normalized_keyword in text)
            else:
                hits += len(re.findall(rf"\b{re.escape(normalized_keyword)}\b", text))
        return hits

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase and normalize whitespace for deterministic text matching."""
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _format_timestamp(timestamp: Any) -> str:
        """Convert yfinance timestamp variants into an ISO-like string."""
        if timestamp is None:
            return ""
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        return str(timestamp)

    @staticmethod
    def _investment_verdict(upside: str, health_rating: str, sentiment: str) -> str:
        """Create a compact rule-based investment verdict."""
        if upside.startswith("-") or health_rating == "DISTRESSED" or sentiment == "BEARISH":
            return "Cautious / risk-managed stance warranted"
        if health_rating in {"STRONG", "MODERATE"} and sentiment in {"BULLISH", "NEUTRAL"}:
            return "Constructive, subject to valuation and execution discipline"
        return "Balanced watchlist stance pending catalyst confirmation"

    @staticmethod
    def _format_percent(value: Any) -> str:
        """Format numeric value as a percentage string."""
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1%}"

    @staticmethod
    def _format_currency(value: Any, precision: int = 1) -> str:
        """Format scalar currency/share values in compact notation."""
        if value is None or pd.isna(value):
            return "N/A"
        amount = float(value)
        sign = "-" if amount < 0 else ""
        absolute = abs(amount)
        for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if absolute >= divisor:
                return f"{sign}${absolute / divisor:.{precision}f}{suffix}"
        return f"{sign}${absolute:.{precision}f}"

    @classmethod
    def _jsonify(cls, value: Any) -> Any:
        """Convert pandas/numpy-heavy objects into JSON-serializable structures."""
        if isinstance(value, pd.DataFrame):
            return value.astype(object).where(pd.notna(value), None).to_dict(orient="index")
        if isinstance(value, pd.Series):
            return value.astype(object).where(pd.notna(value), None).to_dict()
        if isinstance(value, dict):
            return {str(key): cls._jsonify(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._jsonify(item) for item in value]
        if isinstance(value, tuple):
            return [cls._jsonify(item) for item in value]
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value
