"""Executable smoke test for executive intelligence synthesis."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from engine.data_loader import UniversalFinancialDataLoader
from engine.dcf_model import DCFValuationEngine
from engine.financial_ratios import FinancialRatioEngine
from engine.sentiment_analyzer import ExecutiveIntelligenceSynthesizer


def build_sample_financials() -> pd.DataFrame:
    """Load synthetic normalized statements via the universal loader."""
    csv_data = StringIO(
        """Fiscal Year,Sales,Operating Income,Net Income,Cash from Operations,CapEx,D&A,Total Debt,Total Cash,Assets,Liabilities,Book Value,Diluted Shares
2020,$8.0B,$1.12B,$760M,$1.20B,($420M),$300M,$2.0B,$750M,$9.5B,$5.4B,$4.1B,200M
2021,$8.8B,$1.28B,$880M,$1.36B,($460M),$330M,$2.1B,$810M,$10.2B,$5.8B,$4.4B,198M
2022,$9.7B,$1.46B,$1.01B,$1.52B,($510M),$360M,$2.15B,$875M,$11.1B,$6.2B,$4.9B,196M
2023,$10.6B,$1.65B,$1.14B,$1.71B,($550M),$395M,$2.20B,$940M,$11.9B,$6.5B,$5.4B,194M
2024,$11.5B,$1.84B,$1.28B,$1.93B,($610M),$430M,$2.25B,$1.00B,$12.7B,$6.85B,$5.85B,192M
"""
    )
    return UniversalFinancialDataLoader.load_csv(csv_data)


def main() -> None:
    """Run valuation, ratios, sentiment, and print the final memo."""
    company_name = "Acme Industrial Technologies"
    financials = build_sample_financials()

    dcf_engine = DCFValuationEngine(
        financials,
        beta=1.1,
        current_market_price=125.0,
        interest_expense=135_000_000,
        perpetual_growth_rate=0.025,
    )
    dcf_output = dcf_engine.run_valuation()
    valuation_summary = {
        **dcf_output["valuation"],
        "wacc": dcf_output["wacc_components"]["wacc"],
        "perpetual_growth_rate": dcf_engine.perpetual_growth_rate,
        "sensitivity_matrix": dcf_output["sensitivity_matrix"],
    }

    ratio_engine = FinancialRatioEngine(financials)
    ratios_summary = ratio_engine.get_summary_table()
    health_scorecard = ratio_engine.generate_health_scorecard()

    synthesizer = ExecutiveIntelligenceSynthesizer(
        custom_headlines=[
            "Acme Industrial Technologies beats revenue expectations as demand growth accelerates",
            "Management highlights margin expansion program and cash flow growth priorities",
            "Analysts cite macro headwinds but maintain outperform view on Acme Industrial Technologies",
            "Company reviews capital allocation and balance sheet flexibility after debt refinancing",
        ]
    )
    articles = synthesizer.fetch_company_news(company_name, max_articles=4)
    news_sentiment = synthesizer.analyze_news_sentiment(articles)
    memo = synthesizer.generate_executive_memo(
        company_name=company_name,
        valuation_summary=valuation_summary,
        ratios_summary=ratios_summary,
        health_scorecard=health_scorecard,
        news_sentiment=news_sentiment,
    )

    print("Sentiment Breakdown")
    print("=" * 40)
    print(f"Classification: {news_sentiment['sentiment_classification']}")
    print(f"Score: {news_sentiment['sentiment_score']:+.2f}")
    print(f"Themes: {', '.join(news_sentiment['key_themes'])}")
    print("\nExecutive Memo")
    print("=" * 40)
    print(memo)


if __name__ == "__main__":
    main()
