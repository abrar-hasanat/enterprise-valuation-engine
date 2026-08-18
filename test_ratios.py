"""Executable smoke test for the financial ratio engine."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from engine.data_loader import UniversalFinancialDataLoader
from engine.financial_ratios import FinancialRatioEngine


def build_normalized_financials() -> pd.DataFrame:
    """Load synthetic CSV financials through the universal data loader."""
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
    """Run the ratio engine and print analyst-ready diagnostics."""
    engine = FinancialRatioEngine(build_normalized_financials())
    summary_table = engine.get_summary_table()
    scorecard = engine.generate_health_scorecard()

    print("Multi-Year Financial Ratio Summary")
    print("=" * 40)
    print(summary_table.to_string())
    print("\nAutomated Health Scorecard")
    print("=" * 40)
    print(f"Latest Period: {scorecard['latest_period']}")
    print(f"Overall Rating: {scorecard['rating']}")
    print("Flags:")
    for flag in scorecard["flags"]:
        print(flag)


if __name__ == "__main__":
    main()
