"""Executable smoke test for the DCF valuation engine."""

from __future__ import annotations

import pandas as pd

from engine.dcf_model import DCFValuationEngine


def build_synthetic_financials() -> pd.DataFrame:
    """Create normalized synthetic multi-year financial statements."""
    data = {
        "revenue": [8_000_000_000, 8_800_000_000, 9_700_000_000, 10_600_000_000, 11_500_000_000],
        "ebit": [1_120_000_000, 1_280_000_000, 1_455_000_000, 1_650_000_000, 1_840_000_000],
        "net_income": [760_000_000, 880_000_000, 1_010_000_000, 1_140_000_000, 1_280_000_000],
        "operating_cash_flow": [1_200_000_000, 1_360_000_000, 1_520_000_000, 1_710_000_000, 1_930_000_000],
        "capex": [-420_000_000, -460_000_000, -510_000_000, -550_000_000, -610_000_000],
        "depreciation_amortization": [300_000_000, 330_000_000, 360_000_000, 395_000_000, 430_000_000],
        "total_debt": [2_000_000_000, 2_100_000_000, 2_150_000_000, 2_200_000_000, 2_250_000_000],
        "cash_and_equivalents": [750_000_000, 810_000_000, 875_000_000, 940_000_000, 1_000_000_000],
        "total_assets": [9_500_000_000, 10_200_000_000, 11_100_000_000, 11_900_000_000, 12_700_000_000],
        "total_liabilities": [5_400_000_000, 5_800_000_000, 6_200_000_000, 6_500_000_000, 6_850_000_000],
        "shareholder_equity": [4_100_000_000, 4_400_000_000, 4_900_000_000, 5_400_000_000, 5_850_000_000],
        "shares_outstanding": [200_000_000, 198_000_000, 196_000_000, 194_000_000, 192_000_000],
    }
    return pd.DataFrame(data, index=["2020", "2021", "2022", "2023", "2024"])


def main() -> None:
    """Run a synthetic valuation and print an executive summary."""
    engine = DCFValuationEngine(
        build_synthetic_financials(),
        beta=1.1,
        current_market_price=125.0,
        interest_expense=135_000_000,
        perpetual_growth_rate=0.025,
        exit_ebitda_multiple=10.0,
    )
    result = engine.run_valuation()
    valuation = result["valuation"]

    print("Executive DCF Valuation Summary")
    print("=" * 40)
    print(f"Enterprise Value: ${valuation['enterprise_value']:,.0f}")
    print(f"Equity Value:     ${valuation['equity_value']:,.0f}")
    print(f"Target Price:     ${valuation['intrinsic_share_price']:,.2f}")
    print(f"Upside/Downside:  {valuation['upside_downside']:.1%}")
    print("\n5x5 WACC / Perpetual Growth Sensitivity Matrix")
    print(result["sensitivity_matrix"].round(2).to_string())


if __name__ == "__main__":
    main()
