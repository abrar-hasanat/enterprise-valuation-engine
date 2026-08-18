"""Smoke tests for the universal financial CSV loader."""

from __future__ import annotations

from io import StringIO

from engine.data_loader import UniversalFinancialDataLoader


def test_load_dirty_metric_rows_csv() -> None:
    """Dirty row-oriented metrics should normalize to clean floats."""
    dirty_csv = StringIO(
        """Line Item,FY 2024,FY 2023
Total Revenue,"$12,500.0M","$10.1B"
Operating Profit,"($450.0M)",$750M
Net Earnings,$300.5M,($10.0M)
Cash from Operations,$1.25B,$900M
CapEx,($125.0M),($100M)
D&A,$50M,$45M
Total Borrowings,$2.0B,$1.5B
Total Cash,$700M,$600M
Assets,$5.0B,$4.5B
Liabilities,$3.0B,$2.9B
Book Value,$2.0B,$1.6B
Diluted Shares,100M,95M
"""
    )

    frame = UniversalFinancialDataLoader.load_csv(dirty_csv)

    assert frame.loc["FY 2024", "revenue"] == 12_500_000_000.0
    assert frame.loc["FY 2024", "ebit"] == -450_000_000.0
    assert frame.loc["FY 2023", "revenue"] == 10_100_000_000.0
    assert frame.loc["FY 2024", "capex"] == -125_000_000.0
    assert frame.loc["FY 2024", "shares_outstanding"] == 100_000_000.0
    assert all(str(dtype) == "float64" for dtype in frame.dtypes)


def test_load_dirty_period_rows_csv() -> None:
    """Dirty period-oriented rows should remain row-indexed by period."""
    dirty_csv = StringIO(
        """Fiscal Year,Sales,Income from Operations,Bottom Line,CFO,Capital Expenditures
2024,"$12,500.0M","($450.0M)",$300M,$1.2B,($125M)
2023,$10B,$700M,$250M,$950M,($100M)
"""
    )

    frame = UniversalFinancialDataLoader.load_csv(dirty_csv)

    assert frame.loc["2024", "revenue"] == 12_500_000_000.0
    assert frame.loc["2024", "ebit"] == -450_000_000.0
    assert frame.loc["2023", "operating_cash_flow"] == 950_000_000.0
    assert frame.loc["2023", "cash_and_equivalents"] == 0.0
