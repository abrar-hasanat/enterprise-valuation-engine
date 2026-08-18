"""Financial ratio analytics and health diagnostics.

This module provides a quantitative ratio engine for normalized financial
statements produced by :class:`engine.data_loader.UniversalFinancialDataLoader`.
It computes multi-period profitability, leverage, liquidity, efficiency, and
cash-flow quality metrics, plus a concise analyst-style health scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FinancialRatioEngine:
    """Compute multi-year financial ratios and risk diagnostics.

    Args:
        financials: Normalized financial DataFrame with canonical columns from
            ``UniversalFinancialDataLoader``. Optional columns such as
            ``current_assets``, ``current_liabilities``, and ``interest_expense``
            are used automatically when present.
        tax_rate: Tax rate used to convert EBIT into NOPAT for ROIC.
        default_pretax_cost_of_debt: Fallback interest-cost assumption used when
            explicit interest expense is unavailable.
    """

    financials: pd.DataFrame
    tax_rate: float = 0.21
    default_pretax_cost_of_debt: float = 0.055
    _required_columns: tuple[str, ...] = field(
        default=(
            "revenue",
            "ebit",
            "net_income",
            "operating_cash_flow",
            "capex",
            "depreciation_amortization",
            "total_debt",
            "cash_and_equivalents",
            "total_assets",
            "total_liabilities",
            "shareholder_equity",
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate inputs before ratio computation."""
        if self.financials.empty:
            raise ValueError("Financial DataFrame must contain at least one fiscal period.")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be between 0.0 and 1.0.")
        if self.default_pretax_cost_of_debt < 0:
            raise ValueError("default_pretax_cost_of_debt cannot be negative.")
        missing = [column for column in self._required_columns if column not in self.financials.columns]
        if missing:
            raise ValueError(f"Financial DataFrame is missing required columns: {', '.join(missing)}")

    def calculate_ratios(self) -> pd.DataFrame:
        """Return numeric ratios for every reported fiscal period.

        Invalid divisions, infinite results, and metrics without meaningful
        denominators are returned as ``pd.NA`` rather than raising arithmetic
        exceptions.
        """
        frame = self._chronological_financials()
        ebitda = frame["ebit"] + frame["depreciation_amortization"]
        net_debt = frame["total_debt"] - frame["cash_and_equivalents"]
        invested_capital = frame["total_debt"] + frame["shareholder_equity"] - frame["cash_and_equivalents"]
        nopat = frame["ebit"] * (1 - self.tax_rate)
        interest_expense = self._interest_expense(frame)
        current_assets = frame["current_assets"] if "current_assets" in frame.columns else frame["total_assets"]
        current_liabilities = (
            frame["current_liabilities"] if "current_liabilities" in frame.columns else frame["total_liabilities"]
        )
        capital_spending = frame["capex"].abs()

        ratios = pd.DataFrame(index=frame.index)
        ratios["operating_margin"] = self._safe_divide(frame["ebit"], frame["revenue"])
        ratios["net_profit_margin"] = self._safe_divide(frame["net_income"], frame["revenue"])
        ratios["roe"] = self._safe_divide(frame["net_income"], frame["shareholder_equity"])
        ratios["roic"] = self._safe_divide(nopat, invested_capital)
        ratios["current_or_solvency_ratio"] = self._safe_divide(current_assets, current_liabilities)
        ratios["debt_to_equity"] = self._safe_divide(frame["total_debt"], frame["shareholder_equity"])
        ratios["net_debt_to_ebitda"] = self._safe_divide(net_debt, ebitda)
        ratios["interest_coverage"] = self._safe_divide(frame["ebit"], interest_expense)
        ratios["fcf_conversion"] = self._safe_divide(
            frame["operating_cash_flow"] - capital_spending, frame["net_income"]
        )
        ratios["capex_to_revenue"] = self._safe_divide(capital_spending, frame["revenue"])
        ratios["ocf_to_debt"] = self._safe_divide(frame["operating_cash_flow"], frame["total_debt"])
        ratios["ebitda"] = ebitda
        ratios["net_debt"] = net_debt
        ratios["operating_cash_flow"] = frame["operating_cash_flow"]
        return ratios.astype("Float64")

    def generate_health_scorecard(self) -> dict[str, Any]:
        """Generate health rating badge and itemized risk flags.

        Returns:
            Dictionary containing ``rating``, ``flag_count``, ``flags``, and
            ``latest_period``. Flags are bullet-ready strings explaining each
            triggered diagnostic.
        """
        ratios = self.calculate_ratios()
        latest_period = str(ratios.index[-1])
        latest = ratios.iloc[-1]
        flags: list[str] = []

        net_debt_to_ebitda = self._scalar(latest["net_debt_to_ebitda"])
        debt_to_equity = self._scalar(latest["debt_to_equity"])
        if net_debt_to_ebitda is not None and net_debt_to_ebitda > 4.0:
            flags.append(f"• Excessive leverage: Net Debt / EBITDA is {net_debt_to_ebitda:.1f}x, above the 4.0x threshold.")
        if debt_to_equity is not None and debt_to_equity > 2.5:
            flags.append(f"• Excessive leverage: Debt / Equity is {debt_to_equity:.1f}x, above the 2.5x threshold.")

        fcf_conversion = self._scalar(latest["fcf_conversion"])
        operating_cash_flow = self._scalar(latest["operating_cash_flow"])
        if fcf_conversion is not None and fcf_conversion < 0.5:
            flags.append(f"• Cash flow stress: FCF conversion is {fcf_conversion:.1f}x, below the 0.5x threshold.")
        if operating_cash_flow is not None and operating_cash_flow < 0:
            flags.append("• Cash flow stress: Operating cash flow is negative in the latest reported period.")

        if len(ratios) >= 2:
            latest_margin = self._scalar(ratios["operating_margin"].iloc[-1])
            prior_margin = self._scalar(ratios["operating_margin"].iloc[-2])
            if latest_margin is not None and prior_margin is not None:
                margin_change = latest_margin - prior_margin
                if margin_change < -0.02:
                    flags.append(
                        "• Margin contraction: EBIT margin declined "
                        f"{abs(margin_change) * 10_000:.0f} bps year over year."
                    )

        interest_coverage = self._scalar(latest["interest_coverage"])
        if interest_coverage is not None and interest_coverage < 2.5:
            flags.append(
                f"• Solvency risk: Interest coverage is {interest_coverage:.1f}x, below the 2.5x threshold."
            )

        rating = self._rating_from_flags(flags)
        if not flags:
            flags.append("• No major ratio-based risk triggers identified in the latest period.")
        return {
            "rating": rating,
            "flag_count": 0 if rating == "STRONG" else len(flags),
            "latest_period": latest_period,
            "flags": flags,
        }

    def get_summary_table(self) -> pd.DataFrame:
        """Return formatted multi-year ratio summary table for analyst reports."""
        ratios = self.calculate_ratios()
        formatters = {
            "operating_margin": self._format_percent,
            "net_profit_margin": self._format_percent,
            "roe": self._format_percent,
            "roic": self._format_percent,
            "current_or_solvency_ratio": self._format_multiple,
            "debt_to_equity": self._format_multiple,
            "net_debt_to_ebitda": self._format_multiple,
            "interest_coverage": self._format_multiple,
            "fcf_conversion": self._format_multiple,
            "capex_to_revenue": self._format_percent,
            "ocf_to_debt": self._format_percent,
            "ebitda": self._format_currency,
            "net_debt": self._format_currency,
        }
        labels = {
            "operating_margin": "Operating (EBIT) Margin",
            "net_profit_margin": "Net Profit Margin",
            "roe": "Return on Equity",
            "roic": "Return on Invested Capital",
            "current_or_solvency_ratio": "Current / Solvency Ratio",
            "debt_to_equity": "Debt-to-Equity",
            "net_debt_to_ebitda": "Net Debt-to-EBITDA",
            "interest_coverage": "Interest Coverage",
            "fcf_conversion": "Free Cash Flow Conversion",
            "capex_to_revenue": "CapEx-to-Revenue",
            "ocf_to_debt": "Operating Cash Flow-to-Debt",
            "ebitda": "EBITDA",
            "net_debt": "Net Debt",
        }
        formatted = pd.DataFrame(index=[labels[column] for column in formatters], columns=ratios.index)
        for column, formatter in formatters.items():
            formatted.loc[labels[column]] = ratios[column].map(formatter).tolist()
        return formatted

    def _chronological_financials(self) -> pd.DataFrame:
        """Return numeric financials in chronological fiscal-period order."""
        frame = self.financials.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parsed = pd.to_datetime(frame.index, errors="coerce")
        if parsed.notna().sum() == len(frame.index):
            return frame.assign(_period_sort=parsed).sort_values("_period_sort").drop(columns="_period_sort")
        years = pd.Series(frame.index.astype(str), index=frame.index).str.extract(r"(\d{4})", expand=False)
        if years.notna().sum() == len(frame.index):
            return frame.assign(_period_sort=years.astype(int)).sort_values("_period_sort").drop(columns="_period_sort")
        return frame

    def _interest_expense(self, frame: pd.DataFrame) -> pd.Series:
        """Return explicit or estimated interest expense by period."""
        if "interest_expense" in frame.columns:
            return frame["interest_expense"].abs()
        return frame["total_debt"].abs() * self.default_pretax_cost_of_debt

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two numeric series while preserving missing outputs."""
        clean_denominator = denominator.astype(float).mask(denominator.astype(float).eq(0.0))
        ratio = numerator.astype(float).divide(clean_denominator)
        return ratio.replace([float("inf"), float("-inf")], pd.NA)

    @staticmethod
    def _scalar(value: Any) -> float | None:
        """Convert a pandas scalar to ``float`` or ``None`` when unavailable."""
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _rating_from_flags(flags: list[str]) -> str:
        """Translate the number and type of flags into a health badge."""
        if not flags:
            return "STRONG"
        if len(flags) == 1:
            return "MODERATE"
        if len(flags) == 2:
            return "WATCHLIST"
        return "DISTRESSED"

    @staticmethod
    def _format_percent(value: Any) -> str:
        """Format ratio as a percentage string."""
        return "N/A" if pd.isna(value) else f"{float(value):.1%}"

    @staticmethod
    def _format_multiple(value: Any) -> str:
        """Format ratio as an ``x`` multiple string."""
        return "N/A" if pd.isna(value) else f"{float(value):.1f}x"

    @staticmethod
    def _format_currency(value: Any) -> str:
        """Format scalar currency into compact dollar units."""
        if pd.isna(value):
            return "N/A"
        amount = float(value)
        sign = "-" if amount < 0 else ""
        absolute = abs(amount)
        for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if absolute >= divisor:
                return f"{sign}${absolute / divisor:.1f}{suffix}"
        return f"{sign}${absolute:.0f}"
