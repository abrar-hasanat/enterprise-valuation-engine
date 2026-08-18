"""Discounted cash flow valuation engine.

This module contains a production-oriented DCF model that consumes the canonical
financial statement schema produced by :class:`engine.data_loader.UniversalFinancialDataLoader`.
It estimates historical operating trends, projects unlevered free cash flow to
the firm, computes WACC, and returns enterprise value, equity value, intrinsic
share price, and valuation sensitivity tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DCFValuationEngine:
    """Automated discounted cash flow valuation model.

    Args:
        financials: Normalized multi-year financial DataFrame containing the
            canonical schema columns emitted by ``UniversalFinancialDataLoader``.
        forecast_period: Number of explicit annual forecast periods.
        tax_rate: Marginal cash tax rate applied to EBIT.
        capex_to_revenue_ratio: Optional normalized CapEx as a positive
            percentage of revenue. If omitted, the engine derives it from
            historical CapEx/revenue.
        perpetual_growth_rate: Long-run terminal growth rate for the Gordon
            Growth terminal value.
        risk_free_rate: CAPM risk-free rate.
        equity_risk_premium: CAPM equity risk premium.
        beta: Levered equity beta used in CAPM.
        current_market_price: Optional current share price for upside/downside.
        market_cap: Optional current equity market value used for WACC capital
            structure weights. If omitted, the engine uses the latest reported
            equity value or revenue multiple fallback.
        interest_expense: Optional latest annual interest expense. If missing,
            pre-tax cost of debt defaults to ``default_pretax_cost_of_debt``.
        default_pretax_cost_of_debt: Fallback pre-tax cost of debt.
        exit_ebitda_multiple: EV/EBITDA multiple for the alternate terminal value.
        nwc_to_incremental_revenue_ratio: Working-capital investment as a share
            of incremental revenue when explicit NWC changes are unavailable.
    """

    financials: pd.DataFrame
    forecast_period: int = 5
    tax_rate: float = 0.21
    capex_to_revenue_ratio: float | None = None
    perpetual_growth_rate: float = 0.025
    risk_free_rate: float = 0.042
    equity_risk_premium: float = 0.055
    beta: float = 1.0
    current_market_price: float | None = None
    market_cap: float | None = None
    interest_expense: float | None = None
    default_pretax_cost_of_debt: float = 0.055
    exit_ebitda_multiple: float = 10.0
    nwc_to_incremental_revenue_ratio: float = 0.02
    _required_columns: tuple[str, ...] = field(
        default=(
            "revenue",
            "ebit",
            "depreciation_amortization",
            "capex",
            "operating_cash_flow",
            "cash_and_equivalents",
            "total_debt",
            "shares_outstanding",
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate constructor inputs."""
        if self.forecast_period < 1:
            raise ValueError("forecast_period must be at least 1.")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be between 0.0 and 1.0.")
        if self.perpetual_growth_rate < 0:
            raise ValueError("perpetual_growth_rate cannot be negative.")
        missing = [column for column in self._required_columns if column not in self.financials.columns]
        if missing:
            raise ValueError(f"Financial DataFrame is missing required columns: {', '.join(missing)}")
        if self.financials.empty:
            raise ValueError("Financial DataFrame must contain at least one fiscal period.")

    def run_valuation(self) -> dict[str, Any]:
        """Run the complete DCF workflow and return a valuation package."""
        historical = self.compute_historical_metrics()
        projections = self.project_fcff(historical)
        wacc_components = self.calculate_wacc(historical)
        valuation = self.calculate_valuation(projections, wacc_components["wacc"], self.perpetual_growth_rate)
        sensitivity = self.generate_sensitivity_matrix(projections)
        return {
            "historical_metrics": historical,
            "projections": projections,
            "wacc_components": wacc_components,
            "valuation": valuation,
            "sensitivity_matrix": sensitivity,
        }

    def compute_historical_metrics(self) -> dict[str, float]:
        """Extract latest balance-sheet items and historical operating ratios."""
        frame = self._chronological_financials()
        revenue = frame["revenue"].astype(float)
        latest = frame.iloc[-1]
        positive_revenue = revenue[revenue > 0]
        first_positive_revenue = float(positive_revenue.iloc[0]) if not positive_revenue.empty else 0.0
        latest_revenue = float(latest["revenue"])
        periods = max(len(positive_revenue) - 1, 1)
        cagr = (latest_revenue / first_positive_revenue) ** (1 / periods) - 1 if first_positive_revenue > 0 else 0.0

        ebit_margin = self._safe_ratio(frame["ebit"], revenue).tail(min(3, len(frame))).mean()
        da_to_revenue = self._safe_ratio(frame["depreciation_amortization"], revenue).tail(min(3, len(frame))).mean()
        capex_ratio = self.capex_to_revenue_ratio
        if capex_ratio is None:
            capex_ratio = self._safe_ratio(frame["capex"].abs(), revenue).tail(min(3, len(frame))).mean()
        capex_ratio = float(capex_ratio if pd.notna(capex_ratio) else 0.0)

        return {
            "revenue_cagr": float(cagr),
            "latest_revenue": latest_revenue,
            "ebit_margin": float(ebit_margin if pd.notna(ebit_margin) else 0.0),
            "da_to_revenue_ratio": float(da_to_revenue if pd.notna(da_to_revenue) else 0.0),
            "capex_to_revenue_ratio": capex_ratio,
            "operating_cash_flow": float(latest["operating_cash_flow"]),
            "cash_and_equivalents": float(latest["cash_and_equivalents"]),
            "total_debt": max(float(latest["total_debt"]), 0.0),
            "shares_outstanding": float(latest["shares_outstanding"]),
        }

    def project_fcff(self, historical: dict[str, float] | None = None) -> pd.DataFrame:
        """Project revenue, EBIT, D&A, CapEx, NWC investment, and FCFF."""
        metrics = historical or self.compute_historical_metrics()
        rows: list[dict[str, float]] = []
        prior_revenue = metrics["latest_revenue"]
        for year in range(1, self.forecast_period + 1):
            revenue = prior_revenue * (1 + metrics["revenue_cagr"])
            ebit = revenue * metrics["ebit_margin"]
            depreciation_amortization = revenue * metrics["da_to_revenue_ratio"]
            capex = revenue * metrics["capex_to_revenue_ratio"]
            delta_nwc = max(revenue - prior_revenue, 0.0) * self.nwc_to_incremental_revenue_ratio
            fcff = ebit * (1 - self.tax_rate) + depreciation_amortization - capex - delta_nwc
            rows.append({
                "year": float(year),
                "revenue": revenue,
                "ebit": ebit,
                "ebit_margin": metrics["ebit_margin"],
                "depreciation_amortization": depreciation_amortization,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "fcff": fcff,
                "ebitda": ebit + depreciation_amortization,
            })
            prior_revenue = revenue
        return pd.DataFrame(rows).set_index("year")

    def calculate_wacc(self, historical: dict[str, float] | None = None) -> dict[str, float]:
        """Calculate CAPM cost of equity, after-tax debt cost, and WACC."""
        metrics = historical or self.compute_historical_metrics()
        cost_of_equity = self.risk_free_rate + self.beta * self.equity_risk_premium
        total_debt = metrics["total_debt"]
        pretax_cost_of_debt = self.default_pretax_cost_of_debt
        if self.interest_expense is not None and total_debt > 0:
            pretax_cost_of_debt = max(self.interest_expense / total_debt, 0.0)
        after_tax_cost_of_debt = pretax_cost_of_debt * (1 - self.tax_rate)
        equity_value_for_weight = self._capital_structure_equity_value(metrics)
        capital = equity_value_for_weight + total_debt
        equity_weight = equity_value_for_weight / capital if capital > 0 else 1.0
        debt_weight = total_debt / capital if capital > 0 else 0.0
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        return {
            "cost_of_equity": cost_of_equity,
            "pretax_cost_of_debt": pretax_cost_of_debt,
            "after_tax_cost_of_debt": after_tax_cost_of_debt,
            "equity_weight": equity_weight,
            "debt_weight": debt_weight,
            "wacc": wacc,
        }

    def calculate_valuation(self, projections: pd.DataFrame, wacc: float, perpetual_growth_rate: float) -> dict[str, float]:
        """Calculate enterprise value, equity value, intrinsic price, and spread."""
        if wacc <= perpetual_growth_rate:
            raise ValueError("WACC must exceed perpetual growth rate for Gordon Growth terminal value.")
        discount_factors = pd.Series(
            [(1 + wacc) ** int(year) for year in projections.index], index=projections.index
        )
        pv_explicit_fcff = float((projections["fcff"] / discount_factors).sum())
        final_fcff = float(projections["fcff"].iloc[-1])
        terminal_value_gordon = final_fcff * (1 + perpetual_growth_rate) / (wacc - perpetual_growth_rate)
        pv_terminal_value_gordon = terminal_value_gordon / float(discount_factors.iloc[-1])
        terminal_value_exit_multiple = float(projections["ebitda"].iloc[-1]) * self.exit_ebitda_multiple
        pv_terminal_value_exit_multiple = terminal_value_exit_multiple / float(discount_factors.iloc[-1])
        enterprise_value = pv_explicit_fcff + pv_terminal_value_gordon
        metrics = self.compute_historical_metrics()
        equity_value = enterprise_value + metrics["cash_and_equivalents"] - metrics["total_debt"]
        shares = metrics["shares_outstanding"]
        intrinsic_share_price = equity_value / shares if shares > 0 else 0.0
        upside_downside = None
        if self.current_market_price and self.current_market_price > 0:
            upside_downside = intrinsic_share_price / self.current_market_price - 1
        return {
            "pv_explicit_fcff": pv_explicit_fcff,
            "terminal_value_gordon": terminal_value_gordon,
            "pv_terminal_value_gordon": pv_terminal_value_gordon,
            "terminal_value_exit_multiple": terminal_value_exit_multiple,
            "pv_terminal_value_exit_multiple": pv_terminal_value_exit_multiple,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "intrinsic_share_price": intrinsic_share_price,
            "upside_downside": float(upside_downside) if upside_downside is not None else float("nan"),
        }

    def generate_sensitivity_matrix(self, projections: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build a 5x5 WACC/perpetual-growth implied share-price sensitivity matrix."""
        forecast = projections if projections is not None else self.project_fcff()
        base_wacc = self.calculate_wacc()["wacc"]
        wacc_values = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, base_wacc + 0.005, base_wacc + 0.01]
        growth_values = [
            self.perpetual_growth_rate - 0.005,
            self.perpetual_growth_rate - 0.0025,
            self.perpetual_growth_rate,
            self.perpetual_growth_rate + 0.0025,
            self.perpetual_growth_rate + 0.005,
        ]
        data: list[list[float]] = []
        for row_wacc in wacc_values:
            row = []
            for growth in growth_values:
                if row_wacc <= growth:
                    row.append(float("nan"))
                else:
                    row.append(self.calculate_valuation(forecast, row_wacc, growth)["intrinsic_share_price"])
            data.append(row)
        return pd.DataFrame(
            data,
            index=[f"WACC {value:.2%}" for value in wacc_values],
            columns=[f"g {value:.2%}" for value in growth_values],
        )

    def _chronological_financials(self) -> pd.DataFrame:
        """Return numeric financials in chronological order."""
        frame = self.financials.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parsed = pd.to_datetime(frame.index, errors="coerce")
        if parsed.notna().sum() == len(frame.index):
            return frame.assign(_period_sort=parsed).sort_values("_period_sort").drop(columns="_period_sort")
        years = pd.Series(frame.index.astype(str), index=frame.index).str.extract(r"(\d{4})", expand=False)
        if years.notna().sum() == len(frame.index):
            return frame.assign(_period_sort=years.astype(int)).sort_values("_period_sort").drop(columns="_period_sort")
        return frame

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two Series, replacing invalid values with zero."""
        ratio = numerator.astype(float).divide(denominator.astype(float).replace(0.0, pd.NA))
        return ratio.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

    def _capital_structure_equity_value(self, metrics: dict[str, float]) -> float:
        """Infer an equity value for WACC weights when market capitalization is absent."""
        if self.market_cap is not None and self.market_cap > 0:
            return self.market_cap
        if self.current_market_price is not None and metrics["shares_outstanding"] > 0:
            return self.current_market_price * metrics["shares_outstanding"]
        book_equity = metrics["latest_revenue"] * 1.5 - metrics["total_debt"]
        return max(book_equity, metrics["latest_revenue"], 1.0)
