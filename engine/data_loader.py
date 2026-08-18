"""Universal financial data ingestion utilities.

This module normalizes heterogeneous company CSV uploads and Yahoo Finance ticker
history into a single canonical financial statement schema suitable for valuation
models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Any, ClassVar

import pandas as pd


class UniversalFinancialDataLoader:
    """Load and normalize financial statement data from CSVs or Yahoo Finance.

    The loader accepts messy real-world CSVs whose metrics may be arranged either
    down rows (line items in the first column and periods across columns) or
    across columns (periods down rows and line items across columns). Values are
    converted to floats, line-item names are matched through a canonical alias
    map, and absent non-critical metrics are filled with ``0.0``.
    """

    CANONICAL_SCHEMA: ClassVar[dict[str, tuple[str, ...]]] = {
        "revenue": (
            "revenue",
            "total revenue",
            "sales",
            "net sales",
            "turnover",
            "gross revenue",
        ),
        "ebit": (
            "ebit",
            "operating income",
            "operating profit",
            "income from operations",
            "operating earnings",
        ),
        "net_income": (
            "net income",
            "net earnings",
            "profit after tax",
            "bottom line",
            "net profit",
        ),
        "operating_cash_flow": (
            "operating cash flow",
            "cash from operations",
            "cfo",
            "net cash provided by operating activities",
            "operating activities cash flow",
        ),
        "capex": (
            "capex",
            "capital expenditures",
            "capital expenditure",
            "purchase of property equipment",
            "purchase of property and equipment",
            "purchases of property and equipment",
            "property plant equipment purchases",
        ),
        "depreciation_amortization": (
            "d a",
            "d and a",
            "da",
            "depreciation amortization",
            "depreciation and amortization",
            "depreciation expense",
            "amortization depreciation",
        ),
        "total_debt": (
            "total debt",
            "long term debt",
            "short long term debt",
            "short and long term debt",
            "total borrowings",
            "borrowings",
        ),
        "cash_and_equivalents": (
            "cash and cash equivalents",
            "cash cash equivalents",
            "total cash",
            "marketable securities",
            "cash equivalents",
            "cash and short term investments",
        ),
        "total_assets": ("total assets", "assets"),
        "total_liabilities": ("total liabilities", "liabilities"),
        "shareholder_equity": (
            "shareholders equity",
            "shareholder equity",
            "stockholders equity",
            "stockholders' equity",
            "total equity",
            "book value",
            "total stockholder equity",
        ),
        "shares_outstanding": (
            "shares outstanding",
            "diluted shares",
            "common shares",
            "weighted average diluted shares outstanding",
            "ordinary shares number",
        ),
    }

    REQUIRED_METRICS: ClassVar[tuple[str, ...]] = ("revenue",)

    @classmethod
    def load_csv(cls, filepath_or_buffer: str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
        """Load a CSV and return a canonical, numeric financial DataFrame.

        Args:
            filepath_or_buffer: Local path or file-like object accepted by
                :func:`pandas.read_csv`.

        Returns:
            DataFrame indexed by fiscal period with one column per canonical
            metric in :attr:`CANONICAL_SCHEMA`.

        Raises:
            ValueError: If no known financial metrics can be identified.
        """
        raw = pd.read_csv(filepath_or_buffer, dtype=str).dropna(how="all")
        if raw.empty:
            raise ValueError("CSV contains no rows to ingest.")

        oriented = cls._orient_csv_frame(raw)
        normalized = cls._normalize_metric_columns(oriented)
        if not normalized:
            raise ValueError("CSV does not contain recognizable financial metrics.")

        result = pd.DataFrame(normalized)
        result.index = cls._standardize_period_index(oriented.index)
        result.index.name = "period"
        return cls._finalize_frame(result)

    @classmethod
    def load_ticker(cls, ticker_symbol: str) -> pd.DataFrame:
        """Load annual statement history from Yahoo Finance into canonical schema.

        Args:
            ticker_symbol: Public market ticker supported by Yahoo Finance.

        Returns:
            DataFrame indexed by fiscal period with canonical financial metrics.
        """
        import yfinance as yf

        ticker = yf.Ticker(ticker_symbol)
        statements = [ticker.financials, ticker.cashflow, ticker.balance_sheet]
        merged: dict[str, pd.Series] = {}

        for statement in statements:
            if statement is None or statement.empty:
                continue
            for row_label, row in statement.iterrows():
                canonical = cls._canonical_key(row_label)
                if canonical and canonical not in merged:
                    merged[canonical] = row.map(cls._clean_numeric_value)

        if not merged:
            raise ValueError(f"No financial statement data found for ticker {ticker_symbol!r}.")

        result = pd.DataFrame(merged)
        result.index = cls._standardize_period_index(result.index)
        result.index.name = "period"
        return cls._finalize_frame(result)

    @classmethod
    def _orient_csv_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Return data with fiscal periods as rows and raw metrics as columns."""
        frame = frame.dropna(axis=1, how="all").copy()
        header_hits = sum(1 for col in frame.columns if cls._canonical_key(col))
        first_col = frame.iloc[:, 0] if len(frame.columns) else pd.Series(dtype=str)
        first_col_hits = sum(1 for value in first_col.dropna() if cls._canonical_key(value))

        if first_col_hits > header_hits:
            metric_col = frame.columns[0]
            transposed = frame.set_index(metric_col).transpose()
            transposed.index.name = "period"
            return transposed

        period_col = cls._find_period_column(frame)
        if period_col is not None:
            return frame.set_index(period_col)
        return frame

    @classmethod
    def _normalize_metric_columns(cls, frame: pd.DataFrame) -> dict[str, pd.Series]:
        """Map raw columns to canonical metric names and clean numeric values."""
        normalized: dict[str, pd.Series] = {}
        for column in frame.columns:
            canonical = cls._canonical_key(column)
            if canonical is None:
                continue
            cleaned = frame[column].map(cls._clean_numeric_value).astype(float)
            if canonical in normalized:
                normalized[canonical] = normalized[canonical].combine_first(cleaned)
            else:
                normalized[canonical] = cleaned
        return normalized

    @classmethod
    def _finalize_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Add missing schema fields, enforce floats, and sort columns."""
        result = frame.copy()
        for metric in cls.CANONICAL_SCHEMA:
            if metric not in result.columns:
                result[metric] = 0.0
        result = result.loc[:, list(cls.CANONICAL_SCHEMA)]
        result = result.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        missing_required = [metric for metric in cls.REQUIRED_METRICS if result[metric].eq(0.0).all()]
        if missing_required:
            raise ValueError(f"Missing required financial metrics: {', '.join(missing_required)}")
        return result

    @classmethod
    def _canonical_key(cls, label: Any) -> str | None:
        """Return canonical metric key for a noisy label, if recognized."""
        cleaned = cls._standardize_label(label)
        for canonical, aliases in cls.CANONICAL_SCHEMA.items():
            if cleaned == cls._standardize_label(canonical):
                return canonical
            if cleaned in {cls._standardize_label(alias) for alias in aliases}:
                return canonical
        return None

    @staticmethod
    def _standardize_label(label: Any) -> str:
        """Normalize a label for alias matching by stripping punctuation/case."""
        text = "" if pd.isna(label) else str(label)
        text = text.replace("&", " and ")
        text = re.sub(r"[_\-/]+", " ", text.lower())
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_numeric_value(val: Any) -> float:
        """Parse currency, percent, accounting negatives, and K/M/B/T values."""
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)

        text = str(val).strip()
        if not text or text.lower() in {"nan", "none", "null", "-", "n/a", "na"}:
            return 0.0

        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("() ").replace(",", "")
        text = re.sub(r"[$€£¥]", "", text).strip()
        text = text.replace("%", "")
        multiplier = 1.0
        suffix_match = re.search(r"([kmbt])\s*$", text, flags=re.IGNORECASE)
        if suffix_match:
            multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[suffix_match.group(1).lower()]
            text = text[: suffix_match.start()].strip()

        number_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not number_match:
            return 0.0
        value = float(number_match.group(0)) * multiplier
        return -abs(value) if negative else value

    @staticmethod
    def _find_period_column(frame: pd.DataFrame) -> str | None:
        """Identify a likely fiscal-period column in a period-down CSV layout."""
        for column in frame.columns:
            cleaned = UniversalFinancialDataLoader._standardize_label(column)
            if cleaned in {"period", "date", "year", "fiscal year", "fiscal period"}:
                return str(column)
        return None

    @staticmethod
    def _standardize_period_index(index: pd.Index) -> pd.Index:
        """Create stable string fiscal-period labels from dates or raw headers."""
        labels: list[str] = []
        for raw in index:
            text = str(raw).strip()
            if re.fullmatch(r"\d{4}", text) or re.fullmatch(r"(?i)fy\s*\d{4}", text):
                labels.append(text)
                continue
            parsed = pd.to_datetime(text, errors="coerce")
            labels.append(parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else text)
        return pd.Index(labels)
