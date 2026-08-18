"""Unified orchestration runner for the enterprise valuation engine.

The script supports public ticker ingestion, custom CSV uploads, and a quick demo
mode. It connects the data loader, DCF valuation model, financial-ratio engine,
and executive intelligence synthesizer into one end-to-end terminal workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("mock_cache/demo_valuation_cache.json")
DEFAULT_MEMO_PATH = Path("mock_cache/executive_memo.md")


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime configuration for the valuation pipeline."""

    ticker: str | None
    csv_path: Path | None
    wacc_override: float | None
    perpetual_growth_rate: float
    output_path: Path
    memo_path: Path | None
    interactive_demo: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ticker, CSV, model overrides, and exports."""
    parser = argparse.ArgumentParser(
        description="Run the end-to-end enterprise valuation and executive intelligence pipeline."
    )
    parser.add_argument("-t", "--ticker", help="Public ticker symbol to pull live financials via Yahoo Finance.")
    parser.add_argument("-c", "--csv", dest="csv_path", help="Path to a custom financial statement CSV file.")
    parser.add_argument("--wacc", type=float, help="Optional discount-rate override, e.g. 0.09 for 9.0%%.")
    parser.add_argument(
        "-g",
        "--growth",
        type=float,
        default=0.025,
        help="Perpetual growth-rate assumption. Defaults to 0.025.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output directory or JSON filepath for the intelligence payload.",
    )
    parser.add_argument(
        "--save-memo",
        nargs="?",
        const=str(DEFAULT_MEMO_PATH),
        default=None,
        help="Optional Markdown filepath for exporting the executive memo.",
    )
    return parser.parse_args(argv)


def interactive_config() -> RunnerConfig:
    """Prompt the user for ticker, CSV, or demo mode when no CLI args are supplied."""
    print("Enterprise Valuation Engine")
    print("=" * 40)
    print("1) Analyze a live public ticker")
    print("2) Upload/analyze a custom financial CSV file")
    print("3) Run quick demo mode on synthetic financials")
    selection = input("Select an option [1-3]: ").strip() or "3"

    ticker: str | None = None
    csv_path: Path | None = None
    interactive_demo = False
    if selection == "1":
        ticker = input("Ticker symbol (e.g., AAPL): ").strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol cannot be blank.")
    elif selection == "2":
        raw_path = input("CSV filepath: ").strip()
        if not raw_path:
            raise ValueError("CSV filepath cannot be blank.")
        csv_path = Path(raw_path).expanduser()
    elif selection == "3":
        interactive_demo = True
    else:
        raise ValueError("Invalid selection. Please choose 1, 2, or 3.")

    growth = _prompt_optional_float("Perpetual growth rate [0.025]: ", default=0.025)
    wacc = _prompt_optional_float("Optional WACC override, blank to infer: ", default=None)
    output_text = input(f"Output JSON path [{DEFAULT_OUTPUT_PATH}]: ").strip()
    memo_text = input("Save memo markdown? Enter filepath or leave blank: ").strip()
    return RunnerConfig(
        ticker=ticker,
        csv_path=csv_path,
        wacc_override=wacc,
        perpetual_growth_rate=growth,
        output_path=Path(output_text).expanduser() if output_text else DEFAULT_OUTPUT_PATH,
        memo_path=Path(memo_text).expanduser() if memo_text else None,
        interactive_demo=interactive_demo,
    )


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    """Convert parsed arguments into a validated runner configuration."""
    if args.ticker and args.csv_path:
        raise ValueError("Please provide either --ticker or --csv, not both.")
    if args.growth < 0:
        raise ValueError("Perpetual growth rate cannot be negative.")
    if args.wacc is not None and args.wacc <= args.growth:
        raise ValueError("WACC override must exceed the perpetual growth rate.")

    return RunnerConfig(
        ticker=args.ticker.upper() if args.ticker else None,
        csv_path=Path(args.csv_path).expanduser() if args.csv_path else None,
        wacc_override=args.wacc,
        perpetual_growth_rate=args.growth,
        output_path=Path(args.output).expanduser(),
        memo_path=Path(args.save_memo).expanduser() if args.save_memo else None,
        interactive_demo=not args.ticker and not args.csv_path,
    )


def run_pipeline(config: RunnerConfig) -> dict[str, Any]:
    """Run ingestion, valuation, ratios, sentiment, memo generation, and exports."""
    from engine.dcf_model import DCFValuationEngine
    from engine.financial_ratios import FinancialRatioEngine
    from engine.sentiment_analyzer import ExecutiveIntelligenceSynthesizer

    financials, company_name, ticker_or_name = load_financials(config)

    dcf_engine = DCFValuationEngine(financials, perpetual_growth_rate=config.perpetual_growth_rate)
    dcf_output = dcf_engine.run_valuation()
    if config.wacc_override is not None:
        dcf_output = apply_wacc_override(dcf_engine, dcf_output, config.wacc_override, config.perpetual_growth_rate)

    ratio_engine = FinancialRatioEngine(financials)
    ratios_summary = ratio_engine.get_summary_table()
    health_scorecard = ratio_engine.generate_health_scorecard()

    synthesizer = ExecutiveIntelligenceSynthesizer()
    articles = synthesizer.fetch_company_news(ticker_or_name)
    news_sentiment = synthesizer.analyze_news_sentiment(articles)
    valuation_summary = build_valuation_summary(dcf_engine, dcf_output, config.perpetual_growth_rate)
    memo = synthesizer.generate_executive_memo(
        company_name=company_name,
        valuation_summary=valuation_summary,
        ratios_summary=ratios_summary,
        health_scorecard=health_scorecard,
        news_sentiment=news_sentiment,
    )

    payload = {
        "company_name": company_name,
        "ticker_or_name": ticker_or_name,
        "source": source_label(config),
        "financials": jsonify(financials),
        "dcf": jsonify(dcf_output),
        "valuation_summary": jsonify(valuation_summary),
        "ratios_summary": jsonify(ratios_summary),
        "health_scorecard": jsonify(health_scorecard),
        "news_sentiment": jsonify(news_sentiment),
        "executive_memo": memo,
        "intelligence_payload": synthesizer.export_intelligence_payload(),
    }
    save_json_payload(payload, config.output_path)
    if config.memo_path:
        save_text(memo, config.memo_path)
    print_terminal_summary(company_name, valuation_summary, ratios_summary, health_scorecard, news_sentiment, memo, config)
    return payload


def load_financials(config: RunnerConfig) -> tuple[pd.DataFrame, str, str]:
    """Load financials from ticker, CSV, or synthetic demo data."""
    from engine.data_loader import UniversalFinancialDataLoader

    if config.ticker:
        try:
            frame = UniversalFinancialDataLoader.load_ticker(config.ticker)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load ticker '{config.ticker}'. Check the ticker symbol, network access, and Yahoo Finance availability."
            ) from exc
        if frame.empty:
            raise RuntimeError(f"Ticker '{config.ticker}' returned no financial statement data.")
        return frame, config.ticker, config.ticker

    if config.csv_path:
        if not config.csv_path.exists() or not config.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {config.csv_path}")
        try:
            frame = UniversalFinancialDataLoader.load_csv(config.csv_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read and normalize CSV file '{config.csv_path}'.") from exc
        if frame.empty:
            raise RuntimeError(f"CSV file '{config.csv_path}' did not contain usable financial data.")
        company_name = config.csv_path.stem.replace("_", " ").replace("-", " ").title()
        return frame, company_name, company_name

    return build_demo_financials(), "DemoCo Industrial Technologies", "DemoCo Industrial Technologies"


def apply_wacc_override(
    dcf_engine: DCFValuationEngine,
    dcf_output: dict[str, Any],
    wacc_override: float,
    perpetual_growth_rate: float,
) -> dict[str, Any]:
    """Recalculate valuation and sensitivity outputs using a CLI WACC override."""
    projections = dcf_output["projections"]
    valuation = dcf_engine.calculate_valuation(projections, wacc_override, perpetual_growth_rate)
    wacc_components = dict(dcf_output["wacc_components"])
    wacc_components["wacc"] = wacc_override
    dcf_output = dict(dcf_output)
    dcf_output["valuation"] = valuation
    dcf_output["wacc_components"] = wacc_components
    dcf_output["sensitivity_matrix"] = build_sensitivity_matrix(dcf_engine, projections, wacc_override, perpetual_growth_rate)
    return dcf_output


def build_sensitivity_matrix(
    dcf_engine: DCFValuationEngine,
    projections: pd.DataFrame,
    base_wacc: float,
    base_growth: float,
) -> pd.DataFrame:
    """Build a 5x5 sensitivity table around explicit WACC/growth assumptions."""
    wacc_values = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, base_wacc + 0.005, base_wacc + 0.01]
    growth_values = [base_growth - 0.005, base_growth - 0.0025, base_growth, base_growth + 0.0025, base_growth + 0.005]
    matrix: list[list[float]] = []
    for wacc in wacc_values:
        row = []
        for growth in growth_values:
            row.append(
                float("nan")
                if wacc <= growth
                else dcf_engine.calculate_valuation(projections, wacc, growth)["intrinsic_share_price"]
            )
        matrix.append(row)
    return pd.DataFrame(
        matrix,
        index=[f"WACC {value:.2%}" for value in wacc_values],
        columns=[f"g {value:.2%}" for value in growth_values],
    )


def build_valuation_summary(
    dcf_engine: DCFValuationEngine,
    dcf_output: dict[str, Any],
    perpetual_growth_rate: float,
) -> dict[str, Any]:
    """Flatten key DCF outputs for memo and payload consumers."""
    return {
        **dcf_output["valuation"],
        "wacc": dcf_output["wacc_components"]["wacc"],
        "perpetual_growth_rate": perpetual_growth_rate,
        "forecast_period": dcf_engine.forecast_period,
        "sensitivity_matrix": dcf_output["sensitivity_matrix"],
    }


def build_demo_financials() -> pd.DataFrame:
    """Create demo financials by routing synthetic CSV data through the loader."""
    from engine.data_loader import UniversalFinancialDataLoader

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


def save_json_payload(payload: dict[str, Any], output_path: Path) -> Path:
    """Save the compiled intelligence payload to a JSON file."""
    destination = resolve_output_json_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")
    return destination


def save_text(content: str, output_path: Path) -> Path:
    """Save Markdown/text content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def resolve_output_json_path(output_path: Path) -> Path:
    """Treat extension-less or directory output values as payload directories."""
    if output_path.suffix.lower() == ".json":
        return output_path
    return output_path / DEFAULT_OUTPUT_PATH.name


def print_terminal_summary(
    company_name: str,
    valuation_summary: dict[str, Any],
    ratios_summary: pd.DataFrame,
    health_scorecard: dict[str, Any],
    news_sentiment: dict[str, Any],
    memo: str,
    config: RunnerConfig,
) -> None:
    """Print a clean valuation verdict, key ratios, sentiment, and memo."""
    output_path = resolve_output_json_path(config.output_path)
    print("\nEnterprise Valuation Summary")
    print("=" * 40)
    print(f"Company: {company_name}")
    print(f"Target Price: {format_currency(valuation_summary.get('intrinsic_share_price'), precision=2)}")
    print(f"Enterprise Value: {format_currency(valuation_summary.get('enterprise_value'))}")
    print(f"Equity Value: {format_currency(valuation_summary.get('equity_value'))}")
    print(f"WACC: {format_percent(valuation_summary.get('wacc'))}")
    print(f"Terminal Growth: {format_percent(valuation_summary.get('perpetual_growth_rate'))}")
    print(f"Upside/Downside: {format_percent(valuation_summary.get('upside_downside'))}")
    print(f"Health Rating: {health_scorecard.get('rating', 'N/A')}")
    print(
        "Sentiment: "
        f"{news_sentiment.get('sentiment_classification', 'NEUTRAL')} "
        f"({float(news_sentiment.get('sentiment_score', 0.0)):+.2f})"
    )
    print(f"Payload Saved: {output_path}")
    if config.memo_path:
        print(f"Memo Saved: {config.memo_path}")

    print("\nKey Ratio Snapshot")
    print("=" * 40)
    if ratios_summary.empty:
        print("Ratio summary unavailable.")
    else:
        print(ratios_summary.iloc[:, [-1]].to_string())

    print("\nExecutive Memo")
    print("=" * 40)
    print(memo)


def source_label(config: RunnerConfig) -> str:
    """Return a readable data-source label for the payload."""
    if config.ticker:
        return "ticker"
    if config.csv_path:
        return "csv"
    return "demo"


def _prompt_optional_float(prompt: str, default: float | None) -> float | None:
    """Prompt for an optional float with a default value."""
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric input: {raw}") from exc


def format_percent(value: Any) -> str:
    """Format a scalar as a percentage for terminal output."""
    import pandas as pd

    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def format_currency(value: Any, precision: int = 1) -> str:
    """Format currency/share values for terminal output."""
    import pandas as pd

    if value is None or pd.isna(value):
        return "N/A"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    absolute = abs(amount)
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if absolute >= divisor:
            return f"{sign}${absolute / divisor:.{precision}f}{suffix}"
    return f"{sign}${absolute:.{precision}f}"


def jsonify(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-serializable values."""
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return value.astype(object).where(pd.notna(value), None).to_dict(orient="index")
    if isinstance(value, pd.Series):
        return value.astype(object).where(pd.notna(value), None).to_dict()
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [jsonify(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning a process exit code."""
    try:
        if argv is None:
            argv = sys.argv[1:]
        config = interactive_config() if not argv else config_from_args(parse_args(argv))
        run_pipeline(config)
        return 0
    except KeyboardInterrupt:
        print("\nExecution cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
