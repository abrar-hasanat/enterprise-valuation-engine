# Enterprise Financial Intelligence and Valuation Engine

Python tooling for financial-statement normalization, discounted cash flow valuation, ratio diagnostics, sentiment summaries, and Markdown executive memos.

## Capabilities

* Load annual financial statements from CSV files or Yahoo Finance.
* Normalize common statement labels and accounting-formatted numbers into a canonical DataFrame.
* Project five years of free cash flow to firm and calculate CAPM-based WACC.
* Calculate enterprise value, equity value, intrinsic share price, and a 5x5 WACC and perpetual-growth sensitivity matrix.
* Produce ratio summaries, health flags, sentiment summaries, a JSON payload, and a Markdown executive memo.

## Specification alignment

### Valuation engine

* 5x5 WACC sensitivity: implemented by `DCFValuationEngine`.
* $23B+ valuation scenario validation: enforced by the DCF smoke script.
* One-click Morningstar-style PDF report: not implemented. The CLI writes JSON and optionally a Markdown memo.

### Out of scope for this repository

* Agile velocity: 10,000 Monte Carlo trials, P50/P80/P90 release milestones, and RICE feature scoring are not implemented.
* Demand forecasting: 42 months of order history, 91% accuracy, and a 3-week early stockout warning are not implemented.

## Financial methodology

```text
FCFF = EBIT × (1 - Tax Rate) + Depreciation and Amortization - CapEx - ΔNWC

Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium
After-Tax Cost of Debt = Pre-Tax Cost of Debt × (1 - Tax Rate)
WACC = (E / V) × Cost of Equity + (D / V) × After-Tax Cost of Debt

Terminal Value = FCFF₅ × (1 + g) / (WACC - g)
```

The model uses a five-year explicit forecast. When working-capital data is unavailable, the model uses 2.0% of incremental revenue for net working-capital investment. The default exit multiple is 10.0x EV/EBITDA.

## Quickstart

```bash
pip install -r requirements.txt
python main.py -t MSFT
```

Use a CSV input:

```bash
python main.py -c data/raw/company_financials.csv
```

Use the short options below for model overrides and exports:

```bash
python main.py -t MSFT -g 0.025 -o outputs/company_payload.json
```

Run `python main.py` with no arguments for interactive demo mode. Live ticker ingestion requires `yfinance`. Optional OpenAI memo drafting requires `OPENAI_API_KEY`.

## Expected CSV structure

The loader accepts period-oriented and metric-oriented annual CSV files. It recognizes common aliases for revenue, operating income, operating cash flow, capital expenditures, debt, cash, assets, liabilities, equity, and shares.

```csv
Fiscal Year,Sales,Operating Income,Net Income,Cash from Operations,CapEx,D&A,Total Debt,Total Cash,Assets,Liabilities,Book Value,Diluted Shares
2022,$9.7B,$1.46B,$1.01B,$1.52B,($510M),$360M,$2.15B,$875M,$11.1B,$6.2B,$4.9B,196M
2023,$10.6B,$1.65B,$1.14B,$1.71B,($550M),$395M,$2.20B,$940M,$11.9B,$6.5B,$5.4B,194M
2024,$11.5B,$1.84B,$1.28B,$1.93B,($610M),$430M,$2.25B,$1.00B,$12.7B,$6.85B,$5.85B,192M
```

## Verification

Run the executable smoke scripts and unit tests:

```bash
python test_dcf.py
python test_ratios.py
python test_sentiment.py
python test_loader.py
```

## Repository layout

```text
engine/data_loader.py: CSV and ticker ingestion
engine/dcf_model.py: DCF valuation and sensitivity analysis
engine/financial_ratios.py: ratio analytics and health diagnostics
engine/sentiment_analyzer.py: sentiment analysis and memo generation
main.py: CLI and interactive runner
```

## License

MIT License.
