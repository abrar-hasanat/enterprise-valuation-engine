# Enterprise Financial Intelligence & Valuation Engine

**Automated Multi-Year Statement Ingestion, DCF Modeling, Health Scorecarding, and Executive Memo Generation**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerics-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Ready-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-00A86B?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## Executive Summary & Value Proposition

`enterprise-valuation-engine` is an institutional-grade **Enterprise Financial Intelligence & Automated Valuation Platform** for Corporate Finance, Equity Research, Strategic Finance, and M&A teams.

The platform solves a common problem in financial analysis: **financial statement fragmentation**. Real-world company data arrives in inconsistent CSV layouts, exported accounting reports, ticker-based financial statements, and 10-K-style line-item formats. This engine normalizes those disparate sources into a canonical financial schema, then automatically produces:

- **Multi-year standardized financial statements** from messy CSVs or live public tickers.
- **Discounted Cash Flow (DCF) intrinsic valuation** with explicit FCFF forecasting.
- **CAPM/WACC-based cost of capital analysis** with terminal value estimation.
- **Corporate health scorecards** covering liquidity, solvency, profitability, leverage, ROIC, and cash-flow quality.
- **Rule-based market sentiment analysis** using recent company news or deterministic private-company fallback updates.
- **Executive-ready equity research memos** in polished Markdown, optionally enhanced with OpenAI when an API key is available.

The result is a repeatable pipeline that can transform raw financial inputs into valuation outputs, risk diagnostics, and board-ready narrative in seconds.

---

## Architecture & Core Capabilities

```text
Raw CSV / Public Ticker
        │
        ▼
UniversalFinancialDataLoader
        │  Canonical multi-year financial DataFrame
        ├─────────────────────────────┐
        ▼                             ▼
DCFValuationEngine             FinancialRatioEngine
        │                             │
        ▼                             ▼
Intrinsic Value + Sensitivity   Ratio Summary + Health Scorecard
        └───────────────┬─────────────┘
                        ▼
        ExecutiveIntelligenceSynthesizer
                        │
                        ▼
        JSON Payload + Executive Research Memo
```

### 1. Universal Ingestion Layer — `UniversalFinancialDataLoader`

The ingestion layer standardizes heterogeneous financial statement data into a single canonical schema.

**Key capabilities**

- **Alias mapping** for common financial statement labels such as `Sales`, `Total Revenue`, `Operating Income`, `CFO`, `CapEx`, `D&A`, `Total Borrowings`, and `Book Value`.
- **Auto-orientation detection** for both row-oriented financial statements and period-down tabular CSVs.
- **Accounting string normalization** for currency symbols, comma-separated values, parenthetical negatives, percentages, and K/M/B/T suffixes.
- **Canonical DataFrame output** indexed by fiscal period and ready for valuation, ratio diagnostics, and executive reporting.

### 2. Quantitative Valuation Engine — `DCFValuationEngine`

The DCF engine converts historical financials into a forward-looking intrinsic valuation.

**Core modeling features**

- Historical revenue CAGR and margin extraction.
- 5-year explicit **Free Cash Flow to Firm (FCFF)** projection.
- CAPM-derived cost of equity and debt-adjusted WACC calculation.
- Gordon Growth terminal value and exit multiple terminal value.
- Enterprise value, equity value, intrinsic share price, and implied upside/downside.
- Dynamic **5x5 WACC vs. perpetual growth sensitivity grid**.

### 3. Diagnostic Health Scorecard — `FinancialRatioEngine`

The ratio engine evaluates operating quality, balance-sheet durability, and distress risk across all reported fiscal periods.

**Ratio categories**

- **Profitability:** EBIT margin, net profit margin, ROE, ROIC.
- **Liquidity & solvency:** current/solvency ratio, debt-to-equity, net debt-to-EBITDA, interest coverage.
- **Cash-flow quality:** FCF conversion, CapEx-to-revenue, operating cash flow-to-debt.

**Automated diagnostic flags**

- Excessive leverage.
- Cash-flow stress.
- EBIT margin contraction.
- Solvency risk from weak interest coverage.

The scorecard returns a concise rating badge: **STRONG**, **MODERATE**, **WATCHLIST**, or **DISTRESSED**.

### 4. Executive Intelligence Synthesizer — `ExecutiveIntelligenceSynthesizer`

The synthesizer bridges quantitative model output with market narrative.

**Capabilities**

- Fetches recent public-company headlines through `yfinance` when available.
- Supports private-company/custom headlines for non-public targets.
- Scores headline sentiment with a lightweight financial keyword model.
- Extracts key themes such as growth, margin pressure, balance sheet, legal/regulatory risk, and execution risk.
- Generates a deterministic Markdown executive memo out of the box.
- Optionally uses OpenAI memo drafting when `OPENAI_API_KEY` is configured.

---

## Financial Methodology & Formula Reference

### Free Cash Flow to Firm (FCFF)

```text
FCFF = EBIT × (1 - Tax Rate) + Depreciation & Amortization - CapEx - ΔNWC
```

Where:

- `EBIT × (1 - Tax Rate)` = Net Operating Profit After Tax (NOPAT).
- `ΔNWC` defaults to 2.0% of incremental revenue when explicit working-capital data is unavailable.

### Weighted Average Cost of Capital (WACC)

```text
Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

After-Tax Cost of Debt = Pre-Tax Cost of Debt × (1 - Tax Rate)

WACC = (E / V) × Cost of Equity + (D / V) × After-Tax Cost of Debt
```

Where:

- `E` = equity value used for capital-structure weighting.
- `D` = total debt.
- `V` = enterprise capital base, calculated as `E + D`.

### Terminal Value — Gordon Growth

```text
Terminal Value = FCFF₅ × (1 + g) / (WACC - g)
```

Where:

- `FCFF₅` = final-year explicit projected free cash flow.
- `g` = perpetual growth rate.

### Terminal Value — Exit Multiple

```text
Terminal Value = Final-Year EBITDA × Exit EV / EBITDA Multiple
```

The default exit multiple is 10.0x EV/EBITDA and can be overridden in the valuation engine.

### Return on Invested Capital (ROIC)

```text
NOPAT = EBIT × (1 - Tax Rate)

Invested Capital = Total Debt + Shareholder Equity - Cash & Equivalents

ROIC = NOPAT / Invested Capital
```

ROIC provides a capital-efficiency lens for comparing operating returns against the company’s cost of capital.

---

## Quickstart & CLI Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/enterprise-valuation-engine.git
cd enterprise-valuation-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: The platform is designed with deterministic fallbacks for sentiment and memo generation. Live ticker ingestion requires `yfinance`; model-backed memo drafting requires an `OPENAI_API_KEY`.

### 3. Analyze a live public ticker

```bash
python main.py --ticker MSFT
```

With custom assumptions:

```bash
python main.py --ticker MSFT --wacc 0.09 --growth 0.025 --save-memo outputs/msft_memo.md
```

### 4. Analyze a custom CSV

```bash
python main.py --csv data/raw/company_financials.csv
```

Optional custom payload destination:

```bash
python main.py --csv data/raw/company_financials.csv --output outputs/company_payload.json
```

### 5. Use interactive terminal mode

```bash
python main.py
```

Interactive mode prompts for one of three workflows:

1. Analyze a live public ticker.
2. Upload/analyze a custom financial CSV file.
3. Run quick demo mode on pre-built synthetic financials.

### 6. Run smoke scripts directly

```bash
python test_dcf.py
python test_ratios.py
python test_sentiment.py
```

---

## Expected CSV Inputs

The loader accepts both period-down and metric-down CSV structures. Column names do not need to match exactly because the ingestion layer maps common financial aliases into the canonical schema.

Example period-down CSV:

```csv
Fiscal Year,Sales,Operating Income,Net Income,Cash from Operations,CapEx,D&A,Total Debt,Total Cash,Assets,Liabilities,Book Value,Diluted Shares
2022,$9.7B,$1.46B,$1.01B,$1.52B,($510M),$360M,$2.15B,$875M,$11.1B,$6.2B,$4.9B,196M
2023,$10.6B,$1.65B,$1.14B,$1.71B,($550M),$395M,$2.20B,$940M,$11.9B,$6.5B,$5.4B,194M
2024,$11.5B,$1.84B,$1.28B,$1.93B,($610M),$430M,$2.25B,$1.00B,$12.7B,$6.85B,$5.85B,192M
```

---

## Sample Terminal Output

```text
Enterprise Valuation Summary
========================================
Company: DemoCo Industrial Technologies
Target Price: $178.42
Enterprise Value: $35.5B
Equity Value: $34.3B
WACC: 8.9%
Terminal Growth: 2.5%
Upside/Downside: 42.7%
Health Rating: STRONG
Sentiment: BULLISH (+0.44)
Payload Saved: mock_cache/demo_valuation_cache.json

Key Ratio Snapshot
========================================
                                  2024
Operating (EBIT) Margin          16.0%
Net Profit Margin                11.1%
Return on Equity                 21.9%
Return on Invested Capital       20.6%
Current / Solvency Ratio          1.9x
Debt-to-Equity                    0.4x
Net Debt-to-EBITDA                0.6x
Interest Coverage                14.9x
Free Cash Flow Conversion         1.0x
CapEx-to-Revenue                  5.3%
Operating Cash Flow-to-Debt      85.8%
EBITDA                           $2.3B
Net Debt                         $1.2B

5x5 WACC / Perpetual Growth Sensitivity Matrix
========================================
             g 2.00%  g 2.25%  g 2.50%  g 2.75%  g 3.00%
WACC 7.90%   209.31   219.14   229.89   241.70   254.77
WACC 8.40%   186.54   194.19   202.45   211.39   221.10
WACC 8.90%   167.91   173.99   180.49   187.45   194.94
WACC 9.40%   152.42   157.35   162.57   168.12   174.03
WACC 9.90%   139.34   143.41   147.69   152.20   156.96

Executive Memo
========================================
# Equity Research & Strategic Memo: DemoCo Industrial Technologies

## 1. Executive Summary & Investment Verdict
- Investment verdict: Constructive, subject to valuation and execution discipline.
- Intrinsic target price: $178.42; implied upside/downside: 42.7%.
- Financial health badge: STRONG; market sentiment: BULLISH (+0.44).
```

---

## Engineering Standards

This repository is designed for transparent, auditable financial analytics.

### Production-oriented implementation principles

- **Type hints and docstrings:** Core modules are typed and documented for maintainability.
- **Canonical schema design:** The data loader standardizes messy raw inputs before analytics run.
- **Zero-division safeguards:** Ratio and valuation calculations use safe division and validation guards for missing/zero denominators.
- **Deterministic offline fallback mode:** Sentiment and memo generation work without paid APIs or live news feeds.
- **Optional integrations:** `yfinance` and OpenAI are used opportunistically while preserving offline execution paths.
- **Serializable outputs:** Pipeline results are exported to JSON for dashboards, cache layers, and demo rendering.
- **Modular architecture:** Ingestion, valuation, diagnostics, sentiment, and orchestration are isolated into focused components.

---

## Repository Layout

```text
enterprise-valuation-engine/
├── engine/
│   ├── data_loader.py           # UniversalFinancialDataLoader
│   ├── dcf_model.py             # DCFValuationEngine
│   ├── financial_ratios.py      # FinancialRatioEngine
│   └── sentiment_analyzer.py    # ExecutiveIntelligenceSynthesizer
├── main.py                      # Unified CLI and interactive runner
├── test_dcf.py                  # DCF smoke script
├── test_ratios.py               # Ratio diagnostics smoke script
├── test_sentiment.py            # Full intelligence synthesis smoke script
└── README.md
```

---

## Intended Users

- **Corporate Finance teams** evaluating operating plans, capital allocation, and intrinsic value.
- **Equity Research analysts** generating repeatable valuation models and memo-ready outputs.
- **Strategic M&A teams** screening acquisition targets and assessing balance-sheet risk.
- **FP&A and CFO organizations** standardizing financial diagnostics across business units.
- **Builders of AI financial agents** needing structured valuation and reporting primitives.

---

## License

MIT License. See repository license metadata for details.
