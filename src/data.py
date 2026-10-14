"""Fetch a stock universe and raw fundamentals.

Universe defaults to the S&P 500 constituent list maintained by the
open-source `datasets/s-and-p-500-companies` GitHub repo. Fundamentals
come from Yahoo Finance via `yfinance` (free, no API key).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

SP500_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)

# Used only if the live list can't be fetched (offline / rate limited).
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ADBE", "CSCO", "INTC", "AMD", "QCOM", "TXN", "ORCL", "CRM",
]


def get_universe_tickers() -> list[str]:
    """Return the ticker symbols to screen. Falls back to a static list on error."""
    try:
        resp = requests.get(SP500_CONSTITUENTS_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        return sorted(df["Symbol"].str.replace(".", "-", regex=False).unique().tolist())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to static ticker list: %s", exc)
        return FALLBACK_TICKERS


@dataclass
class CompanyFundamentals:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    total_debt: float
    cash_and_equivalents: float
    short_term_investments: float
    accounts_receivable: float
    total_revenue: float
    interest_income: float
    reporting_date: str
    reporting_period: str  # "quarterly" or "annual" — which statement the figures came from
    data_source: str
    market_price_asof: str  # when market_cap / price was last updated


def fetch_fundamentals(ticker: str) -> CompanyFundamentals | None:
    """Pull the fields needed for Sharia financial-ratio screening.

    Prefers the most recent *quarterly* balance sheet/income statement —
    stale annual figures are a common cause of disagreement with other
    screening services. Yahoo Finance data quality varies by ticker; any
    missing field is treated as 0 rather than failing the whole fetch.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        bs = t.quarterly_balance_sheet
        reporting_period = "quarterly"
        if bs is None or bs.empty:
            bs = t.balance_sheet
            reporting_period = "annual"

        fin = t.quarterly_financials
        if fin is None or fin.empty:
            fin = t.financials

        def bs_val(row: str) -> float:
            if bs is None or bs.empty or row not in bs.index:
                return 0.0
            return float(bs.loc[row].iloc[0] or 0.0)

        def fin_val(row: str) -> float:
            if fin is None or fin.empty or row not in fin.index:
                return 0.0
            return float(fin.loc[row].iloc[0] or 0.0)

        reporting_date = "unknown"
        if bs is not None and not bs.empty:
            reporting_date = str(bs.columns[0].date())

        market_price_asof = "unknown"
        regular_market_time = info.get("regularMarketTime")
        if regular_market_time:
            market_price_asof = str(regular_market_time)

        return CompanyFundamentals(
            ticker=ticker,
            name=info.get("longName", ticker),
            sector=info.get("sector", "") or "",
            industry=info.get("industry", "") or "",
            market_cap=float(info.get("marketCap") or 0.0),
            total_debt=bs_val("Total Debt") or float(info.get("totalDebt") or 0.0),
            cash_and_equivalents=bs_val("Cash And Cash Equivalents"),
            short_term_investments=bs_val("Other Short Term Investments"),
            accounts_receivable=bs_val("Receivables"),
            total_revenue=fin_val("Total Revenue") or float(info.get("totalRevenue") or 0.0),
            interest_income=fin_val("Interest Income"),
            reporting_date=reporting_date,
            reporting_period=reporting_period,
            data_source="Yahoo Finance (yfinance)",
            market_price_asof=market_price_asof,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
        return None
