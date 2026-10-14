"""Sharia-compliance screening.

Approximates the widely used AAOIFI / Dow Jones Islamic Market style
screening methodology:

1. Business-activity screen — exclude companies whose primary business
   is in a prohibited sector (interest-based finance, alcohol, tobacco,
   gambling, pork products, adult entertainment, conventional weapons).

2. Financial-ratio screens (all thresholds relative to market cap,
   the common AAOIFI convention):
   - Debt ratio:            total_debt / market_cap        < 33%
   - Liquidity ratio:       (cash + short_term_investments) / market_cap < 33%
   - Receivables ratio:     accounts_receivable / market_cap < 49%
   - Impure income ratio:   interest_income / total_revenue  < 5%
     (small amounts of incidental interest income are tolerated by most
     boards, but that fraction of dividends must be purified/donated)

This is a best-effort automated approximation, not a fatwa. See README
for the disclaimer — users should confirm against a recognized Islamic
index provider or a qualified scholar before investing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import CompanyFundamentals

PROHIBITED_SECTOR_KEYWORDS = [
    "bank", "insurance", "capital markets", "financial services",
    "casino", "gambling", "tobacco", "brewer", "distiller", "wineries",
    "alcohol", "adult", "defense", "aerospace & defense",
]

PROHIBITED_INDUSTRY_KEYWORDS = [
    "bank", "insurance", "credit services", "capital markets",
    "casino", "gambling", "resorts & casinos", "tobacco",
    "beverages—brewers", "beverages—wineries & distilleries",
    "pornography", "adult entertainment", "aerospace & defense",
]

DEBT_RATIO_MAX = 0.33
LIQUIDITY_RATIO_MAX = 0.33
RECEIVABLES_RATIO_MAX = 0.49
IMPURE_INCOME_RATIO_MAX = 0.05


@dataclass
class ScreeningResult:
    ticker: str
    name: str
    is_compliant: bool
    reasons: list[str] = field(default_factory=list)
    debt_ratio: float | None = None
    liquidity_ratio: float | None = None
    receivables_ratio: float | None = None
    impure_income_ratio: float | None = None
    purification_pct: float | None = None  # % of dividends to donate to charity


def _business_activity_flag(f: CompanyFundamentals) -> str | None:
    text = f"{f.sector} {f.industry}".lower()
    for kw in PROHIBITED_SECTOR_KEYWORDS + PROHIBITED_INDUSTRY_KEYWORDS:
        if kw in text:
            return f"Prohibited business activity: sector='{f.sector}', industry='{f.industry}' (matched '{kw}')"
    return None


def screen_company(f: CompanyFundamentals) -> ScreeningResult:
    reasons: list[str] = []

    activity_flag = _business_activity_flag(f)
    if activity_flag:
        reasons.append(activity_flag)

    if f.market_cap <= 0:
        reasons.append("Missing/zero market cap — cannot compute financial ratios")
        return ScreeningResult(f.ticker, f.name, is_compliant=False, reasons=reasons)

    debt_ratio = f.total_debt / f.market_cap
    liquidity_ratio = (f.cash_and_equivalents + f.short_term_investments) / f.market_cap
    receivables_ratio = f.accounts_receivable / f.market_cap
    impure_income_ratio = (f.interest_income / f.total_revenue) if f.total_revenue else 0.0

    if debt_ratio >= DEBT_RATIO_MAX:
        reasons.append(f"Debt ratio {debt_ratio:.1%} >= {DEBT_RATIO_MAX:.0%} threshold")
    if liquidity_ratio >= LIQUIDITY_RATIO_MAX:
        reasons.append(f"Cash+investments ratio {liquidity_ratio:.1%} >= {LIQUIDITY_RATIO_MAX:.0%} threshold")
    if receivables_ratio >= RECEIVABLES_RATIO_MAX:
        reasons.append(f"Receivables ratio {receivables_ratio:.1%} >= {RECEIVABLES_RATIO_MAX:.0%} threshold")
    if impure_income_ratio >= IMPURE_INCOME_RATIO_MAX:
        reasons.append(f"Interest/impure income ratio {impure_income_ratio:.1%} >= {IMPURE_INCOME_RATIO_MAX:.0%} threshold")

    is_compliant = len(reasons) == 0

    return ScreeningResult(
        ticker=f.ticker,
        name=f.name,
        is_compliant=is_compliant,
        reasons=reasons,
        debt_ratio=debt_ratio,
        liquidity_ratio=liquidity_ratio,
        receivables_ratio=receivables_ratio,
        impure_income_ratio=impure_income_ratio,
        purification_pct=impure_income_ratio if is_compliant else None,
    )
