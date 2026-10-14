"""Sharia-compliance screening.

Approximates one specific, named Sharia-screening convention — it is a
variant of the AAOIFI financial-ratio approach, but with ratios measured
against *market capitalization* rather than total assets. This is the
same convention commonly used by Dow Jones Islamic Market / S&P Shariah
style screens, but it is NOT identical to every provider:

- Some providers (e.g. services following AAOIFI Standard 21 literally)
  measure debt against *total assets*, not market cap. A company can
  pass one convention and fail the other at the same point in time.
- Thresholds vary by provider even within the market-cap convention
  (this tool uses 33%; some use 30%).
- "Debt" itself is ambiguous: total interest-bearing debt vs. total
  liabilities vs. a narrower conventional-financing-only figure. This
  tool uses Yahoo Finance's "Total Debt" line, which may not perfectly
  isolate interest-bearing debt.
- Non-permissible-*income* screens (e.g. Tabadulat-style) are a
  different check from the balance-sheet ratio screens here, and are
  not implemented by this tool.

Because of this, a result here disagreeing with another screening
service is NOT evidence that either one is "wrong" — it may simply
mean they're applying a different named standard, a different ratio
denominator, or financial data from a different reporting date. Always
compare the exact formula and reporting date, not just the headline
percentage, before trusting or discarding a result. See README.

1. Business-activity screen — exclude companies whose primary business
   is in a prohibited sector (interest-based finance, alcohol, tobacco,
   gambling, pork products, adult entertainment, conventional weapons).

2. Financial-ratio screens (all thresholds relative to market cap):
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

STANDARD_NAME = (
    "AAOIFI-derived market-cap-denominator screen (this tool's own variant — "
    "not an official AAOIFI, Dow Jones Islamic Market, S&P Shariah, or MSCI "
    "Islamic certification). Debt/liquidity/receivables ratios are measured "
    "against market capitalization, not total assets. See src/screening.py "
    "module docstring and README for why this can disagree with other "
    "screening services."
)

RATIO_FORMULAS = {
    "debt_ratio": "total_debt / market_cap  (threshold: < 33%)",
    "liquidity_ratio": "(cash_and_equivalents + short_term_investments) / market_cap  (threshold: < 33%)",
    "receivables_ratio": "accounts_receivable / market_cap  (threshold: < 49%)",
    "impure_income_ratio": "interest_income / total_revenue  (threshold: < 5%)",
}


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

    # Transparency fields — always populated so a result can be audited
    # against another screening service rather than trusted blindly.
    standard: str = STANDARD_NAME
    formulas: dict[str, str] = field(default_factory=dict)
    source_figures: dict[str, float] = field(default_factory=dict)
    reporting_date: str = "unknown"
    reporting_period: str = "unknown"
    data_source: str = "unknown"
    market_price_asof: str = "unknown"
    explanation: list[str] = field(default_factory=list)


def _business_activity_flag(f: CompanyFundamentals) -> str | None:
    text = f"{f.sector} {f.industry}".lower()
    for kw in PROHIBITED_SECTOR_KEYWORDS + PROHIBITED_INDUSTRY_KEYWORDS:
        if kw in text:
            return f"Prohibited business activity: sector='{f.sector}', industry='{f.industry}' (matched '{kw}')"
    return None


def _source_figures(f: CompanyFundamentals) -> dict[str, float]:
    return {
        "market_cap": f.market_cap,
        "total_debt": f.total_debt,
        "cash_and_equivalents": f.cash_and_equivalents,
        "short_term_investments": f.short_term_investments,
        "accounts_receivable": f.accounts_receivable,
        "total_revenue": f.total_revenue,
        "interest_income": f.interest_income,
    }


def screen_company(f: CompanyFundamentals) -> ScreeningResult:
    reasons: list[str] = []
    explanation: list[str] = []

    activity_flag = _business_activity_flag(f)
    if activity_flag:
        reasons.append(activity_flag)
        explanation.append(activity_flag)

    common_kwargs = dict(
        standard=STANDARD_NAME,
        formulas=dict(RATIO_FORMULAS),
        source_figures=_source_figures(f),
        reporting_date=f.reporting_date,
        reporting_period=f.reporting_period,
        data_source=f.data_source,
        market_price_asof=f.market_price_asof,
    )

    if f.market_cap <= 0:
        reasons.append("Missing/zero market cap — cannot compute financial ratios")
        explanation.append("market_cap is missing or zero — ratios could not be computed")
        return ScreeningResult(
            f.ticker, f.name, is_compliant=False, reasons=reasons,
            explanation=explanation, **common_kwargs,
        )

    debt_ratio = f.total_debt / f.market_cap
    liquidity_ratio = (f.cash_and_equivalents + f.short_term_investments) / f.market_cap
    receivables_ratio = f.accounts_receivable / f.market_cap
    impure_income_ratio = (f.interest_income / f.total_revenue) if f.total_revenue else 0.0

    def verdict(value: float, threshold: float) -> str:
        return "FAIL" if value >= threshold else "PASS"

    explanation.append(
        f"debt_ratio = total_debt ({f.total_debt:,.0f}) / market_cap ({f.market_cap:,.0f}) "
        f"= {debt_ratio:.1%} — threshold < {DEBT_RATIO_MAX:.0%} — {verdict(debt_ratio, DEBT_RATIO_MAX)}"
    )
    explanation.append(
        f"liquidity_ratio = (cash ({f.cash_and_equivalents:,.0f}) + short_term_investments "
        f"({f.short_term_investments:,.0f})) / market_cap ({f.market_cap:,.0f}) = {liquidity_ratio:.1%} "
        f"— threshold < {LIQUIDITY_RATIO_MAX:.0%} — {verdict(liquidity_ratio, LIQUIDITY_RATIO_MAX)}"
    )
    explanation.append(
        f"receivables_ratio = accounts_receivable ({f.accounts_receivable:,.0f}) / market_cap "
        f"({f.market_cap:,.0f}) = {receivables_ratio:.1%} — threshold < {RECEIVABLES_RATIO_MAX:.0%} "
        f"— {verdict(receivables_ratio, RECEIVABLES_RATIO_MAX)}"
    )
    explanation.append(
        f"impure_income_ratio = interest_income ({f.interest_income:,.0f}) / total_revenue "
        f"({f.total_revenue:,.0f}) = {impure_income_ratio:.1%} — threshold < {IMPURE_INCOME_RATIO_MAX:.0%} "
        f"— {verdict(impure_income_ratio, IMPURE_INCOME_RATIO_MAX)}"
    )

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
        explanation=explanation,
        **common_kwargs,
    )
