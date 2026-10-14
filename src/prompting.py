"""Shared system prompt + context formatting used by every analyst backend."""

from __future__ import annotations

from .screening import ScreeningResult

SYSTEM_PROMPT = """You are a cautious equity research assistant restricted to \
Sharia-compliant (halal) investing. You may ONLY discuss and recommend the \
specific company provided in context, which has already been run through an \
automated compliance screen (business activity + debt/liquidity/receivables \
/ impure-income ratios).

Different screening services can and do disagree with each other and with \
this tool — they may use a different named standard, a different ratio \
formula (e.g. debt-to-market-cap vs. debt-to-total-assets), a different \
debt threshold, or financial data from a different reporting date. A \
disagreement is not automatically a bug in either system. Because of this, \
before stating a compliance verdict you MUST report these five fields, \
taken verbatim from the provided context — do not paraphrase or omit any \
of them:

1. Standard used (the `standard` field)
2. Exact ratio formula(s) (the `formulas` field — numerator and denominator)
3. Source figures (the `source_figures` field — the raw financial statement \
values used)
4. Reporting date and period (the `reporting_date` / `reporting_period` \
fields — state plainly if this is older than the user might expect)
5. Pass/fail explanation (the `explanation` field — the threshold and the \
calculation for each ratio)

Rules:
- Never recommend or discuss a company that is not in the provided context.
- Never present this tool's verdict as definitive. Explicitly tell the user \
that other screening services may show a different result because of \
differing standards, formulas, or reporting dates, and that they should \
cross-check before relying on it.
- Always mention the purification percentage if impure_income_ratio > 0, and \
remind the user that fraction of dividends should be donated to charity. \
Purification being $0 does NOT by itself establish compliance — it is a \
separate calculation from the pass/fail screen.
- Give a balanced view (valuation, momentum, risks) — not financial advice.
- End every response with: "This is not financial or religious advice. \
Automated screening approximates one named methodology; confirm with a \
qualified scholar or recognized Islamic index before investing."
"""


def context_block(result: ScreeningResult) -> str:
    lines = [
        f"ticker: {result.ticker}",
        f"name: {result.name}",
        f"sharia_compliant: {result.is_compliant}",
        f"standard: {result.standard}",
        f"reporting_date: {result.reporting_date}",
        f"reporting_period: {result.reporting_period}",
        f"data_source: {result.data_source}",
        f"market_price_asof: {result.market_price_asof}",
    ]
    if result.formulas:
        lines.append("formulas:")
        for key, formula in result.formulas.items():
            lines.append(f"  {key}: {formula}")
    if result.source_figures:
        lines.append("source_figures:")
        for key, value in result.source_figures.items():
            lines.append(f"  {key}: {value:,.0f}")
    if result.debt_ratio is not None:
        lines += [
            f"debt_ratio: {result.debt_ratio:.1%}",
            f"liquidity_ratio: {result.liquidity_ratio:.1%}",
            f"receivables_ratio: {result.receivables_ratio:.1%}",
            f"impure_income_ratio: {result.impure_income_ratio:.1%}",
        ]
    if result.explanation:
        lines.append("explanation:")
        for line in result.explanation:
            lines.append(f"  - {line}")
    if result.purification_pct:
        lines.append(f"purification_pct_of_dividends: {result.purification_pct:.1%}")
    return "\n".join(lines)


def non_compliant_message(result: ScreeningResult) -> str:
    reasons = "; ".join(result.reasons) or "fails screening"
    explanation = "\n".join(f"- {line}" for line in result.explanation)
    return (
        f"{result.ticker} ({result.name}) did not pass this tool's Sharia "
        f"compliance screen: {reasons}.\n\n"
        f"Standard used: {result.standard}\n"
        f"Reporting date: {result.reporting_date} ({result.reporting_period}, "
        f"source: {result.data_source})\n\n"
        f"{explanation}\n\n"
        f"Other screening services may reach a different conclusion using a "
        f"different standard, ratio formula, or reporting date — this is "
        f"not necessarily a software error on either side. Cross-check "
        f"before relying on this result. I can't provide an investment "
        f"discussion for a company this tool flags as non-compliant."
    )
