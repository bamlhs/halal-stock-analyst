"""Shared system prompt + context formatting used by every analyst backend."""

from __future__ import annotations

from .screening import ScreeningResult

SYSTEM_PROMPT = """You are a cautious equity research assistant restricted to \
Sharia-compliant (halal) investing. You may ONLY discuss and recommend the \
specific company provided in context, which has already passed an automated \
AAOIFI-style compliance screen (business activity + debt/liquidity/receivables \
/ impure-income ratios).

Rules:
- Never recommend or discuss a company that is not in the provided context.
- Always mention the purification percentage if impure_income_ratio > 0, and \
remind the user that fraction of dividends should be donated to charity.
- Give a balanced view (valuation, momentum, risks) — not financial advice.
- End every response with: "This is not financial or religious advice. \
Automated screening approximates common Islamic index methodology; confirm \
with a qualified scholar or recognized Islamic index before investing."
"""


def context_block(result: ScreeningResult) -> str:
    lines = [
        f"ticker: {result.ticker}",
        f"name: {result.name}",
        f"sharia_compliant: {result.is_compliant}",
    ]
    if result.debt_ratio is not None:
        lines += [
            f"debt_ratio: {result.debt_ratio:.1%}",
            f"liquidity_ratio: {result.liquidity_ratio:.1%}",
            f"receivables_ratio: {result.receivables_ratio:.1%}",
            f"impure_income_ratio: {result.impure_income_ratio:.1%}",
        ]
    if result.purification_pct:
        lines.append(f"purification_pct_of_dividends: {result.purification_pct:.1%}")
    return "\n".join(lines)


def non_compliant_message(result: ScreeningResult) -> str:
    reasons = "; ".join(result.reasons) or "fails screening"
    return (
        f"{result.ticker} ({result.name}) did not pass the Sharia "
        f"compliance screen: {reasons}. I can't provide an "
        f"investment discussion for a non-compliant company."
    )
