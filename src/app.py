"""FastAPI app: screens the universe once at startup, then answers
chat questions restricted to the halal-compliant subset.

Run locally:   uvicorn src.app:app --reload
Run in Colab:  see notebooks/Halal_Stock_Analyst_Colab.ipynb
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .data import fetch_fundamentals, get_universe_tickers
from .screening import ScreeningResult, screen_company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Halal Stock Analyst")

_screen_cache: dict[str, ScreeningResult] = {}


class Analyst(Protocol):
    def stream_answer(self, result: ScreeningResult, question: str): ...


_analyst: Analyst | None = None


def get_analyst() -> Analyst:
    """Picks the backend based on ANALYST_BACKEND ('local', 'openai', or 'claude').

    'local' (default) loads an open-source model into the Colab GPU —
    free, but needs a GPU runtime and a ~1-2 min load on first use.
    'openai' calls the OpenAI API — needs OPENAI_API_KEY (a separate API
    key from platform.openai.com; a ChatGPT Plus/Pro subscription does
    NOT include API access), billed pay-per-token, but no GPU needed.
    'claude' calls the Anthropic API — needs ANTHROPIC_API_KEY, billed
    pay-per-token, no GPU needed.
    """
    global _analyst
    if _analyst is None:
        backend = os.environ.get("ANALYST_BACKEND", "local").lower()
        if backend == "openai":
            from .openai_analyst import OpenAIAnalyst

            _analyst = OpenAIAnalyst(model=os.environ.get("OPENAI_MODEL", "gpt-4o"))
        elif backend == "claude":
            from .claude_analyst import ClaudeAnalyst

            _analyst = ClaudeAnalyst(model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-8"))
        else:
            from .llm_analyst import LocalAnalyst

            _analyst = LocalAnalyst()
    return _analyst


def get_or_screen(ticker: str) -> ScreeningResult:
    ticker = ticker.upper()
    if ticker not in _screen_cache:
        fundamentals = fetch_fundamentals(ticker)
        if fundamentals is None:
            raise HTTPException(404, f"Could not fetch data for {ticker}")
        _screen_cache[ticker] = screen_company(fundamentals)
    return _screen_cache[ticker]


@app.get("/screen")
def screen_universe(limit: int = 25) -> list[dict]:
    """Screen the first `limit` tickers of the default universe and return results."""
    tickers = get_universe_tickers()[:limit]
    results = []
    for t in tickers:
        fundamentals = fetch_fundamentals(t)
        if fundamentals is None:
            continue
        result = screen_company(fundamentals)
        _screen_cache[t] = result
        results.append(result.__dict__)
    return results


@app.get("/screen/{ticker}")
def screen_one(ticker: str) -> dict:
    return get_or_screen(ticker).__dict__


class ChatRequest(BaseModel):
    ticker: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Streams plain-text progress + the model's answer as it's generated,
    so the client can show live status instead of a blind spinner. Lines
    starting with '[status]' are progress markers, not part of the answer.
    """

    def event_stream():
        ticker = req.ticker.upper()
        yield f"[status] Fetching fundamentals for {ticker}...\n"
        try:
            result = get_or_screen(ticker)
        except HTTPException as exc:
            yield f"[status] Error: {exc.detail}\n"
            return

        yield f"[status] Sharia screen: {'PASS' if result.is_compliant else 'FAIL'}\n"
        if not result.is_compliant:
            yield f"[status] Reasons: {'; '.join(result.reasons)}\n"

        backend = os.environ.get("ANALYST_BACKEND", "local").lower()
        if backend == "openai":
            yield "[status] Calling OpenAI API...\n"
        elif backend == "claude":
            yield "[status] Calling Claude API...\n"
        else:
            yield "[status] Loading local analyst model (first request only, ~1-2 min)...\n"
        analyst = get_analyst()

        yield "[status] Generating response...\n"
        for chunk in analyst.stream_answer(result, req.question):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <html><head><title>Halal Stock Analyst</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
      <h2>Halal Stock Analyst</h2>
      <p>Enter a ticker and a question. Only Sharia-screened companies are discussed.</p>
      <input id="ticker" placeholder="AAPL" style="width:100px"/>
      <input id="question" placeholder="Is this a good long-term hold?" style="width:350px"/>
      <button onclick="ask()">Ask</button>
      <pre id="out" style="white-space:pre-wrap;background:#f4f4f4;padding:12px;margin-top:16px;"></pre>
      <script>
        async function ask() {
          const ticker = document.getElementById('ticker').value;
          const question = document.getElementById('question').value;
          const out = document.getElementById('out');
          out.textContent = '';
          const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker, question})
          });
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            out.textContent += decoder.decode(value, {stream: true});
            out.scrollTop = out.scrollHeight;
          }
        }
      </script>
    </body></html>
    """
