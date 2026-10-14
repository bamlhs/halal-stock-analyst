"""FastAPI app: screens the universe once at startup, then answers
chat questions restricted to the halal-compliant subset.

Run locally:   uvicorn src.app:app --reload
Run in Colab:  see notebooks/Halal_Stock_Analyst_Colab.ipynb
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .data import fetch_fundamentals, get_universe_tickers
from .llm_analyst import LocalAnalyst
from .screening import ScreeningResult, screen_company

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Halal Stock Analyst")

_screen_cache: dict[str, ScreeningResult] = {}
_analyst: LocalAnalyst | None = None


def get_analyst() -> LocalAnalyst:
    global _analyst
    if _analyst is None:
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
def chat(req: ChatRequest) -> dict:
    result = get_or_screen(req.ticker)
    analyst = get_analyst()
    answer = analyst.answer(result, req.question)
    return {"ticker": result.ticker, "compliant": result.is_compliant, "answer": answer}


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
          document.getElementById('out').textContent = 'Thinking...';
          const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker, question})
          });
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        }
      </script>
    </body></html>
    """
