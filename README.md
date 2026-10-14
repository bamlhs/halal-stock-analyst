# Halal Stock Analyst

A Sharia-compliance stock screener paired with a locally-run, open-source
LLM chat analyst. Designed to run for free in Google Colab (GPU runtime)
and be shared via an ngrok tunnel.

> **Disclaimer:** This tool provides an automated approximation of common
> AAOIFI / Dow Jones Islamic Market style screening criteria. It is **not**
> a religious ruling (fatwa) and should not be treated as financial advice.
> Always confirm against a recognized Islamic index (e.g. S&P Shariah,
> Dow Jones Islamic Market, MSCI Islamic) or a qualified Islamic finance
> scholar before making investment decisions.

## How the screening works (`src/screening.py`)

1. **Business activity screen** — excludes companies whose primary sector
   or industry is conventional banking/finance, insurance, alcohol,
   tobacco, gambling, pork products, adult entertainment, or conventional
   weapons.
2. **Financial ratio screens** (relative to market cap):
   - Debt ratio < 33%
   - Cash + short-term investments ratio < 33%
   - Accounts receivable ratio < 49%
   - Interest/impure income ratio < 5% of revenue — if present, that
     percentage of dividends should be donated to charity (purification).

Stock universe defaults to the S&P 500, sourced from the open-source
[`datasets/s-and-p-500-companies`](https://github.com/datasets/s-and-p-500-companies)
GitHub repo. Fundamentals come from Yahoo Finance via `yfinance` (free,
no API key required).

## How the analyst works

The system prompt (`src/prompting.py`) hard-restricts the model to only
discuss the specific ticker passed in, and only after it has already
passed the Sharia screen — it will refuse to discuss anything else.
Three interchangeable backends implement that prompt, selected via the
`ANALYST_BACKEND` env var:

- **`local`** (default, `src/llm_analyst.py`) — loads
  [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  (Apache-2.0, open weights) directly into the Colab GPU runtime in
  4-bit (via `bitsandbytes`), fits comfortably on a free-tier T4. Free,
  but needs a GPU runtime and a ~1-2 min load on first use.
- **`openai`** (`src/openai_analyst.py`) — calls the OpenAI API
  (`gpt-4o` by default, override with `OPENAI_MODEL`). Faster, no GPU
  needed, but requires a separate `OPENAI_API_KEY` from
  [platform.openai.com](https://platform.openai.com) — **a ChatGPT
  Plus/Pro subscription does not include API access**; API calls are
  billed pay-per-token independently of any chatgpt.com subscription.
- **`claude`** (`src/claude_analyst.py`) — calls the Anthropic API
  (`claude-opus-4-8` by default, override with `CLAUDE_MODEL`). Faster,
  no GPU needed, requires a separate `ANTHROPIC_API_KEY` from
  [platform.claude.com](https://platform.claude.com), billed
  pay-per-token against your Anthropic account credit.

## Running

### In Google Colab (recommended)

Open `notebooks/Halal_Stock_Analyst_Colab.ipynb` in Colab, set a
`NGROK_AUTHTOKEN` secret in the Colab Secrets panel (key icon in the
left sidebar — **never paste the token into a cell**), select a GPU
runtime, and run all cells. It prints a public ngrok URL you can open
or share.

To use a hosted API backend instead of the local model, also add the
matching secret (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) and set
`ANALYST_BACKEND` to `"openai"` or `"claude"` in the second code cell —
no GPU runtime is required for either.

### Locally

```bash
pip install -r requirements.txt
export NGROK_AUTHTOKEN=your_token_here   # optional, only needed to tunnel

# Local open-source model (default):
uvicorn src.app:app --reload

# OR, OpenAI backend:
export ANALYST_BACKEND=openai
export OPENAI_API_KEY=your_api_key_here
uvicorn src.app:app --reload

# OR, Claude backend:
export ANALYST_BACKEND=claude
export ANTHROPIC_API_KEY=your_api_key_here
uvicorn src.app:app --reload
```

## API

- `GET /screen?limit=25` — screens the first N tickers of the default universe.
- `GET /screen/{ticker}` — screens a single ticker.
- `POST /chat` — `{"ticker": "AAPL", "question": "..."}` → analyst response,
  restricted to Sharia-compliant companies.

## Security notes

- No credentials are stored in this repo. The ngrok auth token is read
  from an environment variable / Colab secret at runtime only.
- If you ever paste a live token into a notebook cell, chat, or commit
  by mistake, treat it as compromised and regenerate it immediately at
  your provider's dashboard.
