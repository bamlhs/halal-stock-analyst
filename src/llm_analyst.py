"""Local, open-source LLM analyst.

Loads an open-weight instruction-tuned model directly in the Colab GPU
runtime (4-bit quantized via bitsandbytes so it fits a free-tier T4).

Default model: Qwen/Qwen2.5-7B-Instruct (Apache-2.0, no gated access
required, strong at structured financial reasoning). Swap MODEL_ID for
e.g. "meta-llama/Llama-3.1-8B-Instruct" if you have HF access to it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from .screening import ScreeningResult

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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


class LocalAnalyst:
    def __init__(self, model_id: str = MODEL_ID):
        logger.info("Loading %s in 4-bit...", model_id)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )

    def _context_block(self, result: ScreeningResult) -> str:
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

    def stream_answer(self, result: ScreeningResult, question: str) -> Iterator[str]:
        """Yield the answer token-by-token as it's generated, so a caller
        (e.g. a streaming HTTP response) can show live progress instead of
        a blind wait."""
        if not result.is_compliant:
            reasons = "; ".join(result.reasons) or "fails screening"
            yield (
                f"{result.ticker} ({result.name}) did not pass the Sharia "
                f"compliance screen: {reasons}. I can't provide an "
                f"investment discussion for a non-compliant company."
            )
            return

        context = self._context_block(result)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Company data:\n{context}\n\nQuestion: {question}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()
