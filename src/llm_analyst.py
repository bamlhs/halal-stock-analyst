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

from .prompting import SYSTEM_PROMPT, context_block, non_compliant_message
from .screening import ScreeningResult

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


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

    def stream_answer(self, result: ScreeningResult, question: str) -> Iterator[str]:
        """Yield the answer token-by-token as it's generated, so a caller
        (e.g. a streaming HTTP response) can show live progress instead of
        a blind wait."""
        if not result.is_compliant:
            yield non_compliant_message(result)
            return

        context = context_block(result)
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
