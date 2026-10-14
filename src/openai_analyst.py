"""OpenAI API-backed analyst — an alternative to the local open-source model.

Requires OPENAI_API_KEY (an API key from platform.openai.com, billed
pay-per-token). This is separate from a ChatGPT Plus/Pro subscription,
which does not include API access.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from .prompting import SYSTEM_PROMPT, context_block, non_compliant_message
from .screening import ScreeningResult

DEFAULT_MODEL = "gpt-4o"


class OpenAIAnalyst:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def stream_answer(self, result: ScreeningResult, question: str) -> Iterator[str]:
        if not result.is_compliant:
            yield non_compliant_message(result)
            return

        context = context_block(result)
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Company data:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
