"""Claude API-backed analyst — an alternative to the local open-source model
and the OpenAI backend.

Requires ANTHROPIC_API_KEY (from console.anthropic.com / platform.claude.com).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import anthropic

from .prompting import SYSTEM_PROMPT, context_block, non_compliant_message
from .screening import ScreeningResult

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeAnalyst:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def stream_answer(self, result: ScreeningResult, question: str) -> Iterator[str]:
        if not result.is_compliant:
            yield non_compliant_message(result)
            return

        context = context_block(result)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=[
                {
                    "role": "user",
                    "content": f"Company data:\n{context}\n\nQuestion: {question}",
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text
