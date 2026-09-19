"""Approximate token counting with tiktoken (architecture §8.2).

Counts are approximate for Llama / GPT-OSS / Qwen tokenizers; they drive classifier features,
context budgets and client-side pacing. Billed tokens always come from API usage metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import tiktoken

ENCODING_NAME = "cl100k_base"

# Groq documents 2,048 tokens per image (console.groq.com/docs/vision). Billed usage for a
# ≤1600 px figure crop is lower (≈ 1,090 on qwen/qwen3.8-27b), but Groq's per-minute *input*
# gate (ITPM) pre-checks requests at the documented figure, so pacing must assume it too.
IMAGE_TOKEN_ESTIMATE = 2048

# Rough per-message framing overhead (role tags, separators).
MESSAGE_OVERHEAD_TOKENS = 4


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


def estimate_content_tokens(content: Any) -> int:
    """Token estimate for a message `content`: a string or a list of content blocks."""
    if isinstance(content, str):
        return count_tokens(content)
    total = 0
    for block in content or []:
        if isinstance(block, str):
            total += count_tokens(block)
        elif isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                total += count_tokens(str(block.get("text", "")))
            elif block_type in {"image_url", "image"}:
                total += IMAGE_TOKEN_ESTIMATE
    return total


def estimate_messages_tokens(messages: Iterable[Any]) -> int:
    """Token estimate for a list of LangChain messages (anything with a `.content`)."""
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        total += estimate_content_tokens(content) + MESSAGE_OVERHEAD_TOKENS
    return total
