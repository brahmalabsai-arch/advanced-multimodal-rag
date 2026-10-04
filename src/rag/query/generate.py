"""Answer generation — the prompt contract (architecture §4.11).

Large model only (P2); the vision role when a figure image is attached for a VISUAL question.
Output is a validated `Answer` JSON object. Prompt text is versioned (`PROMPT_VERSION` feeds the
cache's `prompt_version` key in Phase 6).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.core.tokens import IMAGE_TOKEN_ESTIMATE, count_tokens
from rag.llm import LLMClient, LLMError, _schema_instruction
from rag.query.assemble import AssembledContext

log = get_logger(__name__)

PROMPT_VERSION = "answer-v2"  # v2: answer every part; computed values are answers

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about NVIDIA's fiscal 2026 annual report (Annual Review, Proxy Statement and Form 10-K). You answer ONLY from the context blocks and calculation blocks provided.

Rules:
1. Use only facts present in the context blocks and calculation blocks. If neither contains what is needed, say exactly what is missing. Never use outside knowledge and never estimate a number.
2. Every figure you state must carry its unit (USD millions unless the block says otherwise) and its fiscal period with the period-end date, e.g. "$206,803 million as of January 25, 2026 (fiscal 2026)".
3. Calculation blocks [K1], [K2] … are pre-computed and authoritative: copy their RESULT verbatim, do not recompute, and cite them like any other block. A growth rate or ratio is rarely printed in the filing; when a calculation block gives it, report it and say it was calculated from the filed figures. Never say the filing does not provide a value that a calculation block gives.
3a. Answer every part of the question. When it asks for several metrics or several things, address each one in turn, in the order asked; if one part cannot be answered, say so for that part and still answer the others.
4. Cite the block ids inline in square brackets, e.g. [C1], [K1]. Cite every block whose numbers you use.
5. NVIDIA's fiscal year ends in late January: fiscal 2026 ended January 25, 2026 and fiscal 2025 ended January 26, 2025. If the question says a bare year, state which fiscal year you interpreted it as.
6. Write the answer in Markdown: a direct answer first, then short supporting detail. Keep it under 180 words unless the question needs a table.
7. answer_class: "filed_fact" for numbers and facts that come straight from the filing, "analytical" for explanations, comparisons or computed ratios, "time_anchored" for anything about scheduled events (e.g. the annual meeting date).
8. confidence: "high" when the context directly answers the question, "medium" when you had to combine or infer, "low" when the context is thin or contradictory.
"""


class FigureUsed(BaseModel):
    value: float | str = Field(description="the number as stated in the context")
    unit: str = Field(description="USD_millions, USD_per_share, ratio, percent, shares_millions, …")
    period: str = Field(description="e.g. FY2026 (Jan 25, 2026)")
    citation: str = Field(description="block id like C1 or K1")


class Answer(BaseModel):
    answer_markdown: str
    figures_used: list[FigureUsed] = Field(default_factory=list)
    citations: list[str] = Field(
        default_factory=list, description="block ids used, e.g. ['C1','K1']"
    )
    confidence: Literal["high", "medium", "low"] = "medium"
    answer_class: Literal["filed_fact", "analytical", "time_anchored"] = "filed_fact"
    fiscal_year_interpretation: str | None = Field(
        default=None, description="only when the question used a bare year"
    )


def build_user_prompt(
    question: str,
    context: AssembledContext,
    retry_note: str | None = None,
    notes: list[str] | None = None,
) -> str:
    parts = [f"QUESTION: {question}", "", "CONTEXT BLOCKS:", context.text]
    if notes:
        parts += ["", "NOTES FROM QUERY ANALYSIS:"] + [f"- {n}" for n in notes]
    if context.images:
        parts += [
            "",
            f"The attached image(s) are the figure(s) cited as {', '.join(context.image_chunk_ids)}; "
            "read values from the companion table block when one is present and treat values "
            "read off the graphic as approximate.",
        ]
    if retry_note:
        parts += ["", "YOUR PREVIOUS ANSWER FAILED VERIFICATION:", retry_note, "Fix these issues."]
    parts += ["", "Respond with the JSON object described in the system message."]
    return "\n".join(parts)


# gpt-oss sometimes cites in its own house style, 【K1】 or 【C3†L1-L4】, instead of [K1]. The
# verifier and the page both read square brackets, so the markers are normalised on the way out.
_HOUSE_CITATION = re.compile(r"【\s*([CK]\d+)[^】]*】")


def normalise_citations(answer: Answer) -> Answer:
    text = _HOUSE_CITATION.sub(r"[\1]", answer.answer_markdown)
    return (
        answer
        if text == answer.answer_markdown
        else answer.model_copy(update={"answer_markdown": text})
    )


# Tokenizer slack: tiktoken undercounts Qwen's tokenizer by up to ~10% on this prose.
INPUT_ESTIMATE_MARGIN = 1.1


def fit_images_to_budget(images: list[str], prompt: str, input_cap: int | None) -> list[str]:
    """Drop images from the end until the estimated input (text + IMAGE_TOKEN_ESTIMATE per
    image) fits `input_cap`; always keeps the first image (the figure the question is about)."""
    if not input_cap:
        return list(images)
    text_tokens = (
        count_tokens(SYSTEM_PROMPT)
        + count_tokens(_schema_instruction(Answer))  # the client appends this to the system text
        + count_tokens(prompt)
        + 16
    ) * INPUT_ESTIMATE_MARGIN
    kept = list(images)
    while len(kept) > 1 and text_tokens + len(kept) * IMAGE_TOKEN_ESTIMATE > input_cap:
        kept.pop()
    return kept


def generate_answer(
    client: LLMClient,
    question: str,
    context: AssembledContext,
    *,
    intent: str,
    request_id: str,
    retry_note: str | None = None,
    notes: list[str] | None = None,
    max_tokens: int = 1200,
) -> tuple[Answer, str]:
    """Returns (answer, role used). If the vision model is unavailable, falls back to the text
    model with the figure description + companion table (architecture §13). `notes` carries
    slot-level hints such as how a bare calendar year was interpreted (§4.2)."""
    prompt = build_user_prompt(question, context, retry_note, notes)
    if context.images and intent == "VISUAL":
        # The vision role is paced on *requested* output tokens (1,000 OTPM on Groq, D-52) and
        # on a small TPM bucket; a request over either bucket is rejected before the call, so
        # cap the budget and shed trailing images (the first one is the figure the question is
        # about) until the request fits.
        cap = client.max_output_tokens("vision")
        vision_budget = min(max_tokens, cap) if cap else max_tokens
        images = fit_images_to_budget(
            context.images, prompt, client.max_input_tokens("vision", vision_budget)
        )
        if len(images) < len(context.images):
            log.info(
                "vision request: %d of %d figure images attached (input budget)",
                len(images),
                len(context.images),
            )
        try:
            answer = client.vision_json(
                prompt,
                images,
                Answer,
                role="vision",
                system=SYSTEM_PROMPT,
                request_id=request_id,
                max_tokens=vision_budget,
            )
            return normalise_citations(answer), "vision"
        except LLMError as exc:
            log.warning("vision role unavailable (%s); answering from descriptions", str(exc)[:120])
            prompt = build_user_prompt(
                question, context.model_copy(update={"images": []}), retry_note, notes
            )
    cap = client.max_output_tokens("large")
    answer = client.json(
        prompt,
        Answer,
        role="large",
        system=SYSTEM_PROMPT,
        request_id=request_id,
        max_tokens=min(max_tokens, cap) if cap else max_tokens,
    )
    return normalise_citations(answer), "large"
