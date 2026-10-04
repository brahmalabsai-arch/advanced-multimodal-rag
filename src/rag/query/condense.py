"""Follow-up condensation: in-session memory for a stateless server (F3, D-71).

The page keeps the conversation in the visitor's browser tab and sends the last turns with each
question, in compact form (the question, the standalone question it was read as, and a one-line
summary of the answer). Nothing about a conversation is stored on the server. This module turns
a follow-up — "and gross profit?", "why did it fall?", "what about FY2025?" — into a standalone
question, and every node after it (slots, cache, retrieval, calculator, verifier, completeness)
works on that standalone question unchanged. That is the hook the v1 design reserved (§15): the
cache keys on the standalone question and its slots, so a follow-up can never hit a cache entry
by its bare wording, and a first-turn question stays LLM-free on the cache path (P1).

Two steps, cheapest first:

1. **Rules** decide whether the question needs the conversation at all: it reads like a
   follow-up (a leading "and / what about / same for", or a reference such as "it", "that",
   "the year before"), or it names no subject of its own (no metric, formula or topic) while a
   conversation exists. Anything else is answered as typed, at no cost.
2. **One `small`-role call** rewrites a follow-up into a standalone question, given the recent
   turns. The model is told to resolve references and keep the ask, never to answer or to add
   asks. Its output is checked (non-empty, bounded length); on any failure the question is
   answered as typed and the trace says so.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.core.tokens import count_tokens
from rag.llm import LLMClient, LLMError

if TYPE_CHECKING:
    from rag.query.slots import QuerySlots

log = get_logger(__name__)

MAX_SUMMARY_CHARS = 280

# A follow-up opener, or a reference that only means something given earlier turns.
_LEADING = re.compile(
    r"^\s*(?:and|also|but|so|then|now|what about|how about|same (?:for|thing|question)|"
    r"compare(?:d)? (?:that|it|this)|vs\.?|versus)\b",
    re.IGNORECASE,
)
_REFERENCE = re.compile(
    r"\b(?:it|its|it's|that|those|these|this|they|them|their|the same|same period|"
    r"the former|the latter|previous(?:ly)?|earlier|above|the other one|the year before|"
    r"prior year|last year|that year|the one before)\b",
    re.IGNORECASE,
)


class Turn(BaseModel):
    """One earlier exchange, as the page sends it."""

    question: str = Field(max_length=2000)
    standalone: str | None = Field(default=None, max_length=2000)
    summary: str | None = Field(default=None, max_length=2000)

    def line(self) -> str:
        asked = (self.standalone or self.question).strip()
        out = f"Q: {asked}"
        if self.summary:
            out += f"\n   A: {self.summary.strip()[:MAX_SUMMARY_CHARS]}"
        return out


class CondenseResult(BaseModel):
    question: str = Field(description="the question every later node answers")
    asked: str = Field(description="the question as the visitor typed it")
    follow_up: bool = False
    rewritten: bool = False
    reason: str = ""
    turns_used: int = 0
    error: str | None = None


def fit_history(history: list[Turn], *, max_turns: int, max_tokens: int) -> list[Turn]:
    """The most recent turns that fit both caps, oldest dropped first, in chronological order."""
    kept: list[Turn] = []
    used = 0
    for turn in reversed(history[-max_turns:] if max_turns else []):
        cost = count_tokens(turn.line())
        if used + cost > max_tokens:
            break
        kept.append(turn)
        used += cost
    return list(reversed(kept))


def needs_context(question: str, slots: QuerySlots, history: list[Turn]) -> tuple[bool, str]:
    """Rules only. True when the question cannot be answered without the conversation."""
    if not history:
        return False, "no conversation"
    q = question.strip()
    has_subject = bool(slots.metrics or slots.formulas or slots.topics)
    if _LEADING.search(q):
        return True, "follow-up opener"
    if not has_subject:
        if _REFERENCE.search(q):
            return True, "refers to an earlier turn"
        if len(q.split()) <= 6:
            return True, "no subject of its own"
        if slots.fiscal_periods or slots.direction or slots.aggregation:
            return True, "asks about a period or change but names no subject"
        return False, "self-contained"
    # names a subject; a reference to "that year" / "the prior year" may still lean on history
    if _REFERENCE.search(q) and not slots.fiscal_periods:
        return True, "refers to an earlier turn"
    return False, "self-contained"


SYSTEM_PROMPT = """You rewrite a follow-up question about NVIDIA's fiscal 2026 annual report into one standalone question.

Use the conversation only to resolve what the follow-up refers to: the company, the metric or topic, the fiscal period, the comparison. Keep exactly what the follow-up asks — its ask type (value, change, reason, comparison) and its scope. When the follow-up only swaps the subject or the period ("and gross profit?", "and the year before?"), carry over what was being asked about the earlier one: after "what was the revenue growth rate?", "and gross profit?" means "what was the gross profit growth rate?". When it asks to compare ("compare that with operating income", "is that higher than last year?"), keep both sides of the comparison: after "what was net income?", "compare that with operating income" means "how does net income compare with operating income?". Do not answer it, do not add new asks, do not merge in earlier questions the follow-up does not refer to. Name metrics and fiscal years explicitly (e.g. "fiscal 2026"). If the question is already standalone, return it unchanged.

Reply with the standalone question only: one line, no quotes, no explanation.
"""


def rewrite(client: LLMClient, question: str, history: list[Turn], *, request_id: str) -> str:
    convo = "\n".join(t.line() for t in history)
    prompt = (
        f"CONVERSATION SO FAR (oldest first):\n{convo}\n\n"
        f"FOLLOW-UP: {question}\n\nSTANDALONE QUESTION:"
    )
    # Plain text, not JSON mode: the answer is one sentence, and in the evaluation Groq's JSON
    # validation rejected rewrites outright (HTTP 400) after gpt-oss's hidden reasoning ate into
    # the budget. Reasoning still counts against max_tokens, hence 600 for one line.
    out = client.text(
        prompt,
        role="small",
        system=SYSTEM_PROMPT,
        request_id=request_id,
        max_tokens=600,
    )
    return _one_line(out)


def _one_line(text: str) -> str:
    """The first non-empty line, without a label or wrapping quotes the model may add."""
    line = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    for label in ("STANDALONE QUESTION:", "Standalone question:", "Standalone:"):
        if line.startswith(label):
            line = line[len(label) :].strip()
    return line.strip().strip('"“”').strip()


def condense(
    question: str,
    history: list[Turn],
    slots: QuerySlots,
    *,
    client: LLMClient | None,
    request_id: str = "",
    max_turns: int = 10,
    max_tokens: int = 1500,
    max_question_chars: int = 1000,
) -> CondenseResult:
    asked = question.strip()
    turns = fit_history(history, max_turns=max_turns, max_tokens=max_tokens)
    follow_up, reason = needs_context(asked, slots, turns)
    result = CondenseResult(question=asked, asked=asked, follow_up=follow_up, reason=reason)
    if not follow_up:
        return result
    if client is None:
        result.error = "no model client for the rewrite"
        return result
    try:
        standalone = rewrite(client, asked, turns, request_id=request_id).strip()
    except LLMError as exc:
        log.warning("follow-up rewrite unavailable for %s: %s", request_id, str(exc)[:160])
        result.error = str(exc)[:200]
        return result
    if not standalone or len(standalone) > max_question_chars:
        result.error = "rewrite was empty or too long"
        return result
    result.question = standalone
    result.rewritten = standalone != asked
    result.turns_used = len(turns)
    return result


__all__ = ["CondenseResult", "Turn", "condense", "fit_history", "needs_context", "rewrite"]
