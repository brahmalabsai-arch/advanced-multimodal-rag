"""Completeness check — did the answer address every part of the question?

The verifier (`verify.py`) proves every number the answer states is traceable. It cannot see a
number the answer *should* have stated and did not: "growth of profit and revenue" answered
with revenue only passes it cleanly. This check runs after the verifier, in two passes.

1. Rules (free, deterministic). The parts the slots name are checked against the values the
   pipeline already holds. A metric with a calculation (the YoY pair, a named ratio) must have
   that calculation's result in the answer; a metric looked up for a period must have its row-fact
   value. Number matching reuses the verifier's tolerance (rounding, million↔billion, ratio↔%).
2. Self-check (one small-model call): the model lists the distinct asks and flags any the
   answer neither answered nor explained as unavailable. It catches parts no slot captures
   ("why did margin fall and what are the risks?"), so it runs only when the question reads as
   several asks, the rules found no gap, and the rules could not have seen every ask: they
   checked nothing, or the question asks for something other than values. "Growth of revenue
   and profit" is settled by the rules alone, at no cost.

A gap feeds the regeneration note, which names each missing part and, where a calculation block
holds its value, the block and the value. An answer still incomplete after the last attempt is
returned with the gap stated and never admitted to the cache.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult
from rag.core.logging import get_logger
from rag.llm import LLMClient, LLMError
from rag.query.generate import Answer
from rag.query.verify import extract_numbers, number_is_supported

if TYPE_CHECKING:
    from rag.calc.calculator import RowFactSource
    from rag.query.slots import QuerySlots

log = get_logger(__name__)

# Intents whose metrics are asked for as values. An EXPLANATORY "why did revenue grow" answer
# explains; it is not required to restate the figure.
VALUE_INTENTS = {"POINT_LOOKUP", "COMPUTATION", "COMPARISON_TREND"}
CHECK_INTENTS = VALUE_INTENTS | {"EXPLANATORY", "CROSS_SECTION", "VISUAL"}
MAX_SELF_CHECK_PARTS = 4

# Phrases that join two asks. Matched on the question with its slot phrases removed, so the
# "and" inside "research and development" or "profit and loss" never counts.
_JOINERS = re.compile(
    r"\b(and|as well as|also|along with|plus|separately|respectively|both|in addition)\b"
)


class Part(BaseModel):
    label: str = Field(description="what the question asks for, in the reader's words")
    expected: list[float] = Field(default_factory=list, description="any of these counts")
    hint: str = Field(default="", description="where the value is, for the regeneration note")


class CoverageResult(BaseModel):
    checked: bool = False
    method: Literal["skipped", "rules", "rules+self_check"] = "skipped"
    parts: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    self_check_error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.missing

    def retry_note(self) -> str | None:
        if self.passed:
            return None
        lines = ["The answer does not address every part of the question. Missing:"]
        lines += [f"- {m}" for m in self.missing]
        lines += [f"  {h}" for h in self.hints if h]
        lines.append(
            "Answer every part. When the filing does not state a value directly but a "
            "calculation block computes it, give the block's RESULT and say it was calculated "
            "from the filed figures. Never say the filing does not provide a value that a "
            "calculation block gives."
        )
        return "\n".join(lines)


# ------------------------------------------------------------------------------- rules


def _label(metric: str) -> str:
    return metric.replace("_", " ")


def _row_value(source: RowFactSource, metric: str, fiscal_year: int) -> float | None:
    statement = source.statement_of(metric)
    chunk = source.get(statement, metric) if statement else None
    if chunk is None:
        return None
    value = getattr(chunk.metadata, f"value_fy{fiscal_year}", None)
    return float(value) if isinstance(value, int | float) else None


def expected_parts(
    slots: QuerySlots,
    calculations: list[CalculationResult],
    intent: str,
    source: RowFactSource | None,
    *,
    default_fiscal_year: int,
) -> list[Part]:
    """The parts of the question the slots name, each with the values that would answer it."""
    parts: list[Part] = []
    by_metric: dict[str, Part] = {}
    for k, calc in enumerate(calculations, start=1):
        if calc.status != "ok" or calc.rounded is None:
            continue
        block = f"K{k}"
        if calc.metric:
            part = by_metric.get(calc.metric)
            if part is None:
                part = by_metric[calc.metric] = Part(
                    label=f"{_label(calc.metric)}: change FY{calc.fiscal_year - 1} to "
                    f"FY{calc.fiscal_year}"
                )
                parts.append(part)
            part.expected.append(float(calc.rounded))
            if calc.formula == "yoy_change_pct":
                part.hint = (
                    f"{_label(calc.metric)}: calculation block [{block}] gives "
                    f"{calc.formatted()} year over year."
                )
        else:
            parts.append(
                Part(
                    label=f"{calc.description} (FY{calc.fiscal_year})",
                    expected=[float(calc.rounded)],
                    hint=f"{calc.formula}: calculation block [{block}] gives {calc.formatted()}.",
                )
            )
    if intent not in VALUE_INTENTS or source is None:
        return parts
    fy = slots.current_fiscal_year or default_fiscal_year
    for metric in slots.metrics:
        if metric in by_metric:
            continue
        value = _row_value(source, metric, fy)
        if value is None:
            continue  # nothing to check against; the model must say it is missing
        parts.append(
            Part(
                label=f"{_label(metric)} (FY{fy})",
                expected=[value],
                hint=f"{_label(metric)}: the row fact for FY{fy} is {value:,.0f}.",
            )
        )
    return parts


def _proves_nothing(token: str) -> bool:
    """The verifier waves small integers and years through for any context; as evidence that a
    part was answered they prove nothing, unless written as a percentage or an amount."""
    if "%" in token or "$" in token:
        return False
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return True
    return value.is_integer() and (abs(value) <= 31 or 1990 <= value <= 2100)


def part_is_answered(part: Part, answer_markdown: str) -> bool:
    allowed = set(part.expected)
    return any(
        number_is_supported(token, allowed)
        for token in extract_numbers(answer_markdown)
        if not _proves_nothing(token)
    )


# --------------------------------------------------------------------------- self-check


def is_multi_part(question: str, slots: QuerySlots, sub_questions: list[str]) -> bool:
    """Several asks: two named metrics or formulas, two decomposed sub-questions, two question
    marks, or a joining word outside the slot phrases."""
    if len(slots.metrics) + len(slots.formulas) >= 2 or len(sub_questions) >= 2:
        return True
    if question.count("?") >= 2:
        return True
    residual = slots.normalized_text or question.lower()
    for m in sorted(slots.matches, key=lambda m: m.start, reverse=True):
        residual = residual[: m.start] + " " + residual[m.end :]
    return bool(_JOINERS.search(residual))


SELF_CHECK_SYSTEM = """You check whether an answer addresses every part of a question about a company's annual report.

Split the question into the distinct things it asks for (at most 4; one if it asks for one thing). For each, decide whether the answer addresses it. A part counts as addressed when the answer gives it, or states clearly that it cannot be given and why. A part does NOT count when the answer is silent on it or gives only a related figure. Judge coverage only, not correctness.
"""


class SelfCheckPart(BaseModel):
    ask: str = Field(description="one thing the question asks for, in a few words")
    addressed: bool


class SelfCheck(BaseModel):
    parts: list[SelfCheckPart] = Field(default_factory=list)


def self_check(
    client: LLMClient, question: str, answer_markdown: str, *, request_id: str
) -> SelfCheck:
    prompt = (
        f"QUESTION: {question}\n\nANSWER:\n{answer_markdown}\n\n"
        "Respond with the JSON object described in the system message."
    )
    return client.json(
        prompt,
        SelfCheck,
        role="small",
        system=SELF_CHECK_SYSTEM,
        request_id=request_id,
        max_tokens=600,  # hidden reasoning counts against it (see condense.rewrite)
    )


# ---------------------------------------------------------------------------- entry point


def check_coverage(
    question: str,
    answer: Answer,
    *,
    slots: QuerySlots,
    intent: str,
    calculations: list[CalculationResult],
    source: RowFactSource | None = None,
    sub_questions: list[str] | None = None,
    client: LLMClient | None = None,
    request_id: str = "",
    default_fiscal_year: int = 2026,
    allow_self_check: bool = True,
) -> CoverageResult:
    """Run the rules, then the self-check when the question has several asks and the rules
    found no gap. A self-check failure (quota, invalid JSON) never fails the answer."""
    if intent not in CHECK_INTENTS:
        return CoverageResult(method="skipped")
    parts = expected_parts(
        slots, calculations, intent, source, default_fiscal_year=default_fiscal_year
    )
    result = CoverageResult(checked=True, method="rules", parts=[p.label for p in parts])
    for p in parts:
        if not part_is_answered(p, answer.answer_markdown):
            result.missing.append(p.label)
            if p.hint:
                result.hints.append(p.hint)
    if result.missing or not allow_self_check or client is None:
        return result
    if not is_multi_part(question, slots, sub_questions or []):
        return result
    if parts and slots.ask == "value" and intent in VALUE_INTENTS:
        return result  # every ask is a value, and the rules checked each one
    result.method = "rules+self_check"
    try:
        check = self_check(client, question, answer.answer_markdown, request_id=request_id)
    except LLMError as exc:
        log.warning("completeness self-check unavailable for %s: %s", request_id, str(exc)[:160])
        result.self_check_error = str(exc)[:200]
        return result
    asks = check.parts[:MAX_SELF_CHECK_PARTS]
    result.parts = list(dict.fromkeys(result.parts + [p.ask for p in asks]))
    result.missing = [p.ask for p in asks if not p.addressed]
    return result


__all__ = [
    "CoverageResult",
    "Part",
    "check_coverage",
    "expected_parts",
    "is_multi_part",
    "part_is_answered",
]
