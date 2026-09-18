"""Answer verifier (architecture §4.12) — cheap, deterministic, no LLM.

1. Every number in `answer_markdown` matches a value in the cited context or a calculation
   result/input (after unit normalisation; a 1,000× scale for million↔billion is tolerated),
   or is a year/date/small count.
2. Every citation id in the answer and in `citations` exists in the assembled context.
3. (Phase 4) period in `figures_used` is consistent with the slots.

A failure triggers one regeneration with the issues appended; a second failure returns the
answer with `confidence=low` and a visible warning, and it is never admitted to the cache.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult
from rag.query.assemble import AssembledContext
from rag.query.generate import Answer

_CITATION = re.compile(r"\[([CK]\d+)\]")
_NUMBER = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\w])")
_YEAR_RANGE = (1990, 2100)
SMALL_INT_ALLOWANCE = 31  # day numbers, list markers, "five-year", "3 segments"
SCALE_FACTORS = (1.0, 1000.0, 0.001)


class VerifyResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    numbers_checked: int = 0
    unmatched_numbers: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citations_found: list[str] = Field(default_factory=list)


def _to_float(token: str) -> float | None:
    cleaned = token.replace("$", "").replace(",", "").replace("%", "").strip()
    if cleaned in {"", "-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_UNICODE_SPACES = str.maketrans(
    {chr(0x202F): " ", chr(0x00A0): " ", chr(0x2009): " ", chr(0x2007): " "}
)


def extract_numbers(text: str) -> list[str]:
    """Numeric tokens in free text, ignoring citation ids; unicode spaces normalised."""
    text = _CITATION.sub(" ", text.translate(_UNICODE_SPACES))
    return [m.group(0) for m in _NUMBER.finditer(text)]


def allowed_values(
    context: AssembledContext, calculations: list[CalculationResult], question: str
) -> set[float]:
    values: set[float] = set()
    for token in extract_numbers(context.text) + extract_numbers(question):
        v = _to_float(token)
        if v is not None:
            values.add(v)
    for calc in calculations:
        for v in (calc.result, calc.rounded):
            if v is not None:
                values.add(float(v))
        for i in calc.inputs:
            if i.value is not None:
                values.add(float(i.value))
    return values


def _decimals(token: str) -> int:
    cleaned = token.replace("$", "").replace(",", "").replace("%", "")
    return len(cleaned.split(".")[1]) if "." in cleaned else 0


def number_is_supported(token: str, allowed: set[float]) -> bool:
    value = _to_float(token)
    if value is None:
        return True
    if value.is_integer() and _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1] and "$" not in token:
        return True  # years
    if (
        value.is_integer()
        and abs(value) <= SMALL_INT_ALLOWANCE
        and "$" not in token
        and "," not in token
    ):
        return True
    decimals = _decimals(token)
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    for a in allowed:
        for scale in SCALE_FACTORS:
            candidate = a * scale
            if abs(abs(candidate) - abs(value)) <= tolerance:
                return True
            # the answer may round a value that the context states with more decimals
            if abs(round(candidate, decimals) - value) <= tolerance:
                return True
    return False


def verify_answer(
    answer: Answer,
    context: AssembledContext,
    calculations: list[CalculationResult],
    question: str,
) -> VerifyResult:
    issues: list[str] = []
    allowed = allowed_values(context, calculations, question)
    tokens = extract_numbers(answer.answer_markdown)
    unmatched = [t for t in tokens if not number_is_supported(t, allowed)]
    if unmatched:
        issues.append(
            "Numbers not found in the context or calculations: "
            + ", ".join(dict.fromkeys(unmatched))
        )

    valid_ids = context.block_ids()
    cited_inline = _CITATION.findall(answer.answer_markdown)
    cited_all = list(dict.fromkeys(cited_inline + [c.strip("[]") for c in answer.citations]))
    invalid = [c for c in cited_all if c not in valid_ids]
    if invalid:
        issues.append("Citations that do not exist in the context: " + ", ".join(invalid))
    if not cited_all and (answer.figures_used or tokens):
        issues.append("The answer states figures but cites no context block.")
    for f in answer.figures_used:
        if f.citation.strip("[]") not in valid_ids:
            issues.append(f"figures_used cites unknown block {f.citation}")

    return VerifyResult(
        passed=not issues,
        issues=issues,
        numbers_checked=len(tokens),
        unmatched_numbers=list(dict.fromkeys(unmatched)),
        invalid_citations=invalid,
        citations_found=cited_all,
    )
