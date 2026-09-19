"""Numeric fidelity guard (architecture §4.8, NFR-6, P3).

Every number token in a compressed text must appear verbatim in its source chunk; any
violation discards the compressed version (the caller reverts to the original chunk) and is
logged. Applied to every compressor output, not only the LLM one — a bug in row selection or
sentence slicing must be caught the same way. Sentence-level verbatim checking is reported
alongside (`non_verbatim`) but only numbers are a hard failure, so re-ordered sentences pass.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_NUMBER = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\w])")
_UNICODE_SPACES = str.maketrans(
    dict.fromkeys(map(chr, (0x202F, 0xA0, 0x2009, 0x2007, 0x2008, 0x200A)), " ")
)
_WS = re.compile(r"\s+")


class FidelityResult(BaseModel):
    ok: bool
    numbers_checked: int = 0
    missing_numbers: list[str] = Field(default_factory=list)
    non_verbatim: list[str] = Field(
        default_factory=list, description="sentences not found verbatim in the source"
    )


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text.translate(_UNICODE_SPACES))]


def _canon(text: str) -> str:
    return _WS.sub(" ", text.translate(_UNICODE_SPACES)).strip().lower()


def check_fidelity(
    compressed: str, source: str, sentences: list[str] | None = None
) -> FidelityResult:
    """Numbers must survive verbatim; `sentences` (optional) are checked as substrings of the
    source after whitespace normalisation and reported, not enforced."""
    source_numbers = set(numbers_in(source))
    checked = numbers_in(compressed)
    missing = sorted({n for n in checked if n not in source_numbers})
    non_verbatim: list[str] = []
    if sentences:
        src = _canon(source)
        non_verbatim = [s for s in sentences if _canon(s) and _canon(s) not in src]
    return FidelityResult(
        ok=not missing,
        numbers_checked=len(checked),
        missing_numbers=missing,
        non_verbatim=non_verbatim,
    )
