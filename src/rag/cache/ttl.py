"""TTL classes — the expiry layer (architecture §5.5, D-05).

    filed_fact      30 d   numbers and facts straight from the filing
    analytical       7 d   explanations, comparisons, computed ratios, figure readings
    time_anchored    1 d   or until the referenced event date if that is sooner
    negative         1 d   "not found in the report" answers
    (not cached)           low confidence / failed verification — decided in admission.py

`answer_class` comes from the generator's structured output; rules override it: any `time_anchor`
slot forces `time_anchored`, and a "not in the report" answer is `negative`. The L1 exact tier
uses min(class TTL, 1 hour).
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Literal

from rag.core.config import TTLSeconds

TTLClass = Literal["filed_fact", "analytical", "time_anchored", "negative"]
TTL_CLASSES: tuple[str, ...] = ("filed_fact", "analytical", "time_anchored", "negative")

L1_TTL_CEILING_S = 3600

_NEGATIVE = re.compile(
    r"\b(not (?:found|reported|disclosed|stated|included|provided|available|contain\w*)|"
    r"does not (?:contain|include|report|disclose|state|provide|specify)|"
    r"do(?:es)? not appear|no (?:information|figure|mention|disclosure)|"
    r"cannot be (?:determined|answered)|is missing|are missing)\b",
    re.IGNORECASE,
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9

_LONG_DATE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def is_negative_answer(answer_markdown: str) -> bool:
    return bool(_NEGATIVE.search(answer_markdown or ""))


def find_dates(text: str) -> list[date]:
    """Calendar dates written out in `text` ("June 24, 2026", "2026-06-24"), de-duplicated,
    ascending. Malformed dates are ignored."""
    found: set[date] = set()
    for m in _LONG_DATE.finditer(text or ""):
        month = _MONTHS.get(m.group(1).lower().rstrip("."))
        if month is None:
            continue
        try:
            found.add(date(int(m.group(3)), month, int(m.group(2))))
        except ValueError:
            continue
    for m in _ISO_DATE.finditer(text or ""):
        try:
            found.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return sorted(found)


def ttl_class(
    generator_class: str,
    *,
    time_anchor: bool,
    answer_markdown: str,
) -> str:
    """Class for an admitted answer (§5.5). Rules override the generator's class."""
    if time_anchor:
        return "time_anchored"
    if is_negative_answer(answer_markdown):
        return "negative"
    if generator_class in TTL_CLASSES:
        return generator_class
    return "analytical"


def ttl_seconds_for(cls: str, ttls: TTLSeconds) -> int:
    return int(getattr(ttls, cls))


def expires_at(
    cls: str,
    *,
    now: int,
    ttls: TTLSeconds,
    answer_markdown: str = "",
) -> int:
    """Epoch-second expiry. `time_anchored` answers expire at the earliest referenced event
    date that is still in the future when that comes before the 1-day TTL; events already in
    the past leave the plain TTL (an answer saying "the meeting was held" stays right)."""
    exp = now + ttl_seconds_for(cls, ttls)
    if cls == "time_anchored":
        today = datetime.fromtimestamp(now, UTC).date()
        future = [d for d in find_dates(answer_markdown) if d > today]
        if future:
            event_ts = int(
                datetime(future[0].year, future[0].month, future[0].day, tzinfo=UTC).timestamp()
            )
            exp = min(exp, event_ts)
    return exp


def l1_expires_at(l2_expires_at: int, now: int, ceiling_s: int = L1_TTL_CEILING_S) -> int:
    return min(l2_expires_at, now + ceiling_s)
