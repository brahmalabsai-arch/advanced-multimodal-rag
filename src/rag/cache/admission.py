"""Admission policy — what may enter the cache (architecture §5.6, D-08).

All five must hold for an L2 write:
    1. `verify_answer` passed
    2. confidence ∈ {high, medium}
    3. intent ≠ OUT_OF_SCOPE (scoped refusals go to L1 only — they are deterministic rule
       output and cost no model call, so a short L1 entry is all they need)
    4. the query is self-contained (v1 is single-turn; a question that reads like a follow-up
       is rejected here — this rule becomes the multi-turn hook in v2)
    5. at least one citation

A degraded answer (model call failed) fails rule 1 because the verifier never ran.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_FOLLOW_UP = re.compile(
    r"^\s*(?:and|also|what about|how about|same (?:for|thing)|then|now)\b|"
    r"\b(?:that|those|it|them|this one|the same|previous|earlier|above)\b.*\?\s*$",
    re.IGNORECASE,
)
_ANCHORED = re.compile(
    r"\b(nvidia|nvda|fiscal|fy\s?\d|20\d\d|total|assets|liabilit|revenue|margin|cash|debt|"
    r"inventor|equity|ratio|meeting|share|dividend|repurchase|h20|export|compensation|pay)\w*",
    re.IGNORECASE,
)


def is_self_contained(question: str) -> bool:
    """Heuristic for rule 4: a question is self-contained unless it reads like a follow-up
    ("and for 2025?", "what about inventories?") *and* names no anchoring subject."""
    q = question.strip()
    if not q:
        return False
    return not (_FOLLOW_UP.search(q) and not _ANCHORED.search(q))


class AdmissionDecision(BaseModel):
    admit_l2: bool
    admit_l1: bool
    reasons: list[str] = Field(default_factory=list, description="failed rules")

    @property
    def admitted(self) -> bool:
        return self.admit_l2 or self.admit_l1


def admission_decision(
    *,
    question: str,
    verify_passed: bool,
    confidence: str,
    intent: str,
    citations: list[str],
    degraded: bool = False,
) -> AdmissionDecision:
    reasons: list[str] = []
    if intent == "OUT_OF_SCOPE":
        # Rule 3: refusals are L1-only. They still need rule 4 (a follow-up refusal is
        # context-dependent) but not citations or verification.
        if not is_self_contained(question):
            reasons.append("R4 not self-contained")
        return AdmissionDecision(
            admit_l2=False, admit_l1=not reasons, reasons=reasons + ["R3 out of scope: L1 only"]
        )
    if degraded:
        reasons.append("R1 degraded answer (model call failed)")
    if not verify_passed:
        reasons.append("R1 verification failed")
    if confidence not in {"high", "medium"}:
        reasons.append(f"R2 confidence={confidence}")
    if not is_self_contained(question):
        reasons.append("R4 not self-contained")
    if not citations:
        reasons.append("R5 no citation")
    ok = not reasons
    return AdmissionDecision(admit_l2=ok, admit_l1=ok, reasons=reasons)
