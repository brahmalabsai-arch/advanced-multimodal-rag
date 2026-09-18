"""Query analysis — Phase 3: rule-based intent only (architecture §4.4 arrives in Phase 4 with
slots, scope gate, expansion and the small-model fallback).

The seven intent classes are defined once in problem statement §4.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "POINT_LOOKUP",
    "COMPUTATION",
    "COMPARISON_TREND",
    "EXPLANATORY",
    "VISUAL",
    "CROSS_SECTION",
    "OUT_OF_SCOPE",
]

_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "OUT_OF_SCOPE",
        re.compile(
            r"\b(should i (buy|sell|hold)|buy or sell|price target|share price (today|now|right now)|"
            r"stock price (today|now|right now)|current (share|stock) price|investment advice|"
            r"is (nvda|nvidia) a (good|bad) (buy|investment))\b",
            re.I,
        ),
    ),
    (
        "VISUAL",
        re.compile(
            r"\b(chart|graph|diagram|figure|plot|five[- ]layer|cake|cumulative total return|"
            r"total shareholder return|nasdaq|s&p ?500|stock performance graph|pay mix|layers?)\b",
            re.I,
        ),
    ),
    (
        "COMPUTATION",
        re.compile(
            r"\b(current ratio|quick ratio|acid[- ]test|working capital|debt[- ]to[- ]equity|"
            r"debt/equity|liabilities[- ]to[- ]equity|equity ratio|leverage ratio|"
            r"cash and investments|cash and marketable securities|calculate|compute)\b",
            re.I,
        ),
    ),
    (
        "EXPLANATORY",
        re.compile(
            r"\b(why|explain|what drove|drivers?|reasons?|because|how did .* (affect|impact)|"
            r"what caused|describe|discuss)\b",
            re.I,
        ),
    ),
    (
        "COMPARISON_TREND",
        re.compile(
            r"\b(year[- ]over[- ]year|yoy|grow|grew|growth|increase[ds]?|decrease[ds]?|change[ds]?|"
            r"compared (to|with)|versus|vs\.?|trend|from .* to)\b",
            re.I,
        ),
    ),
    (
        "CROSS_SECTION",
        re.compile(
            r"\b(pay[- ]versus[- ]performance|pay vs\.? performance|executive (pay|compensation).*"
            r"(revenue|growth|performance)|compensation.*relate)\b",
            re.I,
        ),
    ),
]


class Analysis(BaseModel):
    intent: Intent
    rule: str = Field(description="which rule fired, or 'default'")
    retrieval_queries: list[str]
    needs_image: bool = False


def guess_intent(question: str) -> tuple[Intent, str]:
    for intent, pattern in _RULES:
        m = pattern.search(question)
        if m:
            return intent, m.group(0)
    return "POINT_LOOKUP", "default"


def analyze(question: str) -> Analysis:
    intent, rule = guess_intent(question)
    return Analysis(
        intent=intent,
        rule=rule,
        retrieval_queries=[question.strip()],
        needs_image=intent == "VISUAL",
    )
