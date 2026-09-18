"""Scope gate — rule-first OUT_OF_SCOPE detection with a `small`-role fallback (architecture §4.3).

Out of scope: investment advice (buy / sell / hold, price targets, "is it a good investment"),
live market data ("share price right now"), forecasts of the share price, and questions about
other companies' financials. Everything the filing could answer stays in scope, so a question
that names a metric, a statement or a fiscal period is never refused by rules alone.

Scores: hard patterns → 1.0, borderline patterns → 0.5, in-scope anchors subtract 0.4. A score
≥ 0.8 refuses; 0.4–0.8 asks the `small` model once (one JSON call, ~300 tokens); below 0.4 is in
scope. Refusals return a scoped answer and skip retrieval and generation entirely.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.llm import LLMClient, LLMError
from rag.query.slots import QuerySlots

log = get_logger(__name__)

REFUSE_AT = 0.8
ASK_LLM_AT = 0.4

_STOCK = r"(?:nvda|nvidia|the stock|the shares|shares|stock|its shares|it)"

HARD_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "advice_verb",
        re.compile(
            r"\b(should (i|we|one)|is it (a )?(good|bad|smart|safe) (time|idea|moment) to|"
            r"would you|do you (recommend|suggest|advise)|is now (a|the) (good|right) time to|"
            r"worth (buying|selling|holding|investing))\b.*\b(buy|sell|hold|short|invest|"
            r"purchase|dump|add to|trim)\b",
            re.I,
        ),
    ),
    (
        "buy_sell_hold",
        re.compile(
            r"\b(buy|sell|hold|short)\b (rating|recommendation|call|signal|or (sell|hold|buy))\b|"
            r"\b(buy or sell|sell or buy|buy, sell,? or hold)\b",
            re.I,
        ),
    ),
    (
        "price_target",
        re.compile(
            r"\b(price target|target price|fair value (estimate|of the stock)|"
            r"(over|under)valued|intrinsic value of (the stock|nvda|nvidia))\b",
            re.I,
        ),
    ),
    (
        "live_price",
        re.compile(
            r"\b((share|stock) price|market cap(italization)?|trading at|quote)\b.{0,40}"
            r"\b(now|today|right now|currently|at the moment|live|real[- ]time|this morning|"
            r"this afternoon|tonight)\b|"
            r"\b(current|today'?s|live|real[- ]time|latest) (share|stock) price\b|"
            r"\bwhat('s| is) (nvda|nvidia|the stock) trading at\b",
            re.I,
        ),
    ),
    (
        "forecast",
        re.compile(
            r"\b(will|is going to|going to|gonna|could|might) " + _STOCK + r" "
            r"(go up|go down|rise|fall|drop|crash|rally|hit|reach|double|triple|recover|"
            r"outperform|beat the market)\b|"
            r"\b(predict|forecast|project|estimate)\b.{0,30}\b(share|stock) price\b|"
            r"\b(share|stock) price\b.{0,30}\b(prediction|forecast|projection|outlook)\b|"
            r"\bwhere (will|is) (the stock|nvda|nvidia|the share price) (be|go|head)\b",
            re.I,
        ),
    ),
    (
        "advice_noun",
        re.compile(
            r"\b(investment|financial|trading|portfolio) "
            r"(advice|recommendation|tip|tips|strategy)\b|"
            r"\bshould i invest\b|\bis (nvda|nvidia|it|the stock) a (good|bad|safe|smart|great) "
            r"(buy|investment|stock|bet|long[- ]term hold)\b|\bhow (much|many shares) should i "
            r"(buy|invest|hold)\b",
            re.I,
        ),
    ),
]

BORDERLINE_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "market_terms",
        re.compile(
            r"\b((share|stock) price|market cap(italization)?|valuation|p/e|pe ratio|"
            r"price[- ]to[- ]earnings|analysts?|wall street|consensus|bull case|bear case|"
            r"portfolio|invest(ing)? in|trading|ticker|options|calls|puts)\b",
            re.I,
        ),
    ),
    (
        "future",
        re.compile(
            r"\b(next quarter|next year|next fiscal year|fiscal 2027|fy ?27|forecast|prediction|"
            r"predict|projection|will .* (grow|increase|decrease|decline) (in|next|by)|"
            r"going forward)\b",
            re.I,
        ),
    ),
    (
        "other_company",
        re.compile(
            r"\b(amd|advanced micro devices|intel|apple|microsoft|tesla|alphabet|google|amazon|"
            r"meta|broadcom|tsmc|qualcomm|arm holdings|openai)('s)?\b.{0,40}\b(revenue|profit|"
            r"income|assets|liabilities|equity|balance sheet|cash|debt|margin|earnings)\b",
            re.I,
        ),
    ),
]

IN_SCOPE_ANCHORS = re.compile(
    r"\b(annual report|annual review|10[- ]k|form 10[- ]k|proxy( statement)?|filing|filed|"
    r"balance sheet|income statement|cash flow|statement of|reported|disclosed|according to|"
    r"as of|fiscal|fy ?\d{2,4}|note \d+|item \d+|md&a|risk factors?|auditor|stockholders?'? "
    r"meeting|record date|compensation|five[- ]layer|chart|graph|figure|total return|"
    r"s&p ?500|nasdaq)\b",
    re.I,
)

Category = Literal[
    "filing_question",
    "investment_advice",
    "live_market_data",
    "forecast",
    "other_company",
    "unrelated",
]


class ScopeVerdict(BaseModel):
    """Structured output of the `small`-role fallback call."""

    in_scope: bool = Field(description="true only if NVIDIA's FY2026 annual report can answer it")
    category: Category
    reason: str = Field(description="one short sentence")


class ScopeDecision(BaseModel):
    in_scope: bool
    score: float
    rule: str | None = None
    source: Literal["rule", "llm"] = "rule"
    category: Category = "filing_question"
    reason: str = ""
    llm_called: bool = False
    refusal_markdown: str | None = None


_SCOPE_SYSTEM = (
    "You decide whether a user question can be answered from NVIDIA's fiscal 2026 annual report "
    "(Annual Review, Proxy Statement and Form 10-K for the year ended January 25, 2026). "
    "In scope: reported figures, financial statements, notes, MD&A explanations, risk factors, "
    "executive compensation, governance, the annual meeting, charts in the report. "
    "Out of scope: buy/sell/hold or investment advice, price targets, live or current market "
    "data (share price, market cap today), share-price forecasts, other companies' financials, "
    "and questions unrelated to the company. Respond with JSON only."
)


def rule_score(question: str, slots: QuerySlots | None = None) -> tuple[float, str | None]:
    """Returns (score, first rule that fired). Anchors and slots pull the score down."""
    score = 0.0
    fired: str | None = None
    for name, pattern in HARD_RULES:
        if pattern.search(question):
            score = 1.0
            fired = name
            break
    if fired is None:
        for name, pattern in BORDERLINE_RULES:
            if pattern.search(question):
                score = 0.5
                fired = name
                break
    if score == 0.0:
        return 0.0, None
    anchored = bool(IN_SCOPE_ANCHORS.search(question))
    slot_backed = bool(
        slots and (slots.metrics or slots.formulas or slots.statement or slots.fiscal_periods)
    )
    # Hard advice/live-price rules are not rescued by anchors ("should I buy based on the 10-K?"
    # is still advice); borderline ones are.
    if score < 1.0 and (anchored or (slot_backed and fired != "other_company")):
        score = max(0.0, score - 0.4)
    elif score == 1.0 and fired in {"forecast", "live_price"} and slot_backed:
        score = 0.6
    return round(score, 2), fired


def refusal_text(category: Category) -> str:
    what = {
        "investment_advice": (
            "I can't give investment advice — no buy, sell or hold views, and no price targets."
        ),
        "live_market_data": (
            "I don't have live market data such as today's share price or market capitalization."
        ),
        "forecast": "I can't forecast the share price or future results.",
        "other_company": (
            "I only cover NVIDIA's own annual report, not other companies' financials."
        ),
        "unrelated": "That question is outside what this assistant covers.",
        "filing_question": "That question is outside what this assistant covers.",
    }[category]
    return (
        f"{what}\n\n"
        "This assistant answers questions from **NVIDIA's fiscal 2026 annual report** "
        "(Annual Review, Proxy Statement and Form 10-K for the year ended January 25, 2026): "
        "reported figures, ratios computed from them, management's explanations, risk factors, "
        "executive pay and the annual meeting. For example: *What were total assets as of "
        "January 25, 2026?* or *Why did gross margin decrease in fiscal 2026?*"
    )


def _category_for_rule(rule: str | None) -> Category:
    return {
        "advice_verb": "investment_advice",
        "buy_sell_hold": "investment_advice",
        "price_target": "investment_advice",
        "advice_noun": "investment_advice",
        "live_price": "live_market_data",
        "forecast": "forecast",
        "other_company": "other_company",
        "market_terms": "live_market_data",
        "future": "forecast",
    }.get(rule or "", "unrelated")


def scope_gate(
    question: str,
    slots: QuerySlots | None = None,
    *,
    client: LLMClient | None = None,
    request_id: str | None = None,
) -> ScopeDecision:
    score, rule = rule_score(question, slots)
    if score >= REFUSE_AT:
        cat = _category_for_rule(rule)
        return ScopeDecision(
            in_scope=False,
            score=score,
            rule=rule,
            category=cat,
            reason=f"rule {rule} matched",
            refusal_markdown=refusal_text(cat),
        )
    if score < ASK_LLM_AT or client is None:
        return ScopeDecision(
            in_scope=True,
            score=score,
            rule=rule,
            reason="no out-of-scope rule fired" if rule is None else f"borderline rule {rule}",
        )
    try:
        verdict = client.json(
            f"QUESTION: {question}",
            ScopeVerdict,
            role="small",
            system=_SCOPE_SYSTEM,
            request_id=request_id,
            max_tokens=400,
        )
    except LLMError as exc:
        log.warning("scope fallback failed (%s); treating as in scope", str(exc)[:120])
        return ScopeDecision(
            in_scope=True,
            score=score,
            rule=rule,
            reason=f"borderline rule {rule}; small-model fallback failed",
            llm_called=True,
        )
    cat: Category = "filing_question" if verdict.in_scope else verdict.category
    if not verdict.in_scope and cat == "filing_question":
        cat = "unrelated"
    return ScopeDecision(
        in_scope=verdict.in_scope,
        score=score,
        rule=rule,
        source="llm",
        category=cat,
        reason=verdict.reason,
        llm_called=True,
        refusal_markdown=None if verdict.in_scope else refusal_text(cat),
    )
