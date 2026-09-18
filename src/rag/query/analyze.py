"""Query analysis — intent and expansion (architecture §4.4, plan Phase 4).

Step 1, rules: intent from slots and cue words, with a confidence. Step 2, only when the rules
are not confident (or for EXPLANATORY, which wants a numbers-free HyDE passage): ONE `small`
JSON call returning intent, sub-questions, paraphrases, section hints, `needs_image` and the
HyDE passage together. Rule-confident numeric questions therefore cost zero model calls.

Expansion follows the matrix in §4.4 — glossary expansion for every intent, decomposition
for computations, ≤ 2 paraphrases, HyDE for EXPLANATORY only, at most `max_queries` queries.
Glossary and decomposition queries are phrased like row-fact documents and carry a statement /
modality filter; the original question is always run unfiltered (filters are hints, not walls).

The seven intent classes are defined once in problem statement §4.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.config import FormulasConfig, load_formulas_config
from rag.core.logging import get_logger
from rag.llm import LLMClient, LLMError
from rag.query.slots import QuerySlots, SlotExtractor, get_slot_extractor

log = get_logger(__name__)

Intent = Literal[
    "POINT_LOOKUP",
    "COMPUTATION",
    "COMPARISON_TREND",
    "EXPLANATORY",
    "VISUAL",
    "CROSS_SECTION",
    "OUT_OF_SCOPE",
]
INTENTS: tuple[str, ...] = Intent.__args__  # type: ignore[attr-defined]

MIN_RULE_CONFIDENCE = 0.8
NUMERIC_INTENTS = {"POINT_LOOKUP", "COMPUTATION", "COMPARISON_TREND"}
COMPARISON_AGGREGATIONS = {"yoy", "change", "growth", "pct", "compare"}
ROW_FACT_MODALITIES = ["row_fact", "table"]

# Subsection ids the LLM may use as section hints (from the ingestion section tagger).
KNOWN_SUBSECTIONS = [
    "narrative_spread",
    "item_1_business",
    "item_1a_risk_factors",
    "item_5_market",
    "item_7_mdna",
    "item_7a_market_risk",
    "item_8_financials",
    "notes",
    "item_11_executive_compensation",
    "auditor_report",
    "notice_of_meeting",
    "proxy_summary",
    "corporate_governance",
    "compensation",
    "pay_vs_performance",
    "ownership",
    "proposals",
    "audit_matters",
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
            r"total shareholder return|nasdaq|s&p ?500|stock performance graph|pay mix|layers?|"
            r"infographic|illustration|visual)\b",
            re.I,
        ),
    ),
    (
        "CROSS_SECTION",
        re.compile(
            r"\b(pay[- ]versus[- ]performance|pay vs\.? performance|executive (pay|compensation).*"
            r"(revenue|growth|performance|net income)|compensation.*relate|"
            r"(annual|stockholders?'?|shareholders?'?) meeting|record date|"
            r"proxy.*(10[- ]k|financial)|(10[- ]k|financial).*proxy)\b",
            re.I,
        ),
    ),
    (
        "EXPLANATORY",
        re.compile(
            r"\b(why|explain|what drove|what drives|drivers?|reasons?|because|"
            r"how (did|does|do|has|have) .* (affect|impact|account|treat|recogni[sz]e|measure|"
            r"manage|mitigate|handle)|what caused|what causes|describe|discuss|"
            r"what (is|are) the (risk|risks|policy|policies|purpose)|accounting (policy|treatment)|"
            r"how (is|are) .* (accounted|recogni[sz]ed|measured|determined|calculated)|"
            r"what factors|what led to|what contributed)\b",
            re.I,
        ),
    ),
    (
        "COMPUTATION",
        re.compile(r"\b(calculate|compute|derive|work out)\b", re.I),
    ),
]


# ------------------------------------------------------------------------------ models


class RetrievalQuery(BaseModel):
    kind: Literal["original", "glossary", "decomposition", "paraphrase", "hyde", "section"]
    text: str
    where: dict | None = None


class LLMAnalysis(BaseModel):
    """Structured output of the single `small`-role analysis call."""

    intent: Intent
    # gpt-oss-20b sometimes pads lists with null / "" entries; accept and filter them rather
    # than spending a corrective retry.
    sub_questions: list[str | None] = Field(default_factory=list)
    paraphrases: list[str | None] = Field(default_factory=list, description="at most 2")
    section_hints: list[str | None] = Field(
        default_factory=list, description="subsection ids from the provided list"
    )
    needs_image: bool = False
    hyde_passage: str | None = Field(
        default=None,
        description="EXPLANATORY only: 2-3 sentences the disclosure would contain, no numbers",
    )


class Analysis(BaseModel):
    intent: Intent
    rule: str = Field(description="which rule fired, or 'default'")
    confidence: float = 1.0
    source: Literal["rule", "llm"] = "rule"
    llm_called: bool = False
    llm_intent: Intent | None = None
    retrieval_queries: list[str] = Field(description="query texts, original first")
    queries: list[RetrievalQuery] = Field(default_factory=list)
    section_hints: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    paraphrases: list[str] = Field(default_factory=list)
    hyde_passage: str | None = None
    hyde_rejected: bool = False
    needs_image: bool = False
    notes: list[str] = Field(default_factory=list)


class ExpansionSettings(BaseModel):
    max_queries: int = 4
    max_paraphrases: int = 2
    hyde_enabled: bool = True
    hyde_regenerate: bool = True
    llm_fallback_enabled: bool = True
    min_rule_confidence: float = MIN_RULE_CONFIDENCE
    glossary_enabled: bool = True
    filters_enabled: bool = True


# ----------------------------------------------------------------------------- intent


def rule_intent(question: str, slots: QuerySlots | None = None) -> tuple[Intent, str, float]:
    """Rule-based intent with confidence (§4.4 step 1)."""
    for intent, pattern in _RULES:
        m = pattern.search(question)
        if m:
            if intent == "COMPUTATION" and not (slots and (slots.formulas or slots.metrics)):
                continue  # "calculate" without a known metric: let the slots decide below
            conf = {"OUT_OF_SCOPE": 1.0, "VISUAL": 0.9, "CROSS_SECTION": 0.85}.get(intent, 0.9)
            return intent, f"{intent.lower()}:{m.group(0)}", conf
    if slots is None:
        return "POINT_LOOKUP", "default", 0.5
    if slots.formulas:
        return "COMPUTATION", f"formula:{slots.formulas[0]}", 0.9
    comparison = bool(set(slots.aggregation) & COMPARISON_AGGREGATIONS) or slots.direction
    if slots.metrics and (len(slots.fiscal_periods) >= 2 or comparison):
        return "COMPARISON_TREND", "metric+comparison", 0.9
    if slots.metrics:
        return "POINT_LOOKUP", "metric", 0.9
    if comparison or len(slots.fiscal_periods) >= 2:
        return "COMPARISON_TREND", "comparison-no-metric", 0.6
    return "POINT_LOOKUP", "default", 0.5


def guess_intent(question: str) -> tuple[Intent, str]:
    """Phase 3 entry point kept for callers that only want the rule intent."""
    intent, rule, _ = rule_intent(question, get_slot_extractor().extract(question))
    return intent, rule


# --------------------------------------------------------------------------- expansion


def _period_phrase(fy: int, extractor: SlotExtractor, statement: str | None) -> str:
    end = extractor.calendar.year_end(fy)
    if end is None:
        return f"FY{fy}"
    if statement == "balance_sheet":
        return f"as of {end:%b} {end.day}, {end.year} (FY{fy})"
    return f"for the fiscal year ended {end:%b} {end.day}, {end.year} (FY{fy})"


def canonical_query(
    line_item: str, statement: str | None, fiscal_years: list[int], extractor: SlotExtractor
) -> str:
    """Phrase a metric the way its row-fact document reads, so dense and BM25 both hit it."""
    label = extractor.statement_label(statement) or "financial statements"
    periods = " ".join(_period_phrase(fy, extractor, statement) for fy in fiscal_years)
    return f"NVIDIA {label} — {line_item}: {periods}".strip()


def _where(statement: str | None, enabled: bool) -> dict | None:
    if not enabled or not statement:
        return None
    return {"$and": [{"statement": statement}, {"modality": {"$in": ROW_FACT_MODALITIES}}]}


def build_queries(
    question: str,
    slots: QuerySlots,
    intent: Intent,
    *,
    extractor: SlotExtractor,
    formulas: FormulasConfig,
    settings: ExpansionSettings,
    paraphrases: list[str] | None = None,
    hyde: str | None = None,
    section_hints: list[str] | None = None,
    sub_questions: list[str] | None = None,
) -> list[RetrievalQuery]:
    """Expansion matrix (§4.4). Priority when the cap bites: original → glossary/decomposition
    → HyDE → sub-questions / section queries → paraphrases."""
    fys = slots.resolved_fiscal_years or [extractor.calendar.latest_fiscal_year]
    if intent == "COMPARISON_TREND" and len(fys) == 1:
        fys = [fys[0] - 1, fys[0]] if fys[0] - 1 in extractor.calendar.fiscal_years else fys
    out: list[RetrievalQuery] = [RetrievalQuery(kind="original", text=question.strip())]

    if settings.glossary_enabled:
        # Decomposition: one row-fact query per formula input (COMPUTATION), grouped in pairs
        # when a formula has more inputs than the cap allows.
        if intent == "COMPUTATION" and slots.formulas:
            inputs: list[tuple[str, str | None]] = []
            for fid in slots.formulas:
                f = formulas.formulas.get(fid)
                if f is None:
                    continue
                st = f.statement or formulas.default_statement
                for spec in f.inputs.values():
                    if "{metric}" in spec.line_item:
                        continue
                    if (spec.line_item, st) not in inputs:
                        inputs.append((spec.line_item, st))
            room = max(1, settings.max_queries - len(out))
            groups = _group(inputs, room)
            for group in groups:
                st = group[0][1]
                text = canonical_query("; ".join(li for li, _ in group), st, fys[-1:], extractor)
                out.append(
                    RetrievalQuery(
                        kind="decomposition", text=text, where=_where(st, settings.filters_enabled)
                    )
                )
        # Glossary: canonical row-fact phrasing per metric (max 2), statement-filtered — for
        # numeric intents only: measured in the Phase 4 ablation, row-fact queries push the
        # narrative an EXPLANATORY question needs out of the top 8. Decomposition already covers
        # the inputs of a recognised formula.
        metrics = slots.metrics[:2] if intent in NUMERIC_INTENTS else []
        if intent == "COMPUTATION" and slots.formulas:
            metrics = []
        for metric in metrics:
            st = extractor.metric_statement(metric)
            years = fys if intent == "COMPARISON_TREND" else fys[-1:]
            out.append(
                RetrievalQuery(
                    kind="glossary",
                    text=canonical_query(metric, st, years, extractor),
                    where=_where(st, settings.filters_enabled),
                )
            )

    if hyde and intent == "EXPLANATORY" and settings.hyde_enabled:
        out.append(RetrievalQuery(kind="hyde", text=hyde))

    hints = [h for h in (section_hints or []) if h in KNOWN_SUBSECTIONS]
    if intent == "CROSS_SECTION" and hints:
        for h in hints[:2]:
            out.append(
                RetrievalQuery(
                    kind="section",
                    text=question.strip(),
                    where={"subsection": h} if settings.filters_enabled else None,
                )
            )
    for sq in sub_questions or []:
        if intent in {"COMPARISON_TREND", "EXPLANATORY", "CROSS_SECTION"}:
            out.append(RetrievalQuery(kind="decomposition", text=sq))

    limit = {
        "POINT_LOOKUP": 0,
        "COMPUTATION": 0,
        "COMPARISON_TREND": 1,
        "EXPLANATORY": 2,
        "VISUAL": 1,
        "CROSS_SECTION": 2,
        "OUT_OF_SCOPE": 0,
    }[intent]
    for p in (paraphrases or [])[: min(limit, settings.max_paraphrases)]:
        out.append(RetrievalQuery(kind="paraphrase", text=p))

    # De-duplicate by (text, filter) and apply the cap.
    seen: set[str] = set()
    unique: list[RetrievalQuery] = []
    for q in out:
        key = q.text.strip().lower() + "|" + json.dumps(q.where, sort_keys=True)
        if q.text.strip() and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[: settings.max_queries]


def _group(items: list, room: int) -> list[list]:
    if not items:
        return []
    size = -(-len(items) // room)  # ceil
    return [items[i : i + size] for i in range(0, len(items), size)]


# --------------------------------------------------------------------------- LLM step


_ANALYSIS_SYSTEM = """You analyse questions about NVIDIA's fiscal 2026 annual report for a retrieval system. Classify the intent and propose retrieval expansions. Respond with JSON only.

Intents:
- POINT_LOOKUP: one reported figure or fact for one period.
- COMPUTATION: a ratio or amount derived from reported figures (current ratio, working capital, debt to equity...).
- COMPARISON_TREND: change or comparison across periods or items.
- EXPLANATORY: why / how / what drove — management's explanation, policies, risks.
- VISUAL: about a chart, graph, diagram or figure in the report.
- CROSS_SECTION: needs both the proxy statement and the 10-K, or governance/meeting facts.
- OUT_OF_SCOPE: investment advice, live market data, other companies.

Rules for expansions:
- paraphrases: at most 2 alternative phrasings using the filing's own vocabulary; no new facts.
- sub_questions: only for COMPARISON_TREND / EXPLANATORY / CROSS_SECTION, at most 2.
- section_hints: pick from {sections} — the subsections most likely to hold the answer (max 2).
- hyde_passage: ONLY for EXPLANATORY. Write 2-3 sentences in the style of the Form 10-K describing what the relevant disclosure would discuss. Do NOT state any figures, numbers, percentages, dates or years — describe the drivers in words only.
- needs_image: true only when the answer requires looking at a chart or diagram."""


def _strings(items: list[str | None]) -> list[str]:
    return [str(i).strip() for i in items if i is not None and str(i).strip()]


def _digits(text: str | None) -> bool:
    """True when the passage states a number. Digits glued to letters ("H20", "Q1", "GB200")
    are product / period names, not figures, and are allowed."""
    return bool(text and re.search(r"(?<![A-Za-z0-9])\d", text))


def _ask_llm(
    client: LLMClient,
    question: str,
    slots: QuerySlots,
    *,
    request_id: str | None,
    want_hyde: bool,
) -> LLMAnalysis:
    slot_summary = {
        "fiscal_periods": slots.fiscal_periods,
        "metrics": slots.metrics,
        "formulas": slots.formulas,
        "statement": slots.statement,
    }
    prompt = (
        f"QUESTION: {question}\n"
        f"SLOTS FOUND BY RULES: {slot_summary}\n"
        + ("Include hyde_passage.\n" if want_hyde else "Set hyde_passage to null.\n")
        + "Respond with the JSON object described in the system message."
    )
    return client.json(
        prompt,
        LLMAnalysis,
        role="small",
        system=_ANALYSIS_SYSTEM.replace("{sections}", ", ".join(KNOWN_SUBSECTIONS)),
        request_id=request_id,
        max_tokens=1200,
    )


def _regenerate_hyde(client: LLMClient, question: str, *, request_id: str | None) -> str | None:
    class _Hyde(BaseModel):
        passage: str

    try:
        out = client.json(
            f"QUESTION: {question}\n"
            "Write 2-3 sentences in the style of NVIDIA's Form 10-K describing what the disclosure "
            "answering this question would discuss. Use words only: no digits, no figures, no "
            "percentages, no dates. Respond with JSON only.",
            _Hyde,
            role="small",
            request_id=request_id,
            max_tokens=500,
        )
    except LLMError as exc:
        log.warning("HyDE regeneration failed: %s", str(exc)[:120])
        return None
    return None if _digits(out.passage) else out.passage.strip()


# --------------------------------------------------------------------------- entry


def analyze(
    question: str,
    slots: QuerySlots | None = None,
    *,
    client: LLMClient | None = None,
    request_id: str | None = None,
    settings: ExpansionSettings | None = None,
    extractor: SlotExtractor | None = None,
    formulas: FormulasConfig | None = None,
) -> Analysis:
    settings = settings or ExpansionSettings()
    extractor = extractor or get_slot_extractor()
    formulas = formulas or load_formulas_config()
    slots = slots or extractor.extract(question)

    intent, rule, confidence = rule_intent(question, slots)
    analysis = Analysis(
        intent=intent,
        rule=rule,
        confidence=confidence,
        retrieval_queries=[question.strip()],
        needs_image=intent == "VISUAL",
    )

    want_hyde = intent == "EXPLANATORY" and settings.hyde_enabled
    unsure = confidence < settings.min_rule_confidence
    llm: LLMAnalysis | None = None
    if client is not None and settings.llm_fallback_enabled and (unsure or want_hyde):
        try:
            llm = _ask_llm(client, question, slots, request_id=request_id, want_hyde=want_hyde)
            analysis.llm_called = True
            analysis.llm_intent = llm.intent
        except LLMError as exc:
            analysis.llm_called = True
            analysis.notes.append(f"small-model analysis failed: {str(exc)[:120]}")
            log.warning("analysis call failed (%s); using rule intent", str(exc)[:120])

    if llm is not None:
        if unsure and llm.intent != "OUT_OF_SCOPE":
            analysis.intent = llm.intent
            analysis.source = "llm"
            analysis.confidence = max(confidence, 0.8)
            analysis.rule = f"llm (rule said {intent} @ {confidence:.2f})"
        analysis.sub_questions = _strings(llm.sub_questions)[:2]
        analysis.paraphrases = _strings(llm.paraphrases)[: settings.max_paraphrases]
        analysis.section_hints = [h for h in _strings(llm.section_hints) if h in KNOWN_SUBSECTIONS][
            :2
        ]
        analysis.needs_image = analysis.needs_image or (
            llm.needs_image and analysis.intent == "VISUAL"
        )
        if analysis.intent == "EXPLANATORY" and settings.hyde_enabled:
            passage = (llm.hyde_passage or "").strip() or None
            if _digits(passage):
                analysis.hyde_rejected = True
                analysis.notes.append("HyDE passage contained digits; rejected")
                passage = (
                    _regenerate_hyde(client, question, request_id=request_id)
                    if settings.hyde_regenerate and client is not None
                    else None
                )
                if passage:
                    analysis.notes.append("HyDE regenerated without digits")
            analysis.hyde_passage = passage
        analysis.needs_image = analysis.needs_image or analysis.intent == "VISUAL"

    analysis.queries = build_queries(
        question,
        slots,
        analysis.intent,
        extractor=extractor,
        formulas=formulas,
        settings=settings,
        paraphrases=analysis.paraphrases,
        hyde=analysis.hyde_passage,
        section_hints=analysis.section_hints,
        sub_questions=analysis.sub_questions,
    )
    analysis.retrieval_queries = [q.text for q in analysis.queries]
    return analysis
