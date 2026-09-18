"""Phase 4 — scope gate, intent rules + expansion, rerank gate, slot-based formula selection.

All model calls are scripted (`FakeChat`); nothing here touches the network or the index.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from rag.calc.calculator import select_formulas_from_slots
from rag.core.config import RerankThresholds
from rag.core.schema import Chunk, ChunkMetadata
from rag.llm import LLMClient
from rag.query.analyze import (
    ExpansionSettings,
    analyze,
    build_queries,
    guess_intent,
    rule_intent,
)
from rag.query.rerank import apply_rerank, rerank_gate, rerank_node
from rag.query.retrieve import Candidate, RetrievalResult
from rag.query.scope import refusal_text, rule_score, scope_gate
from rag.query.slots import extract_slots, get_slot_extractor

# ------------------------------------------------------------------------------ helpers


class FakeChat:
    """Scripted chat model: returns the next JSON payload for every invoke."""

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[list] = []

    def bind(self, **kw):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        return AIMessage(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )


def fake_client(settings, payloads: list[dict]) -> tuple[LLMClient, FakeChat]:
    chat = FakeChat(payloads)
    client = LLMClient(settings, chat_factory=lambda cfg, role: chat, pacing_enabled=False)
    return client, chat


ANALYSIS_OK = {
    "intent": "EXPLANATORY",
    "sub_questions": ["What drove the gross margin decrease in fiscal 2026?"],
    "paraphrases": ["reasons for lower gross margin", "gross margin decline drivers"],
    "section_hints": ["item_7_mdna"],
    "needs_image": False,
    "hyde_passage": "Gross margin decreased primarily due to inventory provisions and product "
    "mix as the company transitioned to its next architecture.",
}


# ----------------------------------------------------------------------------- scope


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy NVIDIA stock?",
        "Should we sell NVDA now?",
        "Is NVDA a good investment?",
        "Buy or sell NVIDIA?",
        "What's the price target for NVDA?",
        "Is NVIDIA overvalued?",
        "What is NVIDIA's share price right now?",
        "What is the current stock price?",
        "What is NVDA trading at today?",
        "Will the stock go up next year?",
        "Give me investment advice on NVIDIA",
        "Should I hold my shares given the fiscal 2026 balance sheet?",
    ],
)
def test_scope_refuses_advice_and_live_data_without_a_model_call(question: str) -> None:
    d = scope_gate(question, extract_slots(question), client=None)
    assert not d.in_scope and d.source == "rule" and not d.llm_called
    assert d.refusal_markdown and "annual report" in d.refusal_markdown


@pytest.mark.parametrize(
    "question",
    [
        "What were NVIDIA's total assets as of Jan 25, 2026?",
        "What is the current ratio as of Jan 25, 2026?",
        "How much did inventories grow year-over-year in fiscal 2026?",
        "Why did gross margin decrease in fiscal 2026?",
        "How did the 5-year cumulative total return compare to the S&P 500?",
        "What does the 10-K say about stock price volatility risk?",
        "How many shares were repurchased in fiscal 2026?",
        "When is the 2026 annual meeting of stockholders?",
        "What was the cash pile at FY26 year end?",
        "What is the balance sheet total for liabilities?",
    ],
)
def test_scope_never_refuses_filing_questions(question: str) -> None:
    d = scope_gate(question, extract_slots(question), client=None)
    assert d.in_scope and not d.llm_called


def test_scope_borderline_asks_small_model_once(settings) -> None:
    client, chat = fake_client(
        settings,
        [{"in_scope": False, "category": "live_market_data", "reason": "asks for market cap"}],
    )
    q = "What is NVIDIA's market cap?"
    score, rule = rule_score(q, extract_slots(q))
    assert 0.4 <= score < 0.8 and rule == "market_terms"
    d = scope_gate(q, extract_slots(q), client=client, request_id="t")
    assert d.llm_called and d.source == "llm" and not d.in_scope
    assert d.category == "live_market_data" and len(chat.calls) == 1
    assert d.refusal_markdown == refusal_text("live_market_data")


def test_scope_borderline_in_scope_verdict_passes_through(settings) -> None:
    client, _ = fake_client(
        settings, [{"in_scope": True, "category": "filing_question", "reason": "MD&A outlook"}]
    )
    q = "What does management expect going forward?"
    d = scope_gate(q, extract_slots(q), client=client)
    assert d.in_scope and d.llm_called and d.refusal_markdown is None


def test_scope_other_company_is_not_rescued_by_metric_words() -> None:
    q = "What was AMD's revenue in 2025?"
    score, rule = rule_score(q, extract_slots(q))
    assert rule == "other_company" and score >= 0.4


def test_scope_fallback_failure_defaults_to_in_scope(settings) -> None:
    class Boom:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            raise RuntimeError("boom")

    client = LLMClient(
        settings, chat_factory=lambda cfg, role: Boom(), pacing_enabled=False, max_attempts=1
    )
    d = scope_gate("What is NVIDIA's market cap?", None, client=client)
    assert d.in_scope and d.llm_called


# ----------------------------------------------------------------------------- intent


@pytest.mark.parametrize(
    "question, intent",
    [
        ("What were NVIDIA's total assets as of Jan 25, 2026?", "POINT_LOOKUP"),
        ("What is the current ratio as of Jan 25, 2026?", "COMPUTATION"),
        ("What was the cash pile at FY26 year end?", "COMPUTATION"),
        ("How much did goodwill grow year-over-year?", "COMPARISON_TREND"),
        ("How did cash change from fiscal 2025 to fiscal 2026?", "COMPARISON_TREND"),
        ("Why did gross margin decrease in fiscal 2026?", "EXPLANATORY"),
        ("How does NVIDIA account for inventory provisions?", "EXPLANATORY"),
        ("How did the 5-year total return compare to the Nasdaq 100?", "VISUAL"),
        ("What are the layers in the five-layer cake?", "VISUAL"),
        ("How does executive pay-versus-performance relate to revenue growth?", "CROSS_SECTION"),
        ("When is the annual meeting and what is the record date?", "CROSS_SECTION"),
        ("Should I buy NVIDIA stock?", "OUT_OF_SCOPE"),
        ("What is the share price right now?", "OUT_OF_SCOPE"),
    ],
)
def test_intent_rules(question: str, intent: str) -> None:
    assert guess_intent(question)[0] == intent


def test_rule_confidence_low_without_slots_or_cues() -> None:
    q = "Who are the independent directors?"
    intent, rule, conf = rule_intent(q, extract_slots(q))
    assert intent == "POINT_LOOKUP" and rule == "default" and conf < 0.8


def test_rule_confident_numeric_question_costs_no_model_call(settings) -> None:
    client, chat = fake_client(settings, [ANALYSIS_OK])
    a = analyze("What were total assets as of Jan 25, 2026?", client=client)
    assert a.intent == "POINT_LOOKUP" and a.source == "rule" and not a.llm_called
    assert chat.calls == []
    assert [q.kind for q in a.queries] == ["original", "glossary"]
    assert a.queries[1].where == {
        "$and": [{"statement": "balance_sheet"}, {"modality": {"$in": ["row_fact", "table"]}}]
    }
    assert "Consolidated Balance Sheets" in a.queries[1].text
    assert "Jan 25, 2026" in a.queries[1].text


def test_unsure_question_uses_one_small_call_and_llm_intent(settings) -> None:
    payload = {**ANALYSIS_OK, "intent": "CROSS_SECTION", "hyde_passage": None}
    client, chat = fake_client(settings, [payload])
    a = analyze("Who are the independent directors?", client=client)
    assert a.llm_called and len(chat.calls) == 1
    assert a.intent == "CROSS_SECTION" and a.source == "llm" and a.llm_intent == "CROSS_SECTION"
    kinds = [q.kind for q in a.queries]
    assert kinds[0] == "original" and "section" in kinds and "paraphrase" in kinds
    assert len(a.queries) <= 4


def test_llm_cannot_override_a_confident_rule(settings) -> None:
    payload = {**ANALYSIS_OK, "intent": "POINT_LOOKUP"}
    client, _ = fake_client(settings, [payload])
    a = analyze("Why did gross margin decrease in fiscal 2026?", client=client)
    assert a.intent == "EXPLANATORY" and a.source == "rule" and a.llm_called  # called for HyDE
    assert a.hyde_passage and not a.hyde_rejected
    assert [q.kind for q in a.queries] == ["original", "hyde", "decomposition", "paraphrase"]


# --------------------------------------------------------------------------- expansion


def test_expansion_matrix_caps_and_kinds() -> None:
    ex = get_slot_extractor()
    from rag.core.config import load_formulas_config

    f = load_formulas_config()
    s = ExpansionSettings()
    q = "What is the quick ratio as of Jan 25, 2026?"
    qs = build_queries(q, extract_slots(q), "COMPUTATION", extractor=ex, formulas=f, settings=s)
    assert len(qs) <= 4 and qs[0].kind == "original" and qs[0].where is None
    assert all(x.kind == "decomposition" for x in qs[1:])
    joined = " ".join(x.text for x in qs[1:])
    for li in ("cash and cash equivalents", "marketable securities", "accounts receivable net"):
        assert li in joined

    q = "How much did inventories grow year-over-year in fiscal 2026?"
    qs = build_queries(
        q, extract_slots(q), "COMPARISON_TREND", extractor=ex, formulas=f, settings=s
    )
    assert [x.kind for x in qs] == ["original", "glossary"]
    assert "Jan 26, 2025" in qs[1].text and "Jan 25, 2026" in qs[1].text

    q = "Why did gross margin decrease?"
    qs = build_queries(
        q,
        extract_slots(q),
        "EXPLANATORY",
        extractor=ex,
        formulas=f,
        settings=s,
        paraphrases=["p1", "p2", "p3"],
        hyde="a numbers-free passage",
        sub_questions=["s1", "s2"],
    )
    assert len(qs) == 4 and [x.kind for x in qs] == [
        "original",
        "hyde",
        "decomposition",
        "decomposition",
    ]

    q = "What are the layers of the five-layer cake?"
    qs = build_queries(
        q,
        extract_slots(q),
        "VISUAL",
        extractor=ex,
        formulas=f,
        settings=s,
        paraphrases=["p1", "p2"],
    )
    assert [x.kind for x in qs] == ["original", "paraphrase"]


def test_explanatory_gets_no_row_fact_filter_queries() -> None:
    a = analyze("What drove NVIDIA's revenue growth in fiscal year 2026?")
    assert a.intent == "EXPLANATORY"
    assert all(q.where is None for q in a.queries)


def test_filters_can_be_disabled() -> None:
    a = analyze(
        "What were total assets as of Jan 25, 2026?",
        settings=ExpansionSettings(filters_enabled=False),
    )
    assert len(a.queries) == 2 and a.queries[1].where is None


def test_hyde_with_digits_is_rejected_and_regenerated(settings) -> None:
    bad = {**ANALYSIS_OK, "hyde_passage": "Gross margin fell to 71.1% in fiscal 2026."}
    client, chat = fake_client(settings, [bad, {"passage": "Gross margin fell on product mix."}])
    a = analyze("Why did gross margin decrease in fiscal 2026?", client=client)
    assert a.hyde_rejected and a.hyde_passage == "Gross margin fell on product mix."
    assert len(chat.calls) == 2
    assert any("HyDE regenerated" in n for n in a.notes)


def test_hyde_with_digits_is_dropped_when_regeneration_disabled(settings) -> None:
    bad = {**ANALYSIS_OK, "hyde_passage": "Gross margin fell to 71.1%."}
    client, chat = fake_client(settings, [bad])
    a = analyze(
        "Why did gross margin decrease in fiscal 2026?",
        client=client,
        settings=ExpansionSettings(hyde_regenerate=False),
    )
    assert a.hyde_rejected and a.hyde_passage is None and len(chat.calls) == 1
    assert "hyde" not in [q.kind for q in a.queries]


def test_hyde_regeneration_still_containing_digits_is_dropped(settings) -> None:
    bad = {**ANALYSIS_OK, "hyde_passage": "Margin fell 4 points."}
    client, _ = fake_client(settings, [bad, {"passage": "Still 4 points."}])
    a = analyze("Why did gross margin decrease in fiscal 2026?", client=client)
    assert a.hyde_rejected and a.hyde_passage is None


def test_hyde_disabled_means_no_call_for_explanatory(settings) -> None:
    client, chat = fake_client(settings, [ANALYSIS_OK])
    a = analyze(
        "Why did gross margin decrease in fiscal 2026?",
        client=client,
        settings=ExpansionSettings(hyde_enabled=False),
    )
    assert not a.llm_called and chat.calls == [] and a.queries[0].kind == "original"


# -------------------------------------------------------------------------- rerank gate


def _cand(cid: str, rrf: float, dense: int | None, bm25: int | None, modality="text") -> Candidate:
    return Candidate(
        chunk_id=cid,
        modality=modality,
        page=1,
        section="form_10k",
        subsection="item_8_financials",
        breadcrumb="b",
        rrf=rrf,
        dense_rank=dense,
        bm25_rank=bm25,
        source="both" if dense and bm25 else ("dense" if dense else "bm25"),
    )


def _chunk(cid: str, modality: str = "text", line_item: str | None = None) -> Chunk:
    return Chunk(
        id=cid,
        document=f"[b]\n{cid} body",
        metadata=ChunkMetadata(
            chunk_id=cid,
            modality=modality,
            section="form_10k",
            subsection="item_8_financials",
            page=1,
            breadcrumb="b",
            token_count=10,
            statement="balance_sheet" if line_item else "none",
            line_item_norm=line_item,
        ),
    )


class _Store:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = {c.id: c for c in chunks}

    def get(self, cid: str) -> Chunk | None:
        return self.chunks.get(cid)

    def table_parts(self, pid: str) -> list[Chunk]:
        return []


def _retrieval(pool: list[Candidate], queries: int = 2, fused_count: int = 40) -> RetrievalResult:
    return RetrievalResult(
        queries=["q"] * queries,
        fused_count=fused_count,
        pool=pool,
        candidates=[c.model_copy() for c in pool[:8]],
    )


def test_gate_s1_all_required_metrics_in_top5() -> None:
    store = _Store(
        [
            _chunk("r_assets", "row_fact", "total assets"),
            _chunk("r_liab", "row_fact", "total liabilities"),
            _chunk("t1"),
        ]
    )
    pool = [_cand("t1", 0.03, 1, 2), _cand("r_assets", 0.02, 2, 1), _cand("r_liab", 0.01, 3, 3)]
    r = _retrieval(pool)
    skip = rerank_gate(
        r,
        intent="COMPUTATION",
        required_metrics=["total assets", "total liabilities"],
        store=store,
        final_k=8,
        margin_skip_ratio=0.3,
    )
    assert skip and skip[0] == "S1"
    # counter-example: one metric missing → no S1 skip (and S2/S3 do not fire either)
    skip = rerank_gate(
        r,
        intent="COMPUTATION",
        required_metrics=["total assets", "goodwill"],
        store=store,
        final_k=8,
        margin_skip_ratio=0.3,
    )
    assert skip is None
    # counter-example: explanatory intent never uses S1
    skip = rerank_gate(
        r,
        intent="EXPLANATORY",
        required_metrics=["total assets"],
        store=store,
        final_k=8,
        margin_skip_ratio=0.3,
    )
    assert skip is None


def test_gate_s2_single_query_small_fused_set() -> None:
    pool = [_cand("a", 0.03, 1, 2), _cand("b", 0.02, 2, 1)]
    skip = rerank_gate(
        _retrieval(pool, queries=1, fused_count=6),
        intent="EXPLANATORY",
        required_metrics=[],
        store=None,
        final_k=8,
        margin_skip_ratio=0.3,
    )
    assert skip and skip[0] == "S2"
    for queries, fused in ((2, 6), (1, 9)):
        assert (
            rerank_gate(
                _retrieval(pool, queries=queries, fused_count=fused),
                intent="EXPLANATORY",
                required_metrics=[],
                store=None,
                final_k=8,
                margin_skip_ratio=0.3,
            )
            is None
        )


def test_gate_s3_agreed_top1_with_margin() -> None:
    pool = [_cand("a", 0.0328, 1, 1), _cand("b", 0.0200, 2, 3)]
    skip = rerank_gate(
        _retrieval(pool),
        intent="EXPLANATORY",
        required_metrics=[],
        store=None,
        final_k=8,
        margin_skip_ratio=0.3,
    )
    assert skip and skip[0] == "S3"
    # counter-examples: margin too small; not rank 1 in both lists
    pool_close = [_cand("a", 0.0328, 1, 1), _cand("b", 0.0300, 2, 3)]
    assert (
        rerank_gate(
            _retrieval(pool_close),
            intent="EXPLANATORY",
            required_metrics=[],
            store=None,
            final_k=8,
            margin_skip_ratio=0.3,
        )
        is None
    )
    pool_split = [_cand("a", 0.0328, 1, 2), _cand("b", 0.0200, 2, 1)]
    assert (
        rerank_gate(
            _retrieval(pool_split),
            intent="EXPLANATORY",
            required_metrics=[],
            store=None,
            final_k=8,
            margin_skip_ratio=0.3,
        )
        is None
    )


def test_apply_rerank_reorders_drops_below_floor_and_keeps_min() -> None:
    store = _Store([_chunk(c) for c in "abcdef"])
    pool = [_cand(c, 0.03 - i * 0.001, i + 1, i + 1) for i, c in enumerate("abcdef")]
    th = RerankThresholds(
        model="m", candidates=30, margin_skip_ratio=0.3, drop_floor_logit=-2.0, min_keep=3
    )
    scores = [-5.0, 3.0, -4.0, 1.0, -3.0, -6.0]  # b > d > e > c > a > f
    updated, decision = apply_rerank(
        store, _retrieval(pool), "q", scores=scores, thresholds=th, final_k=8, intent="EXPLANATORY"
    )
    assert decision.applied and updated.rerank_applied
    assert updated.top_ids == ["b", "d", "e"]  # e kept by min_keep, then c/a/f dropped
    assert decision.dropped == ["c", "a", "f"] and decision.kept == 3
    assert updated.pool[0].rerank_rank == 1 and updated.pool[0].rerank_score == 3.0


def test_rerank_node_uses_gate_then_scripted_scores() -> None:
    store = _Store([_chunk(c) for c in "abc"])
    pool = [_cand("a", 0.0328, 1, 2), _cand("b", 0.0300, 2, 1), _cand("c", 0.01, 3, 3)]
    th = RerankThresholds(
        model="m", candidates=30, margin_skip_ratio=0.3, drop_floor_logit=-2.0, min_keep=3
    )

    class Stub:
        model_id = "stub"

        def score(self, query, docs):
            return [0.0, 5.0, 1.0]

    updated, decision = rerank_node(
        store,
        _retrieval(pool),
        "q",
        intent="EXPLANATORY",
        required_metrics=[],
        thresholds=th,
        final_k=8,
        reranker=Stub(),
    )
    assert decision.applied and updated.top_ids == ["b", "c", "a"]
    disabled = th.model_copy(update={"enabled": False})
    _, decision = rerank_node(
        store,
        _retrieval(pool),
        "q",
        intent="EXPLANATORY",
        required_metrics=[],
        thresholds=disabled,
        final_k=8,
        reranker=Stub(),
    )
    assert not decision.applied and decision.gate == "disabled"


# ---------------------------------------------------------------- calculator from slots


LINE_ITEMS = ["total assets", "inventories", "goodwill", "total current assets"]


def test_formula_selection_from_slots_named_formula_wins() -> None:
    reqs = select_formulas_from_slots(extract_slots("current ratio as of Jan 26, 2025"), LINE_ITEMS)
    assert [(r.formula, r.fiscal_year, r.metric) for r in reqs] == [("current_ratio", 2025, None)]


def test_formula_selection_from_slots_yoy_pair() -> None:
    reqs = select_formulas_from_slots(
        extract_slots("How much did inventories grow year-over-year in fiscal 2026?"), LINE_ITEMS
    )
    assert [(r.formula, r.fiscal_year, r.metric) for r in reqs] == [
        ("yoy_change_pct", 2026, "inventories"),
        ("yoy_change_abs", 2026, "inventories"),
    ]


def test_formula_selection_two_periods_uses_latest_as_current() -> None:
    reqs = select_formulas_from_slots(
        extract_slots("How did total assets change from fiscal 2025 to fiscal 2026?"), LINE_ITEMS
    )
    assert reqs and reqs[0].fiscal_year == 2026 and reqs[0].metric == "total assets"


def test_formula_selection_bare_year_resolves_forward() -> None:
    reqs = select_formulas_from_slots(
        extract_slots("How much did goodwill grow in 2025?"), LINE_ITEMS
    )
    assert reqs and reqs[0].fiscal_year == 2026


def test_formula_selection_needs_a_cue() -> None:
    assert select_formulas_from_slots(extract_slots("What were total assets?"), LINE_ITEMS) == []
    assert (
        select_formulas_from_slots(
            extract_slots("How did buybacks compare with dividends?"), LINE_ITEMS
        )
        == []
    )


def test_hyde_digit_check_allows_product_names() -> None:
    from rag.query.analyze import _digits

    assert not _digits("The H20 charge and GB200 ramp in Q1 are discussed.")
    assert _digits("Gross margin fell to 71.1%.")
    assert _digits("in fiscal 2026")
    assert _digits("a $4.5 billion charge")
