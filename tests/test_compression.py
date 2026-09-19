"""Phase 5 — compression classifier rules, compressors, fidelity guard (architecture §4.8, §6)."""

from __future__ import annotations

import numpy as np
import pytest

from rag.compress.classifier import break_even, classify, stage_b_score
from rag.compress.compressors import (
    CompressedChunk,
    apply_compression,
    row_select,
    select_sentences,
)
from rag.compress.features import (
    ChunkFeatures,
    FeatureExtractor,
    QueryFeatures,
    SentenceView,
    numeric_density,
)
from rag.compress.fidelity import check_fidelity
from rag.core.config import CompressionThresholds, Price, StageBThresholds
from rag.core.schema import Chunk, ChunkMetadata, SentenceRecord
from rag.query.assemble import assemble_context, block_body
from rag.query.retrieve import Candidate

TH = CompressionThresholds(
    skip_budget_tokens=1500,
    narrative_pressure_ratio=0.5,
    dedupe_cosine=0.95,
    sentence_relevance_tau=0.62,
    numeric_density_cap=0.15,
    min_reduction_for_llm=0.40,
    stage_b_thresholds=StageBThresholds(keep=0.35, llm=0.60),
)


def _q(intent="POINT_LOOKUP", total=3000, budget=2500, n=8) -> QueryFeatures:
    return QueryFeatures(
        intent=intent,
        n_chunks=n,
        total_tokens=total,
        budget_ratio=total / budget,
        context_budget=budget,
        queries=["q"],
    )


def _f(
    cid,
    modality="text",
    tokens=300,
    density=0.5,
    dup=0.0,
    dup_of=None,
    num=0.02,
    rank=1,
    rows=None,
    score=0.03,
    kind="rrf",
):
    return ChunkFeatures(
        chunk_id=cid,
        rank=rank,
        modality=modality,
        chunk_tokens=tokens,
        rerank_score=score,
        score_kind=kind,
        relevance_density=density,
        max_dup_sim=dup,
        dup_of=dup_of,
        numeric_density=num,
        table_rows=rows,
        n_sentences=8 if modality == "text" else 0,
    )


def _classify(q, feats, **kw):
    return classify(q, feats, thresholds=TH, drop_floor_logit=-2.0, min_keep=3, **kw)


# --------------------------------------------------------------------------- stage A rules


def test_tables_and_row_facts_never_reach_extract_llm() -> None:
    q = _q("POINT_LOOKUP", total=6000)
    feats = [
        _f("r1", "row_fact", tokens=60, density=0.0, rank=1),
        _f("t1", "table", tokens=400, density=0.0, rows=36, rank=2),
        _f("t2", "table", tokens=200, density=0.0, rows=6, rank=3),
        _f("fig", "figure", tokens=300, density=0.0, rank=4),
        _f("x1", "text", tokens=500, density=0.0, rank=5),
    ]
    d = _classify(q, feats)
    actions = {c.chunk_id: c.action for c in d.chunks}
    assert actions["r1"] == "KEEP" and actions["fig"] == "KEEP"
    assert actions["t1"] == "ROW_SELECT"  # > 12 rows, lookup intent
    assert actions["t2"] == "KEEP"  # small table
    assert all(a != "EXTRACT_LLM" for cid, a in actions.items() if cid != "x1")


def test_row_select_only_for_lookup_and_computation_intents() -> None:
    feats = [_f("t1", "table", tokens=400, rows=36)]
    assert _classify(_q("EXPLANATORY", total=6000), feats).chunks[0].action == "KEEP"
    assert _classify(_q("COMPUTATION", total=6000), feats).chunks[0].action == "ROW_SELECT"


def test_context_under_skip_budget_makes_no_compressor_calls() -> None:
    q = _q("EXPLANATORY", total=1200, budget=2500)
    feats = [_f(f"x{i}", tokens=300, density=0.0, rank=i + 1) for i in range(4)]
    d = _classify(q, feats)
    assert not d.query_needs_compression
    assert d.skip_reason and d.skip_reason.startswith("R5")
    assert all(c.action == "KEEP" for c in d.chunks)


def test_dedupe_applies_to_narrative_only_and_never_to_row_facts() -> None:
    q = _q("POINT_LOOKUP", total=6000)
    feats = [
        _f("r1", "row_fact", rank=1),
        _f("r2", "row_fact", dup=0.99, dup_of="r1", rank=2),  # boilerplate similarity
        _f("h20_a", "text", tokens=400, density=0.3, rank=3),
        _f("h20_b", "text", tokens=400, density=0.3, dup=0.97, dup_of="h20_a", rank=4),
        _f("mix", "text", tokens=400, density=0.3, dup=0.97, dup_of="r1", rank=5),
    ]
    actions = {c.chunk_id: c.action for c in _classify(q, feats).chunks}
    assert actions["r2"] == "KEEP"
    assert actions["h20_b"] == "DEDUPE"
    assert actions["mix"] != "DEDUPE"  # duplicate of a non-narrative chunk does not count


def test_drop_below_rerank_floor_only_past_min_keep_and_only_with_logits() -> None:
    q = _q("EXPLANATORY", total=6000)
    feats = [
        _f("a", score=-5.0, kind="rerank", rank=2),
        _f("b", score=-5.0, kind="rerank", rank=4),
        _f("c", score=-5.0, kind="rrf", rank=5),  # RRF scores are not logits
    ]
    actions = {c.chunk_id: c.action for c in _classify(q, feats).chunks}
    assert actions["a"] != "DROP" and actions["b"] == "DROP" and actions["c"] != "DROP"


def test_numeric_density_caps_at_extract_light() -> None:
    q = _q("POINT_LOOKUP", total=8000)
    f = _f("x", tokens=1600, density=0.0, num=0.3)  # above the 1,500 skip budget
    d = _classify(q, [f])
    assert d.chunks[0].action == "EXTRACT_LIGHT" and d.chunks[0].reason.startswith("R6")


# --------------------------------------------------------------------------- stage B


def test_stage_b_score_and_thresholds() -> None:
    q = _q("POINT_LOOKUP", total=8000)  # budget ratio 3.2 → pressure 1.0
    score, parts = stage_b_score(_f("x", tokens=1600, density=0.0), q)
    assert score == pytest.approx(0.45 + 0.25 + 0.20 + 0.10, abs=1e-3)
    filler = _f("filler", tokens=1600, density=1.0, rank=9)  # keeps narrative above the skip budget
    d = _classify(q, [_f("x", tokens=600, density=0.0), filler])
    assert d.chunks[0].action == "EXTRACT_LLM" and d.chunks[0].stage == "B"
    d = _classify(q, [_f("y", tokens=600, density=0.9), filler])
    assert d.chunks[0].action in {"KEEP", "EXTRACT_LIGHT"}
    d = _classify(q, [_f("z", tokens=200, density=0.0), filler])  # too short for the LLM
    assert d.chunks[0].action == "EXTRACT_LIGHT"


def test_break_even_quota_and_price_modes() -> None:
    be = break_even(500, 0.8, narrative_tokens=2000, thresholds=TH)
    assert be.ok and be.mode == "quota" and be.expected_reduction == pytest.approx(0.64)
    assert not break_even(500, 0.3, narrative_tokens=2000, thresholds=TH).ok  # ρ 0.24 < 0.40
    assert not break_even(500, 0.8, narrative_tokens=1000, thresholds=TH).ok  # under skip budget
    cheap_small = Price(input=0.1, output=0.4)
    dear_large = Price(input=3.0, output=15.0)
    be = break_even(
        500,
        0.8,
        narrative_tokens=2000,
        thresholds=TH,
        small_price=cheap_small,
        large_price=dear_large,
    )
    assert be.ok and be.mode == "price"
    be = break_even(
        500,
        0.8,
        narrative_tokens=2000,
        thresholds=TH,
        small_price=dear_large,
        large_price=cheap_small,
    )
    assert not be.ok


# --------------------------------------------------------------------------- fidelity


def test_fidelity_guard_altered_digit_reverts_and_reordered_sentences_pass() -> None:
    src = "Revenue was $215,938 million in fiscal 2026. Gross margin was 71.1%. Inventories rose."
    ok = check_fidelity("Gross margin was 71.1%. Revenue was $215,938 million in fiscal 2026.", src)
    assert ok.ok and ok.numbers_checked == 3
    bad = check_fidelity("Revenue was $215,983 million in fiscal 2026.", src)
    assert not bad.ok and bad.missing_numbers == ["$215,983"]
    rewritten = check_fidelity(
        "Margins fell.", src, sentences=["Margins fell.", "Gross margin was 71.1%."]
    )
    assert rewritten.ok and rewritten.non_verbatim == ["Margins fell."]


def test_numeric_density_feature() -> None:
    assert numeric_density("no numbers here at all") == 0.0
    assert numeric_density("$1,000 and 2% of 300") == pytest.approx(3 / 5)


# --------------------------------------------------------------------------- compressors


BALANCE_SHEET = """Table tbl_p141_0 (PDF p. 141)
Summary: Consolidated balance sheets.
(part 1 of 2)
|  | Jan 25, 2026 | Jan 26, 2025 |
|---|---|---|
| Assets |  |  |
| Current assets: |  |  |
| Cash and cash equivalents | $ 10,605 | $ 8,589 |
| Marketable securities | 51,951 | 34,621 |
| Accounts receivable, net | 38,466 | 23,065 |
| Inventories | 21,403 | 10,080 |
| Total current assets | 125,605 | 80,126 |
| Goodwill | 20,832 | 5,188 |
| Total assets | 206,803 | 111,601 |
| Liabilities and Shareholders' Equity |  |  |
| Accounts payable | 7,032 | 6,310 |
| Total current liabilities | 32,163 | 18,047 |"""


def test_row_select_keeps_query_metrics_sections_and_totals() -> None:
    text, kept, total = row_select(
        BALANCE_SHEET, ["total current assets", "total current liabilities"]
    )
    assert total == 12 and kept == 6
    assert "Total current assets" in text and "Total current liabilities" in text
    assert "Total assets" in text  # totals always kept
    assert "| Assets |" in text and "Current assets:" in text  # section labels
    assert "Goodwill" not in text and "Marketable securities" not in text
    assert text.startswith("Table tbl_p141_0") and "| Jan 25, 2026 |" in text
    assert check_fidelity(text, BALANCE_SHEET).ok
    # no metric matched → table kept whole (kept == 0 signals the caller)
    assert row_select(BALANCE_SHEET, ["revenue"])[1] > 0  # totals + sections still match
    assert row_select("no table here", ["x"]) == ("no table here", 0, 0)


def test_select_sentences_threshold_neighbours_and_fallback() -> None:
    view = SentenceView(
        sentence_ids=list("abcdef"),
        texts=[f"s{i}" for i in range(6)],
        starts=[0] * 6,
        ends=[0] * 6,
        scores=[0.1, 0.9, 0.1, 0.1, 0.1, 0.8],
    )
    assert select_sentences(view, 0.62) == [0, 1, 2, 4, 5]
    low = view.model_copy(update={"scores": [0.1, 0.3, 0.1, 0.1, 0.2, 0.1]})
    assert select_sentences(low, 0.62) == [0, 1, 2, 3, 4, 5]  # top-2 (1, 4) ± neighbours
    assert select_sentences(SentenceView(), 0.62) == []


# ---------------------------------------------------------------- end-to-end with a tiny store

H20_SENTENCE = (
    "As a result of these requirements, we incurred a $4.5 billion charge in the first "
    "quarter of fiscal year 2026 associated with H20."
)


def _chunk(cid, modality, body, **meta) -> Chunk:
    return Chunk(
        id=cid,
        document=f"[Form 10-K > Item 1 > Regulation]\n{body}",
        metadata=ChunkMetadata(
            chunk_id=cid,
            modality=modality,
            section="form_10k",
            subsection="item_1_business",
            page=meta.pop("page", 98),
            breadcrumb="Form 10-K > Item 1 > Regulation",
            token_count=meta.pop("token_count", 50),
            **meta,
        ),
    )


class _Embedder:
    def embed_queries(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


class _Store:
    """Two H20 duplicates (parallel vectors) + one unrelated paragraph + a row fact."""

    def __init__(self):
        h20 = H20_SENTENCE + " In August 2025, licenses were granted."
        self.chunks = {
            "text_p98_0": _chunk("text_p98_0", "text", h20, page=98),
            "text_p114_2": _chunk("text_p114_2", "text", h20, page=114),
            "text_p50_0": _chunk(
                "text_p50_0", "text", "Weather was mild. Nothing to see.", page=50
            ),
            "rowfact_x": _chunk(
                "rowfact_x",
                "row_fact",
                "Inventories: $21,403 million as of Jan 25, 2026.",
                statement="balance_sheet",
                line_item_norm="inventories",
                page=141,
            ),
        }
        self._vec_ids = list(self.chunks)
        self._vectors = np.array(
            [[1.0, 0.0], [0.999, 0.04], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
        )
        self._vectors /= np.linalg.norm(self._vectors, axis=1, keepdims=True)
        self.sentences = [
            SentenceRecord(
                sentence_id="s1",
                chunk_id="text_p98_0",
                start=0,
                end=10,
                text=H20_SENTENCE,
            ),
            SentenceRecord(
                sentence_id="s2",
                chunk_id="text_p98_0",
                start=11,
                end=20,
                text="In August 2025, licenses were granted.",
            ),
            SentenceRecord(
                sentence_id="s3",
                chunk_id="text_p114_2",
                start=0,
                end=10,
                text=H20_SENTENCE,
            ),
            SentenceRecord(
                sentence_id="s4", chunk_id="text_p50_0", start=0, end=10, text="Weather was mild."
            ),
            SentenceRecord(
                sentence_id="s5", chunk_id="text_p50_0", start=11, end=20, text="Nothing to see."
            ),
        ]
        self.sentence_emb = np.array(
            [[1.0, 0.0], [0.2, 0.98], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )
        self.embedder = _Embedder()

    def get(self, cid):
        return self.chunks.get(cid)

    def image_path(self, rel):
        return None


def _cand(chunk: Chunk) -> Candidate:
    m = chunk.metadata
    return Candidate(
        chunk_id=chunk.id,
        modality=m.modality,
        page=m.page,
        section=m.section,
        subsection=m.subsection,
        breadcrumb=m.breadcrumb,
        source="both",
        rrf=0.03,
    )


def test_h20_duplicates_are_deduplicated_with_merged_citations() -> None:
    store = _Store()
    fx = FeatureExtractor(store, sentence_tau=0.62)
    cands = [
        _cand(store.chunks[c]) for c in ("text_p98_0", "text_p114_2", "text_p50_0", "rowfact_x")
    ]
    qf, feats, views = fx.extract(
        cands, ["H20 charge"], intent="EXPLANATORY", context_budget=100, body_of=block_body
    )
    by = {f.chunk_id: f for f in feats}
    assert by["text_p114_2"].max_dup_sim > 0.95 and by["text_p114_2"].dup_of == "text_p98_0"
    assert by["text_p98_0"].relevance_density == 0.5  # one of two sentences ≥ tau
    assert by["text_p50_0"].relevance_density == 0.0
    d = _classify(qf, feats)
    actions = {c.chunk_id: c.action for c in d.chunks}
    assert actions["text_p114_2"] == "DEDUPE" and actions["rowfact_x"] == "KEEP"
    res = apply_compression(
        store, d, views, question="H20 charge", metrics=[], thresholds=TH, body_of=block_body
    )
    by_id = res.by_id()
    assert by_id["text_p114_2"].dropped and by_id["text_p114_2"].merged_into == "text_p98_0"
    assert by_id["text_p98_0"].merged_from == ["text_p114_2"]
    assert res.violations == [] and res.llm_calls == 0
    ctx = assemble_context(
        store, cands, [], token_budget=5000, intent="EXPLANATORY", compressed=by_id
    )
    ids = [b.chunk_id for b in ctx.blocks]
    assert "text_p114_2" not in ids and "text_p98_0" in ids
    kept = next(b for b in ctx.blocks if b.chunk_id == "text_p98_0")
    assert "also PDF p.114" in kept.header and kept.merged_from == ["text_p114_2"]


def test_extract_light_keeps_relevant_sentences_and_passes_fidelity() -> None:
    store = _Store()
    fx = FeatureExtractor(store, sentence_tau=0.62)
    cands = [_cand(store.chunks["text_p98_0"])]
    qf, feats, views = fx.extract(
        cands, ["H20 charge"], intent="EXPLANATORY", context_budget=100, body_of=block_body
    )
    d = _classify(qf, feats)
    d.chunks[0].action = "EXTRACT_LIGHT"
    res = apply_compression(
        store, d, views, question="H20 charge", metrics=[], thresholds=TH, body_of=block_body
    )
    out = res.chunks[0]
    assert out.applied == "EXTRACT_LIGHT" and out.fidelity and out.fidelity.ok
    assert "$4.5 billion" in out.text


def test_extract_llm_output_with_altered_number_is_reverted(settings) -> None:
    import json

    from langchain_core.messages import AIMessage

    from rag.llm import LLMClient

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(
                content=json.dumps({"sentences": ["we incurred a $4.6 billion charge"]}),
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    client = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    store = _Store()
    fx = FeatureExtractor(store, sentence_tau=0.62)
    cands = [_cand(store.chunks["text_p98_0"])]
    qf, feats, views = fx.extract(
        cands, ["H20 charge"], intent="EXPLANATORY", context_budget=100, body_of=block_body
    )
    d = _classify(qf, feats)
    d.chunks[0].action = "EXTRACT_LLM"
    res = apply_compression(
        store,
        d,
        views,
        question="H20",
        metrics=[],
        thresholds=TH,
        body_of=block_body,
        client=client,
    )
    out = res.chunks[0]
    assert out.llm_called and res.llm_calls == 1
    assert (
        out.reverted
        and out.applied == "KEEP"
        and out.text == block_body(store.chunks["text_p98_0"])
    )
    assert res.violations and "$4.6" in res.violations[0]


def test_compressed_chunk_model_roundtrip() -> None:
    c = CompressedChunk(
        chunk_id="x", action="KEEP", applied="KEEP", text="t", tokens_before=1, tokens_after=1
    )
    assert not c.dropped and c.merged_from == []
