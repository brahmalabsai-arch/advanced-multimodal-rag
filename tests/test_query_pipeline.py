"""Phase 3 tests — RRF fusion, verifier, intent rules, context assembly, API contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.calc.calculator import CalcInput, CalculationResult
from rag.core.schema import Chunk, ChunkMetadata
from rag.core.settings import PROJECT_ROOT
from rag.query.analyze import guess_intent
from rag.query.assemble import AssembledContext, ContextBlock, assemble_context, block_body
from rag.query.generate import Answer, FigureUsed
from rag.query.retrieve import Candidate, rrf_fuse
from rag.query.verify import extract_numbers, number_is_supported, verify_answer

# ------------------------------------------------------------------------------- RRF


def test_rrf_fusion_on_synthetic_rankings() -> None:
    dense = ["a", "b", "c", "d"]
    bm25 = ["c", "a", "e"]
    scores = rrf_fuse([dense, bm25], k=60)
    assert set(scores) == {"a", "b", "c", "d", "e"}
    # a: 1/61 + 1/62 ; c: 1/63 + 1/61 ; both retrievers beat single-retriever hits
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["a"] > scores["c"] > scores["b"] > scores["e"] > scores["d"]


def test_rrf_uses_ranks_not_scores() -> None:
    assert rrf_fuse([["x"]], k=60)["x"] == pytest.approx(1 / 61)
    assert rrf_fuse([["x"], ["x"], ["x"]], k=60)["x"] == pytest.approx(3 / 61)


# ---------------------------------------------------------------------------- intent


@pytest.mark.parametrize(
    "question, intent",
    [
        ("What were NVIDIA's total assets as of Jan 25, 2026?", "POINT_LOOKUP"),
        ("What is the current ratio as of Jan 25, 2026?", "COMPUTATION"),
        ("How much did goodwill grow year-over-year?", "COMPARISON_TREND"),
        ("Why did gross margin decrease in fiscal 2026?", "EXPLANATORY"),
        ("How did the 5-year total return compare to the Nasdaq 100?", "VISUAL"),
        ("What are the layers in the five-layer cake?", "VISUAL"),
        ("Should I buy NVIDIA stock?", "OUT_OF_SCOPE"),
        ("What is the share price right now?", "OUT_OF_SCOPE"),
    ],
)
def test_intent_rules(question: str, intent: str) -> None:
    assert guess_intent(question)[0] == intent


# --------------------------------------------------------------------------- verifier


def _ctx(text: str, block_id: str = "C1", modality: str = "row_fact") -> AssembledContext:
    return AssembledContext(
        blocks=[
            ContextBlock(
                block_id=block_id,
                chunk_id="rowfact_p141_x",
                modality=modality,
                page=141,
                header=f"[{block_id} | Form 10-K › Balance Sheets | PDF p.141 | {modality}]",
                text=text,
                token_count=40,
            )
        ],
        token_budget=2500,
        tokens_used=40,
    )


def test_extract_numbers_ignores_citation_ids() -> None:
    assert extract_numbers("Total assets were $206,803 million [C1] in 2026, up 85.3%.") == [
        "$206,803",
        "2026",
        "85.3%",
    ]


def test_number_support_rules() -> None:
    allowed = {206803.0, 3.905, 85.31}
    assert number_is_supported("$206,803", allowed)
    assert number_is_supported("206.8", allowed)  # billion scale, rounded
    assert number_is_supported("3.91", allowed)  # rounded ratio
    assert number_is_supported("85.3%", allowed)
    assert number_is_supported("2026", allowed)  # year
    assert number_is_supported("5", allowed)  # small count ("5-year")
    assert not number_is_supported("$210,000", allowed)
    assert not number_is_supported("4.2", allowed)
    assert number_is_supported("5.4%", {0.054})  # ratio restated as a percentage
    assert number_is_supported("76.1%", {0.761})
    assert not number_is_supported("5.4", {0.054})  # only the percent form gets ×100
    assert not number_is_supported("6.1%", {0.054})


def test_verifier_passes_a_grounded_answer() -> None:
    ctx = _ctx(
        "NVIDIA Consolidated Balance Sheets — Inventories: $21,403 million as of Jan 25, 2026 (FY2026); $10,080 million as of Jan 26, 2025 (FY2025)."
    )
    ans = Answer(
        answer_markdown="Inventories were **$21,403 million** as of January 25, 2026 (fiscal 2026) [C1].",
        figures_used=[FigureUsed(value=21403, unit="USD_millions", period="FY2026", citation="C1")],
        citations=["C1"],
    )
    v = verify_answer(ans, ctx, [], "What were inventories in fiscal 2026?")
    assert (
        v.passed and v.numbers_checked == 4 and v.citations_found == ["C1"]
    )  # $21,403 · 25 · 2026 · 2026


def test_verifier_flags_injected_wrong_number() -> None:
    ctx = _ctx("Inventories: $21,403 million as of Jan 25, 2026 (FY2026).")
    ans = Answer(
        answer_markdown="Inventories were $24,103 million as of Jan 25, 2026 [C1].",
        citations=["C1"],
    )
    v = verify_answer(ans, ctx, [], "q")
    assert not v.passed
    assert "$24,103" in v.unmatched_numbers
    assert any("Numbers not found" in i for i in v.issues)


def test_verifier_flags_invalid_citation_and_missing_citation() -> None:
    ctx = _ctx("Inventories: $21,403 million as of Jan 25, 2026.")
    ans = Answer(answer_markdown="Inventories were $21,403 million [C7].", citations=["C7"])
    v = verify_answer(ans, ctx, [], "q")
    assert not v.passed and v.invalid_citations == ["C7"]
    uncited = Answer(answer_markdown="Inventories were $21,403 million.", citations=[])
    v2 = verify_answer(uncited, ctx, [], "q")
    assert not v2.passed and any("cites no context block" in i for i in v2.issues)


def test_verifier_accepts_calculator_results() -> None:
    ctx = _ctx("Total current assets 125,605; total current liabilities 32,163.")
    calc = CalculationResult(
        formula="current_ratio",
        description="d",
        kind="ratio",
        expression="a / b",
        fiscal_year=2026,
        inputs=[
            CalcInput(
                name="a", line_item="x", line_item_norm="x", value=125605.0, fiscal_year=2026
            ),
            CalcInput(name="b", line_item="y", line_item_norm="y", value=32163.0, fiscal_year=2026),
        ],
        result=3.9053,
        rounded=3.91,
        round_digits=2,
    )
    ctx.blocks.insert(
        0,
        ContextBlock(
            block_id="K1",
            modality="calculation",
            header="[K1 | CALCULATION]",
            text=calc.as_context_block("K1"),
            token_count=50,
        ),
    )
    ans = Answer(
        answer_markdown="The current ratio is **3.91** [K1] (125,605 / 32,163).", citations=["K1"]
    )
    assert verify_answer(ans, ctx, [calc], "current ratio?").passed


# --------------------------------------------------------------------------- assemble


class _FakeStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = {c.id: c for c in chunks}

    def get(self, cid: str) -> Chunk | None:
        return self.chunks.get(cid)

    def image_path(self, rel: str | None) -> Path | None:
        return None


def _chunk(cid: str, modality: str, tokens: int, page: int = 1) -> Chunk:
    body = "word " * max(1, tokens - 20)
    return Chunk(
        id=cid,
        document=f"[Form 10-K > Item 8 > Balance Sheets]\n{body}",
        metadata=ChunkMetadata(
            chunk_id=cid,
            modality=modality,
            section="form_10k",
            subsection="item_8_financials",
            page=page,
            breadcrumb="Form 10-K > Item 8 > Balance Sheets",
            token_count=tokens,
        ),
    )


def _cand(chunk: Chunk, source: str = "both") -> Candidate:
    m = chunk.metadata
    return Candidate(
        chunk_id=chunk.id,
        modality=m.modality,
        page=m.page,
        section=m.section,
        subsection=m.subsection,
        breadcrumb=m.breadcrumb,
        source=source,
        token_count=m.token_count,
    )


def test_assemble_orders_by_modality_and_respects_budget() -> None:
    chunks = [
        _chunk("t1", "text", 300),
        _chunk("r1", "row_fact", 60),
        _chunk("f1", "figure", 100),
        _chunk("tb1", "table", 200),
        _chunk("t2", "text", 500),
    ]
    store = _FakeStore(chunks)
    ctx = assemble_context(store, [_cand(c) for c in chunks], [], token_budget=700)
    order = [b.chunk_id for b in ctx.blocks]
    assert order[:2] == ["r1", "tb1"]  # row facts and tables first
    assert "t2" in ctx.dropped  # did not fit
    assert ctx.tokens_used <= 700
    assert [b.block_id for b in ctx.blocks] == [f"C{i + 1}" for i in range(len(ctx.blocks))]
    assert ctx.blocks[0].header == "[C1 | Form 10-K › Item 8 › Balance Sheets | PDF p.1 | row_fact]"


def test_assemble_visual_intent_leads_with_figure() -> None:
    chunks = [_chunk("t1", "text", 100), _chunk("f1", "figure", 100), _chunk("tb1", "table", 100)]
    ctx = assemble_context(
        _FakeStore(chunks), [_cand(c) for c in chunks], [], token_budget=2500, intent="VISUAL"
    )
    assert [b.chunk_id for b in ctx.blocks] == ["f1", "tb1", "t1"]


def test_block_body_strips_breadcrumb_line() -> None:
    assert block_body(_chunk("x", "text", 30)).startswith("word")


# --------------------------------------------------------------------------- API contract


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from rag.api import main as api_main

    monkeypatch.setattr(
        api_main, "_load_pipeline", lambda app: setattr(app.state, "pipeline", None)
    )
    with TestClient(api_main.app) as c:
        yield c


def test_healthz_and_readyz_before_load(client) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["ready"] is False


def test_ask_validation_and_not_ready(client) -> None:
    assert client.post("/api/ask", json={}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 2000}).status_code == 422
    r = client.post("/api/ask", json={"question": "What were total assets?"})
    assert r.status_code == 503


def test_figure_and_page_endpoints_validate_ids(client) -> None:
    assert client.get("/api/figures/..%2F..%2F.env").status_code == 404
    assert client.get("/api/figures/p999_9").status_code == 404
    assert client.get("/api/pages/0").status_code == 404
    if (PROJECT_ROOT / "data" / "index" / "pages" / "p141.webp").exists():
        r = client.get("/api/pages/141")
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp"


def test_frontend_is_served(client) -> None:
    """The page is mounted at the root, so its relative asset paths resolve, and the API and
    health routes registered before the mount still win (F2)."""
    r = client.get("/")
    assert r.status_code == 200 and "<title>" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200
    assert client.get("/vendor/purify.min.js").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/api/trace/nope").status_code in {404, 503}


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)
def test_ask_contract_with_loaded_pipeline_and_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full /api/ask contract through a real index and a scripted model (no network)."""
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage

    from rag.api import main as api_main
    from rag.graph import Pipeline

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(
                content='{"answer_markdown": "Total assets were $206,803 million as of January 25, 2026 (fiscal 2026) [C1].", '
                '"figures_used": [{"value": 206803, "unit": "USD_millions", "period": "FY2026 (Jan 25, 2026)", "citation": "C1"}], '
                '"citations": ["C1"], "confidence": "high", "answer_class": "filed_fact"}',
                usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

    from rag.core.settings import Settings
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)

    def load(app):
        app.state.pipeline = Pipeline(settings=settings, client=llm)

    monkeypatch.setattr(api_main, "_load_pipeline", load)
    with TestClient(api_main.app) as c:
        assert c.get("/readyz").json()["ready"] is True
        r = c.post(
            "/api/ask",
            json={
                "question": "What were NVIDIA's total assets as of Jan 25, 2026?",
                "bypass_cache": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer_markdown"].startswith("Total assets")
        assert body["citations"][0]["page"] == 141
        assert body["debug"]["citations"][0]["page_image_url"] == "/api/pages/141"
        assert body["debug"]["debug"]["verify"]["passed"] is True
        assert body["debug"]["debug"]["candidates"][0]["chunk_id"] == "rowfact_p141_total_assets"
        assert body["debug"]["cache_tier"] == "bypassed"
        t = c.get(f"/api/trace/{body['debug']['request_id']}")
        assert t.status_code == 200 and t.json()["intent"] == "POINT_LOOKUP"


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)
def test_out_of_scope_is_refused_without_retrieval_or_model_calls() -> None:
    """Plan Phase 4 exit criterion: G14 and paraphrases refused with zero retrieval calls."""
    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    class NeverCalled:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            raise AssertionError("model must not be called for an out-of-scope question")

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: NeverCalled(), pacing_enabled=False)
    pipeline = Pipeline(settings=settings, client=llm)
    for q in ("Should I buy NVIDIA stock?", "Is NVDA a good buy right now?"):
        r = pipeline.ask(q, bypass_cache=True)
        assert r.intent == "OUT_OF_SCOPE" and not r.scope.in_scope
        assert r.retrieval.candidates == [] and r.context.blocks == []
        assert r.tokens_by_model == {} and r.generator_role == "-"
        assert "annual report" in r.answer.answer_markdown
        assert "retrieve" not in r.latency_ms_by_node and "generate" not in r.latency_ms_by_node
    # a balance-sheet question is never refused
    r = pipeline._degraded(
        {"question": "What were total assets as of Jan 25, 2026?", "request_id": "x"}, "n/a"
    )
    assert r["scope"].in_scope and r["slots"].metrics == ["total assets"]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)
def test_debug_payload_exposes_slots_expansion_and_gate_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage

    from rag.api import main as api_main
    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(
                content='{"answer_markdown": "The current ratio is **3.91** [K1] (fiscal 2026, as of January 25, 2026).", '
                '"figures_used": [], "citations": ["K1"], "confidence": "high", "answer_class": "analytical"}',
                usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    monkeypatch.setattr(
        api_main,
        "_load_pipeline",
        lambda app: setattr(app.state, "pipeline", Pipeline(settings=settings, client=llm)),
    )
    with TestClient(api_main.app) as c:
        r = c.post(
            "/api/ask",
            json={
                "question": "What is the current ratio as of Jan 25, 2026?",
                "bypass_cache": True,
            },
        )
        assert r.status_code == 200, r.text
        d = r.json()["debug"]["debug"]
        assert d["slots"]["formulas"] == ["current_ratio"]
        assert d["slots"]["fiscal_periods"] == ["FY2026"]
        assert d["scope"]["in_scope"] is True
        assert d["analysis"]["intent"] == "COMPUTATION" and d["analysis"]["llm_called"] is False
        kinds = [q["kind"] for q in d["analysis"]["queries"]]
        assert kinds[0] == "original" and "decomposition" in kinds
        assert d["analysis"]["queries"][1]["where"]["$and"][0] == {"statement": "balance_sheet"}
        assert d["rerank"]["gate"] in {"S1", "S2", "S3", "disabled"} or d["rerank"]["applied"]
        assert d["calculations"][0]["formula"] == "current_ratio"
        assert d["calculations"][0]["rounded"] == 3.91
        assert d["tokens_by_model"] and sum(t["calls"] for t in d["tokens_by_model"].values()) == 1
        t = c.get(f"/api/trace/{r.json()['debug']['request_id']}").json()
        assert t["slots"]["formulas"] == ["current_ratio"] and "rerank_applied" in t


# ------------------------------------------------------------------- vision budgets (Phase 4)


def test_vision_budgets_follow_role_pacing(settings) -> None:
    from rag.llm import LLMClient

    client = LLMClient(settings, chat_factory=lambda cfg, role: None, pacing_enabled=False)
    assert client.max_output_tokens("vision") == 1000  # OTPM bucket, D-52
    assert client.max_input_tokens("vision", 1000) == 7000  # ITPM gate
    assert client.max_output_tokens("large") is None
    assert client.max_input_tokens("large", 1200) == 8000 - 1200


def test_fit_images_to_budget_sheds_trailing_images_only() -> None:
    from rag.query.generate import fit_images_to_budget

    prompt = "word " * 2500  # ≈ 2.5K tokens of context
    imgs = ["a.png", "b.png", "c.png"]
    assert fit_images_to_budget(imgs, prompt, None) == imgs
    assert fit_images_to_budget(imgs, prompt, 7000) == ["a.png"]  # 2 images would exceed 7K
    assert fit_images_to_budget(imgs, prompt, 20000) == imgs
    assert fit_images_to_budget(["a.png"], "x" * 40000, 7000) == ["a.png"]  # never drops the first


def test_image_data_url_is_bounded_to_vision_max_side(tmp_path: Path) -> None:
    import base64
    import io

    from PIL import Image

    from rag.llm import VISION_MAX_SIDE_PX, _image_to_data_url

    big = tmp_path / "big.png"
    Image.new("RGB", (3300, 1100), "white").save(big)
    url = _image_to_data_url(big)
    assert url.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
        assert max(im.size) == VISION_MAX_SIDE_PX
    small = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(small)
    assert _image_to_data_url(small).startswith("data:image/png;base64,")
