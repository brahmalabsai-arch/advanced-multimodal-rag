"""Phase 8 tests — robustness and security hardening (architecture §13, §14.3; NFR-12).

Covers: input limits, the request timeout, the loopback guard, startup checks, the in-graph
degrade path (retrieval-only view, never admitted), the outer fallback's latency table, the
vision-unavailable fallback and log hygiene for model-error messages.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rag.api import checks
from rag.api.routes_ask import validate_question
from rag.core.config import AppConfig, load_models_config
from rag.core.logging import REDACTED, configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, Settings
from rag.graph import degraded_answer
from rag.llm import LLMCallError
from rag.query.assemble import AssembledContext, ContextBlock
from rag.query.generate import Answer, generate_answer

HAS_INDEX = (PROJECT_ROOT / "data" / "index" / "manifest.json").exists()
needs_index = pytest.mark.skipif(not HAS_INDEX, reason="run `make ingest` first")


# ------------------------------------------------------------------------- input limits


@pytest.mark.parametrize(
    "question, detail",
    [
        ("   ", "empty"),
        ("x" * 1001, "longer than 1000"),
        ("what is\x00 total assets", "control characters"),
        ("\x01\x02\x03abc", "control characters"),
        ("???!!! ...", "no words"),
    ],
)
def test_validate_question_rejects_bad_input(question: str, detail: str) -> None:
    with pytest.raises(HTTPException) as exc:
        validate_question(question, 1000)
    assert exc.value.status_code == 422 and detail in exc.value.detail


def test_validate_question_keeps_ordinary_text() -> None:
    q = "  What were total assets\nas of Jan 25, 2026?  "
    assert validate_question(q, 1000) == q.strip()
    assert validate_question("Q4 2026?", 1000) == "Q4 2026?"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    """App with the pipeline swapped for a stub: `ask` returns a canned PipelineResult-like
    object slowly enough for the timeout test; startup checks run against the real tree."""
    from fastapi.testclient import TestClient

    from rag.api import main as api_main

    def load(app):
        app.state.pipeline = None

    monkeypatch.setattr(api_main, "_load_pipeline", load)
    with TestClient(api_main.app) as c:
        yield c, api_main.app


def test_api_rejects_binary_and_wordless_questions_before_readiness(api) -> None:
    client, _ = api
    r = client.post("/api/ask", json={"question": "abc\x00def"})
    assert r.status_code == 422 and "control" in r.json()["detail"]
    r = client.post("/api/ask", json={"question": "!!! ???"})
    assert r.status_code == 422 and "no words" in r.json()["detail"]
    # over the configured limit but under the wire ceiling → the configured limit wins
    r = client.post("/api/ask", json={"question": "a" * 1500})
    assert r.status_code == 422 and "longer than" in r.json()["detail"]
    # over the wire ceiling → pydantic
    assert client.post("/api/ask", json={"question": "a" * 30000}).status_code == 422


def test_request_timeout_answers_504(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, app = api

    def slow_ask(question, *, bypass_cache=False, client=None):
        time.sleep(0.5)
        raise AssertionError("the response must not wait for this")

    app.state.pipeline = SimpleNamespace(ask=slow_ask, store=None)
    monkeypatch.setattr(app.state.app_config.server, "request_timeout_seconds", 0.05)
    r = client.post("/api/ask", json={"question": "What were total assets?"})
    assert r.status_code == 504 and "exceeded 0.05s" in r.json()["detail"]


def test_readyz_reports_startup_checks_and_limits(api) -> None:
    client, app = api
    r = client.get("/readyz")
    body = r.json()
    names = [c["name"] for c in body["checks"]]
    assert names == ["bind", "public", "admin", "secrets", "index"]
    bind = body["checks"][0]
    assert bind["ok"] is True and bind["fatal"] is True
    # the pipeline is stubbed out, so readiness is false whatever the checks say
    assert r.status_code == 503 and body["ready"] is False


# ------------------------------------------------------------------------ loopback guard


@pytest.mark.parametrize(
    "host, ok",
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("127.0.0.53", True),
        ("localhost", True),
        ("testclient", True),
        (None, True),
        ("10.0.0.5", False),
        ("192.168.1.20", False),
        ("2001:db8::1", False),
    ],
)
def test_is_loopback_client(host: str | None, ok: bool) -> None:
    assert checks.is_loopback_client(host) is ok


def test_non_loopback_peer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from rag.api import main as api_main

    monkeypatch.setattr(
        api_main, "_load_pipeline", lambda app: setattr(app.state, "pipeline", None)
    )
    with TestClient(api_main.app, client=("192.168.1.20", 40000)) as c:
        r = c.get("/healthz")
        assert r.status_code == 403 and "localhost only" in r.json()["detail"]
        assert c.post("/api/ask", json={"question": "total assets?"}).status_code == 403
    with TestClient(api_main.app, client=("127.0.0.1", 40000)) as c:
        assert c.get("/healthz").status_code == 200


# ----------------------------------------------------------------------- startup checks


def test_check_bind_accepts_loopback_only() -> None:
    assert checks.check_bind("127.0.0.1").ok
    assert checks.check_bind("localhost").ok
    r = checks.check_bind("0.0.0.0")
    assert not r.ok and r.fatal and "localhost-only" in r.detail
    assert not checks.check_bind("192.168.1.5").ok


def test_check_admin_gating_flags_admin_outside_dev() -> None:
    dev = AppConfig.model_validate({"env": "dev"})
    assert checks.check_admin_gating(dev).ok
    prod = AppConfig.model_validate({"env": "prod"})
    r_prod = checks.check_admin_gating(prod)
    assert r_prod.ok and "disabled" in r_prod.detail
    leaky = AppConfig.model_validate({"env": "prod", "dev_clock": {"enabled_in": ["dev", "prod"]}})
    r = checks.check_admin_gating(leaky)
    assert not r.ok and "ENABLED" in r.detail and not r.fatal


def test_check_secrets_names_missing_keys_without_values(settings: Settings) -> None:
    models = load_models_config(settings=settings)
    ok = checks.check_secrets(settings, models)
    assert ok.ok and "all provider keys present" in ok.detail
    bare = Settings(_env_file=None, app_env="test", config_dir=settings.config_dir)
    r = checks.check_secrets(bare, models)
    assert not r.ok and "GROQ_API_KEY" in r.detail
    # anthropic template: no key configured → named, never printed
    anth = Settings(
        _env_file=None,
        app_env="test",
        model_profile="anthropic",
        anthropic_api_key="sk-ant-secret-value-000",
        config_dir=settings.config_dir,
    )
    r = checks.check_secrets(anth, load_models_config(settings=anth))
    assert r.ok and "secret-value" not in r.detail


def test_check_secrets_passes_a_keyless_byok_deploy(settings: Settings) -> None:
    """The deployed image carries no provider key by design; the check must not fail it (the
    first CI deploy check did, on exactly this). A key left in that environment is named."""
    models = load_models_config(settings=settings)
    keyless = Settings(
        _env_file=None, app_env="prod", byok_only=True, config_dir=settings.config_dir
    )
    r = checks.check_secrets(keyless, models)
    assert r.ok and "no server keys needed" in r.detail and "never used" not in r.detail
    stray = Settings(
        _env_file=None,
        app_env="prod",
        byok_only=True,
        groq_api_key="gsk_left_behind_0123456789",
        config_dir=settings.config_dir,
    )
    r = checks.check_secrets(stray, models)
    assert r.ok and "GROQ_API_KEY set but never used" in r.detail
    assert "left_behind" not in r.detail


def _write_index(tmp: Path, *, chunks: int = 3, tamper: str | None = None) -> Path:
    from rag.ingest.index import compute_corpus_version

    cfg = {"chunking": {"version": "semantic-v1"}, "embedder": "BAAI/bge-small-en-v1.5"}
    sha = "ab" * 32
    manifest = {
        "corpus_version": compute_corpus_version(sha, cfg),
        "created_at": "2026-09-19T00:00:00+00:00",
        "pdf_sha256": sha,
        "ingestion_config": cfg,
        "ingestion_config_hash": "x",
        "embedder": "BAAI/bge-small-en-v1.5",
        "embedder_alias": "bge-small",
        "embedding_dim": 384,
        "tokenizer_version": "finance-v1",
        "chunk_counts": {"text": chunks},
        "chunks_total": chunks,
        "sentences_total": 10,
    }
    if tamper == "version":
        manifest["corpus_version"] = "deadbeef0000"
    if tamper == "count":
        manifest["chunks_total"] = chunks + 1
        manifest["chunk_counts"] = {"text": chunks + 1}
    if tamper == "embedder":
        manifest["embedder_alias"] = "bge-base"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp / "chunks.jsonl").write_text("".join('{"id": "c"}\n' for _ in range(chunks)), "utf-8")
    for name in ("bm25.pkl", "sentences.jsonl", "sentence_emb.npy"):
        (tmp / name).write_bytes(b"")
    return tmp


def test_check_index_missing_files_is_fatal(tmp_path: Path) -> None:
    r = checks.check_index(tmp_path / "index")
    assert not r.ok and r.fatal and "manifest.json" in r.detail and "make ingest" in r.detail


def test_check_index_consistent_manifest_passes(tmp_path: Path) -> None:
    r = checks.check_index(_write_index(tmp_path / "index"), "bge-small")
    assert r.ok, r.detail
    assert "chunks=3" in r.detail


@pytest.mark.parametrize(
    "tamper, phrase",
    [
        ("version", "does not recompute"),
        ("count", "chunks.jsonl has 3 rows"),
        ("embedder", "thresholds.retrieval.embedder"),
    ],
)
def test_check_index_detects_inconsistency(tmp_path: Path, tamper: str, phrase: str) -> None:
    r = checks.check_index(_write_index(tmp_path / "index", tamper=tamper), "bge-small")
    assert not r.ok and r.fatal and phrase in r.detail


def test_check_index_invalid_manifest(tmp_path: Path) -> None:
    d = _write_index(tmp_path / "index")
    (d / "manifest.json").write_text("{not json", encoding="utf-8")
    r = checks.check_index(d)
    assert not r.ok and "invalid" in r.detail


@needs_index
def test_run_startup_checks_on_the_real_tree(settings: Settings) -> None:
    from rag.core.config import load_app_config, load_thresholds_config

    real = settings.model_copy(update={"data_dir": PROJECT_ROOT / "data"})
    results = checks.run_startup_checks(
        real,
        load_app_config(settings=real),
        load_models_config(settings=real),
        load_thresholds_config(settings=real),
    )
    assert [r.name for r in results] == ["bind", "public", "admin", "secrets", "index"]
    assert all(r.ok for r in results), [r for r in results if not r.ok]
    assert checks.fatal_failures(results) == []


def test_fatal_check_blocks_pipeline_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from rag.api import main as api_main

    monkeypatch.setattr(
        api_main,
        "run_startup_checks",
        lambda *a, **k: [checks.check_bind("0.0.0.0"), checks.check_admin_gating(AppConfig())],
    )

    def must_not_load(app):
        raise AssertionError("pipeline must not load after a fatal check")

    monkeypatch.setattr(api_main, "_load_pipeline", must_not_load)
    with TestClient(api_main.app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False and "bind" in body["error"]
        assert body["checks"][0]["ok"] is False
        assert c.post("/api/ask", json={"question": "total assets?"}).status_code == 503


# ------------------------------------------------------------------------- degrade mode


def _block(bid: str, text: str, page: int = 141, modality: str = "table") -> ContextBlock:
    return ContextBlock(
        block_id=bid,
        chunk_id=f"chunk_{bid}",
        modality=modality,
        page=page,
        section="form_10k",
        breadcrumb="Form 10-K > Item 8 > Balance Sheets",
        header=f"[{bid} | Form 10-K › Item 8 | PDF p.{page} | {modality}]",
        text=text,
        token_count=40,
    )


def test_degraded_answer_is_a_retrieval_only_view_with_citations() -> None:
    ctx = AssembledContext(
        blocks=[
            _block("C1", "Total assets 206,803 " + "filler " * 80),
            _block("C2", "Total liabilities 47,001", page=142),
            _block("K1", "current_ratio = 3.91", modality="calculation"),
        ],
        token_budget=2500,
        tokens_used=120,
    )
    a = degraded_answer(ctx, LLMCallError("boom", status_code=429))
    assert a.confidence == "low" and a.answer_class == "analytical"
    assert "rate-limiting" in a.answer_markdown and "HTTP 429" in a.answer_markdown
    assert "**[C1]**" in a.answer_markdown and "PDF p.141" in a.answer_markdown
    assert "**[K1]**" in a.answer_markdown and "Deterministic calculations" in a.answer_markdown
    assert a.citations == ["C1", "C2", "K1"]
    # long excerpts are cut on a word boundary with an ellipsis; nothing is generated
    line = next(ln for ln in a.answer_markdown.splitlines() if "[C1]" in ln)
    assert line.endswith("…") and "206,803" in line
    b = degraded_answer(ctx, LLMCallError("provider 400", status_code=400))
    assert "rate-limiting" not in b.answer_markdown and "model call failed" in b.answer_markdown


class _Raise:
    """Chat model whose every call fails with a provider-style 429."""

    def __init__(self):
        self.calls = 0

    def bind(self, **kw):
        return self

    def invoke(self, messages):
        self.calls += 1
        exc = RuntimeError("Rate limit reached for model; try again in 19m")
        exc.status_code = 429
        raise exc


@needs_index
def test_generation_failure_degrades_inside_the_graph(tmp_path: Path) -> None:
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    chat = _Raise()
    llm = LLMClient(
        settings,
        chat_factory=lambda cfg, role: chat,
        pacing_enabled=False,
        max_attempts=2,
        sleep=lambda s: None,
    )
    pipeline = Pipeline(settings=settings, client=llm, cache_enabled=False)
    rid = f"degrade-{uuid.uuid4().hex[:8]}"
    r = pipeline.ask("What were total assets as of Jan 25, 2026?", request_id=rid)
    assert r.degraded is True and r.generator_role == "-"
    assert r.answer.confidence == "low" and "rate-limiting" in r.answer.answer_markdown
    assert r.answer.citations and r.answer.citations[0] == "C1"
    assert not r.verify.passed and "model call failed" in r.verify.issues[0]
    assert r.warning and "HTTP 429" in r.warning and r.warning.startswith("Degraded answer")
    # every node that ran kept its latency; verify was skipped
    for node in ("slots", "cache_lookup", "scope", "analyze", "retrieve", "assemble", "generate"):
        assert node in r.latency_ms_by_node, node
    assert "verify" not in r.latency_ms_by_node
    # one generation; the "try again in 19m" hint exceeds MAX_RETRY_AFTER_SECONDS, so the
    # client fails fast instead of spending its second attempt (D-52)
    assert r.generation_attempts == 1 and chat.calls == 1
    # the trace line carries the flag and the error
    t = pipeline.trace_writer.find(rid)
    assert t is not None and t.degraded is True and t.error and "429" in t.error


@needs_index
def test_degraded_answer_is_never_admitted_to_the_cache(tmp_path: Path) -> None:
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=tmp_path / "data",
    )
    # the index lives in the real tree; logs and the cache go to tmp
    from rag.query.store import IndexStore

    index_dir = PROJECT_ROOT / "data" / "index"
    store = IndexStore(index_dir, figures_root=index_dir)
    llm = LLMClient(
        settings,
        chat_factory=lambda cfg, role: _Raise(),
        pacing_enabled=False,
        max_attempts=1,
        sleep=lambda s: None,
    )
    pipeline = Pipeline(settings=settings, client=llm, store=store, cache_enabled=True)
    r = pipeline.ask("What were total assets as of Jan 25, 2026?")
    assert r.degraded and r.cache.tier == "MISS"
    assert r.cache.write is not None and r.cache.write.admitted is False
    assert "degraded" in r.cache.write.reasons[0]
    assert "cache_write" in r.latency_ms_by_node
    # a second ask is a miss again — nothing was stored
    r2 = pipeline.ask("What were total assets as of Jan 25, 2026?")
    assert r2.cache.tier == "MISS" and r2.degraded


@needs_index
def test_outer_fallback_times_the_nodes_it_reruns() -> None:
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: _Raise(), pacing_enabled=False)
    pipeline = Pipeline(settings=settings, client=llm, cache_enabled=False)
    st = pipeline._degraded(
        {"question": "What were total assets as of Jan 25, 2026?", "request_id": "x"},
        LLMCallError("analysis call failed", status_code=503),
    )
    assert st["degraded"] is True and st["error"] == "analysis call failed"
    for node in ("slots", "analyze", "retrieve", "assemble"):
        assert node in st["latency_ms_by_node"], node
    assert st["scope"].in_scope and st["slots"].metrics == ["total assets"]
    assert st["answer"].citations[0] == "C1"


@needs_index
def test_degraded_response_over_http_renders_citation_chips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from rag.api import main as api_main
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(
        settings,
        chat_factory=lambda cfg, role: _Raise(),
        pacing_enabled=False,
        max_attempts=1,
        sleep=lambda s: None,
    )
    monkeypatch.setattr(
        api_main,
        "_load_pipeline",
        lambda app: setattr(app.state, "pipeline", Pipeline(settings=settings, client=llm)),
    )
    with TestClient(api_main.app) as c:
        r = c.post(
            "/api/ask",
            json={"question": "What were total assets as of Jan 25, 2026?", "bypass_cache": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["degraded"] is True and body["debug"]["warning"]
        assert body["confidence"] == "low"
        assert body["citations"] and body["citations"][0]["page"] == 141
        assert body["debug"]["debug"]["verify"]["passed"] is False
        assert "generate" in body["debug"]["debug"]["latency_ms_by_node"]


# ---------------------------------------------------------------- vision-unavailable path


def test_vision_unavailable_falls_back_to_text_model_without_images() -> None:
    """D-55: a VISUAL question whose vision call fails is answered by the large text model
    from the figure description + companion table, with the images stripped from the prompt."""
    seen: dict[str, object] = {}

    class FakeClient:
        def max_output_tokens(self, role):
            return 1000 if role == "vision" else None

        def max_input_tokens(self, role, output_budget=0):
            return 7000

        def vision_json(self, prompt, images, schema, **kw):
            seen["vision_images"] = list(images)
            raise LLMCallError("vision role failed after 1 attempt(s)", status_code=429)

        def json(self, prompt, schema, **kw):
            seen["text_prompt"] = prompt
            seen["text_kwargs"] = kw
            return Answer(answer_markdown="Revenue grew ~114% [C1].", citations=["C1"])

    ctx = AssembledContext(
        blocks=[
            _block(
                "C1", "Figure: revenue by year. Companion table: FY2026 $130.5B.", modality="figure"
            )
        ],
        images=["data:image/png;base64,AAAA"],
        image_chunk_ids=["fig_p10_1"],
        token_budget=2500,
        tokens_used=60,
    )
    answer, role = generate_answer(
        FakeClient(), "How did revenue grow per the chart?", ctx, intent="VISUAL", request_id="v"
    )
    assert role == "large" and answer.citations == ["C1"]
    assert seen["vision_images"] == ["data:image/png;base64,AAAA"]
    assert "ATTACHED IMAGES" not in seen["text_prompt"] and "Companion table" in seen["text_prompt"]
    assert seen["text_kwargs"]["role"] == "large"


# -------------------------------------------------------------------------- log hygiene


KEY = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"


def test_model_error_messages_are_redacted_when_logged() -> None:
    """A provider exception can echo request headers; the pipeline logs the error string, so
    the redactor must catch a key inside it — by registered value and by shape."""
    stream = io.StringIO()
    configure_logging("DEBUG", secrets=[KEY], stream=stream)
    log = get_logger("test.hardening")
    err = LLMCallError(f"role=large failed: 401 Unauthorized (Authorization: Bearer {KEY})", 401)
    log.error("Pipeline failed for %s: %s", "req-1", str(err))
    log.error("degraded: %s", LLMCallError("bad key sk-ant-api03-ZZZZZZZZZZZZZZZZZZZZ", 401))
    out = stream.getvalue()
    assert KEY not in out and "sk-ant-api03-ZZZZ" not in out
    assert out.count(REDACTED) >= 2 and "401" in out


def test_usage_ledger_and_trace_carry_no_prompt_text() -> None:
    """Neither log stores prompts or completions (§9.3–9.4): only counts, ids and status."""
    from rag.core.ledger import UsageRecord
    from rag.core.traces import Trace

    forbidden = {"prompt", "messages", "completion", "content", "system", "answer_markdown"}
    assert not forbidden & set(UsageRecord.model_fields)
    assert not forbidden & set(Trace.model_fields)


def test_secrets_stay_out_of_git() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and "!.env.example" in ignore
    assert "data/**" in ignore
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            assert value.strip() in {"", "groq_build", "dev"}, f"{key} has a value in .env.example"


# ------------------------------------------------------------- console encoding (rehearsal)


def test_utf8_console_makes_cp1252_stdout_print_model_answers(monkeypatch) -> None:
    """The fresh-clone rehearsal crashed `ask_cli.py` and `cache_walkthrough.py` on a cp1252
    console: the answer contained U+202F. `utf8_console()` must make that print succeed."""
    import io

    from rag.core.console import utf8_console

    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr("sys.stdout", fake)
    with pytest.raises(UnicodeEncodeError):
        print("$206,803 million")
    utf8_console()
    print("$206,803 million — ok")
    fake.flush()
    assert " ".encode() in raw.getvalue()


def test_every_cli_script_switches_the_console_to_utf8() -> None:
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.py")) + sorted(
        (PROJECT_ROOT / "eval").glob("*.py")
    )
    missing = [
        p.name
        for p in scripts
        if "def main(" in p.read_text("utf-8") and "utf8_console()" not in p.read_text("utf-8")
    ]
    assert scripts and not missing, missing
