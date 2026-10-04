"""F2 tests — bring-your-own-key and the published page's payload.

The promises this file holds to account: the server never answers from its own key when the
caller brings one, the key never reaches a log line, a trace or the cache, one provider's cached
answers are never served to another's request, and the page's payload keeps its shape when the
pipeline degrades or the cache answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from rag.api import byok
from rag.api.payload import DISCLAIMER, _figure_caption, _figures, headline_from
from rag.cache.versions import generator_model
from rag.core.config import load_models_config
from rag.core.logging import redact, use_key
from rag.core.settings import PROJECT_ROOT, Settings
from rag.llm import PROVIDERS, LLMCallError, LLMClient, UnknownProviderError, profile_for_provider

HAS_INDEX = (PROJECT_ROOT / "data" / "index" / "manifest.json").exists()
needs_index = pytest.mark.skipif(not HAS_INDEX, reason="run `make ingest` first")

GROQ_KEY = "gsk_visitorkey0123456789abcdefghij"
ANTHROPIC_KEY = "sk-ant-api03-visitorkey0123456789"


# ----------------------------------------------------------------- per-request credentials


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_for_request_uses_only_the_callers_key(provider: str, settings: Settings) -> None:
    """The env's keys are blanked in the per-request settings, so a caller cannot fall back
    onto the server's quota, and the profile follows the provider they asked for."""
    env = settings.model_copy(update={"groq_api_key": "gsk_serverkey0123456789abcdef"})
    client = LLMClient.for_request(provider, GROQ_KEY, settings=env, pacing_enabled=False)
    models = load_models_config(settings=env)
    assert client.profile_name == profile_for_provider(provider, models)
    assert client.settings.api_key_for(PROVIDERS[provider]) == GROQ_KEY
    other_keys = (
        {f for p, f in byok.PROVIDER_KEY_FIELDS.items() if p != PROVIDERS[provider]}
        if hasattr(byok, "PROVIDER_KEY_FIELDS")
        else set()
    )
    for field in other_keys:
        assert getattr(client.settings, field) is None


def test_for_request_rejects_an_unknown_provider(settings: Settings) -> None:
    with pytest.raises(UnknownProviderError, match="openrouter"):
        LLMClient.for_request("openrouter", GROQ_KEY, settings=settings)


def test_two_requests_never_share_a_client(settings: Settings) -> None:
    a = LLMClient.for_request("groq", GROQ_KEY, settings=settings, pacing_enabled=False)
    b = LLMClient.for_request("groq", "gsk_someoneelse0123456789abcdef", settings=settings)
    assert a is not b
    assert a.settings.api_key_for("groq") != b.settings.api_key_for("groq")


@pytest.mark.parametrize(
    "headers, status, phrase",
    [
        ({}, 401, "Add your model key"),
        ({"X-Provider": "groq"}, 401, "too short"),
        ({"X-Provider": "openrouter", "X-Provider-Key": GROQ_KEY}, 400, "Unknown provider"),
        ({"X-Provider": "groq", "X-Provider-Key": "short"}, 401, "too short"),
        ({"X-Provider": "groq", "X-Provider-Key": "gsk_with space in the key"}, 401, "spaces"),
        ({"X-Provider": "groq", "X-Provider-Key": "gsk_" + "x" * 600}, 401, "longer"),
        ({"X-Provider": "groq", "X-Provider-Key": "gsk_semi;colon;key;00000"}, 401, "characters"),
    ],
)
def test_credentials_are_validated_before_any_network_call(
    headers: dict[str, str], status: int, phrase: str
) -> None:
    class FakeRequest:
        def __init__(self, h):
            self.headers = h

    with pytest.raises(HTTPException) as exc:
        byok.credentials_from(FakeRequest(headers))
    assert exc.value.status_code == status and phrase in exc.value.detail


def test_credentials_accept_a_valid_pair() -> None:
    class FakeRequest:
        headers = {"X-Provider": "  GROQ ", "X-Provider-Key": f"  {GROQ_KEY}  "}

    creds = byok.credentials_from(FakeRequest())
    assert creds.provider == "groq" and creds.key == GROQ_KEY


# ------------------------------------------------------------------------------ redaction


def test_the_key_never_reaches_a_trace_or_the_ledger(tmp_path: Path) -> None:
    """The two writers that touch disk per request redact before writing (F2)."""
    from rag.core.ledger import UsageLedger, UsageRecord
    from rag.core.traces import Trace, TraceWriter

    key = "some-unbranded-key-0123456789"
    ledger = UsageLedger(tmp_path / "llm_usage.jsonl")
    traces = TraceWriter(tmp_path / "traces.jsonl")
    with use_key(key):
        ledger.append(
            UsageRecord(
                role="large",
                provider="groq",
                model="m",
                status="error",
                error=f"401 Unauthorized (Authorization: Bearer {key})",
            )
        )
        traces.append(
            Trace(
                request_id="r1",
                model_profile="groq_build",
                query="what were total assets?",
                intent="POINT_LOOKUP",
                error=f"provider said: bad key {key}",
            )
        )
    for path in (ledger.path, traces.path):
        text = path.read_text(encoding="utf-8")
        assert key not in text, path.name
        assert "***REDACTED***" in text
    # the question itself is still recorded — only the credential is masked
    assert "total assets" in traces.path.read_text(encoding="utf-8")


def test_redaction_falls_back_to_key_shapes_outside_a_request() -> None:
    assert GROQ_KEY not in redact(f"stray {GROQ_KEY}")
    assert ANTHROPIC_KEY not in redact({"detail": ANTHROPIC_KEY})


def test_provider_errors_are_mapped_without_leaking_the_key() -> None:
    with use_key(GROQ_KEY):
        err = byok.http_error(LLMCallError(f"401 (key={GROQ_KEY})", status_code=401))
        assert err.status_code == 401 and GROQ_KEY not in err.detail
        assert byok.http_error(LLMCallError("no credit", status_code=402)).status_code == 402
        assert byok.http_error(LLMCallError("slow down", status_code=429)).status_code == 429
        assert byok.http_error(LLMCallError("boom", status_code=500)).status_code == 502
        assert byok.http_error(UnknownProviderError("nope")).status_code == 400


# ------------------------------------------------------------------- one cache per provider


def test_generator_model_key_separates_providers() -> None:
    models = load_models_config()
    keys = {
        p: generator_model(
            models.model_copy(update={"active_profile": profile_for_provider(p, models)})
        )
        for p in PROVIDERS
    }
    assert len(set(keys.values())) == len(keys), keys
    assert all(k.startswith(("groq:", "anthropic:", "google:")) for k in keys.values())


@needs_index
def test_registry_shares_one_index_across_providers(settings: Settings) -> None:
    from rag.api.pipelines import PipelineRegistry

    real = settings.model_copy(update={"data_dir": PROJECT_ROOT / "data"})
    registry = PipelineRegistry(settings=real)
    groq = registry.for_provider("groq")
    anthropic = registry.for_provider("anthropic")
    assert groq is not anthropic
    assert groq.store is anthropic.store  # the memory that matters is loaded once
    assert registry.for_provider("groq") is groq  # cached
    assert groq.versions.generator_model != anthropic.versions.generator_model
    assert anthropic.client is None  # no key in .env for it, so BYOK-only
    assert anthropic.context_budget == 6000 and groq.context_budget == 2500


# --------------------------------------------------------------------------- page payload


def _result_from(path: Path):
    """A PipelineResult rebuilt from a recorded debug payload, so the adapter is tested against
    a real pipeline run rather than a hand-made object."""
    from rag.graph import PipelineResult

    return PipelineResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


@needs_index
def test_payload_has_the_shape_the_page_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract in `frontend/README.md`: answer_markdown required, everything else optional
    but correctly shaped when present."""
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage

    from rag.api import main as api_main
    from rag.api.pipelines import PipelineRegistry
    from rag.graph import Pipeline

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(
                content='{"answer_markdown": "The current ratio is **3.91** [K1].", '
                '"figures_used": [{"value": 3.91, "unit": "ratio", "period": "FY2026", '
                '"citation": "K1"}], "citations": ["K1", "C1"], "confidence": "high", '
                '"answer_class": "filed_fact"}',
                usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
            )

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="dev",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)

    def load(app):
        app.state.registry = PipelineRegistry(settings=settings)
        app.state.pipeline = Pipeline(settings=settings, client=llm, store=app.state.registry.store)
        app.state.registry._pipelines["groq"] = app.state.pipeline

    monkeypatch.setattr(api_main, "_load_pipeline", load)
    # the caller's key would build a real client; hand the BYOK path the scripted one instead,
    # so this test exercises the header route without a network call
    monkeypatch.setattr(byok.Credentials, "client", lambda self, **kw: llm)
    with TestClient(api_main.app) as c:
        r = c.post(
            "/api/ask",
            json={
                "question": "What is the current ratio as of Jan 25, 2026?",
                "bypass_cache": True,
            },
            headers={"X-Provider": "groq", "X-Provider-Key": GROQ_KEY},
        )
        assert r.status_code == 200, r.text
        body = r.json()

    assert body["answer_markdown"].startswith("The current ratio")
    assert body["confidence"] == "high" and body["degraded"] is False
    assert body["disclaimer"] == DISCLAIMER
    head = body["headline"]
    assert head["value"] == "3.91" and "FY2026" in head["caption"]
    assert "125,605 / 32,163" in head["working"]
    assert body["citations"] and all({"label", "page"} <= set(c) for c in body["citations"])
    assert all({"label", "value", "unit"} <= set(f) for f in body["figures_used"])
    trace = body["trace"]
    assert trace["cache_tier"] == "miss" and trace["verified"] is True
    assert trace["slots"]["formula"] == "current_ratio"
    labels = [s["label"] for s in trace["steps"]]
    assert "Read the question" in labels and "Verified every number" in labels
    assert all({"label", "kind", "seconds"} <= set(s) for s in trace["steps"])
    # the numbers behind the "how the agent answered" panel
    metrics = trace["metrics"]
    assert metrics["cache"]["tier"] == "bypassed" and metrics["cache"]["written"] is False
    assert metrics["context_tokens"] > 0
    passages = metrics["retrieval"]["passages"]
    assert passages and all({"label", "page", "similarity", "rrf"} <= set(p) for p in passages)
    assert metrics["retrieval"]["top_rrf"] == max(p["rrf"] for p in passages)
    # inline markers resolve: the calculation the answer cites points at a page it was read from
    refs = {r for c in body["citations"] for r in c["refs"]}
    assert "K1" in refs
    # the key is nowhere in the response
    assert GROQ_KEY not in json.dumps(body)


def test_headline_is_omitted_when_no_calculation_succeeded() -> None:
    from rag.calc.calculator import CalculationResult

    failed = CalculationResult(
        formula="current_ratio",
        description="Current assets divided by current liabilities",
        kind="ratio",
        expression="a / b",
        fiscal_year=2026,
        status="missing_inputs",
        missing=["total current assets (FY2026)"],
    )
    assert headline_from([failed]) is None
    assert headline_from([]) is None


@needs_index
def test_byok_only_refuses_a_request_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from rag.api import main as api_main

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="prod",
        byok_only=True,
        data_dir=PROJECT_ROOT / "data",
    )
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api_main, "_load_pipeline", lambda app: setattr(app.state, "pipeline", None)
    )
    with TestClient(api_main.app) as c:
        r = c.post("/api/ask", json={"question": "What were total assets?"})
        assert r.status_code == 401 and "key" in r.json()["detail"].lower()
        # admin routes are gone in prod, whatever the request
        assert c.get("/api/admin/cache/stats").status_code == 404
        checks = {c["name"]: c for c in c.get("/readyz").json()["checks"]}
        assert checks["public"]["ok"] is True and checks["admin"]["ok"] is True


# ------------------------------------------------------------------- public deployment guard


def test_public_deploy_requires_byok_and_no_admin() -> None:
    """The interlock: opening the server to the internet while it still holds a key, or still
    serves admin routes, is a fatal startup failure rather than a warning (F2)."""
    from rag.api.checks import check_public_safety
    from rag.core.config import AppConfig

    prod = AppConfig.model_validate({"env": "prod"})
    dev_cfg = AppConfig.model_validate({"env": "dev"})
    local = Settings(_env_file=None, app_env="dev")
    assert check_public_safety(local, dev_cfg).ok  # localhost build: nothing to enforce

    good = Settings(_env_file=None, app_env="prod", byok_only=True, public_deploy=True)
    assert check_public_safety(good, prod).ok

    leaky = Settings(_env_file=None, app_env="prod", byok_only=False, public_deploy=True)
    bad = check_public_safety(leaky, prod)
    assert not bad.ok and bad.fatal and "BYOK_ONLY" in bad.detail

    admin_open = Settings(_env_file=None, app_env="dev", byok_only=True, public_deploy=True)
    bad = check_public_safety(admin_open, dev_cfg)
    assert not bad.ok and "admin routes enabled" in bad.detail


def test_public_deploy_allows_a_non_loopback_bind_and_visitors() -> None:
    """Behind a platform proxy every visitor arrives with a routable address; the localhost
    guard must be off there, and on everywhere else."""
    from fastapi.testclient import TestClient

    from rag.api import main as api_main
    from rag.api.checks import check_bind

    assert not check_bind("0.0.0.0").ok
    assert check_bind("0.0.0.0", public=True).ok

    with TestClient(api_main.app, client=("10.1.2.3", 5000)) as c:
        api_main.app.state.public_deploy = True
        assert c.get("/healthz").status_code == 200
        api_main.app.state.public_deploy = False
        assert c.get("/healthz").status_code == 403


# --------------------------------------------------------------------------- figures in answers


def test_figure_caption_is_the_parsers_own_title() -> None:
    from types import SimpleNamespace

    chunk = SimpleNamespace(
        document="[Annual Review > Narrative]\n"
        "Figure p3_0 (diagram, PDF p. 3): AI Is a Five-Layer Cake\n"
        "A conceptual diagram."
    )
    assert _figure_caption(chunk, "fallback") == "AI Is a Five-Layer Cake"
    assert _figure_caption(SimpleNamespace(document="no title here"), "fallback") == "fallback"
    assert _figure_caption(None, "fallback") == "fallback"


def test_only_the_strongest_tier_of_figures_is_shown(tmp_path: Path) -> None:
    """A cited figure beats one that merely shares a cited page, which beats one only attached
    to the model; tiers are never mixed, and a crop missing on disk is never offered."""
    from types import SimpleNamespace

    (tmp_path / "figures").mkdir()
    for fid in ("p3_0", "p8_0", "p13_0"):
        (tmp_path / "figures" / f"{fid}.png").write_bytes(b"png")

    def block(bid: str, modality: str, page: int, chunk_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            block_id=bid, modality=modality, page=page, chunk_id=chunk_id, breadcrumb="A > B"
        )

    def fig_chunk(fid: str) -> SimpleNamespace:
        return SimpleNamespace(
            document=f"Figure {fid} (diagram, PDF p. 1): Title {fid}",
            metadata=SimpleNamespace(image_path=f"figures/{fid}.png"),
        )

    chunks = {
        "fig_p3": fig_chunk("p3_0"),
        "fig_p8": fig_chunk("p8_0"),
        "fig_p13": fig_chunk("p13_0"),
        "fig_gone": fig_chunk("p99_0"),
    }
    store = SimpleNamespace(
        get=chunks.get,
        image_path=lambda rel: (tmp_path / rel) if (tmp_path / rel).exists() else None,
    )
    blocks = [
        block("C1", "text", 13, "text_p13"),
        block("C2", "figure", 3, "fig_p3"),
        block("C3", "figure", 8, "fig_p8"),
        block("C4", "figure", 13, "fig_p13"),
        block("C5", "figure", 13, "fig_gone"),
    ]
    result = SimpleNamespace(context=SimpleNamespace(blocks=blocks, image_chunk_ids=["fig_p8"]))

    # the answer cited the p3 figure itself: only that one, not the page-13 or attached ones
    shown = _figures(result, [blocks[0], blocks[1]], store)
    assert [f["url"] for f in shown] == ["/api/figures/p3_0"]
    assert shown[0]["caption"] == "Title p3_0" and shown[0]["page"] == 3

    # only a passage on page 13 was cited: its page's figure, never the missing crop
    shown = _figures(result, [blocks[0]], store)
    assert [f["url"] for f in shown] == ["/api/figures/p13_0"]

    # nothing figure-related cited: fall back to what was attached to the model
    shown = _figures(result, [], store)
    assert [f["url"] for f in shown] == ["/api/figures/p8_0"]

    assert _figures(result, [blocks[1]], None) == []


# --------------------------------------------------------------------------- provider parameters


@pytest.mark.parametrize(
    ("profile", "key_field", "expected"),
    [
        ("gemini", "google_api_key", "max_output_tokens"),
        ("groq_build", "groq_api_key", "max_tokens"),
        ("anthropic", "anthropic_api_key", "max_tokens"),
    ],
)
def test_the_output_cap_uses_each_providers_own_parameter_name(
    profile: str, key_field: str, expected: str
) -> None:
    """Google's SDK rejects `max_tokens` before any network call (GenerateContentConfig forbids
    extra fields), which made every Gemini key fail the key test with a 502."""
    from langchain_core.messages import AIMessage

    bound: list[dict] = []

    class FakeChat:
        def bind(self, **kw):
            bound.append(kw)
            return self

        def invoke(self, messages):
            return AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

    settings = Settings(
        _env_file=None, app_env="test", model_profile=profile, **{key_field: "test-key-0123456789"}
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    assert llm.text("hi", role="small", max_tokens=7) == "ok"
    assert bound == [{expected: 7}]


def test_the_gemini_profile_uses_models_a_free_key_can_call() -> None:
    """Pinned by the 2026-10-04 check (D-72): 2.5 ids 404 for new users, Pro has no free quota,
    and gemini-3.8-flash allows 20 requests a day; flash-lite serves every role."""
    from rag.core.config import load_models_config

    gemini = load_models_config().profiles["gemini"]
    assert gemini.small.model == gemini.large.model == gemini.vision.model
    assert gemini.large.model == "gemini-3.5-flash-lite"
    assert gemini.small.provider_kwargs == {"thinking_level": "minimal"}
    assert gemini.large.provider_kwargs == {"thinking_level": "low"}


# ------------------------------------------------------------------------------- daily quota


GOOGLE_DAILY = (
    "role=large model=gemini-3.8-flash failed after 1 attempt(s) (HTTP 429): GoogleRateLimitError: "
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota', "
    "'details': [{'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
    "'quotaValue': '20'}]}, {'retryDelay': '40604s'}]}}"
)
GROQ_DAILY = (
    "role=large model=openai/gpt-oss-120b failed after 2 attempt(s) (HTTP 429): RateLimitError: "
    "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, "
    "Used 199500, Requested 1200. Please try again in 19m12s."
)
GROQ_MINUTE = (
    "role=large model=openai/gpt-oss-120b failed after 2 attempt(s) (HTTP 429): RateLimitError: "
    "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try again in 7.66s."
)


@pytest.mark.parametrize(
    ("message", "daily"), [(GOOGLE_DAILY, True), (GROQ_DAILY, True), (GROQ_MINUTE, False)]
)
def test_a_spent_daily_quota_is_told_apart_from_a_per_minute_limit(
    message: str, daily: bool
) -> None:
    from rag.llm import LLMCallError

    assert LLMCallError(message, status_code=429).daily_quota is daily
    assert LLMCallError(message, status_code=500).daily_quota is False


def test_the_key_test_says_the_daily_limit_truthfully() -> None:
    from rag.llm import DAILY_LIMIT_MESSAGE, LLMCallError

    err = byok.http_error(LLMCallError(GOOGLE_DAILY, status_code=429))
    assert err.status_code == 429 and err.detail == DAILY_LIMIT_MESSAGE
    assert err.headers == {"X-Limit": "daily"}
    assert "change the provider" in DAILY_LIMIT_MESSAGE and "kept" in DAILY_LIMIT_MESSAGE
    minute = byok.http_error(LLMCallError(GROQ_MINUTE, status_code=429))
    assert "Wait a minute" in minute.detail and not minute.headers


def test_a_daily_limit_mid_answer_degrades_with_the_same_message() -> None:
    from rag.graph import degraded_answer
    from rag.llm import DAILY_LIMIT_MESSAGE, LLMCallError
    from rag.query.assemble import AssembledContext

    ctx = AssembledContext(blocks=[], token_budget=2500, tokens_used=0)
    daily = degraded_answer(ctx, LLMCallError(GROQ_DAILY, status_code=429))
    assert daily.answer_markdown.startswith(DAILY_LIMIT_MESSAGE)
    minute = degraded_answer(ctx, LLMCallError(GROQ_MINUTE, status_code=429))
    assert DAILY_LIMIT_MESSAGE not in minute.answer_markdown


def test_the_status_is_found_on_the_cause_where_google_keeps_it() -> None:
    """LangChain's Google wrapper carries no status of its own; Google's ClientError, its cause,
    has `code`. Without this every Gemini 429 read as a 502 "could not be reached"."""
    from rag.llm import _status_code, is_retryable

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
            self.code = 429

    class GoogleRateLimitError(Exception):
        pass

    try:
        try:
            raise ClientError()
        except ClientError as inner:
            raise GoogleRateLimitError(str(inner)) from inner
    except GoogleRateLimitError as outer:
        assert _status_code(outer) == 429
        assert is_retryable(outer) is False  # a spent daily quota is not retried

    reset = ConnectionResetError(104, "connection reset")  # errno 104 is not HTTP 104
    assert _status_code(reset) is None and is_retryable(reset) is True
