"""Phase 0 tests — model layer: JSON retry, 429 backoff honouring retry-after, ledger lines,
vision message format, lazy provider resolution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from rag.core.settings import PROJECT_ROOT
from rag.llm import (
    LLMCallError,
    LLMClient,
    LLMJSONError,
    is_retryable,
    retry_after_seconds,
)

FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "smoke_image.png"


class Answer(BaseModel):
    value: int
    note: str


class FakeStatusError(Exception):
    """Mimics groq.APIStatusError: `.status_code` and `.response.headers`."""

    def __init__(self, status_code: int, message: str = "boom", headers: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class FakeChat:
    """Stands in for a LangChain chat model: `.bind()` returns self, `.invoke()` pops a script."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[list] = []
        self.bind_kwargs: list[dict] = []

    def bind(self, **kwargs):
        self.bind_kwargs.append(kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return AIMessage(
            content=item,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def make_client(
    settings, models_config, ledger, fake_clock, script, **kwargs
) -> tuple[LLMClient, FakeChat]:
    chat = FakeChat(script)
    client = LLMClient(
        settings,
        models_config,
        ledger=ledger,
        chat_factory=lambda cfg, role: chat,
        clock=fake_clock,
        sleep=fake_clock.sleep,
        **kwargs,
    )
    return client, chat


# --------------------------------------------------------------------------- JSON path


def test_json_retry_once_then_success(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings,
        models_config,
        ledger,
        fake_clock,
        ["not json at all", '{"value": 42, "note": "ok"}'],
    )
    result = client.json("What is 6*7?", Answer, role="large", request_id="r1")

    assert result == Answer(value=42, note="ok")
    assert len(chat.calls) == 2
    # The corrective turn carries the model's bad output and the validation error.
    retry_messages = chat.calls[1]
    assert (
        isinstance(retry_messages[-2], AIMessage)
        and retry_messages[-2].content == "not json at all"
    )
    assert isinstance(retry_messages[-1], HumanMessage)
    assert "not valid JSON" in retry_messages[-1].content
    # Groq JSON mode requested on both attempts.
    assert all(kw.get("response_format") == {"type": "json_object"} for kw in chat.bind_kwargs)

    statuses = [(r.status, r.retries) for r in ledger.read_all()]
    assert statuses == [("ok", 0), ("invalid_json", 0), ("ok", 0)]
    assert all(r.request_id == "r1" for r in ledger.read_all())
    assert ledger.read_all()[-1].model == models_config.active().large.model


def test_json_fails_after_second_bad_response(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings, models_config, ledger, fake_clock, ['{"value": "x"}', "{oops"]
    )
    with pytest.raises(LLMJSONError) as exc_info:
        client.json("q", Answer)
    assert exc_info.value.raw == "{oops"
    assert len(chat.calls) == 2


def test_json_accepts_fenced_output(settings, models_config, ledger, fake_clock) -> None:
    client, _ = make_client(
        settings, models_config, ledger, fake_clock, ['```json\n{"value": 1, "note": "n"}\n```']
    )
    assert client.json("q", Answer).value == 1


def test_json_schema_is_in_system_prompt(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings, models_config, ledger, fake_clock, ['{"value":1,"note":""}']
    )
    client.json("q", Answer, system="You are terse.")
    system = chat.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "You are terse." in system.content
    assert "JSON" in system.content
    assert '"value"' in system.content and '"note"' in system.content


# -------------------------------------------------------------------------- text path


def test_text_call_writes_one_ledger_line(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(settings, models_config, ledger, fake_clock, ["PONG"])
    assert client.text("ping", role="small", max_tokens=8) == "PONG"
    assert chat.bind_kwargs == [{"max_tokens": 8}]  # no JSON mode for text()
    [record] = ledger.read_all()
    assert (record.status, record.role, record.tokens_in, record.tokens_out) == (
        "ok",
        "small",
        10,
        5,
    )
    assert record.provider == "groq"


def test_text_handles_content_blocks(settings, models_config, ledger, fake_clock) -> None:
    client, _ = make_client(
        settings,
        models_config,
        ledger,
        fake_clock,
        [[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]],
    )
    assert client.text("q") == "AB"


# ---------------------------------------------------------------------- retry / 429


def test_429_retries_with_retry_after_and_logs_each_attempt(
    settings, models_config, ledger, fake_clock
) -> None:
    err = FakeStatusError(
        429, "Rate limit reached. Please try again in 7.5s", headers={"retry-after": "9"}
    )
    client, chat = make_client(settings, models_config, ledger, fake_clock, [err, "ok"])

    assert client.text("q") == "ok"
    assert len(chat.calls) == 2
    # tenacity slept at least the retry-after value (9 s) before the second attempt.
    assert fake_clock.sleeps and max(fake_clock.sleeps) >= 9.0

    records = ledger.read_all()
    assert [(r.status, r.retries) for r in records] == [("rate_limited", 0), ("ok", 1)]
    assert "429" not in (records[0].error or "") or "FakeStatusError" in records[0].error


def test_5xx_retries_then_gives_up(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings,
        models_config,
        ledger,
        fake_clock,
        [FakeStatusError(503), FakeStatusError(502), FakeStatusError(500)],
        max_attempts=3,
    )
    with pytest.raises(LLMCallError, match=r"after 3 attempt\(s\)"):
        client.text("q")
    assert len(chat.calls) == 3
    assert [r.status for r in ledger.read_all()] == ["error", "error", "error"]


def test_4xx_is_not_retried(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings, models_config, ledger, fake_clock, [FakeStatusError(400, "bad request")]
    )
    with pytest.raises(LLMCallError, match=r"after 1 attempt\(s\).*bad request"):
        client.text("q")
    assert len(chat.calls) == 1
    assert fake_clock.sleeps == []


def test_retry_after_parsing() -> None:
    assert retry_after_seconds(FakeStatusError(429, headers={"retry-after": "3.5"})) == 3.5
    assert retry_after_seconds(FakeStatusError(429, "Please try again in 750ms")) == 0.75
    assert retry_after_seconds(FakeStatusError(429, "Please try again in 2m")) == 120.0
    assert retry_after_seconds(RuntimeError("no hint")) is None


def test_is_retryable() -> None:
    assert is_retryable(FakeStatusError(429))
    assert is_retryable(FakeStatusError(429, "Please try again in 20.3s"))
    # Daily-quota exhaustion: not worth blocking the request for
    assert not is_retryable(FakeStatusError(429, "Please try again in 19m"))
    assert not is_retryable(FakeStatusError(429, headers={"retry-after": "600"}))
    assert is_retryable(FakeStatusError(503))
    assert not is_retryable(FakeStatusError(401))
    assert is_retryable(type("APIConnectionError", (Exception,), {})())
    assert not is_retryable(ValueError("x"))


# ------------------------------------------------------------------------ pacing hook


def test_pacing_applies_before_call(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(settings, models_config, ledger, fake_clock, ["a"] * 31)
    for _ in range(30):  # groq_build small: rpm=30
        client.text("q", max_tokens=1)
    assert fake_clock.sleeps == []
    client.text("q", max_tokens=1)
    assert len(fake_clock.sleeps) == 1 and fake_clock.sleeps[0] > 0
    assert ledger.read_all()[-1].pacing_wait_ms > 0
    assert len(chat.calls) == 31


# ----------------------------------------------------------------------------- vision


def test_vision_json_builds_image_message(settings, models_config, ledger, fake_clock) -> None:
    client, chat = make_client(
        settings, models_config, ledger, fake_clock, ['{"value": 2, "note": "img"}']
    )
    result = client.vision_json("describe", [FIXTURE_IMAGE], Answer, request_id="v1")
    assert result.value == 2

    human = chat.calls[0][-1]
    assert isinstance(human, HumanMessage)
    assert human.content[0] == {"type": "text", "text": "describe"}
    url = human.content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert ledger.read_all()[0].role == "vision"
    assert ledger.read_all()[0].model == models_config.active().vision.model


def test_vision_json_enforces_image_limit(settings, models_config, ledger, fake_clock) -> None:
    client, _ = make_client(settings, models_config, ledger, fake_clock, [])
    limit = models_config.active().vision.max_images_per_request
    assert limit == 3
    with pytest.raises(ValueError, match="exceed"):
        client.vision_json("d", [b"x"] * (limit + 1), Answer)
    with pytest.raises(ValueError, match="at least one"):
        client.vision_json("d", [], Answer)


# ----------------------------------------------------------------------- lazy import


def test_default_factory_builds_chatgroq_lazily(
    settings, models_config, ledger, fake_clock
) -> None:
    client = LLMClient(
        settings, models_config, ledger=ledger, clock=fake_clock, sleep=fake_clock.sleep
    )
    assert client._models == {}  # nothing constructed until first use
    model = client._model("small")
    assert type(model).__name__ == "ChatGroq"
    assert model.model_name == models_config.active().small.model
    assert model.max_retries == 0


def test_template_profile_without_provider_package_fails_clearly(
    tmp_path: Path, ledger, fake_clock
) -> None:
    from rag.core.config import load_models_config
    from rag.core.settings import Settings
    from rag.llm import ProviderNotInstalledError

    s = Settings(
        _env_file=None,
        model_profile="anthropic",
        anthropic_api_key="sk-ant-test-0123456789abcdef",
        config_dir=PROJECT_ROOT / "config",
        data_dir=tmp_path,
    )
    client = LLMClient(
        s, load_models_config(settings=s), ledger=ledger, clock=fake_clock, sleep=fake_clock.sleep
    )
    if importlib.util.find_spec("langchain_anthropic") is not None:
        pytest.skip("langchain-anthropic installed; lazy-import failure path not exercised")
    with pytest.raises(ProviderNotInstalledError, match="langchain-anthropic"):
        client._model("large")


def test_ledger_lines_are_valid_json(settings, models_config, ledger, fake_clock) -> None:
    client, _ = make_client(settings, models_config, ledger, fake_clock, ["x"])
    client.text("q", ingestion_job="job-1")
    line = ledger.path.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["ingestion_job"] == "job-1"
    assert "request_id" not in data  # None fields are dropped
    assert set(data) >= {
        "ts",
        "role",
        "provider",
        "model",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "retries",
        "status",
    }
