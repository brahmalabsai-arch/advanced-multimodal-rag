"""In-session memory: follow-ups are rewritten into standalone questions (F3, D-71)."""

from __future__ import annotations

import json

import pytest

from rag.core.settings import PROJECT_ROOT
from rag.llm import LLMError
from rag.query.condense import Turn, condense, fit_history, needs_context
from rag.query.slots import get_slot_extractor

needs_index = pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)

HISTORY = [
    Turn(
        question="What was the revenue growth rate in fiscal 2026?",
        standalone="What was the revenue growth rate in fiscal 2026?",
        summary="+65.5% (revenue, FY2026 vs FY2025).",
    )
]


def _needs(question: str, history: list[Turn] = HISTORY) -> bool:
    slots = get_slot_extractor().extract(question)
    return needs_context(question, slots, history)[0]


# ----------------------------------------------------------------------------------- rules


@pytest.mark.parametrize(
    "question",
    [
        "And gross profit?",
        "What about operating income?",
        "and for fiscal 2025?",
        "Why?",
        "Why did it grow so fast?",
        "How does that compare with the prior year?",
        "Same for net income",
        "What drove that?",
    ],
)
def test_follow_ups_need_the_conversation(question: str) -> None:
    assert _needs(question)


@pytest.mark.parametrize(
    "question",
    [
        "What was revenue in fiscal 2026?",
        "Why did gross margin fall in fiscal 2026?",
        "How did inventories change year over year?",
        "What is the current ratio at the latest balance sheet date?",
        "What is NVIDIA's five-layer cake?",
        "When is NVIDIA's annual meeting?",
    ],
)
def test_self_contained_questions_cost_nothing(question: str) -> None:
    assert not _needs(question)


def test_without_a_conversation_nothing_is_a_follow_up() -> None:
    assert not _needs("And gross profit?", history=[])


# --------------------------------------------------------------------------------- history


def test_history_keeps_the_most_recent_turns_within_both_caps() -> None:
    turns = [Turn(question=f"question number {i}", summary="x " * 40) for i in range(25)]
    kept = fit_history(turns, max_turns=10, max_tokens=100_000)
    assert [t.question for t in kept] == [f"question number {i}" for i in range(15, 25)]
    small = fit_history(turns, max_turns=10, max_tokens=120)
    assert 0 < len(small) < 10 and small[-1].question == "question number 24"
    assert fit_history(turns, max_turns=0, max_tokens=1500) == []


def test_a_turn_line_prefers_the_standalone_question_and_bounds_the_summary() -> None:
    t = Turn(
        question="and gross profit?", standalone="What was gross profit growth?", summary="y" * 999
    )
    line = t.line()
    assert line.startswith("Q: What was gross profit growth?") and "and gross profit" not in line
    assert len(line) < 400


# ---------------------------------------------------------------------------------- rewrite


class _Client:
    def __init__(self, reply):  # noqa: ANN001
        self.reply = reply
        self.calls: list[dict] = []

    def text(self, prompt, **kw):  # noqa: ANN001, ANN003
        self.calls.append({"prompt": prompt, **kw})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _condense(question: str, client: _Client | None):
    return condense(
        question,
        HISTORY,
        get_slot_extractor().extract(question),
        client=client,  # type: ignore[arg-type]
        request_id="r1",
    )


def test_a_follow_up_is_rewritten_by_the_small_role_with_the_conversation() -> None:
    client = _Client("What was NVIDIA's gross profit growth rate in fiscal 2026?")
    r = _condense("And gross profit?", client)
    assert r.rewritten and r.follow_up and r.turns_used == 1
    assert r.question == "What was NVIDIA's gross profit growth rate in fiscal 2026?"
    assert r.asked == "And gross profit?"
    call = client.calls[0]
    assert call["role"] == "small" and "FOLLOW-UP: And gross profit?" in call["prompt"]
    assert "+65.5%" in call["prompt"]  # the earlier answer's summary is part of the context


def test_a_self_contained_question_makes_no_call() -> None:
    client = _Client("unused")
    r = _condense("What was revenue in fiscal 2026?", client)
    assert not r.follow_up and not r.rewritten and client.calls == []


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (_Client(LLMError("429 rate limited")), "429"),
        (_Client("   "), "empty"),
        (_Client("x" * 5000), "too long"),
        (None, "no model client"),
    ],
)
def test_any_rewrite_failure_answers_the_question_as_typed(client, error: str) -> None:  # noqa: ANN001
    r = _condense("And gross profit?", client)
    assert r.follow_up and not r.rewritten
    assert r.question == "And gross profit?" and error in (r.error or "")


# --------------------------------------------------------------------------------- pipeline


@needs_index
def test_a_follow_up_runs_the_pipeline_on_its_standalone_question() -> None:
    """End to end: "And gross profit?" after a revenue-growth turn is rewritten, and slots, the
    calculator and the completeness check all work on the rewritten question."""
    from langchain_core.messages import AIMessage

    from rag.api.payload import answer_payload
    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    standalone = (
        "What was NVIDIA's gross profit growth rate in fiscal 2026 compared with fiscal 2025?"
    )
    settings = Settings(
        _env_file=None, groq_api_key="gsk_test_key_0123456789", app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )  # fmt: skip
    seen: list[str] = []

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            system = messages[0].content
            seen.append(system[:40])
            if "follow-up question" in system:
                content = f"Standalone question: {standalone}\n"
            else:
                content = json.dumps(
                    {
                        "answer_markdown": f"Gross profit grew {gross} [K1], calculated "
                        "from the filed figures.",
                        "citations": ["K1"],
                        "confidence": "high",
                        "answer_class": "analytical",
                    }
                )
            return AIMessage(
                content=content,
                usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    pipeline = Pipeline(settings=settings, client=llm)
    gross = pipeline.calculator.compute(
        "yoy_change_pct", fiscal_year=2026, metric="gross profit"
    ).formatted()

    r = pipeline.ask("And gross profit?", bypass_cache=True, history=HISTORY)

    assert r.asked == "And gross profit?" and r.question == standalone
    assert r.condense is not None and r.condense.rewritten
    assert r.slots.metrics == ["gross profit"]
    assert [c.metric for c in r.calculations if c.formula == "yoy_change_pct"] == ["gross profit"]
    assert r.coverage.passed and r.verify.passed
    assert "condense" in r.latency_ms_by_node
    payload = answer_payload(r, [], pipeline.store)
    assert payload["standalone_question"] == standalone
    step = next(s for s in payload["trace"]["steps"] if s["label"] == "Understood the follow-up")
    assert step["detail"] == f'read as "{standalone}"'

    # a first-turn question makes no rewrite call and shows no such step
    seen.clear()
    r2 = pipeline.ask("What was the revenue growth rate in fiscal 2026?", bypass_cache=True)
    assert r2.condense is not None and not r2.condense.follow_up
    assert not any("follow-up" in s for s in seen)
    labels = [s["label"] for s in answer_payload(r2, [], pipeline.store)["trace"]["steps"]]
    assert "Understood the follow-up" not in labels


def test_the_rewrite_keeps_one_line_without_labels_or_quotes() -> None:
    from rag.query.condense import _one_line

    assert _one_line('\n  "What was gross profit growth in fiscal 2026?"\nBecause…') == (
        "What was gross profit growth in fiscal 2026?"
    )
    assert _one_line("STANDALONE QUESTION: Why did revenue grow?") == "Why did revenue grow?"
    assert _one_line("") == ""
