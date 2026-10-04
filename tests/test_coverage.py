"""Multi-part questions: every metric is computed, and the answer is checked for every part.

The failure these pin down (2026-10-04 session): "growth rate of profit and revenue separately"
computed revenue only, because bare "profit" was not a metric and the calculator applied the
YoY pair to the first metric alone; the answer then said the filing does not state the profit
growth rate, and the verifier passed it because every number it *did* state was traceable.
"""

from __future__ import annotations

import json

import pytest

from rag.calc.calculator import CalcInput, CalculationResult, select_formulas_from_slots
from rag.core.settings import PROJECT_ROOT
from rag.llm import LLMError
from rag.query.coverage import (
    CoverageResult,
    check_coverage,
    expected_parts,
    is_multi_part,
    part_is_answered,
)
from rag.query.generate import Answer
from rag.query.slots import get_slot_extractor

needs_index = pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)

LINE_ITEMS = ["revenue", "gross profit", "net income", "operating income", "inventories"]


def _yoy(metric: str, pct: float, abs_change: float) -> list[CalculationResult]:
    def calc(formula: str, kind: str, value: float) -> CalculationResult:
        return CalculationResult(
            formula=formula,
            description="Year-over-year change",
            kind=kind,  # type: ignore[arg-type]
            expression="(current - prior) / prior * 100" if kind == "pct" else "current - prior",
            fiscal_year=2026,
            metric=metric,
            inputs=[
                CalcInput(
                    name="current",
                    line_item=metric,
                    line_item_norm=metric,
                    value=1.0,
                    fiscal_year=2026,
                )
            ],
            result=value,
            rounded=value,
        )

    return [calc("yoy_change_pct", "pct", pct), calc("yoy_change_abs", "amount", abs_change)]


# ------------------------------------------------------------------------ slots + calculator


@pytest.mark.parametrize(
    ("question", "metrics"),
    [
        (
            "What has been the growth rate of Nvidia over last year in terms of profit and "
            "revenue separately?",
            ["net income", "revenue"],
        ),
        ("Calculate the profit growth CY vs LY", ["net income"]),
        (
            "what was the gross profit growth rate and revenue growth rate?",
            ["gross profit", "revenue"],
        ),
        ("How did operating profit change year over year?", ["operating income"]),
    ],
)
def test_bare_profit_is_net_income_and_longer_phrases_still_win(
    question: str, metrics: list[str]
) -> None:
    slots = get_slot_extractor().extract(question)
    assert slots.metrics == metrics


def test_profit_and_loss_statement_is_a_statement_not_a_metric() -> None:
    slots = get_slot_extractor().extract(
        "calculate the revenue growth rate from the profit and loss statement"
    )
    assert slots.metrics == ["revenue"] and slots.statement == "income_statement"


def test_every_metric_gets_its_own_yoy_pair_in_question_order() -> None:
    slots = get_slot_extractor().extract(
        "what was the gross profit growth rate and revenue growth rate?"
    )
    requests = select_formulas_from_slots(slots, LINE_ITEMS)
    assert [(r.formula, r.metric) for r in requests] == [
        ("yoy_change_pct", "gross profit"),
        ("yoy_change_abs", "gross profit"),
        ("yoy_change_pct", "revenue"),
        ("yoy_change_abs", "revenue"),
    ]


def test_a_point_lookup_of_two_metrics_runs_no_yoy() -> None:
    slots = get_slot_extractor().extract("What were revenue and net income in fiscal 2026?")
    assert select_formulas_from_slots(slots, LINE_ITEMS) == []


# ------------------------------------------------------------------------------- the rules


def test_a_metric_with_a_calculation_needs_that_calculation_in_the_answer() -> None:
    slots = get_slot_extractor().extract("growth of revenue and net income year over year")
    calcs = _yoy("revenue", 65.47, 85441.0) + _yoy("net income", 64.75, 47400.0)
    parts = expected_parts(slots, calcs, "COMPARISON_TREND", None, default_fiscal_year=2026)
    assert [p.label for p in parts] == [
        "revenue: change FY2025 to FY2026",
        "net income: change FY2025 to FY2026",
    ]
    revenue_only = "Revenue grew **+65.47%** [K1], an increase of $85,441 million [K2]."
    assert part_is_answered(parts[0], revenue_only)
    assert not part_is_answered(parts[1], revenue_only)
    # rounding and the billion restatement both count, as they do in the verifier
    assert part_is_answered(parts[1], "Net income rose 64.8% [K3].")
    assert part_is_answered(parts[1], "Net income rose by $47.4 billion [K4].")


def test_small_integers_and_years_do_not_count_as_an_answer() -> None:
    slots = get_slot_extractor().extract("growth of revenue year over year")
    parts = expected_parts(slots, _yoy("revenue", 5.0, 3.0), "COMPARISON_TREND", None,
                           default_fiscal_year=2026)  # fmt: skip
    assert not part_is_answered(parts[0], "Revenue changed in fiscal 2026 across 5 segments.")
    assert part_is_answered(parts[0], "Revenue grew 5% [K1].")


def test_rules_report_the_gap_and_point_at_the_block_that_holds_it() -> None:
    question = "What has been the growth rate over last year of profit and revenue separately?"
    slots = get_slot_extractor().extract(question)
    calcs = _yoy("net income", 64.75, 47400.0) + _yoy("revenue", 65.47, 85441.0)
    answer = Answer(
        answer_markdown="Revenue grew +65.47% [K3]. The filing does not state a profit "
        "growth rate explicitly."
    )
    result = check_coverage(
        question, answer, slots=slots, intent="COMPARISON_TREND", calculations=calcs
    )
    assert result.method == "rules" and not result.passed
    assert result.missing == ["net income: change FY2025 to FY2026"]
    note = result.retry_note() or ""
    assert "[K1] gives +64.75%" in note
    assert "Never say the filing does not provide a value that a calculation block gives" in note


def test_failed_calculations_are_not_demanded() -> None:
    slots = get_slot_extractor().extract("growth of revenue year over year")
    failed = CalculationResult(
        formula="yoy_change_pct",
        description="Year-over-year change",
        kind="pct",
        expression="x",
        fiscal_year=2026,
        metric="revenue",
        status="missing_inputs",
    )
    assert expected_parts(slots, [failed], "COMPARISON_TREND", None, default_fiscal_year=2026) == []


def test_explanatory_answers_are_not_held_to_restating_values() -> None:
    slots = get_slot_extractor().extract("why did revenue grow?")
    calcs = _yoy("revenue", 65.47, 85441.0)
    # calculations are always demanded; a bare metric is demanded only for value intents
    assert len(expected_parts(slots, calcs, "EXPLANATORY", None, default_fiscal_year=2026)) == 1
    assert expected_parts(slots, [], "EXPLANATORY", None, default_fiscal_year=2026) == []


# --------------------------------------------------------------------------- multi-part


@pytest.mark.parametrize(
    ("question", "multi"),
    [
        ("What was research and development expense in fiscal 2026?", False),
        ("What is in the profit and loss statement for fiscal 2026?", False),
        ("What were revenue and net income in fiscal 2026?", True),
        ("Why did gross margin fall and what are the main risks?", True),
        ("What is the current ratio? And the quick ratio?", True),
        ("What is the current ratio as of Jan 25, 2026?", False),
    ],
)
def test_multi_part_detection_ignores_and_inside_slot_phrases(question: str, multi: bool) -> None:
    slots = get_slot_extractor().extract(question)
    assert is_multi_part(question, slots, []) is multi


class _ScriptedClient:
    def __init__(self, reply: dict | Exception):
        self.reply = reply
        self.calls: list[dict] = []

    def json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003
        self.calls.append({"prompt": prompt, **kw})
        if isinstance(self.reply, Exception):
            raise self.reply
        return schema.model_validate(self.reply)


def test_self_check_runs_only_for_several_asks_after_the_rules_pass() -> None:
    question = "Why did gross margin fall and what are the main risks?"
    slots = get_slot_extractor().extract(question)
    answer = Answer(answer_markdown="Gross margin fell because of H20 inventory charges [C1].")
    client = _ScriptedClient(
        {
            "parts": [
                {"ask": "why gross margin fell", "addressed": True},
                {"ask": "the main risks", "addressed": False},
            ]
        }
    )
    result = check_coverage(
        question,
        answer,
        slots=slots,
        intent="EXPLANATORY",
        calculations=[],
        client=client,  # type: ignore[arg-type]
        request_id="r1",
    )
    assert result.method == "rules+self_check"
    assert result.missing == ["the main risks"]
    call = client.calls[0]
    assert call["role"] == "small" and question in call["prompt"]

    single = _ScriptedClient({"parts": []})
    check_coverage(
        "Why did gross margin fall?",
        answer,
        slots=get_slot_extractor().extract("Why did gross margin fall?"),
        intent="EXPLANATORY",
        calculations=[],
        client=single,  # type: ignore[arg-type]
    )
    assert single.calls == []


def test_a_self_check_failure_never_fails_the_answer() -> None:
    question = "Why did gross margin fall and what are the main risks?"
    result = check_coverage(
        question,
        Answer(answer_markdown="Because of inventory charges [C1]."),
        slots=get_slot_extractor().extract(question),
        intent="EXPLANATORY",
        calculations=[],
        client=_ScriptedClient(LLMError("429 rate limited")),  # type: ignore[arg-type]
    )
    assert result.passed and result.self_check_error and "429" in result.self_check_error


def test_out_of_scope_and_other_intents_are_skipped() -> None:
    result = check_coverage(
        "what is the weather in Guwahati today?",
        Answer(answer_markdown="That is outside the filing."),
        slots=get_slot_extractor().extract("what is the weather in Guwahati today?"),
        intent="OUT_OF_SCOPE",
        calculations=[],
    )
    assert result == CoverageResult(method="skipped")


# ------------------------------------------------------------------------------ in the graph


@needs_index
def test_the_pipeline_regenerates_an_answer_that_skipped_a_part() -> None:
    """End to end on the real index: the first answer gives revenue growth only, the check
    names net income and the block holding it, and the second answer, which covers both,
    is the one returned. Two generation calls, no self-check (the rules settle it)."""
    from langchain_core.messages import AIMessage

    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    prompts: list[str] = []

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            system, user = messages[0].content, messages[-1].content
            if "financial analyst assistant" not in system:
                raise AssertionError(f"unexpected model call: {system[:80]}")
            prompts.append(user)
            text = (
                f"Revenue grew {revenue} [K1]."
                if len(prompts) == 1
                else f"Revenue grew {revenue} [K1] and net income, read as profit, grew "
                f"{net_income} [K3]; both were calculated from the filed figures."
            )
            return AIMessage(
                content=json.dumps(
                    {
                        "answer_markdown": text,
                        "citations": ["K1", "K3"],
                        "confidence": "high",
                        "answer_class": "analytical",
                    }
                ),
                usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    pipeline = Pipeline(settings=settings, client=llm)
    calc = pipeline.calculator
    revenue = calc.compute("yoy_change_pct", fiscal_year=2026, metric="revenue").formatted()
    net_income = calc.compute("yoy_change_pct", fiscal_year=2026, metric="net income").formatted()

    r = pipeline.ask(
        "What was the year over year growth rate of revenue and profit separately?",
        bypass_cache=True,
    )

    assert [c.metric for c in r.calculations if c.formula == "yoy_change_pct"] == [
        "revenue",
        "net income",
    ]
    assert r.generation_attempts == 2 and len(prompts) == 2
    assert "net income: change FY2025 to FY2026" in prompts[1]
    assert f"[K3] gives {net_income}" in prompts[1]
    assert '"profit" was read as net income' in prompts[0]
    assert r.coverage.passed and r.coverage.method == "rules"
    assert r.verify.passed and r.answer.confidence == "high" and r.warning is None
    assert "net income" in r.answer.answer_markdown


def test_house_style_citations_are_normalised_to_square_brackets() -> None:
    from rag.query.generate import normalise_citations

    a = Answer(answer_markdown="Revenue grew +65.5%【K3】 and net income +64.7%【 K1 †L2-L4】.")
    assert normalise_citations(a).answer_markdown == (
        "Revenue grew +65.5%[K3] and net income +64.7%[K1]."
    )
    plain = Answer(answer_markdown="Revenue grew [K3].")
    assert normalise_citations(plain) is plain
