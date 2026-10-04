"""Evidence for in-session memory (D-71): are follow-ups rewritten correctly?

Scripted conversations, each an earlier turn plus a follow-up, run through `condense` with the
real `small` role. A rewrite is judged deterministically, never by a model: the standalone
question is passed through the slot extractor and must carry the expected metric / formula /
topic / fiscal period / ask type, and must not import a subject the follow-up did not refer to.
Controls check the other direction: a self-contained question, or a follow-up without any
conversation, must make no call and come back unchanged.

    .venv/Scripts/python eval/conversation_memory.py            # ~14 small-role calls
    .venv/Scripts/python eval/conversation_memory.py --report   # also writes the report
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT, get_settings
from rag.llm import LLMClient
from rag.query.condense import Turn, condense
from rag.query.slots import get_slot_extractor

REPORT = PROJECT_ROOT / "docs" / "reports" / "conversation_memory.md"


@dataclass
class Case:
    name: str
    history: list[Turn]
    follow_up: str
    metrics: set[str] = field(default_factory=set)
    formulas: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    periods: set[str] = field(default_factory=set)
    ask: str | None = None
    must_not: set[str] = field(default_factory=set)  # metrics the rewrite must not import
    change: bool = False  # the earlier turn asked for a change / growth, and so must the rewrite
    expect_call: bool = True


def turn(q: str, summary: str) -> Turn:
    return Turn(question=q, standalone=q, summary=summary)


REV = turn(
    "What was the revenue growth rate in fiscal 2026?", "+65.5% (revenue, FY2026 vs FY2025)."
)
CASES = [
    Case(
        "metric swap",
        [REV],
        "And gross profit?",
        metrics={"gross profit"},
        must_not={"revenue"},
        change=True,
    ),
    Case(
        "metric swap, 'what about'",
        [REV],
        "What about net income?",
        metrics={"net income"},
        must_not={"revenue"},
        change=True,
    ),
    Case(
        "reason for the previous figure",
        [REV],
        "Why did it grow so fast?",
        metrics={"revenue"},
        ask="reason",
    ),
    Case(
        "period shift",
        [
            turn(
                "What were total assets as of January 25, 2026?",
                "$206,803 million as of January 25, 2026 (FY2026).",
            )
        ],
        "And the year before?",
        metrics={"total assets"},
        periods={"FY2025"},
    ),
    Case(
        "bare period",
        [
            turn(
                "What were cash and cash equivalents in fiscal 2026?",
                "$10,605 million as of January 25, 2026.",
            )
        ],
        "and FY2025?",
        metrics={"cash and cash equivalents"},
        periods={"FY2025"},
    ),
    Case(
        "formula swap",
        [
            turn(
                "What is the current ratio at the latest balance sheet date?",
                "3.91 (current ratio, as of 2026-01-25).",
            )
        ],
        "How about the quick ratio?",
        formulas={"quick_ratio"},
        must_not={"current_ratio"},
    ),
    Case(
        "bare why",
        [
            turn(
                "How did inventories change year over year?",
                "Inventories rose to $21,403 million from $10,080 million.",
            )
        ],
        "Why?",
        metrics={"inventories"},
        ask="reason",
    ),
    Case(
        "topic reference",
        [
            turn(
                "Why did gross margin fall in fiscal 2026?",
                "Gross margin fell to 71.1% from 75.0%, H20 charges.",
            )
        ],
        "What are the risks to it going forward?",
        topics={"gross_margin"},
        ask="risk",
    ),
    Case(
        "compare with earlier subject",
        [turn("What was net income in fiscal 2026?", "$120,067 million in fiscal 2026.")],
        "Compare that with operating income",
        metrics={"net income", "operating income"},
    ),
    Case(
        "figure follow-up",
        [
            turn(
                "What is NVIDIA's five-layer cake?",
                "A five-layer stack: energy, chips, infrastructure, models, applications.",
            )
        ],
        "Which layer is at the bottom?",
        topics={"five_layer_cake"},
    ),
    Case(
        "event follow-up",
        [
            turn(
                "When is NVIDIA's annual meeting?",
                "The 2026 annual meeting is scheduled for June 24, 2026.",
            )
        ],
        "Can shareholders attend it online?",
        topics={"annual_meeting"},
    ),
    Case(
        "formula, change over time",
        [turn("What is total debt to equity?", "0.05 (debt to equity, FY2026).")],
        "Is that higher than last year?",
        formulas={"debt_to_equity"},
        change=True,
    ),
    # controls: no call, question unchanged
    Case(
        "control: self-contained",
        [REV],
        "What is NVIDIA's five-layer cake?",
        topics={"five_layer_cake"},
        expect_call=False,
    ),
    Case("control: no conversation", [], "And gross profit?", expect_call=False),
]


def judge(case: Case, standalone: str) -> list[str]:
    s = get_slot_extractor().extract(standalone)
    problems = []
    if not case.metrics <= set(s.metrics):
        problems.append(f"metrics {sorted(case.metrics - set(s.metrics))} missing")
    if not case.formulas <= set(s.formulas):
        problems.append(f"formulas {sorted(case.formulas - set(s.formulas))} missing")
    if not case.topics <= set(s.topics):
        problems.append(f"topics {sorted(case.topics - set(s.topics))} missing")
    if not case.periods <= set(s.fiscal_periods):
        problems.append(f"periods {sorted(case.periods - set(s.fiscal_periods))} missing")
    if case.ask and s.ask != case.ask:
        problems.append(f"ask {s.ask!r}, expected {case.ask!r}")
    if case.change and not (
        s.direction or set(s.aggregation) & {"growth", "yoy", "change", "pct", "compare"}
    ):
        problems.append("dropped the change / growth the conversation was about")
    leaked = case.must_not & (set(s.metrics) | set(s.formulas))
    if leaked:
        problems.append(f"imported {sorted(leaked)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    client = LLMClient(get_settings())
    rows = []
    for case in CASES:
        t0 = time.perf_counter()
        r = condense(case.follow_up, case.history,
                     get_slot_extractor().extract(case.follow_up),
                     client=client, request_id="eval-memory")  # fmt: skip
        ms = int((time.perf_counter() - t0) * 1000)
        problems = []
        if case.expect_call != r.follow_up:
            problems.append("called the model" if r.follow_up else "did not detect the follow-up")
        if r.error:
            problems.append(f"error: {r.error[:80]}")
        if case.expect_call and not r.error:
            problems += judge(case, r.question)
        if not case.expect_call and r.question != case.follow_up:
            problems.append("changed a question that needed no rewrite")
        rows.append((case, r, ms, problems))
        mark = "ok " if not problems else "FAIL"
        print(f"{mark} {case.name:32} {case.follow_up!r:45} -> {r.question!r}  {ms} ms")
        for p in problems:
            print("      ", p)
    rewrites = [x for x in rows if x[0].expect_call]
    controls = [x for x in rows if not x[0].expect_call]
    ok_rw = sum(1 for x in rewrites if not x[3])
    ok_ct = sum(1 for x in controls if not x[3])
    lat = sorted(x[2] for x in rewrites)
    p50 = lat[len(lat) // 2] if lat else 0
    print(f"\nrewrites correct {ok_rw}/{len(rewrites)} · controls {ok_ct}/{len(controls)} · "
          f"rewrite p50 {p50} ms")  # fmt: skip
    if args.report:
        REPORT.write_text(render(rows, ok_rw, len(rewrites), ok_ct, len(controls), p50, client),
                          encoding="utf-8")  # fmt: skip
        print("report →", REPORT.relative_to(PROJECT_ROOT))
    return 0 if ok_rw == len(rewrites) and ok_ct == len(controls) else 1


def render(rows, ok_rw, n_rw, ok_ct, n_ct, p50, client) -> str:  # noqa: ANN001
    lines = [
        "# Conversation memory: follow-up rewrites (D-71)",
        "",
        f"Model: `{client.model_id('small')}` (`small` role). Produced by "
        "`eval/conversation_memory.py`.",
        "",
        f"**Rewrites judged correct: {ok_rw}/{n_rw}. Controls (no call, unchanged): {ok_ct}/{n_ct}. "
        f"Rewrite latency p50: {p50} ms.**",
        "",
        "A rewrite is correct when the slot extractor, run on the standalone question, finds the "
        "expected metric / formula / topic / fiscal period / ask type and no subject the "
        "follow-up did not refer to. No model judges another model here.",
        "",
        "| Case | Follow-up | Read as | Result |",
        "|---|---|---|---|",
    ]
    for case, r, _ms, problems in rows:
        read_as = (
            r.question
            if r.rewritten
            else "*(unchanged, no call)*"
            if not r.follow_up
            else r.question
        )
        verdict = "correct" if not problems else "; ".join(problems)
        lines.append(f"| {case.name} | {case.follow_up} | {read_as} | {verdict} |")
    lines += [
        "",
        "Caveats: 14 cases, written by hand against this report; a single run of a sampled model. "
        "The cases cover the follow-up shapes seen in use (metric swap, period shift, bare *why*, "
        "reference to an earlier subject or topic, comparison) — not every phrasing a visitor "
        "will type. The rules that decide whether to call the model are covered by "
        "`tests/test_condense.py`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
