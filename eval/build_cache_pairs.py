"""Generate the cache calibration sets (plan Phase 6, architecture §5.10).

    .venv/Scripts/python eval/build_cache_pairs.py

Writes `eval/adversarial_cache_pairs.jsonl` (≥ 200 pairs that must MISS: period swaps, metric
swaps, direction swaps, negations, plus *same-slot different-question* pairs that only the
similarity threshold can catch) and `eval/paraphrase_pairs.jsonl` (pairs that should HIT: same
slots, different wording). Generation is template-based and deterministic, so the files
are reproducible; hand-written pairs from the golden set are appended at the end.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "eval"

METRICS = [
    ("total assets", "total liabilities"),
    ("cash and cash equivalents", "marketable securities"),
    ("inventories", "accounts receivable"),
    ("goodwill", "intangible assets"),
    ("revenue", "gross profit"),
    ("net income", "operating income"),
    ("long-term debt", "short-term debt"),
    ("total current assets", "total current liabilities"),
    ("research and development", "sales, general and administrative"),
    ("dividends paid", "share repurchases"),
    ("retained earnings", "additional paid-in capital"),
    ("operating cash flow", "capital expenditures"),
]
FORMULAS = [("current ratio", "quick ratio"), ("debt-to-equity ratio", "equity ratio")]

PERIODS = [
    ("as of January 25, 2026", "as of January 26, 2025"),
    ("in fiscal 2026", "in fiscal 2025"),
    ("at the end of FY2026", "at the end of FY2025"),
    ("for fiscal year 2026", "for fiscal year 2024"),
    ("in FY26", "in FY25"),
]

LOOKUP_TEMPLATES = [
    "What were NVIDIA's {m} {p}?",
    "How much was NVIDIA's {m} {p}?",
    "Report NVIDIA's {m} {p}.",
    "{M} {p}?",
]
FORMULA_TEMPLATES = ["What is NVIDIA's {m} {p}?", "Compute the {m} {p}."]

DIRECTIONS = [("increase", "decrease"), ("grow", "decline"), ("rise", "fall"), ("go up", "drop")]
DIRECTION_TEMPLATES = [
    "Why did NVIDIA's {m} {d} in fiscal 2026?",
    "What caused {m} to {d} in FY2026?",
    "Explain the {d} in NVIDIA's {m} during fiscal 2026.",
]
DIRECTION_METRICS = [
    "gross margin",
    "inventories",
    "revenue",
    "operating expenses",
    "net income",
    "cash and cash equivalents",
]

NEGATION_PAIRS = [
    ("Did NVIDIA's {m} increase in fiscal 2026?", "Did NVIDIA's {m} not increase in fiscal 2026?"),
    (
        "Does the report disclose {m} for fiscal 2026?",
        "Does the report not disclose {m} for fiscal 2026?",
    ),
    (
        "Is {m} included in the balance sheet as of January 25, 2026?",
        "Is {m} not included in the balance sheet as of January 25, 2026?",
    ),
    ("Did {m} change year over year?", "Did {m} never change year over year?"),
]
NEGATION_METRICS = [
    "total assets",
    "inventories",
    "goodwill",
    "long-term debt",
    "revenue",
    "dividends paid",
    "net income",
    "cash and cash equivalents",
]

# Same slots (entity, period, metric, direction, negation), different question. These pass the
# hard guard by construction, so they are the pairs the similarity threshold must reject.
SAME_SLOT_TEMPLATES = [
    ("What were NVIDIA's {m} {p}?", "How does NVIDIA account for {m} {p}?"),
    ("What were NVIDIA's {m} {p}?", "Which note to the financial statements discusses {m} {p}?"),
    ("What were NVIDIA's {m} {p}?", "What risks does NVIDIA disclose about {m} {p}?"),
    ("How much was NVIDIA's {m} {p}?", "What assumptions underlie NVIDIA's {m} {p}?"),
]
SAME_SLOT_METRICS = [
    "total assets",
    "inventories",
    "goodwill",
    "revenue",
    "long-term debt",
    "cash and cash equivalents",
    "accounts receivable",
    "net income",
    "dividends paid",
    "marketable securities",
]
SAME_SLOT_PERIODS = ["as of January 25, 2026", "in fiscal 2026"]
SAME_SLOT_EXTRA = [
    ("When is NVIDIA's annual meeting?", "Where is NVIDIA's annual meeting held?"),
    ("When is NVIDIA's annual meeting?", "Who may vote at NVIDIA's annual meeting?"),
    (
        "What are the layers in NVIDIA's five-layer cake?",
        "Who created NVIDIA's five-layer cake figure?",
    ),
    (
        "What drove the H20 charge in fiscal 2026?",
        "How was the H20 charge in fiscal 2026 accounted for?",
    ),
    (
        "What was NVIDIA's revenue in fiscal 2026?",
        "By what percentage did NVIDIA's revenue change in fiscal 2026?",
    ),
    (
        "What were inventories as of January 25, 2026?",
        "What were inventories as a share of total current assets as of January 25, 2026?",
    ),
    (
        "What was net income in fiscal 2026?",
        "How did net income compare with the prior year in fiscal 2026?",
    ),
    (
        "What were NVIDIA's total assets as of Jan 25, 2026?",
        "What were NVIDIA's total assets in fiscal 2026 versus the prior year?",
    ),
    (
        "How much long-term debt did NVIDIA have at the end of fiscal 2026?",
        "What was the change in NVIDIA's long-term debt in fiscal 2026?",
    ),
    (
        "Why did gross margin decrease in fiscal 2026?",
        "By how much did gross margin decrease in fiscal 2026?",
    ),
]

PARAPHRASE_GROUPS = {
    ("total assets", "FY2026"): [
        "What were NVIDIA's total assets as of Jan 25, 2026?",
        "Total assets at the end of fiscal 2026?",
        "How much were NVIDIA's total assets at fiscal year-end 2026?",
        "Report total assets for fiscal 2026.",
        "What was the total asset balance as of January 25, 2026?",
    ],
    ("inventories", "FY2026"): [
        "What were inventories as of January 25, 2026?",
        "How much inventory did NVIDIA hold at the end of fiscal 2026?",
        "Inventory balance at FY2026 year-end?",
        "What was NVIDIA's inventory as of Jan 25, 2026?",
    ],
    ("cash", "FY2026"): [
        "What were cash and cash equivalents as of January 25, 2026?",
        "How much cash and equivalents did NVIDIA have at the end of fiscal 2026?",
        "Cash and cash equivalents at FY2026 year-end?",
    ],
    ("goodwill", "FY2026"): [
        "What was goodwill as of January 25, 2026?",
        "How much goodwill did NVIDIA carry at the end of fiscal 2026?",
        "Goodwill balance at fiscal year-end 2026?",
    ],
    ("current ratio", "FY2026"): [
        "What is the current ratio as of Jan 25, 2026?",
        "Compute NVIDIA's current ratio at the end of fiscal 2026.",
        "What was the current ratio for fiscal 2026?",
        "Current ratio at FY2026 year-end?",
    ],
    ("revenue", "FY2026"): [
        "What was NVIDIA's revenue in fiscal 2026?",
        "How much revenue did NVIDIA report for fiscal year 2026?",
        "Total revenue for FY2026?",
    ],
    ("long-term debt", "FY2026"): [
        "What was NVIDIA's long-term debt as of January 25, 2026?",
        "How much long-term debt did NVIDIA have at the end of fiscal 2026?",
        "Long-term debt at FY2026 year-end?",
    ],
    ("gross margin decrease", "FY2026"): [
        "Why did NVIDIA's gross margin decrease in fiscal 2026?",
        "What caused the decline in NVIDIA's gross margin in FY2026?",
        "Explain the fall in gross margin during fiscal 2026.",
    ],
    ("H20 charge", "FY2026"): [
        "What drove the first-quarter fiscal 2026 charge related to H20?",
        "Why did NVIDIA record an H20-related charge in Q1 fiscal 2026?",
        "Explain the H20 charge taken in the first quarter of fiscal 2026.",
    ],
    ("annual meeting", "-"): [
        "When is NVIDIA's annual meeting?",
        "When is NVIDIA's 2026 annual meeting of stockholders?",
        "What is the date of the 2026 annual meeting of stockholders?",
    ],
    ("five-layer cake", "-"): [
        "What are the layers in NVIDIA's five-layer cake?",
        "Describe the five layers of NVIDIA's AI industry cake framing.",
        "What does the five-layer cake figure show?",
    ],
    ("total return", "-"): [
        "How did NVIDIA's 5-year total return compare to the S&P 500 and Nasdaq 100?",
        "How did NVIDIA's five-year cumulative total return compare with the S&P 500 and the Nasdaq 100?",
        "Compare NVIDIA's 5-year cumulative stock return with the S&P 500 and Nasdaq 100.",
    ],
    ("out of scope", "-"): [
        "Should I buy NVIDIA stock?",
        "Is NVDA a good buy right now?",
        "Would you recommend buying NVIDIA shares?",
    ],
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def adversarial() -> list[dict]:
    rows: list[dict] = []
    n = 0

    def add(kind: str, a: str, b: str, note: str = "") -> None:
        nonlocal n
        n += 1
        rows.append(
            {"id": f"A{n:03d}", "kind": kind, "a": a, "b": b, "expect": "miss", "note": note}
        )

    # period swaps: every metric × every period pair with one template each
    for i, (m, _) in enumerate(METRICS):
        for j, (p1, p2) in enumerate(PERIODS):
            t = LOOKUP_TEMPLATES[(i + j) % len(LOOKUP_TEMPLATES)]
            add(
                "period_swap",
                t.format(m=m, M=_cap(m), p=p1),
                t.format(m=m, M=_cap(m), p=p2),
                f"{p1} vs {p2}",
            )
    for m, _ in FORMULAS:
        for p1, p2 in PERIODS[:3]:
            t = FORMULA_TEMPLATES[0]
            add("period_swap", t.format(m=m, p=p1), t.format(m=m, p=p2), f"{p1} vs {p2}")
    # metric swaps: same template and period, sibling metric
    for i, (m1, m2) in enumerate(METRICS):
        for j, (p, _) in enumerate(PERIODS):
            t = LOOKUP_TEMPLATES[(i + j) % len(LOOKUP_TEMPLATES)]
            add(
                "metric_swap",
                t.format(m=m1, M=_cap(m1), p=p),
                t.format(m=m2, M=_cap(m2), p=p),
                f"{m1} vs {m2}",
            )
    for m1, m2 in FORMULAS:
        for p, _ in PERIODS[:3]:
            add(
                "metric_swap",
                FORMULA_TEMPLATES[0].format(m=m1, p=p),
                FORMULA_TEMPLATES[0].format(m=m2, p=p),
                f"{m1} vs {m2}",
            )
    # direction swaps
    for m in DIRECTION_METRICS:
        for (d1, d2), t in itertools.product(DIRECTIONS, DIRECTION_TEMPLATES):
            add("direction_swap", t.format(m=m, d=d1), t.format(m=m, d=d2), f"{d1} vs {d2}")
    # negations
    for m in NEGATION_METRICS:
        for a, b in NEGATION_PAIRS:
            add("negation", a.format(m=m), b.format(m=m))
    # same slots, different question (guard passes → threshold must reject)
    for i, m in enumerate(SAME_SLOT_METRICS):
        for j, (a, b) in enumerate(SAME_SLOT_TEMPLATES):
            p = SAME_SLOT_PERIODS[(i + j) % len(SAME_SLOT_PERIODS)]
            add("same_slot_other_question", a.format(m=m, p=p), b.format(m=m, p=p))
    for a, b in SAME_SLOT_EXTRA:
        add("same_slot_other_question", a, b)
    return rows


def paraphrases() -> list[dict]:
    rows: list[dict] = []
    n = 0
    for (topic, period), qs in PARAPHRASE_GROUPS.items():
        for a, b in itertools.combinations(qs, 2):
            n += 1
            rows.append(
                {
                    "id": f"P{n:03d}",
                    "kind": "paraphrase",
                    "topic": topic,
                    "period": period,
                    "a": a,
                    "b": b,
                    "expect": "hit",
                }
            )
    return rows


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    utf8_console()
    adv, par = adversarial(), paraphrases()
    write(OUT_DIR / "adversarial_cache_pairs.jsonl", adv)
    write(OUT_DIR / "paraphrase_pairs.jsonl", par)
    kinds: dict[str, int] = {}
    for r in adv:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"adversarial: {len(adv)} {kinds}")
    print(f"paraphrase: {len(par)} pairs from {len(PARAPHRASE_GROUPS)} groups")


if __name__ == "__main__":
    main()
