"""Render an evaluation run (eval/results/<name>.jsonl + .summary.json) as a Markdown report.

    .venv/Scripts/python eval/write_report.py baseline_phase3 --title "Baseline — Phase 3" \
        --out docs/reports/baseline_phase3.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _f(v: float | None, nd: int = 2) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def render(name: str, title: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    o = summary["overall"]
    lines = [f"# {title}\n"]
    lines.append(
        f"Run `{name}` · {summary['generated_at']} · index `{Path(summary['index_dir']).name}` "
        f"({summary['embedder']}, corpus `{summary['corpus_version']}`) · k={summary['k']} · "
        f"{summary.get('completed', len(rows))} questions · {summary['elapsed_s']} s · "
        f"bypass_cache=true\n"
    )
    if summary.get("stopped"):
        lines.append(f"> Run stopped early: {summary['stopped']}\n")
    lines.append("## Headline\n")
    lines.append("| Metric | Value |\n|---|---|")
    lines.append(
        f"| Numeric exact match (value + period + unit) | {_pct(o['exact_match'])} of {o['numeric_n']} |"
    )
    lines.append(f"| Value found (ignoring period/unit wording) | {_pct(o['value_found'])} |")
    lines.append(f"| Calculator agrees with golden value | {_pct(o['calculator_agrees'])} |")
    lines.append(
        f"| Keyword coverage (text questions, indicative) | {_pct(o['keyword_coverage'])} of {o['text_n']} |"
    )
    lines.append(f"| Verifier pass rate | {_pct(o['verify_pass_rate'])} |")
    lines.append(
        f"| recall@{summary['k']} / MRR / hit rate | {_f(o['recall_at_k'])} / {_f(o['mrr'])} / {_f(o['hit_rate'])} |"
    )
    lines.append(
        f"| Latency p50 | {o['latency_ms_p50']} ms (includes client-side pacing waits under the "
        f"Groq free-tier TPM limit; NFR-5 excludes those) |"
    )
    lines.append(f"| Groq tokens in / out | {o['tokens_in']:,} / {o['tokens_out']:,} |")
    lines.append("")
    lines.append("## By intent\n")
    lines.append(
        f"| Intent | n | exact | value | calc | kw | verify | recall@{summary['k']} | MRR | p50 ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for intent, b in summary["by_intent"].items():
        lines.append(
            f"| {intent} | {b['n']} | {_pct(b['exact_match'])} | {_pct(b['value_found'])} | "
            f"{_pct(b['calculator_agrees'])} | {_pct(b['keyword_coverage'])} | {_pct(b['verify_pass_rate'])} | "
            f"{_f(b['recall_at_k'])} | {_f(b['mrr'])} | {b['latency_ms_p50']} |"
        )
    lines.append("")
    lines.append("## Per question\n")
    lines.append(
        "| id | intent | exact / kw | value | period | unit | calc | verify | recall | attempts | ms | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("skipped"):
            lines.append(
                f"| {r['id']} | {r['intent']} | skipped | | | | | | | | | {r['skipped']} |"
            )
            continue
        score = (
            ("✓" if r.get("exact_match") else "✗")
            if "exact_match" in r
            else (_pct(r.get("keyword_coverage")) if r.get("keyword_coverage") is not None else "–")
        )

        def flag(key: str, row: dict[str, Any] = r) -> str:
            return "✓" if row.get(key) else ("✗" if row.get(key) is False else "")

        note = (r.get("warning") or "")[:60]
        lines.append(
            f"| {r['id']} | {r['intent']} | {score} | {flag('value_found')} | {flag('period_stated')} | "
            f"{flag('unit_ok')} | {flag('calculator_agrees')} | {flag('verify_passed')} | "
            f"{_f(r.get('recall_at_k'))} | {r.get('attempts', '')} | {r.get('latency_ms', '')} | {note} |"
        )
    lines.append("")
    misses = [r for r in rows if r.get("exact_match") is False or r.get("keyword_pass") is False]
    if misses:
        lines.append("## Misses (answer excerpts)\n")
        for r in misses:
            ans = (r.get("answer") or "").replace("\n", " ")[:300]
            lines.append(f"- **{r['id']}** ({r['intent']}): {ans}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = [
        json.loads(ln)
        for ln in (RESULTS_DIR / f"{args.name}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    summary = json.loads((RESULTS_DIR / f"{args.name}.summary.json").read_text(encoding="utf-8"))
    text = render(args.name, args.title or f"Evaluation — {args.name}", summary, rows)
    out = args.out or (PROJECT_ROOT / "docs" / "reports" / f"{args.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
