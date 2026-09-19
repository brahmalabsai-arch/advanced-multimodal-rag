"""Embedder gate (plan Phase 3, D-47): bge-small vs bge-base on retrieval metrics, no LLM calls.

Tie-break rule: prefer `bge-small` unless `bge-base` improves recall@8 by >= 3 points.
Writes docs/reports/embedder_gate.md and, with --lock, sets `retrieval.embedder` in
config/thresholds.yaml to the winner.

    .venv/Scripts/python eval/embedder_gate.py [--lock]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import evaluate, load_golden  # noqa: E402

CANDIDATES = {
    "bge-small": PROJECT_ROOT / "data" / "index",
    "bge-base": PROJECT_ROOT / "data" / "index_bge_base",
}
MARGIN_POINTS = 3.0
REPORT = PROJECT_ROOT / "docs" / "reports" / "embedder_gate.md"
THRESHOLDS = PROJECT_ROOT / "config" / "thresholds.yaml"


def decide(small: dict, base: dict) -> tuple[str, str]:
    s = (small["overall"]["recall_at_k"] or 0) * 100
    b = (base["overall"]["recall_at_k"] or 0) * 100
    if b - s >= MARGIN_POINTS:
        return "bge-base", f"bge-base improves recall@8 by {b - s:.1f} points (>= {MARGIN_POINTS})"
    return (
        "bge-small",
        f"bge-base gain is {b - s:+.1f} points (< {MARGIN_POINTS}); keep the smaller, faster model",
    )


def render(results: dict[str, dict], winner: str, reason: str, k: int) -> str:
    lines = ["# Embedder gate — bge-small vs bge-base (Phase 3)\n"]
    lines.append(
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · retrieval-only, dense top-30 ⊕ BM25 top-30 → RRF → top-{k} · golden set n={results['bge-small']['overall']['n']}\n"
    )
    lines.append(
        f"**Decision: `{winner}`** — {reason}. Rule (D-47): prefer bge-small unless bge-base improves recall@{k} by ≥ {MARGIN_POINTS:.0f} points.\n"
    )
    lines.append("## Overall\n")
    lines.append(
        f"| Embedder | Dim | corpus_version | recall@{k} | MRR | hit rate | elapsed s |\n|---|---|---|---|---|---|---|"
    )
    for alias, r in results.items():
        o = r["overall"]
        dim = 384 if alias == "bge-small" else 768
        lines.append(
            f"| {alias} | {dim} | `{r['corpus_version']}` | {o['recall_at_k']:.3f} | {o['mrr']:.3f} | {o['hit_rate']:.3f} | {r['elapsed_s']} |"
        )
    lines.append("")
    lines.append("## By intent\n")
    intents = sorted({i for r in results.values() for i in r["by_intent"]})
    lines.append("| Intent | n | " + " | ".join(f"{a} recall@{k} / MRR" for a in results) + " |")
    lines.append("|---|---|" + "---|" * len(results))
    for it in intents:
        cells = []
        n = 0
        for r in results.values():
            b = r["by_intent"].get(it)
            if b:
                n = b["n"]
                cells.append(f"{b['recall_at_k']:.2f} / {b['mrr']:.2f}")
            else:
                cells.append("–")
        lines.append(f"| {it} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- Chunk boundaries are identical for both indexes (D-53); only the vectors differ, so this is a pure embedder comparison."
    )
    lines.append(
        "- Retrieval here is the Phase 3 baseline (single query, no expansion, no filters, no reranking). Weak spots on COMPUTATION / COMPARISON_TREND questions are addressed in Phase 4 and measured in the retrieval ablation."
    )
    lines.append(
        "- Thresholds calibrated later (rerank floor, cache similarity) are specific to the locked embedder."
    )
    return "\n".join(lines) + "\n"


def lock(winner: str) -> None:
    text = THRESHOLDS.read_text(encoding="utf-8")
    new = re.sub(r"(retrieval:\n\s+embedder: )\S+", rf"\g<1>{winner}", text, count=1)
    THRESHOLDS.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lock", action="store_true", help="write the winner into config/thresholds.yaml"
    )
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args(argv)

    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings

    configure_logging("WARNING", secrets=get_settings().secret_values())
    questions = load_golden()
    results = {}
    for alias, index_dir in CANDIDATES.items():
        if not (index_dir / "manifest.json").exists():
            print(f"missing index for {alias}: {index_dir} (run `make ingest` / `make index-base`)")
            return 1
        results[alias] = evaluate(
            questions,
            index_dir=index_dir,
            retrieval_only=True,
            k=args.k,
            name=f"gate_{alias.replace('-', '_')}",
            log_fn=lambda _m: None,
        )
        print(alias, json.dumps(results[alias]["overall"]))
    winner, reason = decide(results["bge-small"], results["bge-base"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(results, winner, reason, args.k), encoding="utf-8")
    print(f"DECISION: {winner} — {reason}\nreport: {REPORT}")
    if args.lock:
        lock(winner)
        print(f"locked retrieval.embedder = {winner} in {THRESHOLDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
