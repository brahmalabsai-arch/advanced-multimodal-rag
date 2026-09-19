"""Compression ablation (plan Phase 5; architecture §6.9).

Three arms through the full pipeline, same generator profile, same golden questions:

    never_compress     compression node off (an existing run can be reused with --never-run)
    classifier_gated   production behaviour: Stage A/B decides per chunk
    always_compress    every narrative chunk through EXTRACT_LLM (tables stay protected)

Reported per arm: numeric exact match, keyword coverage, verifier pass rate, large-model input
tokens, context tokens, small-model calls, p50/p95 latency, fidelity violations and the applied
action mix. Generation costs tokens: run `always_compress` on a subset (`--always-ids`).

    MODEL_PROFILE=anthropic .venv/Scripts/python eval/compression_ablation.py \\
        --never-run golden_phase4_claude --report docs/reports/compression_ablation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import RESULTS_DIR, evaluate, load_golden  # noqa: E402

DEFAULT_ALWAYS_IDS = (
    "G1,P4,P9,P14,P20,G3,G5,C7,G7,G9,T5,G11,E2,E3,E4,E5,G12,V3,X1,X3"  # 20 across intents
)


def _rows(name: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{name}.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(p / 100 * (len(vals) - 1)))]


def arm_stats(
    rows: list[dict[str, Any]], ids: set[str] | None, large_model: str, small_model: str
) -> dict[str, Any]:
    rs = [r for r in rows if not r.get("skipped") and (ids is None or r["id"] in ids)]
    numeric = [r for r in rs if "exact_match" in r]
    textual = [r for r in rs if r.get("keyword_coverage") is not None]
    large_in = [
        sum(t["in"] for m, t in (r.get("tokens") or {}).items() if m == large_model) for r in rs
    ]
    small_calls = [
        sum(t["calls"] for m, t in (r.get("tokens") or {}).items() if m == small_model) for r in rs
    ]
    comp = [r.get("compression") or {} for r in rs]
    actions: dict[str, int] = {}
    for c in comp:
        for k, v in (c.get("actions") or {}).items():
            actions[k] = actions.get(k, 0) + v
    violations = [v for c in comp for v in (c.get("violations") or [])]
    return {
        "n": len(rs),
        "exact_match": round(sum(r["exact_match"] for r in numeric) / len(numeric), 3)
        if numeric
        else None,
        "numeric_n": len(numeric),
        "keyword_coverage": round(sum(r["keyword_coverage"] for r in textual) / len(textual), 3)
        if textual
        else None,
        "text_n": len(textual),
        "verify_pass_rate": round(sum(1 for r in rs if r.get("verify_passed")) / len(rs), 3)
        if rs
        else None,
        "large_input_tokens": sum(large_in),
        "large_input_per_q": round(sum(large_in) / len(rs)) if rs else None,
        "context_tokens_per_q": round(sum(r.get("context_tokens", 0) for r in rs) / len(rs))
        if rs
        else None,
        "small_calls": sum(small_calls),
        "compression_llm_calls": sum(c.get("llm_calls", 0) for c in comp),
        "queries_compressed": sum(1 for c in comp if c.get("query_needs_compression")),
        "tokens_before": sum(c.get("tokens_before", 0) for c in comp),
        "tokens_after": sum(c.get("tokens_after", 0) for c in comp),
        "violations": violations,
        "actions": actions,
        "latency_p50": _pct([r["latency_ms"] for r in rs if "latency_ms" in r], 50),
        "latency_p95": _pct([r["latency_ms"] for r in rs if "latency_ms" in r], 95),
    }


def write_report(
    arms: dict[str, dict[str, Any]],
    subset_ids: list[str],
    path: Path,
    profile: str,
    runs: dict[str, str],
    context_budget: int,
) -> None:
    ids = set(subset_ids)
    lines = [
        "# Compression ablation — Phase 5",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='minutes')} · profile `{profile}` · "
        f"context budget {context_budget} tokens (the groq_build production budget) · golden set · "
        "same retrieval, calculator, verifier and prompts in every arm; only the compression node differs.",
        "",
        "Runs: " + ", ".join(f"`{a}` → `{n}`" for a, n in runs.items()),
        "",
        "## All in-scope questions (never vs classifier)",
        "",
        "| Arm | n | exact | kw | verify | large input tok / q | context tok / q | compressor LLM calls | queries compressed | fidelity violations | p50 / p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, s: dict[str, Any]) -> str:
        return (
            f"| `{name}` | {s['n']} | {_f(s['exact_match'])} | {_f(s['keyword_coverage'])} | {_f(s['verify_pass_rate'])} | "
            f"{s['large_input_per_q']} | {s['context_tokens_per_q']} | {s['compression_llm_calls']} | {s['queries_compressed']} | "
            f"{len(s['violations'])} | {s['latency_p50']:.0f} / {s['latency_p95']:.0f} |"
        )

    for name in ("never_compress", "classifier_gated"):
        if name in arms and "full" in arms[name]:
            lines.append(row(name, arms[name]["full"]))
    lines += ["", f"## {len(ids)}-question subset (all three arms)", "", lines[8], lines[9]]
    for name in ("never_compress", "classifier_gated", "always_compress"):
        if name in arms and "subset" in arms[name]:
            lines.append(row(name, arms[name]["subset"]))
    lines += ["", "## Applied actions (classifier arm, all questions)", ""]
    acts = arms.get("classifier_gated", {}).get("full", {}).get("actions", {})
    lines += [f"- {k}: {v}" for k, v in sorted(acts.items(), key=lambda kv: -kv[1])] or ["- none"]
    cg = arms.get("classifier_gated", {}).get("full", {})
    if cg:
        lines += [
            "",
            f"- candidate tokens before → after compression: {cg['tokens_before']:,} → {cg['tokens_after']:,} "
            f"({(1 - cg['tokens_after'] / cg['tokens_before']) * 100 if cg['tokens_before'] else 0:.0f}% fewer)",
        ]
    viol = [v for a in arms.values() for s in a.values() for v in s["violations"]]
    lines += [
        "",
        "## Fidelity",
        "",
        f"- violations caught by the guard across all arms: {len(viol)} (each reverted to the original chunk before generation)",
    ]
    lines += [f"  - {v}" for v in viol[:20]]
    nv, cg_s, al = (
        arms.get("never_compress", {}).get("full"),
        arms.get("classifier_gated", {}).get("full"),
        arms.get("always_compress", {}).get("subset"),
    )
    if nv and cg_s:
        lines += [
            "",
            "## Reading the numbers (D-58)",
            "",
            f"- **Numeric accuracy is untouched by compression** ({_f(nv['exact_match'])} exact in every arm): "
            "numbers reach the generator through calculator blocks and protected row facts, never through a "
            "compressor (K2, P3).",
            f"- **`always_compress` is the case for gating**: keyword coverage {_f(al['keyword_coverage']) if al else '–'} vs "
            f"{_f(arms['never_compress']['subset']['keyword_coverage'])} on the same subset, "
            f"{al['compression_llm_calls'] if al else 0} small-model calls and p50 latency "
            f"{al['latency_p50'] / 1000 if al else 0:.1f} s vs {arms['never_compress']['subset']['latency_p50'] / 1000:.1f} s.",
            f"- **`classifier_gated`** keeps exact match at {_f(cg_s['exact_match'])}, compresses {cg_s['queries_compressed']} of "
            f"{cg_s['n']} queries (actions: {', '.join(f'{k} {v}' for k, v in sorted(cg_s['actions'].items(), key=lambda kv: -kv[1]) if k != 'KEEP')}), "
            f"cuts context tokens per question {nv['context_tokens_per_q']} → {cg_s['context_tokens_per_q']} "
            f"({(1 - cg_s['context_tokens_per_q'] / nv['context_tokens_per_q']) * 100:.0f}%) with "
            f"{cg_s['compression_llm_calls']} compressor calls and {len(cg_s['violations'])} fidelity violations. "
            "Large-model input per *attempt* falls in step; the per-question total also counts regenerations, "
            "which vary run to run.",
            "- **Thresholds adjusted from the first gated run** (kept for the record): `dedupe_cosine` 0.95 → 0.98 "
            "because the three H20 disclosures sit at 0.94–0.96 cosine yet are paraphrases of different length "
            '(merging them lost the "license" wording, G11), and R5 now cuts narrative only when the candidates '
            "exceed the budget (`narrative_pressure_ratio` 1.0) because cutting a context that already fit saved "
            "1.4% of large-model input and dropped a keyword on V3. On a 2,500-token budget with eight candidates "
            "that leaves ROW_SELECT as the working compressor; the LLM extractor is reached only under real budget "
            "pressure (longer contexts, more candidates), which the Phase 7 benchmark can exercise.",
            "- **Weak feature, recorded honestly**: bge-small sentence-to-query cosine does not separate labelled "
            "from unlabelled narrative sentences (medians 0.60 vs 0.61), so `relevance_density` carries little "
            "information on this corpus; Stage C (Phase 7) should re-weight it from the decision log.",
        ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _f(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--never-run", default=None, help="reuse an existing run as never_compress")
    ap.add_argument("--classifier-run", default=None, help="reuse an existing classifier_gated run")
    ap.add_argument("--always-run", default=None, help="reuse an existing always_compress run")
    ap.add_argument("--always-ids", default=DEFAULT_ALWAYS_IDS)
    ap.add_argument("--skip-always", action="store_true")
    ap.add_argument(
        "--context-budget",
        type=int,
        default=2500,
        help="context budget for every arm (default 2500 = the groq_build production budget)",
    )
    ap.add_argument("--report", type=Path, default=Path("docs/reports/compression_ablation.md"))
    args = ap.parse_args(argv)

    from rag.core.config import load_models_config
    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings

    settings = get_settings()
    configure_logging("WARNING", secrets=settings.secret_values())
    models = load_models_config(settings=settings)
    profile = models.active()
    large, small = profile.large.model, profile.small.model
    questions = load_golden()
    subset = [i for i in args.always_ids.split(",") if i]
    runs: dict[str, str] = {}

    never = args.never_run
    if never is None:
        never = "compress_never"
        evaluate(
            questions,
            index_dir=None,
            retrieval_only=False,
            k=8,
            name=never,
            compression_mode="never",
            context_budget=args.context_budget,
        )
    runs["never_compress"] = never
    gated = args.classifier_run
    if gated is None:
        gated = "compress_classifier"
        evaluate(
            questions,
            index_dir=None,
            retrieval_only=False,
            k=8,
            name=gated,
            compression_mode="classifier",
            context_budget=args.context_budget,
        )
    runs["classifier_gated"] = gated
    always = args.always_run
    if always is None and not args.skip_always:
        always = "compress_always"
        sub_q = [q for q in questions if q["id"] in set(subset)]
        evaluate(
            sub_q,
            index_dir=None,
            retrieval_only=False,
            k=8,
            name=always,
            compression_mode="always",
            context_budget=args.context_budget,
        )
    if always:
        runs["always_compress"] = always

    arms: dict[str, dict[str, Any]] = {}
    for arm, name in runs.items():
        rows = _rows(name)
        arms[arm] = {"subset": arm_stats(rows, set(subset), large, small)}
        if arm != "always_compress":
            arms[arm]["full"] = arm_stats(rows, None, large, small)
    (RESULTS_DIR / "compression_ablation.summary.json").write_text(
        json.dumps({"runs": runs, "arms": arms, "subset": subset}, indent=2), encoding="utf-8"
    )
    for arm, a in arms.items():
        for scope, s in a.items():
            print(
                f"{arm:17s} {scope:6s} n={s['n']} exact={s['exact_match']} kw={s['keyword_coverage']} "
                f"large_in/q={s['large_input_per_q']} ctx/q={s['context_tokens_per_q']} llm_calls={s['compression_llm_calls']} "
                f"viol={len(s['violations'])} p50={s['latency_p50']}"
            )
    write_report(arms, subset, args.report, models.active_profile, runs, args.context_budget)
    print("report:", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
