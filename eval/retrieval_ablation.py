"""Retrieval ablation (plan Phase 4; architecture §12, D-22/D-23/D-24).

Arms, each adding one component on top of the previous one, scored on the golden set's
labelled supporting chunks (recall@8, MRR, hit rate) with no generation:

    dense            dense only, the question as the single query
    +bm25            hybrid dense ⊕ BM25 → RRF (the Phase 3 baseline)
    +expansion-nf    + rule-based glossary / decomposition queries, no metadata filters
    +expansion       + statement / modality pre-filters on the expansion queries
    +llm             + one `small` call for EXPLANATORY (HyDE, paraphrases) and unsure intents
    +rerank          + gated cross-encoder rerank (S1–S3)
    rerank-always    the gate bypassed: rerank every question (shows what the gate protects)

`+llm` needs `--llm` (Groq calls on the `small` role, ~1.5K tokens per affected question); the
other arms are free. Exact match is not measured here (it needs generation); `--golden-run NAME`
pulls it from a saved `eval/results/NAME.jsonl` for the report.

    .venv/Scripts/python eval/retrieval_ablation.py                 # free arms
    .venv/Scripts/python eval/retrieval_ablation.py --llm           # + the small-model arm
    .venv/Scripts/python eval/retrieval_ablation.py --alt-reranker  # + MiniLM-L-6 timing arm
    .venv/Scripts/python eval/retrieval_ablation.py --report docs/reports/retrieval_ablation.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import RESULTS_DIR, load_golden, retrieval_metrics  # noqa: E402

K = 8


# --------------------------------------------------------------------------------- arms


def run_arms(
    questions: list[dict[str, Any]],
    *,
    use_llm: bool,
    alt_reranker: bool,
    log_fn=print,
) -> dict[str, Any]:
    from rag.core.config import load_formulas_config, load_thresholds_config
    from rag.core.ledger import UsageLedger
    from rag.core.settings import get_settings
    from rag.query.analyze import ExpansionSettings, analyze
    from rag.query.rerank import Reranker, get_reranker, rerank_node
    from rag.query.retrieve import RetrievalSettings, hybrid_retrieve
    from rag.query.slots import get_slot_extractor
    from rag.query.store import get_store

    store = get_store()
    th = load_thresholds_config()
    extractor = get_slot_extractor()
    formulas = load_formulas_config()
    settings = get_settings()
    client = None
    ledger = UsageLedger(settings.logs_dir / "llm_usage.jsonl")
    if use_llm:
        from rag.llm import LLMClient

        client = LLMClient(settings)

    base = RetrievalSettings(final_k=K, pool_k=th.rerank.candidates)
    # The ablation measures the component, so it is switched on here whatever the production
    # default in thresholds.yaml says (D-24 turned it off after this measurement).
    rerank_th = th.rerank.model_copy(update={"enabled": True})
    no_expansion = ExpansionSettings(
        glossary_enabled=False, llm_fallback_enabled=False, hyde_enabled=False
    )
    exp_nf = ExpansionSettings(
        filters_enabled=False, llm_fallback_enabled=False, hyde_enabled=False
    )
    exp = ExpansionSettings(llm_fallback_enabled=False, hyde_enabled=False)
    exp_llm = ExpansionSettings()

    def required(slots, intent: str) -> list[str]:  # noqa: ANN001
        if intent == "COMPUTATION" and slots.formulas:
            items: list[str] = []
            for fid in slots.formulas:
                f = formulas.formulas.get(fid)
                for spec in f.inputs.values() if f else []:
                    if "{metric}" not in spec.line_item and spec.line_item not in items:
                        items.append(spec.line_item)
            return items
        return list(slots.metrics)

    arms: list[str] = ["dense", "+bm25", "+expansion-nf", "+expansion"]
    if use_llm:
        arms.append("+llm")
    arms += ["+rerank", "rerank-always"]
    if alt_reranker:
        arms.append("rerank-minilm")
    alt = Reranker(th.rerank.lightweight_alternative or "Xenova/ms-marco-MiniLM-L-6-v2")
    main_rr = get_reranker(th.rerank.model)

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calibration: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        if q["intent"] == "OUT_OF_SCOPE":
            continue
        gold = q["supporting_chunk_ids"]
        slots = extractor.extract(q["question"])
        a_rule = analyze(
            q["question"], slots, settings=no_expansion, extractor=extractor, formulas=formulas
        )
        intent = a_rule.intent
        req = required(slots, intent)

        def score(  # noqa: ANN001
            arm: str,
            result,
            ms: float,
            calls: int = 0,
            extra: dict | None = None,
            *,
            _q=q,
            _gold=gold,
            _intent=intent,
        ) -> None:
            m = retrieval_metrics(result.top_ids, _gold, K)
            rows[arm].append(
                {
                    "id": _q["id"],
                    "intent": _q["intent"],
                    "intent_pred": _intent,
                    **m,
                    "latency_ms": round(ms, 1),
                    "groq_calls": calls,
                    "n_queries": len(result.queries),
                    "top_ids": result.top_ids[:K],
                    **(extra or {}),
                }
            )

        t0 = time.perf_counter()
        r = hybrid_retrieve(
            store,
            [q["question"]],
            settings=RetrievalSettings(final_k=K, sparse=False),
            intent=intent,
        )
        score("dense", r, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        r = hybrid_retrieve(store, [q["question"]], settings=base, intent=intent)
        score("+bm25", r, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        a = analyze(q["question"], slots, settings=exp_nf, extractor=extractor, formulas=formulas)
        r = hybrid_retrieve(store, a.queries, settings=base, intent=a.intent)
        score("+expansion-nf", r, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        a = analyze(q["question"], slots, settings=exp, extractor=extractor, formulas=formulas)
        r_exp = hybrid_retrieve(store, a.queries, settings=base, intent=a.intent)
        exp_ms = (time.perf_counter() - t0) * 1000
        score("+expansion", r_exp, exp_ms, extra={"filtered_retries": r_exp.filtered_retries})

        r_before_rerank = r_exp
        if use_llm:
            t0 = time.perf_counter()
            before = ledger.count()
            a_llm = analyze(
                q["question"],
                slots,
                client=client,
                request_id=f"ablation-{q['id']}",
                settings=exp_llm,
                extractor=extractor,
                formulas=formulas,
            )
            r_llm = hybrid_retrieve(store, a_llm.queries, settings=base, intent=a_llm.intent)
            calls = sum(
                1 for rec in ledger.read_all()[before:] if rec.request_id == f"ablation-{q['id']}"
            )
            score(
                "+llm",
                r_llm,
                (time.perf_counter() - t0) * 1000,
                calls,
                {
                    "llm_called": a_llm.llm_called,
                    "hyde": bool(a_llm.hyde_passage),
                    "hyde_rejected": a_llm.hyde_rejected,
                    "kinds": [x.kind for x in a_llm.queries],
                },
            )
            r_before_rerank = r_llm

        t0 = time.perf_counter()
        r_rr, dec = rerank_node(
            store,
            r_before_rerank,
            q["question"],
            intent=intent,
            required_metrics=req,
            thresholds=rerank_th,
            final_k=K,
            reranker=main_rr,
        )
        score(
            "+rerank",
            r_rr,
            exp_ms + (time.perf_counter() - t0) * 1000,
            extra={"gate": dec.gate, "applied": dec.applied},
        )

        t0 = time.perf_counter()
        r_force, dec_f = rerank_node(
            store,
            r_before_rerank,
            q["question"],
            intent=intent,
            required_metrics=req,
            thresholds=rerank_th,
            final_k=K,
            reranker=main_rr,
            force=True,
        )
        score(
            "rerank-always",
            r_force,
            exp_ms + (time.perf_counter() - t0) * 1000,
            extra={"dropped": dec_f.dropped},
        )
        for c in r_force.pool:
            calibration.append(
                {
                    "id": q["id"],
                    "chunk_id": c.chunk_id,
                    "gold": c.chunk_id in gold,
                    "score": c.rerank_score,
                }
            )

        if alt_reranker:
            t0 = time.perf_counter()
            r_alt, _ = rerank_node(
                store,
                r_before_rerank,
                q["question"],
                intent=intent,
                required_metrics=req,
                thresholds=rerank_th,
                final_k=K,
                reranker=alt,
                force=True,
            )
            score("rerank-minilm", r_alt, exp_ms + (time.perf_counter() - t0) * 1000)

        line = " ".join(f"{arm}={rows[arm][-1]['recall_at_k']:.2f}" for arm in arms)
        log_fn(
            f"[{i}/{len(questions)}] {q['id']:4s} {intent[:6]} {line} gate={dec.gate or 'rerank'}"
        )

    return {
        "arms": arms,
        "rows": dict(rows),
        "calibration": calibration,
        "rerank_model": th.rerank.model,
        "drop_floor_logit": th.rerank.drop_floor_logit,
        "corpus_version": store.corpus_version,
    }


# ------------------------------------------------------------------------------ summary


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(out: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in out["arms"]:
        rs = out["rows"][arm]
        by_intent: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_intent[r["intent"]].append(r)
        summary[arm] = {
            "n": len(rs),
            "recall_at_8": _mean([r["recall_at_k"] for r in rs]),
            "mrr": _mean([r["mrr"] for r in rs]),
            "hit_rate": _mean([float(r["hit"]) for r in rs]),
            "latency_ms_mean": _mean([r["latency_ms"] for r in rs]),
            "latency_ms_p50": round(statistics.median([r["latency_ms"] for r in rs]), 1),
            "groq_calls_per_query": _mean([r["groq_calls"] for r in rs]),
            "queries_per_question": _mean([r["n_queries"] for r in rs]),
            "reranked": sum(1 for r in rs if r.get("applied")),
            "by_intent": {
                k: _mean([r["recall_at_k"] for r in v]) for k, v in sorted(by_intent.items())
            },
        }
    cal = out["calibration"]
    gold = sorted(c["score"] for c in cal if c["gold"] and c["score"] is not None)
    other = sorted(c["score"] for c in cal if not c["gold"] and c["score"] is not None)

    def pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        return round(vals[min(len(vals) - 1, int(p * (len(vals) - 1)))], 2)

    summary["_calibration"] = {
        "gold_n": len(gold),
        "gold_p10": pct(gold, 0.1),
        "gold_p50": pct(gold, 0.5),
        "gold_min": round(gold[0], 2) if gold else None,
        "other_n": len(other),
        "other_p50": pct(other, 0.5),
        "other_p90": pct(other, 0.9),
        "gold_below_floor": sum(1 for s in gold if s < out["drop_floor_logit"]),
        "other_below_floor": sum(1 for s in other if s < out["drop_floor_logit"]),
        "drop_floor_logit": out["drop_floor_logit"],
    }
    return summary


# ------------------------------------------------------------------------------- report


def exact_match_from_run(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    path = RESULTS_DIR / f"{name}.summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    out: dict[str, Any], summary: dict[str, Any], path: Path, golden_run: str | None
) -> None:
    arms = out["arms"]
    intents = sorted({r["intent"] for rs in out["rows"].values() for r in rs})
    lines = [
        "# Retrieval ablation — Phase 4",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='minutes')} · corpus `{out['corpus_version']}` · "
        f"reranker `{out['rerank_model']}` · k = {K} · {summary[arms[0]]['n']} in-scope golden questions "
        "(OUT_OF_SCOPE excluded: the scope gate refuses them before retrieval).",
        "",
        "Each arm adds one component on top of the previous one (architecture §4.4–4.6). Scores are",
        "against the golden set's labelled supporting chunks; no answers are generated here.",
        "",
        "| Arm | recall@8 | MRR | hit rate | queries / q | Groq calls / q | latency p50 (ms) | reranked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        s = summary[arm]
        lines.append(
            f"| `{arm}` | **{s['recall_at_8']:.2f}** | {s['mrr']:.2f} | {s['hit_rate']:.2f} | "
            f"{s['queries_per_question']:.1f} | {s['groq_calls_per_query']:.2f} | {s['latency_ms_p50']:.0f} | {s['reranked']} |"
        )
    lines += [
        "",
        "## recall@8 by intent",
        "",
        "| Arm | " + " | ".join(intents) + " |",
        "|---|" + "---|" * len(intents),
    ]
    for arm in arms:
        bi = summary[arm]["by_intent"]
        lines.append(
            f"| `{arm}` | " + " | ".join(f"{bi.get(i, 0) or 0:.2f}" for i in intents) + " |"
        )

    lines += [
        "",
        "## Per-question recall@8",
        "",
        "| id | intent | " + " | ".join(f"`{a}`" for a in arms) + " | gate |",
        "|---|---|" + "---|" * (len(arms) + 1),
    ]
    ids = [r["id"] for r in out["rows"][arms[0]]]
    for idx, qid in enumerate(ids):
        cells = []
        for arm in arms:
            r = out["rows"][arm][idx]
            cells.append(f"{r['recall_at_k']:.2f}")
        gate = out["rows"]["+rerank"][idx].get("gate") or "rerank"
        lines.append(
            f"| {qid} | {out['rows'][arms[0]][idx]['intent']} | "
            + " | ".join(cells)
            + f" | {gate} |"
        )

    cal = summary["_calibration"]
    lines += [
        "",
        "## Reranker score calibration (`rerank-always` arm, all pool candidates)",
        "",
        f"- gold chunks: n = {cal['gold_n']}, min {cal['gold_min']}, p10 {cal['gold_p10']}, median {cal['gold_p50']}",
        f"- other chunks: n = {cal['other_n']}, median {cal['other_p50']}, p90 {cal['other_p90']}",
        f"- `drop_floor_logit` = {cal['drop_floor_logit']}: {cal['gold_below_floor']} gold and "
        f"{cal['other_below_floor']} other candidates fall below it",
    ]
    if "+llm" in out["rows"]:
        llm_rows = out["rows"]["+llm"]
        called = [r for r in llm_rows if r.get("llm_called")]
        lines += [
            "",
            "## Small-model arm",
            "",
            f"- questions with a model call: {len(called)} of {len(llm_rows)} "
            f"(rule-confident questions add zero calls, plan Phase 4)",
            f"- HyDE passages used: {sum(1 for r in called if r.get('hyde'))}; rejected for digits: "
            f"{sum(1 for r in called if r.get('hyde_rejected'))}",
        ]
    em = exact_match_from_run(golden_run)
    if em:
        o = em["overall"]
        lines += [
            "",
            f"## Answer quality from the full run `{golden_run}`",
            "",
            f"- numeric exact match: **{o['exact_match']:.0%}** ({o['numeric_n']} questions) · calculator agrees "
            f"{o['calculator_agrees']:.0%} · verifier pass rate {o['verify_pass_rate']:.0%}",
            f"- recall@8 / MRR inside the full pipeline: {o['recall_at_k']:.2f} / {o['mrr']:.2f}",
            f"- tokens: {o['tokens_in']:,} in / {o['tokens_out']:,} out",
        ]
    e, r, ra = summary["+expansion"], summary.get("+rerank"), summary["rerank-always"]
    lines += [
        "",
        "## Reading the numbers (decisions recorded in architecture §16)",
        "",
        f"- **Expansion is the win** (D-57): phrasing each metric / formula input the way its row-fact "
        f"document reads, filtered to the statement, takes recall@8 from {summary['+bm25']['recall_at_8']:.2f} "
        f"to {e['recall_at_8']:.2f} and MRR from {summary['+bm25']['mrr']:.2f} to {e['mrr']:.2f} with zero "
        "model calls. The filter itself is worth "
        f"+{e['recall_at_8'] - summary['+expansion-nf']['recall_at_8']:.2f} recall / "
        f"+{e['mrr'] - summary['+expansion-nf']['mrr']:.2f} MRR over unfiltered expansion.",
        "- **HyDE / paraphrases** (D-23): "
        + (
            f"no measurable change ({summary['+llm']['recall_at_8']:.2f} vs {e['recall_at_8']:.2f}) at "
            f"{summary['+llm']['groq_calls_per_query']:.2f} `small` calls per question — EXPLANATORY recall is "
            "already 1.00 after hybrid retrieval. Off by default (`expansion.hyde: false`)."
            if "+llm" in summary
            else "not measured in this run (`--llm`)."
        ),
        "- **Reranking** (D-24): "
        + (
            f"gated reranking lowers recall@8 to {r['recall_at_8']:.2f} (ungated {ra['recall_at_8']:.2f}, "
            f"MiniLM {summary.get('rerank-minilm', ra)['recall_at_8']:.2f}) and costs ≈ {ra['latency_ms_p50'] / 1000:.1f} s "
            "per reranked question on this CPU; the S1–S3 gate protects the numeric questions but the "
            "narrative ones it does rerank lose labelled chunks. Off by default (`rerank.enabled: false`); "
            "the gate, the reranker and this arm stay so a larger corpus can re-test."
            if r
            else "not measured."
        ),
    ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--llm", action="store_true", help="include the small-model expansion arm")
    ap.add_argument("--alt-reranker", action="store_true", help="time the MiniLM alternative")
    ap.add_argument("--ids", help="comma-separated golden ids")
    ap.add_argument("--name", default="retrieval_ablation")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--golden-run", default=None, help="results name to quote exact match from")
    ap.add_argument(
        "--from-results", action="store_true", help="rewrite the report from saved results only"
    )
    args = ap.parse_args(argv)
    if args.from_results:
        out = json.loads((RESULTS_DIR / f"{args.name}.json").read_text(encoding="utf-8"))
        write_report(
            out,
            summarize(out),
            args.report or Path("docs/reports/retrieval_ablation.md"),
            args.golden_run,
        )
        return 0

    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings

    configure_logging("WARNING", secrets=get_settings().secret_values())
    questions = load_golden()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    out = run_arms(questions, use_llm=args.llm, alt_reranker=args.alt_reranker)
    summary = summarize(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS_DIR / f"{args.name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for arm in out["arms"]:
        s = summary[arm]
        print(
            f"{arm:15s} recall@8={s['recall_at_8']:.3f} mrr={s['mrr']:.3f} hit={s['hit_rate']:.2f} "
            f"p50={s['latency_ms_p50']:.0f}ms calls/q={s['groq_calls_per_query']:.2f} reranked={s['reranked']}"
        )
    print("calibration:", json.dumps(summary["_calibration"]))
    if args.report:
        write_report(out, summary, args.report, args.golden_run)
        print("report:", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
