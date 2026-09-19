"""Golden-set evaluation (plan Phase 3; architecture §12).

Metrics
- retrieval: recall@k and MRR against the labelled supporting chunk ids (no LLM calls);
- numeric questions: exact match = expected value present in the answer (rounding-tolerant,
  million/billion scale tolerant) AND the expected fiscal period stated AND the unit family
  consistent; calculator agreement is recorded separately;
- text questions: keyword coverage of `expected_keywords` (indicative; human review pending);
- always `bypass_cache=true` and request ids prefixed `eval-` (evaluation hygiene, §12).

    .venv/Scripts/python eval/run_eval.py --retrieval-only              # free, no Groq calls
    .venv/Scripts/python eval/run_eval.py --limit 10                    # full pipeline subset
    .venv/Scripts/python eval/run_eval.py --ids G1,G3,G9 --name probe
    .venv/Scripts/python eval/run_eval.py --index data/index_bge_base --retrieval-only
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

GOLDEN = PROJECT_ROOT / "eval" / "golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
_NUMBER = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\w])")


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------- scoring


def retrieval_metrics(
    top_ids: list[str], labelled: list[str], k: int = 8
) -> dict[str, float | None]:
    if not labelled:
        return {"recall_at_k": None, "mrr": None, "hit": None}
    top = top_ids[:k]
    hits = [c for c in labelled if c in top]
    rank = next((i + 1 for i, c in enumerate(top) if c in labelled), None)
    return {
        "recall_at_k": len(hits) / len(labelled),
        "mrr": (1.0 / rank) if rank else 0.0,
        "hit": bool(rank),
    }


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.finditer(re.sub(r"\[[CK]\d+\]", " ", normalize_spaces(text))):
        raw = m.group(0).replace("$", "").replace(",", "").replace("%", "")
        with contextlib.suppress(ValueError):
            out.append(float(raw))
    return out


def value_present(expected: float, text: str, decimals: int) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for v in _numbers(text):
        for scale in (1.0, 1000.0, 0.001):
            if (
                abs(abs(v) * scale - abs(expected)) <= tol
                or abs(round(abs(v) * scale, decimals) - abs(expected)) <= tol
            ):
                return True
    return False


# Models emit narrow no-break spaces (U+202F) and NBSPs inside dates such as "January 25, 2026".
_UNICODE_SPACES = str.maketrans(
    {chr(0x202F): " ", chr(0x00A0): " ", chr(0x2009): " ", chr(0x2007): " "}
)


def normalize_spaces(text: str) -> str:
    return text.translate(_UNICODE_SPACES)


def period_stated(fy: int, period_end: str, text: str) -> bool:
    t = normalize_spaces(text).lower()
    y, m, d = period_end.split("-")
    month = {"01": "jan", "02": "feb", "03": "mar"}.get(m, m)
    return (
        f"fiscal {fy}" in t
        or f"fy{fy}" in t
        or f"fy {fy}" in t
        or f"fiscal year {fy}" in t
        or (f"{month}" in t and f"{int(d)}, {y}" in t)
        or f"{y}-{m}-{d}" in t
    )


def unit_consistent(unit: str, text: str) -> bool:
    t = text.lower()
    if unit == "USD_millions":
        return "million" in t or "billion" in t
    if unit == "USD_per_share":
        return "per share" in t or "/share" in t or "eps" in t
    if unit == "percent":
        return "%" in t or "percent" in t
    return True


def score_numeric(
    q: dict[str, Any], answer_md: str, calculations: list[dict[str, Any]]
) -> dict[str, Any]:
    best = _score_expectation(q, q["expected"], answer_md, calculations)
    for alt in q.get("expected_alternatives") or []:
        if best["exact_match"]:
            break
        cand = _score_expectation(q, alt, answer_md, calculations)
        if cand["exact_match"] or (cand["value_found"] and not best["value_found"]):
            cand["matched_alternative"] = alt
            best = cand
    return best


def _score_expectation(
    q: dict[str, Any], exp: dict[str, Any], answer_md: str, calculations: list[dict[str, Any]]
) -> dict[str, Any]:
    value = float(exp["value"])
    decimals = len(str(exp["value"]).split(".")[1]) if "." in str(exp["value"]) else 0
    has_value = value_present(value, answer_md, decimals)
    has_period = period_stated(exp["fiscal_year"], exp["period_end"], answer_md)
    has_unit = unit_consistent(exp["unit"], answer_md)
    calc_ok = None
    if q.get("expected_calculation"):
        for c in calculations:
            if c["formula"] == q["expected_calculation"] and c.get("rounded") is not None:
                calc_ok = (
                    abs(abs(float(c["rounded"])) - abs(value)) <= 0.5 * 10 ** (-decimals) + 1e-9
                )
                break
        if calc_ok is None:
            calc_ok = False
    return {
        "value_found": has_value,
        "period_stated": has_period,
        "unit_ok": has_unit,
        "exact_match": bool(has_value and has_period and has_unit),
        "calculator_agrees": calc_ok,
    }


def score_text(q: dict[str, Any], answer_md: str) -> dict[str, Any]:
    kws = q.get("expected_keywords") or []
    t = normalize_spaces(answer_md).lower()
    found = [k for k in kws if k.lower() in t]
    coverage = len(found) / len(kws) if kws else None
    return {
        "keywords_found": found,
        "keyword_coverage": coverage,
        "keyword_pass": (coverage or 0) >= 0.5 if kws else None,
    }


# ------------------------------------------------------------------------------- run


def evaluate(
    questions: list[dict[str, Any]],
    *,
    index_dir: Path | None,
    retrieval_only: bool,
    k: int,
    name: str,
    log_fn=print,
    resume: bool = False,
    compression_mode: str | None = None,
    context_budget: int | None = None,
) -> dict[str, Any]:
    from rag.query.retrieve import RetrievalSettings, hybrid_retrieve
    from rag.query.store import IndexStore, get_store

    store = (
        IndexStore(index_dir, figures_root=PROJECT_ROOT / "data" / "index")
        if index_dir
        else get_store()
    )
    pipeline = None
    if not retrieval_only:
        from rag.graph import Pipeline

        pipeline = Pipeline(
            store=store, compression_mode=compression_mode, context_budget=context_budget
        )

    # Rows are appended to the results file as they complete, so a crash or a quota stop loses
    # nothing; `resume=True` skips ids already present.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / f"{name}.jsonl"
    rows: list[dict[str, Any]] = []
    if resume and rows_path.exists():
        rows = [
            json.loads(ln)
            for ln in rows_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        done = {r["id"] for r in rows}
        questions = [q for q in questions if q["id"] not in done]
        log_fn(f"resuming {name}: {len(done)} done, {len(questions)} to go")
    elif rows_path.exists():
        rows_path.unlink()
    started = time.time()
    stopped: str | None = None
    for i, q in enumerate(questions, start=1):
        row: dict[str, Any] = {"id": q["id"], "intent": q["intent"], "question": q["question"]}
        if q["intent"] == "OUT_OF_SCOPE" and retrieval_only:
            rows.append({**row, "skipped": "out_of_scope"})
            continue
        if retrieval_only:
            # Phase 4 chain without model calls: slots -> rule intent + expansion -> hybrid
            # retrieval with per-query filters -> rerank gate / reranker.
            from rag.core.config import load_thresholds_config
            from rag.query.analyze import analyze
            from rag.query.rerank import rerank_node
            from rag.query.slots import extract_slots

            th = load_thresholds_config()
            slots = extract_slots(q["question"])
            a = analyze(q["question"], slots)
            r = hybrid_retrieve(
                store,
                a.queries,
                settings=RetrievalSettings(final_k=k, pool_k=th.rerank.candidates),
                intent=a.intent,
            )
            r, decision = rerank_node(
                store,
                r,
                q["question"],
                intent=a.intent,
                required_metrics=_required_metrics(slots, a.intent),
                thresholds=th.rerank,
                final_k=k,
            )
            row.update(retrieval_metrics(r.top_ids, q["supporting_chunk_ids"], k))
            row["top_ids"] = r.top_ids[:k]
            row["intent_pred"] = a.intent
            row["queries"] = a.retrieval_queries
            row["rerank"] = decision.gate if not decision.applied else "applied"
        else:
            res = pipeline.ask(
                q["question"], bypass_cache=True, request_id=f"eval-{name}-{q['id']}"
            )
            if res.warning and ("tokens per day" in res.warning or "TPD" in res.warning):
                stopped = f"daily quota reached at {q['id']}: {res.warning[:160]}"
                log_fn(f"STOP - {stopped}\nre-run with --resume when the window frees up")
                break
            row.update(retrieval_metrics(res.retrieval.top_ids, q["supporting_chunk_ids"], k))
            row["calculations"] = [c.model_dump() for c in res.calculations]
            row["answer"] = res.answer.answer_markdown
            row["confidence"] = res.answer.confidence
            row["answer_class"] = res.answer.answer_class
            row["verify_passed"] = res.verify.passed
            row["attempts"] = res.generation_attempts
            row["latency_ms"] = res.total_latency_ms
            row["pacing_wait_ms"] = sum(
                t.get("pacing_wait_ms", 0) for t in res.tokens_by_model.values()
            )
            row["tokens"] = res.tokens_by_model
            row["groq_calls"] = sum(t.get("calls", 0) for t in res.tokens_by_model.values())
            row["warning"] = res.warning
            row["intent_pred"] = res.intent
            row["scope"] = res.scope.model_dump(include={"in_scope", "score", "rule", "source"})
            row["queries"] = res.retrieval.queries
            row["rerank"] = res.rerank.gate if not res.rerank.applied else "applied"
            row["compression"] = res.compression.summary()
            row["context_tokens"] = res.context.tokens_used
            calcs = [c.model_dump() for c in res.calculations]
            if q.get("expected"):
                row.update(score_numeric(q, res.answer.answer_markdown, calcs))
            else:
                row.update(score_text(q, res.answer.answer_markdown))
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fn(f"[{i}/{len(questions)}] {q['id']:4s} {q['intent']:16s} " + _one_line(row))

    summary = summarize(rows)
    summary["stopped"] = stopped
    summary["completed"] = len(rows)
    summary["name"] = name
    summary["index_dir"] = str(store.index_dir)
    summary["embedder"] = store.manifest.embedder
    summary["corpus_version"] = store.corpus_version
    summary["retrieval_only"] = retrieval_only
    summary["compression_mode"] = compression_mode
    summary["context_budget"] = context_budget
    summary["k"] = k
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (RESULTS_DIR / f"{name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _required_metrics(slots, intent: str) -> list[str]:  # noqa: ANN001
    from rag.core.config import load_formulas_config

    if intent == "COMPUTATION" and slots.formulas:
        cfg = load_formulas_config()
        items: list[str] = []
        for fid in slots.formulas:
            f = cfg.formulas.get(fid)
            for spec in f.inputs.values() if f else []:
                if "{metric}" not in spec.line_item and spec.line_item not in items:
                    items.append(spec.line_item)
        return items
    return list(slots.metrics)


def _one_line(row: dict[str, Any]) -> str:
    parts = []
    if row.get("recall_at_k") is not None:
        parts.append(f"recall@k={row['recall_at_k']:.2f} mrr={row['mrr']:.2f}")
    if "exact_match" in row:
        parts.append(
            f"exact={'Y' if row['exact_match'] else 'n'} (val={int(row['value_found'])} per={int(row['period_stated'])} unit={int(row['unit_ok'])})"
        )
    if "keyword_coverage" in row and row["keyword_coverage"] is not None:
        parts.append(f"kw={row['keyword_coverage']:.2f}")
    if "verify_passed" in row:
        parts.append(f"verify={'Y' if row['verify_passed'] else 'n'} {row.get('latency_ms')}ms")
    return " ".join(parts)


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("skipped"):
            by_intent[r["intent"]].append(r)

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        numeric = [r for r in rs if "exact_match" in r]
        textual = [r for r in rs if r.get("keyword_coverage") is not None]
        tokens_in = sum(t.get("in", 0) for r in rs for t in (r.get("tokens") or {}).values())
        tokens_out = sum(t.get("out", 0) for r in rs for t in (r.get("tokens") or {}).values())
        return {
            "n": len(rs),
            "recall_at_k": _mean([r.get("recall_at_k") for r in rs]),
            "mrr": _mean([r.get("mrr") for r in rs]),
            "hit_rate": _mean([float(r["hit"]) for r in rs if r.get("hit") is not None]),
            "numeric_n": len(numeric),
            "exact_match": _mean([float(r["exact_match"]) for r in numeric]),
            "value_found": _mean([float(r["value_found"]) for r in numeric]),
            "calculator_agrees": _mean(
                [
                    float(r["calculator_agrees"])
                    for r in numeric
                    if r.get("calculator_agrees") is not None
                ]
            ),
            "text_n": len(textual),
            "keyword_coverage": _mean([r["keyword_coverage"] for r in textual]),
            "verify_pass_rate": _mean(
                [float(r["verify_passed"]) for r in rs if "verify_passed" in r]
            ),
            "latency_ms_p50": _percentile([r["latency_ms"] for r in rs if "latency_ms" in r], 50),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    return {
        "overall": block([r for rs in by_intent.values() for r in rs]),
        "by_intent": {k: block(v) for k, v in sorted(by_intent.items())},
    }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((p / 100) * (len(vals) - 1))))
    return vals[idx]


def rescore(name: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute scores from saved answers (no model calls) and rewrite the run's files."""
    rows_path = RESULTS_DIR / f"{name}.jsonl"
    by_id = {q["id"]: q for q in questions}
    rows = [
        json.loads(ln) for ln in rows_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    for r in rows:
        q = by_id.get(r["id"])
        if q is None or r.get("skipped") or "answer" not in r:
            continue
        if q.get("expected"):
            r.update(score_numeric(q, r["answer"], r.get("calculations", [])))
        else:
            r.update(score_text(q, r["answer"]))
    rows_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    summary_path = RESULTS_DIR / f"{name}.summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary = summarize(rows)
    summary.update({k: v for k, v in old.items() if k not in ("overall", "by_intent")})
    summary["rescored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def drop_rows(name: str, ids: set[str]) -> None:
    rows_path = RESULTS_DIR / f"{name}.jsonl"
    if not rows_path.exists():
        return
    kept = [
        ln
        for ln in rows_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln)["id"] not in ids
    ]
    rows_path.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated golden ids")
    ap.add_argument("--intents", help="comma-separated intents")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true", help="skip ids already in the results file")
    ap.add_argument("--redo", help="comma-separated ids to drop from the results before resuming")
    ap.add_argument("--rescore", action="store_true", help="re-score saved answers; no model calls")
    ap.add_argument(
        "--compression",
        choices=("classifier", "never", "always"),
        default=None,
        help="compression arm (default: thresholds.yaml `compression.mode`)",
    )
    ap.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="override the profile's context_budget_tokens (e.g. 2500 to mirror groq_build)",
    )
    args = ap.parse_args(argv)

    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings

    s = get_settings()
    configure_logging("WARNING", secrets=s.secret_values())

    questions = load_golden()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.intents:
        wanted = set(args.intents.split(","))
        questions = [q for q in questions if q["intent"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if args.rescore:
        if not args.name:
            print("--rescore needs --name")
            return 2
        summary = rescore(args.name, load_golden())
        print(json.dumps(summary["overall"], indent=2))
        return 0
    if args.redo and args.name:
        drop_rows(args.name, set(args.redo.split(",")))
        args.resume = True
    name = args.name or (
        "retrieval" if args.retrieval_only else "golden"
    ) + datetime.now().strftime("_%Y%m%d_%H%M")
    summary = evaluate(
        questions,
        index_dir=args.index,
        retrieval_only=args.retrieval_only,
        k=args.k,
        name=name,
        resume=args.resume,
        compression_mode=args.compression,
        context_budget=args.context_budget,
    )
    print(json.dumps(summary["overall"], indent=2))
    print("results:", RESULTS_DIR / f"{name}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
