"""Local resource profile (plan Phase 7; input for the deferred deployment decision, §14.4).

    .venv/Scripts/python eval/resource_profile.py [--queries 50] [--report docs/reports/local_resource_profile.md]

Measures what a laptop actually pays to serve this system: import time, index load time, RSS at
each startup stage and after the run, and per-node latency percentiles over a **mixed** query
stream — 60 % distinct golden questions (cache misses, the full pipeline) and 40 % repeats and
paraphrases of questions already asked (cache hits). That mix is deliberate: a profile built
only from misses would overstate steady-state latency, and one built only from hits would
understate it.

Model calls: one per miss. With the default 50 queries that is ~30 generations; the cached
requests cost none, which the report shows separately.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import RESULTS_DIR, load_golden  # noqa: E402


def rss_mb() -> float | None:
    try:
        import os

        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1))))
    return round(ordered[k], 1)


def build_stream(n: int, seed: int) -> list[tuple[str, str, bool]]:
    """(golden id, question text, expected_repeat) — 60 % fresh questions, 40 % repeats."""
    rng = random.Random(seed)
    golden = [q for q in load_golden() if q["id"] not in {"G14", "O2"}]  # keep refusals out
    rng.shuffle(golden)
    fresh_n = max(1, int(round(n * 0.6)))
    fresh = golden[:fresh_n]
    stream: list[tuple[str, str, bool]] = [(q["id"], q["question"], False) for q in fresh]
    asked = list(stream)
    while len(stream) < n:
        qid, text, _ = rng.choice(asked)
        stream.append((qid, text, True))
    rng.shuffle(stream)
    # a repeat must come after its first ask, or it is not a repeat
    seen: set[str] = set()
    ordered: list[tuple[str, str, bool]] = []
    deferred: list[tuple[str, str, bool]] = []
    for qid, text, repeat in stream:
        if repeat and qid not in seen:
            deferred.append((qid, text, repeat))
            continue
        seen.add(qid)
        ordered.append((qid, text, repeat))
    ordered.extend(deferred)
    return ordered[:n]


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument(
        "--report", default=str(PROJECT_ROOT / "docs" / "reports" / "local_resource_profile.md")
    )
    ap.add_argument("--name", default="resource_profile")
    ap.add_argument(
        "--purge-first",
        action="store_true",
        help="empty the L2 cache before the run so the miss path is measured; without it a "
        "repeat run of the same seed is served entirely from cache and reports only hits",
    )
    args = ap.parse_args(argv)

    stages: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    rss_start = rss_mb()
    tracemalloc.start()

    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings

    settings = get_settings()
    configure_logging("WARNING", secrets=settings.secret_values())
    t_import = time.perf_counter()
    from rag.graph import Pipeline  # noqa: PLC0415 - timed deliberately

    stages.append(
        {
            "stage": "import rag.graph",
            "seconds": round(time.perf_counter() - t_import, 2),
            "rss_mb": rss_mb(),
        }
    )

    t_store = time.perf_counter()
    pipeline = Pipeline()
    stages.append(
        {
            "stage": "Pipeline() — index, embedder, BM25, cache",
            "seconds": round(time.perf_counter() - t_store, 2),
            "rss_mb": rss_mb(),
        }
    )
    startup_seconds = round(time.perf_counter() - t0, 2)

    purged = None
    if args.purge_first:
        purged = pipeline.cache.purge("all")
        print(f"cache purged before the run: {purged}")
    stream = build_stream(args.queries, args.seed)
    rows: list[dict[str, Any]] = []
    peak_rss = rss_mb() or 0.0
    for i, (qid, text, expect_repeat) in enumerate(stream, start=1):
        started = time.perf_counter()
        r = pipeline.ask(text)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        now_rss = rss_mb() or 0.0
        peak_rss = max(peak_rss, now_rss)
        calls = sum(t["calls"] for t in r.tokens_by_model.values())
        rows.append(
            {
                "n": i,
                "id": qid,
                "intent": r.intent,
                "cache_tier": r.cache_tier,
                "expected_repeat": expect_repeat,
                "latency_ms": elapsed_ms,
                "model_calls": calls,
                "tokens": sum(t["in"] + t["out"] for t in r.tokens_by_model.values()),
                "context_tokens": r.context.tokens_used,
                "latency_ms_by_node": r.latency_ms_by_node,
                "rss_mb": now_rss,
                "warning": r.warning,
            }
        )
        print(
            f"[{i}/{len(stream)}] {qid:4s} {r.cache_tier:8s} {elapsed_ms:>6} ms "
            f"calls={calls} rss={now_rss} MB"
        )
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # A run where the provider was unavailable measures the degrade path, not the system: it
    # would otherwise overwrite a good report with 50 identical "model call failed" rows.
    degraded = [r for r in rows if r["warning"]]
    if len(degraded) > 0.2 * len(rows):
        print(
            f"\nABORTED: {len(degraded)} of {len(rows)} requests failed at the model call "
            f"(first: {degraded[0]['warning'][:160]}).\nNothing was written; re-run when a "
            "provider window is open."
        )
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(
        json.dumps({"stages": stages, "rows": rows}, indent=1), encoding="utf-8"
    )
    report = render(
        rows=rows,
        stages=stages,
        startup_seconds=startup_seconds,
        rss_start=rss_start,
        peak_rss=peak_rss,
        traced_peak_mb=round(traced_peak / (1024 * 1024), 1),
        profile=pipeline.models.active_profile,
        corpus_version=pipeline.store.corpus_version,
        chunks=len(pipeline.store.chunks),
        purged=purged,
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"\n{report}\nreport written to {args.report}")
    return 0


def render(
    *,
    rows: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    startup_seconds: float,
    rss_start: float | None,
    peak_rss: float,
    traced_peak_mb: float,
    profile: str,
    corpus_version: str,
    chunks: int,
    purged: dict[str, int] | None = None,
) -> str:
    hits = [r for r in rows if r["cache_tier"] in {"L1", "L2"}]
    misses = [r for r in rows if r["cache_tier"] == "MISS"]
    all_lat = [float(r["latency_ms"]) for r in rows]
    hit_lat = [float(r["latency_ms"]) for r in hits]
    miss_lat = [float(r["latency_ms"]) for r in misses]
    nodes: dict[str, list[float]] = {}
    for r in rows:
        for node, ms in (r["latency_ms_by_node"] or {}).items():
            nodes.setdefault(node, []).append(float(ms))
    calls = sum(r["model_calls"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)

    lines = [
        "# Local resource profile (Phase 7)",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · `eval/resource_profile.py` · "
        f"{len(rows)} mixed queries ({len(misses)} cache misses, {len(hits)} hits) · profile "
        f"`{profile}` · corpus `{corpus_version}` ({chunks} chunks) · one process, one worker.",
        "",
        "Input for the deployment decision deferred in §14.4: this is what one laptop process pays "
        "to hold the whole system in memory and answer from it."
        + (
            " The L2 cache was emptied before the run, so the misses below are real full-pipeline "
            "runs rather than an artefact of a warm cache."
            if purged
            else " The cache was **not** emptied first, so any question already cached is counted "
            "as a hit."
        ),
        "",
        *(
            []
            if profile == "groq_build"
            else [
                f"> **Provider note.** This run used `{profile}`, not the build profile "
                "`groq_build`, because the Groq free tier's daily token window was already spent on "
                "the RAGAS judge (see `docs/reports/ragas_groq.md`). Startup, RSS and every node "
                "except `generate` are provider-independent; for the free-tier generation latency "
                "read `latency_ms_p50` per intent in `eval/results/golden_phase4.summary.json` and "
                "the walkthrough report.",
                "",
            ]
        ),
        "## 1. Startup",
        "",
        "| stage | seconds | RSS after (MB) |",
        "|---|---|---|",
    ]
    for s in stages:
        lines.append(f"| {s['stage']} | {s['seconds']} | {s['rss_mb']} |")
    lines += [
        f"| **total to first request** | **{startup_seconds}** | — |",
        "",
        f"RSS before any import: {rss_start} MB. Peak RSS during the run: **{peak_rss} MB** "
        f"(Python-object peak measured by `tracemalloc`: {traced_peak_mb} MB — the rest is the ONNX "
        "runtime, the embedding model and NumPy buffers).",
        "",
        "## 2. Latency",
        "",
        "| stream | n | p50 (ms) | p95 (ms) | max (ms) |",
        "|---|---|---|---|---|",
        f"| all requests | {len(all_lat)} | {percentile(all_lat, 50)} | {percentile(all_lat, 95)} | {max(all_lat) if all_lat else '–'} |",
        f"| cache hits | {len(hit_lat)} | {percentile(hit_lat, 50)} | {percentile(hit_lat, 95)} | {max(hit_lat) if hit_lat else '–'} |",
        f"| cache misses (full pipeline) | {len(miss_lat)} | {percentile(miss_lat, 50)} | {percentile(miss_lat, 95)} | {max(miss_lat) if miss_lat else '–'} |",
        "",
        "### Per node",
        "",
        "| node | n | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|",
    ]
    for node in sorted(nodes):
        v = nodes[node]
        lines.append(f"| {node} | {len(v)} | {percentile(v, 50)} | {percentile(v, 95)} |")
    lines += [
        "",
        "## 3. Cost of the stream",
        "",
        f"- model calls: **{calls}** over {len(rows)} requests ({len(hits)} served from cache at zero calls)",
        f"- model tokens: **{tokens:,}**",
        f"- mean context sent per miss: {round(sum(r['context_tokens'] for r in misses) / max(len(misses), 1))} tokens",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
