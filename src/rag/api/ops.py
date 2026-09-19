"""Operations view over the on-disk logs (plan Phase 7, architecture §9.3, §12).

`GET /api/admin/stats` aggregates the last N trace lines plus the usage ledger into the numbers
the UI ops panel shows: cache hit rate over time, model tokens and calls by role, rate-limit
waits, verification failures, the compression decision mix, latency percentiles per node and
`rss_mb`. Reading is streaming and bounded (`tail`), so a long-running server does not re-parse
megabytes on every poll.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    """Last `limit` parseable JSON objects from a .jsonl file (empty list if it is missing)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()[-limit:]
    out: list[dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:  # a partially written last line
            continue
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1))))
    return round(ordered[k], 1)


def _buckets(traces: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(traces), size):
        chunk = traces[i : i + size]
        if chunk:
            yield chunk


def hit_rate_over_time(traces: list[dict[str, Any]], buckets: int = 10) -> list[dict[str, Any]]:
    """Cache hit rate in equal-sized request buckets, oldest first. Bypassed requests (every
    evaluation run) are excluded — they never consult the cache."""
    served = [t for t in traces if t.get("cache_tier") in {"L1", "L2", "MISS"}]
    if not served:
        return []
    size = max(1, len(served) // max(1, buckets))
    out = []
    for chunk in _buckets(served, size):
        hits = sum(1 for t in chunk if t.get("cache_tier") in {"L1", "L2"})
        out.append(
            {
                "from_ts": chunk[0].get("ts"),
                "to_ts": chunk[-1].get("ts"),
                "requests": len(chunk),
                "hits": hits,
                "hit_rate": round(hits / len(chunk), 3),
            }
        )
    return out


def ops_stats(
    *,
    traces_path: Path,
    ledger_path: Path,
    decisions_path: Path,
    limit: int = 500,
) -> dict[str, Any]:
    traces = tail_jsonl(traces_path, limit)
    usage = tail_jsonl(ledger_path, limit * 4)
    decisions = tail_jsonl(decisions_path, limit)

    tiers = Counter(t.get("cache_tier", "unknown") for t in traces)
    served = [t for t in traces if t.get("cache_tier") in {"L1", "L2", "MISS"}]
    hits = sum(1 for t in served if t.get("cache_tier") in {"L1", "L2"})

    by_role: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "pacing_wait_ms": 0, "retries": 0}
    )
    statuses: Counter[str] = Counter()
    for r in usage:
        role = r.get("role", "?")
        b = by_role[role]
        b["calls"] += 1
        b["tokens_in"] += int(r.get("tokens_in", 0))
        b["tokens_out"] += int(r.get("tokens_out", 0))
        b["pacing_wait_ms"] += int(r.get("pacing_wait_ms", 0))
        b["retries"] += int(r.get("retries", 0))
        statuses[str(r.get("status", "?"))] += 1
    for role, b in by_role.items():
        b["model"] = next((r.get("model") for r in reversed(usage) if r.get("role") == role), "?")  # type: ignore[assignment]

    verified = [t for t in traces if t.get("verify_passed") is not None]
    failures = [t for t in verified if not t.get("verify_passed")]
    retried = [t for t in traces if int(t.get("generation_attempts", 0) or 0) > 1]

    actions: Counter[str] = Counter()
    violations = 0
    compressor_calls = 0
    tokens_before = tokens_after = 0
    for d in decisions:
        for c in d.get("chunks", []):
            actions[str(c.get("applied", c.get("action")))] += 1
            if c.get("reverted"):
                violations += 1
        compressor_calls += int(d.get("llm_calls", 0) or 0)
        tokens_before += int(d.get("tokens_before", 0) or 0)
        tokens_after += int(d.get("tokens_after", 0) or 0)

    node_latency: dict[str, dict[str, float | None]] = {}
    all_nodes: set[str] = set()
    for t in traces:
        all_nodes.update((t.get("latency_ms_by_node") or {}).keys())
    for node in sorted(all_nodes):
        values = [
            float(t["latency_ms_by_node"][node])
            for t in traces
            if node in (t.get("latency_ms_by_node") or {})
        ]
        node_latency[node] = {
            "n": len(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        }
    totals = [float(t.get("total_latency_ms", 0)) for t in traces if t.get("total_latency_ms")]
    hit_totals = [
        float(t.get("total_latency_ms", 0))
        for t in traces
        if t.get("cache_tier") in {"L1", "L2"} and t.get("total_latency_ms")
    ]
    miss_totals = [
        float(t.get("total_latency_ms", 0))
        for t in traces
        if t.get("cache_tier") == "MISS" and t.get("total_latency_ms")
    ]
    rss = [float(t["rss_mb"]) for t in traces if t.get("rss_mb")]
    errors = [t for t in traces if t.get("error")]

    return {
        "window": {
            "traces": len(traces),
            "usage_records": len(usage),
            "decision_records": len(decisions),
            "limit": limit,
            "first_ts": traces[0].get("ts") if traces else None,
            "last_ts": traces[-1].get("ts") if traces else None,
        },
        "cache": {
            "by_tier": dict(tiers),
            "served_requests": len(served),
            "hit_rate": round(hits / len(served), 3) if served else 0.0,
            "admitted": sum(1 for t in traces if t.get("admitted")),
            "over_time": hit_rate_over_time(traces),
        },
        "models": {
            "by_role": {k: dict(v) for k, v in sorted(by_role.items())},
            "statuses": dict(statuses),
        },
        "quality": {
            "verified_requests": len(verified),
            "verification_failures": len(failures),
            "failure_rate": round(len(failures) / len(verified), 3) if verified else 0.0,
            "regenerations": len(retried),
            "errors": len(errors),
            "last_issues": [
                {"request_id": t.get("request_id"), "issues": t.get("verify_issues", [])}
                for t in failures[-5:]
            ],
        },
        "compression": {
            "requests": len(decisions),
            "actions": dict(actions),
            "fidelity_reverts": violations,
            "compressor_calls": compressor_calls,
            "context_tokens_before": tokens_before,
            "context_tokens_after": tokens_after,
            "tokens_saved_pct": (
                round(100 * (tokens_before - tokens_after) / tokens_before, 1)
                if tokens_before
                else 0.0
            ),
        },
        "latency": {
            "total_p50": percentile(totals, 50),
            "total_p95": percentile(totals, 95),
            "cache_hit_p50": percentile(hit_totals, 50),
            "cache_miss_p50": percentile(miss_totals, 50),
            "by_node": node_latency,
        },
        "memory": {
            "rss_mb_last": rss[-1] if rss else None,
            "rss_mb_peak": max(rss) if rss else None,
        },
    }
