"""Scripted cache walkthrough — the 10 steps of plan Phase 6 (architecture §5).

    .venv/Scripts/python scripts/cache_walkthrough.py                    # in-process, real models
    .venv/Scripts/python scripts/cache_walkthrough.py --api              # against a running server
    .venv/Scripts/python scripts/cache_walkthrough.py --report docs/reports/cache_walkthrough.md

In-process mode builds the pipeline once, then a *second* pipeline with a fresh L1 for step 3
(what a restart does) and a third with an edited retrieval threshold for step 7 (what a config
edit + restart does). API mode drives a running server through `/api/ask` and the dev-only
admin routes: step 3 clears L1 through `purge {"scope": "l1"}` and step 7 is reported as
"manual" (a config edit needs a real restart). The walkthrough starts from an empty cache
(`purge all`) so results are reproducible; expect about 6 full pipeline runs (plan §Groq usage).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from typing import Any

from rag.core.console import utf8_console

DAY = 86_400
Q_ASSETS = "What were NVIDIA's total assets as of Jan 25, 2026?"
Q_PARA = "Total assets at the end of fiscal 2026?"
Q_PRIOR = "What were total assets as of Jan 26, 2025?"
Q_MEETING = "When is NVIDIA's annual meeting?"


class Step:
    def __init__(self, n: int, action: str, expected: str):
        self.n, self.action, self.expected = n, action, expected
        self.observed = ""
        self.ok: bool | None = None
        self.ms: int | None = None
        self.model_calls: int | None = None


def _admitted(res: dict[str, Any]) -> bool:
    return (res.get("write") or {}).get("admitted") is True


def _fmt_answer(md: str) -> str:
    return " ".join(md.split())[:110]


# --------------------------------------------------------------------------- in-process


class InProcess:
    def __init__(self) -> None:
        from rag.core.clock import Clock
        from rag.core.config import load_thresholds_config
        from rag.core.logging import configure_logging
        from rag.core.settings import get_settings
        from rag.graph import Pipeline

        s = get_settings()
        configure_logging("WARNING", secrets=s.secret_values())
        self.settings = s
        self.clock = Clock(allow_offset=True)
        self.thresholds = load_thresholds_config(settings=s)
        self.pipeline = Pipeline(settings=s, thresholds=self.thresholds, clock=self.clock)
        self._Pipeline = Pipeline
        self.ledger = self.pipeline.ledger

    def ask(self, q: str, *, bypass: bool = False, pipeline=None) -> dict[str, Any]:  # noqa: ANN001
        p = pipeline or self.pipeline
        before = self.ledger.count()
        t0 = time.perf_counter()
        r = p.ask(q, bypass_cache=bypass)
        ms = int((time.perf_counter() - t0) * 1000)
        calls = self.ledger.count() - before
        return {
            "tier": r.cache_tier,
            "similarity": r.cache.similarity,
            "answer": r.answer.answer_markdown,
            "write": r.cache.write.model_dump() if r.cache.write else None,
            "ms": ms,
            "model_calls": calls,
            "request_id": r.request_id,
        }

    def restart(self):  # noqa: ANN201 - Pipeline
        """Fresh process ≈ fresh L1 on the same L2 directory."""
        return self._Pipeline(
            settings=self.settings,
            store=self.pipeline.store,
            thresholds=self.thresholds,
            clock=self.clock,
        )

    def restart_with_config_edit(self):  # noqa: ANN201
        edited = self.thresholds.model_copy(deep=True)
        edited.retrieval.final_k = self.thresholds.retrieval.final_k + 1
        return self._Pipeline(
            settings=self.settings, store=self.pipeline.store, thresholds=edited, clock=self.clock
        )

    def advance(self, seconds: int) -> None:
        self.clock.advance(seconds)

    def set_offset(self, seconds: int) -> None:
        self.clock.set_offset(seconds)

    def reset_clock(self) -> None:
        self.clock.reset()

    def purge(self, scope: str = "all") -> dict[str, Any]:
        return self.pipeline.cache.purge(scope)

    def stats(self) -> dict[str, Any]:
        return self.pipeline.cache.stats()

    def sweep(self) -> dict[str, Any]:
        return self.pipeline.cache.sweeper.sweep()


# ---------------------------------------------------------------------------------- API


class ViaApi:
    def __init__(self, url: str):
        import httpx

        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=240)
        ready = self.http.get(f"{self.url}/readyz").json()
        if not ready.get("ready"):
            raise SystemExit(f"server not ready: {ready}")
        if not ready.get("admin_enabled"):
            raise SystemExit("admin routes are disabled (APP_ENV must be dev)")
        self.ledger_calls = self._ledger_calls()

    def _ledger_calls(self) -> int:
        s = self.http.get(f"{self.url}/api/admin/cache/stats").json()
        return int(s.get("l2", {}).get("misses", 0))  # proxy: misses == pipeline runs

    def ask(self, q: str, *, bypass: bool = False, pipeline=None) -> dict[str, Any]:  # noqa: ANN001
        t0 = time.perf_counter()
        r = self.http.post(f"{self.url}/api/ask", json={"question": q, "bypass_cache": bypass})
        r.raise_for_status()
        ms = int((time.perf_counter() - t0) * 1000)
        body = r.json()
        calls = sum(t.get("calls", 0) for t in body["debug"]["tokens_by_model"].values())
        return {
            "tier": body["cache_tier"],
            "similarity": body["cache"].get("similarity"),
            "answer": body["answer"]["answer_markdown"],
            "write": body["cache"].get("write"),
            "ms": ms,
            "model_calls": calls,
            "request_id": body["request_id"],
        }

    def restart(self):  # noqa: ANN201
        self.http.post(f"{self.url}/api/admin/cache/purge", json={"scope": "l1"}).raise_for_status()
        return None

    def restart_with_config_edit(self):  # noqa: ANN201
        return None  # cannot restart a server from here; reported as manual

    def advance(self, seconds: int) -> None:
        self.http.post(
            f"{self.url}/api/admin/clock", json={"advance_seconds": seconds}
        ).raise_for_status()

    def set_offset(self, seconds: int) -> None:
        self.http.post(
            f"{self.url}/api/admin/clock", json={"offset_seconds": seconds}
        ).raise_for_status()

    def reset_clock(self) -> None:
        self.http.post(f"{self.url}/api/admin/clock", json={"reset": True}).raise_for_status()

    def purge(self, scope: str = "all") -> dict[str, Any]:
        return self.http.post(f"{self.url}/api/admin/cache/purge", json={"scope": scope}).json()

    def stats(self) -> dict[str, Any]:
        return self.http.get(f"{self.url}/api/admin/cache/stats").json()

    def sweep(self) -> dict[str, Any]:
        return self.http.post(f"{self.url}/api/admin/cache/sweep").json()


# --------------------------------------------------------------------------- the steps


def run(drv, *, log) -> list[Step]:  # noqa: ANN001
    steps: list[Step] = []

    def step(n: int, action: str, expected: str) -> Step:
        s = Step(n, action, expected)
        steps.append(s)
        log(f"\n[{n}] {action}\n    expect: {expected}")
        return s

    def record(s: Step, res: dict[str, Any], ok: bool, extra: str = "") -> None:
        w = res.get("write") or {}
        admitted = (
            f" · admitted {w.get('answer_class')} → {'+'.join(w.get('tiers', []))}"
            if w.get("admitted")
            else (f" · not admitted ({'; '.join(w.get('reasons', []))})" if w else "")
        )
        sim = f" · sim {res['similarity']:.3f}" if res.get("similarity") is not None else ""
        s.observed = f"{res['tier']}{sim} · {res['model_calls']} model call(s) · {res['ms']} ms{admitted}{extra}"
        s.ok, s.ms, s.model_calls = ok, res["ms"], res["model_calls"]
        log(f"    got:    {s.observed}\n    answer: {_fmt_answer(res['answer'])}")

    drv.reset_clock()
    drv.purge("all")
    log(f"start: purged · stats {drv.stats()['l2']['entries']} L2 entries")

    s = step(1, f"Ask “{Q_ASSETS}”", "MISS; answer $206,803M; admitted as filed_fact")
    r = drv.ask(Q_ASSETS)
    w = r.get("write") or {}
    record(
        s,
        r,
        r["tier"] == "MISS"
        and w.get("admitted") is True
        and w.get("answer_class") == "filed_fact"
        and "206,803" in r["answer"],
    )

    s = step(2, "Ask the same question again", "L1 hit, no model call")
    r = drv.ask(Q_ASSETS)
    record(s, r, r["tier"] == "L1" and r["model_calls"] == 0)

    s = step(3, "Restart the server (fresh L1); ask again", "L2 hit (L1 was empty), promoted to L1")
    p2 = drv.restart()
    r = drv.ask(Q_ASSETS, pipeline=p2)
    r_again = drv.ask(Q_ASSETS, pipeline=p2)
    record(
        s,
        r,
        r["tier"] == "L2" and r["model_calls"] == 0 and r_again["tier"] == "L1",
        f" · then {r_again['tier']}",
    )
    if p2 is not None:
        drv.pipeline = p2

    s = step(4, f"Ask “{Q_PARA}”", "L2 hit (paraphrase, same slots)")
    r = drv.ask(Q_PARA)
    record(s, r, r["tier"] == "L2" and r["model_calls"] == 0)

    s = step(5, f"Ask “{Q_PRIOR}”", "MISS — period slot differs (G15)")
    r = drv.ask(Q_PRIOR)
    record(s, r, r["tier"] == "MISS" and _admitted(r))

    s = step(6, "Toggle bypass cache; ask step 1", "Pipeline runs; no cache read or write")
    entries_before = drv.stats()["l2"]["entries"]
    r = drv.ask(Q_ASSETS, bypass=True)
    entries_after = drv.stats()["l2"]["entries"]
    record(
        s,
        r,
        r["tier"] == "bypassed"
        and r["model_calls"] >= 1
        and r.get("write") is None
        and entries_after == entries_before,
        f" · L2 entries {entries_before}→{entries_after}",
    )

    s = step(
        7,
        "Change a retrieval threshold; restart; ask step 1 — then revert and ask again",
        "MISS (retrieval_config_hash changed); after revert → L2 hit on the original entry",
    )
    p3 = drv.restart_with_config_edit()
    if p3 is None:
        s.observed = "manual in API mode: edit config/thresholds.yaml retrieval.final_k, restart, ask; revert, restart, ask"
        s.ok = None
        log("    got:    " + s.observed)
    else:
        r = drv.ask(Q_ASSETS, pipeline=p3)
        p4 = drv.restart()  # original config again
        r2 = drv.ask(Q_ASSETS, pipeline=p4)
        record(
            s,
            r,
            r["tier"] == "MISS" and _admitted(r) and r2["tier"] == "L2",
            f" · after revert: {r2['tier']} ({r2['model_calls']} calls)",
        )
        drv.pipeline = p4

    s = step(
        8,
        f"Ask “{Q_MEETING}”; advance clock +1 day; ask again",
        "MISS admitted as time_anchored; then MISS (1-day TTL expired)",
    )
    r = drv.ask(Q_MEETING)
    w = r.get("write") or {}
    drv.advance(DAY)
    r2 = drv.ask(Q_MEETING)
    record(
        s,
        r,
        r["tier"] == "MISS" and w.get("answer_class") == "time_anchored" and r2["tier"] == "MISS",
        f" · after +1d: {r2['tier']} ({r2['model_calls']} calls)",
    )

    s = step(
        9,
        "Advance clock to +31 days; ask step 1",
        "MISS — filed_fact 30-day TTL expired; sweeper removes the stale entry",
    )
    drv.set_offset(31 * DAY)
    swept = drv.sweep()
    r = drv.ask(Q_ASSETS)
    record(
        s,
        r,
        r["tier"] == "MISS" and _admitted(r) and swept.get("expired", 0) >= 1,
        f" · sweep {swept}",
    )

    s = step(10, "Reset clock; purge all", "Stats show zero entries")
    drv.reset_clock()
    purged = drv.purge("all")
    st = drv.stats()
    s.observed = f"purged {purged} · L2 entries {st['l2']['entries']} · L1 entries {st['l1']['entries']} · clock offset {st['clock']['offset_s']}"
    s.ok = st["l2"]["entries"] == 0 and st["l1"]["entries"] == 0 and st["clock"]["offset_s"] == 0
    log("    got:    " + s.observed)
    return steps


def render(steps: list[Step], mode: str, profile: str) -> str:
    total_calls = sum(s.model_calls or 0 for s in steps)
    hit_ms = [s.ms for s in steps if s.ms is not None and s.observed.startswith(("L1", "L2"))]
    lines = [
        "# Cache walkthrough (Phase 6)",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · mode `{mode}` · model profile `{profile}` · "
        f"`scripts/cache_walkthrough.py`. Plan Phase 6 table, scripted; the same steps can be clicked through in the UI cache panel.",
        "",
        "| step | action | expected | observed | ok |",
        "|---|---|---|---|---|",
    ]
    for s in steps:
        ok = "✓" if s.ok else ("manual" if s.ok is None else "✗")
        lines.append(f"| {s.n} | {s.action} | {s.expected} | {s.observed} | {ok} |")
    lines += [
        "",
        f"Full pipeline runs (model calls) across the walkthrough: **{total_calls}**. "
        f"Cache-hit latencies: {', '.join(f'{m} ms' for m in hit_ms) or 'n/a'} "
        f"(NFR-5 target p50 < 300 ms).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--api", action="store_true", help="drive a running server instead of in-process"
    )
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    ap.add_argument("--note", default=None, help="free-text note appended to the report")
    args = ap.parse_args(argv)

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    if args.api:
        drv = ViaApi(args.url)
        profile = "server"
        mode = "api"
    else:
        drv = InProcess()
        profile = drv.pipeline.models.active_profile
        mode = "in-process"
    steps = run(drv, log=log)
    failed = [s for s in steps if s.ok is False]
    report = render(steps, mode, profile)
    if args.note:
        report += f"\n**Note.** {args.note}\n"
    print("\n" + report)
    if args.report:
        from pathlib import Path

        Path(args.report).write_text(report, encoding="utf-8")
        print(f"report written to {args.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
