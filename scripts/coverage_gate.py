"""Coverage gate for the deterministic modules (plan Phase 8, rule G4).

    make coverage   (or)   .venv/Scripts/python scripts/coverage_gate.py [--min 90] [--no-report]

Runs the test suite under `pytest-cov` restricted to the modules where a silent financial error
would hide — slot extraction, the calculator, the fidelity guard, the cache guard and admission,
LFU maths, TTL classes, version keys, classifier rules and features, the verifier and the scope
gate — and fails (exit 1) when any of them is under the threshold. Writes
`docs/reports/coverage_phase8.md` with the per-module table so the number is evidence, not a
claim. No model calls; the suite itself is network-free.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.core.console import utf8_console

ROOT = Path(__file__).resolve().parents[1]

# module → why it is on the list (rule G4 names the first six; the rest guard the same risks)
MODULES: dict[str, str] = {
    "rag.query.slots": "slot extraction: periods, metrics, formulas, ask type, topics (cache keys)",
    "rag.calc.calculator": "deterministic ratios and YoY; whitelisted AST evaluator",
    "rag.compress.fidelity": "numeric fidelity guard on every compressor output",
    "rag.cache.records": "cache guard keys and record schema (false-hit defence)",
    "rag.cache.admission": "cache admission policy (§5.6)",
    "rag.cache.lfu": "Redis-style LFU counter, decay and victim choice",
    "rag.cache.ttl": "TTL classes and time-anchored shortening",
    "rag.cache.versions": "version keys that invalidate independent of TTL",
    "rag.compress.classifier": "Stage A rules, Stage B scoring, break-even test, learned scorer",
    "rag.compress.features": "chunk and query features the classifier reads",
    "rag.query.verify": "answer verifier: every number traceable, every citation valid",
    "rag.query.scope": "scope gate: out-of-scope refusals without retrieval",
}


def run(min_pct: float, write_report: bool) -> int:
    json_path = ROOT / ".coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *[f"--cov={m}" for m in MODULES],
        "--cov-report=term-missing",
        f"--cov-report=json:{json_path}",
        "tests/",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        print("\nTEST RUN FAILED — coverage not evaluated", file=sys.stderr)
        return proc.returncode
    data = json.loads(json_path.read_text(encoding="utf-8"))
    files = data["files"]
    rows: list[tuple[str, int, int, float, str]] = []
    for module, why in MODULES.items():
        rel = "src/" + module.replace(".", "/") + ".py"
        entry = next((v for k, v in files.items() if Path(k).as_posix().endswith(rel)), None)
        if entry is None:
            rows.append((module, 0, 0, 0.0, why))
            continue
        s = entry["summary"]
        rows.append((module, s["num_statements"], s["missing_lines"], s["percent_covered"], why))
    total = data["totals"]["percent_covered"]
    failing = [r for r in rows if r[3] < min_pct]

    print(f"\n{'module':32} {'stmts':>6} {'miss':>5} {'cover':>7}")
    for module, stmts, miss, pct, _ in rows:
        flag = "" if pct >= min_pct else "   <-- below gate"
        print(f"{module:32} {stmts:6d} {miss:5d} {pct:6.1f}%{flag}")
    print(f"{'TOTAL':32} {'':6} {'':5} {total:6.1f}%   (gate: every module >= {min_pct:g}%)")

    if write_report:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        lines = [
            "# Coverage gate — deterministic modules (Phase 8)",
            "",
            f"Generated {stamp} · `scripts/coverage_gate.py` · gate: every module ≥ {min_pct:g} % "
            f"line coverage · full suite under `pytest-cov` (no network, no model calls).",
            "",
            "Rule G4 of the implementation plan makes these modules test-first because a silent "
            "financial error would hide in them. The gate runs the whole suite and reads only "
            "these files' coverage; the rest of the code is exercised by the same run but is not "
            "gated.",
            "",
            "| Module | Statements | Missed | Coverage | Why it is gated |",
            "|---|---:|---:|---:|---|",
        ]
        for module, stmts, miss, pct, why in rows:
            mark = "✅" if pct >= min_pct else "❌"
            lines.append(f"| `{module}` | {stmts} | {miss} | {mark} {pct:.1f} % | {why} |")
        lines += [
            f"| **All gated modules** | {sum(r[1] for r in rows)} | {sum(r[2] for r in rows)} | "
            f"**{total:.1f} %** | |",
            "",
            (
                "Result: **gate passed** — every module clears the threshold."
                if not failing
                else "Result: **gate FAILED** for "
                + ", ".join(f"`{r[0]}` ({r[3]:.1f} %)" for r in failing)
                + "."
            ),
            "",
            "Reproduce: `make coverage` (or `.venv/Scripts/python scripts/coverage_gate.py`). "
            "The per-line `Missing` columns are in the terminal output of that run.",
        ]
        out = ROOT / "docs" / "reports" / "coverage_phase8.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nreport → {out.relative_to(ROOT)}")
    json_path.unlink(missing_ok=True)
    if failing:
        print(f"\nCOVERAGE GATE FAILED: {', '.join(r[0] for r in failing)}", file=sys.stderr)
        return 1
    print("\nCOVERAGE GATE PASSED")
    return 0


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min", type=float, default=90.0, help="minimum line coverage per module")
    ap.add_argument("--no-report", action="store_true", help="do not write the markdown report")
    args = ap.parse_args()
    return run(args.min, not args.no_report)


if __name__ == "__main__":
    sys.exit(main())
