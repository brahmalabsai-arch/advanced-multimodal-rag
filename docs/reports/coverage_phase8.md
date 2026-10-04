# Coverage gate — deterministic modules (Phase 8)

Generated 2026-09-22T15:51:47+00:00 · `scripts/coverage_gate.py` · gate: every module ≥ 90 % line coverage · full suite under `pytest-cov` (no network, no model calls).

Rule G4 of the implementation plan makes these modules test-first because a silent financial error would hide in them. The gate runs the whole suite and reads only these files' coverage; the rest of the code is exercised by the same run but is not gated.

| Module | Statements | Missed | Coverage | Why it is gated |
|---|---:|---:|---:|---|
| `rag.query.slots` | 228 | 4 | ✅ 98.2 % | slot extraction: periods, metrics, formulas, ask type, topics (cache keys) |
| `rag.calc.calculator` | 200 | 1 | ✅ 99.5 % | deterministic ratios and YoY; whitelisted AST evaluator |
| `rag.compress.fidelity` | 24 | 0 | ✅ 100.0 % | numeric fidelity guard on every compressor output |
| `rag.cache.records` | 95 | 1 | ✅ 98.9 % | cache guard keys and record schema (false-hit defence) |
| `rag.cache.admission` | 35 | 2 | ✅ 94.3 % | cache admission policy (§5.6) |
| `rag.cache.lfu` | 41 | 1 | ✅ 97.6 % | Redis-style LFU counter, decay and victim choice |
| `rag.cache.ttl` | 53 | 3 | ✅ 94.3 % | TTL classes and time-anchored shortening |
| `rag.cache.versions` | 40 | 0 | ✅ 100.0 % | version keys that invalidate independent of TTL |
| `rag.compress.classifier` | 160 | 7 | ✅ 95.6 % | Stage A rules, Stage B scoring, break-even test, learned scorer |
| `rag.compress.features` | 105 | 6 | ✅ 94.3 % | chunk and query features the classifier reads |
| `rag.query.verify` | 85 | 6 | ✅ 92.9 % | answer verifier: every number traceable, every citation valid |
| `rag.query.scope` | 73 | 2 | ✅ 97.3 % | scope gate: out-of-scope refusals without retrieval |
| **All gated modules** | 1139 | 33 | **97.1 %** | |

Result: **gate passed** — every module clears the threshold.

Reproduce: `make coverage` (or `.venv/Scripts/python scripts/coverage_gate.py`). The per-line `Missing` columns are in the terminal output of that run.
