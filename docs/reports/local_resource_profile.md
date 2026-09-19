# Local resource profile (Phase 7)

Generated 2026-09-19T06:05:49+00:00 · `eval/resource_profile.py` · 50 mixed queries (30 cache misses, 20 hits) · profile `anthropic` · corpus `747912fc5da6` (689 chunks) · one process, one worker.

Input for the deployment decision deferred in §14.4: this is what one laptop process pays to hold the whole system in memory and answer from it.

> **Provider note.** This run used `anthropic`, not the build profile `groq_build`, because the Groq free tier's daily token window was already spent on the RAGAS judge (see `docs/reports/ragas_groq.md`). Startup, RSS and every node except `generate` are provider-independent; for the free-tier generation latency read `latency_ms_p50` per intent in `eval/results/golden_phase4.summary.json` and the walkthrough report.

## 1. Startup

| stage | seconds | RSS after (MB) |
|---|---|---|
| import rag.graph | 2.51 | 138.1 |
| Pipeline() — index, embedder, BM25, cache | 3.04 | 385.9 |
| **total to first request** | **5.55** | — |

RSS before any import: 46.2 MB. Peak RSS during the run: **478.8 MB** (Python-object peak measured by `tracemalloc`: 122.8 MB — the rest is the ONNX runtime, the embedding model and NumPy buffers).

## 2. Latency

| stream | n | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|
| all requests | 50 | 3161.0 | 9815.0 | 20656.0 |
| cache hits | 20 | 40.0 | 61.0 | 62.0 |
| cache misses (full pipeline) | 30 | 4773.0 | 11795.0 | 20656.0 |

### Per node

| node | n | p50 (ms) | p95 (ms) |
|---|---|---|---|
| analyze | 30 | 0.0 | 0.0 |
| assemble | 30 | 1.0 | 3.0 |
| cache_lookup | 50 | 1.0 | 2.0 |
| cache_write | 30 | 88.0 | 119.0 |
| calculate | 30 | 0.0 | 0.0 |
| compress | 30 | 12.0 | 21.0 |
| generate | 30 | 4567.0 | 11527.0 |
| rerank | 30 | 0.0 | 0.0 |
| retrieve | 30 | 26.0 | 47.0 |
| scope | 30 | 0.0 | 0.0 |
| slots | 50 | 2.0 | 3.0 |
| verify | 30 | 2.0 | 11.0 |

## 3. Cost of the stream

- model calls: **32** over 50 requests (20 served from cache at zero calls)
- model tokens: **136,308**
- mean context sent per miss: 1617 tokens

## 4. What this run changed (post-measurement)

`cache_write` at p50 88 ms stood out: `make_room` read the whole collection on *every* write,
including writes with plenty of capacity. It now checks `count()` first and only scans under
capacity pressure (`src/rag/cache/l2.py`, regression test in `tests/test_cache.py`). Measured
directly on a synthetic L2 afterwards:

| situation | entries | write p50 |
|---|---|---|
| under capacity (the normal case) | 200 | 15.0 ms |
| under capacity | 1,000 | 10.4 ms |
| at capacity, evicting on every write | 1,000 | 182.1 ms |

The last row is the price of the "exact rather than sampled" eviction choice (§5.9): it scales
linearly, so at the configured `L2_MAX_ENTRIES = 5,000` a full cache would cost roughly a second
per write. That is acceptable while the working set stays far below capacity — the policy
benchmark measured capacities 500 and 1,000 as non-binding on this corpus — and it is the first
thing to change if the cache ever runs full.

**Pending.** The table in §2 is the pre-fix run: the `cache_write` column would now read ~10 ms
rather than 88 ms, and nothing else about it changes (the fix touches one code path). The
re-run to confirm that end-to-end is blocked on a model window — on 2026-09-19 the Groq free
tier's daily token budget was spent and the Anthropic account ran out of credit mid-run. The
script now refuses to overwrite this report when most requests fail at the model call, which is
how that attempt was caught.
