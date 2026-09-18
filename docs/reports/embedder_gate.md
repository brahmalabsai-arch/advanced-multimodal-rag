# Embedder gate — bge-small vs bge-base (Phase 3)

Generated 2026-09-17T17:14:40+00:00 · retrieval-only, dense top-30 ⊕ BM25 top-30 → RRF → top-8 · golden set n=46

**Decision: `bge-small`** — bge-base gain is -5.1 points (< 3.0); keep the smaller, faster model. Rule (D-47): prefer bge-small unless bge-base improves recall@8 by ≥ 3 points.

## Overall

| Embedder | Dim | corpus_version | recall@8 | MRR | hit rate | elapsed s |
|---|---|---|---|---|---|---|
| bge-small | 384 | `747912fc5da6` | 0.728 | 0.464 | 0.783 | 0.3 |
| bge-base | 768 | `ffcd60acbd8a` | 0.677 | 0.452 | 0.761 | 0.7 |

## By intent

| Intent | n | bge-small recall@8 / MRR | bge-base recall@8 / MRR |
|---|---|---|---|
| COMPARISON_TREND | 7 | 0.29 / 0.08 | 0.29 / 0.08 |
| COMPUTATION | 8 | 0.44 / 0.23 | 0.29 / 0.11 |
| CROSS_SECTION | 3 | 0.83 / 0.83 | 0.61 / 0.46 |
| EXPLANATORY | 5 | 1.00 / 0.72 | 0.90 / 0.68 |
| POINT_LOOKUP | 20 | 0.88 / 0.49 | 0.88 / 0.57 |
| VISUAL | 3 | 1.00 / 1.00 | 1.00 / 1.00 |

## Notes

- Chunk boundaries are identical for both indexes (D-53); only the vectors differ, so this is a pure embedder comparison.
- Retrieval here is the Phase 3 baseline (single query, no expansion, no filters, no reranking). Weak spots on COMPUTATION / COMPARISON_TREND questions are addressed in Phase 4 and measured in the retrieval ablation.
- Thresholds calibrated later (rerank floor, cache similarity) are specific to the locked embedder.
