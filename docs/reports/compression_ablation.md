# Compression ablation — Phase 5

Generated 2026-09-18T15:40+00:00 · profile `anthropic` · context budget 2500 tokens (the groq_build production budget) · golden set · same retrieval, calculator, verifier and prompts in every arm; only the compression node differs.

Runs: `never_compress` → `compress_never`, `classifier_gated` → `compress_classifier`, `always_compress` → `compress_always`

## All in-scope questions (never vs classifier)

| Arm | n | exact | kw | verify | large input tok / q | context tok / q | compressor LLM calls | queries compressed | fidelity violations | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `never_compress` | 48 | 100% | 93% | 100% | 3744 | 1380 | 0 | 0 | 0 | 4281 / 11557 |
| `classifier_gated` | 48 | 100% | 91% | 96% | 3858 | 1315 | 0 | 15 | 0 | 4239 / 17897 |

## 20-question subset (all three arms)

| Arm | n | exact | kw | verify | large input tok / q | context tok / q | compressor LLM calls | queries compressed | fidelity violations | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `never_compress` | 20 | 100% | 92% | 100% | 4388 | 1654 | 0 | 0 | 0 | 4633 / 11557 |
| `classifier_gated` | 20 | 100% | 89% | 95% | 4620 | 1624 | 0 | 3 | 0 | 4852 / 21690 |
| `always_compress` | 20 | 100% | 72% | 95% | 3965 | 1418 | 47 | 15 | 0 | 9044 / 15781 |

## Applied actions (classifier arm, all questions)

- KEEP: 376
- ROW_SELECT: 18

- candidate tokens before → after compression: 55,805 → 52,677 (6% fewer)

## Fidelity

- violations caught by the guard across all arms: 0 (each reverted to the original chunk before generation)

## Reading the numbers (D-58)

- **Numeric accuracy is untouched by compression** (100% exact in every arm): numbers reach the generator through calculator blocks and protected row facts, never through a compressor (K2, P3).
- **`always_compress` is the case for gating**: keyword coverage 72% vs 92% on the same subset, 47 small-model calls and p50 latency 9.0 s vs 4.6 s.
- **`classifier_gated`** keeps exact match at 100%, compresses 15 of 48 queries (actions: ROW_SELECT 18), cuts context tokens per question 1380 → 1315 (5%) with 0 compressor calls and 0 fidelity violations. Large-model input per *attempt* falls in step; the per-question total also counts regenerations, which vary run to run.
- **Thresholds adjusted from the first gated run** (kept for the record): `dedupe_cosine` 0.95 → 0.98 because the three H20 disclosures sit at 0.94–0.96 cosine yet are paraphrases of different length (merging them lost the "license" wording, G11), and R5 now cuts narrative only when the candidates exceed the budget (`narrative_pressure_ratio` 1.0) because cutting a context that already fit saved 1.4% of large-model input and dropped a keyword on V3. On a 2,500-token budget with eight candidates that leaves ROW_SELECT as the working compressor; the LLM extractor is reached only under real budget pressure (longer contexts, more candidates), which the Phase 7 benchmark can exercise.
- **Weak feature, recorded honestly**: bge-small sentence-to-query cosine does not separate labelled from unlabelled narrative sentences (medians 0.60 vs 0.61), so `relevance_density` carries little information on this corpus; Stage C (Phase 7) should re-weight it from the decision log.
