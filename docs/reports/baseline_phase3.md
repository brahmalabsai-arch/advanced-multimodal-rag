# Baseline metrics — Phase 3 (before expansion, reranking, compression, cache)

Run `baseline_phase3` · 2026-09-18T06:20:27+00:00 · index `index` (BAAI/bge-small-en-v1.5, corpus `747912fc5da6`) · k=8 · 48 questions · 9.4 s · bypass_cache=true

## Headline

| Metric | Value |
|---|---|
| Numeric exact match (value + period + unit) | 94% of 35 |
| Value found (ignoring period/unit wording) | 97% |
| Calculator agrees with golden value | 93% |
| Keyword coverage (text questions, indicative) | 82% of 11 |
| Verifier pass rate | 98% |
| recall@8 / MRR / hit rate | 0.73 / 0.46 / 0.78 |
| Latency p50 | 29690 ms (includes client-side pacing waits under the Groq free-tier TPM limit; NFR-5 excludes those) |
| Groq tokens in / out | 144,183 / 25,205 |

## By intent

| Intent | n | exact | value | calc | kw | verify | recall@8 | MRR | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| COMPARISON_TREND | 7 | 86% | 100% | 86% | – | 100% | 0.29 | 0.08 | 32920 |
| COMPUTATION | 8 | 100% | 100% | 100% | – | 100% | 0.44 | 0.23 | 28868 |
| CROSS_SECTION | 3 | – | – | – | 83% | 100% | 0.83 | 0.83 | 33474 |
| EXPLANATORY | 5 | – | – | – | 80% | 100% | 1.00 | 0.72 | 31318 |
| OUT_OF_SCOPE | 2 | – | – | – | – | 100% | – | – | 29270 |
| POINT_LOOKUP | 20 | 95% | 95% | – | – | 100% | 0.88 | 0.49 | 28466 |
| VISUAL | 3 | – | – | – | 83% | 67% | 1.00 | 1.00 | 34856 |

## Per question

| id | intent | exact / kw | value | period | unit | calc | verify | recall | attempts | ms | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 2471 |  |
| G2 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 1501 |  |
| P3 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 13362 |  |
| P4 | POINT_LOOKUP | ✗ | ✗ | ✓ | ✓ |  | ✓ | 0.00 | 2 | 61317 |  |
| P5 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 28845 |  |
| P6 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 28350 |  |
| P7 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 23138 |  |
| P8 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 32395 |  |
| P9 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 26465 |  |
| P10 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 25587 |  |
| P11 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 33068 |  |
| P12 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 27743 |  |
| P13 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 0.00 | 1 | 33345 |  |
| P14 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 27259 |  |
| P15 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 29893 |  |
| P16 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 24795 |  |
| P17 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 28540 |  |
| P18 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 28466 |  |
| P19 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 30397 |  |
| P20 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 0.50 | 1 | 29690 |  |
| G3 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 28868 |  |
| G4 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.50 | 1 | 23850 |  |
| G6 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 26596 |  |
| C5 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 34896 |  |
| C6 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.50 | 1 | 29006 |  |
| C7 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.50 | 1 | 24074 |  |
| C8 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 32676 |  |
| G7 | COMPARISON_TREND | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | 0.00 | 1 | 32985 |  |
| G8 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 31627 |  |
| G9 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 33761 |  |
| G10 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 32663 |  |
| T5 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 32920 |  |
| T6 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 1.00 | 1 | 33081 |  |
| G11 | EXPLANATORY | 75% |  |  |  |  | ✓ | 1.00 | 1 | 29257 |  |
| E2 | EXPLANATORY | 50% |  |  |  |  | ✓ | 1.00 | 1 | 33018 |  |
| E3 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 32013 |  |
| E4 | EXPLANATORY | 75% |  |  |  |  | ✓ | 1.00 | 1 | 28420 |  |
| E5 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 31318 |  |
| G12 | VISUAL | 100% |  |  |  |  | ✓ | 1.00 | 1 | 34856 |  |
| G13 | VISUAL | 100% |  |  |  |  | ✓ | 1.00 | 1 | 30417 |  |
| V3 | VISUAL | 50% |  |  |  |  | ✗ | 1.00 | 2 | 61790 | Verification failed after regeneration: Numbers not found in |
| X1 | CROSS_SECTION | 50% |  |  |  |  | ✓ | 0.50 | 1 | 33795 |  |
| X2 | CROSS_SECTION | 100% |  |  |  |  | ✓ | 1.00 | 1 | 33083 |  |
| X3 | CROSS_SECTION | 100% |  |  |  |  | ✓ | 1.00 | 1 | 33474 |  |
| G14 | OUT_OF_SCOPE | – |  |  |  |  | ✓ | – | 1 | 29270 |  |
| O2 | OUT_OF_SCOPE | – |  |  |  |  | ✓ | – | 1 | 29629 |  |
| G5 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 3175 |  |
| T7 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.00 | 1 | 6167 |  |

## Misses (answer excerpts)

- **P4** (POINT_LOOKUP): The filing does not state a single total amount for marketable securities on the balance sheet as of the end of fiscal 2026. It provides component amounts (e.g., $21,709 million for Treasury debt, $15,154 million for corporate debt, $2,161 million for government‑agency debt) but no aggregate figure 
- **G7** (COMPARISON_TREND): **Answer:** Total assets grew by $95,202 million, an increase of +85.3% compared with fiscal 2025.  The absolute increase was $95,202 million and the percentage change was +85.3% year‑over‑year.
