# Golden run — Phase 4 pipeline (claude-sonnet-5 generator)

Run `golden_phase4_claude` · 2026-09-18T15:08:59+00:00 · index `index` (BAAI/bge-small-en-v1.5, corpus `747912fc5da6`) · k=8 · 48 questions · 290.7 s · bypass_cache=true


> **Why this run exists.** The Phase 4 exit criterion "no regression on G1–G10" needs a full golden run through the new pipeline, and the Groq `gpt-oss-120b` daily window could not carry it (the Groq run `golden_phase4` reached 34/48 over a day). At the user's request the run was repeated on the `anthropic` profile (`MODEL_PROFILE=anthropic`), a one-off deviation from the Groq-only build rule (G2 / D-45) recorded in the decision log. Retrieval, calculator, verifier and prompts are identical; only the generator differs (`context_budget_tokens` 6,000 vs 2,500 on Groq).
>
> **Verifier note.** The three COMPUTATION verify failures (G6, C5, C6) are answers that restate a ratio as a percentage ("0.054 (i.e. 5.4%)"); the numbers are right and the calculator agreed. `verify.number_is_supported` now accepts the percent form of an allowed ratio, so these pass on re-run.

## Headline

| Metric | Value |
|---|---|
| Numeric exact match (value + period + unit) | 100% of 35 |
| Value found (ignoring period/unit wording) | 100% |
| Calculator agrees with golden value | 100% |
| Keyword coverage (text questions, indicative) | 100% of 11 |
| Verifier pass rate | 94% |
| recall@8 / MRR / hit rate | 0.95 / 0.91 / 1.00 |
| Latency p50 | 4444 ms (no client-side pacing on this profile) |
| Generator tokens in / out (Anthropic, `claude-sonnet-5` / `claude-haiku-4-5`) | 203,409 / 23,030 |

## By intent

| Intent | n | exact | value | calc | kw | verify | recall@8 | MRR | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| COMPARISON_TREND | 7 | 100% | 100% | 100% | – | 100% | 1.00 | 1.00 | 5263 |
| COMPUTATION | 8 | 100% | 100% | 100% | – | 62% | 0.90 | 0.73 | 4904 |
| CROSS_SECTION | 3 | – | – | – | 100% | 100% | 0.72 | 1.00 | 15604 |
| EXPLANATORY | 5 | – | – | – | 100% | 100% | 1.00 | 0.72 | 8240 |
| OUT_OF_SCOPE | 2 | – | – | – | – | 100% | – | – | 1 |
| POINT_LOOKUP | 20 | 100% | 100% | – | – | 100% | 0.97 | 0.97 | 3606 |
| VISUAL | 3 | – | – | – | 100% | 100% | 1.00 | 1.00 | 13710 |

## Per question

| id | intent | exact / kw | value | period | unit | calc | verify | recall | attempts | ms | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3966 |  |
| G2 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 4601 |  |
| P3 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 4108 |  |
| P4 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3490 |  |
| P5 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3606 |  |
| P6 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 5122 |  |
| P7 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3058 |  |
| P8 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3952 |  |
| P9 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 4279 |  |
| P10 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3815 |  |
| P11 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3678 |  |
| P12 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3574 |  |
| P13 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 2854 |  |
| P14 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3325 |  |
| P15 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3220 |  |
| P16 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3506 |  |
| P17 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3394 |  |
| P18 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3290 |  |
| P19 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 1.00 | 1 | 3358 |  |
| P20 | POINT_LOOKUP | ✓ | ✓ | ✓ | ✓ |  | ✓ | 0.50 | 1 | 3834 |  |
| G3 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4016 |  |
| G4 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4252 |  |
| G5 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.50 | 1 | 4904 |  |
| G6 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 0.67 | 2 | 9504 | Verification failed after regeneration: Numbers not found in |
| C5 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 1.00 | 2 | 9006 | Verification failed after regeneration: Numbers not found in |
| C6 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 1.00 | 2 | 9348 | Verification failed after regeneration: Numbers not found in |
| C7 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4061 |  |
| C8 | COMPUTATION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4141 |  |
| G7 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 5263 |  |
| G8 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 5364 |  |
| G9 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4950 |  |
| G10 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 5355 |  |
| T5 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 4498 |  |
| T6 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 5220 |  |
| T7 | COMPARISON_TREND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1.00 | 1 | 6322 |  |
| G11 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 5580 |  |
| E2 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 8792 |  |
| E3 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 8240 |  |
| E4 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 10552 |  |
| E5 | EXPLANATORY | 100% |  |  |  |  | ✓ | 1.00 | 1 | 4611 |  |
| G12 | VISUAL | 100% |  |  |  |  | ✓ | 1.00 | 2 | 25568 |  |
| G13 | VISUAL | 100% |  |  |  |  | ✓ | 1.00 | 1 | 10498 |  |
| V3 | VISUAL | 100% |  |  |  |  | ✓ | 1.00 | 1 | 13710 |  |
| X1 | CROSS_SECTION | 100% |  |  |  |  | ✓ | 0.50 | 1 | 22366 |  |
| X2 | CROSS_SECTION | 100% |  |  |  |  | ✓ | 0.67 | 2 | 15604 |  |
| X3 | CROSS_SECTION | 100% |  |  |  |  | ✓ | 1.00 | 1 | 4444 |  |
| G14 | OUT_OF_SCOPE | – |  |  |  |  | ✓ | – | 0 | 1 |  |
| O2 | OUT_OF_SCOPE | – |  |  |  |  | ✓ | – | 0 | 1 |  |
