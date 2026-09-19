# Generator check — gpt-oss-120b vs qwen3.8-27b (Phase 3 task, run in Phase 4)

Generated 2026-09-18T12:47+00:00 · 15-question subset (5 POINT_LOOKUP, 4 COMPUTATION, 3 COMPARISON_TREND, 2 EXPLANATORY, 1 CROSS_SECTION) · same Phase 4 retrieval, calculator and verifier; only the `large` role differs (`groq_qwen_large` profile in `config/models.yaml`, `reasoning_effort: low`).

gpt-oss-120b rows come from the Phase 4 golden run `golden_phase4` where it had reached the question, otherwise from `baseline_phase3` (marked †; Phase 3 retrieval, same generator).

| id | intent | gpt-oss-120b exact / kw | attempts | latency | qwen3.8-27b exact / kw | attempts | latency |
|---|---|---|---|---|---|---|---|
| G1 | POINT_LOOKUP | Y | 1 | 3.0 s | Y | 1 | 1.5 s |
| G2 | POINT_LOOKUP | Y | 1 | 1.4 s | Y | 1 | 60.4 s |
| P4 | POINT_LOOKUP | Y | 1 | 24.5 s | Y | 1 | 60.2 s |
| P14 | POINT_LOOKUP | Y | 1 | 22.1 s | Y | 1 | 59.9 s |
| P17 | POINT_LOOKUP | Y | 1 | 22.5 s | Y | 1 | 59.9 s |
| G3 | COMPUTATION | Y | 1 | 99.5 s | Y | 1 | 59.9 s |
| G5 | COMPUTATION | Y | 1 | 2.7 s | Y | 2 | 120.7 s |
| G6 | COMPUTATION | Y | 1 | 2.7 s | Y | 2 | 119.8 s |
| C8 | COMPUTATION | Y | 1 | 2.4 s | Y | 1 | 59.5 s |
| G7 | COMPARISON_TREND | Y | 1 | 2.3 s | Y | 1 | 60.3 s |
| G8 | COMPARISON_TREND | Y† | 1 | 31.6 s | Y | 1 | 60.1 s |
| T7 | COMPARISON_TREND | Y† | 1 | 6.2 s | Y | 1 | 59.9 s |
| G11 | EXPLANATORY | kw 0.75† | 1 | 29.3 s | kw 1.00 | 1 | 60.2 s |
| E2 | EXPLANATORY | kw 0.50† | 1 | 33.0 s | kw 1.00 | 1 | 59.9 s |
| X3 | CROSS_SECTION | kw 1.00† | 1 | 33.5 s | kw 1.00 | 1 | 59.6 s |

† from `baseline_phase3`

## Summary

| | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| correct (exact match or keyword pass) | 15 / 15 | 15 / 15 |
| regenerations needed | 0 | 2 |
| latency p50 | 22.1 s | 59.9 s |
| tokens (in + out) | 43,548 | 53,038 |

## Reading

- Accuracy is tied on this subset: both generators get every numeric question exact and every narrative question past the keyword check, because the numbers come from the calculator blocks and the row facts, not from the generator (P3 / D-26).
- Qwen needed a regeneration on two computations (G5, G6) where its first answer failed the verifier; gpt-oss-120b passed those first time.
- The decisive difference is operational: Groq paces qwen3.8-27b at 1,000 *output* tokens per minute charged at the requested `max_tokens`, so every answer waits ≈ 60 s (a regeneration ≈ 120 s); gpt-oss-120b has no OTPM bucket and answers in 2–25 s including TPM pacing waits. Qwen also shares its daily budget with figure annotation and VISUAL answers.
- **Decision:** keep `openai/gpt-oss-120b` as the `large` role (D-50 unchanged); `groq_qwen_large` stays in `models.yaml` for re-checks. The Anthropic-vs-Gemini bake-off remains deferred to F1 (D-38).
