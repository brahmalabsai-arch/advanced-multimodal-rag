# Retrieval ablation — Phase 4

Generated 2026-09-18T09:11+00:00 · corpus `747912fc5da6` · reranker `BAAI/bge-reranker-base` · k = 8 · 46 in-scope golden questions (OUT_OF_SCOPE excluded: the scope gate refuses them before retrieval).

Each arm adds one component on top of the previous one (architecture §4.4–4.6). Scores are
against the golden set's labelled supporting chunks; no answers are generated here.

| Arm | recall@8 | MRR | hit rate | queries / q | Groq calls / q | latency p50 (ms) | reranked |
|---|---|---|---|---|---|---|---|
| `dense` | **0.52** | 0.37 | 0.59 | 1.0 | 0.00 | 22 | 0 |
| `+bm25` | **0.73** | 0.46 | 0.78 | 1.0 | 0.00 | 10 | 0 |
| `+expansion-nf` | **0.93** | 0.86 | 1.00 | 2.0 | 0.00 | 33 | 0 |
| `+expansion` | **0.95** | 0.91 | 1.00 | 2.0 | 0.00 | 45 | 0 |
| `+llm` | **0.95** | 0.91 | 1.00 | 2.2 | 0.17 | 63 | 0 |
| `+rerank` | **0.91** | 0.86 | 0.98 | 2.2 | 0.00 | 77 | 15 |
| `rerank-always` | **0.86** | 0.69 | 0.96 | 2.2 | 0.00 | 5217 | 0 |
| `rerank-minilm` | **0.88** | 0.72 | 0.94 | 2.2 | 0.00 | 1176 | 0 |

## recall@8 by intent

| Arm | COMPARISON_TREND | COMPUTATION | CROSS_SECTION | EXPLANATORY | POINT_LOOKUP | VISUAL |
|---|---|---|---|---|---|---|
| `dense` | 0.29 | 0.25 | 0.81 | 0.50 | 0.62 | 0.83 |
| `+bm25` | 0.29 | 0.44 | 0.83 | 1.00 | 0.88 | 1.00 |
| `+expansion-nf` | 1.00 | 0.77 | 0.72 | 1.00 | 0.97 | 1.00 |
| `+expansion` | 1.00 | 0.90 | 0.72 | 1.00 | 0.97 | 1.00 |
| `+llm` | 1.00 | 0.90 | 0.72 | 1.00 | 0.97 | 1.00 |
| `+rerank` | 0.86 | 0.94 | 0.64 | 0.90 | 0.97 | 0.83 |
| `rerank-always` | 0.71 | 0.75 | 0.64 | 0.90 | 0.97 | 0.83 |
| `rerank-minilm` | 0.71 | 0.78 | 0.72 | 0.90 | 0.97 | 1.00 |

## Per-question recall@8

| id | intent | `dense` | `+bm25` | `+expansion-nf` | `+expansion` | `+llm` | `+rerank` | `rerank-always` | `rerank-minilm` | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| G2 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P3 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P4 | POINT_LOOKUP | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P5 | POINT_LOOKUP | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P6 | POINT_LOOKUP | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P7 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P8 | POINT_LOOKUP | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P9 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P10 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P11 | POINT_LOOKUP | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P12 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P13 | POINT_LOOKUP | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P14 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P15 | POINT_LOOKUP | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P16 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P17 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P18 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P19 | POINT_LOOKUP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| P20 | POINT_LOOKUP | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | S1 |
| G3 | COMPUTATION | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| G4 | COMPUTATION | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | S1 |
| G5 | COMPUTATION | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.75 | rerank |
| G6 | COMPUTATION | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | rerank |
| C5 | COMPUTATION | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | S1 |
| C6 | COMPUTATION | 0.50 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | S1 |
| C7 | COMPUTATION | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| C8 | COMPUTATION | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | S1 |
| G7 | COMPARISON_TREND | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | rerank |
| G8 | COMPARISON_TREND | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S3 |
| G9 | COMPARISON_TREND | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S3 |
| G10 | COMPARISON_TREND | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | S3 |
| T5 | COMPARISON_TREND | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| T6 | COMPARISON_TREND | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| T7 | COMPARISON_TREND | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | S3 |
| G11 | EXPLANATORY | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| E2 | EXPLANATORY | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | rerank |
| E3 | EXPLANATORY | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 1.00 | rerank |
| E4 | EXPLANATORY | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| E5 | EXPLANATORY | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | S1 |
| G12 | VISUAL | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| G13 | VISUAL | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |
| V3 | VISUAL | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 1.00 | rerank |
| X1 | CROSS_SECTION | 0.75 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.25 | 0.50 | rerank |
| X2 | CROSS_SECTION | 0.67 | 1.00 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | rerank |
| X3 | CROSS_SECTION | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | rerank |

## Reranker score calibration (`rerank-always` arm, all pool candidates)

- gold chunks: n = 66, min -3.59, p10 0.0, median 4.44
- other chunks: n = 1314, median 0.31, p90 4.41
- `drop_floor_logit` = -2.0: 3 gold and 391 other candidates fall below it

## Small-model arm

- questions with a model call: 4 of 46 (rule-confident questions add zero calls, plan Phase 4)
- HyDE passages used: 2; rejected for digits: 0

## Reading the numbers (decisions recorded in architecture §16)

- **Expansion is the win** (D-57): phrasing each metric / formula input the way its row-fact document reads, filtered to the statement, takes recall@8 from 0.73 to 0.95 and MRR from 0.46 to 0.91 with zero model calls. The filter itself is worth +0.02 recall / +0.05 MRR over unfiltered expansion.
- **HyDE / paraphrases** (D-23): no measurable change (0.95 vs 0.95) at 0.17 `small` calls per question — EXPLANATORY recall is already 1.00 after hybrid retrieval. Off by default (`expansion.hyde: false`).
- **Reranking** (D-24): gated reranking lowers recall@8 to 0.91 (ungated 0.86, MiniLM 0.88) and costs ≈ 5.2 s per reranked question on this CPU; the S1–S3 gate protects the numeric questions but the narrative ones it does rerank lose labelled chunks. Off by default (`rerank.enabled: false`); the gate, the reranker and this arm stay so a larger corpus can re-test.
