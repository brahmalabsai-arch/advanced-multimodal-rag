# RAGAS-style evaluation (Phase 7)

Generated 2026-09-19T06:02:27+00:00 · `eval/ragas_eval.py` · 16 golden questions · profile `anthropic` · generator `claude-sonnet-5` · judge `claude-sonnet-5`.

> **Indicative, same-family judge.** The judge and the generator come from the same model family and see the same context, so these numbers are a smoke test of grounding, not an independent audit. The golden reference answers are themselves still under review (`review_status` in `eval/golden.jsonl`). The `ragas` package is in `requirements-dev.txt` but could not be imported here — it imports `langchain_community.chat_models.vertexai`, removed from the sunset `langchain-community` 0.4.2 — so the four metric definitions are implemented directly in this script against the project's own `LLMClient`.

## Scores

| metric | mean | scored | definition |
|---|---|---|---|
| faithfulness | **0.994** | 13/16 | answer claims the retrieved context supports |
| answer relevancy | **0.82** | 16/16 | cosine(question, questions reverse-generated from the answer) |
| context precision | **0.705** | 16/16 | judged usefulness of each retrieved block, as precision@k |
| context recall | **0.863** | 16/16 | reference-answer claims the retrieved context supports |

**3 metric(s) could not be scored** and are excluded from the means above rather than counted as zero:

- `E4` faithfulness: no claims extracted
- `G13` faithfulness: role=large model=claude-sonnet-5 returned invalid JSON twice: 1 validation error for Verdicts
  Invalid JSON: EOF while parsing a string at line 1 column 686 [type=json_invalid, input_value='{"verdict
- `X1` faithfulness: no claims extracted

## By intent

| intent | n | faithfulness | relevancy | ctx precision | ctx recall |
|---|---|---|---|---|---|
| COMPARISON_TREND | 2 | 1.000 | 0.814 | 0.837 | 1.000 |
| COMPUTATION | 4 | 1.000 | 0.870 | 0.803 | 1.000 |
| CROSS_SECTION | 2 | 1.000 | 0.761 | 0.669 | 0.484 |
| EXPLANATORY | 4 | 0.972 | 0.758 | 0.412 | 0.875 |
| POINT_LOOKUP | 2 | 1.000 | 0.881 | 1.000 | 1.000 |
| VISUAL | 2 | 1.000 | 0.849 | 0.699 | 0.666 |

## Reading the numbers

- **Faithfulness is near 1.0 by construction.** `verify_answer` (§4.12) already refuses to return an answer whose numbers are not traceable to the context, so this metric mostly confirms the verifier works; a value below 1.0 flags a *qualitative* sentence the judge could not tie to a block, not an invented figure.
- **Context precision is the metric with something to say.** The lowest scores are `E2` (0.17), `E4` (0.21), `E3` (0.32) — the retriever returns eight blocks and, on explanatory questions, the judge considers most of them unnecessary for the reference answer. That is the padding the compression classifier exists to remove, and it is also why `final_k` is worth revisiting per intent rather than globally.
- **Context recall below 1.0** means a claim in the *reference* answer was not in the retrieved context at all: a retrieval miss, not a generation fault. Those rows are the ones to read alongside the golden-set human review.

## Per question

| id | intent | faithfulness | relevancy | ctx precision | ctx recall | claims | notes |
|---|---|---|---|---|---|---|---|
| C5 | COMPUTATION | 1.0 | 0.893 | 0.678 | 1.0 | 5 | – |
| C7 | COMPUTATION | 1.0 | 0.914 | 0.799 | 1.0 | 3 | – |
| E2 | EXPLANATORY | 0.917 | 0.895 | 0.174 | 1.0 | 12 | – |
| E3 | EXPLANATORY | 1.0 | 0.903 | 0.321 | 1.0 | 12 | – |
| E4 | EXPLANATORY | None | 0.582 | 0.21 | 1.0 | 0 | no claims extracted |
| G1 | POINT_LOOKUP | 1.0 | 0.969 | 1.0 | 1.0 | 2 | – |
| G11 | EXPLANATORY | 1.0 | 0.652 | 0.943 | 0.5 | 7 | – |
| G12 | VISUAL | 1.0 | 0.833 | 0.592 | 1.0 | 12 | – |
| G13 | VISUAL | None | 0.865 | 0.806 | 0.333 | 9 | role=large model=claude-sonnet-5 returned invalid JSON twice: 1 validation error for Verdicts
  Invalid JSON: EOF while parsing a string at line 1 column 686 [type=json_invalid, input_value='{"verdict |
| G3 | COMPUTATION | 1.0 | 0.792 | 1.0 | 1.0 | 4 | – |
| G5 | COMPUTATION | 1.0 | 0.879 | 0.736 | 1.0 | 6 | – |
| G7 | COMPARISON_TREND | 1.0 | 0.841 | 0.861 | 1.0 | 4 | – |
| G9 | COMPARISON_TREND | 1.0 | 0.786 | 0.814 | 1.0 | 6 | – |
| P4 | POINT_LOOKUP | 1.0 | 0.794 | 1.0 | 1.0 | 3 | – |
| X1 | CROSS_SECTION | None | 0.605 | 0.771 | 0.167 | 0 | no claims extracted |
| X3 | CROSS_SECTION | 1.0 | 0.917 | 0.567 | 0.8 | 7 | – |

## Claims the judge did not find in the context

- `E2` — The Hopper-to-Blackwell product mix transition carries a different cost/margin profile.

These are worth reading before trusting the score: the verifier (§4.12) already guarantees every *number* is traceable, so a low faithfulness score here usually means the judge disliked a qualitative sentence, not that a figure was invented.
