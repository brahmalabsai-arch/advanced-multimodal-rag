# RAGAS-style evaluation (Phase 7)

Generated 2026-09-19T06:02:39+00:00 · `eval/ragas_eval.py` · 4 golden questions · profile `groq_build` · generator `openai/gpt-oss-120b` · judge `openai/gpt-oss-120b`.

> **Indicative, same-family judge.** The judge and the generator come from the same model family and see the same context, so these numbers are a smoke test of grounding, not an independent audit. The golden reference answers are themselves still under review (`review_status` in `eval/golden.jsonl`). The `ragas` package is in `requirements-dev.txt` but could not be imported here — it imports `langchain_community.chat_models.vertexai`, removed from the sunset `langchain-community` 0.4.2 — so the four metric definitions are implemented directly in this script against the project's own `LLMClient`.

> **Partial run.** Only 4 of the 16 subset questions were judged on `groq_build`: the free tier's rolling 200,000-token daily window closed mid-run (198,953 of 200,000 used that day). The full 16-question subset was judged on the `anthropic` profile instead — `docs/reports/ragas_claude.md` — under the same deviation the Phase 4-6 evaluations used (D-45 note). Re-run `eval/ragas_eval.py --name ragas_groq --resume` when the window reopens.

## Scores

| metric | mean | scored | definition |
|---|---|---|---|
| faithfulness | **1.0** | 4/4 | answer claims the retrieved context supports |
| answer relevancy | **0.777** | 4/4 | cosine(question, questions reverse-generated from the answer) |
| context precision | **0.919** | 4/4 | judged usefulness of each retrieved block, as precision@k |
| context recall | **1.0** | 4/4 | reference-answer claims the retrieved context supports |

## By intent

| intent | n | faithfulness | relevancy | ctx precision | ctx recall |
|---|---|---|---|---|---|
| COMPUTATION | 2 | 1.000 | 0.850 | 0.839 | 1.000 |
| POINT_LOOKUP | 2 | 1.000 | 0.704 | 1.000 | 1.000 |

## Reading the numbers

- **Faithfulness is near 1.0 by construction.** `verify_answer` (§4.12) already refuses to return an answer whose numbers are not traceable to the context, so this metric mostly confirms the verifier works; a value below 1.0 flags a *qualitative* sentence the judge could not tie to a block, not an invented figure.
- **Context precision is the metric with something to say.** The lowest scores are `C5` (0.68), `G1` (1.00), `G3` (1.00) — the retriever returns eight blocks and, on explanatory questions, the judge considers most of them unnecessary for the reference answer. That is the padding the compression classifier exists to remove, and it is also why `final_k` is worth revisiting per intent rather than globally.
- **Context recall below 1.0** means a claim in the *reference* answer was not in the retrieved context at all: a retrieval miss, not a generation fault. Those rows are the ones to read alongside the golden-set human review.

## Per question

| id | intent | faithfulness | relevancy | ctx precision | ctx recall | claims | notes |
|---|---|---|---|---|---|---|---|
| C5 | COMPUTATION | 1.0 | 0.892 | 0.678 | 1.0 | 4 | – |
| G1 | POINT_LOOKUP | 1.0 | 0.723 | 1.0 | 1.0 | 1 | – |
| G3 | COMPUTATION | 1.0 | 0.808 | 1.0 | 1.0 | 4 | – |
| P4 | POINT_LOOKUP | 1.0 | 0.685 | 1.0 | 1.0 | 1 | – |
