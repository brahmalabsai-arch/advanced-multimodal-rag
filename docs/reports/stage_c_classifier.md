# Stage C — learned compression classifier (Phase 7)

Generated 2026-09-19T05:28:20+00:00 · `eval/train_classifier.py` · labels from `stagec_always` (forced `--compression always`) against `golden_phase4_claude` (no compression) · a compressed chunk is positive when the answer is no worse **and** the chunk shrank ≥ 30%.

## 1. Label set

- chunk decisions with an outcome: **54** from 27 golden questions
- positives (compression was right): **46** · negatives: **8**
- applied actions: {'ROW_SELECT': 18, 'EXTRACT_LLM': 36}
- intents: {'POINT_LOOKUP': 17, 'COMPUTATION': 5, 'COMPARISON_TREND': 2, 'EXPLANATORY': 16, 'VISUAL': 10, 'CROSS_SECTION': 4}
- queries where compression cost quality: 1 (E4: keyword coverage 1.00 → 0.25)

## 2. Cross-validated performance

5-fold stratified CV × 3 repeats (the label set is small, so a single split would be noise). The rule row is the hand-set Stage B score (§6.5) scored on the same labels.

| model | n | positives | accuracy | precision (compress) | 90 % CI | recall | ROC AUC |
|---|---|---|---|---|---|---|---|
| “always compress” (base rate) | 54 | 46 | 0.852 | **0.852** | – | 1.0 | – |
| Stage B rules (current) | 54 | 46 | 0.741 | **0.881** | – | 0.804 | 0.5 |
| logistic | 54 | 46 | 0.722 | **0.919** | 0.842–0.977 | 0.739 | 0.745 |
| gradient_boosting | 54 | 46 | 0.907 | **0.956** | 0.902–1.0 | 0.935 | 0.78 |

§6.7 target: precision of “compress” ≥ 0.85 — a wrong compression can lose the answer, a missed one only costs tokens. **Read the base-rate row first:** with a positive rate this high, “always compress” already clears the bar, so precision alone cannot justify a model. Adoption therefore also requires beating that baseline by ≥ 5 pp, beating the rules, and a floor on the number of *negative* labels — the rows that actually teach the model when not to compress.

## 3. Logistic coefficients (what the model learned)

| feature | coefficient |
|---|---|
| `noise` | +0.379 |
| `length` | +1.533 |
| `budget_pressure` | -1.388 |
| `intent_weight` | +0.217 |
| `numeric_density` | -0.075 |
| `rank_norm` | +0.458 |
| `max_dup_sim` | +0.835 |
| `n_sentences_norm` | +1.116 |
| _intercept_ | -1.570 |

Exported to `src/rag/compress/classifier_weights.json` (n=54, positives=46).

NumPy scorer parity with scikit-learn on the training rows: max |Δp| = 1.75e-05 (serving never imports scikit-learn).

## 4. Decision

**Stage C is NOT adopted.** Logistic precision 0.919 (90 % CI 0.842–0.977) vs Stage B rules 0.881 and “always compress” 0.852; best model gradient_boosting at 0.956. Unmet condition(s): ≥ 20 negative labels (have 8). Stage B keeps its hand-set weights — a transparent rule beats a model fitted on this little evidence. The weights file, the NumPy scorer and the `compression.stage_b` flag stay in the repo so the decision is a config change once real traffic (or a larger forced-run set) supplies more negatives.
