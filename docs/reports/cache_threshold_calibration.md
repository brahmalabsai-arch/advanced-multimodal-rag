# Cache threshold calibration (Phase 6)

Generated 2026-09-19T05:47:26+00:00 · embedder `bge-small` · 286 adversarial pairs (`eval/adversarial_cache_pairs.jsonl`) · 52 paraphrase pairs (`eval/paraphrase_pairs.jsonl`) · no model calls (`eval/cache_threshold_calibration.py`).

A pair *hits* when the slot-guard keys (entity, periods, metrics/formulas, direction, negation) agree **and** cosine similarity ≥ the class threshold (slot-rich = metric/formula + period present). Adversarial pairs must miss (NFR-3: false-hit rate ≤ 1 %); paraphrase pairs should hit.

## 1. What the hard slot guard does on its own (threshold 0)

| set | pairs | blocked by the guard | pass the guard |
|---|---|---|---|
| adversarial | 286 | 286 (100.0 %) | 0 |
| paraphrase | 52 | 2 (3.8 %) | 50 |

Per adversarial kind (guard only):

| kind | pairs | guard blocked | would hit without a threshold |
|---|---|---|---|
| period_swap | 66 | 66 | 0 |
| metric_swap | 66 | 66 | 0 |
| direction_swap | 72 | 72 | 0 |
| negation | 32 | 32 | 0 |
| same_slot_other_question | 50 | 50 | 0 |

Paraphrase pairs the guard splits (slot extraction disagrees — these can never hit):

- `P041` ['periods_key']: “When is NVIDIA's annual meeting?” vs “When is NVIDIA's 2026 annual meeting of stockholders?”
- `P042` ['periods_key']: “When is NVIDIA's annual meeting?” vs “What is the date of the 2026 annual meeting of stockholders?”

## 2. Similarity distributions

| mode | kind | n | p50 | max / min | guard-passing n | guard-passing max | p10 |
|---|---|---|---|---|---|---|---|
| raw | direction_swap | 72 | 0.928 | 0.958 | 0 | – | – |
| raw | metric_swap | 66 | 0.909 | 0.980 | 0 | – | – |
| raw | negation | 32 | 0.972 | 0.984 | 0 | – | – |
| raw | period_swap | 66 | 0.972 | 0.993 | 0 | – | – |
| raw | same_slot_other_question | 50 | 0.937 | 0.963 | 0 | – | – |
| raw | paraphrase (all) | 52 | 0.817 | 0.692 | – | – | 0.744 |
| raw | paraphrase slot-rich | 34 | 0.810 | 0.692 | – | – | 0.748 |
| raw | paraphrase slot-poor | 18 | 0.848 | 0.712 | – | – | 0.714 |
| normalized | direction_swap | 72 | 0.929 | 0.967 | 0 | – | – |
| normalized | metric_swap | 66 | 0.902 | 0.977 | 0 | – | – |
| normalized | negation | 32 | 0.968 | 0.981 | 0 | – | – |
| normalized | period_swap | 66 | 0.974 | 0.991 | 0 | – | – |
| normalized | same_slot_other_question | 50 | 0.933 | 0.966 | 0 | – | – |
| normalized | paraphrase (all) | 52 | 0.829 | 0.680 | – | – | 0.768 |
| normalized | paraphrase slot-rich | 34 | 0.821 | 0.716 | – | – | 0.769 |
| normalized | paraphrase slot-poor | 18 | 0.852 | 0.680 | – | – | 0.742 |
| canonical | direction_swap | 72 | 0.938 | 0.975 | 0 | – | – |
| canonical | metric_swap | 66 | 0.889 | 0.969 | 0 | – | – |
| canonical | negation | 32 | 0.975 | 0.991 | 0 | – | – |
| canonical | period_swap | 66 | 0.988 | 0.994 | 0 | – | – |
| canonical | same_slot_other_question | 50 | 0.929 | 0.971 | 0 | – | – |
| canonical | paraphrase (all) | 52 | 0.874 | 0.701 | – | – | 0.813 |
| canonical | paraphrase slot-rich | 34 | 0.869 | 0.701 | – | – | 0.824 |
| canonical | paraphrase slot-poor | 18 | 0.879 | 0.785 | – | – | 0.802 |

## 3. Threshold grid (one threshold for both classes)

| mode | threshold | adversarial false hits | false-hit rate | paraphrase hits | paraphrase hit rate |
|---|---|---|---|---|---|
| raw | 0.80 | 0 | 0.0 % | 29 | 55.8 % |
| raw | 0.82 | 0 | 0.0 % | 23 | 44.2 % |
| raw | 0.84 | 0 | 0.0 % | 19 | 36.5 % |
| raw | 0.85 | 0 | 0.0 % | 18 | 34.6 % |
| raw | 0.86 | 0 | 0.0 % | 17 | 32.7 % |
| raw | 0.88 | 0 | 0.0 % | 13 | 25.0 % |
| raw | 0.90 | 0 | 0.0 % | 11 | 21.1 % |
| raw | 0.92 | 0 | 0.0 % | 10 | 19.2 % |
| raw | 0.95 | 0 | 0.0 % | 5 | 9.6 % |
| normalized | 0.80 | 0 | 0.0 % | 34 | 65.4 % |
| normalized | 0.82 | 0 | 0.0 % | 30 | 57.7 % |
| normalized | 0.84 | 0 | 0.0 % | 21 | 40.4 % |
| normalized | 0.85 | 0 | 0.0 % | 21 | 40.4 % |
| normalized | 0.86 | 0 | 0.0 % | 16 | 30.8 % |
| normalized | 0.88 | 0 | 0.0 % | 14 | 26.9 % |
| normalized | 0.90 | 0 | 0.0 % | 12 | 23.1 % |
| normalized | 0.92 | 0 | 0.0 % | 10 | 19.2 % |
| normalized | 0.95 | 0 | 0.0 % | 5 | 9.6 % |
| canonical | 0.80 | 0 | 0.0 % | 47 | 90.4 % |
| canonical | 0.82 | 0 | 0.0 % | 45 | 86.5 % |
| canonical | 0.84 | 0 | 0.0 % | 39 | 75.0 % |
| canonical | 0.85 | 0 | 0.0 % | 36 | 69.2 % |
| canonical | 0.86 | 0 | 0.0 % | 33 | 63.5 % |
| canonical | 0.88 | 0 | 0.0 % | 22 | 42.3 % |
| canonical | 0.90 | 0 | 0.0 % | 19 | 36.5 % |
| canonical | 0.92 | 0 | 0.0 % | 19 | 36.5 % |
| canonical | 0.95 | 0 | 0.0 % | 11 | 21.1 % |

## 4. Configured thresholds (slot-rich 0.85 / slot-poor 0.9, `embed_text: canonical`)

| mode | adversarial false hits | false-hit rate | paraphrase hits | paraphrase hit rate |
|---|---|---|---|---|
| raw | 0 | 0.0 % | 16 | 30.8 % |
| normalized | 0 | 0.0 % | 18 | 34.6 % |
| canonical **(config)** | 0 | 0.0 % | 29 | 55.8 % |

Paraphrase pairs that pass the guard but fall under the threshold (accepted misses):

- `P022` sim 0.7013 (rich): “How much goodwill did NVIDIA carry at the end of fiscal 2026?” vs “Goodwill balance at fiscal year-end 2026?”
- `P020` sim 0.7226 (rich): “What was goodwill as of January 25, 2026?” vs “How much goodwill did NVIDIA carry at the end of fiscal 2026?”
- `P040` sim 0.7978 (poor): “Why did NVIDIA record an H20-related charge in Q1 fiscal 2026?” vs “Explain the H20 charge taken in the first quarter of fiscal 2026.”
- `P046` sim 0.8042 (poor): “Describe the five layers of NVIDIA's AI industry cake framing.” vs “What does the five-layer cake figure show?”
- `P016` sim 0.8116 (rich): “Inventory balance at FY2026 year-end?” vs “What was NVIDIA's inventory as of Jan 25, 2026?”
- `P014` sim 0.8231 (rich): “How much inventory did NVIDIA hold at the end of fiscal 2026?” vs “Inventory balance at FY2026 year-end?”
- `P038` sim 0.8243 (poor): “What drove the first-quarter fiscal 2026 charge related to H20?” vs “Why did NVIDIA record an H20-related charge in Q1 fiscal 2026?”
- `P005` sim 0.8262 (rich): “Total assets at the end of fiscal 2026?” vs “How much were NVIDIA's total assets at fiscal year-end 2026?”
- `P027` sim 0.8324 (rich): “Compute NVIDIA's current ratio at the end of fiscal 2026.” vs “Current ratio at FY2026 year-end?”
- `P008` sim 0.8353 (rich): “How much were NVIDIA's total assets at fiscal year-end 2026?” vs “Report total assets for fiscal 2026.”
- `P019` sim 0.8371 (rich): “How much cash and equivalents did NVIDIA have at the end of fiscal 2026?” vs “Cash and cash equivalents at FY2026 year-end?”
- `P030` sim 0.8423 (rich): “What was NVIDIA's revenue in fiscal 2026?” vs “Total revenue for FY2026?”
- `P011` sim 0.8465 (rich): “What were inventories as of January 25, 2026?” vs “How much inventory did NVIDIA hold at the end of fiscal 2026?”
- `P031` sim 0.8496 (rich): “How much revenue did NVIDIA report for fiscal year 2026?” vs “Total revenue for FY2026?”
- `P043` sim 0.8599 (poor): “When is NVIDIA's 2026 annual meeting of stockholders?” vs “What is the date of the 2026 annual meeting of stockholders?”
- `P036` sim 0.8699 (poor): “Why did NVIDIA's gross margin decrease in fiscal 2026?” vs “Explain the fall in gross margin during fiscal 2026.”
- `P052` sim 0.8734 (poor): “Is NVDA a good buy right now?” vs “Would you recommend buying NVIDIA shares?”
- `P037` sim 0.8744 (poor): “What caused the decline in NVIDIA's gross margin in FY2026?” vs “Explain the fall in gross margin during fiscal 2026.”
- `P045` sim 0.8789 (poor): “What are the layers in NVIDIA's five-layer cake?” vs “What does the five-layer cake figure show?”
- `P044` sim 0.8798 (poor): “What are the layers in NVIDIA's five-layer cake?” vs “Describe the five layers of NVIDIA's AI industry cake framing.”
- `P050` sim 0.8889 (poor): “Should I buy NVIDIA stock?” vs “Is NVDA a good buy right now?”

## 5. Decision (D-59)

Written after the runs of 2026-09-18 (kept in the script so re-runs keep the reasoning next to the numbers).

1. **The hard slot guard carries all of the measured precision.** The four swap categories the plan
   asked for (period, metric, direction, negation — 236 pairs) embed at cosine 0.90–0.99, *above* every
   threshold in the grid, so similarity could never have rejected them; the guard blocks 236/236. Two
   keys were added to the §5.8 design during calibration: `negation` ("did inventories grow" vs "did
   inventories not grow", 0.945) and `ask` (what the question asks about its slots — value, reason,
   method, location, risk, assumption, person, advice — from the new `lexicons.ask_type` in
   `glossary.yaml`), plus `aggregation_key` (pct / change / compare / ratio / yoy). Without `ask`, the 50
   *same-slot, different-question* pairs ("what were total assets" vs "how does NVIDIA account for total
   assets") sat at 0.93–0.97 — **closer than genuine paraphrases (0.80–0.90)** — and 34 of them still hit
   at 0.95. With the extended guard: 286/286 adversarial pairs miss, false-hit rate 0.0 % (NFR-3 ≤ 1 %).
2. **Embedded text: `canonical`.** Replacing matched spans by canonical ids ("total_assets", "fy2026")
   raises paraphrase similarity (p50 0.869 vs 0.829 normalised, 0.817 raw) without moving the
   adversarial distribution the guard already handles.
3. **Thresholds: slot-rich 0.85, slot-poor 0.90** (design: 0.90 / 0.95). Because bge-small scores
   unrelated-but-same-slot questions *higher* than paraphrases, raising the threshold buys no
   precision on this corpus; it only costs recall (0.90 → 36.5 % paraphrase hits, 0.85 → 65.4 %).
   Slot-poor questions keep a stricter floor because fewer hard keys constrain them. The remaining
   role of the threshold is to reject wording the lexicons do not model; it is a floor, not the defence.
4. **Accepted misses.** Two paraphrase pairs are split by the guard by design: "2026 annual meeting"
   resolves the bare year to `AMBIGUOUS_2026` (a period) while "annual meeting" has none. Paraphrases
   under the threshold (listed above) miss safely and cost one pipeline run each.
5. **Caveat.** The adversarial set is template-generated, so every swap is slot-detectable by
   construction; the same-slot category was added precisely because it is not. A larger hand-written
   set of guard-passing pairs is the right next step (Phase 7 evidence package) and is listed as a
   human-review item.
