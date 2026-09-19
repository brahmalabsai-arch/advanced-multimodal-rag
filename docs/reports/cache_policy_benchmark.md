# Cache-policy benchmark (Phase 7)

Generated 2026-09-19T05:47:42+00:00 · `eval/cache_benchmark.py` · seed 20260919 · 5230 queries over 746 distinct question identities and 30 simulated days · thresholds 0.85 / 0.9 · 7.7 s, no model calls.

The store and the eviction bookkeeping are simulated; slot extraction, the seven-key guard, the canonical embedded text and the bge-small embeddings are the production ones, so a *hit* here means what it means at serving time. A hit is **correct** when the served entry answers the same question identity (topic × period) as the query, **false** otherwise, and **stale** when it is served past its class TTL.

## 1. Workload

- **746 question identities**: a canonical head (every statement line item × three fiscal years × several phrasings, the five formulas, and the report's named topics) plus **630 one-off tail questions** (12% of traffic).
- popularity over the head is Zipf (s ≈ 1.1); three bursts (`h20_charge`, `export_controls`, `annual_meeting`) each concentrate 4 % of traffic into two days.
- every phrasing of one identity is a genuine paraphrase, so a hit that serves another identity is a real error, not a generator artefact.
- **Capacity note.** Only 25, 50, 100 of the requested capacities ever evict; at 500, 1000 the whole working set fits and every policy is identical by construction. The plan asks for 100 / 500 / 1,000; 25 and 50 are added so the comparison has something to compare. This is the "eviction is inert at demo scale" caveat of §5.9, measured.

## 2. Results

| policy | capacity | hit rate | correct-hit rate | false-hit rate | stale-hit rate | evictions | calls avoided |
|---|---|---|---|---|---|---|---|
| redis_volatile_lfu (chosen) | 25 | 45.8% | **45.8%** | 0.00% | 0.00% | 2810 | 2395 |
| redis_volatile_lfu (chosen) | 50 | 59.5% | **59.5%** | 0.00% | 0.00% | 2064 | 3112 |
| redis_volatile_lfu (chosen) | 100 | 73.7% | **73.5%** | 0.19% | 0.00% | 1237 | 3843 |
| redis_volatile_lfu (chosen) | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| redis_volatile_lfu (chosen) | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| lru | 25 | 51.3% | **51.3%** | 0.06% | 0.00% | 2509 | 2681 |
| lru | 50 | 61.9% | **61.8%** | 0.10% | 0.00% | 1909 | 3234 |
| lru | 100 | 71.8% | **71.6%** | 0.25% | 0.00% | 1335 | 3743 |
| lru | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| lru | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| lfu_no_decay | 25 | 46.5% | **46.5%** | 0.00% | 0.00% | 2775 | 2430 |
| lfu_no_decay | 50 | 55.6% | **55.6%** | 0.00% | 0.00% | 2272 | 2908 |
| lfu_no_decay | 100 | 62.2% | **62.2%** | 0.00% | 0.00% | 1875 | 3251 |
| lfu_no_decay | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| lfu_no_decay | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| volatile_ttl | 25 | 37.2% | **37.2%** | 0.00% | 0.00% | 3259 | 1946 |
| volatile_ttl | 50 | 48.2% | **48.2%** | 0.00% | 0.00% | 2661 | 2519 |
| volatile_ttl | 100 | 59.0% | **59.0%** | 0.00% | 0.00% | 2047 | 3083 |
| volatile_ttl | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| volatile_ttl | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| fifo | 25 | 45.4% | **45.3%** | 0.06% | 0.00% | 2831 | 2371 |
| fifo | 50 | 56.5% | **56.4%** | 0.13% | 0.00% | 2223 | 2950 |
| fifo | 100 | 66.7% | **66.3%** | 0.33% | 0.00% | 1604 | 3470 |
| fifo | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| fifo | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| mru | 25 | 37.4% | **37.3%** | 0.04% | 0.00% | 3241 | 1952 |
| mru | 50 | 49.9% | **49.9%** | 0.08% | 0.00% | 2541 | 2608 |
| mru | 100 | 62.6% | **62.5%** | 0.11% | 0.00% | 1785 | 3269 |
| mru | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| mru | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| random | 25 | 45.1% | **45.0%** | 0.08% | 0.00% | 2844 | 2355 |
| random | 50 | 56.2% | **56.1%** | 0.13% | 0.00% | 2227 | 2933 |
| random | 100 | 66.8% | **66.4%** | 0.44% | 0.00% | 1592 | 3471 |
| random | 500 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| random | 1000 | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| ttl_only (unbounded) | ∞ | 84.5% | **83.1%** | 1.36% | 0.00% | 0 | 4346 |
| lru_no_ttl | 25 | 51.4% | **51.4%** | 0.06% | 2.12% | 2514 | 2688 |
| lru_no_ttl | 50 | 62.4% | **62.3%** | 0.10% | 5.51% | 1918 | 3257 |
| lru_no_ttl | 100 | 72.4% | **72.1%** | 0.25% | 5.81% | 1346 | 3771 |
| lru_no_ttl | 500 | 86.5% | **84.1%** | 2.33% | 14.26% | 208 | 4400 |
| lru_no_ttl | 1000 | 86.7% | **84.2%** | 2.47% | 14.51% | 0 | 4406 |

## 3. Cost avoided (chosen policy)

One miss costs a full pipeline run. Tokens per answered question: **2,450** (mean of 653 `large` role calls in data/logs/llm_usage.jsonl).

| capacity | correct hits | large-model calls avoided | tokens avoided |
|---|---|---|---|
| 25 | 2395 | 2395 | 5,867,302 |
| 50 | 3112 | 3112 | 7,623,818 |
| 100 | 3843 | 3843 | 9,414,632 |
| 500 | 4346 | 4346 | 10,646,888 |
| 1000 | 4346 | 4346 | 10,646,888 |

`config/models.yaml` carries no `price_usd_per_mtok` during the Groq free-tier build, so the saving is reported in calls and tokens; multiply by the active profile's published price to get USD.

## 4. Per-TTL-class correct-hit rate (chosen policy)

| capacity | analytical | filed_fact | time_anchored |
|---|---|---|---|
| 25 | 0.0% | 65.2% | 0.0% |
| 50 | 23.3% | 74.8% | 23.4% |
| 100 | 38.7% | 82.4% | 90.0% |
| 500 | 41.4% | 95.2% | 90.5% |
| 1000 | 41.4% | 95.2% | 90.5% |

## 5. False hits — what slips past the guard

The chosen policy's false-hit rate is 0.19% at capacity 100 (worst across all arms: 2.47%, `lru_no_ttl` @ 1000). Every one comes from a *phrasing* whose slots are genuinely identical to another question's, which the calibration set (`docs/reports/cache_threshold_calibration.md`) predicted would be the residual risk. The distinct confusions, most frequent first:

| asked (identity) | served (identity) | n | max similarity | example phrasing |
|---|---|---|---|---|
| `tail:segment:total current assets:FY2025` | `tail:assumptions:total current assets:FY2025` | 1 | 0.877 | “What segment detail is given for total current assets as of January 26, 2025?” |
| `tail:segment:deferred tax assets:FY2024` | `tail:controls:deferred tax assets:FY2024` | 1 | 0.870 | “What segment detail is given for deferred tax assets as of January 28, 2024?” |
| `tail:audit:deferred tax assets:FY2026` | `tail:assumptions:deferred tax assets:FY2026` | 1 | 0.858 | “Is deferred tax assets in fiscal 2026 audited?” |
| `tail:audit:additional paid-in capital:FY2024` | `tail:segment:additional paid-in capital:FY2024` | 1 | 0.899 | “Is additional paid-in capital in fiscal 2024 audited?” |
| `tail:assumptions:capital expenditures:FY2024` | `tail:audit:capital expenditures:FY2024` | 1 | 0.858 | “What assumptions underlie NVIDIA's capital expenditures as of January 28, 2024?” |
| `tail:controls:cost of revenue:FY2025` | `tail:audit:cost of revenue:FY2025` | 1 | 0.879 | “What disclosure controls cover cost of revenue as of January 26, 2025?” |
| `tail:segment:share repurchases:FY2024` | `tail:assumptions:share repurchases:FY2024` | 1 | 0.893 | “What segment detail is given for share repurchases as of January 28, 2024?” |
| `tail:audit:accounts payable:FY2025` | `tail:controls:accounts payable:FY2025` | 1 | 0.869 | “Is accounts payable in fiscal 2025 audited?” |
| `tail:segment:shareholders' equity:FY2024` | `tail:controls:shareholders' equity:FY2024` | 1 | 0.884 | “What segment detail is given for shareholders' equity as of January 28, 2024?” |
| `tail:controls:accrued liabilities:FY2024` | `tail:segment:accrued liabilities:FY2024` | 1 | 0.864 | “What disclosure controls cover accrued liabilities as of January 28, 2024?” |

Read the pattern, not just the rate: what survives the guard is a question *about* a metric — where it is disclosed, who signs it, what policy governs it — served the entry holding that metric's *value*. That is the `ask` slot failing on a phrasing its lexicon does not cover (`lexicons.ask_type` in `config/glossary.yaml`), the same family of defect this benchmark first caught as annual-meeting-vs-record-date (D-60). Each round of cues shrinks the rate; a rule-based ask classifier will never reach zero, which is why the similarity threshold stays as a floor underneath it. The rate is near-identical across policies at a given capacity, so the comparison above is unaffected.

## 6. Promotion rule (D-32)

> The chosen standard stays unless a challenger beats its correct-hit rate by ≥ 5 percentage points at two or more capacities.

- fifo: Δ correct-hit rate vs chosen = -0.5%@25, -3.1%@50, -7.1%@100, +0.0%@500, +0.0%@1000
- lfu_no_decay: Δ correct-hit rate vs chosen = +0.7%@25, -3.9%@50, -11.3%@100, +0.0%@500, +0.0%@1000
- lru: Δ correct-hit rate vs chosen = +5.5%@25, +2.3%@50, -1.9%@100, +0.0%@500, +0.0%@1000 → beats it at 1 capacities
- lru_no_ttl: Δ correct-hit rate vs chosen = +5.6%@25, +2.8%@50, -1.4%@100, +1.0%@500, +1.1%@1000 → beats it at 1 capacities
- mru: Δ correct-hit rate vs chosen = -8.5%@25, -9.6%@50, -11.0%@100, +0.0%@500, +0.0%@1000
- random: Δ correct-hit rate vs chosen = -0.8%@25, -3.4%@50, -7.1%@100, +0.0%@500, +0.0%@1000
- volatile_ttl: Δ correct-hit rate vs chosen = -8.6%@25, -11.3%@50, -14.5%@100, +0.0%@500, +0.0%@1000

**Outcome: the chosen policy stands.**
