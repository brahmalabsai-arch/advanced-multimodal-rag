# Cache walkthrough (Phase 6)

Generated 2026-09-18T17:21:16+00:00 · mode `in-process` · model profile `anthropic` · `scripts/cache_walkthrough.py`. Plan Phase 6 table, scripted; the same steps can be clicked through in the UI cache panel.

| step | action | expected | observed | ok |
|---|---|---|---|---|
| 1 | Ask “What were NVIDIA's total assets as of Jan 25, 2026?” | MISS; answer $206,803M; admitted as filed_fact | MISS · 1 model call(s) · 4578 ms · admitted filed_fact → L2+L1 | ✓ |
| 2 | Ask the same question again | L1 hit, no model call | L1 · 0 model call(s) · 12 ms | ✓ |
| 3 | Restart the server (fresh L1); ask again | L2 hit (L1 was empty), promoted to L1 | L2 · sim 1.000 · 0 model call(s) · 37 ms · then L1 | ✓ |
| 4 | Ask “Total assets at the end of fiscal 2026?” | L2 hit (paraphrase, same slots) | L2 · sim 0.867 · 0 model call(s) · 40 ms | ✓ |
| 5 | Ask “What were total assets as of Jan 26, 2025?” | MISS — period slot differs (G15) | MISS · 1 model call(s) · 4172 ms · admitted filed_fact → L2+L1 | ✓ |
| 6 | Toggle bypass cache; ask step 1 | Pipeline runs; no cache read or write | bypassed · 1 model call(s) · 3711 ms · L2 entries 2→2 | ✓ |
| 7 | Change a retrieval threshold; restart; ask step 1 — then revert and ask again | MISS (retrieval_config_hash changed); after revert → L2 hit on the original entry | MISS · 1 model call(s) · 2844 ms · admitted filed_fact → L2+L1 · after revert: L2 (0 calls) | ✓ |
| 8 | Ask “When is NVIDIA's annual meeting?”; advance clock +1 day; ask again | MISS admitted as time_anchored; then MISS (1-day TTL expired) | MISS · 1 model call(s) · 3658 ms · admitted time_anchored → L2+L1 · after +1d: MISS (1 calls) | ✓ |
| 9 | Advance clock to +31 days; ask step 1 | MISS — filed_fact 30-day TTL expired; sweeper removes the stale entry | MISS · 1 model call(s) · 3586 ms · admitted filed_fact → L2+L1 · sweep {'expired': 4, 'version_mismatch': 0, 'stale_removed': 0, 'remaining': 0, 'l1_cleared': 0} | ✓ |
| 10 | Reset clock; purge all | Stats show zero entries | purged {'l2_removed': 1, 'l1_cleared': 1} · L2 entries 0 · L1 entries 0 · clock offset 0 | ✓ |

Full pipeline runs (model calls) across the walkthrough: **6**. Cache-hit latencies: 12 ms, 37 ms, 40 ms (NFR-5 target p50 < 300 ms).

**Note.** Groq run of the same day (`groq_build`): steps 1-6 passed identically (MISS 2.6 s → L1 6 ms → L2 31 ms → paraphrase L2 sim 0.867 → period-swap MISS → bypass), then the rolling 200K tokens-per-day window was exhausted and steps 7-9 returned degraded answers (correctly *not* admitted: R1). The full 10-step pass above therefore runs on the `anthropic` profile, as the Phase 4-5 evaluations did (D-45 note). Step 7 failed on the first anthropic pass because the write under the edited config swept the original entry as version-mismatched; stale-version entries are now kept until TTL and evicted first under capacity pressure (D-59).
