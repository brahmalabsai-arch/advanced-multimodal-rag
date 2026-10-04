# Model-swap dry run (Phase 8, NFR-11)

Generated 2026-10-04T12:34:32+00:00 · `scripts/check_profile.py --profile all --dry-run` · no network calls.

A profile passes when `models.yaml` validates with it active, every role resolves to a provider + model, the provider chat-model constructor accepts the profile's `provider_kwargs`, the context budget fits under the pacing limits and the structured-output mode matches the provider. A missing key or an uninstalled package is a warning: the template is valid, the switch (F1) supplies them.

| Profile | Result | Errors | Warnings | small | large | vision |
|---|---|---:|---:|---|---|---|
| `groq_build` | ✅ pass | 0 | 1 | groq:`openai/gpt-oss-20b` | groq:`openai/gpt-oss-120b` | groq:`qwen/qwen3.8-27b` |
| `groq_qwen_large` | ✅ pass | 0 | 1 | groq:`openai/gpt-oss-20b` | groq:`qwen/qwen3.8-27b` | groq:`qwen/qwen3.8-27b` |
| `anthropic` | ✅ pass | 0 | 2 | anthropic:`claude-haiku-4-5-20251001` | anthropic:`claude-sonnet-5` | anthropic:`claude-sonnet-5` |
| `gemini` | ✅ pass | 0 | 1 | google:`gemini-3.5-flash-lite` | google:`gemini-3.5-flash-lite` | google:`gemini-3.5-flash-lite` |

## `groq_build`

| Check | Level | Detail |
|---|---|---|
| config | ✅ OK | models.yaml validates; active_profile=groq_build; structured_output=json_mode_validate; context_budget_tokens=2500 |
| roles | ✅ OK | small  → groq:openai/gpt-oss-20b |
| roles | ✅ OK | large  → groq:openai/gpt-oss-120b |
| roles | ✅ OK | vision → groq:qwen/qwen3.8-27b |
| package | ✅ OK | groq: langchain-groq 1.1.3 |
| key | ✅ OK | GROQ_API_KEY present in .env |
| construct | ✅ OK | small  ChatGroq('openai/gpt-oss-20b', reasoning_format='hidden', reasoning_effort='low') |
| construct | ✅ OK | large  ChatGroq('openai/gpt-oss-120b', reasoning_format='hidden', reasoning_effort='medium') |
| construct | ✅ OK | vision ChatGroq('qwen/qwen3.8-27b', reasoning_format='hidden', reasoning_effort='none') |
| budgets | ✅ OK | large  context_budget 2500 vs tpm 8000 → 5500 tokens headroom for prompt scaffolding + output |
| budgets | ✅ OK | vision max_tokens cap 1000 (OTPM) · input cap 7000 · max_images 3 |
| structured | ✅ OK | json_mode_validate for providers groq |
| prices | 🟡 WARN | no per-token prices for small, large, vision → break-even test runs in quota mode (§6.6); fill `price_usd_per_mtok` at F1 |
| enrichment | ✅ OK | figures → vision (qwen/qwen3.8-27b); table_summaries → small (openai/gpt-oss-20b) |
| thresholds | ✅ OK | thresholds.yaml validates; compression.mode=classifier, cache.enabled=True |

## `groq_qwen_large`

| Check | Level | Detail |
|---|---|---|
| config | ✅ OK | models.yaml validates; active_profile=groq_qwen_large; structured_output=json_mode_validate; context_budget_tokens=2500 |
| roles | ✅ OK | small  → groq:openai/gpt-oss-20b |
| roles | ✅ OK | large  → groq:qwen/qwen3.8-27b |
| roles | ✅ OK | vision → groq:qwen/qwen3.8-27b |
| package | ✅ OK | groq: langchain-groq 1.1.3 |
| key | ✅ OK | GROQ_API_KEY present in .env |
| construct | ✅ OK | small  ChatGroq('openai/gpt-oss-20b', reasoning_format='hidden', reasoning_effort='low') |
| construct | ✅ OK | large  ChatGroq('qwen/qwen3.8-27b', reasoning_format='hidden', reasoning_effort='low') |
| construct | ✅ OK | vision ChatGroq('qwen/qwen3.8-27b', reasoning_format='hidden', reasoning_effort='none') |
| budgets | ✅ OK | large  context_budget 2500 vs tpm 8000 → 5500 tokens headroom for prompt scaffolding + output |
| budgets | ✅ OK | vision max_tokens cap 1000 (OTPM) · input cap 7000 · max_images 3 |
| structured | ✅ OK | json_mode_validate for providers groq |
| prices | 🟡 WARN | no per-token prices for small, large, vision → break-even test runs in quota mode (§6.6); fill `price_usd_per_mtok` at F1 |
| enrichment | ✅ OK | figures → vision (qwen/qwen3.8-27b); table_summaries → small (openai/gpt-oss-20b) |
| thresholds | ✅ OK | thresholds.yaml validates; compression.mode=classifier, cache.enabled=True |

## `anthropic`

| Check | Level | Detail |
|---|---|---|
| config | ✅ OK | models.yaml validates; active_profile=anthropic; structured_output=native; context_budget_tokens=6000 |
| roles | ✅ OK | small  → anthropic:claude-haiku-4-5-20251001 |
| roles | ✅ OK | large  → anthropic:claude-sonnet-5 |
| roles | ✅ OK | vision → anthropic:claude-sonnet-5 |
| package | ✅ OK | anthropic: langchain-anthropic 1.7.2 |
| key | 🟡 WARN | ANTHROPIC_API_KEY missing — add it to .env before `MODEL_PROFILE=anthropic` |
| construct | ✅ OK | small  ChatAnthropic('claude-haiku-4-5-20251001') [placeholder key] |
| construct | ✅ OK | large  ChatAnthropic('claude-sonnet-5') [placeholder key] |
| construct | ✅ OK | vision ChatAnthropic('claude-sonnet-5') [placeholder key] |
| budgets | ✅ OK | large  context_budget 6000 vs tpm 40000 → 34000 tokens headroom for prompt scaffolding + output |
| budgets | ✅ OK | vision max_tokens cap none (OTPM) · input cap 40000 · max_images unbounded |
| structured | ✅ OK | native for providers anthropic (schema instruction + validation; native structured output is an F1 step) |
| prices | 🟡 WARN | no per-token prices for small, large, vision → break-even test runs in quota mode (§6.6); fill `price_usd_per_mtok` at F1 |
| enrichment | ✅ OK | figures → vision (claude-sonnet-5); table_summaries → small (claude-haiku-4-5-20251001) |
| thresholds | ✅ OK | thresholds.yaml validates; compression.mode=classifier, cache.enabled=True |

## `gemini`

| Check | Level | Detail |
|---|---|---|
| config | ✅ OK | models.yaml validates; active_profile=gemini; structured_output=native; context_budget_tokens=6000 |
| roles | ✅ OK | small  → google:gemini-3.5-flash-lite |
| roles | ✅ OK | large  → google:gemini-3.5-flash-lite |
| roles | ✅ OK | vision → google:gemini-3.5-flash-lite |
| package | ✅ OK | google: langchain-google-genai 4.4.0 |
| key | ✅ OK | GOOGLE_API_KEY present in .env |
| construct | ✅ OK | small  ChatGoogleGenerativeAI('gemini-3.5-flash-lite', thinking_level='minimal') |
| construct | ✅ OK | large  ChatGoogleGenerativeAI('gemini-3.5-flash-lite', thinking_level='low') |
| construct | ✅ OK | vision ChatGoogleGenerativeAI('gemini-3.5-flash-lite', thinking_level='low') |
| budgets | ✅ OK | large  context_budget 6000 vs tpm 250000 → 244000 tokens headroom for prompt scaffolding + output |
| budgets | ✅ OK | vision max_tokens cap none (OTPM) · input cap 250000 · max_images unbounded |
| structured | ✅ OK | native for providers google (schema instruction + validation; native structured output is an F1 step) |
| prices | 🟡 WARN | no per-token prices for small, large, vision → break-even test runs in quota mode (§6.6); fill `price_usd_per_mtok` at F1 |
| enrichment | ✅ OK | figures → vision (gemini-3.5-flash-lite); table_summaries → small (gemini-3.5-flash-lite) |
| thresholds | ✅ OK | thresholds.yaml validates; compression.mode=classifier, cache.enabled=True |

Reproduce: `make check-profile` (all profiles) or `.venv/Scripts/python scripts/check_profile.py --profile anthropic --dry-run`.
