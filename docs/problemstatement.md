# Problem Statement — Advanced Multimodal RAG for Balance Sheet Analysis (NVIDIA FY2026)

| Field | Value |
|---|---|
| Project | ADVANCED_RAG — Balance Sheet Analysis Assistant |
| Owner | Rakesh (GitHub: `brahmalabsai-arch`) |
| Document version | v0.4 (statuses at the backend-complete gate) |
| Date | 19 September 2026 |
| Change log | v0.1 initial proposal · v0.2 review round 1: simple HTML UI, swappable model profiles, calculator + citations confirmed, single-turn v1, industry-standard cache policy, multimodal strategy · v0.3 review round 2: **localhost-only build**, **Groq-only models during the build**, Kubernetes removed, online deployment and advanced models deferred until the backend is complete · v0.4 Phase 8: NFR table carries measured status (`docs/reports/nfr_results.md`), success criteria ticked with evidence |
| Companion documents | [`architecture.md`](./architecture.md) · [`implementation_plan.md`](./implementation_plan.md) |
| Source corpus | `2026_NVIDIA_ANNUAL_REPORT.pdf` (Annual Review + Notice of Annual Meeting + Proxy Statement + Form 10-K) |

### How discussions are cited in this document

Every design position traces back to a discussion thread. Tags used throughout both documents:

| Tag | Discussion thread |
|---|---|
| **[B1]** | Brainstorm 1 — Which cache eviction / TTL strategy fits this use case |
| **[B2]** | Brainstorm 2 — Logic of the compression-classifier layer |
| **[B3]** | Package selection and reasoning |
| **[B4]** | Corpus inspection findings (what the actual PDF looks like) |
| **[B5]** | Query-side design: expansion, reranking gates, small-vs-large model split |
| **[B6-Qn]** | Review round 1 — resolutions to open questions Q1–Q6 (see `architecture.md` §0.1) |
| **[B7-Qn]** | Review round 2 — localhost-only build, Groq-only models, deployment deferred (see `architecture.md` §0.2) |
| **[R-n]** | External reference (listed at the end of `architecture.md`) |

---

## 1. Background

Balance sheet analysis of a large listed company is a reading-heavy task. For NVIDIA's fiscal year 2026, the relevant material is spread across a single 175-page combined filing that mixes a glossy visual Annual Review, a Proxy Statement, and the full Form 10-K. An analyst answering a question such as *"How did NVIDIA's liquidity position change year-over-year, and what drove the inventory build?"* has to:

1. Locate the Consolidated Balance Sheets (page 141 of the PDF, page 53 of the 10-K).
2. Pull two columns of numbers (Jan 25, 2026 vs Jan 26, 2025).
3. Compute ratios (current ratio, quick ratio, working capital).
4. Cross-reference MD&A narrative (Item 7) and notes to explain the movements.
5. Keep track of units (USD millions), fiscal-period naming, and page references.

A plain "embed-everything-and-ask-an-LLM" RAG handles this poorly (see §5). The goal of this project is an **advanced** retrieval-augmented generation system that answers such questions accurately, with page-level citations, at low and measurable LLM cost.

This is also a portfolio project. It must be explainable end-to-end in an interview: every component should have a stated reason to exist and a measured effect.

---

## 2. Corpus profile [B4]

Findings from direct inspection of the uploaded PDF (poppler + pdfplumber diagnostics, page rasterization):

| Property | Observation | Design consequence |
|---|---|---|
| Size | 175 pages, ~15.9 MB, ~94,000 words | Small enough for a local Chroma index; large enough that naive full-context stuffing is wasteful |
| Composition | pp. 1–18 Annual Review (visual, landscape); pp. 19–88 Notice of Annual Meeting & Proxy Statement; pp. 89–174 Form 10-K; p. 175 back cover | Section metadata on every chunk; section-scoped filtering |
| Page geometry | Mixed: 153 letter-portrait pages, 12 landscape spreads (1188×774 pt), 9 half-spreads, 1 rotated page | Parser must handle layout per page, not assume one template |
| Text layer | Embedded fonts present; text is extractable (not a scanned PDF) | No full OCR needed; OCR only as a fallback |
| Multi-column layout | Annual Review letter pages (e.g., p. 13) interleave columns under naive extraction ("Compute is no longer just a / NVIDIA is building the computing / …") | Layout-aware parsing with reading-order detection is mandatory |
| Tables | ~74 pages with detectable tables (financial statements, compensation tables, share repurchase tables, etc.) | Tables must be first-class retrieval units, not split by a text chunker |
| Balance sheet | Single clean table on p. 141, two period columns, units in USD millions | Row-level "fact" chunks with parsed numeric values enable exact lookup and deterministic ratio math |
| Raster images | 109 embedded images across 36 pages — many are decorative photography (e.g., p. 5 data-center render) | Figure classifier must drop decorative images before indexing |
| Vector charts | p. 123 "Stock Performance Graphs" is drawn with vector operators, so `pdfimages` does not see it and text extraction yields only axis labels | Page-region rasterization is needed to capture charts; the chart's companion data table must be linked |
| Diagrams | p. 3 "AI Is a Five-Layer Cake" stack diagram carries meaning only visually | Vision-model description required for diagram retrieval |
| Duplication | The same disclosure (e.g., the $4.5 billion H20 charge) appears 5 times across Business, Risk Factors and MD&A | Retrieval returns near-duplicates; de-duplication is part of the compression layer [B2] |
| Fiscal calendar | FY2026 ended **Jan 25, 2026**; FY2025 ended **Jan 26, 2025** | "2025" in a user query is ambiguous (calendar 2025 ≈ fiscal 2026). Period normalization is required for retrieval *and* for cache safety [B1] |
| Time-bound content | Annual meeting scheduled for June 24, 2026 — already past as of this document's date | Some answers go stale with wall-clock time even though the PDF does not change; drives TTL classes [B1] |

---

## 3. Problem statement

> **Build an advanced, multimodal retrieval-augmented generation system over NVIDIA's FY2026 combined annual report that answers balance-sheet-analysis questions — lookups, ratios, year-over-year comparisons, narrative explanations, and questions about charts and diagrams — with verifiable page citations, while minimizing large-model usage through semantic caching, gated compression, and a strict small-model / large-model division of labour.**

The system must search across text, tables, and visuals stored in a ChromaDB vector database built with semantic chunking; improve recall with query expansion; improve precision with reranking when needed; shrink context with information compression only when a classifier decides it pays off; and avoid repeat LLM spend with a two-tier semantic cache governed by an explicit TTL and eviction strategy.

---

## 4. Users and question types

**Primary users:** financial analysts, credit/investment research students, finance-literate business users, and interview reviewers evaluating the portfolio build.

The system should handle seven intent classes. These classes are used by the query analyzer, the rerank gate, the compression classifier [B2], and the cache TTL classes [B1], so they are defined once here.

| Intent | Example | What makes it hard |
|---|---|---|
| `POINT_LOOKUP` | "What were NVIDIA's inventories at the end of fiscal 2026?" | Exact number, exact period, exact unit |
| `COMPUTATION` | "What is the current ratio as of Jan 25, 2026?" | Needs two retrieved values + arithmetic the LLM should not do freehand |
| `COMPARISON_TREND` | "How much did goodwill grow year-over-year?" | Two periods; risk of swapping columns |
| `EXPLANATORY` | "Why did gross margin decrease in fiscal 2026?" | Narrative spread across MD&A, duplicates across sections |
| `VISUAL` | "How did NVIDIA's 5-year total return compare to the Nasdaq 100?" | Answer lives in a vector chart + companion table |
| `CROSS_SECTION` | "How does executive pay-versus-performance relate to revenue growth?" | Spans Proxy and 10-K |
| `OUT_OF_SCOPE` | "Should I buy NVDA today?" / "What's the share price right now?" | Must refuse investment advice and live-data questions gracefully, without retrieval spend |

---

## 5. Why a naive RAG fails here

| # | Failure of naive RAG | Evidence from this corpus | Addressed by |
|---|---|---|---|
| F1 | Fixed-size chunking cuts tables mid-row and separates headers from values | 74 table pages; balance sheet columns are period-labelled only in the header | Structure-first + semantic chunking, atomic tables, row-fact chunks |
| F2 | Dense embeddings blur numbers and exact line-item names | "Accounts receivable, net" vs "Accrued and other current liabilities" embed close together | Hybrid dense + BM25 retrieval, finance glossary expansion |
| F3 | Text-only extraction is blind to charts and diagrams | p. 123 vector chart, p. 3 diagram | Vision-model figure descriptions + image passthrough to generator |
| F4 | Naive extraction garbles multi-column pages | p. 13 column interleaving | Layout-aware parser |
| F5 | Top-k fills with duplicates, crowding out useful context | $4.5B H20 charge repeated 5× | Dedupe + compression classifier [B2] |
| F6 | LLM arithmetic errors on ratios | Any `COMPUTATION` query | Deterministic calculator over parsed numeric metadata |
| F7 | A semantic cache returns the *wrong period's* answer | "total assets FY2026" and "total assets FY2025" are near-identical embeddings | Slot-guarded cache lookup [B1] |
| F8 | Token-level compression silently drops digits | Financial tables are dense with numbers | Tables are protected from LLM compression; numeric fidelity check [B2] |
| F9 | Every query hits the large model | Repeated canonical questions (total assets, current ratio…) | Two-tier semantic cache with TTL classes [B1] |

---

## 6. Functional requirements

Requirements FR-1 to FR-7 map one-to-one to the capabilities in the original brief. FR-8 onward are derived requirements that the brainstorm showed are necessary for the brief to work on financial data; each is marked as such so it can be accepted or cut.

### FR-1 Multimodal retrieval (brief item 1)
- FR-1.1 Index narrative text, tables, charts, and diagrams from the PDF into a single searchable store.
- FR-1.2 Each figure is indexed via a text description generated by a vision-capable small model; the original image crop path is stored so the generator can see the actual image at answer time.
- FR-1.3 Decorative images (photography, backgrounds, logos) are classified and excluded.
- FR-1.4 Tables are indexed as (a) a whole-table chunk with a natural-language summary and (b) row-level fact chunks carrying parsed numeric values.

### FR-2 Semantic chunking into ChromaDB (brief item 2)
- FR-2.1 Chunk boundaries respect document structure first (section, heading, table, figure), then semantic similarity between adjacent sentences inside narrative blocks.
- FR-2.2 All chunks are persisted in ChromaDB with the metadata schema defined in `architecture.md` §9.
- FR-2.3 The index carries a `corpus_version` hash so downstream caches can be invalidated when the corpus changes.

### FR-3 Advanced retrieval methods (brief item 3)
- FR-3.1 **Query expansion** — deterministic finance-glossary expansion, fiscal-period normalization, sub-question decomposition for computations, and a small number of LLM paraphrases for explanatory queries.
- FR-3.2 **Information compression** — de-duplication, row selection for tables, and extractive sentence compression for narrative chunks.
- FR-3.3 **Reranking** — cross-encoder reranking, applied through a gate that skips it when retrieval is already unambiguous [B5].

### FR-4 Compression classifier (brief item 4)
- FR-4.1 A classifier decides, per query, whether compression is required at all, and per chunk, which action to take (`KEEP`, `DEDUPE`, `ROW_SELECT`, `EXTRACT`, `DROP`).
- FR-4.2 The classifier itself must not call an LLM (otherwise it spends what it is meant to save) [B2].
- FR-4.3 Decisions and their features are logged for later training of a learned classifier.

### FR-5 Semantic caching (brief item 5)
- FR-5.1 Before any LLM call, look up the query in an exact-match tier and a semantic tier.
- FR-5.2 A semantic hit is only valid if structured slots match (fiscal period, metric, statement/section, comparison direction) and version keys match (corpus, prompt, generator model, retrieval config) [B1].
- FR-5.3 Only verified answers are admitted to the cache (admission policy).

### FR-6 Small-vs-large model rule (brief item 6)
- FR-6.1 Query analysis/expansion, compression, figure description, and reranking use small models (small LLM tier or local cross-encoder).
- FR-6.2 Only final answer generation uses the large model.
- FR-6.3 Model IDs live in configuration, not code, grouped into swappable profiles [B6-Q2][B7-Q2]. **During the whole build only the Groq profile (`groq_build`) is active.** Anthropic and Gemini profiles exist as validated but inactive templates and are switched on after the backend is complete, without code changes. Each profile defines a small model, a large model, a vision-capable model, a context budget, and rate-limit pacing.
- FR-6.4 Answers cached under one model profile are never served under another (the generator model is a cache version key).

### FR-7 TTL strategy for caching (brief item 7)
- FR-7.1 Cache entries carry a TTL assigned by answer class (filed fact, analytical, time-anchored, negative/low-confidence).
- FR-7.2 Version-key mismatch invalidates entries immediately, independent of TTL.
- FR-7.3 The cache follows an **industry-standard policy** [B6-Q5]: the semantic tier uses Redis `volatile-lfu` semantics (every entry has a TTL; eviction removes the lowest decaying-frequency entry), the exact-match tier uses LRU + TTL, and TTLs sit within the commonly used 1–30 day range by content volatility.
- FR-7.4 An offline benchmark compares the chosen standard against LRU, LFU without decay, `volatile-ttl`, FIFO, MRU, random replacement, and TTL-only on a replayed query log; a challenger replaces the standard only if it wins by a defined margin.

### FR-8 (derived, **accepted** in review [B6-Q3]) Deterministic financial calculations
- Ratios and YoY changes are computed in Python from parsed numeric metadata using a formula registry; the LLM explains, it does not calculate.

### FR-9 (derived, **accepted** in review [B6-Q3]) Citations and provenance
- Every answer cites section and PDF page (e.g., *Form 10-K, Consolidated Balance Sheets, PDF p. 141*). Numbers must be traceable to a retrieved chunk.

### FR-10 (derived) Out-of-scope handling
- Investment advice and live-market questions are detected before retrieval and answered with a scoped refusal plus what the report *can* answer.

### FR-11 (derived) Cost and usage telemetry
- Every request logs: cache tier hit/miss, model profile, models called, tokens in/out per model, classifier decisions, latency per stage, and process memory. This is what proves the cost-saving and resource-fit claims.

### FR-12 Localhost web interface [B6-Q1][B7-Q3]
- A simple HTML page (no frontend build toolchain) served by the backend at `http://127.0.0.1:8000`, used to check and test queries and caching.
- Contents: question box, answer with clickable page citations and page thumbnails, figure display for visual answers, and a collapsible debug panel showing retrieval, gate, compression, and cost decisions.
- Cache-testing controls: cache tier indicator (L1 / L2 / miss), similarity and TTL details, a bypass-cache toggle, a purge button, and a dev-only clock to simulate the passage of days so TTL expiry can be tested immediately.
- Online deployment (Render or a similar platform) is **not** part of this build; it is decided after the backend is complete [B7-Q1][B7-Q3].

### FR-13 Conversation scope [B6-Q4]
- v1 is single-turn: each question is answered independently and only self-contained questions are cached.
- v2 adds multi-turn conversations with session management, specifically to test caching behaviour across follow-up questions and sessions.

### FR-14 Demo protection (deferred to the deployment stage [B7-Q3])
- Only needed once a public URL exposes paid model keys: access key, per-IP rate limits, and a daily token cap. Not built in v1.

---

## 7. Non-functional requirements

Targets below were proposed before the build. **Measured status (Phase 7–8)** is in [`docs/reports/nfr_results.md`](./reports/nfr_results.md): NFR-1–6, 9, 10, 12 met; NFR-7 met by the Phase 8 fresh-clone rehearsal (`docs/reports/fresh_clone_rehearsal.md`); NFR-11 met by the Phase 8 model-swap dry run (`docs/reports/model_swap_dry_run.md`); NFR-8 partial — the build ran on the Groq free tier, but the full golden, compression-ablation and RAGAS evaluation runs used the `anthropic` profile when the Groq daily window was exhausted (architecture D-45 note; user-sanctioned deviation).

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Accuracy | ≥ 95% exact-match on numeric `POINT_LOOKUP` golden questions; ≥ 90% on `COMPUTATION` |
| NFR-2 | Faithfulness | RAGAS faithfulness ≥ 0.90 on the golden set |
| NFR-3 | Cache safety | Semantic-cache false-hit rate ≤ 1% on an adversarial near-duplicate set (period swaps, metric swaps, increase/decrease swaps) |
| NFR-4 | Cost | ≥ 30% reduction in large-model calls on a replayed realistic query log versus the no-cache baseline |
| NFR-5 | Latency | On localhost: cache hit p50 < 300 ms; full pipeline p50 < 10 s on the Groq free tier (excluding rate-limit backoff waits, which are logged separately) |
| NFR-6 | Compression fidelity | 0 tolerated cases where a compressed context contains a number not present verbatim in its source chunk |
| NFR-7 | Local runnability | A fresh clone runs end-to-end on a laptop CPU by following the README (ingest → serve → ask in browser). Heavy parsing runs offline in a separate environment; the serving environment has no PyTorch dependency. Memory and latency are recorded for the later hosting decision |
| NFR-8 | Cost ceiling | The entire build runs on the Groq free tier; ingestion enrichment is cached so re-runs cost zero API calls |
| NFR-9 | Reproducibility | Ingestion is deterministic given config + corpus; index artifacts are versioned |
| NFR-10 | Explainability | A debug panel shows retrieved chunks, scores, gate decisions, and cache decisions per answer |
| NFR-11 | Swappability | Switching model profile requires configuration only; the inactive Anthropic/Gemini templates pass a dry-run validation before the backend is declared complete |
| NFR-12 | Security (localhost) | Server binds to 127.0.0.1; admin and dev-clock endpoints are enabled only in dev mode; API keys live only in `.env` |

---

## 8. Constraints and assumptions

- **C1** Vector database is ChromaDB (fixed by the brief).
- **C2** Models [B7-Q2]: **Groq free tier only, for the whole build**. It is rate-limited with low tokens-per-minute ceilings, hence a 2,500-token generation context budget and client-side pacing. Anthropic and Gemini models are switched on later; that switch will need API billing (consumer chat subscriptions do not include API usage). Exact model IDs change often and are verified at build time.
- **C3** Free-tier inputs may be used by providers to improve their products. Acceptable here because the corpus is a public filing; not acceptable for private documents.
- **C4** Local ChromaDB (as of early 2026) does not enable sparse/BM25 indexing — that feature was reported as Chroma Cloud only [R-3]. BM25 is therefore run in-process alongside Chroma unless this has changed by build time.
- **C5** v1 corpus is a single company, single fiscal year. Multi-company support is a roadmap item, but metadata and cache slots include `entity` from day one.
- **C6** The system provides analysis of reported figures, not investment advice.
- **C7** "Year" in user queries is interpreted as NVIDIA fiscal year unless the user specifies a calendar date; ambiguity is surfaced in the answer.
- **C8** Hosting [B7-Q1][B7-Q3]: **localhost only** during the build. No containerization or online hosting until the backend is complete; Render or similar platforms will be evaluated then, using memory and latency measured locally.
- **C9** Groq's only vision-capable model is listed as a preview, so the test profile may lose image-based answering without notice.

---

## 9. Scope

**In scope (v1)**
- Ingestion of the single combined PDF: text, tables, charts, diagrams.
- Balance-sheet-centric Q&A, plus supporting MD&A, notes, and Proxy content.
- Query expansion, hybrid retrieval, gated reranking, classifier-gated compression.
- Two-tier semantic cache with TTL classes, versioned invalidation, industry-standard eviction (Redis `volatile-lfu` semantics for the semantic tier).
- Deterministic ratio calculator and mandatory citations.
- Simple HTML web UI on localhost with debug and cache-testing controls.
- Swappable model layer with the Groq profile active and Anthropic/Gemini templates validated.
- Evaluation harness; cache-policy benchmark; compression ablation.

**Out of scope (v1)**
- Live market data, stock prices, news.
- Multi-company peer comparison (designed for, not built).
- Fine-tuning any model.
- Online deployment (Render or similar), containerization, and public-access protection — decided after backend completion [B7-Q3].
- Switching to Anthropic/Gemini models and the generator bake-off — after backend completion [B7-Q2].
- Multi-turn conversation and session management (planned for v2 [B6-Q4]).
- User accounts and multi-tenant cache isolation (a shared demo access key is in scope; accounts are not).
- Image-embedding retrieval (CLIP / ColPali-style); v1 uses description-based figure retrieval with image passthrough [B6-Q6].
- XBRL ingestion (a strong v2 candidate for ground-truth numbers).

---

## 10. Golden question seeds

Values verified directly from the Consolidated Balance Sheets (PDF p. 141, USD millions). Ratios computed from those values. These seed the evaluation set in `architecture.md` §12.

| # | Intent | Question | Expected answer |
|---|---|---|---|
| G1 | POINT_LOOKUP | Total assets as of Jan 25, 2026? | $206,803M |
| G2 | POINT_LOOKUP | Inventories at fiscal year-end 2025 (Jan 26, 2025)? | $10,080M |
| G3 | COMPUTATION | Current ratio as of Jan 25, 2026? | 125,605 / 32,163 ≈ **3.91** |
| G4 | COMPUTATION | Working capital as of Jan 25, 2026? | 125,605 − 32,163 = **$93,442M** |
| G5 | COMPUTATION | Quick ratio (cash + marketable securities + receivables) / current liabilities? | 101,022 / 32,163 ≈ **3.14** |
| G6 | COMPUTATION | Total debt to equity? | (999 + 7,469) / 157,293 ≈ **0.054** |
| G7 | COMPARISON_TREND | YoY change in total assets? | +85.3% |
| G8 | COMPARISON_TREND | YoY change in inventories? | +112.3% |
| G9 | COMPARISON_TREND | YoY change in goodwill? | $5,188M → $20,832M (+301.5%) |
| G10 | COMPARISON_TREND | YoY change in non-marketable equity securities? | $3,387M → $22,251M (+557.0%) |
| G11 | EXPLANATORY | What drove the first-quarter FY2026 charge related to H20? | Narrative from 10-K; must be de-duplicated across 5 mentions |
| G12 | VISUAL | NVIDIA 5-year cumulative total return vs S&P 500 and Nasdaq 100? | From p. 123 chart and companion table |
| G13 | VISUAL | What are the layers in NVIDIA's "five-layer cake" framing? | Energy, chips, infrastructure, models, applications (p. 3) |
| G14 | OUT_OF_SCOPE | Should I buy NVIDIA stock? | Scoped refusal, no retrieval |
| G15 | Cache-adversarial | "Total assets FY2026" then "Total assets FY2025" | Second query must **miss** the cache |

---

## 11. Success criteria (definition of done)

Status at the backend-complete gate (2026-09-19):

1. ✅ All FR-1…FR-7 implemented and demonstrable in the UI debug panel — debug, cache and ops panels (`frontend/`).
2. ✅ NFR-1 to NFR-6 met on the golden + adversarial sets, or deviations documented with root cause — `docs/reports/nfr_results.md`.
3. ✅ Cache-policy benchmark report produced, comparing all eight options discussed in [B1] — `docs/reports/cache_policy_benchmark.md` (9 policies × 5 capacities).
4. ✅ Compression-classifier ablation produced: always-compress vs never-compress vs classifier-gated, on accuracy, tokens, and latency [B2] — `docs/reports/compression_ablation.md`.
5. ✅ Cost report: large-model calls and tokens per 100 queries, with and without cache/compression — cache: `cache_policy_benchmark.md` (calls avoided per capacity); compression: `compression_ablation.md` (large input tokens per question per arm); per-request tokens in every trace and the ops panel.
6. ✅ README explains every component with the reasoning recorded in the decision log (`architecture.md` §16) — README → *Component by component*.
7. ✅ Localhost runbook: fresh clone → ingest → serve → ask questions and exercise the cache walkthrough in the browser, following the README only — `docs/reports/fresh_clone_rehearsal.md` (scripted pass of the same steps; the browser click-through stays on the human-review list).
8. ✅ Model-swap readiness: Anthropic and Gemini profile templates pass a dry-run validation with no code changes — `docs/reports/model_swap_dry_run.md`.
9. ✅ Phase exit criteria in `implementation_plan.md` are all met — with the documented deviations (D-23, D-24, D-45, D-61, D-62).

---

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Free-tier quota exhaustion or model renames mid-build | Pipeline breaks | Config-driven model IDs, retry with backoff, provider fallback |
| Semantic cache serves a subtly wrong answer | Credibility loss — worst failure for a finance tool | Slot guard, conservative threshold, admission policy, adversarial test set |
| Vision model misreads a chart value | Wrong numbers | Prefer companion data tables when present; label chart-derived numbers as approximate |
| Table parser mis-aligns a column | Wrong period attribution | Cross-validate balance-sheet parse against pdfplumber extraction; totals check (assets = liabilities + equity) |
| Over-engineering for a 175-page corpus | Time sink | Each advanced component is gated and ablated; if a component shows no measured benefit it is documented as such |
| Groq free-tier limits slow the build | Ingestion or evaluation runs stall | Client-side pacing, resumable cached enrichment, evaluation on subsets, retrieval-only degradation mode |
| Groq platform or preview-model changes | Build blocked | Swappable profiles; cached enrichment keeps the index usable |
| Groq-judged evaluation is self-referential | Over-optimistic quality scores | Treat RAGAS scores as indicative; rely on deterministic numeric exact-match; re-run after the model switch |

---

## 13. Glossary

| Term | Meaning here |
|---|---|
| Row-fact chunk | One line item of a financial table, with both period values stored as numeric metadata |
| Slot | Structured attribute extracted from a query: entity, fiscal period, metric, statement, direction |
| TTL | Time-to-live: maximum age of a cache entry before it is considered stale |
| Eviction policy | Rule for which entry to remove when the cache is full (LRU, LFU, …) — distinct from TTL [B1] |
| Admission policy | Rule for which answers are allowed into the cache at all |
| Version keys | `corpus_version`, `prompt_version`, `generator_model`, `retrieval_config_hash` |
| Small model | Small API LLM of the active profile (Llama 3.1 8B Instant on Groq during the build) or a local ONNX cross-encoder / classical ML model |
| Large model | The strongest model of the active profile, used only for final answer generation |
| Model profile | Named set of small/large/vision models, context budget, and pacing (`groq_build` active; `anthropic`, `gemini` templates) |
| Dev clock | A development-only time offset used by all TTL, decay, and sweeper logic so expiry can be tested without waiting |
| Redis `volatile-lfu` | Redis eviction policy that evicts, among entries with a TTL, those with the lowest approximated and time-decayed access frequency |
