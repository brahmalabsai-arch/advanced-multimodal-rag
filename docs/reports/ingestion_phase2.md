# Ingestion report — Phase 2 (chunk, enrich, index)

Generated 2026-09-17T16:24:25+00:00 · chunks: text 447, table 143, row_fact 83, figure 16 (total 689) · sentences 3541 · text blocks 476 · split threshold p90 = 0.4839

## Indexes

| Index | Embedder | Dim | corpus_version | Chunks |
|---|---|---|---|---|
| `index` | BAAI/bge-small-en-v1.5 | 384 | `747912fc5da6` | 689 |
| `index_bge_base` | BAAI/bge-base-en-v1.5 | 768 | `ffcd60acbd8a` | 689 |

## Enrichment (Groq, cached)

Models: {'table_summaries': 'openai/gpt-oss-20b', 'figures': 'qwen/qwen3.8-27b'} · cache hits 141 / misses 2 on the last run · figures dropped: {'logo': 5, 'photo': 15, 'duplicate': 2, 'decorative': 2, 'unannotated': 2}

| Ledger job | Calls | ok | Tokens in | Tokens out |
|---|---|---|---|---|
| ingest-tables | 100 | 100 | 55352 | 6909 |
| ingest-figures | 53 | 37 | 76672 | 7779 |

## Figure classification

| Figure | Type | Title | Data points | Companion table |
|---|---|---|---|---|
| `fig_p3_0` | diagram | AI Is a Five-Layer Cake | 0 |  |
| `fig_p4_0` | diagram | NVIDIA AI Infrastructure Ecosystem | 0 |  |
| `fig_p4_1` | diagram | AI Infrastructure Buildout | 0 |  |
| `fig_p7_0` | chart | NVIDIA is World's Largest Contributor to Open-Source AI | 5 |  |
| `fig_p8_0` | diagram | Agentic AI Architecture: Nemoclaw, OpenShell, and NVIDIA AI Stack | 0 |  |
| `fig_p8_1` | diagram | Agentic AI Enters Production | 0 |  |
| `fig_p24_0` | chart | Business Highlights Fiscal 2026 Shareholder Returns | 9 |  |
| `fig_p43_0` | table_image | Our Lead Director: Stephen C. Neal | 0 |  |
| `fig_p43_1` | table_image | Duties of Our Lead Director | 0 |  |
| `fig_p45_0` | table_image | CC and NCGC Committee Members and Responsibilities | 0 |  |
| `fig_p57_0` | chart | Fiscal 2026 Target Pay Mix | 9 |  |
| `fig_p59_0` | diagram | Fiscal 2026 Executive Compensation Program Timeline | 0 |  |
| `fig_p60_0` | chart | NVIDIA Revenue and Market Capitalization vs. Peer Group Percentiles | 8 |  |
| `fig_p76_0` | chart | NEO CAP versus TSR | 17 |  |
| `fig_p76_1` | chart | Relationships Between Compensation Actually Paid and Financial Perform | 16 |  |
| `fig_p123_0` | chart | Comparison of 5 Year Cumulative Total Return Among NVIDIA Corporation, | 16 | tbl_p123_0 |
