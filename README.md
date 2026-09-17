# Advanced Multimodal RAG — Balance Sheet Analysis (NVIDIA FY2026)

Retrieval-augmented question answering over NVIDIA's FY2026 combined annual report: text, tables,
charts and diagrams in one ChromaDB index; hybrid retrieval with gated reranking and
classifier-gated compression; a two-tier semantic cache with TTL classes; deterministic ratio
maths; page-level citations. Built on Groq's free tier, served on localhost.

| Document | Purpose |
|---|---|
| [docs/problemstatement.md](docs/problemstatement.md) | What is being built and why |
| [docs/architecture.md](docs/architecture.md) | How, with the decision log |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Phase-by-phase build plan and exit criteria |

**Build status:** Phase 2 complete — the index exists. Phase 0 (foundation, Groq model layer),
Phase 1 (Docling parse, statement validation, figure candidates) and Phase 2 (semantic chunks,
row facts, figure descriptions, Chroma + BM25 index) are done; Phase 3 adds the baseline RAG
pipeline and the localhost UI.

## Environment setup (Phase 0)

Two virtual environments keep Docling's PyTorch dependency out of the serving process (D-49):

| Env | Requirements | Used for |
|---|---|---|
| `.venv` | `requirements-dev.txt` (= serve + dev tools) | API, query pipeline, tests, evaluation |
| `.venv-ingest` | `requirements-ingest.txt` | `make ingest` (Docling, spaCy, rasterisation) |

Python 3.11+ (the build machine uses 3.13, see D-51). GNU make is optional; every target is a
one-liner you can run directly (on Windows: `winget install ezwinports.make`).

```bash
# Windows (Git Bash / PowerShell)                 # macOS / Linux
python -m venv .venv                              python3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt -e . --no-deps
python -m venv .venv-ingest                       python3.11 -m venv .venv-ingest
.venv-ingest/Scripts/python -m pip install -r requirements-ingest.txt -e . --no-deps

cp .env.example .env      # add GROQ_API_KEY; keep MODEL_PROFILE=groq_build, APP_ENV=dev
```

Or, with make: `make setup`. Pinned versions from the last clean install are in
`requirements-dev.lock.txt` / `requirements-ingest.lock.txt` (`make lock` regenerates them).

## Checks

```bash
make test     # .venv/Scripts/python -m pytest
make lint     # .venv/Scripts/python -m ruff check . && ruff format --check .
make smoke    # .venv/Scripts/python scripts/smoke_llm.py   (3 Groq calls, needs GROQ_API_KEY)
```

`make smoke` makes one call per role — `small` text, `large` JSON, `vision` JSON with
`tests/fixtures/smoke_image.png` — and checks that three `ok` lines were appended to
`data/logs/llm_usage.jsonl`.

## Ingestion (Phase 1: parse and validate)

```bash
cp /path/to/2026_NVIDIA_ANNUAL_REPORT.pdf data/raw/
make ingest        # .venv-ingest/Scripts/python -m rag.ingest.run   (~5 min on CPU for Docling)
```

Stages (each writes plain files, so later stages never import Docling):

| Stage | Module | Output |
|---|---|---|
| parse | `rag.ingest.parse` | `data/parsed/docling.json` (+ `docling.meta.json`; re-runs skip when the PDF hash is unchanged) |
| elements | `rag.ingest.elements`, `sections.py` | `data/parsed/elements.jsonl` — every text/heading/table/picture with page, bbox, section, subsection, statement |
| validate | `rag.ingest.validate` | `data/parsed/statements.json` — pdfplumber rows for the balance sheet, income statement and cash flow, cross-checked cell by cell against Docling; **ingestion stops if total assets ≠ total liabilities + equity** |
| figures | `rag.ingest.figures` | `data/parsed/figure_candidates.json`, `data/index/figures/*.png` (200 DPI crops), `data/index/pages/*.webp` (thumbnails) |
| report | `rag.ingest.report` | `data/parsed/ingestion_report.json` and [docs/reports/ingestion_phase1.md](docs/reports/ingestion_phase1.md) |

| chunk | `rag.ingest.chunk_semantic`, `tables.py`, `figures.py`, `enrich.py` | `data/parsed/chunks.jsonl` + `sentences.jsonl` — semantic text chunks (120–450 tokens, corpus-level p90 split), table chunks with one-line summaries (small model), row facts for the three statements with numeric metadata, figure chunks described by the vision model (photos/logos/decoratives dropped; charts linked to their companion table). Every LLM result is cached in `data/parsed/enrichment/`, so a re-run makes **zero** Groq calls |
| index | `rag.ingest.index` | `data/index/` — Chroma `report_chunks` (cosine, explicit `bge-small` vectors), `bm25.pkl` (finance-aware tokenizer), `sentences.jsonl` + `sentence_emb.npy` sidecars, `manifest.json` with `corpus_version` |

`--stage <name>` runs one stage; `--force-parse` re-runs Docling; `--no-pages` skips thumbnails;
`--no-llm` chunks without Groq (no summaries or figure chunks). `make index-base` builds the
`bge-base` challenger index under `data/index_bge_base/` for the Phase 3 embedder gate.

Poke at the index from the serving environment:

```bash
.venv/Scripts/python scripts/inspect_index.py --bm25 "inventories" -k 5
.venv/Scripts/python scripts/inspect_index.py --dense "inventories at fiscal year end 2026" -k 5
.venv/Scripts/python scripts/inspect_index.py --modality figure
```

## Configuration

| File | Contents |
|---|---|
| `.env` | `GROQ_API_KEY`, `MODEL_PROFILE` (default `groq_build`), `APP_ENV` (`dev` / `test` / `prod`) |
| `config/models.yaml` | Model profiles. `groq_build` is active; `anthropic` and `gemini` are validated templates for the later switch (F1). Per-model `pacing` holds the provider's RPM/TPM limits. |
| `config/app.yaml` | Bind address (`127.0.0.1:8000`), dev-clock enablement, data paths |
| `config/thresholds.yaml` | Retrieval, rerank, compression and cache tunables (architecture §11.1) |

Model ids are configuration, never code. Groq's Llama 3.x and Llama 4 Scout models were retired
from the free tier before this build started, so `groq_build` uses `openai/gpt-oss-20b`,
`openai/gpt-oss-120b` and `qwen/qwen3.8-27b` (decision D-50).

## Repository layout

```
config/         models.yaml · app.yaml · thresholds.yaml   (glossary, formulas, fiscal calendar arrive in Phase 4)
data/           raw/ parsed/ index/ cache/ logs/            (gitignored; rebuilt locally)
docs/           problem statement · architecture · plan · reports/
src/rag/
  core/         settings.py config.py logging.py tokens.py pacing.py ledger.py
                schema.py (chunk metadata contract) embeddings.py (fastembed) bm25.py
  llm.py        role-based model layer: text() · json() · vision_json(); pacing, retries, ledger
  ingest/       run.py parse.py elements.py sections.py validate.py figures.py report.py
                sentences.py chunk_semantic.py tables.py enrich.py index.py
  query/ compress/ cache/ calc/ api/                         (later phases)
scripts/        smoke_llm.py inspect_index.py
tests/          unit tests (mocked provider; no network). Ingestion tests read the PDF when present
frontend/ eval/                                             (Phase 3+)
```

## The model layer in one paragraph

`rag.llm.LLMClient` resolves the roles `small`, `large` and `vision` through the active profile
and exposes `text()`, `json(schema)` and `vision_json(schema, images)`. Provider packages are
imported lazily (only `langchain-groq` is installed during the build). JSON calls request Groq
JSON mode, validate the reply against the Pydantic schema, and retry once with the validation
error appended before raising `LLMJSONError`. Before every call a per-model token bucket paces
requests and tokens per minute (`core/pacing.py`); on 429/5xx `tenacity` backs off with jitter,
never shorter than the provider's `retry-after`. Every attempt — success, retry or failure — is a
line in `data/logs/llm_usage.jsonl`. Logging masks API keys by value and by shape.
