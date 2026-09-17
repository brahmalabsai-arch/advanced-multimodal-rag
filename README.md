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

**Build status:** Phase 0 — foundation and Groq model layer. Later phases add ingestion, the
query pipeline, the localhost UI and the cache.

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
  llm.py        role-based model layer: text() · json() · vision_json(); pacing, retries, ledger
  ingest/ query/ compress/ cache/ calc/ api/                 (later phases)
scripts/        smoke_llm.py
tests/          unit tests (mocked provider; no network)
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
