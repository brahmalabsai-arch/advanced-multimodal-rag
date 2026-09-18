# Advanced Multimodal RAG — build targets (implementation plan, Phase 0).
# Two environments (D-49): .venv (serve + dev) and .venv-ingest (Docling stack).
# On Windows without GNU make: `winget install ezwinports.make`, or run the commands below by hand.

ifeq ($(OS),Windows_NT)
  VENV_BIN   := .venv/Scripts
  INGEST_BIN := .venv-ingest/Scripts
  SYS_PY     := python
else
  VENV_BIN   := .venv/bin
  INGEST_BIN := .venv-ingest/bin
  SYS_PY     := python3.11
endif

PY        := $(VENV_BIN)/python
PY_INGEST := $(INGEST_BIN)/python

.PHONY: setup setup-serve setup-ingest lock ingest index-base inspect serve test lint format eval eval-retrieval embedder-gate ask bench smoke clean

setup: setup-serve setup-ingest

setup-serve:
	$(SYS_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

setup-ingest:
	$(SYS_PY) -m venv .venv-ingest
	$(PY_INGEST) -m pip install --upgrade pip
	$(PY_INGEST) -m pip install -r requirements-ingest.txt
	$(PY_INGEST) -m pip install -e . --no-deps

# Freeze the resolved versions once both environments install cleanly (Phase 0 task).
lock:
	$(PY) -m pip freeze --exclude-editable > requirements-dev.lock.txt
	$(PY_INGEST) -m pip freeze --exclude-editable > requirements-ingest.lock.txt

# parse -> elements -> validate -> figures -> report -> chunk (Groq, cached) -> index (bge-small)
ingest:
	$(PY_INGEST) -m rag.ingest.run

# Second index with the challenger embedder for the Phase 3 embedder gate (no Groq calls)
index-base:
	$(PY_INGEST) -m rag.ingest.run --stage index --embedder bge-base

inspect:
	$(PY) scripts/inspect_index.py --bm25 "inventories" -k 3
	$(PY) scripts/inspect_index.py --dense "inventories at fiscal year end 2026" -k 3

# Phase 3+: API + HTML on localhost only (D-44). One worker keeps one L1 cache and one Chroma writer.
serve:
	$(PY) -m uvicorn rag.api.main:app --host 127.0.0.1 --port 8000 --workers 1

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

# Golden set through the full pipeline; always bypass_cache=true (rule G8). ~150K Groq tokens.
eval:
	$(PY) eval/run_eval.py --name golden

# Retrieval metrics only (no Groq calls) and the embedder gate
eval-retrieval:
	$(PY) eval/run_eval.py --retrieval-only --name retrieval
embedder-gate:
	$(PY) eval/embedder_gate.py --lock

ask:
	$(PY) scripts/ask_cli.py "$(Q)"

# Phase 7: cache-policy benchmark (simulated, no LLM calls)
bench:
	$(PY) eval/cache_benchmark.py

# Phase 0 exit criterion: one call per role, three ledger lines.
smoke:
	$(PY) scripts/smoke_llm.py

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache']]"
