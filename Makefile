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

.PHONY: setup setup-serve setup-ingest lock ingest serve test lint format eval bench smoke clean

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

# Phase 1+: parse -> chunk -> enrich (Groq, cached) -> index
ingest:
	$(PY_INGEST) -m rag.ingest.run

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

# Phase 3+: golden set, always bypass_cache=true (rule G8)
eval:
	$(PY) eval/run_eval.py --bypass-cache

# Phase 7: cache-policy benchmark (simulated, no LLM calls)
bench:
	$(PY) eval/cache_benchmark.py

# Phase 0 exit criterion: one call per role, three ledger lines.
smoke:
	$(PY) scripts/smoke_llm.py

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache']]"
