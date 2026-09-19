# Fresh-clone rehearsal (Phase 8, NFR-7)

Run 2026-09-19 · Windows 11, Python 3.13 (Windows Store build), Git Bash · scripted as
`rehearsal.sh` (session scratchpad; the commands are the README runbook verbatim) · log kept
with the session, summarised here.

Result: **pass** — after two defects the first pass surfaced were fixed, the full sequence
*new directory → two environments → `.env` + PDF → ingest → tests → profile dry run → serve →
ask G1 → cache walkthrough* runs from the README commands alone.

## What "fresh clone" meant here

Nothing since `b7e1c6b` is committed yet, so a `git clone` would not contain Phases 5–8. The
rehearsal instead copied exactly the files git would commit — `git ls-files --cached --others
--exclude-standard` (159 files: tracked + untracked, not ignored) — into an empty directory.
`data/`, `.env`, both `.venv*`, `eval/results/` and every cache were therefore absent, as in a
real clone. Two conveniences, both noted: the `GROQ_API_KEY` line was copied from the
developer's `.env` rather than typed, and the PDF was copied from the build machine's
`data/raw/`.

## Timings (first pass)

| Step | Command (README §Runbook) | Wall time | Notes |
|---|---|---:|---|
| 1a | `python -m venv .venv` + pip upgrade | 14 s | |
| 1b | `pip install -r requirements-dev.txt` | 225 s | pip cache was cold (18 MB) — everything downloaded |
| 1c | `pip install -e . --no-deps` | 8 s | |
| 1d | `python -m venv .venv-ingest` + pip upgrade | 13 s | |
| 1e | `pip install -r requirements-ingest.txt` | 289 s | torch 2.14 CPU, Docling 2.128, spaCy |
| 1f | `pip install -e . --no-deps` | 8 s | |
| 2 | `.env` from `.env.example`; PDF into `data/raw/` | — | |
| 3 | `python -m rag.ingest.run` | **2,091 s** | see breakdown below |
| 4a | `python -m pytest` | 52 s → 27 s | first pass: 2 failures (defect B); second pass: all passed (1 skipped) |
| 4b | `scripts/check_profile.py --profile all --dry-run` | 3 s | all 4 profiles pass; `anthropic` / `gemini` warn "package not installed" (expected on a clone) |
| 5 | `uvicorn rag.api.main:app --host 127.0.0.1 --port 8000` | ready in ≈ 2 s | startup checks `bind` / `admin` / `secrets` / `index` all ok; pipeline 1.0 s |
| 6 | `scripts/ask_cli.py --api "…total assets as of Jan 25, 2026?"` | 3 s (first pass crashed: defect A) | second pass: `$206,803 million as of Jan 25, 2026 (FY2026).` · `[C1]` PDF p.141 row fact · verify true · 1,599 in / 250 out tokens |
| 7 | `scripts/cache_walkthrough.py --api` | 110 s (first pass crashed: defect A) | second pass: 9 of 9 automatable steps ✓; step 7 (config edit + restart) is "manual" in API mode by design |

Ingestion breakdown (from the log): Docling model download + parse of the 175-page PDF ≈ 4.5 min;
elements / validate / figures / report ≈ 1 min (accounting identity held, `passed=True`); semantic
chunking ≈ 1 min; table summaries ≈ 8 min (101 `small` calls, paced on the 8K TPM bucket);
figure descriptions ≈ 21 min (43 `vision` calls, one 429 retried, paced on the 1,000 OTPM
bucket); index ≈ 1 min. **The rebuilt index has the same `corpus_version` (`747912fc5da6`) and
the same counts (447 text, 143 table, 83 row facts, 16 figures = 689 chunks, 3,541 sentences) as
the build machine's** — ingestion is deterministic across directories (NFR-9), and the manifest
consistency check at startup passed on the rebuilt files.

Groq usage for the whole rehearsal (clone ledger): `gpt-oss-20b` 101 calls / 63K tokens,
`qwen3.8-27b` 43 calls / 95K tokens, `gpt-oss-120b` 30 calls / 25K tokens (4 rate-limited
attempts, all retried). Three separate per-model daily buckets — the rehearsal did not touch the
build machine's `gpt-oss-120b` window beyond G1 and the walkthrough.

## Defects found (and fixed before the second pass)

**A. CLI scripts crashed on a cp1252 console.** `ask_cli.py --api` and `cache_walkthrough.py`
both raised `UnicodeEncodeError` printing the *successful* answer (`'charmap' codec can't
encode U+202F` — the narrow no-break space in "$206,803 million"). The build machine had only
ever run them through a UTF-8 terminal. Fix: `rag.core.console.utf8_console()` called first in
every script under `scripts/` and `eval/` (19 files); two tests guard it
(`test_utf8_console_makes_cp1252_stdout_print_model_answers`,
`test_every_cli_script_switches_the_console_to_utf8`).

**B. A Phase 8 test assumed the future provider packages were installed.**
`test_templates_pass_dry_run_without_keys` expected the constructor line to say "placeholder
key"; on a clone without `langchain-anthropic` / `langchain-google-genai` the dry run correctly
reports "skipped — package not installed". The test now accepts both; the script's wording was
made explicit.

**C. (Caught while writing the runbook, before the rehearsal.)** The old README's one-line
`pip install -r requirements-dev.txt -e . --no-deps` would have skipped every transitive
dependency (`--no-deps` is global). The runbook now uses the two commands the Makefile uses.

Nothing else deviated: no hand edits to config, no extra packages, no environment variables
beyond `.env`.

## What this does and does not show

- Shows: the README is sufficient on a clean Windows machine with Python 3.13; ingestion is
  reproducible byte-for-byte at the manifest level; the serving environment has no PyTorch;
  Phase 8's startup checks and limits appear in `/readyz`; the cache walkthrough's automatable
  steps pass on a freshly built index.
- Does not show: macOS/Linux paths (`.venv/bin/python`) — untested here; a true `git clone`
  (pending the post-Phase-8 commit); the browser click-through, which was not repeated by hand
  (the scripted API-mode pass covers the same requests).
- Wall-clock budget for a newcomer: ≈ 9 min of installs + ≈ 35 min of ingestion on this CPU with
  the free tier's pacing, then seconds per question. A second `make ingest` makes zero model calls
  (enrichment cache), which the Phase 2 report already measured.
