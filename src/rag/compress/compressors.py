"""Compressors (architecture §4.8): DEDUPE, ROW_SELECT, EXTRACT_LIGHT, EXTRACT_LLM, DROP.

Every non-KEEP output passes the numeric fidelity guard against its source chunk; a violation
reverts to the original text and is logged. `EXTRACT_LLM` is the only compressor that calls a
model (the `small` role) and it asks for verbatim sentence copies, never rewrites.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.compress.classifier import Action, CompressionDecision
from rag.compress.features import SentenceView
from rag.compress.fidelity import FidelityResult, check_fidelity
from rag.core.config import CompressionThresholds
from rag.core.logging import get_logger
from rag.core.tokens import count_tokens
from rag.ingest.validate import normalize_label
from rag.llm import LLMClient, LLMError
from rag.query.store import IndexStore

log = get_logger(__name__)

MAX_LLM_SENTENCES = 6
ELLIPSIS = "[…]"


class CompressedChunk(BaseModel):
    chunk_id: str
    action: Action = Field(description="what the classifier decided")
    applied: Action = Field(description="what actually happened after guards / fallbacks")
    text: str = Field(description="body sent to the model (original when reverted)")
    tokens_before: int
    tokens_after: int
    dropped: bool = False
    merged_into: str | None = None
    merged_from: list[str] = Field(default_factory=list)
    kept_sentence_ids: list[str] = Field(default_factory=list)
    kept_rows: int | None = None
    fidelity: FidelityResult | None = None
    reverted: bool = False
    note: str | None = None
    llm_called: bool = False


class CompressionResult(BaseModel):
    chunks: list[CompressedChunk] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    llm_calls: int = 0
    violations: list[str] = Field(default_factory=list)
    applied_actions: dict[str, int] = Field(default_factory=dict)

    def by_id(self) -> dict[str, CompressedChunk]:
        return {c.chunk_id: c for c in self.chunks}


class _Extracted(BaseModel):
    sentences: list[str | None] = Field(default_factory=list)


# ------------------------------------------------------------------------------ helpers


def row_select(body: str, metrics: list[str]) -> tuple[str, int, int]:
    """Keep the table preamble, header + separator, rows whose label matches a query metric,
    section-label rows (no numeric cells) and total rows. Returns (text, rows_kept, rows_total)."""
    lines = body.splitlines()
    preamble: list[str] = []
    table: list[str] = []
    for ln in lines:
        (table if ln.lstrip().startswith("|") or table else preamble).append(ln)
    if len(table) < 3:
        return body, 0, 0
    header, sep, rows = table[0], table[1], table[2:]
    wanted = {normalize_label(m) for m in metrics}
    kept: list[str] = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if not cells:
            continue
        label = normalize_label(cells[0])
        values = cells[1:]
        is_section = not any(any(ch.isdigit() for ch in v) for v in values)
        is_total = label.startswith("total") or " total" in f" {label}"
        if label in wanted or is_section or is_total:
            kept.append(row)
    if not kept:
        return body, 0, len(rows)
    text = "\n".join([*preamble, header, sep, *kept])
    return text, len(kept), len(rows)


def select_sentences(view: SentenceView, tau: float, *, neighbours: int = 1) -> list[int]:
    """Indexes of sentences scoring ≥ tau plus `neighbours` on each side; falls back to the
    top two sentences when nothing clears the threshold."""
    n = len(view.scores)
    if n == 0:
        return []
    hits = [i for i, s in enumerate(view.scores) if s >= tau]
    if not hits:
        hits = sorted(range(n), key=lambda i: -view.scores[i])[:2]
    keep: set[int] = set()
    for i in hits:
        for j in range(max(0, i - neighbours), min(n, i + neighbours + 1)):
            keep.add(j)
    return sorted(keep)


def _join_sentences(view: SentenceView, idx: list[int]) -> str:
    parts: list[str] = []
    prev = None
    for i in idx:
        if prev is not None and i != prev + 1:
            parts.append(ELLIPSIS)
        parts.append(view.texts[i])
        prev = i
    return " ".join(parts)


_LLM_SYSTEM = (
    "You compress a passage from NVIDIA's annual report for a question-answering system. "
    "Return ONLY sentences copied verbatim from the passage that are needed to answer the "
    "question — do not rewrite, paraphrase, summarise or change any number, date or unit. "
    f"At most {MAX_LLM_SENTENCES} sentences. Respond with JSON only."
)


def extract_llm(
    client: LLMClient, question: str, body: str, *, request_id: str | None, max_tokens: int = 700
) -> list[str]:
    out = client.json(
        f"QUESTION: {question}\n\nPASSAGE:\n{body}\n\nRespond with the JSON object described.",
        _Extracted,
        role="small",
        system=_LLM_SYSTEM,
        request_id=request_id,
        max_tokens=max_tokens,
    )
    return [s.strip() for s in out.sentences if s and s.strip()]


# --------------------------------------------------------------------------------- apply


def apply_compression(
    store: IndexStore,
    decision: CompressionDecision,
    views: dict[str, SentenceView],
    *,
    question: str,
    metrics: list[str],
    thresholds: CompressionThresholds,
    body_of,  # noqa: ANN001 - Callable[[Chunk], str]
    client: LLMClient | None = None,
    request_id: str | None = None,
) -> CompressionResult:
    result = CompressionResult()
    by_id: dict[str, CompressedChunk] = {}
    for d in decision.chunks:
        chunk = store.get(d.chunk_id)
        if chunk is None:
            continue
        body = body_of(chunk)
        before = d.features.chunk_tokens or count_tokens(body)
        out = CompressedChunk(
            chunk_id=d.chunk_id,
            action=d.action,
            applied=d.action,
            text=body,
            tokens_before=before,
            tokens_after=before,
        )
        if d.action == "KEEP":
            pass
        elif d.action == "DROP":
            out.dropped = True
            out.tokens_after = 0
        elif d.action == "DEDUPE":
            target = d.features.dup_of
            out.dropped = True
            out.tokens_after = 0
            out.merged_into = target
            if target in by_id:
                by_id[target].merged_from.append(d.chunk_id)
        elif d.action == "ROW_SELECT":
            text, kept, total = row_select(body, metrics)
            if kept == 0:
                out.applied = "KEEP"
                out.note = "no query metric row found; table kept whole"
            else:
                out.text, out.kept_rows = text, kept
                out.note = f"{kept} of {total} rows kept"
        elif d.action in {"EXTRACT_LIGHT", "EXTRACT_LLM"}:
            view = views.get(d.chunk_id) or SentenceView()
            sentences: list[str] | None = None
            if d.action == "EXTRACT_LLM" and client is not None:
                out.llm_called = True
                result.llm_calls += 1
                try:
                    sentences = extract_llm(client, question, body, request_id=request_id)
                except LLMError as exc:
                    out.note = (
                        f"small-model extraction failed ({str(exc)[:80]}); sentence selection used"
                    )
                    sentences = None
            if sentences is not None:
                out.text = " ".join(sentences) if sentences else body
                if not sentences:
                    out.applied = "KEEP"
                    out.note = "model returned no sentences; chunk kept"
            else:
                if d.action == "EXTRACT_LLM":
                    out.applied = "EXTRACT_LIGHT"
                idx = select_sentences(view, thresholds.sentence_relevance_tau)
                if not idx:
                    out.applied = "KEEP"
                    out.note = "no sentence sidecar; chunk kept"
                else:
                    out.text = _join_sentences(view, idx)
                    out.kept_sentence_ids = [view.sentence_ids[i] for i in idx]
        if out.applied not in {"KEEP", "DROP", "DEDUPE"} and not out.dropped:
            fid = check_fidelity(
                out.text,
                body,
                sentences=None if out.applied != "EXTRACT_LLM" else out.text.split(". "),
            )
            out.fidelity = fid
            if not fid.ok:
                result.violations.append(
                    f"{d.chunk_id}: {out.applied} introduced {fid.missing_numbers} → reverted"
                )
                log.warning(
                    "fidelity violation on %s (%s): %s",
                    d.chunk_id,
                    out.applied,
                    fid.missing_numbers,
                )
                out.text = body
                out.reverted = True
                out.applied = "KEEP"
            out.tokens_after = count_tokens(out.text)
            if out.tokens_after >= out.tokens_before and out.applied != "KEEP":
                out.note = (out.note + "; " if out.note else "") + "no reduction"
        by_id[d.chunk_id] = out
        result.chunks.append(out)
    result.tokens_before = sum(c.tokens_before for c in result.chunks)
    result.tokens_after = sum(c.tokens_after for c in result.chunks)
    for c in result.chunks:
        result.applied_actions[c.applied] = result.applied_actions.get(c.applied, 0) + 1
    return result
