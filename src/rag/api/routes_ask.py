"""Core endpoints (architecture §8.3): /api/ask, /api/figures/{id}, /api/pages/{n}, /api/trace/{id}.

Input limits (Phase 8, §13): a question is rejected with 422 when it is empty or whitespace,
longer than `server.max_question_chars`, contains control characters (binary pasted into the
box) or has no letter or digit at all. A request that outlives `server.request_timeout_seconds`
answers 504; the pipeline thread finishes on its own and its trace line is still written.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.api.byok import KEY_HEADER, Credentials, credentials_from, http_error, use_key
from rag.api.payload import answer_payload
from rag.cache.records import CacheInfo
from rag.calc.calculator import CalculationResult
from rag.compress.pipeline import CompressionOutcome
from rag.graph import PipelineResult
from rag.llm import LLMError
from rag.query.assemble import ContextBlock
from rag.query.condense import CondenseResult, Turn
from rag.query.coverage import CoverageResult
from rag.query.generate import Answer
from rag.query.rerank import RerankDecision
from rag.query.retrieve import Candidate
from rag.query.scope import ScopeDecision
from rag.query.slots import QuerySlots
from rag.query.verify import VerifyResult

router = APIRouter(prefix="/api")

# Hard ceiling on the wire; the configured limit (`server.max_question_chars`, default 1,000)
# is enforced per request in `validate_question`.
MAX_QUESTION_CHARS = 20000
_FIGURE_ID = re.compile(r"^p\d{1,3}_\d{1,3}$")
_ALLOWED_CONTROL = {chr(9), chr(10), chr(13)}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    bypass_cache: bool = False
    # in-session memory (D-71): earlier turns, sent by the page; capped again by the pipeline
    history: list[Turn] = Field(default_factory=list, max_length=50)


def validate_question(question: str, max_chars: int) -> str:
    """Return the stripped question or raise the 422 that explains what was wrong."""
    q = question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="question is empty")
    if len(q) > max_chars:
        raise HTTPException(
            status_code=422, detail=f"question is longer than {max_chars} characters"
        )
    if any(unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL for ch in q):
        raise HTTPException(status_code=422, detail="question contains control characters")
    if not any(ch.isalnum() for ch in q):
        raise HTTPException(status_code=422, detail="question has no words")
    return q


class CitationOut(BaseModel):
    block_id: str
    chunk_id: str | None
    modality: str
    page: int | None
    section: str | None
    breadcrumb: str
    label: str
    page_image_url: str | None = None
    figure_image_url: str | None = None


class AnalysisOut(BaseModel):
    intent: str
    rule: str
    confidence: float
    source: str
    llm_called: bool
    llm_intent: str | None
    queries: list[dict[str, Any]]
    section_hints: list[str]
    sub_questions: list[str]
    paraphrases: list[str]
    hyde_passage: str | None
    hyde_rejected: bool
    needs_image: bool
    notes: list[str]


class DebugOut(BaseModel):
    condense: CondenseResult | None = None
    intent: str
    intent_rule: str | None
    slots: QuerySlots
    scope: ScopeDecision
    analysis: AnalysisOut
    rerank: RerankDecision
    compression: dict[str, Any]
    retrieval: dict[str, Any]
    candidates: list[Candidate]
    context_blocks: list[ContextBlock]
    context_tokens: int
    context_budget: int
    dropped: list[str]
    calculations: list[CalculationResult]
    verify: VerifyResult
    coverage: CoverageResult
    generation_attempts: int
    generator_role: str
    tokens_by_model: dict[str, dict[str, int]]
    latency_ms_by_node: dict[str, int]
    total_latency_ms: int
    corpus_version: str


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: Answer
    cache_tier: str
    cache: CacheInfo
    citations: list[CitationOut]
    figures: list[CitationOut]
    warning: str | None = None
    degraded: bool = False
    debug: DebugOut


def _figure_id(image_path: str | None) -> str | None:
    if not image_path:
        return None
    return image_path.rsplit("/", 1)[-1].removesuffix(".png")


def _compression_out(c: CompressionOutcome) -> dict[str, Any]:
    """Summary plus one row per chunk (decision, applied action, tokens, fidelity)."""
    out = c.summary()
    by_id = c.by_id
    rows = []
    for d in c.decision.chunks if c.decision else []:
        applied = by_id.get(d.chunk_id)
        rows.append(
            {
                "chunk_id": d.chunk_id,
                "modality": d.features.modality,
                "action": d.action,
                "applied": applied.applied if applied else d.action,
                "reason": d.reason,
                "stage": d.stage,
                "score": d.score,
                "tokens_before": d.features.chunk_tokens,
                "tokens_after": applied.tokens_after if applied else d.features.chunk_tokens,
                "relevance_density": d.features.relevance_density,
                "max_dup_sim": d.features.max_dup_sim,
                "numeric_density": d.features.numeric_density,
                "note": applied.note if applied else None,
                "reverted": applied.reverted if applied else False,
                "fidelity_ok": applied.fidelity.ok if applied and applied.fidelity else None,
                "merged_into": applied.merged_into if applied else None,
            }
        )
    out["chunks"] = rows
    out["query"] = c.decision.query.model_dump(exclude={"queries"}) if c.decision else None
    return out


def to_response(r: PipelineResult, store) -> AskResponse:  # noqa: ANN001 - IndexStore
    cited = set(r.verify.citations_found) | set(r.answer.citations)
    citations: list[CitationOut] = []
    figures: list[CitationOut] = []
    for b in r.context.blocks:
        chunk = store.get(b.chunk_id) if b.chunk_id else None
        page_url = f"/api/pages/{b.page}" if b.page else None
        fig_url = None
        if chunk is not None and chunk.metadata.modality == "figure":
            fid = _figure_id(chunk.metadata.image_path)
            fig_url = f"/api/figures/{fid}" if fid else None
        out = CitationOut(
            block_id=b.block_id,
            chunk_id=b.chunk_id,
            modality=b.modality,
            page=b.page,
            section=b.section,
            breadcrumb=b.breadcrumb,
            label=b.header.strip("[]"),
            page_image_url=page_url,
            figure_image_url=fig_url,
        )
        if b.block_id in cited:
            citations.append(out)
        if fig_url and (b.block_id in cited or b.chunk_id in r.context.image_chunk_ids):
            figures.append(out)
    return AskResponse(
        request_id=r.request_id,
        question=r.question,
        answer=r.answer,
        cache_tier=r.cache_tier,
        cache=r.cache,
        citations=citations,
        figures=figures,
        warning=r.warning,
        degraded=r.degraded,
        debug=DebugOut(
            condense=r.condense,
            intent=r.intent,
            intent_rule=r.intent_rule,
            slots=r.slots.model_copy(update={"matches": []}),
            scope=r.scope,
            analysis=AnalysisOut(
                intent=r.analysis.intent,
                rule=r.analysis.rule,
                confidence=r.analysis.confidence,
                source=r.analysis.source,
                llm_called=r.analysis.llm_called,
                llm_intent=r.analysis.llm_intent,
                queries=[q.model_dump() for q in r.analysis.queries],
                section_hints=r.analysis.section_hints,
                sub_questions=r.analysis.sub_questions,
                paraphrases=r.analysis.paraphrases,
                hyde_passage=r.analysis.hyde_passage,
                hyde_rejected=r.analysis.hyde_rejected,
                needs_image=r.analysis.needs_image,
                notes=r.analysis.notes,
            ),
            rerank=r.rerank,
            compression=_compression_out(r.compression),
            retrieval={
                "queries": r.retrieval.queries,
                "query_kinds": r.retrieval.query_kinds,
                "filters": r.retrieval.filters,
                "filtered_retries": r.retrieval.filtered_retries,
                "dense_top": r.retrieval.dense_top,
                "bm25_top": r.retrieval.bm25_top,
                "fused_count": r.retrieval.fused_count,
                "filtered_retry": r.retrieval.filtered_retry,
                "rerank_applied": r.retrieval.rerank_applied,
                "rerank_skip_reason": r.retrieval.rerank_skip_reason,
            },
            candidates=r.retrieval.candidates,
            context_blocks=r.context.blocks,
            context_tokens=r.context.tokens_used,
            context_budget=r.context.token_budget,
            dropped=r.context.dropped,
            calculations=r.calculations,
            verify=r.verify,
            coverage=r.coverage,
            generation_attempts=r.generation_attempts,
            generator_role=r.generator_role,
            tokens_by_model=r.tokens_by_model,
            latency_ms_by_node=r.latency_ms_by_node,
            total_latency_ms=r.total_latency_ms,
            corpus_version=r.corpus_version,
        ),
    )


def _pipeline_and_client(request: Request, creds: Credentials | None):
    """Pick the pipeline for this request's provider and the client that will pay for it.

    With credentials: the provider's own pipeline (its own cache version keys, its own context
    budget) and a client built from the caller's key. Without: the process default, which only
    exists when a key is configured locally.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not ready")
        return pipeline, (creds.client() if creds else None)
    provider = creds.provider if creds else registry.default_provider
    try:
        pipeline = registry.for_provider(provider)
    except LLMError as exc:
        raise http_error(exc) from None
    return pipeline, (creds.client() if creds else None)


@router.post("/ask")
async def ask(body: AskRequest, request: Request) -> Any:
    """Answer one question and return the page's payload (`api/payload.py`).

    Credentials are required when the server runs in BYOK mode (`BYOK_ONLY=true`), and accepted
    but optional otherwise, so the same route serves the localhost build and the public demo.
    `?debug=1` adds the full pipeline payload, in dev environments only.
    """
    server = request.app.state.app_config.server
    settings = request.app.state.settings
    question = validate_question(body.question, server.max_question_chars)
    creds = credentials_from(request) if settings.byok or request.headers.get(KEY_HEADER) else None
    pipeline, client = _pipeline_and_client(request, creds)
    with use_key(creds.key if creds else None):
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(
                    pipeline.ask,
                    question,
                    bypass_cache=body.bypass_cache,
                    client=client,
                    history=body.history,
                ),
                timeout=server.request_timeout_seconds,
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"request exceeded {server.request_timeout_seconds:g}s; the model "
                "provider may be backing off — try again or ask a cached question",
            ) from None
        except LLMError as exc:
            # A model failure inside the graph degrades instead of raising; reaching here means
            # the request could not run at all (bad key, provider unreachable, no client).
            raise http_error(exc) from None
        full = to_response(result, pipeline.store)
        payload = answer_payload(
            result,
            [
                b
                for b in result.context.blocks
                if b.block_id in {c.block_id for c in full.citations}
            ],
            pipeline.store,
        )
        # The localhost build keeps the full pipeline payload (the debug view depends on it);
        # a deployed BYOK server returns only what the page renders.
        if settings.is_dev and not settings.byok_only:
            payload["debug"] = full.model_dump(mode="json")
        return payload


@router.get("/figures/{figure_id}")
def figure(figure_id: str, request: Request) -> FileResponse:
    if not _FIGURE_ID.match(figure_id):
        raise HTTPException(status_code=404, detail="unknown figure")
    path = request.app.state.figures_root / "figures" / f"{figure_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown figure")
    return FileResponse(path, media_type="image/png")


@router.get("/pages/{page}")
def page(page: int, request: Request) -> FileResponse:
    if page < 1 or page > 999:
        raise HTTPException(status_code=404, detail="unknown page")
    path = request.app.state.figures_root / "pages" / f"p{page}.webp"
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown page")
    return FileResponse(path, media_type="image/webp")


@router.get("/trace/{request_id}")
def trace(request_id: str, request: Request) -> dict[str, Any]:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    found = pipeline.trace_writer.find(request_id)
    if found is None:
        raise HTTPException(status_code=404, detail="unknown request id")
    return found.model_dump(exclude_none=True)
