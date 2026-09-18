"""Core endpoints (architecture §8.3): /api/ask, /api/figures/{id}, /api/pages/{n}, /api/trace/{id}."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult
from rag.graph import PipelineResult
from rag.query.assemble import ContextBlock
from rag.query.generate import Answer
from rag.query.rerank import RerankDecision
from rag.query.retrieve import Candidate
from rag.query.scope import ScopeDecision
from rag.query.slots import QuerySlots
from rag.query.verify import VerifyResult

router = APIRouter(prefix="/api")

MAX_QUESTION_CHARS = 1000
_FIGURE_ID = re.compile(r"^p\d{1,3}_\d{1,3}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    bypass_cache: bool = False


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
    intent: str
    intent_rule: str | None
    slots: QuerySlots
    scope: ScopeDecision
    analysis: AnalysisOut
    rerank: RerankDecision
    retrieval: dict[str, Any]
    candidates: list[Candidate]
    context_blocks: list[ContextBlock]
    context_tokens: int
    context_budget: int
    dropped: list[str]
    calculations: list[CalculationResult]
    verify: VerifyResult
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
    citations: list[CitationOut]
    figures: list[CitationOut]
    warning: str | None = None
    debug: DebugOut


def _figure_id(image_path: str | None) -> str | None:
    if not image_path:
        return None
    return image_path.rsplit("/", 1)[-1].removesuffix(".png")


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
        citations=citations,
        figures=figures,
        warning=r.warning,
        debug=DebugOut(
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
            generation_attempts=r.generation_attempts,
            generator_role=r.generator_role,
            tokens_by_model=r.tokens_by_model,
            latency_ms_by_node=r.latency_ms_by_node,
            total_latency_ms=r.total_latency_ms,
            corpus_version=r.corpus_version,
        ),
    )


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question is empty")
    result = pipeline.ask(body.question, bypass_cache=body.bypass_cache)
    return to_response(result, pipeline.store)


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
