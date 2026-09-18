"""Rerank gate and cross-encoder reranker (architecture §4.6, D-24, D-34).

The reranker is a local ONNX cross-encoder (`BAAI/bge-reranker-base` via fastembed; zero API
cost) scoring (original question, chunk document) pairs over the fused top-30. It is skipped
when the fused ranking is already decisive:

  S1  numeric intent and every required metric has an exact `line_item_norm` match in the
      top 5 fused results (the row facts are already there);
  S2  a single retrieval query was used and the fused set is no larger than `final_k`;
  S3  the top-1 fused result is rank 1 in both the dense and the BM25 list and its RRF score
      beats the runner-up by ≥ `margin_skip_ratio`.

Otherwise: rerank, keep the top `final_k`, drop candidates whose logit is below
`drop_floor_logit` but always keep ≥ `min_keep`, then re-apply small-to-big expansion.
Reranking always uses the *original* query so expansion improves recall without diluting
precision.
"""

from __future__ import annotations

import time
from functools import lru_cache

from pydantic import BaseModel, Field

from rag.core.config import RerankThresholds
from rag.core.logging import get_logger
from rag.query.retrieve import Candidate, RetrievalResult, expand_small_to_big
from rag.query.store import IndexStore

log = get_logger(__name__)

NUMERIC_INTENTS = {"POINT_LOOKUP", "COMPUTATION"}


class RerankDecision(BaseModel):
    applied: bool
    skip_reason: str | None = None
    gate: str | None = Field(default=None, description="S1 | S2 | S3 | disabled | empty")
    model: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    dropped: list[str] = Field(default_factory=list)
    kept: int = 0
    latency_ms: int = 0
    required_metrics: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------- gate


def rerank_gate(
    retrieval: RetrievalResult,
    *,
    intent: str,
    required_metrics: list[str],
    store: IndexStore | None,
    final_k: int,
    margin_skip_ratio: float,
) -> tuple[str, str] | None:
    """Returns (gate id, reason) when reranking should be skipped, else None."""
    pool = retrieval.pool
    if not pool:
        return "empty", "no fused candidates"
    if intent in NUMERIC_INTENTS and required_metrics and store is not None:
        top5 = [store.get(c.chunk_id) for c in pool[:5]]
        present = {c.metadata.line_item_norm for c in top5 if c is not None}
        missing = [m for m in required_metrics if m not in present]
        if not missing:
            return "S1", f"all required metrics {required_metrics} are row facts in the top 5"
    if len(retrieval.queries) == 1 and retrieval.fused_count <= final_k:
        return "S2", f"single query and only {retrieval.fused_count} fused candidates (≤ {final_k})"
    top = pool[0]
    if top.dense_rank == 1 and top.bm25_rank == 1:
        runner_up = pool[1].rrf if len(pool) > 1 else 0.0
        if runner_up == 0.0 or top.rrf >= runner_up * (1.0 + margin_skip_ratio):
            margin = (top.rrf / runner_up - 1.0) * 100 if runner_up else float("inf")
            return "S3", (
                f"{top.chunk_id} is rank 1 in both dense and BM25 and leads by "
                f"{margin:.0f}% (≥ {margin_skip_ratio:.0%})"
            )
    return None


# ------------------------------------------------------------------------------ reranker


class Reranker:
    """Lazy wrapper around fastembed's cross-encoder; the model loads on first use."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            t0 = time.perf_counter()
            self._model = TextCrossEncoder(self.model_id)
            log.info("reranker %s loaded in %.1fs", self.model_id, time.perf_counter() - t0)
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        return [float(s) for s in model.rerank(query, documents)]


@lru_cache(maxsize=2)
def get_reranker(model_id: str) -> Reranker:
    return Reranker(model_id)


def apply_rerank(
    store: IndexStore,
    retrieval: RetrievalResult,
    question: str,
    *,
    scores: list[float],
    thresholds: RerankThresholds,
    final_k: int,
    intent: str,
    model_id: str | None = None,
) -> tuple[RetrievalResult, RerankDecision]:
    """Re-order the pool by cross-encoder score and rebuild `candidates` (pure; `scores` are
    aligned with `retrieval.pool`, so tests can pass synthetic logits)."""
    pool = retrieval.pool
    if len(scores) != len(pool):
        raise ValueError(f"{len(scores)} scores for {len(pool)} pool candidates")
    scored: list[Candidate] = []
    for c, s in zip(pool, scores, strict=True):
        scored.append(c.model_copy(update={"rerank_score": round(s, 4)}))
    scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
    for rank, c in enumerate(scored, start=1):
        c.rerank_rank = rank

    kept: list[Candidate] = []
    dropped: list[str] = []
    for c in scored:
        if len(kept) >= final_k:
            break
        if (c.rerank_score or 0.0) < thresholds.drop_floor_logit and len(
            kept
        ) >= thresholds.min_keep:
            dropped.append(c.chunk_id)
            continue
        kept.append(c)
    candidates = [c.model_copy() for c in kept]
    candidates.extend(expand_small_to_big(store, candidates, intent))

    decision = RerankDecision(
        applied=True,
        model=model_id or thresholds.model,
        scores={c.chunk_id: c.rerank_score or 0.0 for c in scored},
        dropped=dropped,
        kept=len(kept),
    )
    updated = retrieval.model_copy(
        update={
            "pool": scored,
            "candidates": candidates,
            "rerank_applied": True,
            "rerank_skip_reason": None,
            "rerank_dropped": dropped,
        }
    )
    return updated, decision


def rerank_node(
    store: IndexStore,
    retrieval: RetrievalResult,
    question: str,
    *,
    intent: str,
    required_metrics: list[str],
    thresholds: RerankThresholds,
    final_k: int,
    reranker: Reranker | None = None,
    force: bool = False,
) -> tuple[RetrievalResult, RerankDecision]:
    """Gate, then rerank with the configured model. `force=True` bypasses the gate (ablation)."""
    t0 = time.perf_counter()
    if not thresholds.enabled and not force:
        return retrieval.model_copy(update={"rerank_skip_reason": "disabled"}), RerankDecision(
            applied=False, skip_reason="rerank disabled in thresholds.yaml", gate="disabled"
        )
    skip = (
        None
        if force
        else rerank_gate(
            retrieval,
            intent=intent,
            required_metrics=required_metrics,
            store=store,
            final_k=final_k,
            margin_skip_ratio=thresholds.margin_skip_ratio,
        )
    )
    if skip is not None:
        gate, reason = skip
        return retrieval.model_copy(
            update={"rerank_skip_reason": f"{gate}: {reason}"}
        ), RerankDecision(
            applied=False,
            skip_reason=reason,
            gate=gate,
            required_metrics=required_metrics,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    if not retrieval.pool:
        return retrieval, RerankDecision(applied=False, skip_reason="empty pool", gate="empty")
    rr = reranker or get_reranker(thresholds.model)
    docs = [
        (store.get(c.chunk_id).document if store.get(c.chunk_id) else "") for c in retrieval.pool
    ]
    scores = rr.score(question, docs)
    updated, decision = apply_rerank(
        store,
        retrieval,
        question,
        scores=scores,
        thresholds=thresholds,
        final_k=final_k,
        intent=intent,
        model_id=rr.model_id,
    )
    decision.required_metrics = required_metrics
    decision.latency_ms = int((time.perf_counter() - t0) * 1000)
    return updated, decision
