"""Compression features (architecture §6.3) — all cheap, no model calls (K1).

Query level: total tokens, budget ratio, intent, chunk count. Chunk level: modality, tokens,
relevance score (rerank logit or RRF), relevance density from the sentence sidecar embedded at
ingestion, max duplicate similarity to a higher-ranked chunk (chunk vectors already in the
store), numeric density, table rows. Everything is logged with the decision so Stage C (Phase
7) can be trained from `data/logs/compression_decisions.jsonl`.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from rag.core.schema import Chunk
from rag.core.tokens import count_tokens
from rag.query.retrieve import Candidate
from rag.query.store import IndexStore

_NUMBER_TOKEN = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\w])")
_WORD = re.compile(r"\S+")

NARRATIVE_MODALITIES = {"text"}
PROTECTED_MODALITIES = {"row_fact", "figure"}


class SentenceView(BaseModel):
    """A chunk's sentences with their relevance to the query set (from the sidecar)."""

    sentence_ids: list[str] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)
    starts: list[int] = Field(default_factory=list)
    ends: list[int] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list, description="max cosine to any query")


class ChunkFeatures(BaseModel):
    chunk_id: str
    rank: int
    modality: str
    chunk_tokens: int
    rerank_score: float = Field(description="cross-encoder logit, or RRF score when not reranked")
    score_kind: str = Field(default="rrf", description="rerank | rrf | expanded")
    relevance_density: float = Field(ge=0, le=1)
    max_dup_sim: float = Field(ge=-1, le=1)
    dup_of: str | None = None
    numeric_density: float = Field(ge=0, le=1)
    table_rows: int | None = None
    n_sentences: int = 0


class QueryFeatures(BaseModel):
    intent: str
    n_chunks: int
    total_tokens: int
    budget_ratio: float
    context_budget: int
    queries: list[str]


def numeric_density(text: str) -> float:
    words = _WORD.findall(text)
    if not words:
        return 0.0
    return min(1.0, len(_NUMBER_TOKEN.findall(text)) / len(words))


class FeatureExtractor:
    """Computes features for one retrieval; holds the query embeddings for reuse by the
    compressors (EXTRACT_LIGHT needs the same sentence scores)."""

    def __init__(self, store: IndexStore, *, sentence_tau: float):
        self.store = store
        self.sentence_tau = sentence_tau
        self._chunk_pos = {cid: i for i, cid in enumerate(store._vec_ids)}
        self._sentences_by_chunk: dict[str, list[int]] = {}
        for i, s in enumerate(store.sentences):
            self._sentences_by_chunk.setdefault(s.chunk_id, []).append(i)

    def query_matrix(self, queries: list[str]) -> np.ndarray:
        q = self.store.embedder.embed_queries([t for t in queries if t.strip()])
        if q.size == 0:
            return q
        return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    def sentence_view(self, chunk_id: str, qmat: np.ndarray) -> SentenceView:
        idx = self._sentences_by_chunk.get(chunk_id, [])
        if not idx:
            return SentenceView()
        emb = self.store.sentence_emb[idx]
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        scores = (emb @ qmat.T).max(axis=1) if qmat.size else np.zeros(len(idx))
        recs = [self.store.sentences[i] for i in idx]
        return SentenceView(
            sentence_ids=[r.sentence_id for r in recs],
            texts=[r.text for r in recs],
            starts=[r.start for r in recs],
            ends=[r.end for r in recs],
            scores=[round(float(s), 4) for s in scores],
        )

    def chunk_vector(self, chunk_id: str) -> np.ndarray | None:
        pos = self._chunk_pos.get(chunk_id)
        return None if pos is None else self.store._vectors[pos]

    def extract(
        self,
        candidates: list[Candidate],
        queries: list[str],
        *,
        intent: str,
        context_budget: int,
        body_of,  # noqa: ANN001 - Callable[[Chunk], str]; avoids importing assemble here
    ) -> tuple[QueryFeatures, list[ChunkFeatures], dict[str, SentenceView]]:
        qmat = self.query_matrix(queries)
        feats: list[ChunkFeatures] = []
        views: dict[str, SentenceView] = {}
        vectors: list[tuple[str, np.ndarray]] = []
        total = 0
        for rank, cand in enumerate(candidates, start=1):
            chunk: Chunk | None = self.store.get(cand.chunk_id)
            if chunk is None:
                continue
            body = body_of(chunk)
            tokens = count_tokens(body)
            total += tokens
            # duplicate similarity against every higher-ranked chunk
            vec = self.chunk_vector(chunk.id)
            max_sim, dup_of = -1.0, None
            if vec is not None:
                for other_id, other_vec in vectors:
                    sim = float(vec @ other_vec)
                    if sim > max_sim:
                        max_sim, dup_of = sim, other_id
                vectors.append((chunk.id, vec))
            density = 1.0
            n_sent = 0
            if chunk.metadata.modality in NARRATIVE_MODALITIES:
                view = self.sentence_view(chunk.id, qmat)
                views[chunk.id] = view
                n_sent = len(view.scores)
                if n_sent:
                    density = sum(1 for s in view.scores if s >= self.sentence_tau) / n_sent
            if cand.rerank_score is not None:
                score, kind = cand.rerank_score, "rerank"
            elif cand.source == "expanded":
                score, kind = 0.0, "expanded"
            else:
                score, kind = cand.rrf, "rrf"
            feats.append(
                ChunkFeatures(
                    chunk_id=chunk.id,
                    rank=rank,
                    modality=chunk.metadata.modality,
                    chunk_tokens=tokens,
                    rerank_score=round(float(score), 4),
                    score_kind=kind,
                    relevance_density=round(density, 3),
                    max_dup_sim=round(max_sim, 4),
                    dup_of=dup_of,
                    numeric_density=round(numeric_density(body), 3),
                    table_rows=chunk.metadata.n_rows,
                    n_sentences=n_sent,
                )
            )
        qf = QueryFeatures(
            intent=intent,
            n_chunks=len(feats),
            total_tokens=total,
            budget_ratio=round(total / context_budget, 3) if context_budget else 0.0,
            context_budget=context_budget,
            queries=list(queries),
        )
        return qf, feats, views
