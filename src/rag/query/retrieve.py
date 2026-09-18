"""Hybrid retrieval (architecture §4.5): dense top-k + BM25 top-k → RRF → top-n, small-to-big.

Phase 3 runs one retrieval query (the question itself). Phase 4 adds expansion (several
queries fused together), slot-derived metadata filters with an unfiltered retry, and reranking.
RRF uses ranks only, so BM25's unbounded scores and cosine similarities need no normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rag.core.schema import Chunk
from rag.query.store import IndexStore

EXPAND_INTENTS = {"COMPARISON_TREND", "EXPLANATORY", "VISUAL", "CROSS_SECTION"}


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(id) = Σ over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return scores


class Candidate(BaseModel):
    chunk_id: str
    modality: str
    page: int
    section: str
    subsection: str
    breadcrumb: str
    rrf: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    source: str = Field(description="dense | bm25 | both | expanded")
    expanded_from: str | None = None
    token_count: int = 0


class RetrievalResult(BaseModel):
    queries: list[str]
    where: dict | None = None
    filtered_retry: bool = False
    dense_top: list[str] = Field(default_factory=list)
    bm25_top: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(description="fused top-n plus small-to-big expansions")

    @property
    def top_ids(self) -> list[str]:
        return [c.chunk_id for c in self.candidates if c.source != "expanded"]


@dataclass
class RetrievalSettings:
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    final_k: int = 8
    min_filtered_results: int = 3
    expand_parents: bool = True
    extra: dict = field(default_factory=dict)


def _candidate(chunk: Chunk, **kw) -> Candidate:
    m = chunk.metadata
    return Candidate(
        chunk_id=chunk.id,
        modality=m.modality,
        page=m.page,
        section=m.section,
        subsection=m.subsection,
        breadcrumb=m.breadcrumb,
        token_count=m.token_count,
        **kw,
    )


def hybrid_retrieve(
    store: IndexStore,
    queries: list[str],
    *,
    settings: RetrievalSettings | None = None,
    where: dict | None = None,
    intent: str = "POINT_LOOKUP",
) -> RetrievalResult:
    s = settings or RetrievalSettings()
    dense_lists: list[list[tuple[str, float]]] = []
    sparse_lists: list[list[tuple[str, float]]] = []
    filtered_retry = False

    def run(where_: dict | None) -> None:
        dense_lists.clear()
        sparse_lists.clear()
        for q in queries:
            dense_lists.append(store.dense_search(q, s.dense_top_k, where=where_))
            sparse_lists.append(store.sparse_search(q, s.sparse_top_k, where=where_))

    run(where)
    if where is not None:
        n_unique = len({i for lst in dense_lists + sparse_lists for i, _ in lst})
        if n_unique < s.min_filtered_results:
            filtered_retry = True
            run(None)

    rankings = [[i for i, _ in lst] for lst in dense_lists + sparse_lists]
    fused = rrf_fuse(rankings, k=s.rrf_k)
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: s.final_k]

    dense_best: dict[str, tuple[int, float]] = {}
    for lst in dense_lists:
        for rank, (i, score) in enumerate(lst, start=1):
            if i not in dense_best or rank < dense_best[i][0]:
                dense_best[i] = (rank, score)
    sparse_best: dict[str, tuple[int, float]] = {}
    for lst in sparse_lists:
        for rank, (i, score) in enumerate(lst, start=1):
            if i not in sparse_best or rank < sparse_best[i][0]:
                sparse_best[i] = (rank, score)

    candidates: list[Candidate] = []
    for chunk_id, score in order:
        chunk = store.get(chunk_id)
        if chunk is None:
            continue
        d = dense_best.get(chunk_id)
        b = sparse_best.get(chunk_id)
        source = "both" if d and b else ("dense" if d else "bm25")
        candidates.append(
            _candidate(
                chunk,
                rrf=round(score, 6),
                dense_rank=d[0] if d else None,
                dense_score=round(d[1], 4) if d else None,
                bm25_rank=b[0] if b else None,
                bm25_score=round(b[1], 3) if b else None,
                source=source,
            )
        )

    # Small-to-big: row facts pull their parent table; figures pull their companion table.
    if s.expand_parents:
        seen = {c.chunk_id for c in candidates}
        extra: list[Candidate] = []
        for c in list(candidates):
            chunk = store.get(c.chunk_id)
            if chunk is None:
                continue
            parent_ids: list[str] = []
            if (
                chunk.metadata.modality == "row_fact"
                and intent in EXPAND_INTENTS
                and chunk.metadata.parent_id
            ):
                parent_ids.append(chunk.metadata.parent_id)
            if chunk.metadata.modality == "figure" and chunk.metadata.companion_table_id:
                parent_ids.append(chunk.metadata.companion_table_id)
            for pid in parent_ids:
                for part in store.table_parts(pid) or ([store.get(pid)] if store.get(pid) else []):
                    if part.id in seen:
                        continue
                    seen.add(part.id)
                    extra.append(
                        _candidate(part, rrf=0.0, source="expanded", expanded_from=c.chunk_id)
                    )
        candidates.extend(extra)

    return RetrievalResult(
        queries=queries,
        where=where,
        filtered_retry=filtered_retry,
        dense_top=[i for i, _ in dense_lists[0]][:10] if dense_lists else [],
        bm25_top=[i for i, _ in sparse_lists[0]][:10] if sparse_lists else [],
        candidates=candidates,
    )
