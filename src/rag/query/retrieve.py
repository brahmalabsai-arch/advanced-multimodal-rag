"""Hybrid retrieval (architecture §4.5): dense top-k + BM25 top-k → RRF → top-n, small-to-big.

Phase 4: several retrieval queries are fused together (the original question plus glossary /
decomposition / paraphrase / HyDE expansions from `analyze.py`); each query may carry its own
metadata pre-filter (`where`), with an unfiltered retry for that query when the filtered search
returns fewer than `min_filtered_results` ids — filters are hints, not walls. The fused list is
kept to `pool_k` (30) so the reranker (`rerank.py`) can re-order it; `candidates` holds the
top-`final_k` plus small-to-big expansions. RRF uses ranks only, so BM25's unbounded scores and
cosine similarities need no normalisation.
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


class QuerySpec(BaseModel):
    """One retrieval query; `analyze.RetrievalQuery` is accepted wherever this is."""

    kind: str = "original"
    text: str
    where: dict | None = None


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
    hit_by: list[str] = Field(default_factory=list, description="query kinds that retrieved it")
    rerank_score: float | None = None
    rerank_rank: int | None = None
    token_count: int = 0


class RetrievalResult(BaseModel):
    queries: list[str]
    query_kinds: list[str] = Field(default_factory=list)
    filters: list[dict | None] = Field(default_factory=list)
    where: dict | None = None
    filtered_retry: bool = False
    filtered_retries: list[int] = Field(
        default_factory=list, description="query indexes whose filter was dropped"
    )
    dense_top: list[str] = Field(default_factory=list)
    bm25_top: list[str] = Field(default_factory=list)
    fused_count: int = 0
    pool: list[Candidate] = Field(
        default_factory=list, description="fused top-pool_k before reranking (no expansions)"
    )
    candidates: list[Candidate] = Field(description="fused top-n plus small-to-big expansions")
    rerank_applied: bool = False
    rerank_skip_reason: str | None = None
    rerank_dropped: list[str] = Field(default_factory=list)

    @property
    def top_ids(self) -> list[str]:
        return [c.chunk_id for c in self.candidates if c.source != "expanded"]


@dataclass
class RetrievalSettings:
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    final_k: int = 8
    pool_k: int = 30
    min_filtered_results: int = 3
    expand_parents: bool = True
    dense: bool = True
    sparse: bool = True
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


def _as_specs(queries: list, where: dict | None) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    for q in queries:
        if isinstance(q, str):
            specs.append(QuerySpec(text=q, where=where))
        else:
            specs.append(QuerySpec(kind=q.kind, text=q.text, where=q.where or where))
    return specs


def expand_small_to_big(
    store: IndexStore, candidates: list[Candidate], intent: str
) -> list[Candidate]:
    """Row facts pull their parent table (comparison / explanatory intents); figures pull their
    companion table. Returns the extra candidates (source = "expanded")."""
    seen = {c.chunk_id for c in candidates}
    extra: list[Candidate] = []
    for c in candidates:
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
            parts = store.table_parts(pid) or ([store.get(pid)] if store.get(pid) else [])
            for part in parts:
                if part.id in seen:
                    continue
                seen.add(part.id)
                extra.append(_candidate(part, rrf=0.0, source="expanded", expanded_from=c.chunk_id))
    return extra


def hybrid_retrieve(
    store: IndexStore,
    queries: list,
    *,
    settings: RetrievalSettings | None = None,
    where: dict | None = None,
    intent: str = "POINT_LOOKUP",
) -> RetrievalResult:
    s = settings or RetrievalSettings()
    specs = _as_specs(queries, where)
    dense_lists: list[list[tuple[str, float]]] = []
    sparse_lists: list[list[tuple[str, float]]] = []
    list_kinds: list[str] = []
    retried: list[int] = []

    def run(text: str, where_: dict | None) -> tuple[list, list]:
        d = store.dense_search(text, s.dense_top_k, where=where_) if s.dense else []
        b = store.sparse_search(text, s.sparse_top_k, where=where_) if s.sparse else []
        return d, b

    for idx, spec in enumerate(specs):
        d, b = run(spec.text, spec.where)
        if spec.where is not None:
            n_unique = len({i for i, _ in d} | {i for i, _ in b})
            if n_unique < s.min_filtered_results:
                retried.append(idx)
                d, b = run(spec.text, None)
        if s.dense:
            dense_lists.append(d)
            list_kinds.append(spec.kind)
        if s.sparse:
            sparse_lists.append(b)
            list_kinds.append(spec.kind)

    all_lists = dense_lists + sparse_lists
    kinds_for_lists = list_kinds[: len(dense_lists)] + list_kinds[len(dense_lists) :]
    rankings = [[i for i, _ in lst] for lst in all_lists]
    fused = rrf_fuse(rankings, k=s.rrf_k)
    order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

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
    hit_by: dict[str, list[str]] = {}
    for lst, kind in zip(all_lists, kinds_for_lists, strict=True):
        for i, _ in lst:
            kinds = hit_by.setdefault(i, [])
            if kind not in kinds:
                kinds.append(kind)

    pool: list[Candidate] = []
    for chunk_id, score in order[: max(s.pool_k, s.final_k)]:
        chunk = store.get(chunk_id)
        if chunk is None:
            continue
        d = dense_best.get(chunk_id)
        b = sparse_best.get(chunk_id)
        source = "both" if d and b else ("dense" if d else "bm25")
        pool.append(
            _candidate(
                chunk,
                rrf=round(score, 6),
                dense_rank=d[0] if d else None,
                dense_score=round(d[1], 4) if d else None,
                bm25_rank=b[0] if b else None,
                bm25_score=round(b[1], 3) if b else None,
                source=source,
                hit_by=hit_by.get(chunk_id, []),
            )
        )

    candidates = [c.model_copy() for c in pool[: s.final_k]]
    if s.expand_parents:
        candidates.extend(expand_small_to_big(store, candidates, intent))

    first_dense = dense_lists[0] if dense_lists else []
    first_sparse = sparse_lists[0] if sparse_lists else []
    return RetrievalResult(
        queries=[q.text for q in specs],
        query_kinds=[q.kind for q in specs],
        filters=[q.where for q in specs],
        where=where,
        filtered_retry=bool(retried),
        filtered_retries=retried,
        dense_top=[i for i, _ in first_dense][:10],
        bm25_top=[i for i, _ in first_sparse][:10],
        fused_count=len(fused),
        pool=pool,
        candidates=candidates,
    )
