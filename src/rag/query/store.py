"""Serving-side handle on a built index: Chroma, BM25, chunks by id, sidecars, embedder.

Loaded once per process (FastAPI lifespan / CLI). No Docling, no PyTorch (P7).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from rag.core.bm25 import BM25Index
from rag.core.embeddings import Embedder, get_embedder
from rag.core.logging import get_logger
from rag.core.schema import Chunk, SentenceRecord
from rag.core.settings import Settings, get_settings
from rag.ingest.index import COLLECTION_NAME, Manifest, index_dir_for, load_manifest

log = get_logger(__name__)


class IndexStore:
    def __init__(self, index_dir: Path, *, figures_root: Path | None = None):
        import chromadb

        self.index_dir = Path(index_dir)
        self.figures_root = figures_root or self.index_dir  # figures/ and pages/ live here
        self.manifest: Manifest = load_manifest(self.index_dir)
        self.corpus_version = self.manifest.corpus_version
        self.embedder: Embedder = get_embedder(self.manifest.embedder_alias)
        self._client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        self.collection = self._client.get_collection(COLLECTION_NAME)
        self.bm25 = BM25Index.load(self.index_dir / "bm25.pkl")
        # Exact dense search (D-54): HNSW was measured to drop the true top hit on this small
        # corpus, so vectors are held in memory and scored by brute-force cosine (~1 ms).
        got = self.collection.get(include=["embeddings"])
        self._vec_ids: list[str] = list(got["ids"])
        self._vectors: np.ndarray = np.asarray(got["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        self._vectors = self._vectors / np.maximum(norms, 1e-12)

        self.chunks: dict[str, Chunk] = {}
        with (self.index_dir / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    c = Chunk.model_validate_json(line)
                    self.chunks[c.id] = c
        self.sentences: list[SentenceRecord] = []
        with (self.index_dir / "sentences.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self.sentences.append(SentenceRecord.model_validate_json(line))
        self.sentence_index: dict[str, int] = {
            s.sentence_id: i for i, s in enumerate(self.sentences)
        }
        self.sentence_emb: np.ndarray = np.load(self.index_dir / "sentence_emb.npy")

        self.row_facts: dict[tuple[str, str], list[Chunk]] = {}
        for c in self.chunks.values():
            m = c.metadata
            if m.modality == "row_fact" and m.line_item_norm:
                self.row_facts.setdefault((m.statement, m.line_item_norm), []).append(c)
        self.parts_by_parent: dict[str, list[Chunk]] = {}
        for c in self.chunks.values():
            if c.metadata.modality == "table" and c.metadata.parent_id:
                self.parts_by_parent.setdefault(c.metadata.parent_id, []).append(c)
        for parts in self.parts_by_parent.values():
            parts.sort(key=lambda c: c.metadata.table_part or 0)
        log.info(
            "Index loaded from %s: %d chunks, corpus_version=%s, embedder=%s",
            self.index_dir,
            len(self.chunks),
            self.corpus_version,
            self.manifest.embedder,
        )

    # -- lookups --------------------------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def table_parts(self, parent_id: str) -> list[Chunk]:
        return self.parts_by_parent.get(parent_id, [])

    def row_fact(self, statement: str, line_item_norm: str) -> Chunk | None:
        hits = self.row_facts.get((statement, line_item_norm))
        return hits[0] if hits else None

    def row_fact_line_items(self, statement: str) -> list[str]:
        return sorted(k[1] for k in self.row_facts if k[0] == statement)

    # -- search ----------------------------------------------------------------------

    def dense_search(
        self, query: str, k: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        """Exact cosine search; deterministic ordering (score desc, id asc)."""
        q = self.embedder.embed_query(query)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._vectors @ q
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), self._vec_ids[i]))
        out: list[tuple[str, float]] = []
        for i in order:
            cid = self._vec_ids[i]
            if where is not None and not self._matches(self.chunks[cid], where):
                continue
            out.append((cid, round(float(sims[i]), 6)))
            if len(out) >= k:
                break
        return out

    def dense_search_hnsw(
        self, query: str, k: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        """Chroma's approximate search, kept for comparison / inspection."""
        q = self.embedder.embed_query(query)
        res = self.collection.query(
            query_embeddings=[q.tolist()],
            n_results=min(k, len(self.chunks)),
            where=where,
            include=["distances"],
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists, strict=True)]

    def sparse_search(
        self, query: str, k: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        hits = self.bm25.search(query, k=k * 4 if where else k)
        if where:
            hits = [(i, s) for i, s in hits if self._matches(self.chunks[i], where)]
        return hits[:k]

    @staticmethod
    def _matches(chunk: Chunk, where: dict) -> bool:
        """Minimal evaluator for the `where` subset we use: equality, $in, $and."""
        meta = chunk.metadata.model_dump()
        if "$and" in where:
            return all(IndexStore._matches(chunk, w) for w in where["$and"])
        for key, cond in where.items():
            value = meta.get(key)
            if isinstance(cond, dict):
                if "$in" in cond and value not in cond["$in"]:
                    return False
                if "$eq" in cond and value != cond["$eq"]:
                    return False
            elif value != cond:
                return False
        return True

    def image_path(self, relative: str | None) -> Path | None:
        if not relative:
            return None
        p = self.figures_root / relative
        return p if p.exists() else None


@lru_cache(maxsize=2)
def get_store(index_dir: str | None = None) -> IndexStore:
    settings: Settings = get_settings()
    if index_dir is None:
        from rag.core.config import load_thresholds_config

        alias = load_thresholds_config(settings=settings).retrieval.embedder
        index_dir = str(index_dir_for(settings.data_dir, alias))
    return IndexStore(Path(index_dir), figures_root=settings.data_dir / "index")


def _sanity(index_dir: Path) -> dict:
    """Small helper for /readyz: does the index look complete?"""
    required = ["manifest.json", "bm25.pkl", "chunks.jsonl", "sentences.jsonl", "sentence_emb.npy"]
    missing = [f for f in required if not (index_dir / f).exists()]
    out = {"index_dir": str(index_dir), "missing": missing}
    if not missing:
        out["manifest"] = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return out
