"""Embedding and indexing (architecture §3.6): Chroma `report_chunks` + BM25 + sidecars + manifest.

    .venv-ingest/Scripts/python -m rag.ingest.run --stage index [--embedder bge-small|bge-base]

The default embedder writes `data/index/`; `bge-base` writes `data/index_bge_base/` so the
Phase 3 embedder gate can compare both. Figure crops and page thumbnails stay in `data/index/`
(metadata paths are relative to it). `corpus_version` = sha256(pdf hash + ingestion config)[:12]
and is stamped into every chunk's metadata and the manifest (FR-2.3).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from rag.core.bm25 import TOKENIZER_VERSION, BM25Index
from rag.core.embeddings import Embedder, resolve_embedder
from rag.core.logging import get_logger
from rag.core.schema import Chunk, SentenceRecord

log = get_logger(__name__)

COLLECTION_NAME = "report_chunks"
CHROMA_BATCH = 500


class Manifest(BaseModel):
    corpus_version: str
    created_at: str
    pdf_sha256: str
    ingestion_config: dict[str, Any]
    ingestion_config_hash: str
    embedder: str
    embedder_alias: str
    embedding_dim: int
    tokenizer_version: str
    chunk_counts: dict[str, int]
    chunks_total: int
    sentences_total: int
    enrichment_models: dict[str, str] = Field(default_factory=dict)
    docling_version: str | None = None


def compute_corpus_version(pdf_sha256: str, ingestion_config: dict[str, Any]) -> str:
    """Deterministic: same PDF + same config → same version; any config change → new version."""
    material = pdf_sha256 + json.dumps(ingestion_config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def config_hash(ingestion_config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(ingestion_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]


def write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json(exclude_none=True) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def read_sentences(path: Path) -> list[SentenceRecord]:
    with path.open("r", encoding="utf-8") as fh:
        return [SentenceRecord.model_validate_json(line) for line in fh if line.strip()]


def build_index(
    chunks: list[Chunk],
    sentences: list[SentenceRecord],
    *,
    index_dir: Path,
    embedder: Embedder,
    pdf_sha256: str,
    ingestion_config: dict[str, Any],
    enrichment_models: dict[str, str],
    docling_version: str | None,
) -> Manifest:
    import chromadb

    config = dict(ingestion_config)
    config["embedder"] = embedder.model_id
    config["tokenizer_version"] = TOKENIZER_VERSION
    corpus_version = compute_corpus_version(pdf_sha256, config)
    for c in chunks:
        c.metadata.corpus_version = corpus_version

    index_dir.mkdir(parents=True, exist_ok=True)

    # --- dense: Chroma (rebuilt from scratch for idempotency)
    chroma_dir = index_dir / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    log.info("Embedding %d chunks with %s …", len(chunks), embedder.model_id)
    vectors = embedder.embed_documents([c.document for c in chunks])
    for start in range(0, len(chunks), CHROMA_BATCH):
        batch = chunks[start : start + CHROMA_BATCH]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.document for c in batch],
            embeddings=vectors[start : start + len(batch)].tolist(),
            metadatas=[c.chroma_metadata() for c in batch],
        )
    log.info("Chroma collection %s: %d items", COLLECTION_NAME, collection.count())

    # --- sparse: BM25
    bm25 = BM25Index.build([c.id for c in chunks], [c.document for c in chunks])
    bm25.save(index_dir / "bm25.pkl")

    # --- sidecars (sentence embeddings are embedder-specific)
    write_jsonl(index_dir / "chunks.jsonl", chunks)
    write_jsonl(index_dir / "sentences.jsonl", sentences)
    log.info("Embedding %d sentences for the sidecar …", len(sentences))
    sent_vec = embedder.embed_documents([s.text for s in sentences])
    np.save(index_dir / "sentence_emb.npy", sent_vec.astype(np.float32))

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.metadata.modality] = counts.get(c.metadata.modality, 0) + 1
    manifest = Manifest(
        corpus_version=corpus_version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pdf_sha256=pdf_sha256,
        ingestion_config=config,
        ingestion_config_hash=config_hash(config),
        embedder=embedder.model_id,
        embedder_alias=embedder.alias,
        embedding_dim=embedder.dim,
        tokenizer_version=TOKENIZER_VERSION,
        chunk_counts=dict(sorted(counts.items())),
        chunks_total=len(chunks),
        sentences_total=len(sentences),
        enrichment_models=enrichment_models,
        docling_version=docling_version,
    )
    (index_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info("Manifest written: corpus_version=%s counts=%s", corpus_version, counts)
    return manifest


def load_manifest(index_dir: Path) -> Manifest:
    return Manifest.model_validate_json((index_dir / "manifest.json").read_text(encoding="utf-8"))


def index_dir_for(data_dir: Path, embedder_alias: str) -> Path:
    alias = resolve_embedder(embedder_alias)
    if alias == resolve_embedder("bge-small"):
        return data_dir / "index"
    safe = embedder_alias.replace("/", "_").replace("-", "_")
    return data_dir / f"index_{safe}"
