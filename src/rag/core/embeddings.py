"""Local embeddings via fastembed ONNX (architecture §3.6, §7.5, D-34).

The same runtime embeds documents at ingestion and queries / cache entries at serving, so
vectors are numerically consistent. Documents are embedded without the bge query instruction;
queries with it (`query_embed`). Vectors are L2-normalised, so cosine == dot product.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

EMBEDDERS: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
DEFAULT_EMBEDDER = "bge-small"


def resolve_embedder(name: str) -> str:
    """Accept a short alias ('bge-small') or a full fastembed model id."""
    return EMBEDDERS.get(name, name)


class Embedder:
    def __init__(self, name: str = DEFAULT_EMBEDDER, *, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.alias = name
        self.model_id = resolve_embedder(name)
        self.batch_size = batch_size
        self._model = TextEmbedding(self.model_id)
        self.dim = int(next(iter(self._model.embed(["probe"]))).shape[0])

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(texts, batch_size=self.batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)

    def embed_queries(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(list(self._model.query_embed(texts)), dtype=np.float32)


@lru_cache(maxsize=2)
def get_embedder(name: str = DEFAULT_EMBEDDER) -> Embedder:
    """Process-wide cached embedder (model load takes seconds)."""
    return Embedder(name)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of `a` and rows of `b` (inputs need not be normalised)."""
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_n @ b_n.T
