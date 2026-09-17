"""Cached LLM enrichment at ingestion (architecture §7.3, D-37).

Every enrichment result is stored at
`data/parsed/enrichment/{sha256(content)}__{model_id}.json`, so re-running ingestion makes zero
API calls (NFR-8) and a withdrawn model keeps the existing index usable. The model id in the key
means switching enrichment models produces fresh results and, via the ingestion config hash, a
new `corpus_version`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from rag.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "--").replace(":", "-")


class Enricher:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def cache_path(self, key_material: bytes, model_id: str) -> Path:
        digest = hashlib.sha256(key_material).hexdigest()
        return self.cache_dir / f"{digest}__{safe_model_id(model_id)}.json"

    def cached(
        self,
        key_material: bytes,
        model_id: str,
        schema: type[T],
        compute: Callable[[], T],
        *,
        label: str = "",
    ) -> T:
        """Return the cached result for (content, model) or compute, store and return it."""
        path = self.cache_path(key_material, model_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result = schema.model_validate(payload["result"])
                self.hits += 1
                return result
            except (ValueError, KeyError):
                log.warning("Corrupt enrichment cache entry %s; recomputing", path.name)
        self.misses += 1
        result = compute()
        payload = {"model_id": model_id, "label": label, "result": result.model_dump(mode="json")}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return result

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
