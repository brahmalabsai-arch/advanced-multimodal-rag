"""Version keys that invalidate cache entries independently of TTL (architecture §5.7, D-06).

Every L2 record stores the five keys; lookups filter on all of them, so an entry written under
another corpus, prompt, model or retrieval configuration can never be served. The keys are:

    corpus_version         index manifest (`data/index/<embedder>/manifest.json`)
    prompt_version         `PROMPT_VERSION` + hash of the answer system prompt
    generator_model        "<large model>+<vision model>" of the active profile (VISUAL answers
                           are produced by the vision role, so both ids matter)
    retrieval_config_hash  parsed `thresholds.yaml` minus the `cache:` section (retrieval,
                           expansion, rerank incl. the reranker model, compression) plus
                           `cache.l2.embed_text`, the embedder alias, `glossary.yaml` and
                           `fiscal_calendar.yaml`
    calculator_version     parsed `formulas.yaml`

Hashes are taken over the *parsed* YAML (canonical JSON), so a comment edit does not flush the
cache while any value change does (walkthrough step 7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag.core.config import ModelsConfig, ThresholdsConfig, load_yaml
from rag.core.settings import Settings, get_settings

VERSION_FIELDS = (
    "corpus_version",
    "prompt_version",
    "generator_model",
    "retrieval_config_hash",
    "calculator_version",
)


def _hash(obj: Any, n: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


class VersionKeys(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    prompt_version: str
    generator_model: str
    retrieval_config_hash: str
    calculator_version: str

    def joined(self) -> str:
        """Stable string for the L1 key (`QuerySlots.l1_key(version_keys)`)."""
        return "|".join(getattr(self, f) for f in VERSION_FIELDS)

    def where_clauses(self) -> list[dict[str, str]]:
        return [{f: getattr(self, f)} for f in VERSION_FIELDS]

    def matches(self, metadata: dict[str, Any]) -> bool:
        return all(metadata.get(f) == getattr(self, f) for f in VERSION_FIELDS)


def prompt_version() -> str:
    from rag.query.generate import PROMPT_VERSION, SYSTEM_PROMPT

    return f"{PROMPT_VERSION}-{_hash(SYSTEM_PROMPT, 8)}"


def generator_model(models: ModelsConfig) -> str:
    profile = models.active()
    return f"{profile.large.model}+{profile.vision.model}"


def retrieval_config_hash(
    thresholds: ThresholdsConfig,
    *,
    embedder_alias: str,
    config_dir: Path,
) -> str:
    parts = {
        "thresholds": thresholds.model_dump(mode="json", exclude={"cache"}),
        # the only cache setting that changes what an entry *is*: which text was embedded
        "cache_embed_text": thresholds.cache.l2.embed_text,
        "embedder": embedder_alias,
        "glossary": load_yaml(config_dir / "glossary.yaml"),
        "fiscal_calendar": load_yaml(config_dir / "fiscal_calendar.yaml"),
    }
    return _hash(parts)


def calculator_version(config_dir: Path) -> str:
    return _hash(load_yaml(config_dir / "formulas.yaml"))


def compute_version_keys(
    *,
    corpus_version: str,
    embedder_alias: str,
    thresholds: ThresholdsConfig,
    models: ModelsConfig,
    settings: Settings | None = None,
) -> VersionKeys:
    settings = settings or get_settings()
    return VersionKeys(
        corpus_version=corpus_version,
        prompt_version=prompt_version(),
        generator_model=generator_model(models),
        retrieval_config_hash=retrieval_config_hash(
            thresholds, embedder_alias=embedder_alias, config_dir=settings.config_dir
        ),
        calculator_version=calculator_version(settings.config_dir),
    )
