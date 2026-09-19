"""Shared record types for both cache tiers (architecture §5.11)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult
from rag.query.assemble import ContextBlock
from rag.query.generate import Answer
from rag.query.slots import QuerySlots
from rag.query.verify import VerifyResult

CacheTier = Literal["L1", "L2", "MISS", "bypassed"]

SLOT_FIELDS = (
    "entity",
    "periods_key",
    "metrics_key",
    "topic_key",
    "direction",
    "negation",
    "ask",
    "aggregation_key",
)

_NEGATION = re.compile(
    r"\b(not|never|no|none|neither|nor|without|isnt|arent|wasnt|werent|doesnt|dont|didnt|"
    r"hasnt|havent|hadnt|cannot|cant|wont|wouldnt)\b"
)


def has_negation(normalized_text: str) -> bool:
    return bool(_NEGATION.search(normalized_text or ""))


def slot_keys(slots: QuerySlots, *, default_entity: str) -> dict[str, str]:
    """Slot-guard fields (§5.8). The corpus has one entity, so a question that does not name
    it ("Total assets at the end of fiscal 2026?") is keyed to the default entity rather than
    to a separate `none` bucket — otherwise the paraphrase in walkthrough step 4 could never
    hit. `metrics_key` covers metrics and formula ids (both are canonical glossary ids).
    `negation` and `ask` were added in Phase 6 calibration: "did inventories grow" / "did
    inventories not grow" embed at cosine 0.945, and "what were total assets" / "how does
    NVIDIA account for total assets" at 0.93-0.97 — closer than genuine paraphrases — so
    similarity alone cannot keep them apart (docs/reports/cache_threshold_calibration.md)."""
    return {
        "entity": slots.entity or default_entity,
        "periods_key": "|".join(slots.fiscal_periods),
        "metrics_key": "|".join(sorted(set(slots.metrics) | set(slots.formulas))),
        "direction": slots.direction or "none",
        "negation": "yes" if has_negation(slots.normalized_text) else "no",
        "ask": slots.ask or "value",
        # `total` is almost always part of a metric name that the glossary did not consume
        # ("total return") rather than an aggregation ask, so it does not split entries
        "aggregation_key": "|".join(sorted(set(slots.aggregation) - {"total"})),
        "topic_key": "|".join(sorted(set(slots.topics))),
    }


def embed_text_for(question: str, slots: QuerySlots, mode: str) -> str:
    """Text embedded for the semantic lookup; `mode` is calibrated in
    docs/reports/cache_threshold_calibration.md."""
    if mode == "canonical":
        return slots.canonical_text or slots.normalized_text or question
    if mode == "normalized":
        return slots.normalized_text or question
    return question


class CachedAnswer(BaseModel):
    """Everything needed to render a hit without re-running the pipeline."""

    question: str = Field(description="the question that produced the entry")
    answer: Answer
    intent: str
    intent_rule: str | None = None
    context_blocks: list[ContextBlock] = Field(default_factory=list)
    image_chunk_ids: list[str] = Field(default_factory=list)
    calculations: list[CalculationResult] = Field(default_factory=list)
    verify: VerifyResult
    generator_role: str = "-"
    origin_request_id: str
    tokens_saved_est: int = Field(default=0, description="model tokens of the original run")


class CacheEntryMeta(BaseModel):
    """Metadata view of one L2 record (admin listing, sweeper, eviction)."""

    id: str
    document: str = ""
    entity: str = ""
    periods_key: str = ""
    metrics_key: str = ""
    direction: str = "none"
    negation: str = "no"
    ask: str = "value"
    aggregation_key: str = ""
    topic_key: str = ""
    intent: str = ""
    answer_class: str = ""
    created_at: int = 0
    expires_at: int = 0
    last_hit_at: int = 0
    last_decay_at: int = 0
    lfu_counter: int = 0
    hit_count: int = 0
    corpus_version: str = ""
    prompt_version: str = ""
    generator_model: str = ""
    retrieval_config_hash: str = ""
    calculator_version: str = ""
    tokens_saved_est: int = 0

    @classmethod
    def from_metadata(cls, id_: str, meta: dict[str, Any], document: str = "") -> CacheEntryMeta:
        fields = {k: v for k, v in meta.items() if k in cls.model_fields and k != "answer_json"}
        return cls(id=id_, document=document, **fields)


class CacheHit(BaseModel):
    tier: Literal["L1", "L2"]
    entry_id: str
    similarity: float | None = Field(default=None, description="cosine; None for L1 exact hits")
    answer_class: str
    expires_at: int
    lfu_counter: int = 0
    hit_count: int = 0
    record: CachedAnswer


class CacheWrite(BaseModel):
    admitted: bool
    tiers: list[str] = Field(default_factory=list, description="tiers written: L1 / L2")
    answer_class: str | None = None
    expires_at: int | None = None
    entry_id: str | None = None
    reasons: list[str] = Field(default_factory=list, description="admission rules that failed")
    evicted: int = 0


class CacheInfo(BaseModel):
    """What the API and UI show for one request (cache panel)."""

    tier: CacheTier = "bypassed"
    similarity: float | None = None
    threshold: float | None = None
    entry_id: str | None = None
    answer_class: str | None = None
    expires_at: int | None = None
    lfu_counter: int | None = None
    hit_count: int | None = None
    origin_request_id: str | None = None
    write: CacheWrite | None = None
    lookup_ms: int = 0
    l2_candidates: int = Field(default=0, description="entries passing the slot guard")
    best_rejected_similarity: float | None = Field(
        default=None, description="closest slot-consistent entry that fell under the threshold"
    )
    clock_offset_s: int = 0
