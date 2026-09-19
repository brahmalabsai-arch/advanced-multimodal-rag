"""Per-request trace log — `data/logs/traces.jsonl` (architecture §9.3, FR-11).

One line per request. `rss_mb` is sampled so the later hosting decision rests on measurements.
Phases 4–6 add slots, rerank, compression and cache fields (all optional here).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:  # psutil missing or unsupported platform
        return None


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    clock_offset_s: int = 0
    model_profile: str
    query: str
    slots: dict[str, Any] | None = None
    cache_tier: str = "bypassed"
    cache_similarity: float | None = None
    intent: str
    intent_rule: str | None = None
    expansions: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    fused_top_ids: list[str] = Field(default_factory=list)
    rerank_applied: bool = False
    rerank_skip_reason: str | None = None
    compression_decision: dict[str, Any] | None = None
    context_block_ids: list[str] = Field(default_factory=list)
    context_tokens: int = 0
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    calculator_calls: list[str] = Field(default_factory=list)
    generation_attempts: int = 0
    verify_passed: bool | None = None
    verify_issues: list[str] = Field(default_factory=list)
    admitted: bool = False
    answer_class: str | None = None
    confidence: str | None = None
    latency_ms_by_node: dict[str, int] = Field(default_factory=dict)
    total_latency_ms: int = 0
    rss_mb: float | None = None
    error: str | None = None
    degraded: bool = False


class TraceWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, trace: Trace) -> None:
        line = trace.model_dump_json(exclude_none=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def find(self, request_id: str) -> Trace | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip() and f'"request_id":"{request_id}"' in line:
                    return Trace.model_validate(json.loads(line))
        return None
