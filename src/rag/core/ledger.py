"""LLM usage ledger — `data/logs/llm_usage.jsonl` (architecture §9.4, FR-11).

One line per attempt, including retries and failures, so the ledger answers both
"how much Groq quota did this phase use?" and "how often did we hit 429?".
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.core.logging import redact

UsageStatus = Literal["ok", "invalid_json", "rate_limited", "error"]


class UsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    request_id: str | None = None
    ingestion_job: str | None = None
    role: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    retries: int = 0
    status: UsageStatus
    pacing_wait_ms: int = 0
    error: str | None = None


class UsageLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: UsageRecord) -> None:
        # A provider error can echo the request back, Authorization header included (F2).
        line = redact(record.model_dump_json(exclude_none=True))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_all(self) -> list[UsageRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [UsageRecord.model_validate(json.loads(ln)) for ln in fh if ln.strip()]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for ln in fh if ln.strip())
