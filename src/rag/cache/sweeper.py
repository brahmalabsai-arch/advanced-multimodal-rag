"""Active expiry and version invalidation on a timer (architecture §5.5, §5.7).

`Sweeper.sweep()` runs on app start and every `cache.sweep_interval_seconds` (15 min) from a
FastAPI lifespan task; lookups still filter on `expires_at` themselves (lazy expiry), so the
sweeper is housekeeping, not correctness. Version-mismatched ("stale") records are counted, not
deleted: they are unreachable under the current keys, become the first eviction victims under
capacity pressure, and are removed by `purge stale` or their own TTL (D-59; walkthrough step 7
needs them to survive a config revert). The L1 tier is cleared on every sweep that removed
something, because an L1 entry always mirrors an L2 record.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from rag.cache.l1 import L1Cache
from rag.cache.l2 import L2Cache
from rag.core.logging import get_logger

log = get_logger(__name__)


class Sweeper:
    def __init__(self, l2: L2Cache, l1: L1Cache | None = None, *, now: Callable[[], int]):
        self.l2 = l2
        self.l1 = l1
        self.now = now
        self._lock = threading.Lock()
        self.runs = 0
        self.last_run_at: int | None = None
        self.last_result: dict[str, int] | None = None

    def sweep(self) -> dict[str, Any]:
        with self._lock:
            result = self.l2.sweep()
            removed = result["expired"]
            if removed and self.l1 is not None:
                result["l1_cleared"] = self.l1.clear()
            self.runs += 1
            self.last_run_at = self.now()
            self.last_result = result
        if removed:
            log.info("cache sweep removed %d entries: %s", removed, result)
        return result

    async def run_forever(self, interval_seconds: int) -> None:
        """Background loop for the FastAPI lifespan; cancelled on shutdown."""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await asyncio.to_thread(self.sweep)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # housekeeping must never take the server down
                log.warning("cache sweep failed: %s", exc)

    def stats(self) -> dict[str, Any]:
        return {"runs": self.runs, "last_run_at": self.last_run_at, "last_result": self.last_result}
