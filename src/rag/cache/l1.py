"""L1 exact-match tier: in-process `cachetools.TTLCache`, LRU on capacity (architecture §5.4,
D-03).

Key = `QuerySlots.l1_key(version_keys)` — the canonical slot text plus the five version keys, so a
config or prompt change misses here too without any explicit flush. Every entry carries its own
`expires_at` = min(L2 expiry, 1 h); `TTLCache`'s single TTL is the 1-hour ceiling and the
per-entry value is checked lazily on read.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from cachetools import TTLCache

from rag.cache.records import CachedAnswer, CacheHit


class L1Cache:
    def __init__(self, *, max_entries: int, ttl_seconds: int, now: Callable[[], int]):
        self.now = now
        self.ttl_seconds = ttl_seconds
        self._cache: TTLCache[str, tuple[int, str, str, int, int, CachedAnswer]] = TTLCache(
            maxsize=max_entries, ttl=ttl_seconds, timer=now
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str) -> CacheHit | None:
        now = self.now()
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, entry_id, answer_class, lfu_counter, hit_count, record = item
            if expires_at <= now:
                self._cache.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
        return CacheHit(
            tier="L1",
            entry_id=entry_id,
            answer_class=answer_class,
            expires_at=expires_at,
            lfu_counter=lfu_counter,
            hit_count=hit_count,
            record=record,
        )

    def put(
        self,
        key: str,
        record: CachedAnswer,
        *,
        entry_id: str,
        answer_class: str,
        expires_at: int,
        lfu_counter: int = 0,
        hit_count: int = 0,
    ) -> None:
        with self._lock:
            self._cache[key] = (expires_at, entry_id, answer_class, lfu_counter, hit_count, record)

    def invalidate(self, entry_id: str) -> int:
        """Drop every L1 entry that mirrors a purged L2 record."""
        with self._lock:
            keys = [k for k, v in self._cache.items() if v[1] == entry_id]
            for k in keys:
                self._cache.pop(k, None)
        return len(keys)

    def clear(self) -> int:
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
        return n

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "max_entries": self._cache.maxsize,
            "ttl_ceiling_s": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
