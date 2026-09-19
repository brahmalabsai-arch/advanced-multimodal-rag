"""Redis-style approximated LFU for the L2 tier (architecture §5.9, D-30).

Constants checked against redis.io "Key eviction" (LFU section) and `redis.conf` on
2026-09-18: `lfu-log-factor 10`, `lfu-decay-time 1` (minutes), counter range 0–255, and
`LFU_INIT_VAL 5` (server.h) so a new key is not evicted before it can accumulate hits. The
documented table for factor 10 — ~100 hits → counter 10, ~1,000 hits → 18 — is reproduced by
`test_cache.py::test_lfu_log_increment_matches_redis_table`.

Differences from Redis, both deliberate (§5.9): the decay period is 1 day instead of 1 minute
(demo traffic is tens of queries per day, not thousands per minute), and eviction scans every
entry (≤ 5,000) instead of sampling, so this is the exact form of the same policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

LFU_COUNTER_MAX = 255


@dataclass(frozen=True)
class LFUParams:
    init_val: int = 5
    log_factor: int = 10
    decay_minutes: int = 1440

    @property
    def decay_seconds(self) -> int:
        return self.decay_minutes * 60


class HasLFU(Protocol):
    lfu_counter: int
    last_decay_at: int
    expires_at: int


def decayed(counter: int, last_decay_at: int, now: int, params: LFUParams) -> int:
    """Counter after subtracting one for every whole decay period since `last_decay_at`
    (Redis `LFUDecrAndReturn`)."""
    if params.decay_seconds <= 0:
        return counter
    periods = max(now - last_decay_at, 0) // params.decay_seconds
    return max(counter - periods, 0)


def increment(counter: int, params: LFUParams, rng: random.Random) -> int:
    """Logarithmic Morris increment (Redis `LFULogIncr`): the probability of +1 is
    1 / (base * log_factor + 1) with base = counter - init_val, clamped at 0."""
    if counter >= LFU_COUNTER_MAX:
        return LFU_COUNTER_MAX
    base = max(counter - params.init_val, 0)
    p = 1.0 / (base * params.log_factor + 1)
    if rng.random() < p:
        return counter + 1
    return counter


def on_hit(
    counter: int, last_decay_at: int, now: int, params: LFUParams, rng: random.Random
) -> int:
    """New counter after a hit at `now`: decay first, then the probabilistic increment.
    The caller stores `last_decay_at = now` alongside."""
    return increment(decayed(counter, last_decay_at, now, params), params, rng)


def choose_victim(entries: list[HasLFU], now: int, params: LFUParams) -> HasLFU | None:
    """Lowest decayed counter; tie → soonest expiry (`volatile-ttl` as tie-breaker)."""
    if not entries:
        return None
    return min(
        entries, key=lambda e: (decayed(e.lfu_counter, e.last_decay_at, now, params), e.expires_at)
    )


def expected_counter_after_hits(hits: int, params: LFUParams, rng: random.Random) -> int:
    """Simulate `hits` accesses from a fresh entry (test helper for the Redis table)."""
    c = params.init_val
    for _ in range(hits):
        c = increment(c, params, rng)
    return c
