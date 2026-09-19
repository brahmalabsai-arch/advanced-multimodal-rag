"""Two-tier cache facade used by the graph nodes (architecture §5.4).

    lookup:  L1 exact (slots.l1_key + version keys) → L2 semantic (slot guard + cosine) → miss
    write:   admission (§5.6) → TTL class (§5.5) → L2 upsert (+ LFU make_room) → L1 mirror

`bypass_cache` is handled by the caller (the graph skips both nodes); this class never reads the
flag, so evaluations that construct it directly get the same behaviour as the API.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from rag.cache.admission import AdmissionDecision, admission_decision
from rag.cache.l1 import L1Cache
from rag.cache.l2 import L2Cache
from rag.cache.lfu import LFUParams
from rag.cache.records import (
    CachedAnswer,
    CacheHit,
    CacheInfo,
    CacheWrite,
    embed_text_for,
    slot_keys,
)
from rag.cache.sweeper import Sweeper
from rag.cache.ttl import expires_at, l1_expires_at, ttl_class
from rag.cache.versions import VersionKeys
from rag.core.clock import Clock
from rag.core.config import CacheThresholds
from rag.core.logging import get_logger
from rag.query.slots import QuerySlots

log = get_logger(__name__)


class CacheService:
    def __init__(
        self,
        *,
        cache_dir: Path,
        embed,
        versions: VersionKeys,
        config: CacheThresholds,
        clock: Clock,
        default_entity: str,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.clock = clock
        self.versions = versions
        self.default_entity = default_entity
        ev = config.l2.eviction
        self.l1 = L1Cache(
            max_entries=config.l1.max_entries, ttl_seconds=config.l1.ttl_seconds, now=clock.now
        )
        self.l2 = L2Cache(
            cache_dir / "chroma",
            embed=embed,
            versions=versions,
            now=clock.now,
            max_entries=config.l2.max_entries,
            threshold_slot_rich=config.l2.similarity.slot_rich,
            threshold_slot_poor=config.l2.similarity.slot_poor,
            lfu=LFUParams(
                init_val=ev.lfu_init_val,
                log_factor=ev.lfu_log_factor,
                decay_minutes=ev.lfu_decay_minutes,
            ),
            rng=rng,
        )
        self.sweeper = Sweeper(self.l2, self.l1, now=clock.now)
        self.embed_mode = config.l2.embed_text

    # -- keys -----------------------------------------------------------------------------

    def l1_key(self, slots: QuerySlots) -> str:
        return slots.l1_key(self.versions.joined())

    def keys(self, slots: QuerySlots) -> dict[str, str]:
        return slot_keys(slots, default_entity=self.default_entity)

    def embed_text(self, question: str, slots: QuerySlots) -> str:
        return embed_text_for(question, slots, self.embed_mode)

    # -- lookup ---------------------------------------------------------------------------

    def lookup(self, question: str, slots: QuerySlots) -> tuple[CacheHit | None, CacheInfo]:
        t0 = time.perf_counter()
        info = CacheInfo(tier="MISS", clock_offset_s=self.clock.offset_s)
        hit = self.l1.get(self.l1_key(slots))
        if hit is None:
            res = self.l2.lookup(
                self.embed_text(question, slots), self.keys(slots), slot_rich=slots.slot_rich
            )
            info.threshold = res.threshold
            info.l2_candidates = res.candidates
            if res.hit is None:
                info.best_rejected_similarity = (
                    round(res.best_similarity, 4) if res.best_similarity is not None else None
                )
            else:
                hit = res.hit
                # promote: L1 mirrors the L2 record under the exact key of *this* phrasing
                self.l1.put(
                    self.l1_key(slots),
                    hit.record,
                    entry_id=hit.entry_id,
                    answer_class=hit.answer_class,
                    expires_at=l1_expires_at(
                        hit.expires_at, self.clock.now(), self.config.l1.ttl_seconds
                    ),
                    lfu_counter=hit.lfu_counter,
                    hit_count=hit.hit_count,
                )
        if hit is not None:
            info.tier = hit.tier
            info.similarity = hit.similarity
            info.entry_id = hit.entry_id
            info.answer_class = hit.answer_class
            info.expires_at = hit.expires_at
            info.lfu_counter = hit.lfu_counter
            info.hit_count = hit.hit_count
            info.origin_request_id = hit.record.origin_request_id
        info.lookup_ms = int((time.perf_counter() - t0) * 1000)
        return hit, info

    # -- write ----------------------------------------------------------------------------

    def admit(
        self,
        *,
        question: str,
        intent: str,
        verify_passed: bool,
        confidence: str,
        citations: list[str],
        degraded: bool = False,
    ) -> AdmissionDecision:
        return admission_decision(
            question=question,
            verify_passed=verify_passed,
            confidence=confidence,
            intent=intent,
            citations=citations,
            degraded=degraded,
        )

    def write(
        self,
        question: str,
        slots: QuerySlots,
        record: CachedAnswer,
        *,
        decision: AdmissionDecision,
    ) -> CacheWrite:
        if not decision.admitted:
            return CacheWrite(admitted=False, reasons=decision.reasons)
        now = self.clock.now()
        cls = ttl_class(
            record.answer.answer_class,
            time_anchor=slots.time_anchor,
            answer_markdown=record.answer.answer_markdown,
        )
        exp = expires_at(
            cls,
            now=now,
            ttls=self.config.l2.ttl_seconds,
            answer_markdown=record.answer.answer_markdown,
        )
        write = CacheWrite(
            admitted=True, answer_class=cls, expires_at=exp, reasons=decision.reasons
        )
        entry_id = None
        if decision.admit_l2:
            entry_id, evicted = self.l2.write(
                self.embed_text(question, slots),
                self.keys(slots),
                record,
                intent=record.intent,
                answer_class=cls,
                expires_at=exp,
            )
            write.tiers.append("L2")
            write.evicted = evicted
        if decision.admit_l1:
            entry_id = entry_id or f"l1-{self.l1_key(slots)[:16]}"
            self.l1.put(
                self.l1_key(slots),
                record,
                entry_id=entry_id,
                answer_class=cls,
                expires_at=l1_expires_at(exp, now, self.config.l1.ttl_seconds),
                lfu_counter=self.l2.lfu.init_val,
            )
            write.tiers.append("L1")
        write.entry_id = entry_id
        return write

    # -- admin ------------------------------------------------------------------------------

    def purge(self, scope: str = "all", value: str | None = None) -> dict[str, int]:
        """`l1` clears only the in-process tier (what a server restart does — used by the
        scripted walkthrough for step 3); every other scope purges L2 and mirrors into L1."""
        if scope == "l1":
            return {"l2_removed": 0, "l1_cleared": self.l1.clear()}
        removed = self.l2.purge(scope, value)
        return {"l2_removed": removed, "l1_cleared": self.l1.clear()}

    def stats(self) -> dict:
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
            "sweeper": self.sweeper.stats(),
            "clock": {
                "offset_s": self.clock.offset_s,
                "now": self.clock.now(),
                "now_iso": self.clock.now_iso(),
            },
            "embed_text": self.embed_mode,
        }
