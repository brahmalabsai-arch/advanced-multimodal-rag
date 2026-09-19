"""L2 semantic tier: Chroma collection `semantic_cache` in `data/cache/chroma` (architecture
§5.8–§5.11, D-02, D-07, D-28, D-30, D-31).

Lookup = hard slot guard + version keys + `expires_at > now` as the Chroma `where` clause, then
exact cosine over the (few) survivors and the class threshold (0.90 slot-rich / 0.95 slot-poor).
Scoring the filtered set in numpy rather than through Chroma's HNSW `query` keeps the tier
deterministic on a tiny collection — the same reason retrieval scores dense candidates exactly
(D-54). Eviction is Redis `volatile-lfu` semantics over every entry (`lfu.py`).
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rag.cache.lfu import LFUParams, choose_victim, decayed, on_hit
from rag.cache.records import SLOT_FIELDS, CachedAnswer, CacheEntryMeta, CacheHit
from rag.cache.versions import VERSION_FIELDS, VersionKeys
from rag.core.logging import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "semantic_cache"


class L2Lookup:
    """Outcome of one semantic lookup, hit or not (the UI shows the near misses)."""

    def __init__(
        self, hit: CacheHit | None, *, candidates: int, threshold: float, best: float | None
    ):
        self.hit = hit
        self.candidates = candidates
        self.threshold = threshold
        self.best_similarity = best


def entry_id_for(embed_text: str, versions: VersionKeys) -> str:
    return hashlib.sha256((embed_text + "||" + versions.joined()).encode("utf-8")).hexdigest()[:32]


class L2Cache:
    def __init__(
        self,
        path: Path,
        *,
        embed: Callable[[str], np.ndarray],
        versions: VersionKeys,
        now: Callable[[], int],
        max_entries: int = 5000,
        threshold_slot_rich: float = 0.90,
        threshold_slot_poor: float = 0.95,
        lfu: LFUParams | None = None,
        rng: random.Random | None = None,
    ):
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.embed = embed
        self.versions = versions
        self.now = now
        self.max_entries = max_entries
        self.threshold_slot_rich = threshold_slot_rich
        self.threshold_slot_poor = threshold_slot_poor
        self.lfu = lfu or LFUParams()
        self.rng = rng or random.Random()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self.swept = 0

    # -- helpers ----------------------------------------------------------------------

    def count(self) -> int:
        return int(self.collection.count())

    def threshold_for(self, slot_rich: bool) -> float:
        return self.threshold_slot_rich if slot_rich else self.threshold_slot_poor

    def _where(self, slot_keys: dict[str, str], now: int) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = list(self.versions.where_clauses())
        clauses += [{f: slot_keys[f]} for f in SLOT_FIELDS]
        clauses.append({"expires_at": {"$gt": now}})
        return {"$and": clauses}

    def _all_meta(self) -> list[CacheEntryMeta]:
        got = self.collection.get(include=["metadatas", "documents"])
        out = []
        metas = got["metadatas"] or [{}] * len(got["ids"])
        docs = got["documents"] or [""] * len(got["ids"])
        for id_, meta, doc in zip(got["ids"], metas, docs, strict=True):
            out.append(CacheEntryMeta.from_metadata(id_, meta or {}, doc or ""))
        return out

    # -- lookup -------------------------------------------------------------------------

    def lookup(self, embed_text: str, slot_keys: dict[str, str], *, slot_rich: bool) -> L2Lookup:
        now = self.now()
        threshold = self.threshold_for(slot_rich)
        with self._lock:
            if self.count() == 0:
                self.misses += 1
                return L2Lookup(None, candidates=0, threshold=threshold, best=None)
            got = self.collection.get(
                where=self._where(slot_keys, now),
                include=["embeddings", "metadatas", "documents"],
            )
            ids = list(got["ids"])
            if not ids:
                self.misses += 1
                return L2Lookup(None, candidates=0, threshold=threshold, best=None)
            q = np.asarray(self.embed(embed_text), dtype=np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            vecs = np.asarray(got["embeddings"], dtype=np.float32)
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            sims = vecs @ q
            order = np.argsort(-sims)
            best = float(sims[order[0]])
            if best < threshold:
                self.misses += 1
                return L2Lookup(None, candidates=len(ids), threshold=threshold, best=best)
            i = int(order[0])
            meta = dict((got["metadatas"] or [{}])[i] or {})
            entry_id = ids[i]
            hit = self._on_hit(entry_id, meta, now, best)
            self.hits += 1
            return L2Lookup(hit, candidates=len(ids), threshold=threshold, best=best)

    def _on_hit(self, entry_id: str, meta: dict[str, Any], now: int, sim: float) -> CacheHit:
        counter = on_hit(
            int(meta.get("lfu_counter", 0)),
            int(meta.get("last_decay_at", now)),
            now,
            self.lfu,
            self.rng,
        )
        hit_count = int(meta.get("hit_count", 0)) + 1
        update = {
            "lfu_counter": counter,
            "last_decay_at": now,
            "last_hit_at": now,
            "hit_count": hit_count,
        }
        self.collection.update(ids=[entry_id], metadatas=[{**meta, **update}])
        record = CachedAnswer.model_validate_json(meta["answer_json"])
        return CacheHit(
            tier="L2",
            entry_id=entry_id,
            similarity=round(sim, 4),
            answer_class=str(meta.get("answer_class", "")),
            expires_at=int(meta.get("expires_at", 0)),
            lfu_counter=counter,
            hit_count=hit_count,
            record=record,
        )

    # -- write --------------------------------------------------------------------------

    def write(
        self,
        embed_text: str,
        slot_keys: dict[str, str],
        record: CachedAnswer,
        *,
        intent: str,
        answer_class: str,
        expires_at: int,
    ) -> tuple[str, int]:
        """Upsert one record; returns (entry_id, evicted_count). Runs `make_room` first."""
        now = self.now()
        entry_id = entry_id_for(embed_text, self.versions)
        with self._lock:
            evicted = self.make_room(reserve=1)
            vec = np.asarray(self.embed(embed_text), dtype=np.float32)
            meta: dict[str, Any] = {
                **{f: getattr(self.versions, f) for f in VERSION_FIELDS},
                **{f: slot_keys[f] for f in SLOT_FIELDS},
                "intent": intent,
                "answer_class": answer_class,
                "created_at": now,
                "expires_at": int(expires_at),
                "last_hit_at": now,
                "last_decay_at": now,
                "lfu_counter": self.lfu.init_val,
                "hit_count": 0,
                "tokens_saved_est": int(record.tokens_saved_est),
                "answer_json": record.model_dump_json(),
            }
            self.collection.upsert(
                ids=[entry_id], embeddings=[vec.tolist()], documents=[embed_text], metadatas=[meta]
            )
            self.writes += 1
        return entry_id, evicted

    # -- maintenance --------------------------------------------------------------------

    def sweep(self, *, include_stale: bool = False) -> dict[str, int]:
        """Active expiry (§5.5): delete expired records and report how many live records carry
        other version keys ("stale"). Stale records are invisible to lookups already (§5.7
        filters on every key), so by default they stay until their TTL — that is what lets a
        reverted config change hit its original entries again (walkthrough step 7) — and they
        are the first victims when capacity is short (`make_room`). `include_stale=True`
        (admin `purge stale`) removes them now."""
        now = self.now()
        with self._lock:
            entries = self._all_meta()
            expired = [e.id for e in entries if e.expires_at <= now]
            stale = [
                e.id
                for e in entries
                if e.id not in set(expired) and not self.versions.matches(e.model_dump())
            ]
            doomed = expired + (stale if include_stale else [])
            if doomed:
                self.collection.delete(ids=doomed)
                self.swept += len(doomed)
        return {
            "expired": len(expired),
            "version_mismatch": len(stale),
            "stale_removed": len(stale) if include_stale else 0,
            "remaining": self.count(),
        }

    def make_room(self, reserve: int = 0) -> int:
        """`volatile-lfu` with one refinement: after the expiry sweep, stale-version records go
        first (they can never be served under the current keys), then the lowest decayed LFU
        counter (tie → soonest expiry) until `count + reserve <= max_entries`.

        The capacity check comes first and is a `count()`, not a scan: the Phase 7 resource
        profile measured `cache_write` at p50 88 ms because every write swept the whole
        collection to make room it did not need. Expiry still happens lazily on lookup and
        actively in the 15-minute sweeper."""
        now = self.now()
        with self._lock:
            if self.count() + reserve <= self.max_entries:
                return 0
            self.sweep()
            evicted = 0
            entries = self._all_meta()
            stale = [e for e in entries if not self.versions.matches(e.model_dump())]
            while len(entries) + reserve > self.max_entries and entries:
                victim = (
                    choose_victim(stale, now, self.lfu)  # type: ignore[arg-type]
                    if stale
                    else choose_victim(entries, now, self.lfu)  # type: ignore[arg-type]
                )
                if victim is None:
                    break
                self.collection.delete(ids=[victim.id])
                entries = [e for e in entries if e.id != victim.id]
                stale = [e for e in stale if e.id != victim.id]
                evicted += 1
            self.evictions += evicted
        return evicted

    def entries(self, *, offset: int = 0, limit: int = 50) -> tuple[list[CacheEntryMeta], int]:
        now = self.now()
        entries = self._all_meta()
        for e in entries:
            # decayed counter is what eviction compares; expose it for the admin listing
            e.lfu_counter = decayed(e.lfu_counter, e.last_decay_at, now, self.lfu)
        entries.sort(key=lambda e: (-e.lfu_counter, e.expires_at))
        return entries[offset : offset + limit], len(entries)

    def purge(self, scope: str = "all", value: str | None = None) -> int:
        """scope ∈ {all, class, slot, stale}. `class` takes a TTL class; `slot` takes
        `field=value` (e.g. `periods_key=FY2026`); `stale` removes version-mismatched records."""
        with self._lock:
            before = self.count()
            if scope == "stale":
                return self.sweep(include_stale=True)["stale_removed"]
            if scope == "all":
                self._client.delete_collection(COLLECTION_NAME)
                self.collection = self._client.get_or_create_collection(
                    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
                return before
            if scope == "class":
                if not value:
                    raise ValueError("purge class requires a TTL class")
                self.collection.delete(where={"answer_class": value})
            elif scope == "slot":
                if not value or "=" not in value:
                    raise ValueError("purge slot requires field=value")
                field, val = value.split("=", 1)
                if field not in SLOT_FIELDS:
                    raise ValueError(f"unknown slot field {field!r}; expected one of {SLOT_FIELDS}")
                self.collection.delete(where={field: val})
            else:
                raise ValueError(f"unknown purge scope {scope!r}")
            return before - self.count()

    def stats(self) -> dict[str, Any]:
        now = self.now()
        entries = self._all_meta()
        by_class: dict[str, int] = {}
        expired = 0
        stale = 0
        for e in entries:
            by_class[e.answer_class] = by_class.get(e.answer_class, 0) + 1
            if e.expires_at <= now:
                expired += 1
            elif not self.versions.matches(e.model_dump()):
                stale += 1
        total = self.hits + self.misses
        return {
            "entries": len(entries),
            "max_entries": self.max_entries,
            "by_class": by_class,
            "expired_pending_sweep": expired,
            "stale_version": stale,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "writes": self.writes,
            "evictions": self.evictions,
            "swept": self.swept,
            "thresholds": {
                "slot_rich": self.threshold_slot_rich,
                "slot_poor": self.threshold_slot_poor,
            },
            "lfu": {
                "init_val": self.lfu.init_val,
                "log_factor": self.lfu.log_factor,
                "decay_minutes": self.lfu.decay_minutes,
            },
            "versions": json.loads(self.versions.model_dump_json()),
        }
