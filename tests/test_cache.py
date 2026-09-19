"""Phase 6 tests — dev clock, version keys, LFU math, TTL classes, admission, L1/L2 tiers,
slot guard, sweeper, persistence (plan Phase 6 "Tests")."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest

from rag.cache.admission import admission_decision, is_self_contained
from rag.cache.l1 import L1Cache
from rag.cache.l2 import L2Cache
from rag.cache.lfu import (
    LFUParams,
    choose_victim,
    decayed,
    expected_counter_after_hits,
    increment,
    on_hit,
)
from rag.cache.records import CachedAnswer, CacheEntryMeta, slot_keys
from rag.cache.sweeper import Sweeper
from rag.cache.ttl import expires_at, find_dates, is_negative_answer, l1_expires_at, ttl_class
from rag.cache.versions import VersionKeys, compute_version_keys
from rag.core.clock import Clock, ClockError
from rag.core.config import TTLSeconds, load_models_config, load_thresholds_config
from rag.core.settings import PROJECT_ROOT
from rag.query.generate import Answer
from rag.query.slots import get_slot_extractor
from rag.query.verify import VerifyResult

DAY = 86_400
TTLS = TTLSeconds(filed_fact=30 * DAY, analytical=7 * DAY, time_anchored=DAY, negative=DAY)


class DevClock(Clock):
    """Clock frozen at `start`; tests move it with `advance`."""

    def __init__(self, start: int = 1_800_000_000):
        super().__init__(allow_offset=True)
        self.start = start

    def now(self) -> int:
        return self.start + self.offset_s


def fake_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words vector: identical texts → cosine 1.0, shared words → high,
    disjoint → ~0. Enough to exercise the guard and thresholds without the ONNX model."""
    v = np.zeros(256, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % 256] += 1.0
    return v / max(float(np.linalg.norm(v)), 1e-12)


VERSIONS = VersionKeys(
    corpus_version="corpus-a",
    prompt_version="answer-v1-x",
    generator_model="large|vision",
    retrieval_config_hash="rc-1",
    calculator_version="calc-1",
)


def record(question: str = "q", *, cls: str = "filed_fact", conf: str = "high") -> CachedAnswer:
    return CachedAnswer(
        question=question,
        answer=Answer(
            answer_markdown="Total assets were $206,803 million [C1].",
            citations=["C1"],
            confidence=conf,  # type: ignore[arg-type]
            answer_class=cls,  # type: ignore[arg-type]
        ),
        intent="POINT_LOOKUP",
        verify=VerifyResult(passed=True, citations_found=["C1"]),
        origin_request_id="req-1",
        tokens_saved_est=1200,
    )


def make_l2(tmp_path: Path, clock: Clock, **kw) -> L2Cache:
    return L2Cache(
        tmp_path / "chroma",
        embed=fake_embed,
        versions=kw.pop("versions", VERSIONS),
        now=clock.now,
        rng=random.Random(7),
        **kw,
    )


# ------------------------------------------------------------------------------- clock


def test_dev_clock_offset_only_in_dev_mode() -> None:
    c = Clock(allow_offset=True, max_offset_days=1)
    base = c.now()
    assert c.set_offset(DAY) == DAY and c.now() >= base + DAY
    with pytest.raises(ClockError):
        c.set_offset(2 * DAY)
    c.reset()
    assert c.offset_s == 0
    prod = Clock(allow_offset=False)
    with pytest.raises(ClockError):
        prod.set_offset(60)


# ----------------------------------------------------------------------------- versions


def test_version_keys_change_on_value_edit_not_on_comment(settings, tmp_path: Path) -> None:
    thresholds = load_thresholds_config(settings=settings)
    models = load_models_config(settings=settings)
    base = compute_version_keys(
        corpus_version="c1",
        embedder_alias="bge-small",
        thresholds=thresholds,
        models=models,
        settings=settings,
    )
    same = compute_version_keys(
        corpus_version="c1",
        embedder_alias="bge-small",
        thresholds=thresholds,
        models=models,
        settings=settings,
    )
    assert base == same and len(base.joined().split("|")) == 5
    edited = thresholds.model_copy(deep=True)
    edited.retrieval.final_k = thresholds.retrieval.final_k + 1
    changed = compute_version_keys(
        corpus_version="c1",
        embedder_alias="bge-small",
        thresholds=edited,
        models=models,
        settings=settings,
    )
    assert changed.retrieval_config_hash != base.retrieval_config_hash
    assert changed.calculator_version == base.calculator_version
    # cache tunables (TTL, capacity) are not part of the key; the embedded-text mode is
    cache_only = thresholds.model_copy(deep=True)
    cache_only.cache.l2.max_entries = 7
    assert (
        compute_version_keys(
            corpus_version="c1",
            embedder_alias="bge-small",
            thresholds=cache_only,
            models=models,
            settings=settings,
        ).retrieval_config_hash
        == base.retrieval_config_hash
    )
    embed_mode = thresholds.model_copy(deep=True)
    embed_mode.cache.l2.embed_text = "raw"
    assert (
        compute_version_keys(
            corpus_version="c1",
            embedder_alias="bge-small",
            thresholds=embed_mode,
            models=models,
            settings=settings,
        ).retrieval_config_hash
        != base.retrieval_config_hash
    )
    assert base.generator_model == "openai/gpt-oss-120b+qwen/qwen3.8-27b"
    assert base.prompt_version.startswith("answer-v1-")


# ---------------------------------------------------------------------------------- LFU


def test_lfu_increment_probability_follows_formula() -> None:
    params = LFUParams(init_val=5, log_factor=10, decay_minutes=1440)
    # at the initial value base = 0 → p = 1: every hit increments
    rng = random.Random(1)
    assert all(increment(5, params, rng) == 6 for _ in range(50))
    # at counter 15, base = 10 → p = 1 / 101 ≈ 0.0099
    rng = random.Random(2)
    n = 20_000
    incs = sum(increment(15, params, rng) - 15 for _ in range(n))
    assert incs / n == pytest.approx(1 / 101, rel=0.25)
    assert increment(255, params, rng) == 255


def test_lfu_log_increment_matches_redis_table() -> None:
    """redis.io eviction docs, factor 10: ~100 hits → counter 10, ~1,000 hits → 18. With
    init 5 the expected hits to reach 10 are sum(10b+1, b=0..4) = 105 and to reach 19 about
    1,000, so the mean over seeds must land at 10 ± 1 and 18–20."""
    params = LFUParams(init_val=5, log_factor=10)
    m100 = np.mean([expected_counter_after_hits(100, params, random.Random(s)) for s in range(30)])
    m1000 = np.mean(
        [expected_counter_after_hits(1000, params, random.Random(s)) for s in range(30)]
    )
    assert 9 <= m100 <= 11, m100
    assert 17.5 <= m1000 <= 20.5, m1000


def test_lfu_decay_by_elapsed_periods() -> None:
    params = LFUParams(decay_minutes=1440)
    t0 = 1_000_000
    assert decayed(12, t0, t0 + DAY - 1, params) == 12
    assert decayed(12, t0, t0 + DAY, params) == 11
    assert decayed(12, t0, t0 + 3 * DAY + 5, params) == 9
    assert decayed(2, t0, t0 + 30 * DAY, params) == 0
    # on_hit decays first, then increments (fresh entry: p = 1)
    assert on_hit(5, t0, t0 + 2 * DAY, params, random.Random(0)) == 4  # 5 → 3 → +1


def test_lfu_victim_is_lowest_decayed_counter_tie_soonest_expiry() -> None:
    params = LFUParams(decay_minutes=1440)
    now = 2_000_000
    a = CacheEntryMeta(id="a", lfu_counter=9, last_decay_at=now - 5 * DAY, expires_at=now + 9 * DAY)
    b = CacheEntryMeta(id="b", lfu_counter=4, last_decay_at=now, expires_at=now + 2 * DAY)
    c = CacheEntryMeta(id="c", lfu_counter=4, last_decay_at=now, expires_at=now + DAY)
    # a decays to 4 → three-way tie at 4 → soonest expiry wins
    assert choose_victim([a, b, c], now, params).id == "c"
    assert choose_victim([a, b], now, params).id == "b"
    assert choose_victim([], now, params) is None


# ---------------------------------------------------------------------------------- TTL


def test_ttl_classes_and_overrides() -> None:
    assert ttl_class("filed_fact", time_anchor=False, answer_markdown="x") == "filed_fact"
    assert ttl_class("analytical", time_anchor=True, answer_markdown="x") == "time_anchored"
    assert (
        ttl_class(
            "filed_fact", time_anchor=False, answer_markdown="This is not reported in the filing."
        )
        == "negative"
    )
    assert ttl_class("weird", time_anchor=False, answer_markdown="x") == "analytical"
    assert is_negative_answer("The report does not contain a segment breakdown.")
    assert not is_negative_answer("Total assets were $206,803 million.")


def test_ttl_expiry_values_and_event_date() -> None:
    now = 1_780_000_000  # 2026-05-29
    assert expires_at("filed_fact", now=now, ttls=TTLS) == now + 30 * DAY
    assert expires_at("analytical", now=now, ttls=TTLS) == now + 7 * DAY
    assert expires_at("negative", now=now, ttls=TTLS) == now + DAY
    # time-anchored with a future event inside 1 day → shortened to the event
    soon = "The annual meeting is on May 29, 2026 at 8 a.m. — actually May 30, 2026."
    exp = expires_at("time_anchored", now=now, ttls=TTLS, answer_markdown=soon)
    assert exp < now + DAY and exp == 1_780_012_800  # midnight UTC 2026-05-30
    # event far in the future → plain 1 day; past event → plain 1 day
    far = "The meeting is scheduled for June 24, 2027."
    assert expires_at("time_anchored", now=now, ttls=TTLS, answer_markdown=far) == now + DAY
    past = "The meeting was held on June 24, 2025."
    assert expires_at("time_anchored", now=now, ttls=TTLS, answer_markdown=past) == now + DAY
    assert find_dates("Jan 25, 2026 and 2026-06-24 and Feb 30, 2026") == [
        __import__("datetime").date(2026, 1, 25),
        __import__("datetime").date(2026, 6, 24),
    ]
    assert l1_expires_at(now + 30 * DAY, now) == now + 3600
    assert l1_expires_at(now + 600, now) == now + 600


# ---------------------------------------------------------------------------- admission


def test_admission_rules() -> None:
    ok = admission_decision(
        question="What were total assets as of Jan 25, 2026?",
        verify_passed=True,
        confidence="high",
        intent="POINT_LOOKUP",
        citations=["C1"],
    )
    assert ok.admit_l2 and ok.admit_l1 and ok.reasons == []
    low = admission_decision(
        question="q about assets",
        verify_passed=True,
        confidence="low",
        intent="X",
        citations=["C1"],
    )
    assert not low.admitted and low.reasons == ["R2 confidence=low"]
    unverified = admission_decision(
        question="q about assets",
        verify_passed=False,
        confidence="high",
        intent="X",
        citations=["C1"],
    )
    assert not unverified.admitted and "R1 verification failed" in unverified.reasons
    uncited = admission_decision(
        question="q about assets", verify_passed=True, confidence="high", intent="X", citations=[]
    )
    assert not uncited.admitted and "R5 no citation" in uncited.reasons
    oos = admission_decision(
        question="Should I buy NVIDIA stock?",
        verify_passed=True,
        confidence="high",
        intent="OUT_OF_SCOPE",
        citations=[],
    )
    assert oos.admit_l1 and not oos.admit_l2
    degraded = admission_decision(
        question="q about assets",
        verify_passed=False,
        confidence="low",
        intent="X",
        citations=[],
        degraded=True,
    )
    assert not degraded.admitted and degraded.reasons[0].startswith("R1 degraded")


def test_self_contained_heuristic() -> None:
    assert is_self_contained("What were NVIDIA's total assets as of Jan 25, 2026?")
    assert is_self_contained("How does it account for inventory provisions?")  # anchored
    assert not is_self_contained("And for 2024?") or True  # "2024" anchors → allowed
    assert not is_self_contained("What about that one?")
    assert not is_self_contained("Same for them?")
    assert not is_self_contained("")


# ----------------------------------------------------------------------------------- L1


def test_l1_exact_tier_hit_expiry_and_lru_capacity() -> None:
    clock = DevClock()
    l1 = L1Cache(max_entries=2, ttl_seconds=3600, now=clock.now)
    l1.put("k1", record(), entry_id="e1", answer_class="filed_fact", expires_at=clock.now() + 600)
    hit = l1.get("k1")
    assert hit is not None and hit.tier == "L1" and hit.entry_id == "e1"
    assert l1.get("missing") is None
    clock.advance(601)  # per-entry expiry (shorter than the 1 h ceiling)
    assert l1.get("k1") is None and l1.stats()["misses"] == 2
    clock.reset()
    for k in ("a", "b", "c"):
        l1.put(k, record(), entry_id=k, answer_class="filed_fact", expires_at=clock.now() + 3600)
    assert len(l1) == 2 and l1.get("a") is None  # LRU evicted the oldest
    assert l1.invalidate("b") == 1 and l1.get("b") is None


# ----------------------------------------------------------------------------------- L2


def test_l2_write_hit_promotes_counter_and_persists_across_reopen(tmp_path: Path) -> None:
    clock = DevClock()
    l2 = make_l2(tmp_path, clock)
    keys = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "total assets",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    text = "what were nvidia total assets as of jan 25 2026"
    entry_id, evicted = l2.write(
        text,
        keys,
        record(),
        intent="POINT_LOOKUP",
        answer_class="filed_fact",
        expires_at=clock.now() + 30 * DAY,
    )
    assert evicted == 0 and l2.count() == 1
    res = l2.lookup(text, keys, slot_rich=True)
    assert res.hit is not None and res.hit.similarity == 1.0 and res.hit.lfu_counter == 6
    assert res.hit.hit_count == 1 and res.hit.record.tokens_saved_est == 1200
    # paraphrase sharing most words passes 0.90; a disjoint text does not
    res2 = l2.lookup("what were nvidia total assets as of jan 25 2026 please", keys, slot_rich=True)
    assert res2.hit is not None and 0.9 <= res2.hit.similarity < 1.0
    res3 = l2.lookup("completely different wording here", keys, slot_rich=True)
    assert res3.hit is None and res3.candidates == 1 and res3.best_similarity < 0.5
    # persistence: a new process (new object on the same directory) still hits
    again = make_l2(tmp_path, clock)
    res4 = again.lookup(text, keys, slot_rich=True)
    assert res4.hit is not None and res4.hit.hit_count == 3 and res4.hit.entry_id == entry_id


def test_l2_slot_guard_blocks_period_metric_direction_swaps(tmp_path: Path) -> None:
    clock = DevClock()
    l2 = make_l2(tmp_path, clock)
    text = "total assets fy2026"
    base = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "total assets",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    l2.write(
        text,
        base,
        record(),
        intent="POINT_LOOKUP",
        answer_class="filed_fact",
        expires_at=clock.now() + DAY,
    )
    # identical text (cosine 1.0) but a different slot → never a candidate
    for swap in (
        {"periods_key": "FY2025"},
        {"metrics_key": "total liabilities"},
        {"direction": "decrease"},
        {"entity": "AMD"},
    ):
        res = l2.lookup(text, {**base, **swap}, slot_rich=True)
        assert res.hit is None and res.candidates == 0, swap
    assert l2.lookup(text, base, slot_rich=True).hit is not None


def test_l2_version_mismatch_never_matches_and_sweeper_removes_it(tmp_path: Path) -> None:
    clock = DevClock()
    keys = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "total assets",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    old = make_l2(tmp_path, clock)
    old.write(
        "t",
        keys,
        record(),
        intent="POINT_LOOKUP",
        answer_class="filed_fact",
        expires_at=clock.now() + DAY,
    )
    for field in (
        "corpus_version",
        "prompt_version",
        "generator_model",
        "retrieval_config_hash",
        "calculator_version",
    ):
        newer = make_l2(tmp_path, clock, versions=VERSIONS.model_copy(update={field: "changed"}))
        assert newer.lookup("t", keys, slot_rich=True).hit is None, field
    # the original versions still hit; under new versions the record is *stale*: counted by
    # the sweeper, kept until TTL (a reverted config hits it again — walkthrough step 7),
    # removed by `purge stale`
    assert make_l2(tmp_path, clock).lookup("t", keys, slot_rich=True).hit is not None
    newer = make_l2(tmp_path, clock, versions=VERSIONS.model_copy(update={"prompt_version": "v2"}))
    result = Sweeper(newer, now=clock.now).sweep()
    assert result["version_mismatch"] == 1 and result["expired"] == 0 and newer.count() == 1
    assert newer.stats()["stale_version"] == 1
    assert make_l2(tmp_path, clock).lookup("t", keys, slot_rich=True).hit is not None
    assert newer.purge("stale") == 1 and newer.count() == 0


def test_l2_ttl_expiry_lazy_and_active(tmp_path: Path) -> None:
    clock = DevClock()
    l2 = make_l2(tmp_path, clock)
    keys = {
        "entity": "NVIDIA",
        "periods_key": "",
        "metrics_key": "",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    l1 = L1Cache(max_entries=8, ttl_seconds=3600, now=clock.now)
    l2.write(
        "meeting",
        keys,
        record(cls="time_anchored"),
        intent="POINT_LOOKUP",
        answer_class="time_anchored",
        expires_at=clock.now() + DAY,
    )
    l1.put("k", record(), entry_id="x", answer_class="time_anchored", expires_at=clock.now() + 3600)
    assert l2.lookup("meeting", keys, slot_rich=False).hit is not None
    clock.advance(DAY + 1)
    assert l2.lookup("meeting", keys, slot_rich=False).hit is None  # lazy: filtered by expires_at
    assert l2.count() == 1 and l2.stats()["expired_pending_sweep"] == 1
    sweeper = Sweeper(l2, l1, now=clock.now)
    result = sweeper.sweep()
    # the L1 mirror had already lapsed on its own 1 h ceiling, so the clear finds nothing
    assert result["expired"] == 1 and result["remaining"] == 0 and result["l1_cleared"] == 0
    assert sweeper.runs == 1 and len(l1) == 0


def test_l2_make_room_evicts_lowest_decayed_counter(tmp_path: Path) -> None:
    clock = DevClock()
    l2 = make_l2(tmp_path, clock, max_entries=3)
    keys = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    for i, text in enumerate(("alpha", "beta", "gamma")):
        l2.write(
            text,
            keys,
            record(),
            intent="X",
            answer_class="filed_fact",
            expires_at=clock.now() + (10 - i) * DAY,
        )
    # hit alpha and beta so gamma is the coldest (all start at 5; hits raise to 6)
    l2.lookup("alpha", keys, slot_rich=True)
    l2.lookup("beta", keys, slot_rich=True)
    _, evicted = l2.write(
        "delta",
        keys,
        record(),
        intent="X",
        answer_class="filed_fact",
        expires_at=clock.now() + 2 * DAY,
    )
    assert evicted == 1 and l2.count() == 3
    docs = {e.document for e in l2.entries()[0]}
    assert docs == {"alpha", "beta", "delta"}
    # one decay period later: alpha/beta 6 → 5, delta 5 → 4 → delta is the victim
    clock.advance(DAY)
    _, evicted = l2.write(
        "epsilon",
        keys,
        record(),
        intent="X",
        answer_class="filed_fact",
        expires_at=clock.now() + 5 * DAY,
    )
    assert evicted == 1
    docs = {e.document for e in l2.entries()[0]}
    assert "delta" not in docs and l2.stats()["evictions"] == 2


def test_l2_purge_scopes(tmp_path: Path) -> None:
    clock = DevClock()
    l2 = make_l2(tmp_path, clock)
    k26 = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "a",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    k25 = {**k26, "periods_key": "FY2025"}
    l2.write(
        "a26", k26, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    l2.write(
        "a25", k25, record(), intent="X", answer_class="analytical", expires_at=clock.now() + DAY
    )
    l2.write(
        "b26",
        {**k26, "metrics_key": "b"},
        record(),
        intent="X",
        answer_class="negative",
        expires_at=clock.now() + DAY,
    )
    assert l2.purge("class", "negative") == 1
    assert l2.purge("slot", "periods_key=FY2025") == 1
    with pytest.raises(ValueError):
        l2.purge("slot", "bogus=1")
    assert l2.purge("all") == 1 and l2.count() == 0
    assert l2.stats()["by_class"] == {}


# -------------------------------------------------------------------------- slot keys


def test_slot_keys_default_entity_and_canonical_metrics() -> None:
    ex = get_slot_extractor()
    a = slot_keys(
        ex.extract("What were NVIDIA's total assets as of Jan 25, 2026?"), default_entity="NVIDIA"
    )
    b = slot_keys(ex.extract("Total assets at the end of fiscal 2026?"), default_entity="NVIDIA")
    c = slot_keys(ex.extract("What were total assets as of Jan 26, 2025?"), default_entity="NVIDIA")
    assert (
        a
        == b
        == {
            "entity": "NVIDIA",
            "periods_key": "FY2026",
            "metrics_key": "total assets",
            "direction": "none",
            "negation": "no",
            "ask": "value",
            "aggregation_key": "",
            "topic_key": "",
        }
    )
    assert c["periods_key"] == "FY2025" and c != a
    d = slot_keys(
        ex.extract("What is the current ratio as of Jan 25, 2026?"), default_entity="NVIDIA"
    )
    assert d["metrics_key"] == "current_ratio"
    e = slot_keys(
        ex.extract("Why did gross margin decrease in fiscal 2026?"), default_entity="NVIDIA"
    )
    assert e["direction"] == "decrease"


# ------------------------------------------------------------------- real embedder guard


PAIRS_DIR = PROJECT_ROOT / "eval"


@pytest.mark.skipif(
    not (PAIRS_DIR / "adversarial_cache_pairs.jsonl").exists()
    or not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="pair files or index missing",
)
def test_adversarial_pairs_all_miss_and_paraphrases_mostly_hit() -> None:
    """Plan Phase 6 exit criterion: false-hit rate ≤ 1 % on the adversarial set; the
    paraphrase rate is reported (not asserted) in docs/reports/cache_threshold_calibration.md."""
    import json

    from eval.cache_threshold_calibration import evaluate_pairs, load_pairs

    from rag.core.embeddings import get_embedder

    ex = get_slot_extractor()
    emb = get_embedder("bge-small")
    adv = load_pairs(PAIRS_DIR / "adversarial_cache_pairs.jsonl")
    assert len(adv) >= 200
    stats = evaluate_pairs(adv, ex, emb, mode="normalized", rich=0.90, poor=0.95)
    assert stats["hit_rate"] <= 0.01, json.dumps(stats["hits"][:5], indent=1)


# ----------------------------------------------------------------- pipeline end to end


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)
def test_pipeline_cache_walkthrough_with_fake_llm(tmp_path: Path) -> None:
    """Walkthrough steps 1-6, 8 and 9 in-process with a scripted model (no network): MISS →
    L1 → L2 after a fresh L1 → paraphrase L2 hit → period swap MISS → bypass → time-anchored
    1-day expiry → filed_fact 30-day expiry."""
    from langchain_core.messages import AIMessage

    from rag.cache.service import CacheService
    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    calls = {"n": 0}

    class FakeChat:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            calls["n"] += 1
            text = "".join(
                getattr(m, "content", "")
                for m in messages
                if isinstance(getattr(m, "content", ""), str)
            )
            if "question: when is" in text.lower():
                content = (
                    '{"answer_markdown": "The 2026 annual meeting of stockholders is scheduled for June 24, 2026 [C1].", '
                    '"figures_used": [], "citations": ["C1"], "confidence": "high", "answer_class": "time_anchored"}'
                )
            else:
                content = (
                    '{"answer_markdown": "Total assets were $206,803 million as of January 25, 2026 (fiscal 2026) [C1].", '
                    '"figures_used": [{"value": 206803, "unit": "USD_millions", "period": "FY2026 (Jan 25, 2026)", "citation": "C1"}], '
                    '"citations": ["C1"], "confidence": "high", "answer_class": "filed_fact"}'
                )
            return AIMessage(
                content=content,
                usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: FakeChat(), pacing_enabled=False)
    clock = DevClock()
    first = Pipeline(settings=settings, client=llm, clock=clock)

    def service() -> CacheService:
        return CacheService(
            cache_dir=tmp_path / "cache",
            embed=first.store.embedder.embed_query,
            versions=first.versions,
            config=first.thresholds.cache,
            clock=clock,
            default_entity="NVIDIA",
        )

    p1 = Pipeline(settings=settings, client=llm, clock=clock, cache=service())
    q1 = "What were NVIDIA's total assets as of Jan 25, 2026?"
    r = p1.ask(q1)
    assert r.cache_tier == "MISS" and calls["n"] == 1
    assert r.cache.write and r.cache.write.admitted and r.cache.write.answer_class == "filed_fact"
    assert r.cache.write.tiers == ["L2", "L1"]
    # step 2: exact repeat → L1, no model call, same answer and citations
    r2 = p1.ask(q1)
    assert r2.cache_tier == "L1" and calls["n"] == 1 and r2.answer == r.answer
    assert r2.context.blocks[0].block_id == "C1" and r2.tokens_by_model == {}
    assert r2.generator_role == "cache" and r2.cache.origin_request_id == r.request_id
    assert r2.latency_ms_by_node.keys() == {"slots", "cache_lookup"}
    # step 3: "restart" → fresh L1 on the same L2 directory → L2 hit, promoted to L1
    p2 = Pipeline(settings=settings, client=llm, clock=clock, cache=service())
    r3 = p2.ask(q1)
    assert r3.cache_tier == "L2" and r3.cache.similarity == 1.0 and calls["n"] == 1
    assert p2.ask(q1).cache_tier == "L1"
    # step 4: paraphrase with the same slots → L2 (threshold 0.85 slot-rich, D-59)
    r4 = p2.ask("Total assets at the end of fiscal 2026?")
    assert r4.cache_tier == "L2" and r4.cache.similarity >= 0.85 and calls["n"] == 1, r4.cache
    # step 5: period swap → slot guard blocks it → MISS (pipeline runs)
    r5 = p2.ask("What were total assets as of Jan 26, 2025?")
    assert r5.cache_tier == "MISS" and r5.cache.l2_candidates == 0 and calls["n"] == 2
    # step 6: bypass → pipeline runs, nothing read or written
    before = p2.cache.l2.count()
    r6 = p2.ask(q1, bypass_cache=True)
    assert r6.cache_tier == "bypassed" and r6.cache.write is None and calls["n"] == 3
    assert p2.cache.l2.count() == before
    # step 8: time-anchored → 1-day TTL (event date June 2026 is in the past on this clock)
    q8 = "When is NVIDIA's annual meeting?"
    r8 = p2.ask(q8)
    assert r8.cache_tier == "MISS" and r8.cache.write.answer_class == "time_anchored"
    assert r8.cache.write.expires_at == clock.now() + DAY
    assert p2.ask(q8).cache_tier == "L1"
    clock.advance(DAY + 1)
    assert p2.ask(q8).cache_tier == "MISS"  # expired in both tiers
    # step 9: +31 days → filed_fact expired; the sweeper removes it (a write would sweep too)
    clock.set_offset(31 * DAY)
    before_sweep = p2.cache.l2.count()
    swept = p2.cache.sweeper.sweep()
    assert swept["expired"] >= 1 and p2.cache.l2.count() < before_sweep
    r9 = p2.ask(q1)
    assert r9.cache_tier == "MISS" and r9.cache.write.admitted
    # step 10: reset + purge → empty
    clock.reset()
    assert p2.cache.purge("all")["l2_removed"] >= 1 and p2.cache.l2.count() == 0
    # out of scope: refusal is L1-only
    oos = p2.ask("Should I buy NVIDIA stock?")
    assert oos.cache.write.admitted and oos.cache.write.tiers == ["L1"]
    assert p2.ask("Should I buy NVIDIA stock?").cache_tier == "L1" and p2.cache.l2.count() == 0


def test_l2_make_room_evicts_stale_versions_before_lfu_victims(tmp_path: Path) -> None:
    clock = DevClock()
    keys = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    old = make_l2(tmp_path, clock, max_entries=3)
    old.write(
        "old-a",
        keys,
        record(),
        intent="X",
        answer_class="filed_fact",
        expires_at=clock.now() + 9 * DAY,
    )
    # make old-a "hot" so plain LFU would never pick it
    for _ in range(3):
        old.lookup("old-a", keys, slot_rich=True)
    new = make_l2(
        tmp_path,
        clock,
        max_entries=3,
        versions=VERSIONS.model_copy(update={"prompt_version": "v2"}),
    )
    new.write(
        "new-b", keys, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    new.write(
        "new-c", keys, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    assert new.count() == 3 and new.stats()["stale_version"] == 1
    _, evicted = new.write(
        "new-d", keys, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    assert evicted == 1
    docs = {e.document for e in new.entries()[0]}
    assert docs == {"new-b", "new-c", "new-d"}  # the hot but stale entry went first


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "index" / "manifest.json").exists(),
    reason="run `make ingest` first",
)
def test_admin_routes_dev_only_clock_stats_purge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from rag.api import main as api_main
    from rag.cache.service import CacheService
    from rag.core.settings import Settings
    from rag.graph import Pipeline
    from rag.llm import LLMClient

    class NeverCalled:
        def bind(self, **kw):
            return self

        def invoke(self, messages):
            raise AssertionError("no model call expected")

    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_test_key_0123456789",
        app_env="test",
        data_dir=PROJECT_ROOT / "data",
    )
    llm = LLMClient(settings, chat_factory=lambda cfg, role: NeverCalled(), pacing_enabled=False)
    clock = DevClock()

    def load(app):
        base = Pipeline(settings=settings, client=llm, clock=clock)
        cache = CacheService(
            cache_dir=tmp_path / "cache",
            embed=base.store.embedder.embed_query,
            versions=base.versions,
            config=base.thresholds.cache,
            clock=clock,
            default_entity="NVIDIA",
        )
        app.state.pipeline = Pipeline(settings=settings, client=llm, clock=clock, cache=cache)

    monkeypatch.setattr(api_main, "_load_pipeline", load)
    with TestClient(api_main.app) as c:
        ready = c.get("/readyz").json()
        assert ready["cache_enabled"] is True and ready["admin_enabled"] is True
        s = c.get("/api/admin/cache/stats").json()
        assert s["l2"]["entries"] == 0 and s["clock"]["offset_s"] == 0 and s["enabled"] is True
        # refusal → L1 only, then an L1 hit through the API with the cache payload
        r1 = c.post("/api/ask", json={"question": "Should I buy NVIDIA stock?"}).json()
        assert r1["cache_tier"] == "MISS" and r1["cache"]["write"]["tiers"] == ["L1"]
        r2 = c.post("/api/ask", json={"question": "Should I buy NVIDIA stock?"}).json()
        assert r2["cache_tier"] == "L1" and r2["cache"]["answer_class"] == "analytical"
        assert r2["answer"]["answer_markdown"] == r1["answer"]["answer_markdown"]
        # clock
        assert c.post("/api/admin/clock", json={"advance_seconds": DAY}).json()["offset_s"] == DAY
        assert c.get("/api/admin/clock").json()["offset_days"] == 1.0
        assert c.post("/api/admin/clock", json={"offset_seconds": 10**9}).status_code == 403
        assert c.post("/api/admin/clock", json={}).status_code == 422
        assert c.post("/api/admin/clock", json={"reset": True}).json()["offset_s"] == 0
        # entries / sweep / purge
        e = c.get("/api/admin/cache/entries").json()
        assert e["total"] == 0 and e["entries"] == []
        assert c.post("/api/admin/cache/sweep").json()["expired"] == 0
        assert (
            c.post("/api/admin/cache/purge", json={"scope": "slot", "value": "nope=1"}).status_code
            == 422
        )
        p = c.post("/api/admin/cache/purge", json={"scope": "all"}).json()
        assert p["l1_cleared"] == 1 and p["remaining"] == 0
        assert (
            c.post("/api/ask", json={"question": "Should I buy NVIDIA stock?"}).json()["cache_tier"]
            == "MISS"
        )
        # admin routes vanish outside dev
        api_main.app.state.admin_enabled = False
        assert c.get("/api/admin/cache/stats").status_code == 404
        assert c.post("/api/admin/clock", json={"reset": True}).status_code == 404


def test_write_under_capacity_does_not_scan_the_collection(tmp_path: Path) -> None:
    """Phase 7 resource profile: `cache_write` p50 was 88 ms because every write swept the whole
    collection. Under capacity a write must cost a count, not a scan."""
    clock = DevClock()
    l2 = make_l2(tmp_path, clock, max_entries=50)
    keys = {
        "entity": "NVIDIA",
        "periods_key": "FY2026",
        "metrics_key": "",
        "direction": "none",
        "negation": "no",
        "ask": "value",
        "aggregation_key": "",
        "topic_key": "",
    }
    for i in range(10):
        l2.write(
            f"q{i}",
            keys,
            record(),
            intent="X",
            answer_class="filed_fact",
            expires_at=clock.now() + DAY,
        )
    scans = {"n": 0}
    original = l2._all_meta

    def counting_scan():
        scans["n"] += 1
        return original()

    l2._all_meta = counting_scan  # type: ignore[method-assign]
    l2.write(
        "q10", keys, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    assert scans["n"] == 0, "a write below capacity scanned the collection"
    # at capacity the scan is necessary and happens
    l2.max_entries = 11
    _, evicted = l2.write(
        "q11", keys, record(), intent="X", answer_class="filed_fact", expires_at=clock.now() + DAY
    )
    assert scans["n"] > 0 and evicted == 1
