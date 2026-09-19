"""Phase 7 tests — benchmark reproducibility, the learned scorer's NumPy parity, and the ops
aggregation (plan Phase 7 "Tests")."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rag.api.ops import hit_rate_over_time, ops_stats, percentile, tail_jsonl
from rag.compress.classifier import (
    LEARNED_FEATURES,
    LearnedScorer,
    learned_features,
    load_learned_scorer,
)
from rag.compress.features import ChunkFeatures, QueryFeatures
from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from cache_benchmark import (  # noqa: E402
    POLICIES,
    Identity,
    LFUParams,
    RedisLFU,
    build_log,
    build_tail_identities,
    metrics,
    prepare,
    promotion,
    simulate,
)
from train_classifier import FEATURE_NAMES, feature_vector, not_worse  # noqa: E402

THRESHOLDS = (0.85, 0.90)
LFU = LFUParams()


class FakeEmbedder:
    """Deterministic bag-of-words vectors: no ONNX model in the unit tests."""

    def embed_queries(self, texts) -> np.ndarray:  # noqa: ANN001
        rows = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for tok in str(t).lower().split():
                v[hash(tok) % 64] += 1.0
            rows.append(v)
        return np.asarray(rows, dtype=np.float32)


# ------------------------------------------------------------------ benchmark determinism


def test_query_log_is_reproducible_for_a_seed() -> None:
    a = build_log(400, seed=7)
    b = build_log(400, seed=7)
    c = build_log(400, seed=8)
    assert [(q.identity, q.text, q.ts) for q in a] == [(q.identity, q.text, q.ts) for q in b]
    assert [(q.identity, q.ts) for q in a] != [(q.identity, q.ts) for q in c]
    assert a == sorted(a, key=lambda q: q.ts)  # replayed in time order
    # the three bursts concentrate traffic into two-day windows
    h20 = [q.ts for q in a if q.identity == "topic:h20_charge"]
    assert h20 and (max(h20) - min(h20)) <= 3 * 86_400


def test_tail_generator_is_bounded_not_looping() -> None:
    """The first draft looped forever when more one-offs were asked for than exist."""
    rng = random.Random(1)
    space = build_tail_identities(rng, 10_000)
    assert 100 < len(space) < 10_000
    assert len({i.id for i in space}) == len(space)
    assert all(isinstance(i, Identity) and i.tail for i in space)
    small = build_tail_identities(random.Random(1), 20)
    assert len(small) == 20


def test_same_seed_gives_identical_metrics_every_policy() -> None:
    """Plan Phase 7: benchmark harness reproducibility."""
    log = build_log(300, seed=11)
    vecs = prepare(log, embed_mode="canonical", embedder=FakeEmbedder(), extractor=_extractor())
    for policy in POLICIES:
        cap = 10**9 if not policy.bounded else 40
        first = metrics(
            simulate(log, vecs, policy, cap, seed=11, thresholds=THRESHOLDS, lfu=LFU),
            tokens_per_call=3000,
        )
        second = metrics(
            simulate(log, vecs, policy, cap, seed=11, thresholds=THRESHOLDS, lfu=LFU),
            tokens_per_call=3000,
        )
        assert first == second, policy.name
    # a different seed changes the stochastic policies' bookkeeping, not the harness contract
    seeded = metrics(
        simulate(log, vecs, RedisLFU, 40, seed=99, thresholds=THRESHOLDS, lfu=LFU),
        tokens_per_call=3000,
    )
    assert set(seeded) == set(first)


def _extractor():  # noqa: ANN202
    from rag.query.slots import get_slot_extractor

    return get_slot_extractor()


def test_simulation_invariants_hold() -> None:
    log = build_log(300, seed=5)
    vecs = prepare(log, embed_mode="canonical", embedder=FakeEmbedder(), extractor=_extractor())
    c = simulate(log, vecs, RedisLFU, 30, seed=5, thresholds=THRESHOLDS, lfu=LFU)
    assert c.queries == len(log)
    assert c.hits + c.misses == c.queries
    assert c.correct_hits + c.false_hits == c.hits
    assert c.writes == c.misses  # every miss is admitted in the simulation
    m = metrics(c, tokens_per_call=2500)
    assert m["calls_avoided"] == c.correct_hits
    assert m["tokens_avoided"] == c.correct_hits * 2500
    # a TTL-respecting policy never serves an expired entry
    assert c.stale_hits == 0


def test_promotion_rule_needs_two_capacities() -> None:
    chosen = RedisLFU.name
    caps = [100, 500]
    close = [
        {"policy": chosen, "capacity": 100, "correct_hit_rate": 0.70},
        {"policy": chosen, "capacity": 500, "correct_hit_rate": 0.80},
        {"policy": "lru", "capacity": 100, "correct_hit_rate": 0.76},  # +6 pp, one capacity
        {"policy": "lru", "capacity": 500, "correct_hit_rate": 0.81},
    ]
    winner, notes = promotion(close, caps)
    assert winner == chosen and any("beats it at 1" in n for n in notes)
    clear = [
        *close[:2],
        {"policy": "lru", "capacity": 100, "correct_hit_rate": 0.76},
        {"policy": "lru", "capacity": 500, "correct_hit_rate": 0.86},
    ]
    assert promotion(clear, caps)[0] == "lru"


# ----------------------------------------------------------------------- Stage C parity


def test_learned_features_match_the_trainer_order() -> None:
    """The weights file is written by the trainer and read by serving; a silent reordering
    would score the wrong coefficients against the wrong features."""
    assert list(LEARNED_FEATURES) == FEATURE_NAMES
    f = ChunkFeatures(
        chunk_id="c",
        rank=3,
        modality="text",
        chunk_tokens=200,
        rerank_score=0.4,
        relevance_density=0.25,
        max_dup_sim=0.6,
        numeric_density=0.1,
        n_sentences=10,
    )
    q = QueryFeatures(
        intent="EXPLANATORY",
        n_chunks=8,
        total_tokens=3000,
        budget_ratio=0.8,
        context_budget=2500,
        queries=[],
    )
    serving = learned_features(f, q)
    training = feature_vector(f.model_dump(), q.model_dump())
    assert serving == pytest.approx(training)


def test_numpy_scorer_matches_sklearn(tmp_path: Path) -> None:
    """Plan Phase 7: the NumPy scorer must reproduce scikit-learn's predictions."""
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    X = rng.random((80, len(LEARNED_FEATURES)))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = sklearn_lm.LogisticRegression(max_iter=2000).fit(X, y)
    weights = {
        "version": 1,
        "features": list(LEARNED_FEATURES),
        "coef": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }
    path = tmp_path / "w.json"
    path.write_text(json.dumps(weights), encoding="utf-8")
    scorer = LearnedScorer.model_validate_json(path.read_text(encoding="utf-8"))
    expected = model.predict_proba(X)[:, 1]
    got = [
        1.0 / (1.0 + np.exp(-(float(np.dot(weights["coef"], row)) + weights["intercept"])))
        for row in X
    ]
    assert np.max(np.abs(expected - np.asarray(got))) < 1e-9
    # and through the serving path, on a real feature vector
    f = ChunkFeatures(
        chunk_id="c",
        rank=1,
        modality="text",
        chunk_tokens=300,
        rerank_score=0.0,
        relevance_density=0.4,
        max_dup_sim=0.2,
        numeric_density=0.05,
        n_sentences=12,
    )
    q = QueryFeatures(
        intent="POINT_LOOKUP",
        n_chunks=8,
        total_tokens=2600,
        budget_ratio=1.04,
        context_budget=2500,
        queries=[],
    )
    p = scorer.probability(f, q)
    direct = float(model.predict_proba(np.array([learned_features(f, q)]))[0, 1])
    assert p == pytest.approx(direct, abs=1e-9)
    score, parts = scorer.score(f, q)
    assert 0.0 <= score <= 1.0 and set(parts) == set(LEARNED_FEATURES)


def test_weights_file_is_loadable_and_rejects_reordered_features(tmp_path: Path) -> None:
    shipped = load_learned_scorer()
    if shipped is not None:  # the repo ships a trained file; it must match the serving order
        assert list(shipped.features) == list(LEARNED_FEATURES)
        assert len(shipped.coef) == len(LEARNED_FEATURES)
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {"features": list(reversed(LEARNED_FEATURES)), "coef": [0.0] * 8, "intercept": 0.0}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        LearnedScorer.model_validate_json(bad.read_text(encoding="utf-8"))
    missing = load_learned_scorer(tmp_path / "does-not-exist.json")
    assert missing is None


def test_label_rule_compares_answers_not_just_tokens() -> None:
    never = {"exact_match": True, "keyword_coverage": 0.8, "verify_passed": True}
    assert not_worse({"exact_match": True, "keyword_coverage": 0.9, "verify_passed": True}, never)[
        0
    ]
    assert not not_worse({"exact_match": False, "keyword_coverage": 0.9}, never)[0]
    assert not not_worse({"exact_match": True, "keyword_coverage": 0.5}, never)[0]
    assert not not_worse(
        {"exact_match": True, "keyword_coverage": 0.8, "verify_passed": False}, never
    )[0]


# ---------------------------------------------------------------------------- ops panel


def _trace(**kw: Any) -> dict[str, Any]:
    base = {
        "request_id": "r",
        "ts": "2026-09-19T00:00:00.000+00:00",
        "cache_tier": "MISS",
        "total_latency_ms": 1000,
        "latency_ms_by_node": {"retrieve": 10, "generate": 900},
        "verify_passed": True,
        "generation_attempts": 1,
        "rss_mb": 300.0,
    }
    base.update(kw)
    return base


def test_ops_stats_aggregates_logs(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    ledger = tmp_path / "usage.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    rows = [
        _trace(cache_tier="MISS", admitted=True),
        _trace(cache_tier="L1", total_latency_ms=5, latency_ms_by_node={"slots": 1}),
        _trace(cache_tier="L2", total_latency_ms=30, latency_ms_by_node={"slots": 1}),
        _trace(cache_tier="bypassed"),
        _trace(
            cache_tier="MISS",
            verify_passed=False,
            verify_issues=["bad number"],
            generation_attempts=2,
        ),
    ]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ledger.write_text(
        "\n".join(
            json.dumps(
                {
                    "role": r,
                    "model": m,
                    "tokens_in": 100,
                    "tokens_out": 20,
                    "pacing_wait_ms": 50,
                    "status": "ok",
                }
            )
            for r, m in (("large", "big"), ("large", "big"), ("small", "little"))
        )
        + "\n{ this line is not json\n",  # a partially written tail must not break the panel
        encoding="utf-8",
    )
    decisions.write_text(
        json.dumps(
            {
                "chunks": [
                    {"applied": "KEEP"},
                    {"applied": "EXTRACT_LLM", "reverted": True},
                ],
                "llm_calls": 1,
                "tokens_before": 1000,
                "tokens_after": 800,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    s = ops_stats(traces_path=traces, ledger_path=ledger, decisions_path=decisions, limit=100)
    assert s["cache"]["by_tier"] == {"MISS": 2, "L1": 1, "L2": 1, "bypassed": 1}
    assert s["cache"]["served_requests"] == 4 and s["cache"]["hit_rate"] == 0.5
    assert s["cache"]["admitted"] == 1
    assert s["models"]["by_role"]["large"]["calls"] == 2
    assert s["models"]["by_role"]["large"]["tokens_in"] == 200
    assert s["models"]["by_role"]["small"]["model"] == "little"
    assert s["quality"]["verification_failures"] == 1 and s["quality"]["regenerations"] == 1
    assert s["quality"]["last_issues"][0]["issues"] == ["bad number"]
    assert s["compression"]["actions"] == {"KEEP": 1, "EXTRACT_LLM": 1}
    assert (
        s["compression"]["fidelity_reverts"] == 1 and s["compression"]["tokens_saved_pct"] == 20.0
    )
    assert s["latency"]["cache_hit_p50"] is not None and s["latency"]["cache_miss_p50"] == 1000
    assert s["latency"]["by_node"]["generate"]["n"] == 3
    assert s["memory"]["rss_mb_peak"] == 300.0
    assert s["window"]["traces"] == 5


def test_ops_helpers_handle_empty_and_missing_inputs(tmp_path: Path) -> None:
    assert tail_jsonl(tmp_path / "nope.jsonl", 10) == []
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert hit_rate_over_time([]) == []
    s = ops_stats(
        traces_path=tmp_path / "a.jsonl",
        ledger_path=tmp_path / "b.jsonl",
        decisions_path=tmp_path / "c.jsonl",
    )
    assert s["cache"]["hit_rate"] == 0.0 and s["window"]["traces"] == 0
    buckets = hit_rate_over_time([_trace(cache_tier="L1") for _ in range(10)], buckets=5)
    assert len(buckets) == 5 and all(b["hit_rate"] == 1.0 for b in buckets)
