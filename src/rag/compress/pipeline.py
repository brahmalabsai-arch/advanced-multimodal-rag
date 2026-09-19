"""Compression node: features → classifier → compressors, plus the decision log
(architecture §4.7–4.8, §6.8). One `Compressor` per process; one `CompressionOutcome` per
request, exposed in the debug panel and appended to `data/logs/compression_decisions.jsonl`
once the answer is known (training data for Stage C, Phase 7).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag.compress.classifier import WEIGHTS_FILE, CompressionDecision, classify, load_learned_scorer
from rag.compress.compressors import CompressedChunk, CompressionResult, apply_compression
from rag.compress.features import FeatureExtractor, QueryFeatures
from rag.core.config import Price, ThresholdsConfig
from rag.core.logging import get_logger
from rag.llm import LLMClient
from rag.query.retrieve import Candidate
from rag.query.store import IndexStore

log = get_logger(__name__)


class CompressionOutcome(BaseModel):
    enabled: bool = True
    mode: str = "classifier"
    decision: CompressionDecision | None = None
    result: CompressionResult | None = None
    latency_ms: int = 0

    @property
    def by_id(self) -> dict[str, CompressedChunk]:
        return self.result.by_id() if self.result else {}

    def summary(self) -> dict[str, Any]:
        d, r = self.decision, self.result
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "query_needs_compression": bool(d and d.query_needs_compression),
            "skip_reason": d.skip_reason if d else None,
            "tokens_before": r.tokens_before if r else (d.tokens_before if d else 0),
            "tokens_after": r.tokens_after if r else (d.tokens_before if d else 0),
            "llm_calls": r.llm_calls if r else 0,
            "violations": r.violations if r else [],
            "actions": r.applied_actions if r else {},
            "latency_ms": self.latency_ms,
        }


class Compressor:
    def __init__(
        self,
        store: IndexStore,
        thresholds: ThresholdsConfig,
        *,
        small_price: Price | None = None,
        large_price: Price | None = None,
        body_of,  # noqa: ANN001 - Callable[[Chunk], str]
    ):
        self.store = store
        self.thresholds = thresholds
        self.features = FeatureExtractor(
            store, sentence_tau=thresholds.compression.sentence_relevance_tau
        )
        self.small_price = small_price
        self.large_price = large_price
        self.body_of = body_of
        # Stage C (§6.7): `compression.stage_b: learned` swaps the hand-set Stage B weights for
        # the exported logistic ones. Missing weights fall back to the rules with a warning, so
        # a fresh clone without a trained model still serves.
        self.scorer = None
        if thresholds.compression.stage_b == "learned":
            self.scorer = load_learned_scorer()
            if self.scorer is None:
                log.warning(
                    "compression.stage_b=learned but %s is missing; using the Stage B rules",
                    WEIGHTS_FILE,
                )

    def run(
        self,
        candidates: list[Candidate],
        *,
        queries: list[str],
        question: str,
        intent: str,
        metrics: list[str],
        context_budget: int,
        client: LLMClient | None,
        request_id: str | None,
        mode: str = "classifier",
    ) -> CompressionOutcome:
        """`mode`: classifier (production) | never | always (ablation arms: `always` routes every
        narrative chunk through EXTRACT_LLM while tables stay protected)."""
        t0 = time.perf_counter()
        if mode == "never" or not candidates:
            return CompressionOutcome(
                enabled=mode != "never",
                mode=mode,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        qf, feats, views = self.features.extract(
            candidates,
            queries,
            intent=intent,
            context_budget=context_budget,
            body_of=self.body_of,
        )
        decision = classify(
            qf,
            feats,
            thresholds=self.thresholds.compression,
            drop_floor_logit=self.thresholds.rerank.drop_floor_logit,
            min_keep=self.thresholds.rerank.min_keep,
            small_price=self.small_price,
            large_price=self.large_price,
            scorer=self.scorer,
        )
        if mode == "always":
            for d in decision.chunks:
                if d.features.modality == "text" and d.action in {"KEEP", "EXTRACT_LIGHT"}:
                    d.action = "EXTRACT_LLM"
                    d.reason = "ablation: always_compress"
            decision.query_needs_compression = any(d.action != "KEEP" for d in decision.chunks)
            decision.skip_reason = None
        result = apply_compression(
            self.store,
            decision,
            views,
            question=question,
            metrics=metrics,
            thresholds=self.thresholds.compression,
            body_of=self.body_of,
            client=client,
            request_id=request_id,
        )
        return CompressionOutcome(
            mode=mode,
            decision=decision,
            result=result,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


class DecisionLog:
    """Append-only JSONL: one line per request with features, decisions, applied actions and
    the answer outcome — the Stage C training set (§6.7)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(
        self,
        request_id: str,
        question: str,
        outcome: CompressionOutcome,
        *,
        verify_passed: bool | None,
        generation_attempts: int,
        answer_confidence: str | None,
        large_input_tokens: int | None,
    ) -> None:
        d, r = outcome.decision, outcome.result
        if d is None:
            return
        by_id = r.by_id() if r else {}
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "request_id": request_id,
            "question": question,
            "mode": outcome.mode,
            "query": d.query.model_dump(exclude={"queries"}),
            "query_needs_compression": d.query_needs_compression,
            "skip_reason": d.skip_reason,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "action": c.action,
                    "applied": by_id[c.chunk_id].applied if c.chunk_id in by_id else c.action,
                    "stage": c.stage,
                    "score": c.score,
                    "reason": c.reason,
                    "features": c.features.model_dump(exclude={"chunk_id"}),
                    "tokens_after": by_id[c.chunk_id].tokens_after if c.chunk_id in by_id else None,
                    "reverted": by_id[c.chunk_id].reverted if c.chunk_id in by_id else False,
                }
                for c in d.chunks
            ],
            "tokens_before": r.tokens_before if r else d.tokens_before,
            "tokens_after": r.tokens_after if r else d.tokens_before,
            "llm_calls": r.llm_calls if r else 0,
            "violations": r.violations if r else [],
            "outcome": {
                "verify_passed": verify_passed,
                "generation_attempts": generation_attempts,
                "confidence": answer_confidence,
                "large_input_tokens": large_input_tokens,
            },
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def query_features_of(outcome: CompressionOutcome) -> QueryFeatures | None:
    return outcome.decision.query if outcome.decision else None
