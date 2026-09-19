"""Compression classifier — Stage A rules, Stage B score, Stage C learned scorer (§6.4–6.7).

No model call is made here (K1). Tables, row facts and figures are protected from LLM
compression (K2). `EXTRACT_LLM` is chosen only when the break-even test says compressing pays
for itself (K3): in *quota mode* (free tiers, no prices configured) when the expected
reduction is at least `min_reduction_for_llm` and the narrative exceeds the skip budget; in
*price mode* when the large-model input saving exceeds the small-model cost. Every decision
carries a reason string for the debug panel and the decision log (K4).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from rag.compress.features import (
    NARRATIVE_MODALITIES,
    PROTECTED_MODALITIES,
    ChunkFeatures,
    QueryFeatures,
)
from rag.core.config import CompressionThresholds, Price

Action = Literal["KEEP", "DEDUPE", "ROW_SELECT", "EXTRACT_LIGHT", "EXTRACT_LLM", "DROP"]

INTENT_WEIGHT = {
    "POINT_LOOKUP": 1.0,
    "COMPUTATION": 1.0,
    "COMPARISON_TREND": 0.7,
    "VISUAL": 0.5,
    "CROSS_SECTION": 0.4,
    "EXPLANATORY": 0.2,
}
PROMPT_OVERHEAD_TOKENS = 150  # compressor instructions around the chunk
MIN_LLM_CHUNK_TOKENS = 250
ROW_SELECT_MIN_ROWS = 12
ROW_SELECT_INTENTS = {"POINT_LOOKUP", "COMPUTATION"}
# Expected reduction from EXTRACT_LLM as a function of noise (§6.6); replaced by the running
# mean observed in the decision log once Stage C exists.
EXPECTED_REDUCTION_FACTOR = 0.8


class ChunkDecision(BaseModel):
    chunk_id: str
    action: Action
    reason: str
    stage: Literal["A", "B"] = "A"
    score: float | None = None
    features: ChunkFeatures


class CompressionDecision(BaseModel):
    query_needs_compression: bool
    mode: Literal["quota", "price"] = "quota"
    chunks: list[ChunkDecision] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after_est: int = 0
    narrative_tokens: int = 0
    skip_reason: str | None = Field(
        default=None, description="query-level reason when no compressor runs"
    )
    query: QueryFeatures


class BreakEven(BaseModel):
    ok: bool
    mode: Literal["quota", "price"]
    expected_reduction: float
    reason: str


def break_even(
    chunk_tokens: int,
    noise: float,
    *,
    narrative_tokens: int,
    thresholds: CompressionThresholds,
    small_price: Price | None = None,
    large_price: Price | None = None,
    expected_reduction: float | None = None,
) -> BreakEven:
    """§6.6. Price mode when both roles carry prices, else quota mode."""
    rho = (
        expected_reduction if expected_reduction is not None else EXPECTED_REDUCTION_FACTOR * noise
    )
    rho = max(0.0, min(1.0, rho))
    if small_price is not None and large_price is not None:
        t_in = chunk_tokens + PROMPT_OVERHEAD_TOKENS
        saving = large_price.input * rho * chunk_tokens
        cost = small_price.input * t_in + small_price.output * (1.0 - rho) * chunk_tokens
        ok = saving > cost
        return BreakEven(
            ok=ok,
            mode="price",
            expected_reduction=round(rho, 3),
            reason=f"price: saving {saving:.2f} vs cost {cost:.2f} (per-MTok units, ρ={rho:.2f})",
        )
    ok = (
        rho >= thresholds.min_reduction_for_llm and narrative_tokens > thresholds.skip_budget_tokens
    )
    return BreakEven(
        ok=ok,
        mode="quota",
        expected_reduction=round(rho, 3),
        reason=(
            f"quota: ρ={rho:.2f} {'≥' if rho >= thresholds.min_reduction_for_llm else '<'} "
            f"{thresholds.min_reduction_for_llm:.2f}, narrative {narrative_tokens} "
            f"{'>' if narrative_tokens > thresholds.skip_budget_tokens else '≤'} "
            f"{thresholds.skip_budget_tokens}"
        ),
    )


def stage_b_score(f: ChunkFeatures, q: QueryFeatures) -> tuple[float, dict[str, float]]:
    noise = 1.0 - f.relevance_density
    budget_pressure = max(0.0, min(1.0, (q.budget_ratio - 0.5) / 0.5))
    intent_weight = INTENT_WEIGHT.get(q.intent, 0.5)
    length = min(f.chunk_tokens / 400.0, 1.0)
    score = 0.45 * noise + 0.25 * length + 0.20 * budget_pressure + 0.10 * intent_weight
    return round(score, 3), {
        "noise": round(noise, 3),
        "length": round(length, 3),
        "budget_pressure": round(budget_pressure, 3),
        "intent_weight": intent_weight,
    }


# ---------------------------------------------------------------- Stage C (§6.7, learned)

# Feature order of `classifier_weights.json`; `eval/train_classifier.py` writes the file and
# must keep this list and `learned_features()` in step (a test asserts they match).
LEARNED_FEATURES = (
    "noise",
    "length",
    "budget_pressure",
    "intent_weight",
    "numeric_density",
    "rank_norm",
    "max_dup_sim",
    "n_sentences_norm",
)
WEIGHTS_FILE = Path(__file__).with_name("classifier_weights.json")


def learned_features(f: ChunkFeatures, q: QueryFeatures) -> list[float]:
    return [
        1.0 - f.relevance_density,
        min(f.chunk_tokens / 400.0, 1.0),
        max(0.0, min(1.0, (q.budget_ratio - 0.5) / 0.5)),
        INTENT_WEIGHT.get(q.intent, 0.5),
        f.numeric_density,
        min(f.rank / 10.0, 1.0),
        max(0.0, f.max_dup_sim),
        min(f.n_sentences / 20.0, 1.0),
    ]


class LearnedScorer(BaseModel):
    """Exported logistic weights scored with NumPy — scikit-learn is never imported at serving
    time (P7). Trained and evaluated by `eval/train_classifier.py` (§6.7)."""

    version: int = 1
    features: list[str]
    coef: list[float]
    intercept: float
    decision_threshold: float = 0.5
    trained_at: str | None = None
    n_samples: int | None = None
    n_positive: int | None = None
    seed: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if tuple(self.features) != LEARNED_FEATURES:
            raise ValueError(f"weights file features {self.features} != {list(LEARNED_FEATURES)}")
        if len(self.coef) != len(self.features):
            raise ValueError("coef and features lengths differ")

    def probability(self, f: ChunkFeatures, q: QueryFeatures) -> float:
        x = np.asarray(learned_features(f, q), dtype=np.float64)
        z = float(np.dot(np.asarray(self.coef, dtype=np.float64), x)) + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))

    def score(self, f: ChunkFeatures, q: QueryFeatures) -> tuple[float, dict[str, float]]:
        """Same contract as `stage_b_score`: a 0-1 score plus its parts, so the two are
        interchangeable behind `compression.stage_b`."""
        x = learned_features(f, q)
        p = self.probability(f, q)
        return round(p, 3), {name: round(v, 3) for name, v in zip(LEARNED_FEATURES, x, strict=True)}


@lru_cache(maxsize=1)
def load_learned_scorer(path: Path | None = None) -> LearnedScorer | None:
    """`None` when no weights have been exported — the caller falls back to the rules."""
    p = Path(path) if path is not None else WEIGHTS_FILE
    if not p.exists():
        return None
    return LearnedScorer.model_validate_json(p.read_text(encoding="utf-8"))


def classify(
    query: QueryFeatures,
    chunks: list[ChunkFeatures],
    *,
    thresholds: CompressionThresholds,
    drop_floor_logit: float,
    min_keep: int,
    small_price: Price | None = None,
    large_price: Price | None = None,
    expected_reduction: float | None = None,
    scorer: LearnedScorer | None = None,
) -> CompressionDecision:
    mode: Literal["quota", "price"] = (
        "price" if small_price is not None and large_price is not None else "quota"
    )
    decisions: dict[str, ChunkDecision] = {}
    narrative_ids = {f.chunk_id for f in chunks if f.modality in NARRATIVE_MODALITIES}
    if query.intent == "OUT_OF_SCOPE":
        return CompressionDecision(
            query_needs_compression=False,
            mode=mode,
            skip_reason="R0: out of scope — no retrieval, no compression",
            query=query,
        )

    # ---- Stage A, chunk rules R1–R4 ------------------------------------------------
    narrative: list[ChunkFeatures] = []
    for f in chunks:
        # R1 is for repeated disclosures (the 5x H20 paragraph). Row facts and table parts of
        # one statement embed within 0.95 of each other because they share the statement
        # boilerplate, yet each carries a different number — so only narrative text is deduped,
        # and only against another narrative chunk.
        if (
            f.modality in NARRATIVE_MODALITIES
            and f.max_dup_sim >= thresholds.dedupe_cosine
            and f.dup_of
            and f.dup_of in narrative_ids
        ):
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id,
                action="DEDUPE",
                reason=f"R1: duplicate of {f.dup_of} (cosine {f.max_dup_sim:.2f})",
                features=f,
            )
            continue
        if f.score_kind == "rerank" and f.rerank_score < drop_floor_logit and f.rank > min_keep:
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id,
                action="DROP",
                reason=(
                    f"R2: rerank logit {f.rerank_score:.2f} < floor {drop_floor_logit} "
                    f"at rank {f.rank}"
                ),
                features=f,
            )
            continue
        if f.modality in PROTECTED_MODALITIES:
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id,
                action="KEEP",
                reason=f"R3: {f.modality} is protected",
                features=f,
            )
            continue
        if f.modality == "table":
            rows = f.table_rows or 0
            if query.intent in ROW_SELECT_INTENTS and rows > ROW_SELECT_MIN_ROWS:
                decisions[f.chunk_id] = ChunkDecision(
                    chunk_id=f.chunk_id,
                    action="ROW_SELECT",
                    reason=(
                        f"R4: {rows}-row table, intent {query.intent} → header + query metric "
                        "rows + totals"
                    ),
                    features=f,
                )
            else:
                decisions[f.chunk_id] = ChunkDecision(
                    chunk_id=f.chunk_id,
                    action="KEEP",
                    reason=f"R4: table kept ({rows} rows, intent {query.intent})",
                    features=f,
                )
            continue
        if f.modality in NARRATIVE_MODALITIES:
            narrative.append(f)
        else:
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id, action="KEEP", reason="unknown modality kept", features=f
            )

    # ---- Stage A, query rule R5 ------------------------------------------------------
    narrative_tokens = sum(f.chunk_tokens for f in narrative)
    skip_reason: str | None = None
    if narrative_tokens <= thresholds.skip_budget_tokens:
        skip_reason = (
            f"R5: narrative {narrative_tokens} tokens ≤ skip budget {thresholds.skip_budget_tokens}"
        )
    elif query.budget_ratio <= thresholds.narrative_pressure_ratio:
        skip_reason = (
            f"R5: budget ratio {query.budget_ratio:.2f} ≤ {thresholds.narrative_pressure_ratio} "
            "(context fits; narrative kept)"
        )
    if skip_reason:
        for f in narrative:
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id, action="KEEP", reason=skip_reason, features=f
            )
    else:
        # ---- Stage B --------------------------------------------------------------
        for f in narrative:
            # Stage C swaps the hand-set weights for the learned ones; Stage A's rules and R6
            # stay hard overrides either way (§6.7).
            if scorer is not None:
                score, parts = scorer.score(f, query)
                detail = (
                    f"learned p={score:.2f} (noise {parts['noise']:.2f}, "
                    f"len {parts['length']:.2f}, pressure {parts['budget_pressure']:.2f}, "
                    f"intent {parts['intent_weight']:.1f})"
                )
            else:
                score, parts = stage_b_score(f, query)
                detail = (
                    f"score {score:.2f} (noise {parts['noise']:.2f}, len {parts['length']:.2f}, "
                    f"pressure {parts['budget_pressure']:.2f}, intent {parts['intent_weight']:.1f})"
                )
            capped = f.numeric_density > thresholds.numeric_density_cap  # R6
            if score < thresholds.stage_b_thresholds.keep:
                action: Action = "KEEP"
                reason = f"B: {detail} < keep threshold {thresholds.stage_b_thresholds.keep}"
            elif score < thresholds.stage_b_thresholds.llm:
                action = "EXTRACT_LIGHT"
                reason = f"B: {detail} → sentence selection"
            else:
                be = break_even(
                    f.chunk_tokens,
                    parts["noise"],
                    narrative_tokens=narrative_tokens,
                    thresholds=thresholds,
                    small_price=small_price,
                    large_price=large_price,
                    expected_reduction=expected_reduction,
                )
                if capped:
                    action = "EXTRACT_LIGHT"
                    reason = (
                        f"R6: numeric density {f.numeric_density:.2f} > "
                        f"{thresholds.numeric_density_cap} caps at EXTRACT_LIGHT ({detail})"
                    )
                elif f.chunk_tokens >= MIN_LLM_CHUNK_TOKENS and be.ok:
                    action = "EXTRACT_LLM"
                    reason = (
                        f"B: {detail}, {f.chunk_tokens} tok ≥ {MIN_LLM_CHUNK_TOKENS}, "
                        f"break-even ok ({be.reason})"
                    )
                else:
                    action = "EXTRACT_LIGHT"
                    why = (
                        f"{f.chunk_tokens} tok < {MIN_LLM_CHUNK_TOKENS}"
                        if f.chunk_tokens < MIN_LLM_CHUNK_TOKENS
                        else f"break-even failed ({be.reason})"
                    )
                    reason = f"B: {detail} but {why} → sentence selection"
            decisions[f.chunk_id] = ChunkDecision(
                chunk_id=f.chunk_id,
                action=action,
                reason=reason,
                stage="B",
                score=score,
                features=f,
            )

    ordered = [decisions[f.chunk_id] for f in chunks if f.chunk_id in decisions]
    est_after = 0
    for d in ordered:
        t = d.features.chunk_tokens
        if d.action in {"DEDUPE", "DROP"}:
            continue
        if d.action == "ROW_SELECT":
            est_after += int(t * 0.5)
        elif d.action == "EXTRACT_LIGHT":
            est_after += int(t * max(0.3, d.features.relevance_density + 0.2))
        elif d.action == "EXTRACT_LLM":
            est_after += int(
                t * (1.0 - EXPECTED_REDUCTION_FACTOR * (1.0 - d.features.relevance_density))
            )
        else:
            est_after += t
    return CompressionDecision(
        query_needs_compression=any(d.action != "KEEP" for d in ordered),
        mode=mode,
        chunks=ordered,
        tokens_before=query.total_tokens,
        tokens_after_est=est_after,
        narrative_tokens=narrative_tokens,
        skip_reason=skip_reason if not any(d.action != "KEEP" for d in ordered) else None,
        query=query,
    )
