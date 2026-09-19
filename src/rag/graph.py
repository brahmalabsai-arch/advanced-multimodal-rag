"""Query pipeline as a LangGraph state graph (architecture §4.1, §8.3).

Phase 6 flow:

    slots → cache_lookup ─┬─ L1 / L2 hit ──────────────────────────────────────────→ END
                          └─ miss / bypass → scope_gate ─┬─ OUT_OF_SCOPE ─────→ cache_write → END
                                                         └─ in scope → analyze → retrieve
                                → rerank → compress → calculate → assemble → generate
                                → verify ─┬─ pass ───→ cache_write → END
                                 ▲        └─ fail (1×) ┘

Nodes are plain functions over `PipelineState`. The two cache nodes wrap the Phase 4/5 core:
`cache_lookup` needs the slots (L1 key, slot guard) so it sits after `slots`; `cache_write`
applies the admission policy after verification. `bypass_cache` skips both. Every node records
its latency; the `Pipeline` wrapper writes the trace line and returns a `PipelineResult`.

Degrade mode (Phase 8, architecture §13): when the generator's model call fails after retries
(Groq 429 past the retry budget, a provider 400, an outage), `generate` itself returns a
retrieval-only view — the top context blocks with their citations — flagged `degraded`, and the
graph routes straight to `cache_write`, which never admits it (§5.6 rule 1). Because the failure
is handled *inside* the graph, every node that did run keeps its latency and the cache lookup
that preceded generation stands: a cached answer, when one exists, is served before the model is
ever called. `Pipeline._degraded` remains as the outer fallback for a model error raised
anywhere else.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag.cache.records import CachedAnswer, CacheHit, CacheInfo, CacheWrite
from rag.cache.service import CacheService
from rag.cache.versions import VersionKeys, compute_version_keys
from rag.calc.calculator import (
    CalculationResult,
    Calculator,
    RowFactSource,
    select_formulas,
    select_formulas_from_slots,
)
from rag.compress.pipeline import CompressionOutcome, Compressor, DecisionLog
from rag.core.clock import Clock, get_clock
from rag.core.config import (
    ModelsConfig,
    ThresholdsConfig,
    load_models_config,
    load_thresholds_config,
)
from rag.core.ledger import UsageLedger
from rag.core.logging import get_logger
from rag.core.settings import Settings, get_settings
from rag.core.traces import Trace, TraceWriter, rss_mb
from rag.llm import LLMCallError, LLMClient, LLMError
from rag.query.analyze import Analysis, ExpansionSettings, analyze
from rag.query.assemble import AssembledContext, assemble_context, block_body
from rag.query.generate import Answer, generate_answer
from rag.query.rerank import RerankDecision, Reranker, rerank_node
from rag.query.retrieve import RetrievalResult, RetrievalSettings, hybrid_retrieve
from rag.query.scope import ScopeDecision, scope_gate
from rag.query.slots import QuerySlots, SlotExtractor, get_slot_extractor
from rag.query.store import IndexStore, get_store
from rag.query.verify import VerifyResult, verify_answer

log = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 2


class PipelineState(TypedDict, total=False):
    question: str
    request_id: str
    bypass_cache: bool
    slots: QuerySlots
    cache_hit: CacheHit | None
    cache_info: CacheInfo
    scope: ScopeDecision
    analysis: Analysis
    retrieval: RetrievalResult
    rerank: RerankDecision
    compression: CompressionOutcome
    calculations: list[CalculationResult]
    context: AssembledContext
    answer: Answer
    generator_role: str
    verify: VerifyResult
    attempts: int
    retry_note: str | None
    latency_ms_by_node: dict[str, int]
    error: str | None
    degraded: bool


class PipelineResult(BaseModel):
    request_id: str
    question: str
    answer: Answer
    intent: str
    intent_rule: str | None = None
    slots: QuerySlots
    scope: ScopeDecision
    analysis: Analysis
    retrieval: RetrievalResult
    rerank: RerankDecision
    compression: CompressionOutcome = Field(default_factory=CompressionOutcome)
    calculations: list[CalculationResult] = Field(default_factory=list)
    context: AssembledContext
    verify: VerifyResult
    generation_attempts: int
    generator_role: str
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    latency_ms_by_node: dict[str, int] = Field(default_factory=dict)
    total_latency_ms: int
    cache_tier: str = "bypassed"
    cache: CacheInfo = Field(default_factory=CacheInfo)
    corpus_version: str
    warning: str | None = None
    degraded: bool = Field(
        default=False,
        description="the model call failed and the answer is a retrieval-only view (§13)",
    )


def _timed(state: PipelineState, node: str, started: float) -> dict[str, int]:
    timings = dict(state.get("latency_ms_by_node", {}))
    timings[node] = timings.get(node, 0) + int((time.perf_counter() - started) * 1000)
    return timings


_EMPTY_RETRIEVAL = RetrievalResult(queries=[], candidates=[])

DEGRADED_TOP_BLOCKS = 5
_EXCERPT_CHARS = 220


def degraded_answer(context: AssembledContext, error: LLMError) -> Answer:
    """Retrieval-only view served when no model is available (architecture §13).

    The answer lists the top context blocks with their citation ids, source and page, plus a
    short verbatim excerpt, so the reader can still open the cited pages from the chips. Every
    excerpt is copied from the context, never generated; the block ids go into `citations` so
    the UI renders them. Confidence is `low` and the result is never admitted to the cache.
    """
    if isinstance(error, LLMCallError) and error.rate_limited:
        why = (
            "The model provider is rate-limiting this account (HTTP 429) and the retry budget "
            "is spent, so no generated answer is available right now."
        )
    else:
        why = "The model call failed, so no generated answer is available."
    blocks = [b for b in context.blocks if b.modality != "calculation"][:DEGRADED_TOP_BLOCKS]
    lines = [why, "", "The most relevant passages retrieved for this question:", ""]
    for b in blocks:
        source = b.breadcrumb or b.section or ""
        where = " · ".join(p for p in (source, f"PDF p.{b.page}" if b.page else "") if p)
        excerpt = " ".join(b.text.split())
        if len(excerpt) > _EXCERPT_CHARS:
            excerpt = excerpt[:_EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"
        lines.append(f"- **[{b.block_id}]** {where} ({b.modality}) — {excerpt}")
    calcs = [b for b in context.blocks if b.modality == "calculation"]
    if calcs:
        lines += ["", "Deterministic calculations already available (no model needed):", ""]
        lines += [
            f"- **[{b.block_id}]** {' '.join(b.text.split())[:_EXCERPT_CHARS]}" for b in calcs
        ]
    lines += [
        "",
        "Try again in a minute, or ask a question that was answered before — cached answers "
        "do not need the model.",
    ]
    return Answer(
        answer_markdown="\n".join(lines),
        citations=[b.block_id for b in blocks] + [b.block_id for b in calcs],
        confidence="low",
        answer_class="analytical",
    )


class Pipeline:
    def __init__(
        self,
        *,
        store: IndexStore | None = None,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        thresholds: ThresholdsConfig | None = None,
        models: ModelsConfig | None = None,
        trace_writer: TraceWriter | None = None,
        extractor: SlotExtractor | None = None,
        reranker: Reranker | None = None,
        compression_mode: str | None = None,
        context_budget: int | None = None,
        cache: CacheService | None = None,
        clock: Clock | None = None,
        cache_enabled: bool | None = None,
    ):
        self.settings = settings or get_settings()
        self.thresholds = thresholds or load_thresholds_config(settings=self.settings)
        self.models = models or load_models_config(settings=self.settings)
        self.store = store or get_store()
        self.client = client or LLMClient(self.settings, self.models)
        self.extractor = extractor or get_slot_extractor()
        self.reranker = reranker
        self.calculator = Calculator(RowFactSource(self.store.chunks.values()))
        self.trace_writer = trace_writer or TraceWriter(self.settings.logs_dir / "traces.jsonl")
        self.ledger = UsageLedger(self.settings.logs_dir / "llm_usage.jsonl")
        r = self.thresholds.retrieval
        self.retrieval_settings = RetrievalSettings(
            dense_top_k=r.dense_top_k,
            sparse_top_k=r.sparse_top_k,
            rrf_k=r.rrf_k,
            final_k=r.final_k,
            pool_k=self.thresholds.rerank.candidates,
        )
        e = self.thresholds.expansion
        self.expansion_settings = ExpansionSettings(
            max_queries=r.max_retrieval_queries,
            max_paraphrases=e.max_paraphrases,
            hyde_enabled=e.hyde,
            hyde_regenerate=e.hyde_regenerate,
            llm_fallback_enabled=e.llm_fallback,
            min_rule_confidence=e.min_rule_confidence,
            glossary_enabled=e.glossary,
            filters_enabled=e.filters,
        )
        # `context_budget` overrides the profile value for evaluations that must compare
        # against the production (Groq, 2,500-token) budget on another provider.
        self.context_budget = context_budget or self.models.active().context_budget_tokens
        profile = self.models.active()
        self.compression_mode = compression_mode or self.thresholds.compression.mode
        self.compressor = Compressor(
            self.store,
            self.thresholds,
            small_price=profile.small.price_usd_per_mtok,
            large_price=profile.large.price_usd_per_mtok,
            body_of=block_body,
        )
        self.decision_log = DecisionLog(self.settings.logs_dir / "compression_decisions.jsonl")
        # Phase 6: two-tier cache. Version keys are computed once per process (§5.7); a config
        # edit therefore needs a restart to take effect — as the walkthrough (step 7) assumes.
        self.clock = clock or get_clock()
        self.cache_enabled = (
            self.thresholds.cache.enabled if cache_enabled is None else cache_enabled
        )
        self.versions: VersionKeys = compute_version_keys(
            corpus_version=self.store.corpus_version,
            embedder_alias=self.store.manifest.embedder_alias,
            thresholds=self.thresholds,
            models=self.models,
            settings=self.settings,
        )
        self.cache = cache or CacheService(
            cache_dir=self.settings.data_dir / "cache",
            embed=self.store.embedder.embed_query,
            versions=self.versions,
            config=self.thresholds.cache,
            clock=self.clock,
            default_entity=self.extractor.calendar.entity,
        )
        self.graph = self._build()

    # -- nodes ------------------------------------------------------------------------

    def node_slots(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        slots = self.extractor.extract(state["question"])
        return {"slots": slots, "latency_ms_by_node": _timed(state, "slots", t0)}

    def node_cache_lookup(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        if state.get("bypass_cache") or not self.cache_enabled:
            info = CacheInfo(tier="bypassed", clock_offset_s=self.clock.offset_s)
            return {
                "cache_hit": None,
                "cache_info": info,
                "latency_ms_by_node": _timed(state, "cache_lookup", t0),
            }
        hit, info = self.cache.lookup(state["question"], state["slots"])
        out: dict[str, Any] = {"cache_hit": hit, "cache_info": info}
        if hit is not None:
            out.update(self._state_from_hit(hit))
        out["latency_ms_by_node"] = _timed(state, "cache_lookup", t0)
        return out

    def _state_from_hit(self, hit: CacheHit) -> dict[str, Any]:
        """Fill the downstream state from a cached record so `ask()` renders a hit exactly
        like a fresh answer (context blocks for citation chips, calculator rows, verifier)."""
        rec = hit.record
        return {
            "scope": ScopeDecision(in_scope=rec.intent != "OUT_OF_SCOPE", score=1.0, rule="cache"),
            "analysis": Analysis(
                intent=rec.intent,  # type: ignore[arg-type]
                rule=rec.intent_rule or "cache",
                confidence=1.0,
                retrieval_queries=[],
            ),
            "retrieval": _EMPTY_RETRIEVAL,
            "rerank": RerankDecision(applied=False, skip_reason="cache hit", gate="cache"),
            "compression": CompressionOutcome(enabled=False, mode="cache"),
            "calculations": list(rec.calculations),
            "context": AssembledContext(
                blocks=list(rec.context_blocks),
                image_chunk_ids=list(rec.image_chunk_ids),
                token_budget=self.context_budget,
                tokens_used=sum(b.token_count for b in rec.context_blocks),
            ),
            "answer": rec.answer.model_copy(deep=True),
            "generator_role": "cache",
            "verify": rec.verify,
            "attempts": 0,
        }

    @staticmethod
    def route_after_cache(state: PipelineState) -> str:
        return END if state.get("cache_hit") is not None else "scope"

    def node_cache_write(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = state.get("cache_info") or CacheInfo(tier="bypassed")
        if state.get("bypass_cache") or not self.cache_enabled:
            return {"latency_ms_by_node": _timed(state, "cache_write", t0)}
        if state.get("error"):
            # Degraded view (§13): recorded as a refused write so the cache panel says why.
            info = info.model_copy(
                update={
                    "write": CacheWrite(
                        admitted=False, reasons=["R1 degraded answer (model call failed)"]
                    )
                }
            )
            return {"cache_info": info, "latency_ms_by_node": _timed(state, "cache_write", t0)}
        answer = state["answer"]
        verify = state["verify"]
        analysis = state["analysis"]
        decision = self.cache.admit(
            question=state["question"],
            intent=analysis.intent,
            verify_passed=verify.passed,
            confidence=answer.confidence,
            citations=list(verify.citations_found or answer.citations),
        )
        record = CachedAnswer(
            question=state["question"],
            answer=answer,
            intent=analysis.intent,
            intent_rule=analysis.rule,
            context_blocks=list(state["context"].blocks),
            image_chunk_ids=list(state["context"].image_chunk_ids),
            calculations=list(state.get("calculations", [])),
            verify=verify,
            generator_role=state.get("generator_role", "-"),
            origin_request_id=state["request_id"],
            tokens_saved_est=self._tokens_this_request(state["request_id"]),
        )
        try:
            write = self.cache.write(state["question"], state["slots"], record, decision=decision)
        except Exception as exc:  # a cache write failure must never fail the answer
            log.warning("cache write failed for %s: %s", state["request_id"], exc)
            write = CacheWrite(admitted=False, reasons=[f"write failed: {exc}"[:200]])
        info = info.model_copy(update={"write": write})
        return {"cache_info": info, "latency_ms_by_node": _timed(state, "cache_write", t0)}

    def _tokens_this_request(self, request_id: str) -> int:
        try:
            return sum(
                r.tokens_in + r.tokens_out
                for r in self.ledger.read_all()
                if r.request_id == request_id
            )
        except Exception:
            return 0

    def node_scope(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        decision = scope_gate(
            state["question"],
            state["slots"],
            client=self.client if self.thresholds.expansion.llm_fallback else None,
            request_id=state["request_id"],
        )
        out: dict[str, Any] = {"scope": decision}
        if not decision.in_scope:
            # Scoped refusal: no retrieval, no generation, nothing to verify (§4.3).
            out.update(
                analysis=Analysis(
                    intent="OUT_OF_SCOPE",
                    rule=f"scope:{decision.rule or decision.source}",
                    confidence=decision.score,
                    source=decision.source,
                    llm_called=decision.llm_called,
                    retrieval_queries=[],
                ),
                retrieval=_EMPTY_RETRIEVAL,
                rerank=RerankDecision(applied=False, skip_reason="out of scope", gate="scope"),
                compression=CompressionOutcome(enabled=False, mode="scope"),
                calculations=[],
                context=AssembledContext(
                    blocks=[], token_budget=self.context_budget, tokens_used=0
                ),
                answer=Answer(
                    answer_markdown=decision.refusal_markdown or "",
                    confidence="high",
                    answer_class="analytical",
                ),
                generator_role="-",
                verify=VerifyResult(passed=True, issues=[]),
                attempts=0,
            )
        out["latency_ms_by_node"] = _timed(state, "scope", t0)
        return out

    @staticmethod
    def route_after_scope(state: PipelineState) -> str:
        return "cache_write" if not state["scope"].in_scope else "analyze"

    def node_analyze(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        analysis = analyze(
            state["question"],
            state["slots"],
            client=self.client,
            request_id=state["request_id"],
            settings=self.expansion_settings,
            extractor=self.extractor,
            formulas=self.calculator.config,
        )
        return {"analysis": analysis, "latency_ms_by_node": _timed(state, "analyze", t0)}

    def node_retrieve(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = state["analysis"]
        result = hybrid_retrieve(
            self.store, a.queries, settings=self.retrieval_settings, intent=a.intent
        )
        return {"retrieval": result, "latency_ms_by_node": _timed(state, "retrieve", t0)}

    def required_metrics(self, slots: QuerySlots, intent: str) -> list[str]:
        """Line items whose row facts must be present for the S1 gate: formula inputs for a
        COMPUTATION, the question's metrics otherwise."""
        if intent == "COMPUTATION" and slots.formulas:
            items: list[str] = []
            for fid in slots.formulas:
                f = self.calculator.config.formulas.get(fid)
                if f is None:
                    continue
                for spec in f.inputs.values():
                    if "{metric}" not in spec.line_item and spec.line_item not in items:
                        items.append(spec.line_item)
            return items
        return list(slots.metrics)

    def node_rerank(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = state["analysis"]
        retrieval, decision = rerank_node(
            self.store,
            state["retrieval"],
            state["question"],
            intent=a.intent,
            required_metrics=self.required_metrics(state["slots"], a.intent),
            thresholds=self.thresholds.rerank,
            final_k=self.retrieval_settings.final_k,
            reranker=self.reranker,
        )
        return {
            "retrieval": retrieval,
            "rerank": decision,
            "latency_ms_by_node": _timed(state, "rerank", t0),
        }

    def node_compress(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = state["analysis"]
        if not self.thresholds.compression.enabled:
            outcome = CompressionOutcome(enabled=False, mode="disabled")
        else:
            outcome = self.compressor.run(
                state["retrieval"].candidates,
                queries=a.retrieval_queries or [state["question"]],
                question=state["question"],
                intent=a.intent,
                metrics=self.required_metrics(state["slots"], a.intent),
                context_budget=self.context_budget,
                client=self.client,
                request_id=state["request_id"],
                mode=self.compression_mode,
            )
        return {"compression": outcome, "latency_ms_by_node": _timed(state, "compress", t0)}

    def node_calculate(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        calcs: list[CalculationResult] = []
        if state["analysis"].intent in {"COMPUTATION", "COMPARISON_TREND"}:
            line_items = self.calculator.source.line_items()
            requests = select_formulas_from_slots(
                state["slots"], line_items, self.calculator.config
            ) or select_formulas(state["question"], line_items, self.calculator.config)
            for req in requests:
                calcs.append(
                    self.calculator.compute(
                        req.formula, fiscal_year=req.fiscal_year, metric=req.metric
                    )
                )
        return {"calculations": calcs, "latency_ms_by_node": _timed(state, "calculate", t0)}

    def node_assemble(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = state["analysis"]
        context = assemble_context(
            self.store,
            state["retrieval"].candidates,
            state.get("calculations", []),
            token_budget=self.context_budget,
            intent=a.intent,
            needs_image=a.needs_image,
            compressed=state["compression"].by_id if state.get("compression") else None,
        )
        return {"context": context, "latency_ms_by_node": _timed(state, "assemble", t0)}

    def node_generate(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        attempts = state.get("attempts", 0) + 1
        notes: list[str] = []
        slots = state.get("slots")
        if (
            slots is not None
            and slots.period_note
            and state["analysis"].intent
            in {
                "POINT_LOOKUP",
                "COMPUTATION",
                "COMPARISON_TREND",
            }
        ):
            notes.append(f"{slots.period_note}; say so in the answer.")
        try:
            answer, role = generate_answer(
                self.client,
                state["question"],
                state["context"],
                intent=state["analysis"].intent,
                request_id=state["request_id"],
                retry_note=state.get("retry_note"),
                notes=notes,
            )
        except LLMError as exc:
            # Degrade inside the graph (§13): keep every node's latency, skip verification and
            # let `cache_write` see `error` so the view is never admitted.
            error = str(exc)
            log.error(
                "generation failed for %s, serving retrieval-only view: %s",
                state["request_id"],
                error,
            )
            return {
                "answer": degraded_answer(state["context"], exc),
                "generator_role": "-",
                "attempts": attempts,
                "verify": VerifyResult(passed=False, issues=[f"model call failed: {error[:200]}"]),
                "retry_note": None,
                "error": error,
                "degraded": True,
                "latency_ms_by_node": _timed(state, "generate", t0),
            }
        return {
            "answer": answer,
            "generator_role": role,
            "attempts": attempts,
            "latency_ms_by_node": _timed(state, "generate", t0),
        }

    @staticmethod
    def route_after_generate(state: PipelineState) -> str:
        return "cache_write" if state.get("error") else "verify"

    def node_verify(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = verify_answer(
            state["answer"], state["context"], state.get("calculations", []), state["question"]
        )
        retry_note = None if result.passed else "\n".join(f"- {i}" for i in result.issues)
        return {
            "verify": result,
            "retry_note": retry_note,
            "latency_ms_by_node": _timed(state, "verify", t0),
        }

    @staticmethod
    def route_after_verify(state: PipelineState) -> str:
        if state["verify"].passed or state.get("attempts", 0) >= MAX_GENERATION_ATTEMPTS:
            return "cache_write"
        return "generate"

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("slots", self.node_slots)
        g.add_node("cache_lookup", self.node_cache_lookup)
        g.add_node("cache_write", self.node_cache_write)
        g.add_node("scope", self.node_scope)
        g.add_node("analyze", self.node_analyze)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node("rerank", self.node_rerank)
        g.add_node("compress", self.node_compress)
        g.add_node("calculate", self.node_calculate)
        g.add_node("assemble", self.node_assemble)
        g.add_node("generate", self.node_generate)
        g.add_node("verify", self.node_verify)
        g.add_edge(START, "slots")
        g.add_edge("slots", "cache_lookup")
        g.add_conditional_edges(
            "cache_lookup", self.route_after_cache, {END: END, "scope": "scope"}
        )
        g.add_conditional_edges(
            "scope", self.route_after_scope, {"cache_write": "cache_write", "analyze": "analyze"}
        )
        g.add_edge("analyze", "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "compress")
        g.add_edge("compress", "calculate")
        g.add_edge("calculate", "assemble")
        g.add_edge("assemble", "generate")
        g.add_conditional_edges(
            "generate",
            self.route_after_generate,
            {"verify": "verify", "cache_write": "cache_write"},
        )
        g.add_conditional_edges(
            "verify",
            self.route_after_verify,
            {"cache_write": "cache_write", "generate": "generate"},
        )
        g.add_edge("cache_write", END)
        return g.compile()

    # -- entry point ------------------------------------------------------------------

    def ask(
        self, question: str, *, bypass_cache: bool = False, request_id: str | None = None
    ) -> PipelineResult:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        ledger_before = self.ledger.count()
        state: PipelineState = {
            "question": question.strip(),
            "request_id": request_id,
            "bypass_cache": bypass_cache,
            "attempts": 0,
            "latency_ms_by_node": {},
        }
        error: str | None = None
        try:
            final: PipelineState = self.graph.invoke(state)
            error = final.get("error")
        except LLMError as exc:
            error = str(exc)
            log.error("Pipeline failed for %s: %s", request_id, error)
            final = self._degraded(state, exc)
        total_ms = int((time.perf_counter() - started) * 1000)

        tokens_by_model: dict[str, dict[str, int]] = {}
        for rec in self.ledger.read_all()[ledger_before:]:
            if rec.request_id != request_id:
                continue
            t = tokens_by_model.setdefault(
                rec.model, {"in": 0, "out": 0, "calls": 0, "pacing_wait_ms": 0}
            )
            t["in"] += rec.tokens_in
            t["out"] += rec.tokens_out
            t["calls"] += 1
            t["pacing_wait_ms"] += rec.pacing_wait_ms

        answer = final["answer"]
        verify = final["verify"]
        warning = None
        if not verify.passed:
            answer.confidence = "low"
            warning = "Verification failed after regeneration: " + "; ".join(verify.issues)
        if error:
            warning = f"Degraded answer (retrieval-only view, not cached): {error}"
        cache_info = final.get("cache_info") or CacheInfo(
            tier="bypassed", clock_offset_s=self.clock.offset_s
        )

        result = PipelineResult(
            request_id=request_id,
            question=state["question"],
            answer=answer,
            intent=final["analysis"].intent,
            intent_rule=final["analysis"].rule,
            slots=final["slots"],
            scope=final["scope"],
            analysis=final["analysis"],
            retrieval=final["retrieval"],
            rerank=final.get("rerank") or RerankDecision(applied=False, skip_reason="not run"),
            compression=final.get("compression") or CompressionOutcome(enabled=False, mode="-"),
            calculations=final.get("calculations", []),
            context=final["context"],
            verify=verify,
            generation_attempts=final.get("attempts", 0),
            generator_role=final.get("generator_role", "-"),
            tokens_by_model=tokens_by_model,
            latency_ms_by_node=final.get("latency_ms_by_node", {}),
            total_latency_ms=total_ms,
            cache_tier=cache_info.tier,
            cache=cache_info,
            corpus_version=self.store.corpus_version,
            warning=warning,
            degraded=bool(final.get("degraded")),
        )
        self._write_trace(result, error)
        if cache_info.tier not in {"L1", "L2"}:
            self._log_compression(result)
        return result

    def _log_compression(self, r: PipelineResult) -> None:
        try:
            large_model = self.models.active().large.model
            self.decision_log.append(
                r.request_id,
                r.question,
                r.compression,
                verify_passed=r.verify.passed,
                generation_attempts=r.generation_attempts,
                answer_confidence=r.answer.confidence,
                large_input_tokens=r.tokens_by_model.get(large_model, {}).get("in"),
            )
        except Exception as exc:  # logging must never break answering
            log.warning("compression decision log failed: %s", exc)

    def _degraded(self, state: PipelineState, error: LLMError | str) -> PipelineState:
        """Outer fallback: retrieval-only view when a model error escapes the graph (§13).

        The generator's own failure is handled inside `node_generate`; this path covers a model
        error raised anywhere else. LangGraph does not return the partial state when a node
        raises, so the LLM-free nodes are re-run here — each one timed, so the latency table
        still says what ran (the Phase 7 resource profile found the table empty on this path).
        """
        exc = error if isinstance(error, LLMError) else LLMError(str(error))
        message = str(exc)
        st: PipelineState = {
            **state,
            "latency_ms_by_node": dict(state.get("latency_ms_by_node", {})),
        }
        if not st.get("slots"):
            t0 = time.perf_counter()
            st["slots"] = self.extractor.extract(st["question"])
            st["latency_ms_by_node"] = _timed(st, "slots", t0)
        if not st.get("scope"):
            st["scope"] = ScopeDecision(in_scope=True, score=0.0, rule="degraded")
        if not st.get("analysis"):
            t0 = time.perf_counter()
            st["analysis"] = analyze(
                st["question"],
                st["slots"],
                settings=self.expansion_settings,
                extractor=self.extractor,
            )
            st["latency_ms_by_node"] = _timed(st, "analyze", t0)
        if not st.get("retrieval"):
            t0 = time.perf_counter()
            st["retrieval"] = hybrid_retrieve(
                self.store,
                st["analysis"].queries,
                settings=self.retrieval_settings,
                intent=st["analysis"].intent,
            )
            st["latency_ms_by_node"] = _timed(st, "retrieve", t0)
        if not st.get("context"):
            t0 = time.perf_counter()
            st["context"] = assemble_context(
                self.store,
                st["retrieval"].candidates,
                st.get("calculations", []),
                token_budget=self.context_budget,
            )
            st["latency_ms_by_node"] = _timed(st, "assemble", t0)
        return {
            **st,
            "answer": degraded_answer(st["context"], exc),
            "verify": VerifyResult(passed=False, issues=[f"model call failed: {message[:200]}"]),
            "generator_role": "-",
            "error": message,
            "degraded": True,
            # the lookup ran (and missed, or generation would not have been reached); the
            # degraded answer is never admitted (§5.6 rule 1)
            "cache_info": state.get("cache_info")
            or CacheInfo(
                tier="bypassed" if state.get("bypass_cache") or not self.cache_enabled else "MISS",
                clock_offset_s=self.clock.offset_s,
                write=None
                if state.get("bypass_cache") or not self.cache_enabled
                else CacheWrite(admitted=False, reasons=["R1 degraded answer (model call failed)"]),
            ),
        }

    def _write_trace(self, r: PipelineResult, error: str | None) -> None:
        try:
            self.trace_writer.append(
                Trace(
                    request_id=r.request_id,
                    model_profile=self.models.active_profile,
                    query=r.question,
                    slots=r.slots.model_dump(exclude={"matches", "normalized_text"}),
                    clock_offset_s=r.cache.clock_offset_s,
                    cache_tier=r.cache.tier,
                    cache_similarity=r.cache.similarity,
                    admitted=bool(r.cache.write and r.cache.write.admitted),
                    intent=r.intent,
                    intent_rule=r.intent_rule,
                    expansions=[f"{q.kind}: {q.text[:120]}" for q in r.analysis.queries[1:]],
                    retrieval_queries=r.retrieval.queries,
                    fused_top_ids=r.retrieval.top_ids,
                    rerank_applied=r.rerank.applied,
                    rerank_skip_reason=(
                        f"{r.rerank.gate}: {r.rerank.skip_reason}" if not r.rerank.applied else None
                    ),
                    compression_decision=r.compression.summary(),
                    context_block_ids=[b.block_id for b in r.context.blocks],
                    context_tokens=r.context.tokens_used,
                    tokens_by_model={
                        m: {"in": t["in"], "out": t["out"]} for m, t in r.tokens_by_model.items()
                    },
                    calculator_calls=[c.formula for c in r.calculations],
                    generation_attempts=r.generation_attempts,
                    verify_passed=r.verify.passed,
                    verify_issues=r.verify.issues,
                    answer_class=r.answer.answer_class,
                    confidence=r.answer.confidence,
                    latency_ms_by_node=r.latency_ms_by_node,
                    total_latency_ms=r.total_latency_ms,
                    rss_mb=rss_mb(),
                    error=error,
                    degraded=r.degraded,
                )
            )
        except Exception as exc:  # tracing must never break answering
            log.warning("trace write failed: %s", exc)
