"""Query pipeline as a LangGraph state graph (architecture §4.1, §8.3).

Phase 3 is linear with one conditional edge:

    analyze → retrieve → calculate → assemble → generate → verify ─┬─ pass ──→ END
                                          ▲                       └─ fail (1×) ┘

Nodes are plain functions over `PipelineState`; later phases add slots, the scope gate,
expansion, reranking, compression and the two cache tiers around this core. Every node records
its latency; the `Pipeline` wrapper writes the trace line and returns a `PipelineResult`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult, Calculator, RowFactSource, select_formulas
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
from rag.llm import LLMClient, LLMError
from rag.query.analyze import Analysis, analyze
from rag.query.assemble import AssembledContext, assemble_context
from rag.query.generate import Answer, generate_answer
from rag.query.retrieve import RetrievalResult, RetrievalSettings, hybrid_retrieve
from rag.query.store import IndexStore, get_store
from rag.query.verify import VerifyResult, verify_answer

log = get_logger(__name__)

MAX_GENERATION_ATTEMPTS = 2


class PipelineState(TypedDict, total=False):
    question: str
    request_id: str
    bypass_cache: bool
    analysis: Analysis
    retrieval: RetrievalResult
    calculations: list[CalculationResult]
    context: AssembledContext
    answer: Answer
    generator_role: str
    verify: VerifyResult
    attempts: int
    retry_note: str | None
    latency_ms_by_node: dict[str, int]
    error: str | None


class PipelineResult(BaseModel):
    request_id: str
    question: str
    answer: Answer
    intent: str
    intent_rule: str | None = None
    retrieval: RetrievalResult
    calculations: list[CalculationResult] = Field(default_factory=list)
    context: AssembledContext
    verify: VerifyResult
    generation_attempts: int
    generator_role: str
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    latency_ms_by_node: dict[str, int] = Field(default_factory=dict)
    total_latency_ms: int
    cache_tier: str = "bypassed"
    corpus_version: str
    warning: str | None = None


def _timed(state: PipelineState, node: str, started: float) -> dict[str, int]:
    timings = dict(state.get("latency_ms_by_node", {}))
    timings[node] = timings.get(node, 0) + int((time.perf_counter() - started) * 1000)
    return timings


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
    ):
        self.settings = settings or get_settings()
        self.thresholds = thresholds or load_thresholds_config(settings=self.settings)
        self.models = models or load_models_config(settings=self.settings)
        self.store = store or get_store()
        self.client = client or LLMClient(self.settings, self.models)
        self.calculator = Calculator(RowFactSource(self.store.chunks.values()))
        self.trace_writer = trace_writer or TraceWriter(self.settings.logs_dir / "traces.jsonl")
        self.ledger = UsageLedger(self.settings.logs_dir / "llm_usage.jsonl")
        r = self.thresholds.retrieval
        self.retrieval_settings = RetrievalSettings(
            dense_top_k=r.dense_top_k, sparse_top_k=r.sparse_top_k, rrf_k=r.rrf_k, final_k=r.final_k
        )
        self.context_budget = self.models.active().context_budget_tokens
        self.graph = self._build()

    # -- nodes ------------------------------------------------------------------------

    def node_analyze(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        analysis = analyze(state["question"])
        return {"analysis": analysis, "latency_ms_by_node": _timed(state, "analyze", t0)}

    def node_retrieve(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = state["analysis"]
        result = hybrid_retrieve(
            self.store, a.retrieval_queries, settings=self.retrieval_settings, intent=a.intent
        )
        return {"retrieval": result, "latency_ms_by_node": _timed(state, "retrieve", t0)}

    def node_calculate(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        calcs: list[CalculationResult] = []
        if state["analysis"].intent in {"COMPUTATION", "COMPARISON_TREND"}:
            line_items = self.calculator.source.line_items()
            for req in select_formulas(state["question"], line_items, self.calculator.config):
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
        )
        return {"context": context, "latency_ms_by_node": _timed(state, "assemble", t0)}

    def node_generate(self, state: PipelineState) -> dict[str, Any]:
        t0 = time.perf_counter()
        attempts = state.get("attempts", 0) + 1
        answer, role = generate_answer(
            self.client,
            state["question"],
            state["context"],
            intent=state["analysis"].intent,
            request_id=state["request_id"],
            retry_note=state.get("retry_note"),
        )
        return {
            "answer": answer,
            "generator_role": role,
            "attempts": attempts,
            "latency_ms_by_node": _timed(state, "generate", t0),
        }

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
            return END
        return "generate"

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("analyze", self.node_analyze)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node("calculate", self.node_calculate)
        g.add_node("assemble", self.node_assemble)
        g.add_node("generate", self.node_generate)
        g.add_node("verify", self.node_verify)
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "retrieve")
        g.add_edge("retrieve", "calculate")
        g.add_edge("calculate", "assemble")
        g.add_edge("assemble", "generate")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify", self.route_after_verify, {END: END, "generate": "generate"}
        )
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
        except LLMError as exc:
            error = str(exc)
            log.error("Pipeline failed for %s: %s", request_id, error)
            final = self._degraded(state, error)
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
            warning = error

        result = PipelineResult(
            request_id=request_id,
            question=state["question"],
            answer=answer,
            intent=final["analysis"].intent,
            intent_rule=final["analysis"].rule,
            retrieval=final["retrieval"],
            calculations=final.get("calculations", []),
            context=final["context"],
            verify=verify,
            generation_attempts=final.get("attempts", 0),
            generator_role=final.get("generator_role", "-"),
            tokens_by_model=tokens_by_model,
            latency_ms_by_node=final.get("latency_ms_by_node", {}),
            total_latency_ms=total_ms,
            corpus_version=self.store.corpus_version,
            warning=warning,
        )
        self._write_trace(result, error)
        return result

    def _degraded(self, state: PipelineState, error: str) -> PipelineState:
        """Retrieval-only view when the model call fails after retries (architecture §13)."""
        analysis = state.get("analysis") or analyze(state["question"])
        retrieval = state.get("retrieval") or hybrid_retrieve(
            self.store,
            analysis.retrieval_queries,
            settings=self.retrieval_settings,
            intent=analysis.intent,
        )
        context = state.get("context") or assemble_context(
            self.store,
            retrieval.candidates,
            state.get("calculations", []),
            token_budget=self.context_budget,
        )
        answer = Answer(
            answer_markdown=(
                "The model call failed, so no generated answer is available. "
                "The most relevant passages retrieved are listed in the debug panel.\n\n"
                + "\n".join(f"- {b.header}" for b in context.blocks[:5])
            ),
            confidence="low",
            answer_class="analytical",
        )
        return {
            **state,
            "analysis": analysis,
            "retrieval": retrieval,
            "context": context,
            "answer": answer,
            "verify": VerifyResult(passed=False, issues=[f"model call failed: {error[:200]}"]),
            "generator_role": "-",
        }

    def _write_trace(self, r: PipelineResult, error: str | None) -> None:
        try:
            self.trace_writer.append(
                Trace(
                    request_id=r.request_id,
                    model_profile=self.models.active_profile,
                    query=r.question,
                    cache_tier="bypassed",
                    intent=r.intent,
                    intent_rule=r.intent_rule,
                    retrieval_queries=r.retrieval.queries,
                    fused_top_ids=r.retrieval.top_ids,
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
                )
            )
        except Exception as exc:  # tracing must never break answering
            log.warning("trace write failed: %s", exc)
