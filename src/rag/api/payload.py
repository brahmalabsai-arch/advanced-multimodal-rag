"""Adapter from `PipelineResult` to the payload the published page renders (F2).

`frontend/` is a public demo, not the localhost debug view: it shows one headline number with
its working, the answer prose, the figures the answer used, citation chips that open page
images, and a "method" rail of what the pipeline did. The debug payload of `routes_ask.py`
carries all of that, in a shape built for a different page. This module is the translation, and
it is deliberately one-way and total: every field the page treats as optional is omitted rather
than faked, so a degraded or cache-served answer still renders.

Shape (see `frontend/README.md` for the page's own contract):

    answer_markdown  str                       — the only required field
    headline         {value, caption, working} — present only for a successful calculation
    figures_used     [{label, value, unit}]    — the numbers the model said it used
    citations        [{label, page, refs}]     — deduplicated by page, in citation order
    confidence       high | medium | low
    degraded         bool                      — retrieval-only view (§13)
    disclaimer       str
    incomplete       [str]                     — parts of the question the answer did not cover
    figures          [{url, caption, page}]    — figure crops the answer drew on, shown under it
    trace            {cache_tier, model, total_seconds, verified, tokens, slots, steps[],
                      metrics: {cache, retrieval, context_tokens}}

Nothing provider-specific and nothing secret crosses this boundary: the trace names the model
id the answer came from, never the key or the request's headers.
"""

from __future__ import annotations

import re
from typing import Any

from rag.calc.calculator import CalculationResult
from rag.graph import PipelineResult
from rag.llm import DAILY_LIMIT_MESSAGE
from rag.query.slots import QuerySlots

DISCLAIMER = (
    "Figures are read from the filed report and checked against it. "
    "This is analysis of a public filing, not investment advice."
)

# node name in `latency_ms_by_node` → (label on the rail, kind the page styles it with)
NODE_LABELS: dict[str, tuple[str, str]] = {
    "condense": ("Understood the follow-up", "normal"),
    "slots": ("Read the question", "normal"),
    "cache_lookup": ("Checked the cache", "normal"),
    "scope": ("Checked it is answerable from the report", "checked"),
    "analyze": ("Planned the search", "normal"),
    "retrieve": ("Searched the filing", "normal"),
    "rerank": ("Re-ranked the passages", "normal"),
    "compress": ("Trimmed the context", "normal"),
    "calculate": ("Computed the ratio", "computed"),
    "assemble": ("Assembled the context", "normal"),
    "generate": ("Wrote the answer", "normal"),
    "verify": ("Verified every number", "checked"),
    "complete": ("Checked every part was answered", "checked"),
    "cache_write": ("Cached the answer", "normal"),
}
# nodes whose presence says nothing to a reader when they did no work
_QUIET_NODES = {"cache_write", "rerank", "compress", "complete", "condense"}


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}".rstrip("0").rstrip(".")


def _unit_label(unit: str) -> str:
    return {
        "USD_millions": "USD millions",
        "USD_per_share": "USD per share",
        "shares_millions": "millions of shares",
        "ratio": "",
        "percent": "%",
    }.get(unit, unit.replace("_", " "))


def headline_from(calcs: list[CalculationResult]) -> dict[str, str] | None:
    """The one number to put above the answer: the first calculation that succeeded.

    `working` is the arithmetic as it was actually performed — the formula's own expression with
    its inputs substituted — so a reader can check it without opening the debug view.
    """
    calc = next((c for c in calcs if c.status == "ok" and c.rounded is not None), None)
    if calc is None:
        return None
    period = next((i.period_end for i in calc.inputs if i.period_end), None)
    caption = calc.description if not calc.metric else f"{calc.description}: {calc.metric}"
    caption += f", as of {period} (FY{calc.fiscal_year})" if period else f" (FY{calc.fiscal_year})"
    working = calc.expression
    for i in calc.inputs:
        working = working.replace(i.name, _fmt(i.value or 0.0))
    unit = _unit_label(calc.inputs[0].unit if calc.inputs and calc.inputs[0].unit else calc.unit)
    return {
        "value": calc.formatted(),
        "caption": caption.lower() if caption.isupper() else caption,
        "working": f"{working}  {unit}" if unit else working,
    }


def _citations(
    result: PipelineResult, cited_blocks: list[Any], store: Any | None = None
) -> list[dict[str, Any]]:
    """One chip per cited page, labelled by the section it came from.

    A calculation block has no page of its own, so an answer that cites only the arithmetic
    would show no sources at all. Its inputs carry the row-fact pages they were read from, and
    those are the pages a reader needs to check the figure, so they are cited too. `refs` lists
    the context ids ("C1", "K1") behind each chip, so the page can turn the answer's inline
    markers into links to the same pages.
    """
    out: list[dict[str, Any]] = []
    by_page: dict[int, dict[str, Any]] = {}

    def add(page: int | None, label: str, ref: str) -> None:
        if page is None:
            return
        chip = by_page.get(page)
        if chip is None:
            chip = by_page[page] = {"label": label, "page": page, "refs": []}
            out.append(chip)
        if ref not in chip["refs"]:
            chip["refs"].append(ref)

    for block in cited_blocks:
        if block.modality == "calculation":
            continue
        breadcrumb = (getattr(block, "breadcrumb", "") or "").split(" > ")
        label = breadcrumb[-1] if breadcrumb and breadcrumb[-1] else (block.section or "Filing")
        if block.modality == "figure" and store is not None and block.chunk_id:
            label = _figure_caption(store.get(block.chunk_id), label)
        add(getattr(block, "page", None), label, block.block_id)

    calc_refs = {
        _calc_formula(b): b.block_id for b in result.context.blocks if b.modality == "calculation"
    }
    for calc in result.calculations:
        if calc.status != "ok":
            continue
        for inp in calc.inputs:
            add(inp.page, _page_label(inp.page, result), calc_refs.get(calc.formula, ""))
    for chip in out:
        chip["refs"] = [r for r in chip["refs"] if r]
    return out


def _calc_formula(block: Any) -> str:
    """A calculation block's header is "[K1 | CALCULATION | <formula> | FY2026]"."""
    parts = [p.strip() for p in (block.header or "").strip("[]").split("|")]
    return parts[2] if len(parts) > 2 else ""


def _page_label(page: int | None, result: PipelineResult) -> str:
    """Name a page by the statement that sits on it, so the chip reads like a filing reference.

    The context already holds blocks from that page; their breadcrumb is the statement's own
    heading. Falling back to a generic label is better than inventing a specific one.
    """
    if page is not None:
        for block in result.context.blocks:
            if getattr(block, "page", None) == page:
                tail = (getattr(block, "breadcrumb", "") or "").split(" > ")[-1]
                if tail:
                    return tail
    return "Financial statements"


_FIGURE_TITLE = re.compile(r"^Figure \S+ \([^)]*\):\s*(.+)$", re.MULTILINE)
MAX_FIGURES = 2


def _figure_caption(chunk: Any, fallback: str) -> str:
    """The figure's own title, as the parser wrote it ("Figure p3_0 (diagram, PDF p. 3): …")."""
    match = _FIGURE_TITLE.search(getattr(chunk, "document", "") or "")
    return match.group(1).strip() if match else fallback


def _figures(result: PipelineResult, cited_blocks: list[Any], store: Any | None) -> list[dict]:
    """Figure crops worth showing under the answer, strongest evidence first.

    Three tiers, and only the best non-empty one is shown: figures the answer cited; else
    figures on a page the answer cited (a narrative passage and its diagram share a page, and
    the answer often cites only the passage); else figures attached to the model as images.
    Mixing tiers put loosely related diagrams under a precise answer. Only crops that exist on
    disk are offered: the deployed index prunes the ones nothing references.
    """
    if store is None:
        return []
    cited_ids = {b.block_id for b in cited_blocks}
    cited_pages = {b.page for b in cited_blocks if getattr(b, "page", None)}
    attached = set(result.context.image_chunk_ids)
    tiers: list[list[dict]] = [[], [], []]
    seen: set[str] = set()
    for b in result.context.blocks:
        if b.modality != "figure" or not b.chunk_id:
            continue
        if b.block_id in cited_ids:
            tier = 0
        elif b.page in cited_pages:
            tier = 1
        elif b.chunk_id in attached:
            tier = 2
        else:
            continue
        chunk = store.get(b.chunk_id)
        relative = getattr(getattr(chunk, "metadata", None), "image_path", None)
        if not relative or store.image_path(relative) is None:
            continue
        figure_id = relative.rsplit("/", 1)[-1].removesuffix(".png")
        if figure_id in seen:
            continue
        seen.add(figure_id)
        tail = (b.breadcrumb or "").split(" > ")[-1]
        tiers[tier].append(
            {
                "url": f"/api/figures/{figure_id}",
                "caption": _figure_caption(chunk, tail or "Figure from the filing"),
                "page": b.page,
            }
        )
    best = next((t for t in tiers if t), [])
    return best[:MAX_FIGURES]


def _passage_label(candidate: Any, store: Any | None) -> str:
    """A row fact is named by its line item; anything else by the heading it sits under."""
    chunk = store.get(candidate.chunk_id) if store is not None else None
    line_item = getattr(getattr(chunk, "metadata", None), "line_item", None)
    if line_item:
        return str(line_item)
    return (candidate.breadcrumb or candidate.section).split(" > ")[-1]


def _metrics(result: PipelineResult, store: Any | None = None) -> dict[str, Any]:
    """The numbers behind the method rail, for the "how the agent answered" panel.

    Similarities are cosine (the store scores unit vectors); RRF is the fused reciprocal-rank
    score, so it is comparable between passages of one answer and nothing else.
    """
    cache = result.cache
    write = cache.write
    out: dict[str, Any] = {
        "cache": {
            "tier": result.cache_tier,
            "similarity": cache.similarity,
            "threshold": cache.threshold,
            "best_rejected_similarity": cache.best_rejected_similarity,
            "lookup_ms": cache.lookup_ms,
            "written": bool(write and write.admitted),
        },
        "context_tokens": result.context.tokens_used,
    }
    ranked = [c for c in result.retrieval.candidates if c.source != "expanded"]
    if ranked:
        dense = [c.dense_score for c in ranked if c.dense_score is not None]
        out["retrieval"] = {
            "queries": len(result.retrieval.queries),
            "top_similarity": max(dense) if dense else None,
            "top_rrf": round(max(c.rrf for c in ranked), 4),
            "passages": [
                {
                    "label": _passage_label(c, store),
                    "page": c.page,
                    "modality": c.modality,
                    "similarity": c.dense_score,
                    "rrf": round(c.rrf, 4),
                    "dense_rank": c.dense_rank,
                    "bm25_rank": c.bm25_rank,
                }
                for c in ranked[:5]
            ],
        }
    return out


def _slots(slots: QuerySlots) -> dict[str, str]:
    out = {
        "entity": slots.entity or "",
        "period": ", ".join(slots.fiscal_periods),
        "formula": ", ".join(slots.formulas) or ", ".join(slots.metrics),
    }
    return {k: v for k, v in out.items() if v}


def _compression_detail(result: PipelineResult) -> str:
    """What the classifier did, in the reader's terms: tokens saved, or why it stood aside."""
    s = result.compression.summary()
    before, after = s.get("tokens_before") or 0, s.get("tokens_after") or 0
    if before and after and after < before:
        return f"context trimmed {before:,} → {after:,} tokens, every number checked"
    return s.get("skip_reason") or "context left intact"


def _condense_detail(result: PipelineResult) -> str:
    """How a follow-up was read; empty for a question that did not need the conversation."""
    c = result.condense
    if c is None or not c.follow_up:
        return ""
    if c.rewritten:
        return f'read as "{c.question}"'
    if c.error:
        return "answered as typed: the follow-up could not be rewritten"
    return "already standalone"


def _coverage_detail(result: PipelineResult) -> str:
    """What the completeness check concluded, in the reader's terms."""
    c = result.coverage
    if not c.checked:
        return ""
    how = "rules and a self-check" if c.method == "rules+self_check" else "rules"
    if c.missing:
        return f"not answered: {'; '.join(c.missing)} ({how})"
    n = len(c.parts)
    if n == 0:
        return f"no separate parts to check ({how})"
    return f"{n} of {n} part{'s' if n != 1 else ''} answered ({how})"


def _steps(result: PipelineResult) -> list[dict[str, Any]]:
    """The method rail: one entry per node that ran, in pipeline order, with what it decided."""
    detail_for: dict[str, str] = {
        "slots": ", ".join(f"{k}: {v}" for k, v in _slots(result.slots).items()),
        "cache_lookup": (
            f"{result.cache_tier} hit, no model call"
            if result.cache_tier in {"L1", "L2"}
            else "no usable cached answer"
        ),
        "scope": "answerable from the filing" if result.scope.in_scope else "out of scope",
        "analyze": f"{result.intent.replace('_', ' ').lower()}; "
        f"{len(result.analysis.queries)} search quer"
        f"{'y' if len(result.analysis.queries) == 1 else 'ies'}",
        "retrieve": f"{len(result.retrieval.candidates)} passages from "
        f"{len(result.retrieval.queries)} queries",
        "rerank": result.rerank.skip_reason or "re-ordered the candidates",
        "compress": _compression_detail(result),
        "calculate": "; ".join(
            f"{c.formula} = {c.formatted()}" for c in result.calculations if c.status == "ok"
        ),
        "assemble": f"{result.context.tokens_used} tokens of context",
        "generate": f"{result.generator_role} model, "
        f"{result.generation_attempts} attempt{'s' if result.generation_attempts != 1 else ''}",
        "verify": (
            "every number traced to the filing"
            if result.verify.passed
            else "; ".join(result.verify.issues[:2])
        ),
        "complete": _coverage_detail(result),
        "condense": _condense_detail(result),
        "cache_write": (
            "answer cached for the next visitor"
            if result.cache.write and result.cache.write.admitted
            else ""
        ),
    }
    steps: list[dict[str, Any]] = []
    for node, ms in result.latency_ms_by_node.items():
        label, kind = NODE_LABELS.get(node, (node.replace("_", " "), "normal"))
        detail = detail_for.get(node, "")
        if node in _QUIET_NODES and not detail:
            continue
        step: dict[str, Any] = {"label": label, "kind": kind, "seconds": round(ms / 1000, 2)}
        if detail:
            step["detail"] = detail
        if node == "slots":
            tags = [*result.slots.fiscal_periods, *result.slots.metrics[:2]]
            if tags:
                step["tags"] = tags
        steps.append(step)
    return steps


def _figure_label(
    citation: str,
    blocks: dict[str, Any],
    store: Any | None,
    calcs: list[CalculationResult] | None = None,
) -> str:
    """Name the number the answer used: the calculation that produced it, the line item where
    the block is a row fact, else the section it came from. Falls back to the citation id, which
    is always meaningful on the page because the same id labels the chip."""
    block = blocks.get(citation)
    if block is None:
        return citation
    if block.modality == "calculation":
        formula = _calc_formula(block)
        match = next((c for c in calcs or [] if c.formula == formula), None)
        if match is not None:
            return match.description if not match.metric else f"{match.description}: {match.metric}"
        return formula.replace("_", " ").capitalize() or "Calculated"
    chunk = store.get(block.chunk_id) if store is not None and block.chunk_id else None
    line_item = getattr(getattr(chunk, "metadata", None), "line_item", None)
    if line_item:
        return str(line_item)
    tail = (block.breadcrumb or "").split(" > ")[-1]
    return tail or block.section or citation


def answer_payload(
    result: PipelineResult, cited_blocks: list[Any], store: Any | None = None
) -> dict[str, Any]:
    """Build the page's payload. `cited_blocks` are the context blocks the answer cited."""
    tokens_in = sum(t.get("in", 0) for t in result.tokens_by_model.values())
    tokens_out = sum(t.get("out", 0) for t in result.tokens_by_model.values())
    # the model that wrote the answer, not the first one in the ledger (often a support call)
    model = result.generator_model or next(iter(result.tokens_by_model), None)
    payload: dict[str, Any] = {
        "answer_markdown": result.answer.answer_markdown,
        # the page keeps this in its conversation memory, so later follow-ups build on the
        # resolved question rather than on "and gross profit?"
        "standalone_question": result.question,
        "confidence": result.answer.confidence,
        "degraded": result.degraded,
        "citations": _citations(result, cited_blocks, store),
        "trace": {
            "cache_tier": result.cache_tier if result.cache_tier in {"L1", "L2"} else "miss",
            "total_seconds": round(result.total_latency_ms / 1000, 2),
            "verified": result.verify.passed,
            "slots": _slots(result.slots),
            "steps": _steps(result),
            "metrics": _metrics(result, store),
        },
    }
    if model:
        payload["trace"]["model"] = model
    if tokens_in or tokens_out:
        payload["trace"]["tokens"] = {"in": tokens_in, "out": tokens_out}
    headline = headline_from(result.calculations)
    if headline:
        payload["headline"] = headline
    by_id = {b.block_id: b for b in result.context.blocks}
    figures = [
        {
            "label": _figure_label(f.citation, by_id, store, result.calculations),
            "value": _fmt(f.value) if isinstance(f.value, int | float) else str(f.value),
            "unit": _unit_label(f.unit),
        }
        for f in result.answer.figures_used
    ]
    if figures:
        payload["figures_used"] = figures
    figures_shown = _figures(result, cited_blocks, store)
    if figures_shown:
        payload["figures"] = figures_shown
    if result.coverage.missing:
        payload["incomplete"] = list(result.coverage.missing)
    if result.daily_quota:
        payload["quota"] = {"limit": "daily", "message": DAILY_LIMIT_MESSAGE}
    if result.answer.confidence != "low":
        payload["disclaimer"] = DISCLAIMER
    if result.answer.fiscal_year_interpretation:
        payload["trace"]["slots"]["note"] = result.answer.fiscal_year_interpretation
    return payload


__all__ = ["DISCLAIMER", "answer_payload", "headline_from"]
