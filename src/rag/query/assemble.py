"""Context assembly (architecture §4.9).

Order: calculation blocks, then row facts and tables, then narrative text, then figures —
within each group by retrieval rank. Every block carries a citation header
`[C3 | Form 10-K › Item 8 › Consolidated Balance Sheets | PDF p.141 | table]`. Phase 5: the
compression node's output (`compressed`, keyed by chunk id) replaces chunk bodies, removes
dropped / de-duplicated chunks and appends merged citations to the header; the budget is then
filled by rank, so truncation only bites after compression. For VISUAL intents the figure PNGs
(max 2) are attached as images.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rag.calc.calculator import CalculationResult
from rag.core.schema import Chunk
from rag.core.tokens import count_tokens
from rag.query.retrieve import Candidate
from rag.query.store import IndexStore

MODALITY_ORDER = {"row_fact": 0, "table": 1, "text": 2, "figure": 3}
# For visual questions the figure description leads, its companion table follows.
VISUAL_MODALITY_ORDER = {"figure": 0, "table": 1, "row_fact": 2, "text": 3}
SECTION_LABELS = {
    "annual_review": "Annual Review",
    "proxy": "Proxy Statement",
    "form_10k": "Form 10-K",
    "back_cover": "Back Cover",
}
MAX_IMAGES = 2


class ContextBlock(BaseModel):
    block_id: str = Field(description="C1… for chunks, K1… for calculations")
    chunk_id: str | None = None
    modality: str
    page: int | None = None
    section: str | None = None
    breadcrumb: str = ""
    header: str
    text: str
    token_count: int
    rank: int | None = None
    rrf: float | None = None
    truncated: bool = False
    compression: str | None = Field(default=None, description="applied compressor action")
    merged_from: list[str] = Field(default_factory=list, description="de-duplicated chunk ids")


class AssembledContext(BaseModel):
    blocks: list[ContextBlock]
    dropped: list[str] = Field(default_factory=list, description="chunk ids that did not fit")
    images: list[str] = Field(default_factory=list, description="figure image paths attached")
    image_chunk_ids: list[str] = Field(default_factory=list)
    token_budget: int
    tokens_used: int

    @property
    def text(self) -> str:
        return "\n\n".join(f"{b.header}\n{b.text}" for b in self.blocks)

    def block_ids(self) -> set[str]:
        return {b.block_id for b in self.blocks}


def citation_label(chunk: Chunk) -> str:
    m = chunk.metadata
    section = SECTION_LABELS.get(m.section, m.section)
    crumb = m.breadcrumb.split(" > ", 1)[1] if " > " in m.breadcrumb else m.breadcrumb
    crumb = crumb.replace(" > ", " › ")
    return f"{section} › {crumb} | PDF p.{m.page} | {m.modality}"


def block_body(chunk: Chunk) -> str:
    """Chunk document minus its leading breadcrumb line (the header carries that)."""
    doc = chunk.document
    if doc.startswith("[") and "\n" in doc:
        first, rest = doc.split("\n", 1)
        if first.endswith("]"):
            return rest.strip()
    return doc.strip()


def assemble_context(
    store: IndexStore,
    candidates: list[Candidate],
    calculations: list[CalculationResult],
    *,
    token_budget: int,
    intent: str = "POINT_LOOKUP",
    needs_image: bool = False,
    compressed: dict | None = None,
) -> AssembledContext:
    """`compressed` maps chunk id -> `CompressedChunk` (rag.compress.compressors); chunks it
    marks dropped are skipped and its `text` replaces the chunk body."""
    blocks: list[ContextBlock] = []
    used = 0
    compressed = compressed or {}

    # Calculations first: they are small and the model must use them verbatim.
    for i, calc in enumerate(calculations, start=1):
        text = calc.as_context_block(f"K{i}")
        header, body = text.split("\n", 1)
        tokens = count_tokens(text)
        blocks.append(
            ContextBlock(
                block_id=f"K{i}",
                modality="calculation",
                header=header,
                text=body,
                token_count=tokens,
            )
        )
        used += tokens

    order = VISUAL_MODALITY_ORDER if (intent == "VISUAL" or needs_image) else MODALITY_ORDER
    ranked = sorted(
        enumerate(candidates),
        key=lambda x: (order.get(x[1].modality, 9), x[1].source == "expanded", x[0]),
    )
    dropped: list[str] = []
    cid = 0
    for rank, cand in ranked:
        chunk = store.get(cand.chunk_id)
        if chunk is None:
            continue
        comp = compressed.get(chunk.id)
        if comp is not None and comp.dropped:
            continue
        body = comp.text if comp is not None else block_body(chunk)
        header_id = f"C{cid + 1}"
        label = citation_label(chunk)
        merged = list(comp.merged_from) if comp is not None else []
        if merged:
            pages = sorted({store.get(m).metadata.page for m in merged if store.get(m)})
            label += " | also PDF p." + ", ".join(str(p) for p in pages)
        header = f"[{header_id} | {label}]"
        tokens = count_tokens(header + "\n" + body)
        if used + tokens > token_budget:
            dropped.append(cand.chunk_id)
            continue
        cid += 1
        blocks.append(
            ContextBlock(
                block_id=header_id,
                chunk_id=chunk.id,
                modality=chunk.metadata.modality,
                page=chunk.metadata.page,
                section=chunk.metadata.section,
                breadcrumb=chunk.metadata.breadcrumb,
                header=header,
                text=body,
                token_count=tokens,
                rank=rank + 1,
                rrf=cand.rrf,
                compression=comp.applied if comp is not None and comp.applied != "KEEP" else None,
                merged_from=merged,
            )
        )
        used += tokens

    images: list[str] = []
    image_chunks: list[str] = []
    if needs_image or intent == "VISUAL":
        for b in blocks:
            if b.modality != "figure" or b.chunk_id is None:
                continue
            chunk = store.get(b.chunk_id)
            path: Path | None = store.image_path(chunk.metadata.image_path) if chunk else None
            if path is not None:
                images.append(str(path))
                image_chunks.append(b.chunk_id)
            if len(images) >= MAX_IMAGES:
                break

    return AssembledContext(
        blocks=blocks,
        dropped=dropped,
        images=images,
        image_chunk_ids=image_chunks,
        token_budget=token_budget,
        tokens_used=used,
    )
