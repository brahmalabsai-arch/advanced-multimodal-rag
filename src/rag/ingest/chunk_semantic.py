"""Structure-first semantic chunking of narrative text (architecture §3.3).

Pass 1 — structural blocks: consecutive text elements bounded by headings, tables, figures and
section/subsection changes. Tables and figures never enter this chunker.

Pass 2 — inside each block: sentences → embeddings → neighbour distances → split where the
distance exceeds the *corpus-level* 90th percentile → merge pieces under MIN_TOKENS into the
neighbour with the smaller boundary distance → split pieces over MAX_TOKENS at their largest
internal distance. Every chunk gets a breadcrumb prefix and a sentence span for the sidecars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from rag.core.embeddings import Embedder
from rag.core.logging import get_logger
from rag.core.schema import Chunk, ChunkMetadata, SentenceRecord
from rag.core.tokens import count_tokens
from rag.ingest.elements import Element
from rag.ingest.sentences import normalize_whitespace, sentencize

log = get_logger(__name__)

MIN_TOKENS = 120
MAX_TOKENS = 450
SPLIT_PERCENTILE = 90.0
CHUNKING_VERSION = "semantic-v1"  # part of the ingestion config hash

TEXT_TYPES = {"text", "list_item", "footnote"}
BOUNDARY_TYPES = {"table", "picture"}
SKIP_TYPES = {
    "page_header",
    "page_footer",
    "checkbox",
    "document_index",
    "caption",
    "code",
    "formula",
}

SECTION_LABELS = {
    "annual_review": "Annual Review",
    "proxy": "Proxy Statement",
    "form_10k": "Form 10-K",
    "back_cover": "Back Cover",
}
SUBSECTION_LABELS = {
    "narrative_spread": "Narrative",
    "shareholder_letter": "Shareholder Letter",
    "proxy_front_matter": "Front Matter",
    "notice_of_meeting": "Notice of Annual Meeting",
    "proxy_summary": "Proxy Summary",
    "proposals": "Proposals",
    "corporate_governance": "Corporate Governance",
    "compensation": "Executive Compensation",
    "pay_vs_performance": "Pay Versus Performance",
    "ownership": "Stock Ownership",
    "audit_matters": "Audit Matters",
    "additional_information": "Additional Information",
    "form_10k_front_matter": "Front Matter",
    "item_1_business": "Item 1 Business",
    "item_1a_risk_factors": "Item 1A Risk Factors",
    "item_1b_unresolved_staff_comments": "Item 1B",
    "item_1c_cybersecurity": "Item 1C Cybersecurity",
    "item_2_properties": "Item 2 Properties",
    "item_3_legal_proceedings": "Item 3 Legal Proceedings",
    "item_4_mine_safety": "Item 4",
    "item_5_market": "Item 5 Market for Common Equity",
    "item_6_reserved": "Item 6",
    "item_7_mdna": "Item 7 MD&A",
    "item_7a_market_risk": "Item 7A Market Risk",
    "item_8_financials": "Item 8 Financial Statements",
    "item_9a_controls": "Item 9A Controls",
    "item_9b_other_information": "Item 9B Other Information",
    "item_11_executive_compensation": "Item 11",
    "item_15_exhibits": "Item 15 Exhibits",
    "auditor_report": "Auditor's Report",
    "notes": "Notes to Financial Statements",
    "exhibit_index": "Exhibit Index",
    "signatures": "Signatures",
    "table_of_contents": "Contents",
    "back_cover": "Back Cover",
}


# Statement headings carry the running head and the unit note; neither belongs in a breadcrumb.
_RUNNING_HEAD = re.compile(
    r"^NVIDIA Corporation and Subsidiaries\s*|\s*\((In|in) (millions|thousands|billions)[^)]*\)\s*$"
)


def pretty_subsection(tag: str) -> str:
    return SUBSECTION_LABELS.get(tag, tag.replace("_", " ").title())


def make_breadcrumb(section: str, subsection: str, heading: str | None) -> str:
    parts = [SECTION_LABELS.get(section, section), pretty_subsection(subsection)]
    if heading:
        h = _RUNNING_HEAD.sub("", normalize_whitespace(heading)).strip(" -–—:")
        if len(h) > 80:
            h = h[:80].rsplit(" ", 1)[0] + "…"
        parts.append(h)
    return " > ".join(parts)


def compute_breadcrumbs(elements: list[Element]) -> dict[str, str]:
    """Breadcrumb for every element (used by table and figure chunks as well)."""
    heading: str | None = None
    current_sub: tuple[str, str] | None = None
    out: dict[str, str] = {}
    for e in sorted(elements, key=lambda x: x.order):
        key = (e.section, e.subsection)
        if key != current_sub:
            current_sub = key
            heading = None
        if e.type == "heading" and e.text:
            heading = e.text
        out[e.element_id] = make_breadcrumb(e.section, e.subsection, heading)
    return out


# --------------------------------------------------------------------------- blocks


@dataclass
class BlockSentence:
    text: str
    page: int
    element_id: str


@dataclass
class Block:
    section: str
    subsection: str
    breadcrumb: str
    heading: str | None
    sentences: list[BlockSentence] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.sentences[0].page

    @property
    def page_end(self) -> int:
        return self.sentences[-1].page


def build_blocks(elements: list[Element]) -> list[Block]:
    blocks: list[Block] = []
    current: Block | None = None
    heading: str | None = None
    current_sub: tuple[str, str] | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current.sentences:
            blocks.append(current)
        current = None

    for e in sorted(elements, key=lambda x: x.order):
        key = (e.section, e.subsection)
        if key != current_sub:
            close()
            current_sub = key
            heading = None
        if e.type == "heading":
            close()
            heading = e.text
            continue
        if e.type in BOUNDARY_TYPES:
            close()
            continue
        if e.type in SKIP_TYPES or e.type not in TEXT_TYPES or not e.text:
            continue
        text = normalize_whitespace(e.text)
        if not text:
            continue
        if current is None:
            current = Block(
                e.section, e.subsection, make_breadcrumb(e.section, e.subsection, heading), heading
            )
        for s in sentencize(text):
            current.sentences.append(BlockSentence(s.text, e.page, e.element_id))
    close()
    return blocks


# -------------------------------------------------------------------- splitting


def _tokens_of(sentences: list[BlockSentence], breadcrumb: str) -> int:
    return count_tokens(f"[{breadcrumb}]\n" + " ".join(s.text for s in sentences))


def split_by_distance(distances: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Piece boundaries [start, end) over n = len(distances)+1 sentences."""
    n = len(distances) + 1
    pieces: list[tuple[int, int]] = []
    start = 0
    for i, d in enumerate(distances):
        if d > threshold:
            pieces.append((start, i + 1))
            start = i + 1
    pieces.append((start, n))
    return pieces


def merge_small_pieces(
    pieces: list[tuple[int, int]],
    token_of: list[int],
    distances: np.ndarray,
    min_tokens: int,
    max_tokens: int | None = None,
) -> list[tuple[int, int]]:
    """Merge pieces under `min_tokens` into the neighbour with the smaller boundary distance.

    With `max_tokens`, a merge that would exceed the cap is skipped (the other neighbour is
    tried first); a piece that fits nowhere is left small.
    """
    pieces = list(pieces)
    stuck: set[tuple[int, int]] = set()
    while len(pieces) > 1:
        sizes = [sum(token_of[a:b]) for a, b in pieces]
        small = [i for i, s in enumerate(sizes) if s < min_tokens and pieces[i] not in stuck]
        if not small:
            break
        i = min(small, key=lambda k: sizes[k])
        a, b = pieces[i]
        left_d = distances[a - 1] if i > 0 else np.inf
        right_d = distances[b - 1] if i < len(pieces) - 1 else np.inf
        options = sorted([(left_d, "left"), (right_d, "right")], key=lambda x: x[0])
        merged = False
        for _d, side in options:
            if side == "left" and i > 0:
                if max_tokens is None or sizes[i - 1] + sizes[i] <= max_tokens:
                    pieces[i - 1] = (pieces[i - 1][0], b)
                    del pieces[i]
                    merged = True
                    break
            elif (
                side == "right"
                and i < len(pieces) - 1
                and (max_tokens is None or sizes[i + 1] + sizes[i] <= max_tokens)
            ):
                pieces[i + 1] = (a, pieces[i + 1][1])
                del pieces[i]
                merged = True
                break
        if not merged:
            stuck.add((a, b))
    return pieces


def split_large_pieces(
    pieces: list[tuple[int, int]],
    token_of: list[int],
    distances: np.ndarray,
    max_tokens: int,
) -> list[tuple[int, int]]:
    """Split pieces over `max_tokens` at their largest internal distance (recursively)."""
    out: list[tuple[int, int]] = []
    stack = list(reversed(pieces))
    while stack:
        a, b = stack.pop()
        if sum(token_of[a:b]) <= max_tokens or b - a <= 1:
            out.append((a, b))
            continue
        inner = distances[a : b - 1]
        cut = a + 1 + int(np.argmax(inner))
        stack.append((cut, b))
        stack.append((a, cut))
    return out


@dataclass
class ChunkingResult:
    chunks: list[Chunk]
    sentences: list[SentenceRecord]
    threshold: float
    n_blocks: int


def chunk_text(
    elements: list[Element],
    embedder: Embedder,
    *,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    percentile: float = SPLIT_PERCENTILE,
) -> ChunkingResult:
    blocks = build_blocks(elements)
    all_sents = [s for b in blocks for s in b.sentences]
    log.info(
        "%d text blocks, %d sentences; embedding for boundary detection…",
        len(blocks),
        len(all_sents),
    )
    emb = embedder.embed_documents([s.text for s in all_sents])

    # Neighbour distances within blocks; the threshold is a corpus-level percentile.
    block_slices: list[tuple[int, int]] = []
    offset = 0
    for b in blocks:
        block_slices.append((offset, offset + len(b.sentences)))
        offset += len(b.sentences)
    block_dist: list[np.ndarray] = []
    for a, b_ in block_slices:
        if b_ - a >= 2:
            v = emb[a:b_]
            block_dist.append(1.0 - np.sum(v[:-1] * v[1:], axis=1))
        else:
            block_dist.append(np.zeros(0, dtype=np.float32))
    pooled = (
        np.concatenate([d for d in block_dist if len(d)])
        if any(len(d) for d in block_dist)
        else np.zeros(1)
    )
    threshold = float(np.percentile(pooled, percentile))
    log.info(
        "Neighbour-distance threshold (p%.0f over %d pairs) = %.4f",
        percentile,
        len(pooled),
        threshold,
    )

    chunks: list[Chunk] = []
    records: list[SentenceRecord] = []
    sentence_counter = 0
    per_block_pieces: list[list[tuple[Block, list[BlockSentence]]]] = []

    for block, dist in zip(blocks, block_dist, strict=True):
        token_of = [count_tokens(s.text) + 1 for s in block.sentences]  # +1 for the joining space
        body_budget = max_tokens - count_tokens(f"[{block.breadcrumb}]") - 1
        pieces = split_by_distance(dist, threshold) if len(dist) else [(0, len(block.sentences))]
        pieces = merge_small_pieces(pieces, token_of, dist, min_tokens, body_budget)
        pieces = split_large_pieces(pieces, token_of, dist, body_budget)
        pieces = merge_small_pieces(pieces, token_of, dist, min_tokens, body_budget)
        per_block_pieces.append([(block, block.sentences[a:b]) for a, b in pieces])

    # Cross-block merge: a whole block smaller than MIN_TOKENS joins the next block in the same
    # subsection (its heading is kept inline so nothing is lost); if that is impossible it joins
    # the previous chunk of the same subsection instead.
    merged: list[tuple[Block, list[BlockSentence]]] = []
    pending: tuple[Block, list[BlockSentence]] | None = None

    def _same_sub(a: Block, b: Block) -> bool:
        return a.section == b.section and a.subsection == b.subsection

    def _heading_lead(block: Block, sents: list[BlockSentence]) -> list[BlockSentence]:
        if not block.heading:
            return []
        return [BlockSentence(f"{block.heading}:", sents[0].page, sents[0].element_id)]

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        pb, ps = pending
        if merged and _same_sub(merged[-1][0], pb):
            prev_block, prev_sents = merged[-1]
            combined = prev_sents + _heading_lead(pb, ps) + ps
            if _tokens_of(combined, prev_block.breadcrumb) <= max_tokens:
                merged[-1] = (prev_block, combined)
                pending = None
                return
        merged.append(pending)
        pending = None

    for pieces in per_block_pieces:
        for block, sents in pieces:
            if pending is not None:
                pb, ps = pending
                if _same_sub(pb, block):
                    combined = ps + _heading_lead(block, sents) + sents
                    if _tokens_of(combined, pb.breadcrumb) <= max_tokens:
                        sents = combined
                        block = Block(
                            pb.section, pb.subsection, pb.breadcrumb, pb.heading, combined
                        )
                        pending = None
                _flush_pending()
            if _tokens_of(sents, block.breadcrumb) < min_tokens and len(pieces) == 1:
                pending = (block, sents)
            else:
                merged.append((block, sents))
    _flush_pending()

    seq_by_page: dict[int, int] = {}
    for block, sents in merged:
        body = " ".join(s.text for s in sents)
        document = f"[{block.breadcrumb}]\n{body}"
        page = sents[0].page
        seq = seq_by_page.get(page, 0)
        seq_by_page[page] = seq + 1
        chunk_id = f"text_p{page}_{seq}"

        first_sid = sentence_counter
        pos = 0
        for s in sents:
            sentence_counter += 1
            records.append(
                SentenceRecord(
                    sentence_id=f"s_{sentence_counter:06d}",
                    chunk_id=chunk_id,
                    start=pos,
                    end=pos + len(s.text),
                    text=s.text,
                )
            )
            pos += len(s.text) + 1
        span = f"s_{first_sid + 1:06d}:s_{sentence_counter:06d}"
        element_ids = ",".join(dict.fromkeys(s.element_id for s in sents))
        chunks.append(
            Chunk(
                id=chunk_id,
                document=document,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    modality="text",
                    section=block.section,
                    subsection=block.subsection,
                    page=page,
                    page_end=sents[-1].page,
                    breadcrumb=block.breadcrumb,
                    token_count=count_tokens(document),
                    sentence_span_ref=span,
                    element_ids=element_ids[:500],
                ),
            )
        )

    sizes = [c.metadata.token_count for c in chunks]
    log.info(
        "%d text chunks (tokens min/median/max = %d/%d/%d; %d under %d, %d over %d)",
        len(chunks),
        min(sizes) if sizes else 0,
        int(np.median(sizes)) if sizes else 0,
        max(sizes) if sizes else 0,
        sum(1 for s in sizes if s < min_tokens),
        min_tokens,
        sum(1 for s in sizes if s > max_tokens),
        max_tokens,
    )
    return ChunkingResult(
        chunks=chunks, sentences=records, threshold=threshold, n_blocks=len(blocks)
    )
