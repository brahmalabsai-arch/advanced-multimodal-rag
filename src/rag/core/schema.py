"""Shared chunk schema — the `report_chunks` metadata contract (architecture §9.1).

Written by ingestion, read by retrieval, the calculator, the compressor and the UI. Chroma
metadata must be flat scalars, so `Chunk.chroma_metadata()` drops `None` values and flattens
nothing else; `Chunk.from_chroma()` reverses it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Modality = Literal["text", "table", "row_fact", "figure"]
FigureType = Literal["chart", "diagram", "table_image", "photo", "logo", "decorative"]


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    modality: Modality
    section: str
    subsection: str
    page: int
    page_end: int | None = None
    breadcrumb: str
    statement: str = "none"
    token_count: int
    corpus_version: str = ""

    # tables and row facts
    parent_id: str | None = Field(
        default=None, description="logical table id for parts / row facts"
    )
    table_part: int | None = None
    table_parts: int | None = None
    n_rows: int | None = None
    n_cols: int | None = None

    # row facts
    line_item: str | None = None
    line_item_norm: str | None = None
    line_group: str | None = None
    value_fy2026: float | None = None
    value_fy2025: float | None = None
    value_fy2024: float | None = None
    unit: str | None = None
    period_end_fy2026: str | None = None
    period_end_fy2025: str | None = None
    period_end_fy2024: str | None = None

    # figures
    figure_type: FigureType | None = None
    image_path: str | None = None
    page_image_path: str | None = None
    companion_table_id: str | None = None
    numbers_are_approximate: bool | None = None

    # text
    sentence_span_ref: str | None = Field(default=None, description="s_000812:s_000820 (inclusive)")
    element_ids: str | None = Field(default=None, description="comma-separated source element ids")


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document: str = Field(description="what gets embedded and shown: breadcrumb prefix + body")
    metadata: ChunkMetadata

    def chroma_metadata(self) -> dict[str, Any]:
        return {k: v for k, v in self.metadata.model_dump().items() if v is not None}

    @classmethod
    def from_chroma(cls, id_: str, document: str, metadata: dict[str, Any]) -> Chunk:
        return cls(id=id_, document=document, metadata=ChunkMetadata.model_validate(metadata))


class SentenceRecord(BaseModel):
    """One line of `sentences.jsonl`; offsets are into the chunk *body* (after the breadcrumb)."""

    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    chunk_id: str
    start: int
    end: int
    text: str
