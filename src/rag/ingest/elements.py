"""Phase 1 — typed elements from the Docling parse (plan: "Sections and element typing").

Every Docling item becomes one `Element` with page, bounding box (top-left origin, crop-box
points), section/subsection (from `sections.py`) and, for statement pages, the statement it
belongs to. Written to `data/parsed/elements.jsonl`; later phases (chunking, tables, figures)
read this file and never touch Docling again.

Needs `docling_core` (ingest environment). The `Element` schema itself is plain Pydantic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.ingest.sections import (
    Heading,
    Section,
    Statement,
    assign_subsections,
    detect_statement,
    page_section,
)

log = get_logger(__name__)

ElementType = Literal[
    "text",
    "heading",
    "list_item",
    "caption",
    "footnote",
    "table",
    "picture",
    "page_header",
    "page_footer",
    "document_index",
    "checkbox",
    "code",
    "formula",
    "other",
]

# Docling DocItemLabel value -> our element type
LABEL_MAP: dict[str, ElementType] = {
    "text": "text",
    "paragraph": "text",
    "section_header": "heading",
    "title": "heading",
    "list_item": "list_item",
    "caption": "caption",
    "footnote": "footnote",
    "table": "table",
    "picture": "picture",
    "chart": "picture",
    "page_header": "page_header",
    "page_footer": "page_footer",
    "document_index": "document_index",
    "checkbox_selected": "checkbox",
    "checkbox_unselected": "checkbox",
    "code": "code",
    "formula": "formula",
}

# A statement heading must sit in the top part of the page to mark that page as a statement.
STATEMENT_HEADING_MAX_TOP_FRACTION = 0.35


class Element(BaseModel):
    element_id: str = Field(description="stable id derived from the Docling ref, e.g. tables_64")
    ref: str = Field(description="Docling self_ref, e.g. #/tables/64")
    order: int = Field(description="reading-order index across the document")
    type: ElementType
    page: int
    bbox: list[float] = Field(description="[l, t, r, b] in crop-box points, top-left origin")
    page_width: float
    page_height: float
    section: Section
    subsection: str
    statement: Statement = "none"
    text: str | None = None
    level: int | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    grid: list[list[str]] | None = Field(default=None, description="table cell texts, row-major")
    caption: str | None = None
    has_image: bool = False
    image_ref: str | None = Field(default=None, description="Docling artifact path for pictures")

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class ElementsSummary(BaseModel):
    count: int
    by_type: dict[str, int]
    by_section: dict[str, int]
    by_type_and_section: dict[str, dict[str, int]]
    statement_pages: dict[int, Statement]
    subsections_by_page: dict[int, str]
    toc_pages: list[int]


def _ref_to_id(ref: str) -> str:
    return ref.lstrip("#/").replace("/", "_")


def build_elements(docling_json: Path) -> tuple[list[Element], ElementsSummary]:
    from docling_core.types.doc import DoclingDocument, PictureItem, TableItem, TextItem

    doc = DoclingDocument.load_from_json(docling_json)
    n_pages = len(doc.pages)

    # Pass 1: headings -> subsections, statement pages.
    headings: list[Heading] = []
    statement_pages: dict[int, Statement] = {}
    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem) or not item.prov:
            continue
        if LABEL_MAP.get(str(item.label.value), "other") != "heading":
            continue
        prov = item.prov[0]
        headings.append(Heading(prov.page_no, item.text))
        statement = detect_statement(item.text)
        if statement != "none":
            page_h = doc.pages[prov.page_no].size.height
            top = page_h - prov.bbox.t  # distance from the top edge
            if top <= STATEMENT_HEADING_MAX_TOP_FRACTION * page_h:
                statement_pages.setdefault(prov.page_no, statement)
    subsections = assign_subsections(headings, last_page=n_pages)

    # Pass 2: every item -> Element.
    elements: list[Element] = []
    order = 0
    for item, level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        page = prov.page_no
        size = doc.pages[page].size
        bbox_tl = prov.bbox.to_top_left_origin(size.height)
        etype = LABEL_MAP.get(str(item.label.value), "other")
        el = Element(
            element_id=_ref_to_id(item.self_ref),
            ref=item.self_ref,
            order=order,
            type=etype,
            page=page,
            bbox=[
                round(bbox_tl.l, 2),
                round(bbox_tl.t, 2),
                round(bbox_tl.r, 2),
                round(bbox_tl.b, 2),
            ],
            page_width=size.width,
            page_height=size.height,
            section=page_section(page),
            subsection=subsections[page],
            statement=statement_pages.get(page, "none")
            if etype in {"table", "heading"}
            else "none",
        )
        if isinstance(item, TextItem):
            el.text = item.text
            if etype == "heading":
                el.level = level
        elif isinstance(item, TableItem):
            el.n_rows = item.data.num_rows
            el.n_cols = item.data.num_cols
            el.grid = [[(cell.text or "") for cell in row] for row in item.data.grid]
            el.caption = item.caption_text(doc) or None
        elif isinstance(item, PictureItem):
            el.has_image = item.image is not None
            if item.image is not None and item.image.uri is not None:
                el.image_ref = str(item.image.uri)
            el.caption = item.caption_text(doc) or None
        elements.append(el)
        order += 1

    by_type = Counter(e.type for e in elements)
    by_section = Counter(e.section for e in elements)
    cross: dict[str, dict[str, int]] = {}
    for e in elements:
        cross.setdefault(e.type, {}).setdefault(e.section, 0)
        cross[e.type][e.section] += 1
    summary = ElementsSummary(
        count=len(elements),
        by_type=dict(sorted(by_type.items())),
        by_section=dict(sorted(by_section.items())),
        by_type_and_section=cross,
        statement_pages=dict(sorted(statement_pages.items())),
        subsections_by_page=subsections,
        toc_pages=sorted(p for p, s in subsections.items() if s == "table_of_contents"),
    )
    log.info(
        "%d elements: %s; statement pages %s",
        summary.count,
        summary.by_type,
        summary.statement_pages,
    )
    return elements, summary


def write_elements(elements: list[Element], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in elements:
            fh.write(e.model_dump_json(exclude_none=True) + "\n")


def read_elements(path: Path) -> list[Element]:
    with path.open("r", encoding="utf-8") as fh:
        return [Element.model_validate(json.loads(line)) for line in fh if line.strip()]
