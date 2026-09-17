"""Tables → parent table chunks + row-fact children (architecture §3.4).

(a) One table chunk per Docling table: breadcrumb + one-line summary (small model, cached) +
    the table as Markdown. Tables over MAX_TOKENS are split by row groups with the header
    repeated; a row is never split.
(b) Row-fact chunks for the validated financial statements, built from `statements.json`
    (pdfplumber values that passed the identity check — never from the Docling grid), with the
    numeric values in metadata so the calculator never parses a number out of text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.core.schema import Chunk, ChunkMetadata
from rag.core.tokens import count_tokens
from rag.ingest.elements import Element
from rag.ingest.validate import StatementRow, StatementTable

log = get_logger(__name__)

MAX_TABLE_TOKENS = 450
SUMMARY_PROMPT_VERSION = "table-summary-v1"
SUMMARY_MAX_ROWS = 30

STATEMENT_TITLES = {
    "balance_sheet": "Consolidated Balance Sheets",
    "income_statement": "Consolidated Statements of Income",
    "cash_flow": "Consolidated Statements of Cash Flows",
}


class TableSummary(BaseModel):
    summary: str = Field(description="one sentence: what the table shows, entities, periods, units")


# ------------------------------------------------------------------------ markdown


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).replace("|", "\\|").strip()


def table_markdown(grid: list[list[str]], rows: slice | None = None) -> str:
    """GitHub-style Markdown; the first grid row is the header (repeated for every part)."""
    if not grid:
        return ""
    header = [_cell(c) for c in grid[0]]
    body = grid[1:] if rows is None else grid[1:][rows]
    n = max(len(header), *(len(r) for r in body)) if body else len(header)
    header += [""] * (n - len(header))
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * n]
    for r in body:
        cells = [_cell(c) for c in r] + [""] * (n - len(r))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_rows_by_budget(grid: list[list[str]], prefix: str, max_tokens: int) -> list[slice]:
    """Row-group slices (over `grid[1:]`) so that prefix + header + rows ≤ max_tokens each."""
    body = grid[1:]
    header_tokens = count_tokens(prefix + "\n" + table_markdown(grid[:1]))
    slices: list[slice] = []
    start = 0
    running = header_tokens
    for i, row in enumerate(body):
        row_tokens = count_tokens("| " + " | ".join(_cell(c) for c in row) + " |") + 1
        if running + row_tokens > max_tokens and i > start:
            slices.append(slice(start, i))
            start = i
            running = header_tokens
        running += row_tokens
    slices.append(slice(start, len(body)))
    return slices


# ------------------------------------------------------------------- table chunks


@dataclass
class TableContext:
    element: Element
    table_id: str
    breadcrumb: str


def table_ids(elements: list[Element]) -> dict[str, str]:
    """element_id -> logical table id `tbl_p{page}_{idx}` (idx = order of the table on its page)."""
    out: dict[str, str] = {}
    per_page: dict[int, int] = {}
    for e in sorted(elements, key=lambda x: x.order):
        if e.type != "table":
            continue
        idx = per_page.get(e.page, 0)
        per_page[e.page] = idx + 1
        out[e.element_id] = f"tbl_p{e.page}_{idx}"
    return out


def summary_prompt(breadcrumb: str, caption: str | None, grid: list[list[str]]) -> str:
    md = table_markdown(grid, rows=slice(0, SUMMARY_MAX_ROWS))
    more = (
        f"\n(… {len(grid) - 1 - SUMMARY_MAX_ROWS} more rows)"
        if len(grid) - 1 > SUMMARY_MAX_ROWS
        else ""
    )
    return (
        "You are indexing tables from NVIDIA's fiscal 2026 annual report (Annual Review, Proxy "
        "Statement and Form 10-K). Write ONE sentence (max 45 words) describing what this table "
        "shows: the entities or line items, the periods compared, and the units. Do not invent "
        "numbers; do not list every row.\n\n"
        f"Location: {breadcrumb}\n"
        f"Caption: {caption or '(none)'}\n\n"
        f"Table:\n{md}{more}"
    )


SummaryFn = Callable[[str, str], TableSummary]  # (prompt, label) -> summary


def build_table_chunks(
    elements: list[Element],
    breadcrumbs: dict[str, str],
    summarize: SummaryFn | None,
    *,
    max_tokens: int = MAX_TABLE_TOKENS,
) -> list[Chunk]:
    ids = table_ids(elements)
    chunks: list[Chunk] = []
    for e in sorted(elements, key=lambda x: x.order):
        if e.type != "table" or not e.grid:
            continue
        table_id = ids[e.element_id]
        breadcrumb = breadcrumbs.get(e.element_id, "")
        summary = ""
        if summarize is not None:
            try:
                summary = summarize(summary_prompt(breadcrumb, e.caption, e.grid), table_id).summary
            except Exception as exc:  # enrichment must not sink ingestion; the table still indexes
                log.warning("Summary failed for %s: %s", table_id, exc)
        prefix_lines = [f"[{breadcrumb}]", f"Table {table_id} (PDF p. {e.page})"]
        if e.caption:
            prefix_lines.append(f"Caption: {e.caption}")
        if summary:
            prefix_lines.append(f"Summary: {summary}")
        prefix = "\n".join(prefix_lines)
        parts = split_rows_by_budget(e.grid, prefix, max_tokens)
        for k, rows in enumerate(parts):
            md = table_markdown(e.grid, rows=rows)
            part_note = f"\n(part {k + 1} of {len(parts)})" if len(parts) > 1 else ""
            document = f"{prefix}{part_note}\n{md}"
            chunk_id = table_id if len(parts) == 1 else f"{table_id}_part{k + 1}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document=document,
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        modality="table",
                        section=e.section,
                        subsection=e.subsection,
                        page=e.page,
                        breadcrumb=breadcrumb,
                        statement=e.statement,
                        token_count=count_tokens(document),
                        parent_id=table_id,
                        table_part=k + 1,
                        table_parts=len(parts),
                        n_rows=e.n_rows,
                        n_cols=e.n_cols,
                        element_ids=e.element_id,
                    ),
                )
            )
    log.info("%d table chunks from %d tables", len(chunks), len(ids))
    return chunks


# --------------------------------------------------------------------- row facts


def _fmt_money(v: float | None, unit: str) -> str:
    if v is None:
        return "— (nil)"
    if unit == "USD_millions":
        return f"-${abs(v):,.0f} million" if v < 0 else f"${v:,.0f} million"
    if unit == "USD_per_share":
        return f"${v:,.2f} per share"
    if unit == "shares_millions":
        return f"{v:,.0f} million shares"
    return f"{v:,.2f}"


def _row_unit(row: StatementRow, table_unit: str) -> str:
    g = row.group
    if "weighted_average_shares" in g or g.startswith("shares"):
        return "shares_millions"
    if "per_share" in g or "per share" in g:
        return "USD_per_share"
    return table_unit


def _slug(label_norm: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label_norm).strip("_")[:60] or "row"


def _pretty_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return d.strftime("%b %d, %Y").replace(" 0", " ")


def build_row_facts(
    statements: dict[str, StatementTable],
    table_id_by_page: dict[int, str],
    breadcrumb_by_page: dict[int, str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for key, table in statements.items():
        title = STATEMENT_TITLES.get(key, table.title)
        parent = table_id_by_page.get(table.page, f"tbl_p{table.page}_0")
        breadcrumb = breadcrumb_by_page.get(
            table.page, f"Form 10-K > Item 8 Financial Statements > {title}"
        )
        phrase = "as of" if key == "balance_sheet" else "for the fiscal year ended"
        seen: dict[str, int] = {}
        for row in table.rows:
            unit = _row_unit(row, table.unit)
            parts = []
            for v, iso, fy in zip(row.values, table.period_ends, table.fiscal_years, strict=True):
                parts.append(f"{_fmt_money(v, unit)} {phrase} {_pretty_date(iso)} (FY{fy})")
            group_note = (
                f" [{row.group.replace('_', ' ')}]" if row.group not in {"body", "other"} else ""
            )
            document = (
                f"[{breadcrumb}]\nNVIDIA {title} — {row.label}{group_note}: "
                + "; ".join(parts)
                + "."
            )
            slug = _slug(row.label_norm)
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            chunk_id = f"rowfact_p{table.page}_{slug}" + (f"_{n + 1}" if n else "")
            meta: dict = {
                "chunk_id": chunk_id,
                "modality": "row_fact",
                "section": "form_10k",
                "subsection": "item_8_financials",
                "page": table.page,
                "breadcrumb": breadcrumb,
                "statement": key,
                "token_count": count_tokens(document),
                "parent_id": parent,
                "line_item": row.label,
                "line_item_norm": row.label_norm,
                "line_group": row.group,
                "unit": unit,
            }
            for v, iso, fy in zip(row.values, table.period_ends, table.fiscal_years, strict=True):
                if f"value_fy{fy}" in ChunkMetadata.model_fields:
                    meta[f"value_fy{fy}"] = v
                    meta[f"period_end_fy{fy}"] = iso
            chunks.append(Chunk(id=chunk_id, document=document, metadata=ChunkMetadata(**meta)))
    log.info("%d row facts from %d statements", len(chunks), len(statements))
    return chunks
