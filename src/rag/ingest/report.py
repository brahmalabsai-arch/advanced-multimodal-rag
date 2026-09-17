"""Phase 1 — ingestion report (plan: `data/parsed/ingestion_report.json` + Markdown summary).

Also holds the p. 13 reading-order spot check: the multi-column shareholder-letter page whose
naive extraction interleaved columns (problem statement §2). The check asserts that (a) the
known fragment "Compute is no longer just a" continues inside the same paragraph and (b) text
items are read column by column (x advances monotonically, y restarts at each column).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.ingest.elements import Element, ElementsSummary
from rag.ingest.figures import FigureCandidate
from rag.ingest.validate import StatementTable

READING_ORDER_PAGE = 13
READING_ORDER_FRAGMENT = "Compute is no longer just a"
READING_ORDER_CONTINUATION = "cost to support software"
COLUMN_X_TOLERANCE_PT = 30.0


class ReadingOrderCheck(BaseModel):
    page: int
    text_items: int
    columns: int
    fragment_intact: bool = Field(description="known split fragment continues in one paragraph")
    column_order_ok: bool = Field(description="items are read column by column, top to bottom")
    passed: bool
    first_items: list[str]


def check_reading_order(
    elements: list[Element], page: int = READING_ORDER_PAGE
) -> ReadingOrderCheck:
    items = [e for e in elements if e.page == page and e.type in {"text", "heading", "list_item"}]
    items.sort(key=lambda e: e.order)
    fragment_intact = any(
        READING_ORDER_FRAGMENT in (e.text or "") and READING_ORDER_CONTINUATION in (e.text or "")
        for e in items
    )
    column_order_ok = True
    columns = 1 if items else 0
    prev = None
    for e in items:
        if prev is not None:
            same_column = abs(e.bbox[0] - prev.bbox[0]) <= COLUMN_X_TOLERANCE_PT
            if same_column:
                if e.bbox[1] < prev.bbox[1]:  # moved back up inside the same column
                    column_order_ok = False
            elif e.bbox[0] > prev.bbox[0]:
                columns += 1
            else:  # jumped back to a column on the left
                column_order_ok = False
        prev = e
    return ReadingOrderCheck(
        page=page,
        text_items=len(items),
        columns=columns,
        fragment_intact=fragment_intact,
        column_order_ok=column_order_ok,
        passed=fragment_intact and column_order_ok,
        first_items=[(e.text or "")[:80] for e in items[:6]],
    )


class IngestionReport(BaseModel):
    generated_at: str
    pdf: str
    pdf_sha256: str
    docling: dict[str, Any]
    elements: ElementsSummary
    statements: dict[str, StatementTable]
    figure_candidates: dict[str, Any]
    reading_order: ReadingOrderCheck
    passed: bool


def build_report(
    *,
    pdf: Path,
    docling_meta: dict[str, Any],
    elements: list[Element],
    summary: ElementsSummary,
    statements: dict[str, StatementTable],
    candidates: list[FigureCandidate],
    pages_rendered: int,
) -> IngestionReport:
    reading_order = check_reading_order(elements)
    by_source: dict[str, int] = {}
    for c in candidates:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    fig = {
        "count": len(candidates),
        "by_source": by_source,
        "pages": sorted({c.page for c in candidates}),
        "pages_rendered": pages_rendered,
        "required_pages_present": {p: any(c.page == p for c in candidates) for p in (3, 5, 123)},
    }
    bs = statements.get("balance_sheet")
    passed = (
        reading_order.passed
        and bs is not None
        and bs.identity is not None
        and bs.identity.holds
        and all(fig["required_pages_present"].values())
    )
    return IngestionReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pdf=pdf.name,
        pdf_sha256=docling_meta.get("pdf_sha256", ""),
        docling={
            k: docling_meta.get(k)
            for k in (
                "docling_version",
                "elapsed_seconds",
                "pages",
                "texts",
                "tables",
                "pictures",
                "conversion_status",
            )
        },
        elements=summary,
        statements=statements,
        figure_candidates=fig,
        reading_order=reading_order,
        passed=passed,
    )


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def render_markdown(report: IngestionReport, candidates: list[FigureCandidate]) -> str:
    r = report
    lines: list[str] = []
    lines.append("# Ingestion report — Phase 1 (parse and validate)\n")
    lines.append(
        f"Generated {r.generated_at} · `{r.pdf}` · sha256 `{r.pdf_sha256[:12]}…` · "
        f"Docling {r.docling.get('docling_version')} in {r.docling.get('elapsed_seconds')} s · "
        f"**overall: {'PASS' if r.passed else 'FAIL'}**\n"
    )

    lines.append("## Parse summary\n")
    lines.append("| Metric | Value |\n|---|---|")
    lines.append(f"| Pages | {r.docling.get('pages')} |")
    lines.append(
        f"| Docling texts / tables / pictures | {r.docling.get('texts')} / "
        f"{r.docling.get('tables')} / {r.docling.get('pictures')} |"
    )
    lines.append(f"| Elements written | {r.elements.count} |")
    lines.append(
        "| Statement pages | "
        + ", ".join(f"p{p} {s}" for p, s in r.elements.statement_pages.items())
        + " |"
    )
    lines.append("")

    lines.append("## Elements by type and section\n")
    sections = sorted(r.elements.by_section)
    lines.append("| Type | " + " | ".join(sections) + " | Total |")
    lines.append("|---|" + "---|" * (len(sections) + 1))
    for t, count in r.elements.by_type.items():
        row = r.elements.by_type_and_section.get(t, {})
        lines.append(
            f"| {t} | " + " | ".join(str(row.get(s, 0)) for s in sections) + f" | {count} |"
        )
    lines.append("")

    lines.append("## Section map (subsection transitions)\n")
    prev = None
    parts = []
    for page in sorted(r.elements.subsections_by_page):
        sub = r.elements.subsections_by_page[page]
        if sub != prev:
            parts.append(f"p{page} → `{sub}`")
            prev = sub
    lines.append("; ".join(parts) + "\n")

    lines.append("## Statement validation (pdfplumber vs Docling)\n")
    for key, t in r.statements.items():
        lines.append(f"### {t.title} (p. {t.page}, {t.unit}, periods {', '.join(t.period_ends)})\n")
        lines.append(
            f"Rows: {len(t.rows)} · Docling rows matched: "
            f"{t.docling_rows_matched}/{t.docling_rows_total} · "
            f"disagreements: **{len(t.disagreements)}**"
        )
        if t.identity is not None:
            lines.append(
                f"· accounting identity: **{'holds' if t.identity.holds else 'FAILS'}** "
                f"({' / '.join(_fmt(v) for v in t.identity.left)} = "
                f"{' / '.join(_fmt(v) for v in t.identity.right)})"
            )
        lines.append("")
        if t.disagreements:
            lines.append("| Row | Column | pdfplumber | Docling |\n|---|---|---|---|")
            for d in t.disagreements:
                lines.append(
                    f"| {d.label} | {d.column} | {_fmt(d.pdfplumber)} | {_fmt(d.docling)} |"
                )
            lines.append("")
        if t.subtotals:
            lines.append("| Subtotal | Stated | Computed | OK |\n|---|---|---|---|")
            for s in t.subtotals:
                lines.append(
                    f"| {s.total_label} | {' / '.join(_fmt(v) for v in s.expected)} | "
                    f"{' / '.join(_fmt(v) for v in s.computed)} | {'✓' if s.holds else '✗'} |"
                )
            lines.append("")
        if key == "balance_sheet":
            lines.append(
                "| Line item | Group | "
                + " | ".join(f"FY{fy}" for fy in t.fiscal_years)
                + " | Docling agrees |"
            )
            lines.append("|---|---|" + "---|" * (len(t.fiscal_years) + 1))
            for row in t.rows:
                agree = "✓" if row.agree else ("✗" if row.agree is False else "n/a")
                lines.append(
                    f"| {row.label[:60]} | {row.group} | "
                    + " | ".join(_fmt(v) for v in row.values)
                    + f" | {agree} |"
                )
            lines.append("")

    fc = r.figure_candidates
    lines.append("## Figure candidates\n")
    lines.append(
        f"{fc['count']} candidates on {len(fc['pages'])} pages · by source: "
        + ", ".join(f"{k} {v}" for k, v in fc["by_source"].items())
        + " · required pages present: "
        + ", ".join(f"p{p} {'✓' if ok else '✗'}" for p, ok in fc["required_pages_present"].items())
        + f" · page thumbnails rendered: {fc['pages_rendered']}\n"
    )
    lines.append("| Figure | Page | Source | Size (px) | Caption |\n|---|---|---|---|---|")
    for c in candidates:
        lines.append(
            f"| `{c.figure_id}` | {c.page} | {c.source} | {c.width_px}×{c.height_px} | "
            f"{(c.caption or '')[:70]} |"
        )
    lines.append("")

    ro = r.reading_order
    lines.append("## Reading-order spot check (p. 13)\n")
    lines.append(
        f"Text items: {ro.text_items} · columns detected: {ro.columns} · "
        f"fragment intact: {'✓' if ro.fragment_intact else '✗'} · "
        f"column order: {'✓' if ro.column_order_ok else '✗'} · "
        f"**{'PASS' if ro.passed else 'FAIL'}**\n"
    )
    for t in ro.first_items:
        lines.append(f"- {t}…")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: IngestionReport, candidates: list[FigureCandidate], json_path: Path, md_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report, candidates), encoding="utf-8")


# ================================================================= Phase 2 report


def render_phase2_markdown(
    chunk_summary: dict[str, Any],
    manifests: list[dict[str, Any]],
    figure_rows: list[tuple[str, str, str, int, str | None]],
    ledger_totals: dict[str, dict[str, int]],
) -> str:
    """Markdown summary of chunking, enrichment and indexing for docs/reports."""
    lines: list[str] = ["# Ingestion report — Phase 2 (chunk, enrich, index)\n"]
    counts = chunk_summary["chunk_counts"]
    lines.append(
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · chunks: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
        + f" (total {chunk_summary['chunks_total']}) · "
        f"sentences {chunk_summary['sentences_total']} · "
        f"text blocks {chunk_summary['text_blocks']} · split threshold p90 = "
        f"{chunk_summary['split_threshold']:.4f}\n"
    )
    lines.append("## Indexes\n")
    lines.append("| Index | Embedder | Dim | corpus_version | Chunks |\n|---|---|---|---|---|")
    for m in manifests:
        lines.append(
            f"| `{m['dir']}` | {m['embedder']} | {m['embedding_dim']} | `{m['corpus_version']}` | "
            f"{m['chunks_total']} |"
        )
    lines.append("")
    lines.append("## Enrichment (Groq, cached)\n")
    enr = chunk_summary.get("enrichment", {})
    lines.append(
        f"Models: {chunk_summary.get('enrichment_models')} · cache hits {enr.get('hits')} / "
        f"misses {enr.get('misses')} on the last run · figures dropped: "
        f"{chunk_summary.get('figures_dropped')}\n"
    )
    lines.append("| Ledger job | Calls | ok | Tokens in | Tokens out |\n|---|---|---|---|---|")
    for job, t in ledger_totals.items():
        lines.append(f"| {job} | {t['calls']} | {t['ok']} | {t['tokens_in']} | {t['tokens_out']} |")
    lines.append("")
    lines.append("## Figure classification\n")
    lines.append("| Figure | Type | Title | Data points | Companion table |\n|---|---|---|---|---|")
    for fid, ftype, title, npts, companion in figure_rows:
        lines.append(f"| `{fid}` | {ftype} | {title[:70]} | {npts} | {companion or ''} |")
    lines.append("")
    return "\n".join(lines)
