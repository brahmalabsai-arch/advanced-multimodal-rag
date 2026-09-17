"""Ingestion entry point — `make ingest` (runs in `.venv-ingest`).

Phase 1 stages: parse → elements → validate → figures → report. Phase 2 appends
chunk → enrich → index. Each stage writes plain files under `data/parsed` / `data/index`,
so later stages never need Docling.

    .venv-ingest/Scripts/python -m rag.ingest.run
        [--stage all|parse|elements|validate|figures|report]
        [--pdf PATH] [--force-parse] [--no-pages]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.core.logging import configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, Settings, get_settings
from rag.ingest.elements import (
    Element,
    ElementsSummary,
    build_elements,
    read_elements,
    write_elements,
)
from rag.ingest.figures import FigureCandidate, FigureCandidatesFile, detect_candidates, rasterize
from rag.ingest.parse import ParsePaths, parse_pdf, sha256_file
from rag.ingest.report import build_report, write_report
from rag.ingest.validate import (
    StatementsFile,
    StatementTable,
    ValidationError,
    compare_with_docling,
    docling_grid_to_rows,
    extract_statement_page,
    validate_balance_sheet,
)

log = get_logger(__name__)

STAGES = ("parse", "elements", "validate", "figures", "report")


class IngestPaths:
    def __init__(self, settings: Settings, pdf: Path | None = None):
        self.parse = ParsePaths(settings, pdf)
        self.pdf = self.parse.pdf
        self.parsed_dir = self.parse.parsed_dir
        self.index_dir = settings.data_dir / "index"
        self.elements = self.parsed_dir / "elements.jsonl"
        self.elements_summary = self.parsed_dir / "elements_summary.json"
        self.statements = self.parsed_dir / "statements.json"
        self.figures = self.parsed_dir / "figure_candidates.json"
        self.report_json = self.parsed_dir / "ingestion_report.json"
        self.report_md = PROJECT_ROOT / "docs" / "reports" / "ingestion_phase1.md"


# --------------------------------------------------------------------------- stages


def stage_parse(paths: IngestPaths, *, force: bool) -> None:
    parse_pdf(paths.parse, force=force)


def stage_elements(paths: IngestPaths) -> tuple[list[Element], ElementsSummary]:
    elements, summary = build_elements(paths.parse.docling_json)
    write_elements(elements, paths.elements)
    paths.elements_summary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    log.info("Wrote %d elements to %s", len(elements), paths.elements)
    return elements, summary


def _load_elements(paths: IngestPaths) -> tuple[list[Element], ElementsSummary]:
    if not paths.elements.exists():
        return stage_elements(paths)
    elements = read_elements(paths.elements)
    summary = ElementsSummary.model_validate_json(
        paths.elements_summary.read_text(encoding="utf-8")
    )
    return elements, summary


def stage_validate(
    paths: IngestPaths, elements: list[Element], summary: ElementsSummary
) -> dict[str, StatementTable]:
    """pdfplumber extraction + Docling cross-check for every detected statement page.

    Raises `ValidationError` (stops ingestion) if the balance-sheet identity fails.
    """
    tables_by_page: dict[int, list[Element]] = {}
    for e in elements:
        if e.type == "table" and e.statement != "none":
            tables_by_page.setdefault(e.page, []).append(e)

    statements: dict[str, StatementTable] = {}
    for page, statement in summary.statement_pages.items():
        table = extract_statement_page(paths.pdf, page)
        docling_rows = []
        for t in tables_by_page.get(page, []):
            docling_rows.extend(docling_grid_to_rows(t.grid or [], len(table.period_ends)))
        table = compare_with_docling(table, docling_rows)
        if statement == "balance_sheet":
            table = validate_balance_sheet(table)
        statements[statement] = table
        log.info(
            "%s p%d: %d rows, docling matched %d/%d, %d disagreements",
            statement,
            page,
            len(table.rows),
            table.docling_rows_matched,
            table.docling_rows_total,
            len(table.disagreements),
        )
    if "balance_sheet" not in statements:
        raise ValidationError("No balance sheet page detected — cannot validate the corpus")
    out = StatementsFile(pdf_sha256=sha256_file(paths.pdf), statements=statements)
    paths.statements.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    log.info("Wrote %s", paths.statements)
    return statements


def stage_figures(
    paths: IngestPaths, elements: list[Element], *, render_pages: bool
) -> tuple[list[FigureCandidate], int]:
    candidates = detect_candidates(elements, paths.pdf)
    rendered = rasterize(candidates, paths.pdf, paths.index_dir, render_pages=render_pages)
    out = FigureCandidatesFile(figure_dpi=200, candidates=candidates, pages_rendered=rendered)
    paths.figures.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    log.info("Wrote %s", paths.figures)
    return candidates, rendered


def stage_report(
    paths: IngestPaths,
    elements: list[Element],
    summary: ElementsSummary,
    statements: dict[str, StatementTable],
    candidates: list[FigureCandidate],
    rendered: int,
) -> bool:
    meta = json.loads(paths.parse.meta_json.read_text(encoding="utf-8"))
    report = build_report(
        pdf=paths.pdf,
        docling_meta=meta,
        elements=elements,
        summary=summary,
        statements=statements,
        candidates=candidates,
        pages_rendered=rendered,
    )
    write_report(report, candidates, paths.report_json, paths.report_md)
    log.info("Report: %s and %s (passed=%s)", paths.report_json, paths.report_md, report.passed)
    return report.passed


# ------------------------------------------------------------------------------ main


def run(stage: str, *, pdf: Path | None, force_parse: bool, render_pages: bool) -> int:
    settings = get_settings()
    paths = IngestPaths(settings, pdf)
    run_all = stage == "all"

    if run_all or stage == "parse":
        stage_parse(paths, force=force_parse)
        if not run_all:
            return 0

    if run_all or stage == "elements":
        elements, summary = stage_elements(paths)
        if not run_all:
            return 0
    else:
        elements, summary = _load_elements(paths)

    statements: dict[str, StatementTable] = {}
    if run_all or stage == "validate":
        statements = stage_validate(paths, elements, summary)
        if not run_all:
            return 0

    candidates: list[FigureCandidate] = []
    rendered = 0
    if run_all or stage == "figures":
        candidates, rendered = stage_figures(paths, elements, render_pages=render_pages)
        if not run_all:
            return 0

    if stage == "report":
        statements = StatementsFile.model_validate_json(
            paths.statements.read_text(encoding="utf-8")
        ).statements
        fc = FigureCandidatesFile.model_validate_json(paths.figures.read_text(encoding="utf-8"))
        candidates, rendered = fc.candidates, fc.pages_rendered
    passed = stage_report(paths, elements, summary, statements, candidates, rendered)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("all", *STAGES), default="all")
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument(
        "--force-parse", action="store_true", help="re-run Docling even if the parse is current"
    )
    ap.add_argument("--no-pages", action="store_true", help="skip page thumbnails (faster)")
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    try:
        return run(
            args.stage, pdf=args.pdf, force_parse=args.force_parse, render_pages=not args.no_pages
        )
    except ValidationError as exc:
        log.error("INGESTION STOPPED: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
