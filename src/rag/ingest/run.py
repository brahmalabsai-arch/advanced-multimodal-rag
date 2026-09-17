"""Ingestion entry point — `make ingest` (runs in `.venv-ingest`).

Stages: parse → elements → validate → figures → report (Phase 1) → chunk → index (Phase 2).
Each stage writes plain files under `data/parsed` / `data/index`, so later stages never need
Docling. `chunk` is the only stage that calls Groq (table summaries + figure descriptions), and
every result is cached, so a second run makes zero calls.

    .venv-ingest/Scripts/python -m rag.ingest.run
        [--stage all|parse|elements|validate|figures|report|chunk|index]
        [--embedder bge-small|bge-base] [--pdf PATH] [--force-parse] [--no-pages] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from rag.ingest.parse import ParsePaths, parse_pdf, settings_hash, sha256_file
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

STAGES = ("parse", "elements", "validate", "figures", "report", "chunk", "index")
CHUNK_EMBEDDER = "bge-small"  # boundary detection always uses the default embedder


class IngestPaths:
    def __init__(self, settings: Settings, pdf: Path | None = None):
        self.parse = ParsePaths(settings, pdf)
        self.pdf = self.parse.pdf
        self.parsed_dir = self.parse.parsed_dir
        self.data_dir = settings.data_dir
        self.index_dir = settings.data_dir / "index"
        self.elements = self.parsed_dir / "elements.jsonl"
        self.elements_summary = self.parsed_dir / "elements_summary.json"
        self.statements = self.parsed_dir / "statements.json"
        self.figures = self.parsed_dir / "figure_candidates.json"
        self.report_json = self.parsed_dir / "ingestion_report.json"
        self.report_md = PROJECT_ROOT / "docs" / "reports" / "ingestion_phase1.md"
        self.enrichment_dir = self.parsed_dir / "enrichment"
        self.chunks = self.parsed_dir / "chunks.jsonl"
        self.sentences = self.parsed_dir / "sentences.jsonl"
        self.chunk_summary = self.parsed_dir / "chunk_summary.json"


# ------------------------------------------------------------------- Phase 1 stages


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


def _load_statements(paths: IngestPaths) -> dict[str, StatementTable]:
    return StatementsFile.model_validate_json(
        paths.statements.read_text(encoding="utf-8")
    ).statements


def _load_candidates(paths: IngestPaths) -> tuple[list[FigureCandidate], int]:
    fc = FigureCandidatesFile.model_validate_json(paths.figures.read_text(encoding="utf-8"))
    return fc.candidates, fc.pages_rendered


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


# ------------------------------------------------------------------- Phase 2 stages


def stage_chunk(
    paths: IngestPaths,
    elements: list[Element],
    statements: dict[str, StatementTable],
    candidates: list[FigureCandidate],
    *,
    use_llm: bool,
) -> dict[str, Any]:
    """Text semantic chunks + table chunks + row facts + figure chunks → chunks.jsonl."""
    from rag.core.embeddings import Embedder
    from rag.ingest.chunk_semantic import (
        CHUNKING_VERSION,
        MAX_TOKENS,
        MIN_TOKENS,
        SPLIT_PERCENTILE,
        chunk_text,
        compute_breadcrumbs,
    )
    from rag.ingest.enrich import Enricher
    from rag.ingest.figures import (
        FIGURE_PROMPT_VERSION,
        FigureAnnotation,
        annotate_figures,
        build_figure_chunks,
    )
    from rag.ingest.index import write_jsonl
    from rag.ingest.tables import (
        SUMMARY_PROMPT_VERSION,
        TableSummary,
        build_row_facts,
        build_table_chunks,
        table_ids,
    )

    breadcrumbs = compute_breadcrumbs(elements)
    ids = table_ids(elements)

    # --- text
    text_result = chunk_text(elements, Embedder(CHUNK_EMBEDDER))

    # --- enrichment (cached)
    enricher = Enricher(paths.enrichment_dir)
    models: dict[str, str] = {}
    summarize = None
    annotate = None
    if use_llm:
        from rag.core.config import load_models_config
        from rag.llm import LLMClient

        settings = get_settings()
        cfg = load_models_config(settings=settings)
        client = LLMClient(settings, cfg)
        table_role = cfg.ingestion_enrichment.table_summaries
        figure_role = cfg.ingestion_enrichment.figures
        models = {
            "table_summaries": client.model_id(table_role),
            "figures": client.model_id(figure_role),
        }

        def summarize(prompt: str, label: str) -> TableSummary:
            return enricher.cached(
                (SUMMARY_PROMPT_VERSION + "\n" + prompt).encode("utf-8"),
                models["table_summaries"],
                TableSummary,
                lambda: client.json(
                    prompt,
                    TableSummary,
                    role=table_role,
                    ingestion_job="ingest-tables",
                    max_tokens=200,
                ),
                label=label,
            )

        def annotate(prompt: str, image_data_url: str, label: str) -> FigureAnnotation:
            return enricher.cached(
                (FIGURE_PROMPT_VERSION + "\n" + prompt + "\n" + image_data_url).encode("utf-8"),
                models["figures"],
                FigureAnnotation,
                lambda: client.vision_json(
                    prompt,
                    [image_data_url],
                    FigureAnnotation,
                    role=figure_role,
                    ingestion_job="ingest-figures",
                    max_tokens=500,  # descriptions run ~150 tokens; Groq charges the request
                ),
                label=label,
            )

    # --- tables + row facts
    table_chunks = build_table_chunks(elements, breadcrumbs, summarize)
    statement_table_by_page = {
        e.page: ids[e.element_id] for e in elements if e.type == "table" and e.statement != "none"
    }
    breadcrumb_by_page = {}
    for e in sorted(elements, key=lambda x: x.order):
        breadcrumb_by_page.setdefault(e.page, breadcrumbs[e.element_id])
    row_facts = build_row_facts(statements, statement_table_by_page, breadcrumb_by_page)

    # --- figures
    figure_chunks = []
    dropped: dict[str, int] = {}
    if annotate is not None:
        annotations = annotate_figures(candidates, elements, paths.index_dir, annotate)
        figure_chunks, dropped = build_figure_chunks(
            candidates, annotations, elements, breadcrumbs, ids
        )
    else:
        log.warning("--no-llm: table summaries and figure chunks skipped")

    chunks = text_result.chunks + table_chunks + row_facts + figure_chunks
    write_jsonl(paths.chunks, chunks)
    write_jsonl(paths.sentences, text_result.sentences)

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.metadata.modality] = counts.get(c.metadata.modality, 0) + 1
    ingestion_config = {
        "docling_settings_hash": settings_hash(),
        "chunking": {
            "version": CHUNKING_VERSION,
            "min_tokens": MIN_TOKENS,
            "max_tokens": MAX_TOKENS,
            "percentile": SPLIT_PERCENTILE,
            "boundary_embedder": CHUNK_EMBEDDER,
        },
        "tables": {
            "summary_prompt": SUMMARY_PROMPT_VERSION,
            "model": models.get("table_summaries"),
        },
        "figures": {"prompt": FIGURE_PROMPT_VERSION, "model": models.get("figures")},
    }
    summary = {
        "chunk_counts": counts,
        "chunks_total": len(chunks),
        "sentences_total": len(text_result.sentences),
        "text_blocks": text_result.n_blocks,
        "split_threshold": text_result.threshold,
        "figures_dropped": dropped,
        "enrichment": enricher.stats(),
        "enrichment_models": models,
        "ingestion_config": ingestion_config,
    }
    paths.chunk_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "Chunks by modality: %s (total %d); enrichment cache hits=%d misses=%d",
        counts,
        len(chunks),
        enricher.hits,
        enricher.misses,
    )
    return summary


def stage_index(paths: IngestPaths, embedder_alias: str) -> None:
    from rag.core.embeddings import Embedder
    from rag.ingest.index import build_index, index_dir_for, read_chunks, read_sentences

    if not paths.chunks.exists():
        raise FileNotFoundError(f"{paths.chunks} missing — run --stage chunk first")
    chunk_summary = json.loads(paths.chunk_summary.read_text(encoding="utf-8"))
    meta = json.loads(paths.parse.meta_json.read_text(encoding="utf-8"))
    chunks = read_chunks(paths.chunks)
    sentences = read_sentences(paths.sentences)
    index_dir = index_dir_for(paths.data_dir, embedder_alias)
    manifest = build_index(
        chunks,
        sentences,
        index_dir=index_dir,
        embedder=Embedder(embedder_alias),
        pdf_sha256=meta["pdf_sha256"],
        ingestion_config=chunk_summary["ingestion_config"],
        enrichment_models=chunk_summary.get("enrichment_models", {}),
        docling_version=meta.get("docling_version"),
    )
    log.info(
        "Index %s ready: corpus_version=%s, %s",
        index_dir,
        manifest.corpus_version,
        manifest.chunk_counts,
    )
    write_phase2_report(paths)


def write_phase2_report(paths: IngestPaths) -> None:
    """docs/reports/ingestion_phase2.md from chunk_summary, manifests, enrichment cache, ledger."""
    from rag.core.ledger import UsageLedger
    from rag.ingest.enrich import Enricher  # noqa: F401 - cache layout documented there
    from rag.ingest.index import read_chunks
    from rag.ingest.report import render_phase2_markdown

    chunk_summary = json.loads(paths.chunk_summary.read_text(encoding="utf-8"))
    manifests = []
    for d in sorted(paths.data_dir.glob("index*")):
        mf = d / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            m["dir"] = d.name
            manifests.append(m)
    figure_rows = []
    for c in read_chunks(paths.chunks):
        if c.metadata.modality == "figure":
            title = c.document.splitlines()[1].split("): ", 1)[-1]
            npts = c.document.count(" @ ")
            figure_rows.append(
                (c.id, c.metadata.figure_type or "", title, npts, c.metadata.companion_table_id)
            )
    totals: dict[str, dict[str, int]] = {}
    for r in UsageLedger(get_settings().logs_dir / "llm_usage.jsonl").read_all():
        job = r.ingestion_job or r.request_id or "-"
        if not job.startswith("ingest-"):
            continue
        t = totals.setdefault(job, {"calls": 0, "ok": 0, "tokens_in": 0, "tokens_out": 0})
        t["calls"] += 1
        t["ok"] += r.status == "ok"
        t["tokens_in"] += r.tokens_in
        t["tokens_out"] += r.tokens_out
    out = PROJECT_ROOT / "docs" / "reports" / "ingestion_phase2.md"
    out.write_text(
        render_phase2_markdown(chunk_summary, manifests, figure_rows, totals), encoding="utf-8"
    )
    log.info("Phase 2 report: %s", out)


# ------------------------------------------------------------------------------ main


def run(
    stage: str,
    *,
    pdf: Path | None,
    force_parse: bool,
    render_pages: bool,
    embedder: str,
    use_llm: bool,
) -> int:
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

    if run_all or stage == "validate":
        statements = stage_validate(paths, elements, summary)
        if not run_all:
            return 0
    elif stage in {"report", "chunk"}:
        statements = _load_statements(paths)
    else:
        statements = {}

    if run_all or stage == "figures":
        candidates, rendered = stage_figures(paths, elements, render_pages=render_pages)
        if not run_all:
            return 0
    elif stage in {"report", "chunk"}:
        candidates, rendered = _load_candidates(paths)
    else:
        candidates, rendered = [], 0

    if run_all or stage == "report":
        passed = stage_report(paths, elements, summary, statements, candidates, rendered)
        if not run_all:
            return 0 if passed else 1
        if not passed:
            log.error("Phase 1 report did not pass; stopping before chunking")
            return 1

    if run_all or stage == "chunk":
        stage_chunk(paths, elements, statements, candidates, use_llm=use_llm)
        if not run_all:
            return 0

    stage_index(paths, embedder)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("all", *STAGES), default="all")
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--embedder", default=CHUNK_EMBEDDER, help="bge-small (default) or bge-base")
    ap.add_argument(
        "--force-parse", action="store_true", help="re-run Docling even if the parse is current"
    )
    ap.add_argument("--no-pages", action="store_true", help="skip page thumbnails (faster)")
    ap.add_argument(
        "--no-llm", action="store_true", help="chunk without Groq (no summaries, no figure chunks)"
    )
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    try:
        return run(
            args.stage,
            pdf=args.pdf,
            force_parse=args.force_parse,
            render_pages=not args.no_pages,
            embedder=args.embedder,
            use_llm=not args.no_llm,
        )
    except ValidationError as exc:
        log.error("INGESTION STOPPED: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
