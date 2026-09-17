"""Phase 1 — layout-aware PDF parsing with Docling (architecture §3.1).

Runs in `.venv-ingest` only. Produces `data/parsed/docling.json` (the DoclingDocument) plus a
small sidecar `docling.meta.json` recording the PDF hash and pipeline settings, so a re-run on
an unchanged PDF is skipped (NFR-9 reproducibility, plan Phase 1).

    .venv-ingest/Scripts/python -m rag.ingest.parse [--pdf data/raw/X.pdf] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from rag.core.logging import configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, Settings, get_settings

log = get_logger(__name__)

DEFAULT_PDF_NAME = "2026_NVIDIA_ANNUAL_REPORT.pdf"

# Pipeline settings are part of the skip-check hash: changing them re-parses.
PIPELINE_SETTINGS: dict[str, Any] = {
    "do_ocr": False,  # text layer is present (problem statement §2); OCR only as fallback
    "do_table_structure": True,
    "table_cell_matching": True,
    "generate_picture_images": True,
    "images_scale": 2.0,  # ~144 DPI crops from Docling; pypdfium2 renders the final figures
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def settings_hash() -> str:
    return hashlib.sha256(json.dumps(PIPELINE_SETTINGS, sort_keys=True).encode()).hexdigest()[:12]


class ParsePaths:
    def __init__(self, settings: Settings | None = None, pdf: Path | None = None):
        s = settings or get_settings()
        self.raw_dir = s.data_dir / "raw"
        self.parsed_dir = s.data_dir / "parsed"
        self.pdf = pdf or (self.raw_dir / DEFAULT_PDF_NAME)
        self.docling_json = self.parsed_dir / "docling.json"
        self.meta_json = self.parsed_dir / "docling.meta.json"
        self.artifacts_dir = self.parsed_dir / "docling_artifacts"


def _docling_version() -> str:
    import docling

    return docling.__version__


def is_parse_current(paths: ParsePaths, pdf_hash: str) -> bool:
    if not (paths.docling_json.exists() and paths.meta_json.exists()):
        return False
    try:
        meta = json.loads(paths.meta_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("pdf_sha256") == pdf_hash
        and meta.get("settings_hash") == settings_hash()
        and meta.get("docling_version") == _docling_version()
    )


def parse_pdf(paths: ParsePaths, *, force: bool = False) -> Path:
    """Convert the PDF with Docling and save the document JSON. Returns the JSON path."""
    if not paths.pdf.exists():
        raise FileNotFoundError(f"Corpus PDF not found: {paths.pdf} (copy it into data/raw/)")
    pdf_hash = sha256_file(paths.pdf)
    if not force and is_parse_current(paths, pdf_hash):
        log.info("Parse is current for %s (sha256 %s…); skipping", paths.pdf.name, pdf_hash[:12])
        return paths.docling_json

    # Heavy imports stay inside the function so the module can be imported in the serving env.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode

    opts = PdfPipelineOptions()
    opts.do_ocr = PIPELINE_SETTINGS["do_ocr"]
    opts.do_table_structure = PIPELINE_SETTINGS["do_table_structure"]
    opts.table_structure_options.do_cell_matching = PIPELINE_SETTINGS["table_cell_matching"]
    opts.generate_picture_images = PIPELINE_SETTINGS["generate_picture_images"]
    opts.images_scale = PIPELINE_SETTINGS["images_scale"]

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    log.info("Docling %s converting %s …", _docling_version(), paths.pdf)
    started = time.monotonic()
    result = converter.convert(paths.pdf)
    elapsed = time.monotonic() - started
    doc = result.document
    log.info(
        "Converted %d pages in %.0fs: %d texts, %d tables, %d pictures (status=%s)",
        len(doc.pages),
        elapsed,
        len(doc.texts),
        len(doc.tables),
        len(doc.pictures),
        result.status.name,
    )

    paths.parsed_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(
        paths.docling_json, artifacts_dir=paths.artifacts_dir, image_mode=ImageRefMode.REFERENCED
    )
    meta = {
        "pdf": str(paths.pdf.relative_to(PROJECT_ROOT))
        if paths.pdf.is_relative_to(PROJECT_ROOT)
        else str(paths.pdf),
        "pdf_sha256": pdf_hash,
        "settings": PIPELINE_SETTINGS,
        "settings_hash": settings_hash(),
        "docling_version": _docling_version(),
        "conversion_status": result.status.name,
        "elapsed_seconds": round(elapsed, 1),
        "pages": len(doc.pages),
        "texts": len(doc.texts),
        "tables": len(doc.tables),
        "pictures": len(doc.pictures),
    }
    paths.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved %s and %s", paths.docling_json, paths.meta_json)
    return paths.docling_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--pdf", type=Path, default=None, help=f"PDF path (default: data/raw/{DEFAULT_PDF_NAME})"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-parse even if the cached parse is current"
    )
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    paths = ParsePaths(settings, args.pdf)
    parse_pdf(paths, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
