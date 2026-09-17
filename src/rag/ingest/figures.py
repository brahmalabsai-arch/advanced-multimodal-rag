"""Phase 1 — figure candidates: detect → rasterize (architecture §3.5 steps 1–2).

Candidate sources, in priority order:
1. `docling_picture` — picture regions from Docling's layout model (it sees the rendered page,
   so vector charts such as the p. 123 stock-performance graph are included).
2. `raster_uncovered` — large embedded raster images that no Docling picture covers. The
   Annual Review spreads (pp. 1–11) are full-bleed photographs Docling treats as background;
   they must become candidates so the Phase 2 classifier can *decide* to drop them (FR-1.3).
3. `vector_density` — pages with many shaped vector paths outside every Docling table or
   picture (a chart the layout model missed). Statement pages are excluded by construction
   because their paths sit inside table regions.

Every candidate is rasterized with pypdfium2 to `data/index/figures/p{page}_{idx}.png` (for the
vision model); every page gets a WebP thumbnail in `data/index/pages/p{page}.webp` (for the UI).
Phase 2 classifies, describes and links companion tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.ingest.elements import Element

log = get_logger(__name__)

CandidateSource = Literal["docling_picture", "raster_uncovered", "vector_density"]

FIGURE_DPI = 200
THUMBNAIL_WIDTH_PX = 1000
MIN_SIDE_PT = 40.0  # drop icons / bullets
MIN_AREA_FRACTION = 0.01  # of the page
RASTER_MIN_AREA_FRACTION = 0.05
RASTER_COVERED_FRACTION = 0.5  # a raster image ≥ 50 % inside a Docling picture is not new
VECTOR_MIN_SHAPED_PATHS = 25
VECTOR_PAD_PT = 8.0
OCCUPIED_MARGIN_PT = 15.0  # table row shading often spills past Docling's table box
BAND_MAX_HEIGHT_PT = 20.0  # full-width, short paths are shading bands / rules, not chart ink
BAND_MIN_WIDTH_FRACTION = 0.6

BBox = tuple[float, float, float, float]  # l, t, r, b — top-left origin, crop-box points


class FigureCandidate(BaseModel):
    figure_id: str = Field(description="p{page}_{idx}")
    page: int
    bbox: list[float] = Field(description="[l, t, r, b] crop-box points, top-left origin")
    page_width: float
    page_height: float
    source: CandidateSource
    element_id: str | None = Field(default=None, description="Docling picture element id")
    caption: str | None = None
    image_path: str | None = Field(default=None, description="relative to data/index")
    page_image_path: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    section: str | None = None
    subsection: str | None = None


class FigureCandidatesFile(BaseModel):
    figure_dpi: int
    candidates: list[FigureCandidate]
    pages_rendered: int


# --------------------------------------------------------------------------- geometry


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    return _area((left, top, right, bottom)) if right > left and bottom > top else 0.0


def _clip(b: BBox, w: float, h: float) -> BBox:
    return (max(0.0, b[0]), max(0.0, b[1]), min(w, b[2]), min(h, b[3]))


def _big_enough(b: BBox, w: float, h: float) -> bool:
    return (
        (b[2] - b[0]) >= MIN_SIDE_PT
        and (b[3] - b[1]) >= MIN_SIDE_PT
        and _area(b) >= MIN_AREA_FRACTION * w * h
    )


def _contains_point(b: BBox, x: float, y: float) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


# ------------------------------------------------------------------------- detection


def detect_candidates(elements: list[Element], pdf_path: Path) -> list[FigureCandidate]:
    import pypdfium2 as pdfium
    from pypdfium2.raw import FPDF_PAGEOBJ_IMAGE, FPDF_PAGEOBJ_PATH

    by_page: dict[int, list[Element]] = {}
    for e in elements:
        by_page.setdefault(e.page, []).append(e)

    pdf = pdfium.PdfDocument(str(pdf_path))
    candidates: list[FigureCandidate] = []
    try:
        for page_no in range(1, len(pdf) + 1):
            page = pdf[page_no - 1]
            width, height = page.get_size()
            crop_x0, crop_y0, _, _ = page.get_cropbox()
            page_elements = by_page.get(page_no, [])
            section = page_elements[0].section if page_elements else None
            subsection = page_elements[0].subsection if page_elements else None
            pictures = [e for e in page_elements if e.type == "picture"]
            tables = [e for e in page_elements if e.type == "table"]
            picture_boxes: list[BBox] = [tuple(e.bbox) for e in pictures]  # type: ignore[misc]
            occupied: list[BBox] = [
                (
                    b[0] - OCCUPIED_MARGIN_PT,
                    b[1] - OCCUPIED_MARGIN_PT,
                    b[2] + OCCUPIED_MARGIN_PT,
                    b[3] + OCCUPIED_MARGIN_PT,
                )
                for b in picture_boxes + [tuple(e.bbox) for e in tables]  # type: ignore[misc]
            ]
            page_cands: list[FigureCandidate] = []

            # 1. Docling pictures
            for e in pictures:
                b = _clip(tuple(e.bbox), width, height)  # type: ignore[arg-type]
                if not _big_enough(b, width, height):
                    continue
                page_cands.append(
                    FigureCandidate(
                        figure_id="",
                        page=page_no,
                        bbox=[round(v, 2) for v in b],
                        page_width=width,
                        page_height=height,
                        source="docling_picture",
                        element_id=e.element_id,
                        caption=e.caption,
                        section=section,
                        subsection=subsection,
                    )
                )

            # 2. Large raster images not covered by a Docling picture; 3. shaped vector paths
            shaped: list[BBox] = []
            for obj in page.get_objects(max_depth=4):
                left, bottom, right, top = obj.get_bounds()  # media-box, bottom-left origin
                b: BBox = _clip(
                    (
                        left - crop_x0,
                        height - (top - crop_y0),
                        right - crop_x0,
                        height - (bottom - crop_y0),
                    ),
                    width,
                    height,
                )
                if obj.type == FPDF_PAGEOBJ_IMAGE:
                    if _area(b) < RASTER_MIN_AREA_FRACTION * width * height:
                        continue
                    covered = max((_intersection(b, p) for p in picture_boxes), default=0.0)
                    if _area(b) and covered / _area(b) >= RASTER_COVERED_FRACTION:
                        continue
                    if any(
                        _intersection(b, c) / _area(b) > 0.9
                        for c in map(lambda x: tuple(x.bbox), page_cands)
                    ):  # type: ignore[arg-type]
                        continue
                    page_cands.append(
                        FigureCandidate(
                            figure_id="",
                            page=page_no,
                            bbox=[round(v, 2) for v in b],
                            page_width=width,
                            page_height=height,
                            source="raster_uncovered",
                            section=section,
                            subsection=subsection,
                        )
                    )
                elif obj.type == FPDF_PAGEOBJ_PATH:
                    w_, h_ = b[2] - b[0], b[3] - b[1]
                    if w_ <= 2 or h_ <= 2:
                        continue  # rules and hairlines
                    if h_ < BAND_MAX_HEIGHT_PT and w_ > BAND_MIN_WIDTH_FRACTION * width:
                        continue  # shading bands
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if not any(_contains_point(o, cx, cy) for o in occupied):
                        shaped.append(b)
            if len(shaped) >= VECTOR_MIN_SHAPED_PATHS:
                union: BBox = (
                    min(s[0] for s in shaped) - VECTOR_PAD_PT,
                    min(s[1] for s in shaped) - VECTOR_PAD_PT,
                    max(s[2] for s in shaped) + VECTOR_PAD_PT,
                    max(s[3] for s in shaped) + VECTOR_PAD_PT,
                )
                union = _clip(union, width, height)
                if _big_enough(union, width, height):
                    page_cands.append(
                        FigureCandidate(
                            figure_id="",
                            page=page_no,
                            bbox=[round(v, 2) for v in union],
                            page_width=width,
                            page_height=height,
                            source="vector_density",
                            section=section,
                            subsection=subsection,
                        )
                    )

            for idx, c in enumerate(page_cands):
                c.figure_id = f"p{page_no}_{idx}"
            candidates.extend(page_cands)
            page.close()
    finally:
        pdf.close()

    by_source = {
        s: sum(1 for c in candidates if c.source == s)
        for s in ("docling_picture", "raster_uncovered", "vector_density")
    }
    log.info(
        "%d figure candidates on %d pages: %s",
        len(candidates),
        len({c.page for c in candidates}),
        by_source,
    )
    return candidates


# ---------------------------------------------------------------------- rasterization


def rasterize(
    candidates: list[FigureCandidate],
    pdf_path: Path,
    index_dir: Path,
    *,
    dpi: int = FIGURE_DPI,
    thumbnail_width: int = THUMBNAIL_WIDTH_PX,
    render_pages: bool = True,
) -> int:
    """Write figure PNG crops and page WebP thumbnails. Returns the number of pages rendered."""
    import pypdfium2 as pdfium

    figures_dir = index_dir / "figures"
    pages_dir = index_dir / "pages"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    by_page: dict[int, list[FigureCandidate]] = {}
    for c in candidates:
        by_page.setdefault(c.page, []).append(c)

    pdf = pdfium.PdfDocument(str(pdf_path))
    rendered = 0
    try:
        for page_no in range(1, len(pdf) + 1):
            page = pdf[page_no - 1]
            width, height = page.get_size()
            if render_pages:
                scale = thumbnail_width / width
                image = page.render(scale=scale).to_pil().convert("RGB")
                out = pages_dir / f"p{page_no}.webp"
                image.save(out, "WEBP", quality=80, method=4)
                rendered += 1
            scale = dpi / 72.0
            for c in by_page.get(page_no, []):
                left, top, right, bottom = c.bbox
                crop = (
                    left,
                    height - bottom,
                    width - right,
                    top,
                )  # left, bottom, right, top margins to cut
                image = page.render(scale=scale, crop=crop).to_pil().convert("RGB")
                out = figures_dir / f"{c.figure_id}.png"
                image.save(out, "PNG", optimize=True)
                c.image_path = f"figures/{c.figure_id}.png"
                c.page_image_path = f"pages/p{page_no}.webp"
                c.width_px, c.height_px = image.size
            page.close()
    finally:
        pdf.close()
    log.info(
        "Rasterized %d figure crops at %d DPI; %d page thumbnails", len(candidates), dpi, rendered
    )
    return rendered
