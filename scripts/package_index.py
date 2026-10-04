"""Package `data/index` as a release asset the deployed image downloads at build time (F2).

    .venv/Scripts/python scripts/package_index.py            # writes dist/index.tar.gz
    .venv/Scripts/python scripts/package_index.py --keep-all # no pruning, no re-encoding

Render's build cannot run ingestion — Docling and PyTorch are not in the serving image, and the
parse takes 35 minutes — so the index travels as a tarball attached to a GitHub release and the
image pulls it in. The raw directory is 84 MB, two thirds of it images that do not need to be
that large:

* `figures/` holds a 200-DPI PNG for every *candidate* the detector found, including the logos
  and photographs ingestion then discarded. Only the crops referenced by a figure chunk can ever
  be served, so the rest are dropped.
* Those survivors are re-encoded to `VISION_MAX_SIDE_PX` — the bound `rag/llm.py` already applies
  before any vision request — so nothing the model or the page sees changes.
* `pages/` is kept whole: chunks cite 168 of the 175 thumbnails, so pruning saves nothing.

The manifest, Chroma collection, BM25 index and sidecars are copied verbatim: `corpus_version`
must still recompute at startup or `check_index` fails (D-64).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

INDEX_DIR = PROJECT_ROOT / "data" / "index"
OUT_DIR = PROJECT_ROOT / "dist"
MAX_SIDE_PX = 1600
JPEG_QUALITY = 85


def referenced_figures(index_dir: Path) -> set[str]:
    """File names of the figure crops a chunk can actually cite."""
    names: set[str] = set()
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            meta = json.loads(line)["metadata"]
            if meta.get("modality") == "figure" and meta.get("image_path"):
                names.add(Path(meta["image_path"]).name)
    return names


def _shrink(src: Path, dst: Path) -> tuple[int, int]:
    """Copy an image, bounding its longer side. Returns (bytes before, bytes after)."""
    before = src.stat().st_size
    try:
        from PIL import Image

        with Image.open(src) as im:
            if max(im.size) > MAX_SIDE_PX:
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
                im.save(dst, "PNG", optimize=True)
                return before, dst.stat().st_size
    except ImportError:
        pass
    shutil.copy2(src, dst)
    return before, dst.stat().st_size


def build(index_dir: Path, out: Path, *, keep_all: bool) -> Path:
    utf8_console()
    if not (index_dir / "manifest.json").exists():
        raise SystemExit(f"no index at {index_dir} — run `make ingest` first")
    keep = referenced_figures(index_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    dropped = kept = saved = 0

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "index"
        staged.mkdir()
        for item in sorted(index_dir.iterdir()):
            if item.name == "figures" and not keep_all:
                (staged / "figures").mkdir()
                for png in sorted(item.iterdir()):
                    if png.name not in keep:
                        dropped += 1
                        continue
                    before, after = _shrink(png, staged / "figures" / png.name)
                    saved += before - after
                    kept += 1
            elif item.is_dir():
                shutil.copytree(item, staged / item.name)
            elif item.name != ".gitkeep":
                shutil.copy2(item, staged / item.name)

        with tarfile.open(out, "w:gz") as tar:
            for path in sorted(staged.rglob("*")):
                tar.add(path, arcname=str(path.relative_to(staged)))

    size = out.stat().st_size / 1_048_576
    original = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file()) / 1_048_576
    print(f"index  {original:6.1f} MB on disk")
    if not keep_all:
        print(
            f"figures kept {kept}, dropped {dropped} unreferenced, {saved / 1_048_576:.1f} MB saved"
        )
    print(f"→ {out.relative_to(PROJECT_ROOT)}  {size:.1f} MB")
    print(
        "\nPublish it, then paste the download URL into deploy/render.yaml (INDEX_URL):\n"
        f"  gh release create v1.1-index {out.relative_to(PROJECT_ROOT)} "
        '--title "v1.1-index" --notes "Prebuilt index for the deployed demo"'
    )
    return out


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "index.tar.gz")
    ap.add_argument("--keep-all", action="store_true", help="no pruning, no re-encoding")
    args = ap.parse_args()
    build(args.index_dir, args.out, keep_all=args.keep_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
