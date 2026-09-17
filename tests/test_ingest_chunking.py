"""Phase 2 tests — chunking guards, table parts, row facts, figure linking, corpus_version,
enrichment cache, BM25 tokenizer. Pure/synthetic tests run anywhere; artifact tests read the
outputs of `make ingest` when present."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from rag.core.bm25 import BM25Index, finance_tokenize
from rag.core.schema import Chunk, ChunkMetadata
from rag.core.settings import PROJECT_ROOT
from rag.ingest.chunk_semantic import (
    MAX_TOKENS,
    MIN_TOKENS,
    make_breadcrumb,
    merge_small_pieces,
    split_by_distance,
    split_large_pieces,
)
from rag.ingest.elements import Element
from rag.ingest.enrich import Enricher, safe_model_id
from rag.ingest.figures import DataPoint, FigureAnnotation, FigureCandidate, link_companion_table
from rag.ingest.index import compute_corpus_version
from rag.ingest.tables import (
    build_row_facts,
    build_table_chunks,
    split_rows_by_budget,
    table_markdown,
)
from rag.ingest.validate import StatementRow, StatementTable

PARSED = PROJECT_ROOT / "data" / "parsed"
INDEX = PROJECT_ROOT / "data" / "index"

# ------------------------------------------------------------------ splitting rules


def test_split_by_distance_and_size_guards() -> None:
    d = np.array([0.1, 0.9, 0.1, 0.1, 0.95, 0.1])  # 7 sentences, two big jumps
    pieces = split_by_distance(d, threshold=0.5)
    assert pieces == [(0, 2), (2, 5), (5, 7)]

    tokens = [10, 10, 100, 100, 100, 10, 10]
    merged = merge_small_pieces(pieces, tokens, d, min_tokens=50)
    assert merged == [(0, 5), (5, 7)] or merged == [(0, 7)]
    assert all(sum(tokens[a:b]) >= 20 for a, b in merged)

    big = split_large_pieces([(0, 7)], tokens, d, max_tokens=250)
    assert all(sum(tokens[a:b]) <= 250 for a, b in big)
    assert big[0][0] == 0 and big[-1][1] == 7


def test_merge_small_pieces_respects_cap() -> None:
    d = np.array([0.2, 0.2])
    tokens = [400, 30, 400]
    pieces = merge_small_pieces([(0, 1), (1, 2), (2, 3)], tokens, d, min_tokens=50, max_tokens=420)
    assert pieces == [(0, 1), (1, 2), (2, 3)]  # neither neighbour can absorb it under the cap


def test_breadcrumb_strips_running_head_and_units() -> None:
    bc = make_breadcrumb(
        "form_10k",
        "item_8_financials",
        "NVIDIA Corporation and Subsidiaries Consolidated Balance Sheets "
        "(In millions, except par value)",
    )
    assert bc == "Form 10-K > Item 8 Financial Statements > Consolidated Balance Sheets"
    assert (
        make_breadcrumb("proxy", "compensation", None) == "Proxy Statement > Executive Compensation"
    )


# --------------------------------------------------------------------------- tables


def _table_element(
    grid: list[list[str]], page: int = 141, statement: str = "balance_sheet"
) -> Element:
    return Element(
        element_id="tables_64",
        ref="#/tables/64",
        order=5,
        type="table",
        page=page,
        bbox=[40, 70, 554, 620],
        page_width=594,
        page_height=774,
        section="form_10k",
        subsection="item_8_financials",
        statement=statement,
        n_rows=len(grid),
        n_cols=len(grid[0]),
        grid=grid,
    )


def test_table_markdown_escapes_pipes_and_pads() -> None:
    md = table_markdown([["", "FY26"], ["Revenue | net", "1"], ["Short"]])
    assert md.splitlines()[0] == "|  | FY26 |"
    assert "Revenue \\| net" in md
    assert md.splitlines()[-1] == "| Short |  |"


def test_large_table_splits_by_rows_with_header_repeated() -> None:
    grid = [["Item", "FY2026", "FY2025"]] + [
        [f"Line item number {i} with a long label", f"{i},000", f"{i},500"] for i in range(60)
    ]
    el = _table_element(grid)
    chunks = build_table_chunks([el], {"tables_64": "Form 10-K > Item 8"}, None, max_tokens=300)
    assert len(chunks) > 1
    assert all(c.metadata.token_count <= 300 for c in chunks)
    assert all(c.metadata.parent_id == "tbl_p141_0" for c in chunks)
    assert [c.metadata.table_part for c in chunks] == list(range(1, len(chunks) + 1))
    for c in chunks:
        lines = c.document.splitlines()
        assert "| Item | FY2026 | FY2025 |" in lines  # header repeated
        data_rows = [ln for ln in lines if ln.startswith("| Line item")]
        for ln in data_rows:
            assert ln.count("|") == 4  # a row is never split across parts
    # every data row appears exactly once across the parts
    all_rows = [
        ln for c in chunks for ln in c.document.splitlines() if ln.startswith("| Line item")
    ]
    assert len(all_rows) == 60 and len(set(all_rows)) == 60


def test_split_rows_by_budget_never_empty() -> None:
    grid = [["a", "b"], ["x" * 2000, "y"]]  # single oversized row still yields one part
    assert split_rows_by_budget(grid, "prefix", 50) == [slice(0, 1)]


def _statements() -> dict[str, StatementTable]:
    rows = [
        StatementRow(
            label="Inventories",
            label_norm="inventories",
            group="current_assets",
            values=[21403.0, 10080.0],
        ),
        StatementRow(
            label="Short-term debt",
            label_norm="short term debt",
            group="current_liabilities",
            values=[999.0, None],
        ),
        StatementRow(label="Other", label_norm="other", group="equity", values=[1.0, 2.0]),
        StatementRow(label="Other", label_norm="other", group="equity", values=[3.0, 4.0]),
    ]
    return {
        "balance_sheet": StatementTable(
            statement="balance_sheet",
            page=141,
            title="Consolidated Balance Sheets",
            unit="USD_millions",
            period_ends=["2026-01-25", "2025-01-26"],
            fiscal_years=[2026, 2025],
            rows=rows,
        )
    }


def test_row_facts_carry_values_and_periods() -> None:
    chunks = build_row_facts(
        _statements(), {141: "tbl_p141_0"}, {141: "Form 10-K > Item 8 > Balance Sheets"}
    )
    by_id = {c.id: c for c in chunks}
    inv = by_id["rowfact_p141_inventories"]
    assert inv.metadata.modality == "row_fact"
    assert inv.metadata.value_fy2026 == 21403.0 and inv.metadata.value_fy2025 == 10080.0
    assert inv.metadata.period_end_fy2026 == "2026-01-25"
    assert inv.metadata.page == 141 and inv.metadata.parent_id == "tbl_p141_0"
    assert inv.metadata.line_group == "current_assets" and inv.metadata.unit == "USD_millions"
    assert "$21,403 million as of Jan 25, 2026 (FY2026)" in inv.document
    assert "$10,080 million as of Jan 26, 2025 (FY2025)" in inv.document
    std = by_id["rowfact_p141_short_term_debt"]
    assert std.metadata.value_fy2025 is None and "nil" in std.document
    assert (
        "rowfact_p141_other" in by_id and "rowfact_p141_other_2" in by_id
    )  # duplicates disambiguated


# -------------------------------------------------------------------------- figures


def test_companion_table_links_by_label_overlap() -> None:
    ann = FigureAnnotation(
        figure_type="chart",
        title="Comparison of 5 Year Cumulative Total Return",
        description="...",
        numbers_are_approximate=True,
        data_points=[
            DataPoint(series="NVIDIA Corporation", x="1/31/2021", y=100),
            DataPoint(series="Nasdaq 100", x="1/25/2026", y=220),
        ],
    )
    cand = FigureCandidate(
        figure_id="p123_0",
        page=123,
        bbox=[0, 0, 1, 1],
        page_width=594,
        page_height=774,
        source="docling_picture",
        caption="*$100 invested on 1/31/2021",
    )
    data_table = _table_element(
        [
            ["", "1/31/2021", "1/30/2022"],
            ["NVIDIA Corporation", "100", "176"],
            ["Nasdaq 100", "100", "113"],
        ],
        page=123,
        statement="none",
    )
    other = _table_element([["Name", "Age"], ["Jensen Huang", "63"]], page=123, statement="none")
    assert (
        link_companion_table(ann, cand, [("tbl_p123_1", other), ("tbl_p123_0", data_table)])
        == "tbl_p123_0"
    )
    assert link_companion_table(ann, cand, [("tbl_p123_1", other)]) is None


# ---------------------------------------------------------------- corpus_version


def test_corpus_version_changes_only_with_config_or_pdf() -> None:
    base = {
        "chunking": {"version": "semantic-v1", "max_tokens": 450},
        "figures": {"model": "qwen/qwen3.8-27b"},
        "embedder": "BAAI/bge-small-en-v1.5",
    }
    v0 = compute_corpus_version("abc", base)
    assert v0 == compute_corpus_version("abc", json.loads(json.dumps(base)))
    assert len(v0) == 12
    changed = json.loads(json.dumps(base))
    changed["figures"]["model"] = "other-vision-model"
    assert compute_corpus_version("abc", changed) != v0
    changed = json.loads(json.dumps(base))
    changed["chunking"]["max_tokens"] = 400
    assert compute_corpus_version("abc", changed) != v0
    changed = json.loads(json.dumps(base))
    changed["embedder"] = "BAAI/bge-base-en-v1.5"
    assert compute_corpus_version("abc", changed) != v0
    assert compute_corpus_version("different-pdf", base) != v0


# --------------------------------------------------------------- enrichment cache


class _Out(BaseModel):
    text: str


def test_enrichment_cache_hit_makes_no_call(tmp_path: Path) -> None:
    enr = Enricher(tmp_path / "enrichment")
    calls = {"n": 0}

    def compute() -> _Out:
        calls["n"] += 1
        return _Out(text="summary")

    a = enr.cached(b"table-content", "openai/gpt-oss-20b", _Out, compute, label="t1")
    b = enr.cached(b"table-content", "openai/gpt-oss-20b", _Out, compute, label="t1")
    assert a == b == _Out(text="summary")
    assert calls["n"] == 1 and enr.stats() == {"hits": 1, "misses": 1}
    # a different model id is a different cache entry
    enr.cached(b"table-content", "some/other-model", _Out, compute)
    assert calls["n"] == 2
    files = sorted(p.name for p in (tmp_path / "enrichment").glob("*.json"))
    assert len(files) == 2 and all("__" in f for f in files)
    assert safe_model_id("qwen/qwen3.8-27b") == "qwen--qwen3.8-27b"


# --------------------------------------------------------------------------- BM25


def test_finance_tokenizer_keeps_numbers_and_amounts() -> None:
    toks = finance_tokenize(
        "Inventories were $21,403 million; non-marketable securities grew 557.0% (H20)."
    )
    assert "$21403" in toks and "21403" in toks
    assert "non-marketable" in toks and "non" in toks and "marketable" in toks
    assert "557.0%" in toks and "h20" in toks


def test_bm25_index_roundtrip(tmp_path: Path) -> None:
    ids = ["a", "b", "c"]
    docs = ["Inventories 21,403 10,080", "Goodwill 20,832", "Total assets 206,803"]
    idx = BM25Index.build(ids, docs)
    assert idx.search("inventories")[0][0] == "a"
    assert idx.search("21403")[0][0] == "a"
    idx.save(tmp_path / "bm25.pkl")
    loaded = BM25Index.load(tmp_path / "bm25.pkl")
    assert loaded.search("goodwill")[0][0] == "b" and len(loaded) == 3


def test_chunk_chroma_metadata_drops_none() -> None:
    c = Chunk(
        id="x",
        document="d",
        metadata=ChunkMetadata(
            chunk_id="x",
            modality="text",
            section="proxy",
            subsection="proposals",
            page=3,
            breadcrumb="b",
            token_count=1,
        ),
    )
    meta = c.chroma_metadata()
    assert "line_item" not in meta and meta["page"] == 3
    assert Chunk.from_chroma("x", "d", meta) == c


# ----------------------------------------------------------- artifacts (make ingest)


def _chunks() -> list[dict]:
    return [
        json.loads(ln)
        for ln in (PARSED / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


@pytest.mark.skipif(not (PARSED / "chunks.jsonl").exists(), reason="run `make ingest` first")
def test_chunks_within_bounds_and_inside_sections() -> None:
    chunks = _chunks()
    text = [c for c in chunks if c["metadata"]["modality"] == "text"]
    assert text
    assert all(c["metadata"]["token_count"] <= MAX_TOKENS for c in text)
    small = [c for c in text if c["metadata"]["token_count"] < MIN_TOKENS]
    assert len(small) <= 0.03 * len(text), [c["id"] for c in small]
    elements = {}
    for ln in (PARSED / "elements.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(ln)
        elements[e["element_id"]] = e
    for c in text:
        subs = {
            (elements[eid]["section"], elements[eid]["subsection"])
            for eid in c["metadata"]["element_ids"].split(",")
        }
        assert len(subs) == 1, f"{c['id']} crosses a section boundary: {subs}"
    tables = [c for c in chunks if c["metadata"]["modality"] == "table"]
    assert all(c["metadata"]["token_count"] <= MAX_TOKENS + 20 for c in tables)


@pytest.mark.skipif(not (PARSED / "chunks.jsonl").exists(), reason="run `make ingest` first")
def test_row_fact_inventories_artifact() -> None:
    inv = next(c for c in _chunks() if c["id"] == "rowfact_p141_inventories")
    m = inv["metadata"]
    assert (m["value_fy2026"], m["value_fy2025"], m["page"]) == (21403.0, 10080.0, 141)
    assert m["statement"] == "balance_sheet" and m["parent_id"].startswith("tbl_p141")


@pytest.mark.skipif(not (PARSED / "chunk_summary.json").exists(), reason="run `make ingest` first")
def test_figures_p5_dropped_p123_linked() -> None:
    summary = json.loads((PARSED / "chunk_summary.json").read_text(encoding="utf-8"))
    if not summary.get("enrichment_models"):
        pytest.skip("chunk stage ran with --no-llm")
    figs = {c["id"]: c for c in _chunks() if c["metadata"]["modality"] == "figure"}
    assert "fig_p5_0" not in figs, "p. 5 decorative render must not be indexed"
    assert "fig_p123_0" in figs
    m = figs["fig_p123_0"]["metadata"]
    assert m["figure_type"] == "chart" and m["companion_table_id"]
    assert m["image_path"] == "figures/p123_0.png" and m["page_image_path"] == "pages/p123.webp"
    assert "fig_p3_0" in figs and figs["fig_p3_0"]["metadata"]["figure_type"] == "diagram"


@pytest.mark.skipif(not (INDEX / "manifest.json").exists(), reason="run `make ingest` first")
def test_index_manifest_and_sidecars() -> None:
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["corpus_version"]) == 12
    assert manifest["chunk_counts"]["row_fact"] >= 80
    emb = np.load(INDEX / "sentence_emb.npy")
    n_sent = sum(
        1
        for ln in (INDEX / "sentences.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    )
    assert emb.shape == (n_sent, manifest["embedding_dim"])
    assert (INDEX / "bm25.pkl").exists() and (INDEX / "chroma").exists()
