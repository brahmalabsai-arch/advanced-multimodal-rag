"""Phase 1 tests — section mapping, statement detection, figure candidates, reading order."""

from __future__ import annotations

import json

import pytest

from rag.core.settings import PROJECT_ROOT
from rag.ingest.elements import Element
from rag.ingest.report import check_reading_order
from rag.ingest.sections import (
    Heading,
    assign_subsections,
    detect_statement,
    page_section,
    subsection_for_heading,
)

PARSED = PROJECT_ROOT / "data" / "parsed"

# ---------------------------------------------------------------------- sections


@pytest.mark.parametrize(
    "page, section",
    [
        (1, "annual_review"),
        (18, "annual_review"),
        (19, "proxy"),
        (88, "proxy"),
        (89, "form_10k"),
        (141, "form_10k"),
        (174, "form_10k"),
        (175, "back_cover"),
    ],
)
def test_page_section_boundaries(page: int, section: str) -> None:
    assert page_section(page) == section


def test_page_section_out_of_range() -> None:
    with pytest.raises(ValueError):
        page_section(176)


@pytest.mark.parametrize(
    "heading, statement",
    [
        ("Consolidated Balance Sheets", "balance_sheet"),
        (
            "NVIDIA Corporation and Subsidiaries Consolidated Balance Sheets "
            "(In millions, except par value)",
            "balance_sheet",
        ),
        (
            "NVIDIA Corporation and Subsidiaries Consolidated Statements of Income (In millions)",
            "income_statement",
        ),
        ("Consolidated Statements of Cash Flows", "cash_flow"),
        ("Consolidated Statements of Comprehensive Income", "none"),
        ("Item 8. Financial Statements and Supplementary Data", "none"),
    ],
)
def test_detect_statement(heading: str, statement: str) -> None:
    assert detect_statement(heading) == statement


@pytest.mark.parametrize(
    "section, heading, tag",
    [
        ("form_10k", "Item 1A. Risk Factors", "item_1a_risk_factors"),
        ("form_10k", "Item  7.  Management's Discussion and Analysis", "item_7_mdna"),
        (
            "form_10k",
            "NVIDIA Corporation and Subsidiaries Notes to the Consolidated Financial Statements "
            "(Continued)",
            "notes",
        ),
        (
            "form_10k",
            "Note 1 - Organization and Summary of Significant Accounting Policies",
            "notes",
        ),
        (
            "form_10k",
            "NVIDIA Corporation and Subsidiaries Consolidated Balance Sheets",
            "item_8_financials",
        ),
        ("form_10k", "Our Company", None),
        ("proxy", "Pay Versus Performance", "pay_vs_performance"),
        ("proxy", "Compensation Discussion and Analysis", "compensation"),
        ("proxy", "Proposal 1 - Election of Directors", "proposals"),
        ("annual_review", "Dear Shareholders,", "shareholder_letter"),
    ],
)
def test_subsection_for_heading(section: str, heading: str, tag: str | None) -> None:
    assert subsection_for_heading(section, heading) == tag


def test_assign_subsections_carries_forward_and_uses_last_marker_on_page() -> None:
    headings = [
        Heading(92, "Item 1. Business"),
        Heading(100, "Item 1A. Risk Factors"),
        Heading(121, "Item 2. Properties"),
        Heading(121, "Item 3. Legal Proceedings"),
        Heading(121, "Item 5. Market for Registrant's Common Equity"),
        Heading(124, "Item 7. Management's Discussion and Analysis"),
        Heading(20, "Notice of 2026 Annual Meeting of Stockholders"),
    ]
    sub = assign_subsections(headings)
    assert sub[91] == "form_10k_front_matter"
    assert sub[92] == "item_1_business"
    assert sub[99] == "item_1_business"
    assert sub[100] == "item_1a_risk_factors"
    assert sub[121] == "item_2_properties"  # first marker on the page tags the page
    assert sub[122] == "item_5_market"  # following pages continue from the last marker
    assert sub[124] == "item_7_mdna"
    assert sub[19] == "proxy_front_matter"
    assert sub[20] == "notice_of_meeting"
    assert sub[88] == "notice_of_meeting"
    assert sub[89] == "form_10k_front_matter"  # section change resets the cursor
    assert sub[175] == "back_cover"


def test_assign_subsections_flags_a_real_table_of_contents() -> None:
    toc = [Heading(91, f"Item {n}. Something") for n in range(1, 10)]
    sub = assign_subsections([*toc, Heading(92, "Item 1. Business")])
    assert sub[91] == "table_of_contents"
    assert sub[92] == "item_1_business"


# ----------------------------------------------------------------- reading order


def _el(order: int, page: int, x: float, y: float, text: str) -> Element:
    return Element(
        element_id=f"texts_{order}",
        ref=f"#/texts/{order}",
        order=order,
        type="text",
        page=page,
        bbox=[x, y, x + 150, y + 40],
        page_width=594,
        page_height=774,
        section="annual_review",
        subsection="narrative_spread",
        text=text,
    )


def test_reading_order_check_passes_on_column_wise_order() -> None:
    els = [
        _el(0, 13, 63, 100, "Compute is no longer just a cost to support software; it is"),
        _el(1, 13, 63, 300, "second paragraph"),
        _el(2, 13, 232, 100, "NVIDIA is building the computing infrastructure"),
        _el(3, 13, 232, 300, "fourth"),
        _el(4, 13, 401, 100, "fifth"),
    ]
    check = check_reading_order(els)
    assert check.passed and check.columns == 3 and check.fragment_intact


def test_reading_order_check_fails_on_interleaved_columns() -> None:
    els = [
        _el(0, 13, 63, 100, "Compute is no longer just a"),
        _el(1, 13, 232, 100, "NVIDIA is building the computing"),
        _el(2, 13, 63, 300, "cost to support software"),
    ]
    check = check_reading_order(els)
    assert not check.fragment_intact
    assert not check.column_order_ok
    assert not check.passed


# ------------------------------------------------------- artifacts from a real run


@pytest.mark.skipif(
    not (PARSED / "figure_candidates.json").exists(), reason="run `make ingest` first"
)
def test_figure_candidates_include_p3_and_p123() -> None:
    data = json.loads((PARSED / "figure_candidates.json").read_text(encoding="utf-8"))
    pages = {c["page"] for c in data["candidates"]}
    assert 3 in pages, "p. 3 five-layer-cake diagram must be a candidate"
    assert 123 in pages, "p. 123 vector stock-performance chart must be a candidate"
    assert 5 in pages, "p. 5 decorative render must be a candidate so the classifier can drop it"
    p123 = [c for c in data["candidates"] if c["page"] == 123 and c["source"] == "docling_picture"]
    assert p123 and p123[0]["caption"] and "$100 invested" in p123[0]["caption"]
    for c in data["candidates"]:
        assert (PROJECT_ROOT / "data" / "index" / c["image_path"]).exists()
        assert (PROJECT_ROOT / "data" / "index" / c["page_image_path"]).exists()


@pytest.mark.skipif(
    not (PARSED / "elements_summary.json").exists(), reason="run `make ingest` first"
)
def test_elements_summary_statement_pages_and_sections() -> None:
    summary = json.loads((PARSED / "elements_summary.json").read_text(encoding="utf-8"))
    assert summary["statement_pages"] == {
        "139": "income_statement",
        "141": "balance_sheet",
        "143": "cash_flow",
    }
    subs = summary["subsections_by_page"]
    assert subs["141"] == "item_8_financials"
    assert subs["150"] == "notes"
    assert subs["13"] in {"narrative_spread", "shareholder_letter"}


@pytest.mark.skipif(
    not (PARSED / "ingestion_report.json").exists(), reason="run `make ingest` first"
)
def test_ingestion_report_passed_and_p13_reading_order() -> None:
    report = json.loads((PARSED / "ingestion_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["reading_order"]["passed"] is True
    assert report["reading_order"]["columns"] == 3
    bs = report["statements"]["balance_sheet"]
    assert bs["identity"]["holds"] is True
    assert bs["disagreements"] == []
