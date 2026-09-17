"""Phase 1 tests — numeric normaliser, row parsing, balance-sheet parse (p. 141), identity check."""

from __future__ import annotations

import pytest

from rag.core.settings import PROJECT_ROOT
from rag.ingest.validate import (
    NotNumeric,
    ParsedRow,
    StatementRow,
    ValidationError,
    assign_balance_sheet_groups,
    check_identity,
    check_subtotals,
    docling_grid_to_rows,
    find_period_ends,
    merge_wrapped_labels,
    normalize_label,
    normalize_number,
    parse_row_tokens,
)

PDF = PROJECT_ROOT / "data" / "raw" / "2026_NVIDIA_ANNUAL_REPORT.pdf"

# ------------------------------------------------------------------ numeric normaliser


@pytest.mark.parametrize(
    "token, expected",
    [
        ("$ 10,605", 10605.0),
        ("$10,605", 10605.0),
        ("10,605", 10605.0),
        ("(1,234)", -1234.0),
        ("(259)", -259.0),
        ("—", None),
        ("–", None),
        ("-", None),
        ("4.93", 4.93),
        ("1,100.68", 1100.68),
        ("24", 24.0),
        ("206,803(a)", 206803.0),
        ("12.5%", 12.5),
    ],
)
def test_normalize_number(token: str, expected: float | None) -> None:
    assert normalize_number(token) == expected


@pytest.mark.parametrize("token", ["Inventories", "$", "", "Note 12x", "1,23", "FY2026"])
def test_normalize_number_rejects_non_numeric(token: str) -> None:
    with pytest.raises(NotNumeric):
        normalize_number(token)


def test_normalize_label_keeps_apostrophes_and_strips_footnotes() -> None:
    assert normalize_label("Total liabilities and shareholders’ equity") == (
        "total liabilities and shareholders' equity"
    )
    assert normalize_label("Goodwill (1)") == "goodwill"
    assert normalize_label("  Current assets:  ") == "current assets"


# ------------------------------------------------------------------------ row parsing


def test_parse_row_tokens_reads_values_from_the_right() -> None:
    row = parse_row_tokens(["Cash", "and", "cash", "equivalents", "$", "10,605", "$", "8,589"], 2)
    assert row.label == "Cash and cash equivalents"
    assert row.values == [10605.0, 8589.0]


def test_parse_row_tokens_does_not_eat_numbers_inside_labels() -> None:
    tokens = ["Commitments", "and", "contingencies", "-", "see", "Note", "12", "—", "—"]
    row = parse_row_tokens(tokens, 2)
    assert row.label == "Commitments and contingencies - see Note 12"
    assert row.values == [None, None]

    row = parse_row_tokens(["outstanding", "as", "of", "January", "26,", "2025", "24", "24"], 2)
    assert row.label == "outstanding as of January 26, 2025"
    assert row.values == [24.0, 24.0]


def test_parse_row_tokens_short_term_debt_dash() -> None:
    row = parse_row_tokens(["Short-term", "debt", "999", "—"], 2)
    assert row.values == [999.0, None]


def test_parse_row_tokens_three_periods() -> None:
    row = parse_row_tokens(["Interest", "expense", "(259)", "(247)", "(257)"], 3)
    assert row.values == [-259.0, -247.0, -257.0]


def test_merge_wrapped_labels_joins_same_indent_fragments() -> None:
    rows = [
        ParsedRow(
            "Common stock, $0.001 par value; 80,000 shares authorized; 24,304 shares", [], x0=50
        ),
        ParsedRow(
            "issued and outstanding as of January 25, 2026; 24,477 shares issued and", [], x0=50
        ),
        ParsedRow("outstanding as of January 26, 2025", [24.0, 24.0], x0=50),
    ]
    merged = merge_wrapped_labels(rows)
    assert len(merged) == 1
    assert merged[0].label.startswith("Common stock") and merged[0].label.endswith("2025")
    assert merged[0].values == [24.0, 24.0]


def test_merge_wrapped_labels_keeps_headers_above_indented_rows() -> None:
    rows = [
        ParsedRow("Operating expenses", [], x0=40),
        ParsedRow("Research and development", [18497.0], x0=52),
    ]
    merged = merge_wrapped_labels(rows)
    assert [r.label for r in merged] == ["Operating expenses", "Research and development"]


def test_merge_wrapped_labels_title_line_is_not_a_fragment_of_colon_header() -> None:
    rows = [
        ParsedRow("Assets", [], x0=40),
        ParsedRow("Current assets:", [], x0=40),
        ParsedRow("Inventories", [21403.0, 10080.0], x0=52),
    ]
    merged = merge_wrapped_labels(rows)
    assert [r.label for r in merged] == ["Assets", "Current assets:", "Inventories"]


def test_merge_wrapped_labels_joins_mid_sentence_fragment_into_colon_header() -> None:
    rows = [
        ParsedRow(
            "Adjustments to reconcile net income to net cash provided by operating", [], x0=40
        ),
        ParsedRow("activities:", [], x0=40),
        ParsedRow("Depreciation", [1.0], x0=52),
    ]
    merged = merge_wrapped_labels(rows)
    assert merged[0].label.endswith("operating activities:")
    assert len(merged) == 2


def test_find_period_ends() -> None:
    ends = find_period_ends("Jan 25, 2026 Jan 26, 2025 Jan 28, 2024")
    assert [d.isoformat() for d in ends] == ["2026-01-25", "2025-01-26", "2024-01-28"]


# ------------------------------------------------------------------- groups / checks


def _bs_rows() -> list[StatementRow]:
    parsed = [
        ParsedRow("Assets", []),
        ParsedRow("Current assets:", []),
        ParsedRow("Cash and cash equivalents", [10605.0, 8589.0]),
        ParsedRow("Inventories", [21403.0, 10080.0]),
        ParsedRow("Total current assets", [32008.0, 18669.0]),
        ParsedRow("Goodwill", [20832.0, 5188.0]),
        ParsedRow("Total assets", [52840.0, 23857.0]),
        ParsedRow("Liabilities and Shareholders' Equity", []),
        ParsedRow("Current liabilities:", []),
        ParsedRow("Accounts payable", [9812.0, 6310.0]),
        ParsedRow("Total current liabilities", [9812.0, 6310.0]),
        ParsedRow("Total liabilities", [9812.0, 6310.0]),
        ParsedRow("Shareholders' equity:", []),
        ParsedRow("Retained earnings", [43028.0, 17547.0]),
        ParsedRow("Total shareholders' equity", [43028.0, 17547.0]),
        ParsedRow("Total liabilities and shareholders' equity", [52840.0, 23857.0]),
    ]
    groups = assign_balance_sheet_groups(parsed)
    return [
        StatementRow(label=r.label, label_norm=normalize_label(r.label), group=g, values=r.values)
        for r, g in zip(parsed, groups, strict=True)
        if r.values
    ]


def test_balance_sheet_groups() -> None:
    rows = {r.label: r.group for r in _bs_rows()}
    assert rows["Inventories"] == "current_assets"
    assert rows["Total current assets"] == "current_assets"
    assert rows["Goodwill"] == "non_current_assets"
    assert rows["Total assets"] == "assets"
    assert rows["Accounts payable"] == "current_liabilities"
    assert rows["Retained earnings"] == "equity"
    assert rows["Total liabilities and shareholders' equity"] == "liabilities_and_equity"


def test_identity_holds_on_real_values() -> None:
    check = check_identity(_bs_rows())
    assert check.holds
    assert check.left == [52840.0, 23857.0]


def test_identity_fails_on_altered_copy() -> None:
    rows = _bs_rows()
    total_assets = next(r for r in rows if r.label_norm == "total assets")
    total_assets.values = [52841.0, 23857.0]  # one dollar off in one column
    assert not check_identity(rows).holds


def test_identity_requires_both_rows() -> None:
    rows = [r for r in _bs_rows() if r.label_norm != "total assets"]
    with pytest.raises(ValidationError, match="identity rows missing"):
        check_identity(rows)


def test_subtotal_checks() -> None:
    from rag.ingest.validate import StatementTable

    table = StatementTable(
        statement="balance_sheet",
        page=141,
        title="t",
        unit="USD_millions",
        period_ends=["2026-01-25", "2025-01-26"],
        fiscal_years=[2026, 2025],
        rows=_bs_rows(),
    )
    checks = {c.total_label: c.holds for c in check_subtotals(table)}
    assert checks["Total current assets"] is True
    assert checks["Total shareholders' equity"] is True


def test_docling_grid_rows_treat_valueless_rows_as_headers() -> None:
    grid = [
        ["", "Jan 25, 2026", "Jan 26, 2025"],
        ["Operating expenses", "", ""],
        ["Research and development", "18,497", "12,914"],
    ]
    rows = docling_grid_to_rows(grid, 2)
    assert [r.label for r in rows if r.values] == ["Research and development"]
    assert rows[-1].values == [18497.0, 12914.0]


# ------------------------------------------------------------- live balance sheet parse


@pytest.mark.skipif(not PDF.exists(), reason="corpus PDF not present in data/raw")
def test_balance_sheet_page_141_values() -> None:
    pytest.importorskip("pdfplumber")
    from rag.ingest.validate import extract_statement_page, validate_balance_sheet

    table = extract_statement_page(PDF, 141)
    assert table.statement == "balance_sheet"
    assert table.unit == "USD_millions"
    assert table.period_ends == ["2026-01-25", "2025-01-26"]
    assert table.fiscal_years == [2026, 2025]

    assert table.row("inventories").values == [21403.0, 10080.0]
    assert table.row("total current liabilities").values == [32163.0, 18047.0]
    assert table.row("total current assets").values == [125605.0, 80126.0]
    assert table.row("total assets").values == [206803.0, 111601.0]
    assert table.row("goodwill").values == [20832.0, 5188.0]
    assert table.row(
        "non marketable equity securities"
    ).values  # hyphens normalise to spaces == [22251.0, 3387.0]
    assert table.row("short term debt").values == [999.0, None]
    assert table.row("long term debt").values == [7469.0, 8463.0]
    assert table.row("total shareholders' equity").values == [157293.0, 79327.0]
    assert table.row("inventories").group == "current_assets"

    validate_balance_sheet(table)
    assert table.identity is not None and table.identity.holds
    assert all(c.holds for c in table.subtotals)
    assert not any(r.label == "" for r in table.rows)


@pytest.mark.skipif(not PDF.exists(), reason="corpus PDF not present in data/raw")
def test_income_statement_and_cash_flow_pages_parse() -> None:
    pytest.importorskip("pdfplumber")
    from rag.ingest.validate import extract_statement_page

    income = extract_statement_page(PDF, 139)
    assert income.statement == "income_statement"
    assert income.row("revenue").values == [215938.0, 130497.0, 60922.0]
    assert income.row("net income").values == [120067.0, 72880.0, 29760.0]
    assert income.row("research and development").values == [18497.0, 12914.0, 8675.0]

    cash = extract_statement_page(PDF, 143)
    assert cash.statement == "cash_flow"
    assert cash.row("net cash provided by operating activities").values == [
        102718.0,
        64089.0,
        28090.0,
    ]
