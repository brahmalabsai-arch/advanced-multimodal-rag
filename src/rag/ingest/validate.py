"""Phase 1 — independent extraction and validation of the financial statements (architecture §3.1).

pdfplumber reads the statement pages straight from the text layer (exact character positions);
Docling's TableFormer parse of the same pages is compared cell by cell. On a numeric disagreement
pdfplumber wins and the disagreement is logged. The accounting identity
`Total assets == Total liabilities and shareholders' equity` must hold for every period column,
otherwise ingestion stops (P3: numbers are sacred).

Pure helpers (`normalize_number`, `parse_row_tokens`, `check_identity`, grouping) have no PDF
dependency and are unit-tested from the dev environment; `extract_statement_page` needs pdfplumber.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.core.logging import get_logger
from rag.ingest.sections import Statement, detect_statement

log = get_logger(__name__)


class ValidationError(RuntimeError):
    """A hard validation failure that must stop ingestion."""


# ------------------------------------------------------------------ numeric normaliser

DASHES = {"-", "–", "—", "‒", "―", "−"}
_FOOTNOTE_SUFFIX = re.compile(r"(\(\s*[a-z]\s*\)|[*†‡§]+)$", re.I)
_NUMBER = re.compile(r"^\(?\$?\s*(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s*\)?%?$")
_CURRENCY_ONLY = re.compile(r"^\$$")


class NotNumeric(ValueError):
    pass


def is_dash(token: str) -> bool:
    return token.strip() in DASHES


def normalize_number(token: str) -> float | None:
    """`"$ 10,605"` → 10605.0; `"(1,234)"` → -1234.0; `"—"` → None; footnote markers stripped.

    Raises `NotNumeric` for anything else (labels, words, standalone `$`).
    """
    text = token.strip()
    if not text:
        raise NotNumeric(token)
    text = _FOOTNOTE_SUFFIX.sub("", text).strip()
    if is_dash(text):
        return None
    if _CURRENCY_ONLY.match(text):
        raise NotNumeric(token)
    if not _NUMBER.match(text):
        raise NotNumeric(token)
    negative = text.startswith("(") and text.endswith(")")
    digits = re.sub(r"[^\d.]", "", text)
    value = float(digits)
    return -value if negative else value


def is_numeric_token(token: str) -> bool:
    try:
        normalize_number(token)
        return True
    except NotNumeric:
        return False


def normalize_label(label: str) -> str:
    text = label.replace("’", "'").replace("‘", "'").lower()
    text = re.sub(r"\(\s*[a-z0-9]\s*\)", " ", text)  # footnote markers
    text = re.sub(r"[^a-z0-9$%.' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .:")
    return text


_FOOTER_PREFIXES = ("see accompanying notes", "see notes to", "the accompanying notes")


def is_page_furniture(label: str) -> bool:
    return normalize_label(label).startswith(_FOOTER_PREFIXES)


# ------------------------------------------------------------------------ row parsing


@dataclass
class ParsedRow:
    label: str
    values: list[float | None]
    raw: str = ""
    top: float = 0.0
    x0: float | None = None  # left edge of the first token; None when unknown (Docling grids)


def parse_row_tokens(tokens: list[str], n_periods: int) -> ParsedRow:
    """Split a statement line into a label and up to `n_periods` trailing numeric values.

    Values are read from the right so labels that contain numbers ("see Note 12",
    "$0.001 par value") are not mistaken for cells. Standalone `$` tokens are skipped.
    """
    values: list[float | None] = []
    idx = len(tokens)
    while idx > 0 and len(values) < n_periods:
        token = tokens[idx - 1]
        if _CURRENCY_ONLY.match(token.strip()):
            idx -= 1
            continue
        try:
            values.append(normalize_number(token))
        except NotNumeric:
            break
        idx -= 1
    # A trailing `$` that belonged to the last consumed value.
    while idx > 0 and _CURRENCY_ONLY.match(tokens[idx - 1].strip()):
        idx -= 1
    label = " ".join(tokens[:idx]).strip()
    values.reverse()
    return ParsedRow(label=label, values=values, raw=" ".join(tokens))


INDENT_TOLERANCE_PT = 2.0


def _ends_mid_sentence(label: str) -> bool:
    """A wrapped fragment ends on a lowercase word or a joining mark; a title line does not."""
    words = label.split()
    if not words:
        return False
    last = words[-1]
    return last[-1] in ",;-–—/" or (last[0].islower() and last.isalpha())


def _same_indent(a: ParsedRow, b: ParsedRow) -> bool:
    """Unknown positions (Docling grids) are treated as same-indent, i.e. as wrapping."""
    if a.x0 is None or b.x0 is None:
        return True
    return abs(a.x0 - b.x0) <= INDENT_TOLERANCE_PT


def merge_wrapped_labels(rows: list[ParsedRow]) -> list[ParsedRow]:
    """Re-join labels that wrapped onto several lines.

    A value-less line is a *fragment* of the next line when the next line starts at the same
    indent (wrapped cell text keeps its left edge). A value-less line followed by a more-indented
    line is a group header ("Operating expenses" above "Research and development"). Fragments
    that precede a ':'-terminated line form one header with it.
    """
    merged: list[ParsedRow] = []
    pending: list[ParsedRow] = []
    for i, row in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if not row.values:
            if row.label.endswith(":"):
                if pending and _ends_mid_sentence(pending[-1].label):
                    # "…net cash provided by operating" + "activities:" is one wrapped header
                    label = " ".join([*(p.label for p in pending), row.label])
                    row = ParsedRow(label, [], row.raw, pending[0].top, pending[0].x0)
                    pending = []
                elif pending:
                    # "Assets" above "Current assets:" is a title line, not a fragment
                    merged.extend(pending)
                    pending = []
                merged.append(row)
            elif not row.label:
                continue
            elif nxt is not None and _same_indent(row, nxt):
                pending.append(row)
            else:
                # Header without a colon (or last line): keep as a value-less row.
                if pending:
                    label = " ".join([*(p.label for p in pending), row.label])
                    row = ParsedRow(label, [], row.raw, pending[0].top, pending[0].x0)
                    pending = []
                merged.append(row)
            continue
        if pending:
            label = " ".join([*(p.label for p in pending), row.label])
            row = ParsedRow(label, row.values, row.raw, pending[0].top, pending[0].x0)
            pending = []
        merged.append(row)
    if pending:
        label = " ".join(p.label for p in pending)
        merged.append(ParsedRow(label, [], "", pending[0].top, pending[0].x0))
    return merged


# ------------------------------------------------------------------------ row groups

_BS_TOTALS = {
    "total current assets": "current_assets",
    "total assets": "assets",
    "total current liabilities": "current_liabilities",
    "total liabilities": "liabilities",
    "total shareholders' equity": "equity",
    "total stockholders' equity": "equity",
    "total liabilities and shareholders' equity": "liabilities_and_equity",
    "total liabilities and stockholders' equity": "liabilities_and_equity",
}


def assign_balance_sheet_groups(rows: list[ParsedRow]) -> list[str]:
    """`current_assets`, `non_current_assets`, `assets`, `current_liabilities`,
    `non_current_liabilities`, `liabilities`, `equity`, `liabilities_and_equity`, `header`."""
    groups: list[str] = []
    current = "header"
    for row in rows:
        norm = normalize_label(row.label)
        if not row.values:
            if norm.startswith("current assets"):
                current = "current_assets"
            elif norm.startswith("current liabilities"):
                current = "current_liabilities"
            elif "equity" in norm and "liabilities" not in norm:
                current = "equity"
            elif norm == "assets":
                current = "assets_header"
            elif "liabilities" in norm:
                current = "liabilities_header"
            groups.append("header")
            continue
        if current == "done":
            groups.append("footer")
            continue
        total_group = _BS_TOTALS.get(norm)
        if total_group is not None:
            groups.append(total_group)
            current = {
                "current_assets": "non_current_assets",
                "assets": "liabilities_header",
                "current_liabilities": "non_current_liabilities",
                "liabilities": "commitments",
                "equity": "equity",
                "liabilities_and_equity": "done",
            }[total_group]
            continue
        if current in {"assets_header", "liabilities_header", "commitments", "header"}:
            # Rows outside a named block: commitments line, or an unusual layout.
            groups.append("other")
        else:
            groups.append(current)
    return groups


def assign_generic_groups(rows: list[ParsedRow]) -> list[str]:
    groups: list[str] = []
    current = "body"
    for row in rows:
        if not row.values:
            norm = normalize_label(row.label)
            norm = re.sub(r"^(cash flows? (from|used in) )", "", norm)
            norm = re.sub(r"^adjustments to reconcile .*", "adjustments", norm)
            current = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")[:60] or "body"
            groups.append("header")
        else:
            groups.append(current)
    return groups


# ----------------------------------------------------------------------- data model


class StatementRow(BaseModel):
    label: str
    label_norm: str
    group: str
    values: list[float | None]
    docling_values: list[float | None] | None = None
    agree: bool | None = None  # None = no Docling counterpart found
    source: str = "pdfplumber"


class Disagreement(BaseModel):
    label: str
    column: int
    pdfplumber: float | None
    docling: float | None


class IdentityCheck(BaseModel):
    left_label: str
    right_label: str
    left: list[float | None]
    right: list[float | None]
    holds: bool


class SubtotalCheck(BaseModel):
    total_label: str
    group: str
    expected: list[float | None]
    computed: list[float | None]
    holds: bool


class StatementTable(BaseModel):
    statement: Statement
    page: int
    title: str
    unit: str
    period_ends: list[str] = Field(description="ISO dates, one per value column")
    fiscal_years: list[int]
    rows: list[StatementRow]
    disagreements: list[Disagreement] = Field(default_factory=list)
    identity: IdentityCheck | None = None
    subtotals: list[SubtotalCheck] = Field(default_factory=list)
    docling_rows_matched: int = 0
    docling_rows_total: int = 0

    def row(self, label_norm: str) -> StatementRow | None:
        for r in self.rows:
            if r.label_norm == label_norm:
                return r
        return None


class StatementsFile(BaseModel):
    pdf_sha256: str
    statements: dict[str, StatementTable]


# ----------------------------------------------------------------- period detection

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE = re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
_MONTH_NUM["sept"] = 9


def find_period_ends(text: str, max_periods: int = 4) -> list[date]:
    """Dates like "Jan 25, 2026" in the column header area, left to right."""
    found: list[date] = []
    for m in _DATE.finditer(text):
        month = _MONTH_NUM[m.group(1).lower()[:3]] if m.group(1).lower() != "sept" else 9
        d = date(int(m.group(3)), month, int(m.group(2)))
        if d not in found:
            found.append(d)
        if len(found) >= max_periods:
            break
    return found


def fiscal_year_of(period_end: date) -> int:
    """NVIDIA's fiscal year is named by the calendar year in which it ends (late January)."""
    return period_end.year


def detect_unit(text: str) -> str:
    head = text[:600].lower()
    if "in millions" in head:
        return "USD_millions"
    if "in thousands" in head:
        return "USD_thousands"
    if "in billions" in head:
        return "USD_billions"
    return "USD"


# ------------------------------------------------------------------- pdfplumber path


def _lines_from_words(words: list[dict[str, Any]], y_tolerance: float = 3.0) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def extract_statement_page(pdf_path: Path, page_no: int) -> StatementTable:
    """pdfplumber extraction of one statement page into labelled, grouped rows."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_no - 1]
        text = page.extract_text() or ""
        words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)

    lines = _lines_from_words(words)
    line_texts = [" ".join(w["text"] for w in line) for line in lines]

    title = ""
    statement: Statement = "none"
    for candidate in line_texts[:6]:
        statement = detect_statement(candidate)
        if statement != "none":
            title = candidate.strip()
            break
    if statement == "none":
        raise ValidationError(f"page {page_no} has no recognised statement heading")

    period_ends = find_period_ends("\n".join(line_texts[:12]))
    if len(period_ends) < 2:
        raise ValidationError(f"page {page_no}: could not find period-end dates in the header")
    n_periods = len(period_ends)
    unit = detect_unit(text)

    parsed: list[ParsedRow] = []
    header_done = False
    for line, ltext in zip(lines, line_texts, strict=True):
        if not header_done:
            if _DATE.search(ltext):
                header_done = True
            continue
        tokens = [w["text"] for w in line]
        row = parse_row_tokens(tokens, n_periods)
        row.top = float(line[0]["top"])
        row.x0 = float(line[0]["x0"])
        parsed.append(row)
    parsed = merge_wrapped_labels(parsed)
    parsed = [r for r in parsed if r.label or r.values]

    # Drop page furniture (page numbers, "See accompanying notes...") that carry no values.
    parsed = [
        r
        for r in parsed
        if r.values
        or r.label.endswith(":")
        or normalize_label(r.label) in {"assets"}
        or "equity" in normalize_label(r.label)
    ]

    groups = (
        assign_balance_sheet_groups(parsed)
        if statement == "balance_sheet"
        else assign_generic_groups(parsed)
    )
    rows = [
        StatementRow(
            label=r.label.rstrip(":"),
            label_norm=normalize_label(r.label),
            group=g,
            values=r.values + [None] * (n_periods - len(r.values)),
        )
        for r, g in zip(parsed, groups, strict=True)
        # header rows are kept only as group markers; footers and bare page numbers are furniture
        if r.values and r.label and g != "footer" and not is_page_furniture(r.label)
    ]
    return StatementTable(
        statement=statement,
        page=page_no,
        title=title,
        unit=unit,
        period_ends=[d.isoformat() for d in period_ends],
        fiscal_years=[fiscal_year_of(d) for d in period_ends],
        rows=rows,
    )


# ----------------------------------------------------------- Docling cross-validation


def docling_grid_to_rows(grid: list[list[str]], n_periods: int) -> list[ParsedRow]:
    """Turn a Docling table grid (cell texts) into label/value rows using the same rules."""
    rows: list[ParsedRow] = []
    for cells in grid:
        if not cells or not (cells[0] or "").strip():
            continue  # column-header rows ("", "Jan 25, 2026", …) have no label cell
        tokens: list[str] = []
        for cell in cells:
            cell = (cell or "").strip()
            if not cell:
                continue
            # Multi-token cells (e.g. "$ 10,605") split like the text layer would.
            tokens.extend(cell.split())
        if not tokens:
            continue
        rows.append(parse_row_tokens(tokens, n_periods))
    # Docling cells already hold whole (unwrapped) labels; value-less rows are headers.
    return rows


def compare_with_docling(table: StatementTable, docling_rows: list[ParsedRow]) -> StatementTable:
    """Cell-by-cell comparison; pdfplumber values stay authoritative, disagreements are logged.

    Rows are matched by normalised label *in order* (monotonic cursor), so repeated labels such
    as "Basic" / "Diluted" (per-share vs share counts) or "Other" pair with the right twin.
    """
    valued = [(normalize_label(r.label), r) for r in docling_rows if r.values]
    n = len(table.period_ends)
    matched = 0
    cursor = 0
    disagreements: list[Disagreement] = []
    for row in table.rows:
        d = None
        for j in range(cursor, len(valued)):
            if valued[j][0] == row.label_norm:
                d, cursor = valued[j][1], j + 1
                break
        if d is None:  # fall back to any unmatched-position match (row order differs)
            for j in range(0, cursor):
                if valued[j][0] == row.label_norm:
                    d = valued[j][1]
                    break
        if d is None:
            row.docling_values, row.agree = None, None
            continue
        matched += 1
        dvals = (d.values + [None] * n)[:n]
        row.docling_values = dvals
        row.agree = True
        for col, (a, b) in enumerate(zip(row.values, dvals, strict=True)):
            if a != b:
                row.agree = False
                disagreements.append(
                    Disagreement(label=row.label, column=col, pdfplumber=a, docling=b)
                )
                log.warning(
                    "%s p%d %r col%d: pdfplumber=%s docling=%s (pdfplumber wins)",
                    table.statement,
                    table.page,
                    row.label,
                    col,
                    a,
                    b,
                )
    table.docling_rows_matched = matched
    table.docling_rows_total = len([r for r in docling_rows if r.values])
    table.disagreements = disagreements
    return table


# ----------------------------------------------------------------- accounting checks


def check_identity(
    rows: list[StatementRow],
    left_label: str = "total assets",
    right_label: str = "total liabilities and shareholders' equity",
) -> IdentityCheck:
    left = next((r for r in rows if r.label_norm == left_label), None)
    right = next((r for r in rows if r.label_norm == right_label), None)
    if left is None or right is None:
        raise ValidationError(
            f"identity rows missing: {left_label!r}={left is not None}, "
            f"{right_label!r}={right is not None}"
        )
    holds = all(
        a is not None and b is not None and a == b
        for a, b in zip(left.values, right.values, strict=True)
    )
    return IdentityCheck(
        left_label=left.label,
        right_label=right.label,
        left=left.values,
        right=right.values,
        holds=holds,
    )


def check_subtotals(table: StatementTable) -> list[SubtotalCheck]:
    """Soft checks: each `Total X` equals the sum of its group's rows (warn only)."""
    checks: list[SubtotalCheck] = []
    n = len(table.period_ends)
    group_sums: dict[str, list[float]] = {}
    for row in table.rows:
        if row.label_norm.startswith("total "):
            continue
        sums = group_sums.setdefault(row.group, [0.0] * n)
        for i, v in enumerate(row.values):
            sums[i] += v or 0.0
    for row in table.rows:
        if row.group in group_sums and row.label_norm.startswith("total "):
            computed = group_sums[row.group]
            holds = all(
                v is not None and abs(v - c) < 0.5
                for v, c in zip(row.values, computed, strict=True)
            )
            checks.append(
                SubtotalCheck(
                    total_label=row.label,
                    group=row.group,
                    expected=row.values,
                    computed=computed,
                    holds=holds,
                )
            )
    return checks


def validate_balance_sheet(table: StatementTable) -> StatementTable:
    """Hard identity assertion (stops ingestion) plus soft subtotal checks."""
    table.identity = check_identity(table.rows)
    table.subtotals = check_subtotals(table)
    for c in table.subtotals:
        if not c.holds:
            log.warning(
                "Subtotal mismatch %r: stated=%s computed=%s", c.total_label, c.expected, c.computed
            )
    if not table.identity.holds:
        raise ValidationError(
            "Accounting identity failed on the balance sheet: "
            f"total assets {table.identity.left} != "
            f"total liabilities and equity {table.identity.right}"
        )
    log.info("Balance sheet identity holds: %s", table.identity.left)
    return table
