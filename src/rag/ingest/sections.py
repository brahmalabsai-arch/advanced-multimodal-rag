"""Section tagging (architecture §3.2).

Page ranges from corpus inspection seed the `section`; Docling headings refine the
`subsection`. Statement pages are detected by their headings. Pure logic — no Docling
imports — so it is unit-testable from the serving/dev environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Section = Literal["annual_review", "proxy", "form_10k", "back_cover"]
Statement = Literal["balance_sheet", "income_statement", "cash_flow", "none"]

# Inclusive page ranges from problem statement §2 / architecture §3.2.
SECTION_RANGES: tuple[tuple[int, int, Section], ...] = (
    (1, 18, "annual_review"),
    (19, 88, "proxy"),
    (89, 174, "form_10k"),
    (175, 175, "back_cover"),
)

DEFAULT_SUBSECTION: dict[Section, str] = {
    "annual_review": "narrative_spread",
    "proxy": "proxy_front_matter",
    "form_10k": "form_10k_front_matter",
    "back_cover": "back_cover",
}


def page_section(page: int) -> Section:
    for lo, hi, name in SECTION_RANGES:
        if lo <= page <= hi:
            return name
    raise ValueError(f"page {page} is outside the known page ranges (1–175)")


# ---------------------------------------------------------------------- statements

# Unanchored: Docling merges the running head, e.g.
# "NVIDIA Corporation and Subsidiaries Consolidated Balance Sheets (In millions, except par value)".
STATEMENT_HEADINGS: tuple[tuple[re.Pattern[str], Statement], ...] = (
    (re.compile(r"\bconsolidated balance sheets?\b", re.I), "balance_sheet"),
    (re.compile(r"\bconsolidated statements? of income\b", re.I), "income_statement"),
    (re.compile(r"\bconsolidated statements? of cash flows?\b", re.I), "cash_flow"),
)

# Any primary statement page (also comprehensive income, shareholders' equity).
FINANCIAL_STATEMENT_HEADING = re.compile(
    r"\bconsolidated (balance sheets?|statements? of [a-z' ]+)\b", re.I
)


def detect_statement(heading: str) -> Statement:
    text = _clean(heading)
    for pattern, statement in STATEMENT_HEADINGS:
        if pattern.search(text):
            return statement
    return "none"


# --------------------------------------------------------------------- subsections

# (regex on the cleaned heading, subsection tag). First match wins; evaluated per section.
_10K_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^item\s+1\.\s*business", re.I), "item_1_business"),
    (re.compile(r"^item\s+1a\.", re.I), "item_1a_risk_factors"),
    (re.compile(r"^item\s+1b\.", re.I), "item_1b_unresolved_staff_comments"),
    (re.compile(r"^item\s+1c\.", re.I), "item_1c_cybersecurity"),
    (re.compile(r"^item\s+2\.", re.I), "item_2_properties"),
    (re.compile(r"^item\s+3\.", re.I), "item_3_legal_proceedings"),
    (re.compile(r"^item\s+4\.", re.I), "item_4_mine_safety"),
    (re.compile(r"^item\s+5\.", re.I), "item_5_market"),
    (re.compile(r"^item\s+6\.", re.I), "item_6_reserved"),
    (re.compile(r"^item\s+7\.", re.I), "item_7_mdna"),
    (re.compile(r"^item\s+7a\.", re.I), "item_7a_market_risk"),
    (re.compile(r"^item\s+8\.", re.I), "item_8_financials"),
    (re.compile(r"^item\s+9\.", re.I), "item_9_accountant_changes"),
    (re.compile(r"^item\s+9a\.", re.I), "item_9a_controls"),
    (re.compile(r"^item\s+9b\.", re.I), "item_9b_other_information"),
    (re.compile(r"^item\s+9c\.", re.I), "item_9c_foreign_jurisdictions"),
    (re.compile(r"^item\s+10\.", re.I), "item_10_directors"),
    (re.compile(r"^item\s+11\.", re.I), "item_11_executive_compensation"),
    (re.compile(r"^item\s+12\.", re.I), "item_12_security_ownership"),
    (re.compile(r"^item\s+13\.", re.I), "item_13_relationships"),
    (re.compile(r"^item\s+14\.", re.I), "item_14_accountant_fees"),
    (re.compile(r"^item\s+15\.", re.I), "item_15_exhibits"),
    (re.compile(r"^item\s+16\.", re.I), "item_16_summary"),
    (re.compile(r"\bnotes to (the )?consolidated financial statements", re.I), "notes"),
    (re.compile(r"^note\s+\d+\s*[-–—]", re.I), "notes"),
    (FINANCIAL_STATEMENT_HEADING, "item_8_financials"),
    (
        re.compile(r"^report of independent registered public accounting firm", re.I),
        "auditor_report",
    ),
    (re.compile(r"^signatures?$", re.I), "signatures"),
    (re.compile(r"^exhibit index", re.I), "exhibit_index"),
)

_PROXY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^notice of (the )?(\d{4} )?annual meeting", re.I), "notice_of_meeting"),
    (re.compile(r"^proxy (statement )?summary", re.I), "proxy_summary"),
    (re.compile(r"^(proposal \d|election of directors)", re.I), "proposals"),
    (
        re.compile(r"^(corporate governance|board of directors|director independence)", re.I),
        "corporate_governance",
    ),
    (
        re.compile(r"^(executive compensation|compensation discussion and analysis)", re.I),
        "compensation",
    ),
    (
        re.compile(
            r"^(summary compensation table|grants of plan-based|outstanding equity awards)", re.I
        ),
        "compensation",
    ),
    (re.compile(r"^(pay versus performance|pay ratio|ceo pay ratio)", re.I), "pay_vs_performance"),
    (
        re.compile(
            r"^(security ownership|stock ownership|equity compensation plan information)", re.I
        ),
        "ownership",
    ),
    (
        re.compile(r"^(audit matters|report of the audit committee|audit committee report)", re.I),
        "audit_matters",
    ),
    (
        re.compile(r"^(additional information|questions and answers|other matters)", re.I),
        "additional_information",
    ),
)

_ANNUAL_REVIEW_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(dear (fellow )?shareholders|to our shareholders|letter to shareholders)", re.I
        ),
        "shareholder_letter",
    ),
)

SUBSECTION_PATTERNS: dict[Section, tuple[tuple[re.Pattern[str], str], ...]] = {
    "annual_review": _ANNUAL_REVIEW_PATTERNS,
    "proxy": _PROXY_PATTERNS,
    "form_10k": _10K_PATTERNS,
    "back_cover": (),
}

_ITEM_HEADING = re.compile(r"^item\s+\d{1,2}[abc]?\.", re.I)
# A page with this many "Item N." *headings* is a table of contents. Set high on purpose:
# p. 121 legitimately carries Items 2, 3, 4 and 5, and the real 10-K contents page is a
# Docling `document_index`, not headings.
TOC_ITEM_THRESHOLD = 8


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def subsection_for_heading(section: Section, heading: str) -> str | None:
    """The subsection a heading opens, or None if the heading is not a section marker."""
    text = _clean(heading)
    for pattern, tag in SUBSECTION_PATTERNS[section]:
        if pattern.search(text):
            return tag
    return None


@dataclass(frozen=True)
class Heading:
    page: int
    text: str


def assign_subsections(headings: list[Heading], last_page: int = 175) -> dict[int, str]:
    """Map every page to a subsection.

    Headings are walked in reading order. A page is tagged with the first marker heading it
    carries; pages that follow continue from the *last* marker on that page (p. 121 opens with
    Item 2 but ends inside Item 5). Table-of-contents pages (≥ `TOC_ITEM_THRESHOLD` item
    headings) are tagged `table_of_contents` and do not move the cursor.
    """
    per_page_items: dict[int, int] = {}
    for h in headings:
        if _ITEM_HEADING.search(_clean(h.text)):
            per_page_items[h.page] = per_page_items.get(h.page, 0) + 1
    toc_pages = {p for p, n in per_page_items.items() if n >= TOC_ITEM_THRESHOLD}

    markers: dict[int, list[str]] = {}
    for h in headings:  # reading order
        if h.page in toc_pages:
            continue
        tag = subsection_for_heading(page_section(h.page), h.text)
        if tag is not None:
            markers.setdefault(h.page, []).append(tag)

    result: dict[int, str] = {}
    current: dict[Section, str] = {}
    for page in range(1, last_page + 1):
        section = page_section(page)
        if page in toc_pages:
            result[page] = "table_of_contents"
            continue
        tags = markers.get(page)
        if tags:
            result[page] = tags[0]
            current[section] = tags[-1]
        else:
            result[page] = current.get(section, DEFAULT_SUBSECTION[section])
    return result
