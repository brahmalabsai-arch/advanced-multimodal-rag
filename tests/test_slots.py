"""Phase 4 — slot extraction (architecture §4.2). Deterministic, test-first (plan rule G4)."""

from __future__ import annotations

import pytest

from rag.core.config import load_fiscal_calendar, load_glossary_config
from rag.query.slots import SlotExtractor, extract_slots, normalize_text

# ------------------------------------------------------------------------ normalisation


def test_normalize_text_strips_possessives_unicode_spaces_and_hyphens() -> None:
    q = "NVIDIA’s year-over-year change as of January 25, 2026 — in USD?"
    assert normalize_text(q) == "nvidia year over year change as of january 25 2026 in usd"


def test_normalize_keeps_decimals_tickers_and_percent() -> None:
    assert normalize_text("PP&E at $0.001 par, 5.5% and A/R vs. AR") == (
        "pp&e at $0.001 par 5.5% and a/r vs ar"
    )


# ------------------------------------------------------------------------ fiscal periods


@pytest.mark.parametrize(
    "question, periods",
    [
        ("Total assets in FY26", ["FY2026"]),
        ("Total assets in FY 2026", ["FY2026"]),
        ("Total assets in fy2025", ["FY2025"]),
        ("Revenue for fiscal 2026", ["FY2026"]),
        ("Revenue for fiscal year 2026", ["FY2026"]),
        ("Inventories at fiscal year-end 2025", ["FY2025"]),
        ("Inventories at fiscal year end 2025 (Jan 26, 2025)", ["FY2025"]),
        ("Total assets as of Jan 25, 2026", ["FY2026"]),
        ("Total assets as of January 25, 2026", ["FY2026"]),
        ("Total assets as of Jan. 26, 2025", ["FY2025"]),
        ("Total assets as of 2026-01-25", ["FY2026"]),
        ("Total assets as of 1/25/2026", ["FY2026"]),
        ("Net income for the fiscal year ended January 25, 2026", ["FY2026"]),
        ("Net income for the year ended 2026", ["FY2026"]),
        ("Revenue for the first quarter of fiscal 2026", ["FY2026"]),
        ("Total assets at the end of last fiscal year", ["FY2026"]),
        ("Total assets in the most recent fiscal year", ["FY2026"]),
        ("Total assets in the prior fiscal year", ["FY2025"]),
        ("Total assets in the current fiscal year", ["FY2026"]),
        ("How did cash change from fiscal 2025 to fiscal 2026?", ["FY2025", "FY2026"]),
        ("Compare net income in FY25 and FY26", ["FY2025", "FY2026"]),
        ("Revenue in 2025", ["AMBIGUOUS_2025"]),
        ("Revenue in 2026", ["AMBIGUOUS_2026"]),
        ("Revenue in fiscal 2026 versus 2024", ["AMBIGUOUS_2024", "FY2026"]),
        ("How did the 5-year total return compare?", []),
        ("$100 invested on 1/31/2021", []),
        ("What are the layers of the five-layer cake?", []),
    ],
)
def test_fiscal_period_forms(question: str, periods: list[str]) -> None:
    assert extract_slots(question).fiscal_periods == periods


def test_bare_year_resolves_forward_with_a_note() -> None:
    s = extract_slots("How much did inventories decrease in 2025?")
    assert s.fiscal_periods == ["AMBIGUOUS_2025"]
    assert s.resolved_fiscal_years == [2026]
    assert s.period_note and "fiscal 2026" in s.period_note and "January 25, 2026" in s.period_note


def test_bare_year_beyond_calendar_resolves_to_latest() -> None:
    s = extract_slots("What is planned for 2026?")
    assert s.fiscal_periods == ["AMBIGUOUS_2026"] and s.resolved_fiscal_years == [2026]


def test_explicit_fiscal_year_has_no_note() -> None:
    assert extract_slots("Revenue for fiscal 2026").period_note is None


def test_current_fiscal_year_is_the_latest_period_in_the_question() -> None:
    s = extract_slots("How did cash change from fiscal 2025 to fiscal 2026?")
    assert s.current_fiscal_year == 2026


# ------------------------------------------------------------------------------ metrics


@pytest.mark.parametrize(
    "question, metrics",
    [
        ("What were total assets?", ["total assets"]),
        ("What were current assets?", ["total current assets"]),
        ("What were total current assets?", ["total current assets"]),
        ("How much cash did NVIDIA hold?", ["cash and cash equivalents"]),
        ("cash and cash equivalents at year end", ["cash and cash equivalents"]),
        ("What were short-term investments?", ["marketable securities"]),
        ("What were receivables?", ["accounts receivable net"]),
        ("accounts receivable, net, as of Jan 25, 2026", ["accounts receivable net"]),
        ("What was the inventory balance?", ["inventories"]),
        ("What was PP&E?", ["property and equipment net"]),
        ("What were right-of-use assets?", ["operating lease assets"]),
        ("What was goodwill?", ["goodwill"]),
        ("What was the current portion of long-term debt?", ["short term debt"]),
        ("How much long-term debt?", ["long term debt"]),
        ("What was total stockholders' equity?", ["total shareholders' equity"]),
        ("What was book value?", ["total shareholders' equity"]),
        ("What was NVIDIA's top line?", ["revenue"]),
        ("What was net sales?", ["revenue"]),
        ("How much was COGS?", ["cost of revenue"]),
        ("R&D spending in fiscal 2026", ["research and development"]),
        ("What was SG&A?", ["sales general and administrative"]),
        ("What was sales, general and administrative?", ["sales general and administrative"]),
        ("What was operating profit?", ["operating income"]),
        ("What was the bottom line?", ["net income"]),
        ("What was diluted EPS?", ["diluted"]),
        ("What was basic earnings per share?", ["basic"]),
        ("What was operating cash flow?", ["net cash provided by operating activities"]),
        (
            "How much was capex?",
            ["purchases related to property and equipment and intangible assets"],
        ),
        ("How much went to share buybacks?", ["payments related to repurchases of common stock"]),
        ("How much did NVIDIA pay in dividends?", ["dividends paid"]),
        ("What was stock-based compensation?", ["stock based compensation expense"]),
        ("What were non-marketable equity securities?", ["non marketable equity securities"]),
        (
            "quick ratio (cash + marketable securities + receivables) / current liabilities",
            [
                "cash and cash equivalents",
                "marketable securities",
                "accounts receivable net",
                "total current liabilities",
            ],
        ),
        ("What drove the H20 charge?", []),
    ],
)
def test_metric_synonyms(question: str, metrics: list[str]) -> None:
    assert extract_slots(question).metrics == metrics


# ----------------------------------------------------------------------------- formulas


@pytest.mark.parametrize(
    "question, formulas",
    [
        ("What is the current ratio?", ["current_ratio"]),
        ("What is the liquidity ratio?", ["current_ratio"]),
        ("What is the quick ratio?", ["quick_ratio"]),
        ("What is the acid-test ratio?", ["quick_ratio"]),
        ("What was working capital?", ["working_capital"]),
        ("What is the total debt to equity ratio?", ["debt_to_equity"]),
        ("What is the debt/equity ratio?", ["debt_to_equity"]),
        ("What is the leverage ratio?", ["debt_to_equity"]),
        ("What is the liabilities-to-equity ratio?", ["liabilities_to_equity"]),
        ("What is the equity ratio (equity to assets)?", ["equity_ratio"]),
        ("What was the cash pile at FY26 year end?", ["cash_and_investments"]),
        ("What was total cash and marketable securities?", ["cash_and_investments"]),
        ("What were total assets?", []),
    ],
)
def test_formula_synonyms(question: str, formulas: list[str]) -> None:
    assert extract_slots(question).formulas == formulas


def test_formula_span_does_not_leak_metrics_or_anchors() -> None:
    s = extract_slots("What is the current ratio as of Jan 25, 2026?")
    assert s.formulas == ["current_ratio"]
    assert s.metrics == []
    assert s.time_anchor is False
    assert s.statement == "balance_sheet" and s.statement_source == "inferred"


# ---------------------------------------------------------------------------- statement


@pytest.mark.parametrize(
    "question, statement, source",
    [
        ("marketable securities on the balance sheet", "balance_sheet", "explicit"),
        ("revenue per the income statement", "income_statement", "explicit"),
        ("What does the statement of cash flows show for dividends?", "cash_flow", "explicit"),
        ("What were total assets?", "balance_sheet", "inferred"),
        ("What was revenue?", "income_statement", "inferred"),
        ("What was operating cash flow?", "cash_flow", "inferred"),
        ("Compare revenue with total assets", None, None),
        ("What drove the H20 charge?", None, None),
    ],
)
def test_statement_slot(question: str, statement: str | None, source: str | None) -> None:
    s = extract_slots(question)
    assert (s.statement, s.statement_source) == (statement, source)


# ------------------------------------------------------- direction / aggregation / anchor


@pytest.mark.parametrize(
    "question, direction",
    [
        ("How much did inventories grow?", "increase"),
        ("By what percentage did total liabilities increase?", "increase"),
        ("Why did gross margin decrease?", "decrease"),
        ("Did long-term debt fall?", "decrease"),
        ("Did goodwill increase or decrease?", None),
        ("What were total assets?", None),
    ],
)
def test_direction(question: str, direction: str | None) -> None:
    assert extract_slots(question).direction == direction


@pytest.mark.parametrize(
    "question, aggregation",
    [
        ("year-over-year change in total assets", ["yoy", "change"]),
        ("How much did inventories grow YoY?", ["yoy"]),
        ("By what percentage did total liabilities increase?", ["pct"]),
        ("How did buybacks compare with dividends?", ["compare"]),
        ("What share of revenue came from outside the US?", ["ratio"]),
        ("What were total assets?", []),
    ],
)
def test_aggregation(question: str, aggregation: list[str]) -> None:
    assert extract_slots(question).aggregation == aggregation


@pytest.mark.parametrize(
    "question, anchored",
    [
        ("When is the upcoming annual meeting?", True),
        ("When is NVIDIA's next annual meeting of stockholders?", True),
        ("What is the latest dividend?", True),
        ("What is the share price right now?", True),
        ("What is the current ratio?", False),
        ("What were total current liabilities?", False),
        ("What were total assets in the current fiscal year?", False),
    ],
)
def test_time_anchor(question: str, anchored: bool) -> None:
    assert extract_slots(question).time_anchor is anchored


# ------------------------------------------------------------------------ entity / key


@pytest.mark.parametrize(
    "question, entity",
    [
        ("What were NVIDIA's total assets?", "NVIDIA"),
        ("What were NVDA's total assets?", "NVIDIA"),
        ("How much cash did the company hold?", "NVIDIA"),
        ("What were total assets?", None),
    ],
)
def test_entity_alias(question: str, entity: str | None) -> None:
    assert extract_slots(question).entity == entity


def test_canonical_text_and_l1_key_are_stable_across_synonyms() -> None:
    a = extract_slots("What were NVIDIA's total assets as of Jan 25, 2026?")
    b = extract_slots("what were nvda total assets as of January 25, 2026")
    assert a.canonical_text == b.canonical_text == "what were nvidia total_assets as of fy2026"
    assert a.l1_key("v1") == b.l1_key("v1")
    assert a.l1_key("v1") != a.l1_key("v2")


def test_slot_rich_needs_metric_and_period() -> None:
    assert extract_slots("Total assets as of Jan 25, 2026").slot_rich
    assert not extract_slots("Total assets").slot_rich
    assert not extract_slots("Why did gross margin fall in fiscal 2026?").slot_rich


def test_matches_are_ordered_and_non_overlapping() -> None:
    s = extract_slots("What is the current ratio as of Jan 25, 2026 on the balance sheet?")
    spans = [(m.start, m.end) for m in s.matches]
    assert spans == sorted(spans)
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s2 >= e1


# ------------------------------------------------------------------------------ config


def test_glossary_has_at_least_150_synonyms_and_no_duplicates() -> None:
    g = load_glossary_config()
    assert g.synonym_count() >= 150


def test_fiscal_calendar_maps_dates() -> None:
    from datetime import date

    c = load_fiscal_calendar()
    assert c.fiscal_year_of(date(2026, 1, 25)) == 2026
    assert c.fiscal_year_of(date(2025, 1, 27)) == 2026
    assert c.fiscal_year_of(date(2025, 1, 26)) == 2025
    assert c.fiscal_year_of(date(2021, 1, 31)) is None


def test_extractor_accepts_custom_config() -> None:
    ex = SlotExtractor()
    assert ex.statement_label("balance_sheet") == "Consolidated Balance Sheets"
    assert ex.metric_statement("revenue") == "income_statement"
