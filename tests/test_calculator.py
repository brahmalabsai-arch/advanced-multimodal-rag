"""Phase 3 tests — deterministic calculator against golden values (problem statement §10)."""

from __future__ import annotations

import pytest

from rag.calc.calculator import (
    Calculator,
    RowFactSource,
    evaluate_expression,
    select_formulas,
)
from rag.core.schema import Chunk, ChunkMetadata

# Balance sheet values verified in Phase 1 (USD millions).
BS = {
    "total current assets": (125605.0, 80126.0),
    "total current liabilities": (32163.0, 18047.0),
    "cash and cash equivalents": (10605.0, 8589.0),
    "marketable securities": (51951.0, 34621.0),
    "accounts receivable net": (38466.0, 23065.0),
    "short term debt": (999.0, None),
    "long term debt": (7469.0, 8463.0),
    "total shareholders' equity": (157293.0, 79327.0),
    "total liabilities": (49510.0, 32274.0),
    "total assets": (206803.0, 111601.0),
    "inventories": (21403.0, 10080.0),
    "goodwill": (20832.0, 5188.0),
    "non marketable equity securities": (22251.0, 3387.0),
}


def _rowfacts() -> list[Chunk]:
    out = []
    for norm, (v26, v25) in BS.items():
        cid = "rowfact_p141_" + norm.replace(" ", "_").replace(",", "").replace("'", "")
        out.append(
            Chunk(
                id=cid,
                document=f"NVIDIA Consolidated Balance Sheets — {norm}",
                metadata=ChunkMetadata(
                    chunk_id=cid,
                    modality="row_fact",
                    section="form_10k",
                    subsection="item_8_financials",
                    page=141,
                    breadcrumb="Form 10-K > Item 8 > Consolidated Balance Sheets",
                    statement="balance_sheet",
                    token_count=30,
                    line_item=norm.title(),
                    line_item_norm=norm,
                    line_group="x",
                    value_fy2026=v26,
                    value_fy2025=v25,
                    unit="USD_millions",
                    period_end_fy2026="2026-01-25",
                    period_end_fy2025="2025-01-26",
                ),
            )
        )
    return out


@pytest.fixture
def calc() -> Calculator:
    return Calculator(RowFactSource(_rowfacts()))


def test_expression_evaluator_is_restricted() -> None:
    assert evaluate_expression("(a + b) / c * 100", {"a": 1, "b": 2, "c": 4}) == 75.0
    assert evaluate_expression("abs(x)", {"x": -3}) == 3
    with pytest.raises(ValueError):
        evaluate_expression("__import__('os')", {})
    with pytest.raises(ValueError):
        evaluate_expression("a ** 99999999", {"a": 2})


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("current_ratio", 3.91),
        ("working_capital", 93442.0),
        ("quick_ratio", 3.14),
        ("debt_to_equity", 0.054),
        ("liabilities_to_equity", 0.315),
        ("equity_ratio", 0.761),
        ("cash_and_investments", 62556.0),
    ],
)
def test_golden_ratios_fy2026(calc: Calculator, formula: str, expected: float) -> None:
    result = calc.compute(formula, fiscal_year=2026)
    assert result.status == "ok"
    assert result.rounded == expected
    assert result.fiscal_year == 2026
    assert all(i.chunk_id.startswith("rowfact_p141_") and i.page == 141 for i in result.inputs)


def test_prior_year_ratio(calc: Calculator) -> None:
    result = calc.compute("current_ratio", fiscal_year=2025)
    assert result.rounded == pytest.approx(round(80126 / 18047, 2))
    assert result.inputs[0].fiscal_year == 2025


@pytest.mark.parametrize(
    "metric, expected_pct, expected_abs",
    [
        ("total assets", 85.3, 95202.0),
        ("inventories", 112.3, 11323.0),
        ("goodwill", 301.5, 15644.0),
        ("non marketable equity securities", 557.0, 18864.0),
    ],
)
def test_golden_yoy(
    calc: Calculator, metric: str, expected_pct: float, expected_abs: float
) -> None:
    pct = calc.compute("yoy_change_pct", fiscal_year=2026, metric=metric)
    assert pct.status == "ok" and pct.rounded == expected_pct
    assert pct.kind == "pct"
    absolute = calc.compute("yoy_change_abs", fiscal_year=2026, metric=metric)
    assert absolute.rounded == expected_abs


def test_optional_nil_input_counts_as_zero(calc: Calculator) -> None:
    # FY2025 short-term debt is nil → debt/equity uses long-term debt only.
    result = calc.compute("debt_to_equity", fiscal_year=2025)
    assert result.status == "ok"
    assert result.rounded == round(8463 / 79327, 3)
    nil_input = next(i for i in result.inputs if i.name == "short_term_debt")
    assert nil_input.value == 0.0 and nil_input.nil is True


def test_missing_inputs_reported_not_guessed(calc: Calculator) -> None:
    result = calc.compute("yoy_change_pct", fiscal_year=2026, metric="deferred revenue")
    assert result.status == "missing_inputs"
    assert result.result is None
    assert "deferred revenue" in " ".join(result.missing)
    assert "not found" in result.message.lower()


def test_generic_formula_requires_metric(calc: Calculator) -> None:
    result = calc.compute("yoy_change_pct", fiscal_year=2026)
    assert result.status == "missing_inputs"


def test_unknown_formula(calc: Calculator) -> None:
    with pytest.raises(KeyError):
        calc.compute("nope", fiscal_year=2026)


def test_select_formulas_by_keyword(calc: Calculator) -> None:
    line_items = list(BS)
    assert [
        r.formula
        for r in select_formulas("What is the current ratio as of Jan 25, 2026?", line_items)
    ] == ["current_ratio"]
    reqs = select_formulas("How much did goodwill grow year-over-year?", line_items)
    assert {r.formula for r in reqs} == {"yoy_change_pct", "yoy_change_abs"}
    assert all(r.metric == "goodwill" for r in reqs)
    reqs = select_formulas("YoY change in non-marketable equity securities?", line_items)
    assert reqs and reqs[0].metric == "non marketable equity securities"
    assert select_formulas("What were inventories at fiscal year-end 2025?", line_items) == []
    assert (
        select_formulas(
            "Quick ratio (cash + marketable securities + receivables) / current liabilities?",
            line_items,
        )[0].formula
        == "quick_ratio"
    )


def test_select_formulas_picks_fiscal_year() -> None:
    reqs = select_formulas("What was the current ratio as of January 26, 2025?", ["total assets"])
    assert reqs[0].fiscal_year == 2025
    reqs = select_formulas("current ratio for fiscal 2025", ["total assets"])
    assert reqs[0].fiscal_year == 2025
    reqs = select_formulas("current ratio FY26", ["total assets"])
    assert reqs[0].fiscal_year == 2026
    # a range names the comparison year first; the latest year is the current period
    reqs = select_formulas(
        "How did cash and cash equivalents change from fiscal 2025 to fiscal 2026?",
        ["cash and cash equivalents"],
    )
    assert reqs and reqs[0].fiscal_year == 2026 and reqs[0].metric == "cash and cash equivalents"


def test_result_block_is_verbatim_usable(calc: Calculator) -> None:
    result = calc.compute("current_ratio", fiscal_year=2026)
    block = result.as_context_block("K1")
    assert "3.91" in block and "125,605" in block and "32,163" in block
    assert "rowfact_p141_total_current_assets" in block
