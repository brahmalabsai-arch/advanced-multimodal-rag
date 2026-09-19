"""Generate `eval/golden.jsonl` (plan Phase 3) and check that every labelled chunk id exists.

Numeric expectations come from the Phase 1-validated statements (USD millions). Explanatory,
visual and cross-section reference answers are drafted from the retrieved passages and flagged
`needs_human_review` (architecture §18.2-1).

    .venv/Scripts/python eval/build_golden.py
"""

from __future__ import annotations

import json
from collections import Counter

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

OUT = PROJECT_ROOT / "eval" / "golden.jsonl"
BS, IS, CF = "rowfact_p141_", "rowfact_p139_", "rowfact_p143_"
M = "USD_millions"
PERIOD_END = {2026: "2026-01-25", 2025: "2025-01-26", 2024: "2024-01-28"}


def num(id_, intent, q, value, unit, fy, chunks, notes=None, calc=None, alternatives=None):
    d = {
        "id": id_,
        "intent": intent,
        "question": q,
        "expected": {"value": value, "unit": unit, "fiscal_year": fy, "period_end": PERIOD_END[fy]},
        "supporting_chunk_ids": chunks,
        "review_status": "verified",
    }
    if alternatives:  # other correct readings of the question, e.g. dollar change vs percent
        d["expected_alternatives"] = [
            {"value": v, "unit": u, "fiscal_year": fy, "period_end": PERIOD_END[fy]}
            for v, u in alternatives
        ]
    if calc:
        d["expected_calculation"] = calc
    if notes:
        d["notes"] = notes
    return d


def text(id_, intent, q, keywords, reference, chunks, notes=None, behavior=None):
    d = {
        "id": id_,
        "intent": intent,
        "question": q,
        "expected": None,
        "expected_keywords": keywords,
        "reference_answer": reference,
        "supporting_chunk_ids": chunks,
        "review_status": "needs_human_review",
    }
    if notes:
        d["notes"] = notes
    if behavior:
        d["expected_behavior"] = behavior
    return d


ROWS = [
    # --- POINT_LOOKUP (balance sheet)
    num(
        "G1",
        "POINT_LOOKUP",
        "What were NVIDIA's total assets as of Jan 25, 2026?",
        206803,
        M,
        2026,
        [BS + "total_assets"],
    ),
    num(
        "G2",
        "POINT_LOOKUP",
        "What were NVIDIA's inventories at fiscal year-end 2025 (Jan 26, 2025)?",
        10080,
        M,
        2025,
        [BS + "inventories"],
    ),
    num(
        "P3",
        "POINT_LOOKUP",
        "How much cash and cash equivalents did NVIDIA hold as of January 25, 2026?",
        10605,
        M,
        2026,
        [BS + "cash_and_cash_equivalents"],
    ),
    num(
        "P4",
        "POINT_LOOKUP",
        "What was the value of marketable securities on the balance sheet at the end of fiscal 2026?",
        51951,
        M,
        2026,
        [BS + "marketable_securities"],
    ),
    num(
        "P5",
        "POINT_LOOKUP",
        "What were accounts receivable, net, as of Jan 25, 2026?",
        38466,
        M,
        2026,
        [BS + "accounts_receivable_net"],
    ),
    num(
        "P6",
        "POINT_LOOKUP",
        "What were total current liabilities at the end of fiscal 2026?",
        32163,
        M,
        2026,
        [BS + "total_current_liabilities"],
    ),
    num(
        "P7",
        "POINT_LOOKUP",
        "What were NVIDIA's total liabilities as of January 25, 2026?",
        49510,
        M,
        2026,
        [BS + "total_liabilities"],
    ),
    num(
        "P8",
        "POINT_LOOKUP",
        "What was total shareholders' equity as of Jan 25, 2026?",
        157293,
        M,
        2026,
        [BS + "total_shareholders_equity"],
    ),
    num(
        "P9",
        "POINT_LOOKUP",
        "What was goodwill on the balance sheet as of January 26, 2025?",
        5188,
        M,
        2025,
        [BS + "goodwill"],
    ),
    num(
        "P10",
        "POINT_LOOKUP",
        "How much long-term debt did NVIDIA report as of Jan 25, 2026?",
        7469,
        M,
        2026,
        [BS + "long_term_debt"],
    ),
    num(
        "P11",
        "POINT_LOOKUP",
        "What were retained earnings at the end of fiscal 2026?",
        146973,
        M,
        2026,
        [BS + "retained_earnings"],
    ),
    num(
        "P12",
        "POINT_LOOKUP",
        "What was NVIDIA's short-term debt as of January 25, 2026?",
        999,
        M,
        2026,
        [BS + "short_term_debt"],
    ),
    num(
        "P13",
        "POINT_LOOKUP",
        "What were total assets as of January 26, 2025?",
        111601,
        M,
        2025,
        [BS + "total_assets"],
        notes="Cache-adversarial twin of G1 (G15): must not be served from G1's cache entry.",
    ),
    # --- POINT_LOOKUP (income statement / cash flow)
    num(
        "P14",
        "POINT_LOOKUP",
        "What was NVIDIA's revenue for fiscal year 2026?",
        215938,
        M,
        2026,
        [IS + "revenue"],
    ),
    num(
        "P15",
        "POINT_LOOKUP",
        "What was net income for the fiscal year ended January 25, 2026?",
        120067,
        M,
        2026,
        [IS + "net_income"],
    ),
    num(
        "P16",
        "POINT_LOOKUP",
        "How much did NVIDIA spend on research and development in fiscal 2026?",
        18497,
        M,
        2026,
        [IS + "research_and_development"],
    ),
    num(
        "P17",
        "POINT_LOOKUP",
        "What was diluted net income per share for fiscal 2026?",
        4.90,
        "USD_per_share",
        2026,
        [IS + "diluted"],
    ),
    num(
        "P18",
        "POINT_LOOKUP",
        "What was net cash provided by operating activities in fiscal year 2026?",
        102718,
        M,
        2026,
        [CF + "net_cash_provided_by_operating_activities"],
    ),
    num(
        "P19",
        "POINT_LOOKUP",
        "How much did NVIDIA pay to repurchase common stock in fiscal 2026?",
        40086,
        M,
        2026,
        [CF + "payments_related_to_repurchases_of_common_stock"],
        notes="Stated as (40,086) in the cash flow statement; the answer may present it as a positive payment.",
    ),
    num(
        "P20",
        "POINT_LOOKUP",
        "How much did NVIDIA pay in dividends during fiscal year 2026?",
        974,
        M,
        2026,
        [CF + "dividends_paid", "text_p131_4"],
        notes="Cash flow shows (974); MD&A states $974 million.",
    ),
    # --- COMPUTATION
    num(
        "G3",
        "COMPUTATION",
        "What is the current ratio as of Jan 25, 2026?",
        3.91,
        "ratio",
        2026,
        [BS + "total_current_assets", BS + "total_current_liabilities"],
        calc="current_ratio",
    ),
    num(
        "G4",
        "COMPUTATION",
        "What was NVIDIA's working capital as of Jan 25, 2026?",
        93442,
        M,
        2026,
        [BS + "total_current_assets", BS + "total_current_liabilities"],
        calc="working_capital",
    ),
    num(
        "G5",
        "COMPUTATION",
        "What is the quick ratio (cash + marketable securities + receivables) / current liabilities as of Jan 25, 2026?",
        3.14,
        "ratio",
        2026,
        [
            BS + "cash_and_cash_equivalents",
            BS + "marketable_securities",
            BS + "accounts_receivable_net",
            BS + "total_current_liabilities",
        ],
        calc="quick_ratio",
    ),
    num(
        "G6",
        "COMPUTATION",
        "What is NVIDIA's total debt to equity ratio as of Jan 25, 2026?",
        0.054,
        "ratio",
        2026,
        [BS + "short_term_debt", BS + "long_term_debt", BS + "total_shareholders_equity"],
        calc="debt_to_equity",
    ),
    num(
        "C5",
        "COMPUTATION",
        "What is the liabilities-to-equity ratio as of January 25, 2026?",
        0.315,
        "ratio",
        2026,
        [BS + "total_liabilities", BS + "total_shareholders_equity"],
        calc="liabilities_to_equity",
    ),
    num(
        "C6",
        "COMPUTATION",
        "What is NVIDIA's equity ratio (equity to assets) at fiscal year-end 2026?",
        0.761,
        "ratio",
        2026,
        [BS + "total_shareholders_equity", BS + "total_assets"],
        calc="equity_ratio",
    ),
    num(
        "C7",
        "COMPUTATION",
        "What was NVIDIA's total cash and marketable securities as of Jan 25, 2026?",
        62556,
        M,
        2026,
        [BS + "cash_and_cash_equivalents", BS + "marketable_securities"],
        calc="cash_and_investments",
    ),
    num(
        "C8",
        "COMPUTATION",
        "What was the current ratio as of January 26, 2025?",
        4.44,
        "ratio",
        2025,
        [BS + "total_current_assets", BS + "total_current_liabilities"],
        calc="current_ratio",
        notes="80,126 / 18,047 = 4.4398",
    ),
    # --- COMPARISON_TREND
    num(
        "G7",
        "COMPARISON_TREND",
        "What was the year-over-year change in total assets in fiscal 2026?",
        85.3,
        "percent",
        2026,
        [BS + "total_assets"],
        calc="yoy_change_pct",
    ),
    num(
        "G8",
        "COMPARISON_TREND",
        "How much did inventories grow year-over-year in fiscal 2026?",
        112.3,
        "percent",
        2026,
        [BS + "inventories"],
        calc="yoy_change_pct",
    ),
    num(
        "G9",
        "COMPARISON_TREND",
        "How much did goodwill grow year-over-year?",
        301.5,
        "percent",
        2026,
        [BS + "goodwill"],
        calc="yoy_change_pct",
        notes="$5,188M to $20,832M",
    ),
    num(
        "G10",
        "COMPARISON_TREND",
        "What was the year-over-year change in non-marketable equity securities?",
        557.0,
        "percent",
        2026,
        [BS + "non_marketable_equity_securities"],
        calc="yoy_change_pct",
        notes="$3,387M to $22,251M",
    ),
    num(
        "T5",
        "COMPARISON_TREND",
        "By what percentage did total liabilities increase year-over-year in fiscal 2026?",
        53.4,
        "percent",
        2026,
        [BS + "total_liabilities"],
        calc="yoy_change_pct",
        notes="32,274 to 49,510",
    ),
    num(
        "T6",
        "COMPARISON_TREND",
        "How much did NVIDIA's long-term debt change year-over-year in fiscal 2026?",
        -11.7,
        "percent",
        2026,
        [BS + "long_term_debt"],
        calc="yoy_change_pct",
        notes="8,463 to 7,469 (decrease); a $994 million decrease is also correct",
        alternatives=[(-994, M)],
    ),
    num(
        "T7",
        "COMPARISON_TREND",
        "How did cash and cash equivalents change from fiscal 2025 to fiscal 2026?",
        23.5,
        "percent",
        2026,
        [BS + "cash_and_cash_equivalents"],
        calc="yoy_change_pct",
        notes="8,589 to 10,605; a $2,016 million increase is also correct",
        alternatives=[(2016, M)],
    ),
    # --- EXPLANATORY
    text(
        "G11",
        "EXPLANATORY",
        "What drove the first-quarter fiscal 2026 charge related to H20?",
        ["H20", "license", "4.5 billion", "export"],
        "In April 2025 the U.S. government informed NVIDIA that a license is required to export H20 products to China; as a result NVIDIA incurred a $4.5 billion charge in the first quarter of fiscal 2026 for excess inventory and purchase obligations tied to H20. The disclosure is repeated in Item 1 (Government Regulations), Item 1A (Risk Factors) and Item 7 (MD&A).",
        ["text_p98_0", "text_p124_2", "text_p114_2"],
        notes="Appears in several sections; the compressor must dedupe.",
    ),
    text(
        "E2",
        "EXPLANATORY",
        "Why did NVIDIA's gross margin decrease in fiscal 2026?",
        ["71.1%", "75.0%", "Blackwell", "Hopper"],
        "Gross margin decreased to 71.1% in fiscal 2026 from 75.0% in fiscal 2025 as the business model transitioned from offering Hopper HGX systems to Blackwell full-scale systems, together with the H20-related charges.",
        ["text_p129_2", "text_p125_1"],
    ),
    text(
        "E3",
        "EXPLANATORY",
        "What drove NVIDIA's revenue growth in fiscal year 2026?",
        ["data center", "65%", "Blackwell"],
        "Revenue was $215.9 billion, up 65%; growth was driven by data center compute and networking platforms for accelerated computing and AI, with Blackwell architectures representing the majority of Data Center revenue (up 68%). Gaming, Professional Visualization and Automotive also grew.",
        ["text_p124_1", "text_p125_1"],
    ),
    text(
        "E4",
        "EXPLANATORY",
        "How does NVIDIA account for inventory provisions and what causes excess inventory?",
        ["lower of cost", "net realizable value", "excess", "obsolete"],
        "NVIDIA charges cost of sales for provisions that write inventory down to the lower of cost or net realizable value and for excess/obsolete inventory and excess purchase commitments; provisions depend on management judgment about future demand and can be driven by demand decreases, order cancellations and technology obsolescence.",
        ["text_p126_1"],
    ),
    text(
        "E5",
        "EXPLANATORY",
        "What share of fiscal 2026 revenue came from customers headquartered outside the United States?",
        ["31%"],
        "Revenue from customers headquartered outside the United States accounted for 31% of total revenue in fiscal 2026 (MD&A, Concentration of Revenue).",
        ["text_p129_1"],
        notes="Numeric check: 31%.",
    ),
    # --- VISUAL
    text(
        "G12",
        "VISUAL",
        "How did NVIDIA's 5-year cumulative total return compare to the S&P 500 and the Nasdaq 100?",
        ["1,448", "S&P 500", "Nasdaq 100"],
        "Per the stock performance graph and its data table (PDF p. 123), $100 invested on 1/31/2021 grew to about $1,448.75 in NVIDIA stock by 1/25/2026, versus roughly $200-$220 for the S&P 500 and the Nasdaq 100; NVIDIA outperformed both indices by a wide margin.",
        ["fig_p123_0", "tbl_p123_0"],
        notes="Companion table values take precedence over chart-read values.",
    ),
    text(
        "G13",
        "VISUAL",
        'What are the layers in NVIDIA\'s "five-layer cake" framing of the AI industry?',
        ["Energy", "Chips", "Infrastructure", "Models", "Applications"],
        "The five layers, bottom to top, are Energy, Chips, Infrastructure, Models and Applications (Annual Review diagram, PDF p. 3).",
        ["fig_p3_0"],
        notes="Reference verified against the figure.",
    ),
    text(
        "V3",
        "VISUAL",
        "What does the fiscal 2026 target pay mix chart show about how NEO compensation is weighted?",
        ["equity", "at-risk"],
        "The target pay mix chart (Proxy, PDF p. 57) shows executive pay heavily weighted towards equity awards, with the large majority of target compensation at-risk/performance-based and only a small cash portion.",
        ["fig_p57_0", "text_p57_0"],
    ),
    # --- CROSS_SECTION
    text(
        "X1",
        "CROSS_SECTION",
        "How does executive pay-versus-performance relate to revenue growth?",
        ["compensation actually paid", "revenue"],
        "The Pay Versus Performance section (Proxy, PDF pp. 74-76) shows compensation actually paid moving with NVIDIA's total shareholder return and revenue growth; revenue (up 65% to $215.9 billion in fiscal 2026 per the 10-K) is one of the most important financial performance measures.",
        ["tbl_p74_0_part1", "text_p74_1", "fig_p76_1", "rowfact_p139_revenue"],
    ),
    text(
        "X2",
        "CROSS_SECTION",
        "How did NVIDIA's share repurchases compare with dividends paid in fiscal 2026?",
        ["40,086", "974"],
        "NVIDIA repurchased $40,086 million of common stock and paid $974 million of dividends in fiscal 2026 (cash flow statement, PDF p. 143; MD&A capital return section); repurchases were about 41 times the dividends.",
        [
            "rowfact_p143_payments_related_to_repurchases_of_common_stock",
            "rowfact_p143_dividends_paid",
            "text_p131_4",
        ],
    ),
    text(
        "X3",
        "CROSS_SECTION",
        "When is NVIDIA's 2026 annual meeting of stockholders and what is the record date?",
        ["June 24, 2026", "April 27, 2026"],
        "The 2026 annual meeting is Wednesday, June 24, 2026 at 9:00 a.m. Pacific Time, held virtually; the record date is April 27, 2026 (Notice of Annual Meeting, PDF p. 20).",
        ["text_p20_0"],
        notes="time_anchored answer class expected.",
    ),
    # --- OUT_OF_SCOPE
    text(
        "G14",
        "OUT_OF_SCOPE",
        "Should I buy NVIDIA stock?",
        [],
        "Scoped refusal: the assistant analyses reported figures and does not give investment advice; offers what the report can answer.",
        [],
        behavior="refuse",
        notes="Phase 4 scope gate; no retrieval spend.",
    ),
    text(
        "O2",
        "OUT_OF_SCOPE",
        "What is NVIDIA's share price right now?",
        [],
        "Scoped refusal: live market data is not in the annual report.",
        [],
        behavior="refuse",
        notes="Phase 4 scope gate.",
    ),
]


def main() -> int:
    utf8_console()
    with OUT.open("w", encoding="utf-8") as fh:
        for r in ROWS:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(ROWS)} questions -> {OUT}")
    print(dict(Counter(r["intent"] for r in ROWS)))
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    chunks_file = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
    if chunks_file.exists():
        known = {
            json.loads(ln)["id"]
            for ln in chunks_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        missing = [(r["id"], c) for r in ROWS for c in r["supporting_chunk_ids"] if c not in known]
        print("missing chunk ids:", missing or "none")
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
