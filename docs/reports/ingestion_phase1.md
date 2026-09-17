# Ingestion report — Phase 1 (parse and validate)

Generated 2026-09-17T14:38:10+00:00 · `2026_NVIDIA_ANNUAL_REPORT.pdf` · sha256 `0e725ba04822…` · Docling 2.128.0 in 280.4 s · **overall: PASS**

## Parse summary

| Metric | Value |
|---|---|
| Pages | 175 |
| Docling texts / tables / pictures | 2422 / 105 / 96 |
| Elements written | 2343 |
| Statement pages | p139 income_statement, p141 balance_sheet, p143 cash_flow |

## Elements by type and section

| Type | annual_review | form_10k | proxy | Total |
|---|---|---|---|---|
| caption | 1 | 1 | 6 | 8 |
| checkbox | 0 | 10 | 40 | 50 |
| document_index | 0 | 2 | 1 | 3 |
| footnote | 0 | 4 | 23 | 27 |
| heading | 25 | 236 | 191 | 452 |
| list_item | 0 | 106 | 173 | 279 |
| picture | 12 | 2 | 82 | 96 |
| table | 0 | 56 | 46 | 102 |
| text | 136 | 620 | 570 | 1326 |

## Section map (subsection transitions)

p1 → `narrative_spread`; p19 → `proxy_front_matter`; p20 → `notice_of_meeting`; p25 → `proxy_summary`; p26 → `proposals`; p27 → `corporate_governance`; p28 → `compensation`; p46 → `corporate_governance`; p54 → `ownership`; p56 → `proposals`; p57 → `compensation`; p67 → `ownership`; p69 → `compensation`; p73 → `pay_vs_performance`; p77 → `proposals`; p79 → `audit_matters`; p80 → `proposals`; p88 → `additional_information`; p89 → `form_10k_front_matter`; p92 → `item_1_business`; p100 → `item_1a_risk_factors`; p120 → `item_1b_unresolved_staff_comments`; p121 → `item_2_properties`; p122 → `item_5_market`; p124 → `item_7_mdna`; p132 → `item_7a_market_risk`; p133 → `item_8_financials`; p134 → `item_9b_other_information`; p135 → `item_11_executive_compensation`; p136 → `item_15_exhibits`; p137 → `auditor_report`; p139 → `item_8_financials`; p144 → `notes`; p170 → `exhibit_index`; p171 → `item_16_summary`; p172 → `signatures`; p175 → `back_cover`

## Statement validation (pdfplumber vs Docling)

### Consolidated Statements of Income (p. 139, USD_millions, periods 2026-01-25, 2025-01-26, 2024-01-28)

Rows: 18 · Docling rows matched: 18/18 · disagreements: **0**

### Consolidated Balance Sheets (p. 141, USD_millions, periods 2026-01-25, 2025-01-26)

Rows: 30 · Docling rows matched: 30/30 · disagreements: **0**
· accounting identity: **holds** (206,803 / 111,601 = 206,803 / 111,601)

| Subtotal | Stated | Computed | OK |
|---|---|---|---|
| Total current assets | 125,605 / 80,126 | 125,605 / 80,126 | ✓ |
| Total current liabilities | 32,163 / 18,047 | 32,163 / 18,047 | ✓ |
| Total shareholders' equity | 157,293 / 79,327 | 157,293 / 79,327 | ✓ |

| Line item | Group | FY2026 | FY2025 | Docling agrees |
|---|---|---|---|---|
| Cash and cash equivalents | current_assets | 10,605 | 8,589 | ✓ |
| Marketable securities | current_assets | 51,951 | 34,621 | ✓ |
| Accounts receivable, net | current_assets | 38,466 | 23,065 | ✓ |
| Inventories | current_assets | 21,403 | 10,080 | ✓ |
| Prepaid expenses and other current assets | current_assets | 3,180 | 3,771 | ✓ |
| Total current assets | current_assets | 125,605 | 80,126 | ✓ |
| Property and equipment, net | non_current_assets | 10,383 | 6,283 | ✓ |
| Operating lease assets | non_current_assets | 2,867 | 1,793 | ✓ |
| Goodwill | non_current_assets | 20,832 | 5,188 | ✓ |
| Intangible assets, net | non_current_assets | 3,306 | 807 | ✓ |
| Deferred income tax assets | non_current_assets | 13,258 | 10,979 | ✓ |
| Non-marketable equity securities | non_current_assets | 22,251 | 3,387 | ✓ |
| Other assets | non_current_assets | 8,301 | 3,038 | ✓ |
| Total assets | assets | 206,803 | 111,601 | ✓ |
| Accounts payable | current_liabilities | 9,812 | 6,310 | ✓ |
| Accrued and other current liabilities | current_liabilities | 21,352 | 11,737 | ✓ |
| Short-term debt | current_liabilities | 999 | — | ✓ |
| Total current liabilities | current_liabilities | 32,163 | 18,047 | ✓ |
| Long-term debt | non_current_liabilities | 7,469 | 8,463 | ✓ |
| Long-term operating lease liabilities | non_current_liabilities | 2,572 | 1,519 | ✓ |
| Other long-term liabilities | non_current_liabilities | 7,306 | 4,245 | ✓ |
| Total liabilities | liabilities | 49,510 | 32,274 | ✓ |
| Commitments and contingencies - see Note 12 | other | — | — | ✓ |
| Preferred stock, $0.001 par value; 2 shares authorized; none | equity | — | — | ✓ |
| Common stock, $0.001 par value; 80,000 shares authorized; 24 | equity | 24 | 24 | ✓ |
| Additional paid-in capital | equity | 10,118 | 11,237 | ✓ |
| Accumulated other comprehensive income | equity | 178 | 28 | ✓ |
| Retained earnings | equity | 146,973 | 68,038 | ✓ |
| Total shareholders' equity | equity | 157,293 | 79,327 | ✓ |
| Total liabilities and shareholders' equity | liabilities_and_equity | 206,803 | 111,601 | ✓ |

### Consolidated Statements of Cash Flows (p. 143, USD_millions, periods 2026-01-25, 2025-01-26, 2024-01-28)

Rows: 35 · Docling rows matched: 35/35 · disagreements: **0**

## Figure candidates

42 candidates on 29 pages · by source: raster_uncovered 12, docling_picture 29, vector_density 1 · required pages present: p3 ✓, p5 ✓, p123 ✓ · page thumbnails rendered: 175

| Figure | Page | Source | Size (px) | Caption |
|---|---|---|---|---|
| `p1_0` | 1 | raster_uncovered | 3300×2150 |  |
| `p2_0` | 2 | raster_uncovered | 3300×2150 |  |
| `p3_0` | 3 | docling_picture | 2885×1046 |  |
| `p3_1` | 3 | raster_uncovered | 3300×2150 |  |
| `p4_0` | 4 | docling_picture | 2742×988 |  |
| `p4_1` | 4 | raster_uncovered | 3300×2150 |  |
| `p5_0` | 5 | raster_uncovered | 3300×2150 |  |
| `p6_0` | 6 | raster_uncovered | 3300×2150 |  |
| `p7_0` | 7 | docling_picture | 1141×780 |  |
| `p7_1` | 7 | docling_picture | 1279×453 |  |
| `p7_2` | 7 | raster_uncovered | 3300×2150 |  |
| `p8_0` | 8 | docling_picture | 2652×848 |  |
| `p8_1` | 8 | raster_uncovered | 3300×2150 |  |
| `p9_0` | 9 | raster_uncovered | 3300×2150 |  |
| `p10_0` | 10 | raster_uncovered | 3300×2150 |  |
| `p11_0` | 11 | docling_picture | 488×222 |  |
| `p11_1` | 11 | docling_picture | 1276×1741 |  |
| `p11_2` | 11 | raster_uncovered | 3300×2150 |  |
| `p12_0` | 12 | docling_picture | 909×786 |  |
| `p14_0` | 14 | docling_picture | 910×799 |  |
| `p17_0` | 17 | docling_picture | 1380×652 |  |
| `p18_0` | 18 | docling_picture | 1380×798 |  |
| `p18_1` | 18 | docling_picture | 418×191 | Jensen Huang Founder and CEO, NVIDIA May 2026 |
| `p20_0` | 20 | docling_picture | 248×185 |  |
| `p23_0` | 23 | docling_picture | 353×197 | Data Center |
| `p23_1` | 23 | docling_picture | 350×198 |  |
| `p23_2` | 23 | docling_picture | 350×198 | Professional Visualization |
| `p23_3` | 23 | docling_picture | 352×198 |  |
| `p24_0` | 24 | docling_picture | 1284×688 |  |
| `p29_0` | 29 | docling_picture | 249×188 |  |
| `p43_0` | 43 | docling_picture | 1058×417 |  |
| `p43_1` | 43 | docling_picture | 1127×523 |  |
| `p45_0` | 45 | vector_density | 1466×1828 |  |
| `p57_0` | 57 | docling_picture | 922×446 |  |
| `p59_0` | 59 | docling_picture | 1425×260 |  |
| `p60_0` | 60 | docling_picture | 1370×520 |  |
| `p76_0` | 76 | docling_picture | 1096×767 |  |
| `p76_1` | 76 | docling_picture | 1122×715 |  |
| `p80_0` | 80 | docling_picture | 564×113 | Proposal 4 - Majority Vote Standard |
| `p89_0` | 89 | docling_picture | 397×172 |  |
| `p123_0` | 123 | docling_picture | 1427×890 | *$100 invested on 1/31/2021 in stock and in indices, including reinves |
| `p175_0` | 175 | raster_uncovered | 3300×2150 |  |

## Reading-order spot check (p. 13)

Text items: 16 · columns detected: 3 · fragment intact: ✓ · column order: ✓ · **PASS**

- Compute is no longer just a cost to support software; it is a productive asset t…
- A computer used to be a tool. In the age of AI, it is manufacturing equipment. A…
- NVIDIA delivers a full-stack AI computing platform that spans the entire AI life…
- NVIDIA is a vertically integrated, horizontally open platform. We architect with…
- Our platform powers AI wherever it runs-across clouds, on-premises data centers,…
- NVIDIA is building the computing infrastructure of the AI era.…
