"""Calibrate the L2 similarity thresholds and the embedded-text mode (architecture §5.8, §5.10;
plan Phase 6). No model calls — slot extractor + local embedder only.

    .venv/Scripts/python eval/cache_threshold_calibration.py [--report docs/reports/cache_threshold_calibration.md]

For every pair the script asks two questions the cache asks at lookup time: do the slot-guard
keys of A and B agree (hard filter), and what is the cosine similarity of the embedded texts
(soft score)? A pair "hits" when the guard passes and similarity ≥ the class threshold of B.
Adversarial pairs must miss (false-hit rate ≤ 1 %, NFR-3); paraphrase pairs should hit.
Three embedded-text modes are compared (`raw` question, `normalized`, `canonical` slot text)
over a threshold grid, and the report records which combination the config adopts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag.cache.records import embed_text_for, slot_keys
from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "eval"
MODES = ("raw", "normalized", "canonical")
GRID = [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]


def load_pairs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def score_pairs(
    pairs: list[dict[str, Any]], extractor, embedder, mode: str
) -> list[dict[str, Any]]:  # noqa: ANN001
    """Attach guard result, slot-richness of B and cosine similarity for one embed mode."""
    slots = {}
    texts: list[str] = []
    for p in pairs:
        for q in (p["a"], p["b"]):
            if q not in slots:
                slots[q] = extractor.extract(q)
                texts.append(embed_text_for(q, slots[q], mode))
    uniq = list(dict.fromkeys(texts))
    vecs = embedder.embed_queries(uniq)
    vec_of = {t: vecs[i] for i, t in enumerate(uniq)}
    rows = []
    for p in pairs:
        sa, sb = slots[p["a"]], slots[p["b"]]
        ka, kb = slot_keys(sa, default_entity="NVIDIA"), slot_keys(sb, default_entity="NVIDIA")
        va, vb = vec_of[embed_text_for(p["a"], sa, mode)], vec_of[embed_text_for(p["b"], sb, mode)]
        sim = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
        diff = [f for f in ka if ka[f] != kb[f]]
        rows.append(
            {
                **p,
                "guard_pass": not diff,
                "guard_diff": diff,
                "slot_rich": sb.slot_rich,
                "similarity": round(sim, 4),
            }
        )
    return rows


def evaluate_pairs(
    pairs, extractor, embedder, *, mode: str, rich: float, poor: float
) -> dict[str, Any]:  # noqa: ANN001
    rows = score_pairs(pairs, extractor, embedder, mode)
    return summarize(rows, rich=rich, poor=poor)


def summarize(rows: list[dict[str, Any]], *, rich: float, poor: float) -> dict[str, Any]:
    hits = []
    by_kind: dict[str, dict[str, int]] = {}
    guard_blocked = 0
    for r in rows:
        thr = rich if r["slot_rich"] else poor
        hit = r["guard_pass"] and r["similarity"] >= thr
        k = by_kind.setdefault(r["kind"], {"n": 0, "hits": 0, "guard_blocked": 0})
        k["n"] += 1
        if not r["guard_pass"]:
            k["guard_blocked"] += 1
            guard_blocked += 1
        if hit:
            k["hits"] += 1
            hits.append({"id": r["id"], "a": r["a"], "b": r["b"], "similarity": r["similarity"]})
    n = len(rows)
    return {
        "n": n,
        "hits_n": len(hits),
        "hit_rate": round(len(hits) / n, 4) if n else 0.0,
        "guard_blocked": guard_blocked,
        "by_kind": by_kind,
        "hits": hits,
    }


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "docs" / "reports" / "cache_threshold_calibration.md"),
    )
    ap.add_argument("--embedder", default="bge-small")
    args = ap.parse_args(argv)

    from rag.core.config import load_thresholds_config
    from rag.core.embeddings import get_embedder
    from rag.query.slots import get_slot_extractor

    extractor = get_slot_extractor()
    embedder = get_embedder(args.embedder)
    adv = load_pairs(EVAL_DIR / "adversarial_cache_pairs.jsonl")
    par = load_pairs(EVAL_DIR / "paraphrase_pairs.jsonl")
    cfg = load_thresholds_config().cache.l2

    scored = {
        m: (score_pairs(adv, extractor, embedder, m), score_pairs(par, extractor, embedder, m))
        for m in MODES
    }

    # 1. guard-only view (thresholds at 0 → what the hard filter alone does)
    guard_adv = summarize(scored["normalized"][0], rich=0.0, poor=0.0)
    guard_par = summarize(scored["normalized"][1], rich=0.0, poor=0.0)

    # 2. grid: same threshold for both classes, then the (rich, poor) split the config uses
    grid_rows = []
    for m in MODES:
        a_rows, p_rows = scored[m]
        for t in GRID:
            sa, sp = summarize(a_rows, rich=t, poor=t), summarize(p_rows, rich=t, poor=t)
            grid_rows.append(
                {
                    "mode": m,
                    "thr": t,
                    "adv_hit_rate": sa["hit_rate"],
                    "adv_hits": sa["hits_n"],
                    "par_hit_rate": sp["hit_rate"],
                    "par_hits": sp["hits_n"],
                }
            )
    config_rows = []
    for m in MODES:
        a_rows, p_rows = scored[m]
        sa = summarize(a_rows, rich=cfg.similarity.slot_rich, poor=cfg.similarity.slot_poor)
        sp = summarize(p_rows, rich=cfg.similarity.slot_rich, poor=cfg.similarity.slot_poor)
        config_rows.append({"mode": m, "adv": sa, "par": sp})

    # 3. similarity distributions per kind (guard-passing adversarial pairs are the risk)
    dist_rows = []
    for m in MODES:
        a_rows, p_rows = scored[m]
        for kind in sorted({r["kind"] for r in a_rows}):
            sims = [r["similarity"] for r in a_rows if r["kind"] == kind]
            passing = [r["similarity"] for r in a_rows if r["kind"] == kind and r["guard_pass"]]
            dist_rows.append(
                {
                    "mode": m,
                    "kind": kind,
                    "n": len(sims),
                    "p50": float(np.median(sims)),
                    "max": max(sims),
                    "guard_pass_n": len(passing),
                    "guard_pass_max": max(passing) if passing else None,
                }
            )
        sims = [r["similarity"] for r in p_rows]
        rich = [r["similarity"] for r in p_rows if r["slot_rich"]]
        poor = [r["similarity"] for r in p_rows if not r["slot_rich"]]
        dist_rows.append(
            {
                "mode": m,
                "kind": "paraphrase (all)",
                "n": len(sims),
                "p50": float(np.median(sims)),
                "min": min(sims),
                "p10": float(np.percentile(sims, 10)),
            }
        )
        dist_rows.append(
            {
                "mode": m,
                "kind": "paraphrase slot-rich",
                "n": len(rich),
                "p50": float(np.median(rich)) if rich else None,
                "min": min(rich) if rich else None,
                "p10": float(np.percentile(rich, 10)) if rich else None,
            }
        )
        dist_rows.append(
            {
                "mode": m,
                "kind": "paraphrase slot-poor",
                "n": len(poor),
                "p50": float(np.median(poor)) if poor else None,
                "min": min(poor) if poor else None,
                "p10": float(np.percentile(poor, 10)) if poor else None,
            }
        )

    report = render_report(
        adv_n=len(adv),
        par_n=len(par),
        guard_adv=guard_adv,
        guard_par=guard_par,
        grid_rows=grid_rows,
        config_rows=config_rows,
        dist_rows=dist_rows,
        cfg=cfg,
        embedder=args.embedder,
        par_scored=scored[cfg.embed_text][1],
    )
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    (EVAL_DIR / "results" / "cache_threshold_calibration.json").write_text(
        json.dumps({"grid": grid_rows, "config": config_rows, "dist": dist_rows}, indent=1),
        encoding="utf-8",
    )
    print(report)
    return 0


DECISION = """
Written after the runs of 2026-09-18 (kept in the script so re-runs keep the reasoning next to the numbers).

1. **The hard slot guard carries all of the measured precision.** The four swap categories the plan
   asked for (period, metric, direction, negation — 236 pairs) embed at cosine 0.90–0.99, *above* every
   threshold in the grid, so similarity could never have rejected them; the guard blocks 236/236. Two
   keys were added to the §5.8 design during calibration: `negation` ("did inventories grow" vs "did
   inventories not grow", 0.945) and `ask` (what the question asks about its slots — value, reason,
   method, location, risk, assumption, person, advice — from the new `lexicons.ask_type` in
   `glossary.yaml`), plus `aggregation_key` (pct / change / compare / ratio / yoy). Without `ask`, the 50
   *same-slot, different-question* pairs ("what were total assets" vs "how does NVIDIA account for total
   assets") sat at 0.93–0.97 — **closer than genuine paraphrases (0.80–0.90)** — and 34 of them still hit
   at 0.95. With the extended guard: 286/286 adversarial pairs miss, false-hit rate 0.0 % (NFR-3 ≤ 1 %).
2. **Embedded text: `canonical`.** Replacing matched spans by canonical ids ("total_assets", "fy2026")
   raises paraphrase similarity (p50 0.869 vs 0.829 normalised, 0.817 raw) without moving the
   adversarial distribution the guard already handles.
3. **Thresholds: slot-rich 0.85, slot-poor 0.90** (design: 0.90 / 0.95). Because bge-small scores
   unrelated-but-same-slot questions *higher* than paraphrases, raising the threshold buys no
   precision on this corpus; it only costs recall (0.90 → 36.5 % paraphrase hits, 0.85 → 65.4 %).
   Slot-poor questions keep a stricter floor because fewer hard keys constrain them. The remaining
   role of the threshold is to reject wording the lexicons do not model; it is a floor, not the defence.
4. **Accepted misses.** Two paraphrase pairs are split by the guard by design: "2026 annual meeting"
   resolves the bare year to `AMBIGUOUS_2026` (a period) while "annual meeting" has none. Paraphrases
   under the threshold (listed above) miss safely and cost one pipeline run each.
5. **Caveat.** The adversarial set is template-generated, so every swap is slot-detectable by
   construction; the same-slot category was added precisely because it is not. A larger hand-written
   set of guard-passing pairs is the right next step (Phase 7 evidence package) and is listed as a
   human-review item.
"""


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def render_report(
    *,
    adv_n,
    par_n,
    guard_adv,
    guard_par,
    grid_rows,
    config_rows,
    dist_rows,
    cfg,
    embedder,
    par_scored,
) -> str:  # noqa: ANN001
    lines = [
        "# Cache threshold calibration (Phase 6)",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · embedder `{embedder}` · "
        f"{adv_n} adversarial pairs (`eval/adversarial_cache_pairs.jsonl`) · {par_n} paraphrase pairs "
        "(`eval/paraphrase_pairs.jsonl`) · no model calls (`eval/cache_threshold_calibration.py`).",
        "",
        "A pair *hits* when the slot-guard keys (entity, periods, metrics/formulas, direction, negation) agree "
        "**and** cosine similarity ≥ the class threshold (slot-rich = metric/formula + period present). "
        "Adversarial pairs must miss (NFR-3: false-hit rate ≤ 1 %); paraphrase pairs should hit.",
        "",
        "## 1. What the hard slot guard does on its own (threshold 0)",
        "",
        "| set | pairs | blocked by the guard | pass the guard |",
        "|---|---|---|---|",
        f"| adversarial | {guard_adv['n']} | {guard_adv['guard_blocked']} ({_pct(guard_adv['guard_blocked'] / guard_adv['n'])}) | {guard_adv['n'] - guard_adv['guard_blocked']} |",
        f"| paraphrase | {guard_par['n']} | {guard_par['guard_blocked']} ({_pct(guard_par['guard_blocked'] / guard_par['n'])}) | {guard_par['n'] - guard_par['guard_blocked']} |",
        "",
        "Per adversarial kind (guard only):",
        "",
        "| kind | pairs | guard blocked | would hit without a threshold |",
        "|---|---|---|---|",
    ]
    for k, v in guard_adv["by_kind"].items():
        lines.append(f"| {k} | {v['n']} | {v['guard_blocked']} | {v['hits']} |")
    if guard_par["guard_blocked"]:
        lines += [
            "",
            "Paraphrase pairs the guard splits (slot extraction disagrees — these can never hit):",
            "",
        ]
        blocked = [r for r in par_scored if not r["guard_pass"]]
        for r in blocked:
            lines.append(f"- `{r['id']}` {r['guard_diff']}: “{r['a']}” vs “{r['b']}”")
    lines += [
        "",
        "## 2. Similarity distributions",
        "",
        "| mode | kind | n | p50 | max / min | guard-passing n | guard-passing max | p10 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def f3(x: Any) -> str:
        return "–" if x is None else f"{x:.3f}"

    for d in dist_rows:
        lines.append(
            f"| {d['mode']} | {d['kind']} | {d['n']} | {f3(d.get('p50'))} | "
            f"{f3(d.get('max', d.get('min')))} | {d.get('guard_pass_n', '–')} | "
            f"{f3(d.get('guard_pass_max'))} | {f3(d.get('p10'))} |"
        )
    lines += [
        "",
        "## 3. Threshold grid (one threshold for both classes)",
        "",
        "| mode | threshold | adversarial false hits | false-hit rate | paraphrase hits | paraphrase hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for g in grid_rows:
        lines.append(
            f"| {g['mode']} | {g['thr']:.2f} | {g['adv_hits']} | {_pct(g['adv_hit_rate'])} | {g['par_hits']} | {_pct(g['par_hit_rate'])} |"
        )
    lines += [
        "",
        f"## 4. Configured thresholds (slot-rich {cfg.similarity.slot_rich} / slot-poor {cfg.similarity.slot_poor}, `embed_text: {cfg.embed_text}`)",
        "",
        "| mode | adversarial false hits | false-hit rate | paraphrase hits | paraphrase hit rate |",
        "|---|---|---|---|---|",
    ]
    for c in config_rows:
        mark = " **(config)**" if c["mode"] == cfg.embed_text else ""
        lines.append(
            f"| {c['mode']}{mark} | {c['adv']['hits_n']} | {_pct(c['adv']['hit_rate'])} | {c['par']['hits_n']} | {_pct(c['par']['hit_rate'])} |"
        )
    chosen = next(c for c in config_rows if c["mode"] == cfg.embed_text)
    if chosen["adv"]["hits"]:
        lines += ["", "Adversarial pairs that still hit under the configured thresholds:", ""]
        for h in chosen["adv"]["hits"]:
            lines.append(f"- `{h['id']}` sim {h['similarity']}: “{h['a']}” vs “{h['b']}”")
    misses = [
        r
        for r in par_scored
        if r["guard_pass"]
        and r["similarity"]
        < (cfg.similarity.slot_rich if r["slot_rich"] else cfg.similarity.slot_poor)
    ]
    if misses:
        lines += [
            "",
            "Paraphrase pairs that pass the guard but fall under the threshold (accepted misses):",
            "",
        ]
        for r in sorted(misses, key=lambda r: r["similarity"]):
            lines.append(
                f"- `{r['id']}` sim {r['similarity']} ({'rich' if r['slot_rich'] else 'poor'}): “{r['a']}” vs “{r['b']}”"
            )
    lines += ["", "## 5. Decision (D-59)", "", DECISION.strip(), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
