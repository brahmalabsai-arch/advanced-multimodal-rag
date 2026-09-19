"""Stage C — learned compression classifier (architecture §6.7, plan Phase 7).

    .venv/Scripts/python eval/train_classifier.py --always-run stagec_always --never-run golden_phase4_claude

Offline only: scikit-learn trains, a small JSON of logistic weights ships, and serving scores it
with NumPy (`compress/classifier.py`, `stage_b: learned`), so the serving dependency set stays
lean (P7).

**Labels (§6.7).** The decision log records, per request, every chunk's features and the action
Stage A/B chose; the evaluation results record the answer outcome for the same `request_id`
(`eval-<run>-<id>`). A forced `--compression always` run therefore compresses chunks that the
production gate would have kept, and the paired never-compress run says whether that hurt. A
compressed chunk is labelled

    1 (compress was right)  the query's answer did not get worse than the never arm
                            (numeric exact match preserved, keyword coverage not lower)
                            AND the chunk actually shrank by >= `--min-saving` (default 30 %)
    0 (compress was wrong)  otherwise

This is the "chunk group" labelling the architecture calls for: attribution is per query, so
every compressed chunk in a query shares its outcome. Chunks the run kept are labelled 0 only
when keeping them was under budget pressure; otherwise they carry no signal and are dropped
(`--keep-negatives` includes them).

**Models.** `LogisticRegression` (interpretable, exportable) against `GradientBoostingClassifier`
(reference), both under repeated stratified cross-validation because the label set is small.
The target from §6.7 is precision of "compress" >= 0.85. The rule-based Stage B score is
evaluated on the same labels as the baseline that has to be beaten.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag.compress.classifier import INTENT_WEIGHT, stage_b_score
from rag.compress.features import ChunkFeatures, QueryFeatures
from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT, get_settings

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
WEIGHTS_PATH = PROJECT_ROOT / "src" / "rag" / "compress" / "classifier_weights.json"
COMPRESS_ACTIONS = {"EXTRACT_LIGHT", "EXTRACT_LLM", "ROW_SELECT"}

FEATURE_NAMES = [
    "noise",
    "length",
    "budget_pressure",
    "intent_weight",
    "numeric_density",
    "rank_norm",
    "max_dup_sim",
    "n_sentences_norm",
]


def feature_vector(f: dict[str, Any], q: dict[str, Any]) -> list[float]:
    """The Stage B inputs plus the cheap extras the rules ignore. Same order as FEATURE_NAMES."""
    budget_ratio = float(q.get("budget_ratio", 0.0))
    return [
        1.0 - float(f.get("relevance_density", 0.0)),
        min(float(f.get("chunk_tokens", 0)) / 400.0, 1.0),
        max(0.0, min(1.0, (budget_ratio - 0.5) / 0.5)),
        INTENT_WEIGHT.get(q.get("intent", ""), 0.5),
        float(f.get("numeric_density", 0.0)),
        min(float(f.get("rank", 1)) / 10.0, 1.0),
        max(0.0, float(f.get("max_dup_sim", 0.0))),
        min(float(f.get("n_sentences", 0)) / 20.0, 1.0),
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def outcome_index(run: str) -> dict[str, dict[str, Any]]:
    """golden id → evaluation row for one run."""
    return {r["id"]: r for r in load_jsonl(RESULTS_DIR / f"{run}.jsonl")}


def not_worse(always: dict[str, Any], never: dict[str, Any]) -> tuple[bool, str]:
    """Did compressing this query's context cost answer quality?"""
    if (
        always.get("exact_match") is not None
        and never.get("exact_match") is not None
        and bool(never["exact_match"])
        and not bool(always["exact_match"])
    ):
        return False, "numeric exact match lost"
    a_cov, n_cov = always.get("keyword_coverage"), never.get("keyword_coverage")
    if a_cov is not None and n_cov is not None and a_cov + 1e-9 < n_cov:
        return False, f"keyword coverage {n_cov:.2f} → {a_cov:.2f}"
    if never.get("verify_passed") and not always.get("verify_passed"):
        return False, "verification failed"
    return True, "answer preserved"


def build_dataset(
    *,
    decisions: list[dict[str, Any]],
    always_run: str,
    never_run: str,
    min_saving: float,
    keep_negatives: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    always_rows = outcome_index(always_run)
    never_rows = outcome_index(never_run)
    prefix = f"eval-{always_run}-"
    rows: list[dict[str, Any]] = []
    for d in decisions:
        rid = str(d.get("request_id", ""))
        if not rid.startswith(prefix):
            continue
        qid = rid[len(prefix) :]
        a, n = always_rows.get(qid), never_rows.get(qid)
        if a is None or n is None:
            continue
        preserved, why = not_worse(a, n)
        q = d.get("query", {})
        for c in d.get("chunks", []):
            applied = str(c.get("applied") or c.get("action"))
            f = c.get("features", {})
            before = float(f.get("chunk_tokens", 0) or 0)
            after = float(c.get("tokens_after") or before)
            saving = (before - after) / before if before else 0.0
            compressed = applied in COMPRESS_ACTIONS
            if not compressed and not keep_negatives:
                continue
            label = int(compressed and preserved and saving >= min_saving)
            rows.append(
                {
                    "request_id": rid,
                    "golden_id": qid,
                    "chunk_id": c.get("chunk_id"),
                    "intent": q.get("intent"),
                    "modality": f.get("modality"),
                    "applied": applied,
                    "stage": c.get("stage"),
                    "rule_score": c.get("score"),
                    "tokens_before": before,
                    "tokens_after": after,
                    "saving": round(saving, 3),
                    "answer_preserved": preserved,
                    "why": why,
                    "label": label,
                    "x": feature_vector(f, q),
                }
            )
    X = (
        np.array([r["x"] for r in rows], dtype=np.float64)
        if rows
        else np.zeros((0, len(FEATURE_NAMES)))
    )
    y = np.array([r["label"] for r in rows], dtype=int) if rows else np.zeros(0, dtype=int)
    return X, y, rows


def rule_baseline(rows: list[dict[str, Any]], keep_threshold: float) -> dict[str, float]:
    """How the hand-set Stage B score would have decided the same chunks."""
    preds = []
    for r in rows:
        score = r["rule_score"]
        if score is None:  # Stage A decided; reconstruct the Stage B score from features
            f = ChunkFeatures(
                chunk_id=r["chunk_id"] or "x",
                rank=1,
                modality=r["modality"] or "text",
                chunk_tokens=int(r["tokens_before"]),
                rerank_score=0.0,
                relevance_density=1.0 - r["x"][0],
                max_dup_sim=r["x"][6],
                numeric_density=r["x"][4],
                n_sentences=int(r["x"][7] * 20),
            )
            q = QueryFeatures(
                intent=r["intent"] or "POINT_LOOKUP",
                n_chunks=8,
                total_tokens=0,
                budget_ratio=0.5 + r["x"][2] * 0.5,
                context_budget=2500,
                queries=[],
            )
            score = stage_b_score(f, q)[0]
        preds.append(int(score >= keep_threshold))
    y = np.array([r["label"] for r in rows])
    return scores(y, np.array(preds), np.array([r["rule_score"] or 0.0 for r in rows]))


def scores(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
    }
    if prob is not None and len(set(y.tolist())) > 1:
        with contextlib.suppress(ValueError):
            out["roc_auc"] = round(float(roc_auc_score(y, prob)), 3)
    return out


def bootstrap_precision(
    y: np.ndarray, prob: np.ndarray, *, seed: int, n: int = 2000
) -> tuple[float, float]:
    """90 % percentile interval for precision of the positive class — the honest width of a
    number computed from a few dozen rows."""
    rng = np.random.default_rng(seed)
    pred = (prob >= 0.5).astype(int)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        tp = int(((pred[s] == 1) & (y[s] == 1)).sum())
        fp = int(((pred[s] == 1) & (y[s] == 0)).sum())
        if tp + fp:
            vals.append(tp / (tp + fp))
    if not vals:
        return (0.0, 0.0)
    return (round(float(np.percentile(vals, 5)), 3), round(float(np.percentile(vals, 95)), 3))


def majority_baseline(y: np.ndarray) -> dict[str, float]:
    """ "Always compress" — the predictor Stage C has to beat before it is worth its weights."""
    pred = np.ones_like(y)
    out = scores(y, pred)
    out["note"] = "always predicts compress"  # type: ignore[assignment]
    return out


def cross_validate(X: np.ndarray, y: np.ndarray, *, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    folds = min(5, max(2, min(n_pos, n_neg)))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=3, random_state=seed)
    out: dict[str, Any] = {"folds": folds, "repeats": 3, "models": {}}
    models = {
        "logistic": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
    }
    for name, make in models.items():
        preds = np.zeros(len(y), dtype=float)
        counts = np.zeros(len(y), dtype=float)
        for train, test in cv.split(X, y):
            m = make()
            m.fit(X[train], y[train])
            preds[test] += m.predict_proba(X[test])[:, 1]
            counts[test] += 1
        prob = preds / np.maximum(counts, 1)
        m = scores(y, (prob >= 0.5).astype(int), prob)
        lo, hi = bootstrap_precision(y, prob, seed=seed)
        m["precision_ci90"] = f"{lo}–{hi}"  # type: ignore[assignment]
        out["models"][name] = m
        out.setdefault("oof_prob", {})[name] = prob.tolist()
    return out


def fit_final(X: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[dict[str, Any], Any]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    weights = {
        "version": 1,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURE_NAMES,
        "coef": [round(float(c), 4) for c in model.coef_[0]],
        "intercept": round(float(model.intercept_[0]), 4),
        "decision_threshold": 0.5,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "seed": seed,
    }
    return weights, model


def numpy_probability(x: list[float], weights: dict[str, Any]) -> float:
    z = float(np.dot(np.asarray(weights["coef"], dtype=float), np.asarray(x, dtype=float)))
    return float(1.0 / (1.0 + np.exp(-(z + float(weights["intercept"])))))


def render(
    *,
    rows: list[dict[str, Any]],
    cv: dict[str, Any] | None,
    weights: dict[str, Any] | None,
    rule: dict[str, float] | None,
    always_run: str,
    never_run: str,
    min_saving: float,
    adopted: bool,
    reason: str,
    parity_max_diff: float | None,
) -> str:
    import collections

    by_action = collections.Counter(r["applied"] for r in rows)
    by_label = collections.Counter(r["label"] for r in rows)
    by_intent = collections.Counter(r["intent"] for r in rows)
    lost = [r for r in rows if not r["answer_preserved"]]
    lines = [
        "# Stage C — learned compression classifier (Phase 7)",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · `eval/train_classifier.py` · "
        f"labels from `{always_run}` (forced `--compression always`) against `{never_run}` "
        f"(no compression) · a compressed chunk is positive when the answer is no worse **and** the "
        f"chunk shrank ≥ {min_saving:.0%}.",
        "",
        "## 1. Label set",
        "",
        f"- chunk decisions with an outcome: **{len(rows)}** from {len({r['golden_id'] for r in rows})} golden questions",
        f"- positives (compression was right): **{by_label.get(1, 0)}** · negatives: **{by_label.get(0, 0)}**",
        f"- applied actions: {dict(by_action)}",
        f"- intents: {dict(by_intent)}",
        f"- queries where compression cost quality: {len({r['golden_id'] for r in lost})}"
        + (
            f" ({', '.join(sorted({r['golden_id'] + ': ' + r['why'] for r in lost}))})"
            if lost
            else ""
        ),
        "",
    ]
    if cv is None:
        lines += [
            "## 2. Training",
            "",
            "**Not trained.** " + reason,
            "",
        ]
        return "\n".join(lines)
    lines += [
        "## 2. Cross-validated performance",
        "",
        f"{cv['folds']}-fold stratified CV × {cv['repeats']} repeats (the label set is small, so a single "
        "split would be noise). The rule row is the hand-set Stage B score (§6.5) scored on the same labels.",
        "",
        "| model | n | positives | accuracy | precision (compress) | 90 % CI | recall | ROC AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = cv.get("majority_baseline")
    if base:
        lines.append(
            f"| “always compress” (base rate) | {base['n']} | {base['positives']} | "
            f"{base['accuracy']} | **{base['precision']}** | – | {base['recall']} | – |"
        )
    if rule:
        lines.append(
            f"| Stage B rules (current) | {rule['n']} | {rule['positives']} | {rule['accuracy']} | "
            f"**{rule['precision']}** | – | {rule['recall']} | {rule.get('roc_auc', '–')} |"
        )
    for name, m in cv["models"].items():
        lines.append(
            f"| {name} | {m['n']} | {m['positives']} | {m['accuracy']} | **{m['precision']}** | "
            f"{m.get('precision_ci90', '–')} | {m['recall']} | {m.get('roc_auc', '–')} |"
        )
    lines += [
        "",
        "§6.7 target: precision of “compress” ≥ 0.85 — a wrong compression can lose the answer, a "
        "missed one only costs tokens. **Read the base-rate row first:** with a positive rate this "
        "high, “always compress” already clears the bar, so precision alone cannot justify a model. "
        "Adoption therefore also requires beating that baseline by ≥ 5 pp, beating the rules, and a "
        "floor on the number of *negative* labels — the rows that actually teach the model when not "
        "to compress.",
        "",
    ]
    if weights:
        lines += [
            "## 3. Logistic coefficients (what the model learned)",
            "",
            "| feature | coefficient |",
            "|---|---|",
        ]
        for name, c in zip(weights["features"], weights["coef"], strict=True):
            lines.append(f"| `{name}` | {c:+.3f} |")
        lines += [
            f"| _intercept_ | {weights['intercept']:+.3f} |",
            "",
            f"Exported to `src/rag/compress/classifier_weights.json` (n={weights['n_samples']}, "
            f"positives={weights['n_positive']}).",
            "",
        ]
        if parity_max_diff is not None:
            lines += [
                f"NumPy scorer parity with scikit-learn on the training rows: max |Δp| = "
                f"{parity_max_diff:.2e} (serving never imports scikit-learn).",
                "",
            ]
    lines += [
        "## 4. Decision",
        "",
        f"**Stage C is {'adopted' if adopted else 'NOT adopted'}.** {reason}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--always-run", default="stagec_always")
    ap.add_argument("--never-run", default="golden_phase4_claude")
    ap.add_argument("--min-saving", type=float, default=0.30)
    ap.add_argument(
        "--keep-negatives", action="store_true", help="also label chunks that were kept"
    )
    ap.add_argument("--min-samples", type=int, default=40, help="below this, training is refused")
    ap.add_argument(
        "--min-negatives",
        type=int,
        default=20,
        help="adoption floor: fewer negative labels than this and the rules stay (§6.7)",
    )
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--write-weights", action="store_true", help="export classifier_weights.json")
    ap.add_argument(
        "--report", default=str(PROJECT_ROOT / "docs" / "reports" / "stage_c_classifier.md")
    )
    args = ap.parse_args(argv)

    settings = get_settings()
    decisions = load_jsonl(settings.logs_dir / "compression_decisions.jsonl")
    X, y, rows = build_dataset(
        decisions=decisions,
        always_run=args.always_run,
        never_run=args.never_run,
        min_saving=args.min_saving,
        keep_negatives=args.keep_negatives,
    )
    print(f"labelled chunk decisions: {len(rows)} (positives {int(y.sum()) if len(y) else 0})")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "stage_c_labels.json").write_text(
        json.dumps({"features": FEATURE_NAMES, "rows": rows}, indent=1), encoding="utf-8"
    )

    cv = weights = rule = None
    parity = None
    adopted = False
    if len(rows) < args.min_samples or len(set(y.tolist())) < 2:
        reason = (
            f"Only {len(rows)} labelled chunk decisions with "
            f"{len(set(y.tolist()))} distinct label(s) are available — below the {args.min_samples}-sample "
            "floor this script refuses. Training on this would fit noise, and §6.7 asks for precision "
            "≥ 0.85 on decisions that can lose an answer."
        )
        print(reason)
    else:
        rule = rule_baseline(rows, keep_threshold=0.35)
        cv = cross_validate(X, y, seed=args.seed)
        weights, model = fit_final(X, y, seed=args.seed)
        probs_sk = model.predict_proba(X)[:, 1]
        probs_np = np.array([numpy_probability(r["x"], weights) for r in rows])
        parity = float(np.max(np.abs(probs_sk - probs_np)))
        best = max(cv["models"].items(), key=lambda kv: kv[1]["precision"])
        base = majority_baseline(y)
        log_m = cv["models"]["logistic"]
        n_neg = int(len(y) - y.sum())
        clears_bar = log_m["precision"] >= 0.85
        beats_rules = log_m["precision"] > rule["precision"]
        beats_baseline = log_m["precision"] > base["precision"] + 0.05
        enough_negatives = n_neg >= args.min_negatives
        adopted = clears_bar and beats_rules and beats_baseline and enough_negatives
        blockers = [
            name
            for name, ok in (
                ("precision ≥ 0.85", clears_bar),
                ("beats the Stage B rules", beats_rules),
                (f"beats “always compress” ({base['precision']}) by ≥ 5 pp", beats_baseline),
                (f"≥ {args.min_negatives} negative labels (have {n_neg})", enough_negatives),
            )
            if not ok
        ]
        cv["majority_baseline"] = base
        reason = (
            f"Logistic precision {log_m['precision']} (90 % CI {log_m['precision_ci90']}) vs Stage B "
            f"rules {rule['precision']} and “always compress” {base['precision']}; best model "
            f"{best[0]} at {best[1]['precision']}. "
            + (
                "All four adoption conditions hold, so the learned weights ship behind "
                "`compression.stage_b: learned`."
                if adopted
                else "Unmet condition(s): "
                + "; ".join(blockers)
                + ". Stage B keeps its hand-set weights — a transparent rule beats a model fitted on "
                "this little evidence. The weights file, the NumPy scorer and the "
                "`compression.stage_b` flag stay in the repo so the decision is a config change once "
                "real traffic (or a larger forced-run set) supplies more negatives."
            )
        )
        print(json.dumps(cv, indent=1))
        print(reason)
        if args.write_weights:
            WEIGHTS_PATH.write_text(json.dumps(weights, indent=1), encoding="utf-8")
            print(f"weights written to {WEIGHTS_PATH}")

    report = render(
        rows=rows,
        cv=cv,
        weights=weights,
        rule=rule,
        always_run=args.always_run,
        never_run=args.never_run,
        min_saving=args.min_saving,
        adopted=adopted,
        reason=reason,
        parity_max_diff=parity,
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
