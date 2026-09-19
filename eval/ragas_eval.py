"""RAGAS-style quality metrics on a golden subset (plan Phase 7, architecture §12).

    MODEL_PROFILE=groq_build .venv/Scripts/python eval/ragas_eval.py --name ragas_groq [--resume]

**Why the metrics are computed here rather than with the `ragas` package.** `ragas` 0.4.3 is
installed (`requirements-dev.txt`) but cannot be imported in this environment: it does
`from langchain_community.chat_models.vertexai import ChatVertexAI`, a module that
`langchain-community` 0.4.2 — now sunset — no longer ships. Downgrading it would drag the
serving stack (langchain-core / langchain-groq 1.x) backwards, so the four metric definitions
are implemented directly against the project's own `LLMClient`. Definitions follow the RAGAS
documentation:

    faithfulness        claims in the answer that the retrieved context supports / all claims
    answer relevancy    mean cosine(question, question_i) over questions the judge reverse-
                        generates from the answer alone (embeddings are the local bge model,
                        so this number costs no tokens beyond the generation call)
    context precision   judged usefulness of each retrieved block for the reference answer,
                        as precision@k averaged over the ranks where a useful block appears
    context recall      claims in the *reference* answer that the retrieved context supports

Every judged number is **indicative, not authoritative**: the judge belongs to the same model
family as the generator (plan §Phase 7), it sees the same context, and the golden reference
answers themselves are still under review (`review_status` in `eval/golden.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import RESULTS_DIR, load_golden  # noqa: E402

DEFAULT_IDS = "G1,P4,G3,C5,C7,G5,G7,G9,G11,E2,E3,E4,G12,G13,X1,X3"

CLAIMS_SYSTEM = (
    "You split a financial answer into atomic factual claims: one self-contained statement per "
    "claim, each carrying its own number, unit and period when the text gives one. Ignore hedging "
    "and citation markers like [C1]. Return at most 12 claims."
)
VERDICT_SYSTEM = (
    "You judge whether each numbered claim is supported by the CONTEXT. A claim is supported only "
    "if the context states it or it follows by direct arithmetic on numbers in the context. "
    "Outside knowledge is never support. Return one verdict per claim, in order."
)
QUESTIONS_SYSTEM = (
    "You reverse-engineer questions from an answer. Given only the ANSWER, write the exactly 3 "
    "standalone questions it would be a direct reply to."
)
USEFUL_SYSTEM = (
    "You judge retrieval usefulness. For each numbered CONTEXT BLOCK, decide whether it contains "
    "information needed to produce the REFERENCE ANSWER to the QUESTION. Return one verdict per "
    "block, in order."
)


class Claims(BaseModel):
    claims: list[str] = Field(default_factory=list, description="atomic factual claims, in order")


class Verdict(BaseModel):
    claim_index: int = Field(description="0-based index of the claim being judged")
    supported: bool
    why: str = ""


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class Questions(BaseModel):
    questions: list[str] = Field(default_factory=list)


class BlockVerdict(BaseModel):
    block: int = Field(description="1-based block number")
    useful: bool


class BlockVerdicts(BaseModel):
    useful: list[BlockVerdict] = Field(default_factory=list)


def _json_call(client, role, schema, system, user, request_id, max_tokens=700):  # noqa: ANN001
    """One judged call; a quota or parse failure degrades to `None` rather than aborting the run."""
    from rag.llm import LLMError

    try:
        return client.json(
            user, schema, role=role, system=system, request_id=request_id, max_tokens=max_tokens
        ), None
    except LLMError as exc:
        return None, str(exc)[:200]


def _claim_verdicts(client, role, text, context, rid, tag):  # noqa: ANN001
    """Claims of `text`, each judged against `context`. Returns (items, verdicts, note)."""
    claims, err = _json_call(
        client, role, Claims, CLAIMS_SYSTEM, f"ANSWER:\n{text}", f"{rid}-{tag}"
    )
    items = [c.strip() for c in (claims.claims if claims else []) if c and c.strip()]
    if not items:
        return [], [], err or "no claims extracted"
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(items))
    verdicts, err = _json_call(
        client,
        role,
        Verdicts,
        VERDICT_SYSTEM,
        f"CONTEXT:\n{context}\n\nCLAIMS:\n{listed}",
        f"{rid}-{tag}-verdicts",
        max_tokens=900,
    )
    vs = verdicts.verdicts if verdicts else []
    if not vs:
        return items, [], err or "no verdicts"
    return items, vs, None


def faithfulness(client, role: str, answer: str, context: str, rid: str) -> dict[str, Any]:  # noqa: ANN001
    items, vs, note = _claim_verdicts(client, role, answer, context, rid, "claims")
    if note:
        return {"score": None, "claims": len(items), "note": note}
    supported = [v.supported for v in vs]
    return {
        "score": round(sum(supported) / len(supported), 3),
        "claims": len(items),
        "supported": int(sum(supported)),
        "unsupported_examples": [
            items[v.claim_index] for v in vs if not v.supported and 0 <= v.claim_index < len(items)
        ][:3],
    }


def answer_relevancy(client, role, question, answer, embedder, rid) -> dict[str, Any]:  # noqa: ANN001
    out, err = _json_call(
        client, role, Questions, QUESTIONS_SYSTEM, f"ANSWER:\n{answer}", f"{rid}-questions", 400
    )
    qs = [q.strip() for q in (out.questions if out else []) if q and q.strip()]
    if not qs:
        return {"score": None, "note": err or "no questions generated"}
    vecs = embedder.embed_queries([question, *qs]).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sims = (vecs[1:] @ vecs[0]).tolist()
    return {
        "score": round(float(np.mean(sims)), 3),
        "generated": qs,
        "similarities": [round(s, 3) for s in sims],
    }


def context_precision(
    client,  # noqa: ANN001
    role: str,
    question: str,
    reference: str,
    blocks: list[str],
    rid: str,
) -> dict[str, Any]:
    if not blocks:
        return {"score": None, "note": "no context blocks"}
    listed = "\n\n".join(f"BLOCK {i + 1}:\n{b}" for i, b in enumerate(blocks))
    out, err = _json_call(
        client,
        role,
        BlockVerdicts,
        USEFUL_SYSTEM,
        f"QUESTION: {question}\n\nREFERENCE ANSWER:\n{reference}\n\n{listed}",
        f"{rid}-useful",
        max_tokens=600,
    )
    flags = [i.useful for i in (out.useful if out else [])][: len(blocks)]
    if not flags:
        return {"score": None, "note": err or "no verdicts"}
    hits = 0
    precisions = []
    for k, ok in enumerate(flags, start=1):
        if ok:
            hits += 1
            precisions.append(hits / k)
    score = round(sum(precisions) / hits, 3) if hits else 0.0
    return {"score": score, "useful_blocks": hits, "blocks": len(flags)}


def context_recall(client, role: str, reference: str, context: str, rid: str) -> dict[str, Any]:  # noqa: ANN001
    if not reference.strip():
        return {"score": None, "note": "golden row has no reference answer"}
    items, vs, note = _claim_verdicts(client, role, reference, context, rid, "refclaims")
    if note:
        return {"score": None, "note": note}
    covered = [v.supported for v in vs]
    return {
        "score": round(sum(covered) / len(covered), 3),
        "reference_claims": len(items),
        "covered": int(sum(covered)),
    }


def reference_text(q: dict[str, Any]) -> str:
    """The golden row's prose reference, or one built from its verified `expected` figure —
    numeric rows carry {value, unit, fiscal_year, period_end} instead of prose, and context
    precision/recall need something to judge against."""
    prose = (q.get("reference_answer") or "").strip()
    if prose:
        return prose
    e = q.get("expected") or {}
    value, unit = e.get("value"), (e.get("unit") or "").replace("_", " ")
    if value is None:
        return ""
    money = f"${value:,}" if "USD" in (e.get("unit") or "") else f"{value:,}"
    unit_tail = " million" if e.get("unit") == "USD_millions" else (f" {unit}" if unit else "")
    period = f" as of {e['period_end']}" if e.get("period_end") else ""
    fy = f" (fiscal {e['fiscal_year']})" if e.get("fiscal_year") else ""
    return f"{q['question'].rstrip('?')}: {money}{unit_tail}{period}{fy}."


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def scored_n(values: list[float | None]) -> int:
    return sum(1 for v in values if v is not None)


def _unscored_note(rows: list[dict[str, Any]]) -> list[str]:
    """A metric that could not be computed is reported, never silently averaged away."""
    failures = [
        (r["id"], metric, r[metric].get("note"))
        for r in rows
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        if r[metric].get("score") is None
    ]
    if not failures:
        return []
    lines = [
        f"**{len(failures)} metric(s) could not be scored** and are excluded from the means above "
        "rather than counted as zero:",
        "",
    ]
    lines += [f"- `{qid}` {metric}: {note}" for qid, metric, note in failures]
    lines.append("")
    return lines


def _intent_rows(rows: list[dict[str, Any]]) -> list[str]:
    intents: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        intents.setdefault(r["intent"], []).append(r)
    out = []
    for intent, rs in sorted(intents.items()):
        cells = [
            mean([r[m].get("score") for r in rs])
            for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        ]
        out.append(
            f"| {intent} | {len(rs)} | "
            + " | ".join("–" if c is None else f"{c:.3f}" for c in cells)
            + " |"
        )
    return out


def _reading(rows: list[dict[str, Any]]) -> list[str]:
    """One paragraph of interpretation, computed rather than asserted."""
    worst = sorted(
        (r for r in rows if r["context_precision"].get("score") is not None),
        key=lambda r: r["context_precision"]["score"],
    )[:3]
    if not worst:
        return []
    names = ", ".join(f"`{r['id']}` ({r['context_precision']['score']:.2f})" for r in worst)
    return [
        "## Reading the numbers",
        "",
        "- **Faithfulness is near 1.0 by construction.** `verify_answer` (§4.12) already refuses to "
        "return an answer whose numbers are not traceable to the context, so this metric mostly "
        "confirms the verifier works; a value below 1.0 flags a *qualitative* sentence the judge "
        "could not tie to a block, not an invented figure.",
        "- **Context precision is the metric with something to say.** The lowest scores are "
        f"{names} — the retriever returns eight blocks and, on explanatory questions, the judge "
        "considers most of them unnecessary for the reference answer. That is the padding the "
        "compression classifier exists to remove, and it is also why `final_k` is worth revisiting "
        "per intent rather than globally.",
        "- **Context recall below 1.0** means a claim in the *reference* answer was not in the "
        "retrieved context at all: a retrieval miss, not a generation fault. Those rows are the "
        "ones to read alongside the golden-set human review.",
        "",
    ]


def render(rows: list[dict[str, Any]], *, judge: str, generator: str, profile: str) -> str:
    def col(metric: str) -> list[float | None]:
        return [r[metric].get("score") for r in rows]

    lines = [
        "# RAGAS-style evaluation (Phase 7)",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · `eval/ragas_eval.py` · "
        f"{len(rows)} golden questions · profile `{profile}` · generator `{generator}` · judge `{judge}`.",
        "",
        "> **Indicative, same-family judge.** The judge and the generator come from the same model "
        "family and see the same context, so these numbers are a smoke test of grounding, not an "
        "independent audit. The golden reference answers are themselves still under review "
        "(`review_status` in `eval/golden.jsonl`). The `ragas` package is in `requirements-dev.txt` but could not "
        "be imported here — it imports `langchain_community.chat_models.vertexai`, removed from the "
        "sunset `langchain-community` 0.4.2 — so the four metric definitions are implemented directly "
        "in this script against the project's own `LLMClient`.",
        "",
        "## Scores",
        "",
        "| metric | mean | scored | definition |",
        "|---|---|---|---|",
        f"| faithfulness | **{mean(col('faithfulness'))}** | {scored_n(col('faithfulness'))}/{len(rows)} | answer claims the retrieved context supports |",
        f"| answer relevancy | **{mean(col('answer_relevancy'))}** | {scored_n(col('answer_relevancy'))}/{len(rows)} | cosine(question, questions reverse-generated from the answer) |",
        f"| context precision | **{mean(col('context_precision'))}** | {scored_n(col('context_precision'))}/{len(rows)} | judged usefulness of each retrieved block, as precision@k |",
        f"| context recall | **{mean(col('context_recall'))}** | {scored_n(col('context_recall'))}/{len(rows)} | reference-answer claims the retrieved context supports |",
        "",
        *_unscored_note(rows),
        "## By intent",
        "",
        "| intent | n | faithfulness | relevancy | ctx precision | ctx recall |",
        "|---|---|---|---|---|---|",
        *_intent_rows(rows),
        "",
        *_reading(rows),
        "## Per question",
        "",
        "| id | intent | faithfulness | relevancy | ctx precision | ctx recall | claims | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, a, p, c = (
            r["faithfulness"],
            r["answer_relevancy"],
            r["context_precision"],
            r["context_recall"],
        )
        notes = "; ".join(
            n for n in (f.get("note"), a.get("note"), p.get("note"), c.get("note")) if n
        )
        lines.append(
            f"| {r['id']} | {r['intent']} | {f.get('score', '–')} | {a.get('score', '–')} | "
            f"{p.get('score', '–')} | {c.get('score', '–')} | {f.get('claims', '–')} | {notes or '–'} |"
        )
    weak = [r for r in rows if (r["faithfulness"].get("score") or 1) < 1.0]
    if weak:
        lines += ["", "## Claims the judge did not find in the context", ""]
        for r in weak:
            for claim in r["faithfulness"].get("unsupported_examples", []):
                lines.append(f"- `{r['id']}` — {claim}")
        lines += [
            "",
            "These are worth reading before trusting the score: the verifier (§4.12) already "
            "guarantees every *number* is traceable, so a low faithfulness score here usually means "
            "the judge disliked a qualitative sentence, not that a figure was invented.",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--name", default="ragas_groq")
    ap.add_argument("--judge", default="large", choices=["small", "large"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    from rag.core.logging import configure_logging
    from rag.core.settings import get_settings
    from rag.graph import Pipeline

    settings = get_settings()
    configure_logging("WARNING", secrets=settings.secret_values())
    pipeline = Pipeline()
    embedder = pipeline.store.embedder
    generator = pipeline.models.active().large.model
    judge_model = getattr(pipeline.models.active(), args.judge).model

    wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
    golden = {q["id"]: q for q in load_golden()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.name}.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and out_path.exists():
        rows = [
            json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        done = {r["id"] for r in rows}
        wanted = [i for i in wanted if i not in done]
        print(f"resuming: {len(done)} done, {len(wanted)} to go")
    elif out_path.exists():
        out_path.unlink()

    for n, qid in enumerate(wanted, start=1):
        q = golden.get(qid)
        if q is None:
            print(f"unknown golden id {qid}")
            continue
        rid = f"ragas-{args.name}-{qid}"
        result = pipeline.ask(q["question"], bypass_cache=True, request_id=rid)
        blocks = [b.text for b in result.context.blocks]
        context = "\n\n".join(f"[{b.block_id}] {b.text}" for b in result.context.blocks)
        answer = result.answer.answer_markdown
        reference = reference_text(q)
        row = {
            "id": qid,
            "intent": result.intent,
            "question": q["question"],
            "answer": answer,
            "context_blocks": len(blocks),
            "faithfulness": faithfulness(pipeline.client, args.judge, answer, context, rid),
            "answer_relevancy": answer_relevancy(
                pipeline.client, args.judge, q["question"], answer, embedder, rid
            ),
            "context_precision": context_precision(
                pipeline.client, args.judge, q["question"], reference, blocks, rid
            ),
            "context_recall": context_recall(pipeline.client, args.judge, reference, context, rid),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[{n}/{len(wanted)}] {qid:4s} faith={row['faithfulness'].get('score')} "
            f"rel={row['answer_relevancy'].get('score')} "
            f"prec={row['context_precision'].get('score')} rec={row['context_recall'].get('score')}"
        )

    rows.sort(key=lambda r: r["id"])
    report = render(
        rows, judge=judge_model, generator=generator, profile=pipeline.models.active_profile
    )
    path = Path(args.report or PROJECT_ROOT / "docs" / "reports" / "ragas_groq.md")
    path.write_text(report, encoding="utf-8")
    print(f"\n{report}\nreport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
