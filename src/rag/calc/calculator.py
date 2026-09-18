"""Deterministic financial calculator (architecture §4.10, FR-8, P3).

Inputs come from row-fact metadata (`value_fy{YYYY}`), never from generated text. Formulas
live in `config/formulas.yaml`; expressions are evaluated by a tiny whitelisted AST evaluator
(no `eval`). A missing line item produces a `missing_inputs` result that names it — the model
is told to say so, never to estimate.

Phase 3 selects formulas by keyword (`select_formulas`); Phase 4 selects them from slots.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.config import Formula, FormulasConfig, load_formulas_config
from rag.core.schema import Chunk

# ------------------------------------------------------------- expression evaluator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "min": min, "max": max}


def evaluate_expression(expression: str, variables: dict[str, float]) -> float:
    """Evaluate `+ - * /`, unary minus, parentheses, abs/min/max over named variables."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"bad expression {expression!r}: {exc}") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable {node.id!r}")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
        ):
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            return float(_FUNCS[node.func.id](*(walk(a) for a in node.args)))
        raise ValueError(f"disallowed syntax in expression: {ast.dump(node)[:60]}")

    return walk(tree)


# ------------------------------------------------------------------------ data model


class CalcInput(BaseModel):
    name: str
    line_item: str
    line_item_norm: str
    value: float | None
    nil: bool = False
    fiscal_year: int
    period_end: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    unit: str | None = None


class CalculationResult(BaseModel):
    formula: str
    description: str
    kind: Literal["ratio", "amount", "pct"]
    expression: str
    fiscal_year: int
    metric: str | None = None
    inputs: list[CalcInput] = Field(default_factory=list)
    result: float | None = None
    rounded: float | None = None
    round_digits: int = 2
    unit: str = "USD_millions"
    status: Literal["ok", "missing_inputs", "error"] = "ok"
    missing: list[str] = Field(default_factory=list)
    message: str = ""

    def formatted(self) -> str:
        if self.rounded is None:
            return "n/a"
        if self.kind == "amount":
            sign = "-" if self.rounded < 0 else ""
            return f"{sign}${abs(self.rounded):,.0f} million"
        if self.kind == "pct":
            return f"{self.rounded:+.{self.round_digits}f}%"
        return f"{self.rounded:.{self.round_digits}f}"

    def as_context_block(self, block_id: str) -> str:
        """Pre-computed fact block injected into the prompt; the model must use it verbatim."""
        title = self.description
        if self.metric:
            title = f"{title} — {self.metric}"
        lines = [f"[{block_id} | CALCULATION | {self.formula} | FY{self.fiscal_year}]", title]
        if self.status != "ok":
            lines.append(f"STATUS: {self.status} — {self.message}")
            return "\n".join(lines)
        for i in self.inputs:
            v = "— (nil, treated as 0)" if i.nil else f"{i.value:,.0f}"
            lines.append(
                f"  {i.name} = {i.line_item} FY{i.fiscal_year} ({i.period_end}) = {v} "
                f"{i.unit or ''} [source {i.chunk_id}, PDF p.{i.page}]"
            )
        lines.append(f"  {self.expression} = {self.result:.6g}")
        lines.append(f"  RESULT: {self.formatted()} (rounded to {self.round_digits} decimals)")
        return "\n".join(lines)


# -------------------------------------------------------------------- row-fact source


class RowFactSource:
    """Lookup of row facts by (statement, line_item_norm) — from the index store or a list."""

    def __init__(self, chunks: Iterable[Chunk]):
        self._by_key: dict[tuple[str, str], Chunk] = {}
        for c in chunks:
            m = c.metadata
            if m.modality == "row_fact" and m.line_item_norm:
                self._by_key.setdefault((m.statement, m.line_item_norm), c)

    def get(self, statement: str, line_item_norm: str) -> Chunk | None:
        return self._by_key.get((statement, line_item_norm))

    def line_items(self, statement: str | None = None) -> list[str]:
        return sorted({k[1] for k in self._by_key if statement is None or k[0] == statement})

    def statement_of(self, line_item_norm: str) -> str | None:
        for st, li in self._by_key:
            if li == line_item_norm:
                return st
        return None


# --------------------------------------------------------------------------- calculator


class Calculator:
    def __init__(self, source: RowFactSource, config: FormulasConfig | None = None):
        self.source = source
        self.config = config or load_formulas_config()

    def formulas(self) -> dict[str, Formula]:
        return self.config.formulas

    def compute(
        self, formula_name: str, *, fiscal_year: int, metric: str | None = None
    ) -> CalculationResult:
        formula = self.config.formulas[formula_name]  # KeyError for unknown formulas
        result = CalculationResult(
            formula=formula_name,
            description=formula.description,
            kind=formula.kind,
            expression=formula.expression,
            fiscal_year=fiscal_year,
            metric=metric,
            round_digits=formula.round,
        )
        if formula.generic and not metric:
            result.status = "missing_inputs"
            result.missing = ["metric"]
            result.message = "This formula needs a line item (metric) to apply to."
            return result

        variables: dict[str, float] = {}
        for name, spec in formula.inputs.items():
            line_item = spec.line_item.replace("{metric}", metric or "")
            fy = fiscal_year if spec.period == "current" else fiscal_year - 1
            statement = (
                formula.statement
                or self.source.statement_of(line_item)
                or self.config.default_statement
            )
            chunk = self.source.get(statement, line_item)
            value = getattr(chunk.metadata, f"value_fy{fy}", None) if chunk else None
            has_period = chunk is not None and f"value_fy{fy}" in chunk.metadata.model_fields_set
            nil = chunk is not None and has_period and value is None
            if chunk is None or not has_period:
                result.missing.append(f"{line_item} (FY{fy})")
                continue
            if nil and not spec.optional:
                result.missing.append(f"{line_item} (FY{fy} is nil)")
                continue
            calc_input = CalcInput(
                name=name,
                line_item=chunk.metadata.line_item or line_item,
                line_item_norm=line_item,
                value=0.0 if nil else value,
                nil=nil,
                fiscal_year=fy,
                period_end=getattr(chunk.metadata, f"period_end_fy{fy}", None),
                chunk_id=chunk.id,
                page=chunk.metadata.page,
                unit=chunk.metadata.unit,
            )
            result.inputs.append(calc_input)
            variables[name] = calc_input.value or 0.0
            if calc_input.unit and formula.kind == "amount":
                result.unit = calc_input.unit

        if result.missing:
            result.status = "missing_inputs"
            result.message = "Line item not found in the indexed statements: " + "; ".join(
                result.missing
            )
            return result
        try:
            value = evaluate_expression(formula.expression, variables)
        except (ValueError, ZeroDivisionError) as exc:
            result.status = "error"
            result.message = f"could not evaluate: {exc}"
            return result
        result.result = value
        result.rounded = round(value, formula.round) if formula.round else float(round(value))
        if formula.kind in {"ratio", "pct"}:
            result.unit = "ratio" if formula.kind == "ratio" else "percent"
        return result


# ------------------------------------------------------------------ keyword selection


@dataclass(frozen=True)
class FormulaRequest:
    formula: str
    fiscal_year: int
    metric: str | None = None


_FY_PATTERNS = [
    (re.compile(r"\bfy\s?(20)?(\d{2})\b", re.I), lambda m: 2000 + int(m.group(2))),
    (re.compile(r"\bfiscal(?: year)?\s+(20\d{2})\b", re.I), lambda m: int(m.group(1))),
    (
        re.compile(r"\b(?:jan(?:uary)?\.?\s+\d{1,2},?\s+)(20\d{2})\b", re.I),
        lambda m: int(m.group(1)),
    ),
    (re.compile(r"\byear[- ]end(?:ed)?\s+(20\d{2})\b", re.I), lambda m: int(m.group(1))),
]
DEFAULT_FISCAL_YEAR = 2026
YOY_KEYWORDS = (
    "year-over-year",
    "year over year",
    "yoy",
    "grow",
    "growth",
    "change",
    "increase",
    "decrease",
    "compared",
)


def detect_fiscal_year(question: str, default: int = DEFAULT_FISCAL_YEAR) -> int:
    """The fiscal year a computation applies to. When a question spans years ("from fiscal 2025
    to fiscal 2026") the latest one is the current period and the earlier one the comparison."""
    years = [conv(m) for pattern, conv in _FY_PATTERNS for m in pattern.finditer(question)]
    return max(years) if years else default


def match_line_item(question: str, line_items: list[str]) -> str | None:
    """Longest line item whose normalised form appears in the question."""
    q = re.sub(r"[^a-z0-9$%.' ]+", " ", question.lower().replace("’", "'"))
    q = re.sub(r"\s+", " ", q)
    best: str | None = None
    for item in line_items:
        if item in q and (best is None or len(item) > len(best)):
            best = item
    return best


def select_formulas(
    question: str, line_items: list[str], config: FormulasConfig | None = None
) -> list[FormulaRequest]:
    """Phase 3 keyword selection. Named ratios win; YoY applies when a change keyword and a
    line item are both present."""
    cfg = config or load_formulas_config()
    q = question.lower()
    fy = detect_fiscal_year(question)
    requests: list[FormulaRequest] = []
    for name, formula in cfg.formulas.items():
        if formula.generic:
            continue
        if any(k in q for k in formula.keywords):
            requests.append(FormulaRequest(name, fy))
    if requests:
        return requests
    if any(k in q for k in YOY_KEYWORDS):
        metric = match_line_item(question, line_items)
        if metric:
            return [
                FormulaRequest("yoy_change_pct", fy, metric),
                FormulaRequest("yoy_change_abs", fy, metric),
            ]
    return []
