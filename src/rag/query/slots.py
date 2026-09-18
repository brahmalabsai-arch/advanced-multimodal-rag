"""Query normalisation and slot extraction — rules only, no LLM (architecture §4.2, plan Phase 4).

The `QuerySlots` object feeds the scope gate, intent rules, glossary expansion, retrieval
filters, the calculator (formula selection) and, in Phase 6, the cache guard and the L1 key.

Extraction runs longest-phrase-first over a normalised copy of the question and marks every
matched span as consumed, so "current ratio" never leaks a `current` time anchor, "fiscal 2026"
never leaves a bare "2026" behind, and "total current assets" wins over "current assets".
Order: fiscal periods (relative phrases, FY / fiscal / date forms, bare years) → glossary
(statements, formulas, metrics) → entity aliases → lexicons (direction, aggregation, anchor).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from rag.core.config import (
    FiscalCalendar,
    FormulasConfig,
    GlossaryConfig,
    load_fiscal_calendar,
    load_formulas_config,
    load_glossary_config,
)

MatchKind = Literal[
    "period", "statement", "formula", "metric", "entity", "direction", "aggregation", "anchor"
]

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9,
    "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}  # fmt: skip
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Unicode spaces the models and PDFs emit inside dates ("January 25, 2026").
_UNICODE_SPACES = str.maketrans(
    dict.fromkeys(map(chr, (0x202F, 0xA0, 0x2009, 0x2007, 0x2008, 0x200A)), " ")
)
_CURLY_QUOTES = str.maketrans(dict.fromkeys(map(chr, (0x2019, 0x2018)), "'"))
_DASHES = str.maketrans(dict.fromkeys(map(chr, (0x2013, 0x2014, 0x2D, 0x5F)), " "))


# ------------------------------------------------------------------------ normalisation


def normalize_text(question: str) -> str:
    """Lower-case, NFKC, straight quotes, possessives dropped, punctuation stripped except
    `$ % .` inside numbers and `& /` inside tickers ("pp&e", "a/r"); hyphens become spaces."""
    t = unicodedata.normalize("NFKC", question).translate(_UNICODE_SPACES).lower()
    t = t.translate(_CURLY_QUOTES)
    t = re.sub(r"'s\b", "", t)
    t = t.replace("'", "")
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = t.translate(_DASHES)
    t = re.sub(r"[^a-z0-9$%.&/ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # "/" and "&" bind tokens together ("debt/equity", "pp&e"), so they are not boundaries.
    return re.compile(r"(?<![a-z0-9/&])" + re.escape(phrase) + r"(?![a-z0-9/&])")


# ------------------------------------------------------------------------------- output


class SlotMatch(BaseModel):
    kind: MatchKind
    phrase: str = Field(description="the matched text in the normalised question")
    canonical: str
    start: int
    end: int


class QuerySlots(BaseModel):
    entity: str | None = None
    fiscal_periods: list[str] = Field(
        default_factory=list, description="FY2026 | AMBIGUOUS_2025, ascending"
    )
    resolved_fiscal_years: list[int] = Field(default_factory=list)
    period_note: str | None = Field(
        default=None, description="set when a bare calendar year had to be interpreted"
    )
    metrics: list[str] = Field(default_factory=list, description="line_item_norm keys")
    formulas: list[str] = Field(default_factory=list, description="formula ids")
    statement: str | None = None
    statement_source: Literal["explicit", "inferred"] | None = None
    direction: Literal["increase", "decrease"] | None = None
    aggregation: list[str] = Field(default_factory=list)
    time_anchor: bool = False
    normalized_text: str = ""
    canonical_text: str = Field(
        default="", description="normalised text with matched spans canonicalised (L1 key input)"
    )
    matches: list[SlotMatch] = Field(default_factory=list)

    @property
    def current_fiscal_year(self) -> int | None:
        return max(self.resolved_fiscal_years) if self.resolved_fiscal_years else None

    @property
    def slot_rich(self) -> bool:
        """Slot-rich queries get the looser cache similarity threshold (§5.8)."""
        return bool((self.metrics or self.formulas) and self.fiscal_periods)

    def l1_key(self, version_keys: str = "") -> str:
        return hashlib.sha256((self.canonical_text + "|" + version_keys).encode()).hexdigest()


# ---------------------------------------------------------------------------- extractor


@dataclass
class _Phrase:
    text: str
    kind: MatchKind
    canonical: str
    pattern: re.Pattern[str]


class SlotExtractor:
    def __init__(
        self,
        glossary: GlossaryConfig | None = None,
        calendar: FiscalCalendar | None = None,
        formulas: FormulasConfig | None = None,
    ):
        self.glossary = glossary or load_glossary_config()
        self.calendar = calendar or load_fiscal_calendar()
        self.formulas = formulas or load_formulas_config()
        self._phrases = self._build_phrases()
        self._relative = self._build_relative()
        self._entity = [
            _Phrase(a, "entity", self.calendar.entity, _phrase_pattern(a))
            for a in sorted(
                {normalize_text(a) for a in self.calendar.entity_aliases}, key=len, reverse=True
            )
        ]
        lex = self.glossary.lexicons
        self._direction = self._lexicon(
            {"increase": lex.direction.increase, "decrease": lex.direction.decrease}, "direction"
        )
        self._aggregation = self._lexicon(lex.aggregation, "aggregation")
        self._anchor = self._lexicon({"anchor": lex.time_anchor}, "anchor")
        self._years = sorted(self.calendar.fiscal_years)

    # -- construction ------------------------------------------------------------------

    def _build_phrases(self) -> list[_Phrase]:
        out: list[_Phrase] = []
        for sid, st in self.glossary.statements.items():
            out += [
                _Phrase(s, "statement", sid, _phrase_pattern(s)) for s in _norm_set(st.synonyms)
            ]
        for fid, f in self.glossary.formulas.items():
            out += [_Phrase(s, "formula", fid, _phrase_pattern(s)) for s in _norm_set(f.synonyms)]
        for mid, m in self.glossary.metrics.items():
            out += [_Phrase(s, "metric", mid, _phrase_pattern(s)) for s in _norm_set(m.synonyms)]
        out.sort(key=lambda p: len(p.text), reverse=True)
        return out

    def _build_relative(self) -> list[_Phrase]:
        latest = self.calendar.latest_fiscal_year
        out = [
            _Phrase(s, "period", f"FY{latest}", _phrase_pattern(s))
            for s in _norm_set(self.calendar.relative_periods.latest)
        ]
        out += [
            _Phrase(s, "period", f"FY{latest - 1}", _phrase_pattern(s))
            for s in _norm_set(self.calendar.relative_periods.prior)
        ]
        out.sort(key=lambda p: len(p.text), reverse=True)
        return out

    @staticmethod
    def _lexicon(table: dict[str, list[str]], kind: MatchKind) -> list[_Phrase]:
        out = [
            _Phrase(s, kind, key, _phrase_pattern(s))
            for key, words in table.items()
            for s in _norm_set(words)
        ]
        out.sort(key=lambda p: len(p.text), reverse=True)
        return out

    # -- extraction --------------------------------------------------------------------

    def extract(self, question: str) -> QuerySlots:
        text = normalize_text(question)
        consumed = [False] * len(text)
        matches: list[SlotMatch] = []

        def take(start: int, end: int, kind: MatchKind, canonical: str) -> bool:
            if any(consumed[start:end]):
                return False
            for i in range(start, end):
                consumed[i] = True
            matches.append(
                SlotMatch(
                    kind=kind, phrase=text[start:end], canonical=canonical, start=start, end=end
                )
            )
            return True

        def scan(phrases: list[_Phrase], stop_after_first: bool = False) -> None:
            for p in phrases:
                for m in p.pattern.finditer(text):
                    if take(m.start(), m.end(), p.kind, p.canonical) and stop_after_first:
                        return

        # 1. fiscal periods
        scan(self._relative)
        periods, note = self._periods(text, take)
        periods = sorted(
            set(periods) | {m.canonical for m in matches if m.kind == "period"}, key=self._resolve
        )

        # 2. glossary (statements, formulas, metrics), longest first across all kinds
        scan(self._phrases)
        # 3. entity
        scan(self._entity)
        # 4. lexicons over what is left
        scan(self._direction)
        scan(self._aggregation)
        scan(self._anchor)

        matches.sort(key=lambda m: m.start)
        metrics = _unique(m.canonical for m in matches if m.kind == "metric")
        formulas = _unique(m.canonical for m in matches if m.kind == "formula")
        statements = _unique(m.canonical for m in matches if m.kind == "statement")
        directions = _unique(m.canonical for m in matches if m.kind == "direction")
        aggregation = _unique(m.canonical for m in matches if m.kind == "aggregation")

        statement, source = self._statement(statements, metrics, formulas)
        return QuerySlots(
            entity=next((m.canonical for m in matches if m.kind == "entity"), None),
            fiscal_periods=periods,
            resolved_fiscal_years=sorted({self._resolve(p) for p in periods}),
            period_note=note,
            metrics=metrics,
            formulas=formulas,
            statement=statement,
            statement_source=source,
            direction=directions[0] if len(directions) == 1 else None,
            aggregation=aggregation,
            time_anchor=any(m.kind == "anchor" for m in matches),
            normalized_text=text,
            canonical_text=_canonicalize(text, matches),
            matches=matches,
        )

    # -- fiscal periods ---------------------------------------------------------------

    def _periods(self, text: str, take) -> tuple[list[str], str | None]:  # noqa: ANN001
        found: set[str] = set()
        note: str | None = None

        def add_fy(fy: int, start: int, end: int, phrase_kind: str = "FY") -> None:
            if fy in self.calendar.fiscal_years and take(
                start, end, "period", f"{phrase_kind}{fy}"
            ):
                found.add(f"{phrase_kind}{fy}")

        # fy26 / fy 2026 / fy2026
        for m in re.finditer(r"(?<![a-z0-9])fy ?(\d{4}|\d{2})(?![a-z0-9])", text):
            add_fy(_expand_year(m.group(1)), m.start(), m.end())
        # fiscal 2026 / fiscal year 2026 / fiscal year ended 2026 / fiscal year end 2026
        for m in re.finditer(
            r"(?<![a-z0-9])fiscal(?: year)?(?: end(?:ed|ing)?)?(?: of)? ?(\d{4}|\d{2})(?![a-z0-9])",
            text,
        ):
            add_fy(_expand_year(m.group(1)), m.start(), m.end())
        # fiscal 2026 year end
        for m in re.finditer(r"(?<![a-z0-9])fiscal (\d{4}) year end(?![a-z0-9])", text):
            add_fy(int(m.group(1)), m.start(), m.end())
        # january 25 2026 / jan 25 2026 / january 2026 → fiscal year containing the date
        for m in re.finditer(
            rf"(?<![a-z0-9])({_MONTH_RE}) ?(\d{{1,2}})? ?(\d{{4}})(?![a-z0-9])", text
        ):
            day = date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2) or 15))
            fy = self.calendar.fiscal_year_of(day)
            if fy is not None:
                add_fy(fy, m.start(), m.end())
        # 2026 01 25 (from 2026-01-25) and 1/25/2026
        for m in re.finditer(r"(?<![a-z0-9])(\d{4}) (\d{2}) (\d{2})(?![a-z0-9])", text):
            fy = self.calendar.fiscal_year_of(_safe_date(m.group(1), m.group(2), m.group(3)))
            if fy is not None:
                add_fy(fy, m.start(), m.end())
        for m in re.finditer(r"(?<![a-z0-9/])(\d{1,2})/(\d{1,2})/(\d{4})(?![a-z0-9/])", text):
            fy = self.calendar.fiscal_year_of(_safe_date(m.group(3), m.group(1), m.group(2)))
            if fy is not None:
                add_fy(fy, m.start(), m.end())
        # year end 2025 / year ended 2026
        for m in re.finditer(r"(?<![a-z0-9])year end(?:ed|ing)? (\d{4})(?![a-z0-9])", text):
            add_fy(int(m.group(1)), m.start(), m.end())
        # bare calendar year → AMBIGUOUS_<year>, resolved to the fiscal year that holds most of it
        for m in re.finditer(r"(?<![a-z0-9$%.])(20\d{2})(?![a-z0-9%.])", text):
            year = int(m.group(1))
            shifted = year + self.calendar.bare_year_offset
            if year not in self.calendar.fiscal_years and shifted not in self.calendar.fiscal_years:
                continue  # "2021" in "$100 invested on 1/31/2021" is not a fiscal period here
            resolved = self._resolve(f"AMBIGUOUS_{year}")
            if take(m.start(), m.end(), "period", f"AMBIGUOUS_{year}"):
                found.add(f"AMBIGUOUS_{year}")
                note = (
                    f"'{year}' was read as fiscal {resolved} (fiscal {resolved} ended "
                    f"{self.calendar.year_end(resolved):%B %d, %Y})"
                )
        return sorted(found, key=self._resolve), note

    def _resolve(self, period: str) -> int:
        if period.startswith("FY"):
            return int(period[2:])
        year = int(period.rsplit("_", 1)[1])
        shifted = year + self.calendar.bare_year_offset
        if shifted in self.calendar.fiscal_years:
            return shifted
        return min(max(year, self._years[0]), self.calendar.latest_fiscal_year)

    # -- statement --------------------------------------------------------------------

    def _statement(
        self, explicit: list[str], metrics: list[str], formulas: list[str]
    ) -> tuple[str | None, Literal["explicit", "inferred"] | None]:
        if explicit:
            return explicit[0], "explicit"
        inferred = {self.glossary.metrics[m].statement for m in metrics}
        for f in formulas:
            formula = self.formulas.formulas.get(f)
            if formula is not None:
                inferred.add(formula.statement or self.formulas.default_statement)
        if len(inferred) == 1:
            return next(iter(inferred)), "inferred"
        return None, None

    # -- helpers for other nodes -----------------------------------------------------

    def statement_label(self, statement: str | None) -> str | None:
        st = self.glossary.statements.get(statement or "")
        return st.label if st else None

    def metric_statement(self, metric: str) -> str | None:
        m = self.glossary.metrics.get(metric)
        return m.statement if m else None


# ------------------------------------------------------------------------------ helpers


def _norm_set(phrases: list[str]) -> list[str]:
    return sorted({normalize_text(p) for p in phrases if normalize_text(p)}, key=len, reverse=True)


def _unique(items) -> list[str]:  # noqa: ANN001
    out: list[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


def _expand_year(raw: str) -> int:
    return int(raw) if len(raw) == 4 else 2000 + int(raw)


def _safe_date(y: str, m: str, d: str) -> date:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return date(1900, 1, 1)


def _canonicalize(text: str, matches: list[SlotMatch]) -> str:
    """Replace matched spans with canonical tokens (periods, metrics, formulas, statements,
    entity); lexicon words stay as written so 'increase' and 'decrease' keys stay distinct."""
    out: list[str] = []
    cursor = 0
    for m in matches:
        if m.kind in {"direction", "aggregation", "anchor"}:
            continue
        out.append(text[cursor : m.start])
        out.append(m.canonical.lower().replace(" ", "_").replace("'", ""))
        cursor = m.end
    out.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


@lru_cache(maxsize=1)
def get_slot_extractor() -> SlotExtractor:
    return SlotExtractor()


def extract_slots(question: str) -> QuerySlots:
    return get_slot_extractor().extract(question)
