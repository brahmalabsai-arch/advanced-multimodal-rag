"""Sentence segmentation for semantic chunking (architecture §3.3, §8.1).

spaCy blank English pipeline + rule-based `sentencizer` + a small custom component that
undoes splits after filing-specific abbreviations the tokenizer does not know
("No.", "approx.", "vs.", "Nos.", "Sec.") and after "Item 1A." / "Note 12." style references.
Runs only at ingestion; serving reads the precomputed sentence sidecars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Tokens (case-sensitive, as they appear) after which a period never ends a sentence.
ABBREVIATIONS = {
    "No.",
    "Nos.",
    "approx.",
    "Approx.",
    "vs.",
    "Sec.",
    "Secs.",
    "Ann.",
    "Gen.",
    "Rev.",
    "Reg.",
    "Art.",
    "Fig.",
    "Figs.",
    "p.",
    "pp.",
    "cf.",
    "viz.",
    "Ex.",
    "Sched.",
    "Div.",
}
# Case-sensitive on purpose: "No." is an abbreviation, "no." ends a sentence.
_ABBREV_STEMS = {a.rstrip(".") for a in ABBREVIATIONS}
# Only "Item 1A." — "Note 12." and "Section 5." routinely end sentences in filings.
_ITEM_REF = re.compile(r"^Item$", re.I)
_ITEM_NUMBER = re.compile(r"^\d{1,2}[A-Za-z]?\.$")


def _filing_boundaries(doc):  # noqa: ANN001 — spaCy Doc
    """spaCy component: clear sentence starts that follow known non-terminal periods."""
    tokens = list(doc)
    for i, tok in enumerate(tokens[1:], start=1):
        prev = tokens[i - 1]
        if not tok.is_sent_start:
            continue
        prev_text = prev.text
        # "approx." kept as one token, or split into "Approx" + "."
        if prev_text in ABBREVIATIONS or (
            prev_text == "." and i >= 2 and tokens[i - 2].text in _ABBREV_STEMS
        ):
            tok.is_sent_start = False
            continue
        # "Item 1A ." where the period became its own token
        if (
            prev_text == "."
            and i >= 3
            and _ITEM_REF.match(tokens[i - 3].text)
            and re.match(r"^\d{1,2}[A-Za-z]?$", tokens[i - 2].text)
        ):
            tok.is_sent_start = False
            continue
        # "Item 1A." kept as one token
        if i >= 2 and _ITEM_NUMBER.match(prev_text) and _ITEM_REF.match(tokens[i - 2].text):
            tok.is_sent_start = False
    return doc


@lru_cache(maxsize=1)
def build_nlp():
    import spacy
    from spacy.language import Language

    if "filing_boundaries" not in Language.factories:
        Language.component("filing_boundaries", func=_filing_boundaries)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("filing_boundaries", after="sentencizer")
    return nlp


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str


_WS = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Docling emits double spaces inside paragraphs; collapse them, keep newlines out."""
    return _WS.sub(" ", text).strip()


def sentencize(text: str) -> list[Sentence]:
    """Split `text` (already whitespace-normalised) into sentences with char offsets."""
    nlp = build_nlp()
    doc = nlp(text)
    out: list[Sentence] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue
        start = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
        out.append(Sentence(start=start, end=start + len(s), text=s))
    return out
