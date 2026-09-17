"""Lexical retrieval: finance-aware tokenizer + BM25Okapi index (architecture §3.6, §8.2).

Local Chroma has no sparse index, so BM25 runs in-process behind this `SparseRetriever`-shaped
class and is pickled next to the Chroma directory. The tokenizer keeps numbers and `$` amounts as
tokens (with thousands separators removed, so "21,403" == "21403"), keeps hyphenated names whole
*and* split ("non-marketable" → "non-marketable", "non", "marketable"), and lower-cases everything.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

TOKENIZER_VERSION = "finance-v1"

_TOKEN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)
_NUMBER = re.compile(r"^\$?\d[\d,]*(?:\.\d+)?%?$")


def finance_tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if _NUMBER.match(tok):
            plain = tok.replace(",", "")
            out.append(plain)
            if plain.startswith("$"):
                out.append(plain[1:])  # "$21403" also matches a bare "21403"
            continue
        out.append(tok)
        if "-" in tok or "'" in tok:
            out.extend(p for p in re.split(r"[-']", tok) if p)
    return out


class BM25Index:
    def __init__(self, ids: list[str], tokenized: list[list[str]]):
        from rank_bm25 import BM25Okapi

        self.ids = list(ids)
        self._model = BM25Okapi(tokenized)
        self.tokenizer_version = TOKENIZER_VERSION

    @classmethod
    def build(cls, ids: list[str], documents: list[str]) -> BM25Index:
        return cls(ids, [finance_tokenize(d) for d in documents])

    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        scores = self._model.get_scores(finance_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "tokenizer_version": self.tokenizer_version,
                    "ids": self.ids,
                    "model": self._model,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - our own artifact
        obj = cls.__new__(cls)
        obj.ids = payload["ids"]
        obj._model = payload["model"]
        obj.tokenizer_version = payload["tokenizer_version"]
        return obj

    def __len__(self) -> int:
        return len(self.ids)
