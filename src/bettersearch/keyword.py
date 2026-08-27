"""BM25 keyword baseline - the thing semantic search has to beat.

Hand-rolled rather than pulling in ``rank_bm25``: it is about forty lines of
well-specified arithmetic, and the comparison is more credible when the baseline
is visible and auditable rather than a black box.

BM25 ranks by term overlap. That is precisely why it cannot answer "Norman
conquest" with a document that only ever says "1066" and "Hastings" - there is
no shared term to score. Keeping an honest, properly-tuned baseline matters: a
deliberately weak one would make the semantic results look better than they are.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .types import Chunk, ScoredChunk

# Standard BM25 parameters.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")

# A small stopword list. Without it, "the ship that sank on its maiden voyage"
# scores mostly on "the" and "on", which would misrepresent BM25's real ability.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can did do does for from had has have he
    her his how i if in into is it its of on or our that the their them then
    there these they this to was were what when where which who why will with
    you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """In-memory BM25 over a fixed set of chunks."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._documents = [tokenize(f"{c.title} {c.text}") for c in self._chunks]
        self._lengths = [len(d) for d in self._documents]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._frequencies = [Counter(d) for d in self._documents]

        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            document_frequency.update(set(document))

        total = len(self._documents)
        # Standard BM25 IDF with the +1 that keeps it non-negative.
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, top_k: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._documents:
            return []

        scores = [0.0] * len(self._documents)
        for index, frequencies in enumerate(self._frequencies):
            length = self._lengths[index]
            if length == 0:
                continue
            norm = K1 * (1 - B + B * length / (self._average_length or 1.0))
            total = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                total += self._idf.get(term, 0.0) * (
                    frequency * (K1 + 1) / (frequency + norm)
                )
            scores[index] = total

        ranked = sorted(
            (i for i, s in enumerate(scores) if s > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            ScoredChunk(chunk=self._chunks[i], score=scores[i], rank=rank)
            for rank, i in enumerate(ranked, start=1)
        ]
