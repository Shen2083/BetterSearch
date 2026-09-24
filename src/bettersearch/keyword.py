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

#: How much more a term in the title counts than the same term in the body.
#:
#: Without this a record is one flat bag of words, so "Harry" in an author's
#: name scores exactly as much as "Harry" in a title - which is how a search
#: for `Harry Potter` returned *Woodworking* by Harry Zarchy above anything
#: about wizards. Every library catalogue weights these differently.
#:
#: This is the BM25F idea in its simplest form: term frequency becomes a
#: weighted sum over fields, and the document length is weighted to match so
#: the length normalisation stays honest. Only the title is weighted, and
#: deliberately: author and subject headings live inside the body text as
#: labelled lines, and parsing those here would tie this module to one
#: catalogue's text format. Every document has a title.
#:
#: **Measured, and left at 1.0 - which is no weighting at all.**
#:
#: The sweep in scripts/tune_title_weight.py says this does not earn its place
#: on this corpus:
#:
#:   * the five exact-name lookups it was meant to help already score a perfect
#:     nDCG@10 of 1.000 with no weighting, so there is nothing to win;
#:   * nDCG@10 over the judged set falls from 0.443 to 0.403 at weight 3,
#:     though pool coverage falls with it (100% -> 94%), so part of that drop is
#:     the lane finding records no judge ever saw rather than real harm;
#:   * and on the query that prompted it, `Harry Potter`, it trades one kind of
#:     noise for another - Woodworking and Fool Moon leave, a book about a
#:     potter's wheel and a Beatrix Potter knitting guide arrive.
#:
#: The knob and the sweep stay because the next person will have the same idea,
#: and re-deriving the answer costs more than reading it. Raise it only with a
#: re-pooled eval: the keyword lane built the pool, so changing what it
#: retrieves makes its own new finds unjudged and therefore unscored.
TITLE_WEIGHT = 1.0

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

    def __init__(self, chunks: Sequence[Chunk], *,
                 title_weight: float = TITLE_WEIGHT) -> None:
        self._chunks = list(chunks)

        self._frequencies: list[Counter[str]] = []
        self._lengths: list[float] = []
        vocabularies: list[set[str]] = []
        for chunk in self._chunks:
            title = tokenize(chunk.title)
            body = tokenize(chunk.text)
            counts: Counter[str] = Counter()
            for term in title:
                counts[term] += title_weight
            for term in body:
                counts[term] += 1
            self._frequencies.append(counts)
            # The length has to be weighted the same way, or a weighted title
            # inflates the numerator while the denominator still thinks the
            # document is short, and every titled record drifts upward.
            self._lengths.append(len(title) * title_weight + len(body))
            vocabularies.append(set(counts))

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for vocabulary in vocabularies:
            document_frequency.update(vocabulary)

        total = len(self._chunks)
        # Standard BM25 IDF with the +1 that keeps it non-negative.
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, top_k: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []

        scores = [0.0] * len(self._chunks)
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
