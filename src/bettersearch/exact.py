"""Exact matches: the one signal neither retrieval lane can see.

BM25 scores a record as a bag of words, so `secret life bees` and
`bees secret life` are the same query to it. An embedding blurs proper nouns
into their neighbourhood, which is why searching an author's name returns
books *like* that author rather than that author. **Word order is invisible to
both**, and a phrase appearing in sequence is therefore free information.

This module finds records a reader has arguably *named* rather than described.
It loads no model and touches no vectors: it is string work over the title and
author fields, which every catalogue has.

WHY THE THRESHOLDS ARE WHAT THEY ARE
------------------------------------
Measured against the 36 need-shaped queries in data/eval_real.json and the 5
known-item controls that sit alongside them:

    signal                      fires on needs    catches controls
    2-word phrase in a title         11/36              3/5
    3-word phrase in a title          1/36              1/5
    query equals an author name       0/36              3/5

A two-word rule fires on nearly a third of ordinary questions and is unusable.
Three words plus an author-field match is precise, and between them they cover
four of the five controls; whole-title equality picks up the fifth, `Joy of
Cooking`, which is two tokens after stopwords and so can never contain a
3-gram.

The one need query that trips the phrase rule is `second world war in the
pacific`, and a book whose title contains that exact phrase is a defensible
thing to put in front of that reader rather than a bug.

WHAT IT IS WORTH, MEASURED
--------------------------
Not a relevance improvement, and the page should not be sold as one.

    nDCG@10 over the 41 judged queries    0.658 -> 0.658
    right record at rank 1, 113 names   103/113 -> 104/113
    right record in the top 5           109/113 -> 110/113

Identical on needs at every setting, and one query better on names. The gain
is inside the noise. bge-base already answers name lookups well, which is the
same finding that killed the router, and promotion has little left to add.

It earns its place for three other reasons. It **labels** - a card saying "by
this author" is legible in a way a cosine score never is. It makes a
**guarantee** rather than an improvement: a record whose exact title someone
typed will be on page one because a rule says so, not because the encoder
happened to agree. And it costs nothing to be wrong about, since the judged
score does not move.

WHAT THIS IS NOT
----------------
It is not fusion and it is not routing, both of which were measured here and
rejected. Reciprocal rank fusion scored 0.741 against semantic's 0.890, because
blending two full rankings drags a good list down with a mostly-empty one. A
router is pointless because meaning-based search wins *both* query types
(103/113 against 99/113 on name lookups, 0.658 against 0.443 on needs). This
leaves the meaning ranking as the spine and promotes a handful of named records
into it, labelled so a reader knows why they are there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .keyword import tokenize

#: Shortest run of words that counts as a phrase. Two fires on a third of
#: ordinary questions; see the table above. Three and four score identically,
#: so three is kept for the extra reach at no measured cost.
PHRASE_TOKENS = 3

#: How many named records may be promoted above the meaning ranking.
#:
#: **Chosen by judgement, not by measurement, and the sweep says so.** Caps of
#: 1, 2, 3 and 5 produce identical numbers on every metric, because promotion
#: rarely changes the top of a list that bge-base had already ordered well.
#: Three is a display decision: a reader who types an author's name is better
#: served by three of their books than by one.
MAX_PROMOTED = 3


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """A record the reader arguably named, and which of the rules said so."""

    doc_id: str
    #: "title", "phrase" or "author" - shown to the reader, so it has to be
    #: something true and sayable rather than an internal code.
    reason: str

    def describe(self) -> str:
        return {
            "title": "this exact title",
            "phrase": "contains your exact phrase",
            "author": "by this author",
        }[self.reason]


def _runs(terms: list[str], length: int) -> set[str]:
    """Every consecutive run of `length` terms, order preserved.

    The terms are already stopword-free, so "three words" here means three
    *content* words and is stricter than three typed ones: `the secret life of
    bees` is the run `secret life bees`. That is why it does not match *The
    Secret Life of Mermaids*, which shares only two content words in sequence.
    """
    return {" ".join(terms[i:i + length]) for i in range(len(terms) - length + 1)}


def find(query: str, records: Iterable[dict], *,
         phrase_tokens: int = PHRASE_TOKENS) -> list[ExactMatch]:
    """Records the query names, in the order title, author, phrase.

    The order matters: a whole-title match is a stronger claim than a phrase
    inside a longer title, and both are stronger than an author whose name the
    reader may only have half meant.
    """
    terms = tokenize(query)
    if not terms:
        return []

    asked = " ".join(terms)
    phrases = _runs(terms, phrase_tokens) if len(terms) >= phrase_tokens else set()

    titles: list[ExactMatch] = []
    authors: list[ExactMatch] = []
    contained: list[ExactMatch] = []
    for record in records:
        doc_id = record["doc_id"]
        title_terms = tokenize(record.get("title", ""))
        title = " ".join(title_terms)

        if title and title == asked:
            titles.append(ExactMatch(doc_id, "title"))
            continue

        author_terms = tokenize(record.get("author", ""))
        if author_terms:
            author = " ".join(author_terms)
            # The author's name as a run inside the query, so `books by sue
            # monk kidd` still matches, but `monk` alone does not.
            if author == asked or author in _runs(terms, len(author_terms)):
                authors.append(ExactMatch(doc_id, "author"))
                continue

        if phrases and title and any(p in title for p in phrases):
            contained.append(ExactMatch(doc_id, "phrase"))

    return titles + authors + contained


def promote(ranked: list, matches: list[ExactMatch],
            catalogue: dict[str, dict], *,
            limit: int = MAX_PROMOTED) -> tuple[list, dict[str, str]]:
    """Label every named record, and lift the first `limit` of them.

    Inserting is the point rather than a detail. Reordering alone could only
    shuffle what meaning-based search already found, and the case worth fixing
    is the record it *missed* - the reader typed a title, it is in the
    catalogue, and it is nowhere on page one. So a match absent from the
    ranking is put in rather than skipped.

    Nothing is dropped. A promoted record already present moves up instead of
    appearing twice, and everything else keeps its relative order, so a reader
    who would have found something at rank 7 still finds it, one place lower.

    `ranked` is a list of (record, score) as run_search builds it. Inserted
    rows carry a score of 0.0: the page never displays a score, and inventing a
    similarity for a record that was matched by its title rather than by a
    vector would be making a number up.

    Returns the reordered list and a doc_id -> reason map covering **every**
    match, not only the ones that moved - see the comment below.
    """
    if not matches:
        return ranked, {}

    # **Labelling and lifting are different jobs, and the cap governs only the
    # second.** Merging them is the obvious mistake and it shipped once: with
    # one slice doing both, a fourth book by the author a reader had just named
    # sat unlabelled directly beneath three labelled ones, because the ranking
    # had already placed it and so it was never lifted. That distinction is
    # real in here and invisible, and meaningless, to the person reading the
    # page. Every match gets its reason; only `limit` of them move.
    reasons = {m.doc_id: m.describe() for m in matches if m.doc_id in catalogue}
    if not reasons:
        return ranked, {}

    # A prolific author or a common phrase can label many records, which is
    # correct - ten Christie novels on one page should all say so.
    moving = list(reasons)[:limit]
    existing = {row[0]["doc_id"]: row for row in ranked}
    lifted = [existing.get(doc_id) or (catalogue[doc_id], 0.0)
              for doc_id in moving]
    rest = [row for row in ranked if row[0]["doc_id"] not in set(moving)]
    return lifted + rest, reasons
