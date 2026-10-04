"""Cross-encoder reranking: reorder what retrieval already found.

Retrieval compares two vectors made independently - the query was embedded
without ever seeing the record, and the record without ever seeing the query.
A cross-encoder reads the pair together, which is why it can order better and
why it costs more: one model call per candidate per query rather than one per
query.

It is the largest measured quality win in this project, at rerank depth 20 on
the 4,000-record judged corpus:

    arm                                      nDCG@10   recall@10   controls
    retrieval as it ships                     0.6582      0.4196      1.000
    + ms-marco-MiniLM over enriched records   0.7149      0.4780      0.877
      … + exact-match promotion               0.7280      0.4780      1.000

TWO THINGS THAT ARE NOT OPTIONAL

**It needs prose, not a catalogue record.** Measured on records as they ship -
a title, an author, a semicolon list of subject headings, 24 words median - a
cross-encoder is worth +0.005 and -0.001. That is not a weak model; it is a
model trained to read passages being handed something with nothing to read.
On the enriched text, 95 words median, the same model is worth +0.073. So
`rerank` scores only records that *have* enriched text and leaves every other
one exactly where retrieval put it. A catalogue with no enrichment gets no
benefit and should not pay the latency.

**It breaks exact-name lookup, and something must put that back.** Both
enriched arms regressed the known-item controls - MiniLM to 0.877, `Sue Monk
Kidd` worst - because a model reasoning over prose optimises what a record is
*about* at the cost of which record it *is*. Finding a named book is the thing
library users do most. `bettersearch.exact` fixes it and must run **after**
this, never before: promotion lifts named records by rule, and reranking would
simply push them back down.

WHAT IT DOES NOT DO

It never changes the candidate set. The same records come back in a different
order, which is what makes the gain attributable to ordering rather than to
retrieval - and what lets the eval measure it at 100% judged coverage. A
`rerank` that returned different records would be a different retriever wearing
this module's name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Measured against `BAAI/bge-reranker-base`, which scores 0.7308 - 0.0028
#: better - for 2,809 ms/query against this one's 356 ms. Eight times the cost
#: for 4% of the gain is not a trade worth making on a request path.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: How many of the retrieved candidates to reorder. 20 because that is what the
#: page displays and what the eval measures; deeper was swept and decays
#: monotonically - +0.066 at 20 against +0.010 at 200, for seven times the
#: latency. See scripts/tune_reranking.py.
DEFAULT_DEPTH = 20


def load_enriched(path: str | Path, titles: dict[str, str]) -> dict[str, str]:
    """The records as prose, from the enrichment store, keyed by doc_id.

    The same shape `Enrichment.embed_text` builds for indexing. Reranking reads
    it at query time rather than embedding it, so adding reranking to a service
    needs no re-ingest and no index change.

    Shared with `scripts/tune_reranking.py` deliberately. If the measurement
    scored one rendering of a record and the server reranked another, the number
    in the README would describe something nobody runs.
    """
    store = Path(path)
    if not store.exists():
        return {}
    out: dict[str, str] = {}
    for line in store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        item_id = entry["item_id"]
        out[item_id] = "\n".join(filter(None, [
            titles.get(item_id, ""),
            " ".join(entry.get("questions") or []),
            entry.get("synopsis") or "",
            "Topics: " + ", ".join(entry.get("topics") or []),
            "Related: " + ", ".join(entry.get("entities") or []),
        ]))
    return out


class Reranker:
    """A cross-encoder, loaded once on first use.

    Lazy because `api/catalogue.py` is imported by tests that have no model and
    no network. Constructing this at import time would make the whole test suite
    depend on a 90 MB download, which is the property the CI workflow exists to
    protect.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, max_length: int = 512) -> None:
        self.model_name = model_name
        self._max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self._max_length)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One score per text. Higher is more relevant to `query`."""
        if not texts:
            return []
        model = self._load()
        return [float(s) for s in model.predict([(query, t) for t in texts],
                                                show_progress_bar=False)]


def rerank(query: str, ranked: list[tuple[dict, float]], texts: dict[str, str],
           reranker: Reranker, *, depth: int = DEFAULT_DEPTH) -> list[tuple[dict, float]]:
    """Reorder the first `depth` candidates that have prose to score.

    `ranked` is `[(record, score), ...]` as `run_search` builds it. The return
    value holds exactly the same records - this is a permutation, never a
    substitution.

    Records without enriched text keep their retrieval position rather than
    being scored on a 24-word stub, which was measured at nothing. That means a
    mixed catalogue - enriched books beside un-enriched events - reranks the
    part that benefits and leaves the rest alone, instead of being all-or-
    nothing about it.

    The returned score is the retrieval score, unchanged. A cross-encoder's
    output is an uncalibrated logit on a different scale, and letting it reach
    the response would silently break anything reading `score` as a cosine -
    the card's explanation, the confidence break, every threshold downstream.
    Order is the only thing reranking is allowed to change.
    """
    head = ranked[:depth]
    tail = ranked[depth:]

    scorable = [(i, r) for i, (r, _) in enumerate(head) if texts.get(r["doc_id"])]
    if len(scorable) < 2:
        # Nothing to reorder - one candidate, or a catalogue with no enrichment.
        return list(ranked)

    scores = reranker.score(query, [texts[r["doc_id"]] for _, r in scorable])
    order = [i for i, _ in sorted(zip([i for i, _ in scorable], scores),
                                  key=lambda pair: -pair[1])]

    # Reordered records go back into the positions reranking was allowed to
    # touch; un-enriched candidates keep the slots they already held, so the
    # result is still the same records in the same count.
    slots = iter(order)
    out = [head[next(slots)] if texts.get(r["doc_id"]) else head[i]
           for i, (r, _) in enumerate(head)]
    return out + tail
