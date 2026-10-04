"""Tests for cross-encoder reranking.

No model is loaded here. `Reranker.score` is the only thing that touches
`sentence_transformers`, so a subclass overriding it exercises every decision
this module makes - which is also the point: the suite must keep needing no
model, no network and no API key.

What is actually being guarded is that reranking **reorders and nothing else**.
The eval attributes +0.073 nDCG to ordering alone, and that attribution is only
true while the candidate set is untouched. A rerank that quietly dropped or
added a record would still look like an improvement and would be measuring
something else entirely.
"""

from __future__ import annotations

import json

import pytest

from bettersearch.rerank import Reranker, load_enriched, rerank


class FakeReranker(Reranker):
    """Scores from a lookup table instead of a cross-encoder."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [self.table.get(t, 0.0) for t in texts]


def ranked_of(*doc_ids):
    """`[(record, score), ...]` as run_search builds it, descending."""
    return [({"doc_id": d, "title": d.upper()}, 1.0 - i / 100)
            for i, d in enumerate(doc_ids)]


# ------------------------------------------------- reorder, and nothing else

def test_reranking_reorders_by_score():
    ranked = ranked_of("a", "b", "c")
    texts = {"a": "ta", "b": "tb", "c": "tc"}
    out = rerank("q", ranked, texts, FakeReranker({"ta": 0.1, "tb": 0.9, "tc": 0.5}))
    assert [r["doc_id"] for r, _ in out] == ["b", "c", "a"]


def test_the_candidate_set_is_never_changed():
    """The invariant the measured gain depends on."""
    ranked = ranked_of(*"abcdefgh")
    texts = {d: f"t{d}" for d in "abcdefgh"}
    scores = {f"t{d}": i for i, d in enumerate("hgfedcba")}
    out = rerank("q", ranked, texts, FakeReranker(scores))
    assert len(out) == len(ranked)
    assert sorted(r["doc_id"] for r, _ in out) == sorted(r["doc_id"] for r, _ in ranked)


def test_each_record_keeps_its_own_retrieval_score():
    """`score` is a cosine and must stay one.

    A cross-encoder emits an uncalibrated logit on a different scale. Letting it
    reach the response would redefine a published field and break everything
    reading it as a similarity.
    """
    ranked = ranked_of("a", "b")
    texts = {"a": "ta", "b": "tb"}
    out = rerank("q", ranked, texts, FakeReranker({"ta": 0.0, "tb": 1.0}))
    by_id = {r["doc_id"]: s for r, s in out}
    assert by_id == {"a": 1.0, "b": 0.99}


# ----------------------------------------------- records with nothing to read

def test_a_record_with_no_prose_keeps_its_retrieval_position():
    """Reranking a 24-word stub is measured at nothing, so it is not attempted.

    An events collection beside an enriched book collection is the real case:
    the part that benefits is reordered and the rest is left exactly alone,
    rather than the whole search being all-or-nothing about it.
    """
    ranked = ranked_of("book1", "event", "book2")
    texts = {"book1": "prose one", "book2": "prose two"}
    out = rerank("q", ranked, texts, FakeReranker({"prose one": 0.1, "prose two": 0.9}))
    ids = [r["doc_id"] for r, _ in out]
    assert ids == ["book2", "event", "book1"]
    assert ids[1] == "event", "an un-enriched record must not move"


def test_nothing_enriched_means_nothing_reranked():
    ranked = ranked_of("a", "b", "c")
    fake = FakeReranker({})
    out = rerank("q", ranked, {}, fake)
    assert [r["doc_id"] for r, _ in out] == ["a", "b", "c"]
    assert fake.calls == 0, "the model must not be called with nothing to score"


def test_a_single_scorable_candidate_is_not_worth_a_model_call():
    ranked = ranked_of("a", "b")
    fake = FakeReranker({"ta": 1.0})
    rerank("q", ranked, {"a": "ta"}, fake)
    assert fake.calls == 0


# -------------------------------------------------------------------- depth

def test_only_the_top_depth_is_reordered():
    """Beyond `depth` the list is untouched - that is what bounds the cost."""
    ranked = ranked_of(*"abcdef")
    texts = {d: f"t{d}" for d in "abcdef"}
    out = rerank("q", ranked, texts,
                 FakeReranker({f"t{d}": i for i, d in enumerate("fedcba")}), depth=3)
    ids = [r["doc_id"] for r, _ in out]
    assert ids[3:] == ["d", "e", "f"], "the tail must not move"
    assert sorted(ids[:3]) == ["a", "b", "c"], "the head must stay the head"


def test_depth_beyond_the_candidate_count_is_harmless():
    ranked = ranked_of("a", "b")
    out = rerank("q", ranked, {"a": "ta", "b": "tb"},
                 FakeReranker({"ta": 0.0, "tb": 1.0}), depth=500)
    assert [r["doc_id"] for r, _ in out] == ["b", "a"]


def test_an_empty_candidate_list_is_not_an_error():
    assert rerank("q", [], {}, FakeReranker({})) == []


# ------------------------------------------------------------ the prose store

def test_enriched_text_is_built_from_the_stores_fields(tmp_path):
    store = tmp_path / "e.jsonl"
    store.write_text(json.dumps({
        "item_id": "b1", "synopsis": "A book about bees.",
        "questions": ["how do I keep bees?"], "topics": ["beekeeping"],
        "entities": ["Langstroth"],
    }) + "\n", encoding="utf-8")
    texts = load_enriched(store, {"b1": "Beekeeping for All"})
    assert "Beekeeping for All" in texts["b1"]
    assert "how do I keep bees?" in texts["b1"]
    assert "A book about bees." in texts["b1"]
    assert "Topics: beekeeping" in texts["b1"]


def test_a_missing_store_is_empty_rather_than_an_error(tmp_path):
    """A catalogue nobody has enriched simply does not get this feature."""
    assert load_enriched(tmp_path / "nope.jsonl", {}) == {}


def test_blank_lines_in_the_store_are_skipped(tmp_path):
    store = tmp_path / "e.jsonl"
    store.write_text('\n{"item_id": "b1", "synopsis": "x"}\n\n', encoding="utf-8")
    assert set(load_enriched(store, {})) == {"b1"}
