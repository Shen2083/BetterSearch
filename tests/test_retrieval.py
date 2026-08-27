from __future__ import annotations

import pytest

from bettersearch.chunking import chunk_documents
from bettersearch.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank
from bettersearch.fusion import reciprocal_rank_fusion
from bettersearch.ingest import ingest_documents
from bettersearch.keyword import BM25Index, tokenize
from bettersearch.search import MODES, Searcher
from bettersearch.types import Chunk, ScoredChunk


def _chunk(chunk_id: str, doc_id: str, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        ordinal=0,
        text=text,
        embed_text=f"{title}\n\n{text}",
        content_hash=chunk_id,
    )


def _scored(ids: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=_chunk(i, i, i, i), score=1.0, rank=rank)
        for rank, i in enumerate(ids, start=1)
    ]


# ------------------------------------------------------------------- keyword
def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The ship that SANK!") == ["ship", "sank"]


def test_bm25_ranks_term_overlap_first():
    chunks = [
        _chunk("a", "a", "Yeast", "A fungus that releases carbon dioxide in dough."),
        _chunk("b", "b", "Hastings", "An invading army met the English shield wall."),
    ]
    results = BM25Index(chunks).search("carbon dioxide dough", top_k=2)
    assert results[0].chunk.chunk_id == "a"


def test_bm25_returns_nothing_without_shared_terms():
    """The failure mode the whole project is about."""
    chunks = [_chunk("a", "a", "Hastings", "An invading army met the shield wall.")]
    assert BM25Index(chunks).search("carbon dioxide") == []


def test_bm25_on_an_empty_corpus_is_safe():
    assert BM25Index([]).search("anything") == []


def test_bm25_ignores_a_pure_stopword_query():
    chunks = [_chunk("a", "a", "T", "Some body text here.")]
    assert BM25Index(chunks).search("the and of") == []


# -------------------------------------------------------------------- fusion
def test_rrf_rewards_appearing_in_both_lists():
    """A document both retrievers find beats one only a single retriever finds.

    This is the property that makes hybrid worth having: it promotes documents
    that keyword and semantic search agree on.
    """
    # "b" is second in both lists; "a" and "c" each appear in one list only.
    fused = reciprocal_rank_fusion([_scored(["a", "b"]), _scored(["c", "b"])])
    assert fused[0].chunk.chunk_id == "b"


def test_rrf_prefers_topping_one_list_over_being_middling_in_both():
    """A real and slightly counter-intuitive property of the formula.

    Because 1/x is convex, 1/(k+1) + 1/(k+3) is always greater than 2/(k+2) -
    so ranks {1st, 3rd} outscore {2nd, 2nd} at every value of k. Worth pinning
    down: it means RRF does not simply reward consensus, and a strong single
    signal is never buried by a pair of mediocre ones.
    """
    fused = reciprocal_rank_fusion([_scored(["a", "b", "c"]), _scored(["c", "b", "a"])])
    assert [f.chunk.chunk_id for f in fused][:2] == ["a", "c"]
    assert fused[2].chunk.chunk_id == "b"


def test_rrf_preserves_order_for_a_single_list():
    fused = reciprocal_rank_fusion([_scored(["a", "b", "c"])])
    assert [f.chunk.chunk_id for f in fused] == ["a", "b", "c"]


def test_rrf_deduplicates_and_reranks():
    fused = reciprocal_rank_fusion([_scored(["a", "b"]), _scored(["a", "c"])])
    assert [f.chunk.chunk_id for f in fused] == ["a", "b", "c"]
    assert [f.rank for f in fused] == [1, 2, 3]


def test_rrf_respects_top_k():
    fused = reciprocal_rank_fusion([_scored(["a", "b", "c", "d"])], top_k=2)
    assert len(fused) == 2


def test_rrf_scores_match_the_formula():
    fused = reciprocal_rank_fusion([_scored(["a"]), _scored(["a"])], k=60)
    assert fused[0].score == pytest.approx(2 / 61)


# ------------------------------------------------------------------- metrics
def test_recall_counts_relevant_documents_in_the_cut():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 5) == pytest.approx(0.5)
    assert recall_at_k(["x"], ["a"], 5) == 0.0
    assert recall_at_k(["a", "b"], [], 5) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "a"], ["a"], 10) == pytest.approx(0.5)
    assert reciprocal_rank(["a"], ["a"], 10) == pytest.approx(1.0)
    assert reciprocal_rank(["x"], ["a"], 10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 10) == pytest.approx(1.0)


def test_ndcg_penalises_a_lower_rank():
    good = ndcg_at_k(["a", "x", "y"], ["a"], 10)
    worse = ndcg_at_k(["x", "y", "a"], ["a"], 10)
    assert good > worse


# ------------------------------------------------------------------ searcher
def test_searcher_runs_every_mode(index, provider, documents, settings):
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    searcher = Searcher(provider=provider, index=index, settings=settings)

    for mode in MODES:
        response = searcher.search("carbon dioxide dough", mode=mode, top_k=3)
        assert response.mode == mode
        assert len(response.results) <= 3


def test_keyword_mode_never_loads_the_embedding_model(index, provider, documents, settings):
    """A keyword search must not pay the cost of loading a model."""
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    searcher = Searcher(index=index, settings=settings)  # no provider supplied

    results = searcher.keyword("carbon dioxide", top_k=3)
    assert results  # worked without ever touching Searcher.provider


def test_compare_returns_all_modes(index, provider, documents, settings):
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    searcher = Searcher(provider=provider, index=index, settings=settings)
    assert set(searcher.compare("dough", top_k=2)) == set(MODES)


def test_unknown_mode_is_rejected(index, provider, documents, settings):
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    searcher = Searcher(provider=provider, index=index, settings=settings)
    with pytest.raises(ValueError):
        searcher.search("q", mode="magic")


def test_keyword_and_semantic_search_the_same_corpus(index, provider, documents, settings):
    """Otherwise the comparison measures corpus differences, not retrieval."""
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    searcher = Searcher(provider=provider, index=index, settings=settings)

    indexed = {c.chunk_id for c in index.all_chunks()}
    assert indexed == {c.chunk_id for c in chunk_documents(documents)}
    assert len(searcher._keyword_index()) == len(indexed)
