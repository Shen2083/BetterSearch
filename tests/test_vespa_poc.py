"""Tests for the Vespa proof of concept's pure functions.

There is no Vespa in CI, so nothing here talks to a node. What is worth pinning
is the part that would silently produce a wrong measurement rather than an
error: the feed's faithfulness check, the year coercion, and the query bodies.
A lane file scored against a query string that does not match the eval set is
reported as a missing query, not a failure, so the query form is load-bearing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "vespa" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"vespa_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_feed = _load("make_feed")
run_eval = _load("run_eval")
probe_significance = _load("probe_significance")
compare_approximation = _load("compare_approximation")


RECORD = {
    "doc_id": "ol-OL1W", "title": "Psychology", "text": "Author: Carole Wade",
    "tags": ["large print"], "author": "Carole Wade", "author_dates": "",
    "year": "1987", "publisher": "Prentice-Hall", "format": "Large print",
    "language": "English", "fiction": True, "subjects": ["Textbooks", "Psychology"],
    "location": "Ashcombe Branch", "available": 0, "copies": 1,
    "blurb": None, "cover": "#1F3A3D",
}


class TestYearInt:
    def test_a_plain_year_survives(self):
        assert make_feed.year_int("1987") == 1987

    def test_an_empty_year_is_zero_not_an_error(self):
        # The corpus stores year as a string and some are blank. A feed that
        # raised here would be unable to carry the record at all.
        assert make_feed.year_int("") == 0

    def test_a_year_with_decoration_keeps_only_the_digits(self):
        assert make_feed.year_int("c. 1842?") == 1842

    def test_a_long_run_of_digits_is_truncated_to_four(self):
        assert make_feed.year_int("19871987") == 1987


class TestFeedDocument:
    def test_the_document_id_carries_the_doc_id(self):
        doc = make_feed.feed_document(RECORD, np.zeros(3, dtype=np.float32))
        assert doc["put"] == "id:northfield:record::ol-OL1W"

    def test_embed_text_is_title_blank_line_text(self):
        # This must match src/bettersearch/chunking.py:125 exactly, because the
        # ColBERT field is derived from it inside Vespa.
        doc = make_feed.feed_document(RECORD, np.zeros(3, dtype=np.float32))
        assert doc["fields"]["embed_text"] == "Psychology\n\nAuthor: Carole Wade"

    def test_the_vector_becomes_plain_floats(self):
        doc = make_feed.feed_document(RECORD, np.array([0.5, -0.25], dtype=np.float32))
        assert doc["fields"]["embedding"] == {"values": [0.5, -0.25]}
        assert all(isinstance(v, float) for v in doc["fields"]["embedding"]["values"])

    def test_multi_valued_fields_stay_lists(self):
        doc = make_feed.feed_document(RECORD, np.zeros(1, dtype=np.float32))
        assert doc["fields"]["subjects"] == ["Textbooks", "Psychology"]
        assert doc["fields"]["tags"] == ["large print"]

    def test_blurb_is_not_fed_because_it_is_null_on_every_book(self):
        doc = make_feed.feed_document(RECORD, np.zeros(1, dtype=np.float32))
        assert "blurb" not in doc["fields"]


class TestBuildBody:
    def test_keyword_mode_asks_for_any_not_weak_and(self):
        # Our BM25 scores every record containing any query term. Vespa
        # defaults to weakAnd, an OR with early termination, and that
        # approximation would show up as a scoring difference that has nothing
        # to do with BM25.
        body = run_eval.build_body("bm25", "bees", 20, "bm25_flat", None, {})
        assert body["type"] == "any"
        assert body["query"] == "bees"

    def test_dense_mode_sends_no_query_string(self):
        body = run_eval.build_body("dense", "bees", 20, "dense", [0.1, 0.2], {})
        assert "query" not in body
        assert body["input.query(q)"] == {"values": [0.1, 0.2]}

    def test_dense_target_hits_follows_the_requested_depth(self):
        body = run_eval.build_body("dense", "bees", 75, "dense", [0.1], {})
        assert "{targetHits:75}" in body["yql"]

    def test_colbert_mode_lets_vespa_embed_the_query(self):
        body = run_eval.build_body("colbert", "bees", 20, "colbert", [0.1], {})
        assert body["input.query(qt)"] == "embed(colbert, @qtext)"
        assert body["qtext"] == "bees"

    def test_dense_native_sends_no_vector_at_all(self):
        # The whole point of the lane: Vespa tokenises and embeds the query
        # itself. A vector in the request would mean our encoder was still
        # involved and the measurement would be of nothing in particular.
        body = run_eval.build_body("dense_native", "bees", 20, "dense_native",
                                   None, {})
        assert body["input.query(qn)"] == "embed(bge, @qtext)"
        assert body["qtext"] == "bees"
        assert "input.query(q)" not in body
        assert "embedding_native" in body["yql"]

    def test_binary_packs_the_query_the_way_vespa_packs_documents(self):
        # Most significant bit first, eight dimensions per byte, signed. A
        # wrong bit order does not error, it silently returns nonsense.
        vector = [1.0] * 8 + [-1.0] * 8 + [0.0] * 8
        body = run_eval.build_body("binary", "bees", 20, "binary", vector, {})
        packed = body["input.query(qb)"]["values"]
        assert len(packed) == 3
        assert packed[0] == -1   # 0b11111111 as int8
        assert packed[1] == 0    # eight negatives
        assert packed[2] == 0    # zeros are not > 0

    def test_hnsw_mode_forces_the_approximate_path(self):
        # Without approximate:true Vespa may quietly run an exact scan, and the
        # lane would read as perfect recall while measuring nothing.
        body = run_eval.build_body("dense_hnsw", "bees", 100, "dense_hnsw",
                                   [0.1], {})
        assert "approximate:true" in body["yql"]
        assert "embedding_hnsw" in body["yql"]

    def test_explore_additional_hits_lands_in_the_annotation(self):
        # It is an annotation on the operator. Sent as a query property it is
        # accepted and ignored, and the sweep would show no sensitivity to a
        # parameter that never arrived.
        body = run_eval.build_body("dense_hnsw", "bees", 100, "dense_hnsw",
                                   [0.1], {"hnsw.exploreAdditionalHits": 500})
        assert "hnsw.exploreAdditionalHits:500" in body["yql"]
        assert "hnsw.exploreAdditionalHits" not in body

    def test_build_body_consumes_the_annotation_from_the_dict_it_is_given(self):
        # build_body pops the annotation out of `extra`. A caller that reuses
        # one dict across a loop therefore gets the parameter on the first
        # query and not on the rest - and a sweep would read as insensitive to
        # a parameter that only ever arrived once. compare_approximation.py
        # passes a copy for this reason; this pins the behaviour it copies for.
        shared = {"hnsw.exploreAdditionalHits": 500}
        run_eval.build_body("dense_hnsw", "a", 100, "p", [0.1], shared)
        assert shared == {}, "build_body must consume it, or the comment is wrong"

        extra = {"hnsw.exploreAdditionalHits": 500}
        bodies = [run_eval.build_body("dense_hnsw", q, 100, "p", [0.1], dict(extra))
                  for q in ("a", "b", "c")]
        assert all("hnsw.exploreAdditionalHits:500" in b["yql"] for b in bodies)

    def test_extra_properties_are_passed_through(self):
        body = run_eval.build_body("bm25", "bees", 20, "bm25_fielded", None,
                                   {"model.defaultIndex": "fielded"})
        assert body["model.defaultIndex"] == "fielded"

    def test_an_unknown_mode_fails_loudly(self):
        with pytest.raises(SystemExit):
            run_eval.build_body("nonsense", "bees", 20, "p", None, {})


class TestDocIds:
    def test_hits_without_a_doc_id_are_skipped_rather_than_crashing(self):
        result = {"root": {"children": [
            {"fields": {"doc_id": "a"}}, {"fields": {}}, {"fields": {"doc_id": "b"}}]}}
        assert run_eval.doc_ids(result) == ["a", "b"]

    def test_no_children_gives_an_empty_ranking(self):
        assert run_eval.doc_ids({"root": {"fields": {"totalCount": 0}}}) == []


class TestAnnotatedYql:
    def test_each_term_carries_its_own_significance(self):
        yql = probe_significance.annotated_yql("grow vegetables", [0.6, 0.7])
        assert '{significance:0.600000}"grow"' in yql
        assert '{significance:0.700000}"vegetables"' in yql
        assert " or " in yql

    def test_extra_words_without_a_significance_are_dropped(self):
        # Vespa reports significance per term index; if it reported fewer than
        # the query has words, pairing by position must not invent one.
        yql = probe_significance.annotated_yql("a b c", [0.5, 0.5])
        assert yql.count("significance") == 2

    def test_quotes_in_a_query_cannot_break_out_of_the_yql(self):
        yql = probe_significance.annotated_yql('say "hi"', [0.5, 0.5])
        assert yql.count('"') == 4


class TestApproximationRecall:
    def test_a_perfect_approximation_recalls_everything(self):
        exact = ["a", "b", "c"]
        assert compare_approximation.recall_at(exact, ["a", "b", "c"], 3) == 1.0

    def test_recall_counts_set_membership_not_order(self):
        # A reordering inside the top k is not a recall failure; it is a
        # ranking difference, and first_divergence is what reports it.
        assert compare_approximation.recall_at(["a", "b"], ["b", "a"], 2) == 1.0

    def test_a_missed_document_lowers_recall(self):
        assert compare_approximation.recall_at(["a", "b"], ["a", "z"], 2) == 0.5

    def test_recall_of_an_empty_exact_list_is_one_not_a_crash(self):
        # Division by zero here would take out the whole sweep for one query
        # that legitimately matched nothing.
        assert compare_approximation.recall_at([], ["a"], 10) == 1.0

    def test_recall_only_looks_at_the_top_k_of_both(self):
        exact = ["a", "b", "c", "d"]
        approx = ["a", "b", "z", "y"]
        assert compare_approximation.recall_at(exact, approx, 2) == 1.0
        assert compare_approximation.recall_at(exact, approx, 4) == 0.5


class TestFirstDivergence:
    def test_identical_lists_never_diverge(self):
        assert compare_approximation.first_divergence(["a", "b"], ["a", "b"]) is None

    def test_divergence_is_one_based(self):
        assert compare_approximation.first_divergence(["a", "b"], ["z", "b"]) == 1
        assert compare_approximation.first_divergence(["a", "b"], ["a", "z"]) == 2

    def test_a_short_list_diverges_where_it_runs_out(self):
        assert compare_approximation.first_divergence(["a", "b", "c"], ["a", "b"]) == 3

    def test_two_empty_lists_are_identical(self):
        assert compare_approximation.first_divergence([], []) is None
