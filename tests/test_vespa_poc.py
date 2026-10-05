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
