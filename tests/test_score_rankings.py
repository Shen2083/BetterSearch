"""Tests for the engine-independent scorer.

This script is the bar a Vespa rebuild will be held to, so the arithmetic has
to be right and the qualifications around it have to be honest. The cases below
are the ways a scorer can flatter a system: scoring fewer queries than it was
asked about, counting a repeated record twice, or reporting a confident number
over records no judge ever saw.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "score_rankings.py"
_spec = importlib.util.spec_from_file_location("score_rankings", SCRIPT)
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def q(query, relevant, graded=None, note=""):
    """One eval query in the shape data/eval_real.json uses."""
    return {
        "query": query,
        "relevant_doc_ids": list(relevant),
        # `grades` is every record a judge looked at, relevant or not.
        "grades": {d: 2 for d in (relevant if graded is None else graded)},
        "note": note,
    }


def test_a_perfect_ranking_scores_one():
    queries = [q("bees", ["a", "b"])]
    s = sr.score({"bees": ["a", "b", "z"]}, queries)
    assert s["ndcg@10"] == pytest.approx(1.0)
    assert s["recall@5"] == pytest.approx(1.0)
    assert s["mrr@10"] == pytest.approx(1.0)


def test_rank_order_matters():
    queries = [q("bees", ["a"])]
    first = sr.score({"bees": ["a", "x", "y"]}, queries)["ndcg@10"]
    third = sr.score({"bees": ["x", "y", "a"]}, queries)["ndcg@10"]
    assert first > third > 0


def test_a_missing_query_is_reported_rather_than_skipped():
    """Scoring only the queries you did well on is the easiest way to cheat."""
    queries = [q("bees", ["a"]), q("knitting", ["b"])]
    s = sr.score({"bees": ["a"]}, queries)
    assert s["missing"] == ["knitting"]
    assert s["queries"] == 1
    # and the mean is over what was scored, not padded with zeros that would
    # hide the gap behind a merely-bad number
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_a_repeated_doc_id_counts_once():
    """Engines that return one record per chunk must not be rewarded for it."""
    queries = [q("bees", ["a", "b"])]
    s = sr.score({"bees": ["a", "a", "a", "b"]}, queries)
    assert s["ndcg@10"] == pytest.approx(1.0)
    assert s["rows"][0]["returned"] == 2


def test_unknown_doc_ids_do_not_crash():
    """A first export from another engine will contain ids we have never seen."""
    queries = [q("bees", ["a"])]
    s = sr.score({"bees": ["unknown-1", "unknown-2", "a"]}, queries)
    assert 0 < s["ndcg@10"] < 1
    assert s["coverage"] == pytest.approx(1 / 3)


def test_coverage_counts_judged_records_not_relevant_ones():
    """Coverage must use `grades`, not `relevant_doc_ids`.

    If it used the relevant set, coverage and precision would be the same
    number and the check could never warn about anything.
    """
    # "b" was judged and found irrelevant; it still counts as covered.
    queries = [q("bees", ["a"], graded=["a", "b"])]
    s = sr.score({"bees": ["a", "b"]}, queries)
    assert s["coverage"] == pytest.approx(1.0)
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_coverage_falls_when_an_engine_finds_unpooled_records():
    """The pooling trap, as a number: a new lane's finds are unjudged."""
    queries = [q("bees", ["a"], graded=["a", "b", "c"])]
    s = sr.score({"bees": ["a", "new-1", "new-2", "new-3"]}, queries)
    assert s["coverage"] == pytest.approx(0.25)


def test_controls_are_read_from_the_note():
    queries = [
        q("Agatha Christie", ["a"], note="control (keyword should win) · Exact author"),
        q("bees", ["b"], note="natural-language subject · whatever"),
    ]
    s = sr.score({"Agatha Christie": ["a"], "bees": ["zzz"]}, queries)
    assert s["control_count"] == 1
    assert s["controls"] == pytest.approx(1.0)      # the control, not the mean


def test_an_event_on_a_known_item_lookup_is_flagged():
    queries = [q("Agatha Christie", ["a"])]
    s = sr.score({"Agatha Christie": ["a", "ev-bd9d439a9d"]}, queries)
    assert s["events_intruded"] == ["Agatha Christie"]


def test_an_event_elsewhere_is_not_flagged():
    queries = [q("somewhere to go on a Tuesday", ["ev-1"])]
    s = sr.score({"somewhere to go on a Tuesday": ["ev-1"]}, queries)
    assert s["events_intruded"] == []


def test_the_shipped_eval_set_loads_and_has_its_controls():
    """Guards the constants above against the data moving underneath them."""
    queries, meta = sr.load_eval(Path("data/eval_real.json"))
    assert len(queries) == 41
    controls = [x for x in queries if x.get("note", "").startswith(sr.CONTROL_PREFIX)]
    assert len(controls) == 5
    assert {x["query"] for x in controls} == set(sr.KNOWN_ITEM)
    assert meta["judge_model"] and meta["lanes_pooled"]


# ---------------------------------------------------------------- reranking
# tune_reranking.py's whole claim rests on one property: an arm may reorder the
# candidates but never substitute them. If that breaks, judged coverage moves
# and the arms stop being comparable to each other - which is the failure the
# rest of this project keeps running into from other directions.

_rr_spec = importlib.util.spec_from_file_location(
    "tune_reranking", SCRIPT.parent / "tune_reranking.py")
rr = importlib.util.module_from_spec(_rr_spec)
_rr_spec.loader.exec_module(rr)


def test_promotion_returns_a_permutation_not_a_new_list():
    catalogue = {
        "a": {"doc_id": "a", "title": "The Secret Life of Bees", "author": "Sue Monk Kidd"},
        "b": {"doc_id": "b", "title": "Something Else", "author": "Someone"},
        "c": {"doc_id": "c", "title": "Third Book", "author": "Another"},
    }
    c = {"query": "Sue Monk Kidd", "doc_ids": ["a", "b", "c"]}
    out = rr.promote_named(c, ["c", "b", "a"], catalogue)
    assert sorted(out) == ["a", "b", "c"], "promotion must not add or drop records"
    assert out[0] == "a", "the named author's record should be lifted"


def test_promotion_keeps_records_absent_from_the_catalogue():
    """Events and anything else outside the catalogue file must survive."""
    catalogue = {"a": {"doc_id": "a", "title": "T", "author": "A"}}
    c = {"query": "nothing matches this", "doc_ids": ["a", "ev-1"]}
    out = rr.promote_named(c, ["a", "ev-1"], catalogue)
    assert sorted(out) == ["a", "ev-1"]


def test_minmax_survives_a_flat_signal():
    """BM25 returns zero for every candidate when no term matches."""
    import numpy as np
    flat = rr.minmax(np.zeros(5, dtype=np.float32))
    assert flat.tolist() == [0.0] * 5


def test_measure_rejects_an_arm_that_changes_the_candidate_set():
    cands = [{"query": "q", "note": "", "relevant": ["a"], "graded": {"a", "b"},
              "doc_ids": ["a", "b"], "idx": [0, 1]}]
    with pytest.raises(SystemExit, match="changed the candidate set"):
        rr.measure(cands, lambda c: ["a", "substituted"], name="bad arm")
