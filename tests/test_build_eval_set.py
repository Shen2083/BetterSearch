"""Tests for the two mechanisms that let the pool grow without re-paying for it.

Both are places where a mistake would be invisible. Judgement reuse attaches a
grade a judge gave for one query to a (query, record) pair; if it keys that
wrongly, the ground truth is corrupted silently and every number computed from
it afterwards is wrong while still looking plausible. A file lane is how an arm
that is not just an index and a mode - a reranked lane - gets into the pool at
all; if its candidates do not reach the pool, the arm is measured against
judgements that never saw what it found, which is the exact bias this project
has been fighting.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_eval_set.py"
_spec = importlib.util.spec_from_file_location("build_eval_set", SCRIPT)
bes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bes)


def write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def eval_file(tmp_path: Path, queries) -> Path:
    return write(tmp_path / "eval.json", {"queries": queries})


# ------------------------------------------------------------------ file lanes

def test_a_file_lane_puts_its_ranking_into_the_pool(tmp_path):
    lane = write(tmp_path / "lane.json", {"bees": ["a", "b", "c"]})
    built = bes.build_pools(
        [{"kind": "file", "label": "reranked", "path": str(lane)}],
        [{"query": "bees"}], per_lane=10)
    assert set(built["pools"]["bees"]) == {"a", "b", "c"}
    assert built["contributed"]["reranked"] == 3


def test_a_file_lane_is_cut_to_per_lane(tmp_path):
    """The dump is depth-long; the pool takes the top K, in rank order."""
    lane = write(tmp_path / "lane.json", {"bees": list("abcdefgh")})
    built = bes.build_pools(
        [{"kind": "file", "label": "reranked", "path": str(lane)}],
        [{"query": "bees"}], per_lane=3)
    assert set(built["pools"]["bees"]) == {"a", "b", "c"}


def test_a_file_lane_contributes_only_what_other_lanes_missed(tmp_path):
    """unique_to is what would be lost by leaving the lane out of the pool."""
    a = write(tmp_path / "a.json", {"bees": ["x", "y"]})
    b = write(tmp_path / "b.json", {"bees": ["y", "z"]})
    built = bes.build_pools(
        [{"kind": "file", "label": "a", "path": str(a)},
         {"kind": "file", "label": "b", "path": str(b)}],
        [{"query": "bees"}], per_lane=10)
    assert set(built["pools"]["bees"]) == {"x", "y", "z"}
    assert built["unique_to"] == {"a": 1, "b": 1}


def test_a_query_the_file_lane_cannot_answer_is_counted_not_hidden(tmp_path):
    """Silently skipping it would thin the pool for that query unannounced."""
    lane = write(tmp_path / "lane.json", {"bees": ["a"]})
    built = bes.build_pools(
        [{"kind": "file", "label": "reranked", "path": str(lane)}],
        [{"query": "bees"}, {"query": "wasps"}], per_lane=10)
    assert built["pools"]["wasps"] == []
    assert built["missing_from_file"]["reranked"] == 1


def test_a_repeated_doc_id_is_pooled_once(tmp_path):
    lane = write(tmp_path / "lane.json", {"bees": ["a", "a", "b"]})
    built = bes.build_pools(
        [{"kind": "file", "label": "reranked", "path": str(lane)}],
        [{"query": "bees"}], per_lane=10)
    assert sorted(built["pools"]["bees"]) == ["a", "b"]


# ---------------------------------------------------------------------- reuse

def test_reuse_is_keyed_on_query_text_not_query_index(tmp_path):
    """The failure this guards against.

    The judgements file keys grades `q{index}--{doc_id}`. Reorder the query list
    - which re-pooling invites, since lanes change - and every index shifts, so
    a grade given for one query would attach to another. Keying on the query's
    text cannot drift that way.
    """
    ev = eval_file(tmp_path, [
        {"query": "bees", "grades": {"a": 2}},
        {"query": "wasps", "grades": {"a": 0}},
    ])
    reuse = bes.load_reuse(ev, tmp_path / "absent.json")
    assert reuse[("bees", "a")]["grade"] == 2
    assert reuse[("wasps", "a")]["grade"] == 0

    # Same grades, queries swapped. Index-keyed reuse would now return 0 for
    # "bees"; text-keyed reuse is unmoved.
    swapped = eval_file(tmp_path, [
        {"query": "wasps", "grades": {"a": 0}},
        {"query": "bees", "grades": {"a": 2}},
    ])
    assert bes.load_reuse(swapped, tmp_path / "absent.json") == reuse


def test_reuse_carries_the_judges_reason_across(tmp_path):
    ev = eval_file(tmp_path, [{"query": "bees", "grades": {"a": 2}}])
    whys = write(tmp_path / "j.json",
                 {"judgements": {"q0--a": {"grade": 2, "why": "a book about bees"}}})
    reuse = bes.load_reuse(ev, whys)
    assert reuse[("bees", "a")] == {"grade": 2, "why": "a book about bees"}


def test_reuse_survives_a_missing_judgements_file(tmp_path):
    """Grades are the ground truth; the reasons are commentary on them."""
    ev = eval_file(tmp_path, [{"query": "bees", "grades": {"a": 1}}])
    reuse = bes.load_reuse(ev, tmp_path / "nope.json")
    assert reuse[("bees", "a")] == {"grade": 1, "why": ""}


def test_no_eval_file_means_no_reuse(tmp_path):
    """A first run must judge everything rather than quietly judging nothing."""
    assert bes.load_reuse(tmp_path / "nope.json", tmp_path / "nope2.json") == {}


def test_reuse_does_not_invent_grades_for_unjudged_records(tmp_path):
    """`grades` is what a judge saw. A record outside it must come back as new.

    Returning a default - 0, say - would be the pooling bias moved inside the
    tool: an unjudged record would be recorded as irrelevant without anyone
    having looked at it, and it would never be queued for judging.
    """
    ev = eval_file(tmp_path, [{"query": "bees", "grades": {"a": 2}}])
    reuse = bes.load_reuse(ev, tmp_path / "nope.json")
    assert ("bees", "b") not in reuse


def test_a_query_with_no_grades_yet_reuses_nothing(tmp_path):
    ev = eval_file(tmp_path, [{"query": "bees", "grades": {}},
                              {"query": "wasps"}])
    assert bes.load_reuse(ev, tmp_path / "nope.json") == {}


# --------------------------------------------------- the split that spends money

def corpus_of(*doc_ids):
    return {d: {"doc_id": d, "title": f"Title {d}", "text": f"text {d}"}
            for d in doc_ids}


def test_only_unseen_pairs_are_queued_for_judging():
    queries = [{"query": "bees"}]
    pools = {"bees": ["a", "b"]}
    reuse = {("bees", "a"): {"grade": 2, "why": "seen before"}}
    pairs, grades = bes.split_pairs(queries, pools, corpus_of("a", "b"), reuse)
    assert [k for k, _, _ in pairs] == ["q0--b"]
    assert grades == {"q0--a": {"grade": 2, "why": "seen before"}}


def test_the_same_record_under_two_queries_is_judged_per_query():
    """Relevance is a property of the pair, not of the record."""
    queries = [{"query": "bees"}, {"query": "wasps"}]
    pools = {"bees": ["a"], "wasps": ["a"]}
    reuse = {("bees", "a"): {"grade": 2, "why": ""}}
    pairs, grades = bes.split_pairs(queries, pools, corpus_of("a"), reuse)
    assert [k for k, _, _ in pairs] == ["q1--a"]
    assert list(grades) == ["q0--a"]


def test_keys_follow_this_runs_query_order():
    """`grades` is written back by the same index, so the two must agree."""
    queries = [{"query": "wasps"}, {"query": "bees"}]
    pools = {"bees": ["a"], "wasps": ["b"]}
    pairs, _ = bes.split_pairs(queries, pools, corpus_of("a", "b"), {})
    assert [(k, q) for k, q, _ in pairs] == [("q0--b", "wasps"), ("q1--a", "bees")]


def test_an_empty_reuse_judges_the_whole_pool():
    queries = [{"query": "bees"}]
    pools = {"bees": ["a", "b", "c"]}
    pairs, grades = bes.split_pairs(queries, pools, corpus_of("a", "b", "c"), {})
    assert len(pairs) == 3 and grades == {}


def test_the_judge_sees_title_and_text_and_not_the_lane():
    """Pool position is shuffled and the label discarded; nothing here may leak it."""
    pairs, _ = bes.split_pairs([{"query": "bees"}], {"bees": ["a"]},
                               corpus_of("a"), {})
    (key, query, text) = pairs[0]
    assert query == "bees"
    assert text == "Title a\ntext a"


# ------------------------------------------- the pool may grow, never shrink

def test_a_judged_record_no_current_lane_finds_stays_in_the_pool():
    """The quiet way re-pooling corrupts a comparison.

    The new pool is built from the lanes this run has. A record the old lanes
    retrieved and the new ones do not would drop out, shrinking the denominator
    every published figure was measured against - so arms would look better
    without anything having improved.
    """
    pools = {"bees": ["new"]}
    reuse = {("bees", "old"): {"grade": 2, "why": ""}}
    restored = bes.keep_judged(pools, [{"query": "bees"}], reuse)
    assert restored == 1
    assert set(pools["bees"]) == {"new", "old"}


def test_a_record_both_old_and_new_lanes_find_is_not_duplicated():
    pools = {"bees": ["a", "b"]}
    reuse = {("bees", "a"): {"grade": 2, "why": ""}}
    restored = bes.keep_judged(pools, [{"query": "bees"}], reuse)
    assert restored == 0
    assert sorted(pools["bees"]) == ["a", "b"]


def test_judgements_do_not_leak_across_queries():
    pools = {"bees": [], "wasps": []}
    reuse = {("bees", "a"): {"grade": 2, "why": ""},
             ("wasps", "b"): {"grade": 1, "why": ""}}
    bes.keep_judged(pools, [{"query": "bees"}, {"query": "wasps"}], reuse)
    assert pools["bees"] == ["a"] and pools["wasps"] == ["b"]


def test_a_query_dropped_from_the_eval_does_not_resurrect_its_pool():
    """Only queries in this run get a pool; stale ones are left behind."""
    pools = {"bees": ["a"]}
    reuse = {("bees", "a"): {"grade": 2, "why": ""},
             ("retired query", "z"): {"grade": 2, "why": ""}}
    bes.keep_judged(pools, [{"query": "bees"}], reuse)
    assert "retired query" not in pools


def test_the_restored_pool_is_shuffled_deterministically():
    """Position must not encode whether a record is new this run.

    Appending the carried-forward records would put every one of them after
    every freshly-retrieved one, which tells the judge exactly which is which.
    """
    reuse = {("bees", d): {"grade": 0, "why": ""} for d in "cdefghij"}
    runs = []
    for _ in range(2):
        pools = {"bees": ["a", "b"]}
        bes.keep_judged(pools, [{"query": "bees"}], reuse)
        runs.append(list(pools["bees"]))
    assert runs[0] == runs[1]                      # reproducible
    assert runs[0] != ["a", "b"] + sorted(reuse)   # and not merely appended
    assert set(runs[0]) == {"a", "b", *"cdefghij"}


def test_an_empty_reuse_file_is_not_a_crash(tmp_path):
    """`--reuse /dev/null` is the documented way to re-judge from scratch."""
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert bes.load_reuse(empty, tmp_path / "nope.json") == {}
