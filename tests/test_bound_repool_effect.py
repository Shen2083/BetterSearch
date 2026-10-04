"""Tests for the pre-spend bound on what re-pooling can change.

The script's job is to be trustworthy before any money is spent, so the two
exact bounds have to be exactly right: a zero-relevant assumption must leave
today's figures untouched, and a sweep must never claim a figure can move when
nothing new is pooled.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bound_repool_effect.py"
_spec = importlib.util.spec_from_file_location("bound_repool_effect", SCRIPT)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)


def test_no_new_records_relevant_leaves_recall_where_it_is():
    per_q = {"bees": {"relevant": {"a"}, "new": ["x", "y"]}}
    r = br.recall_at_10({"bees": ["a"]}, per_q, {"bees": set()})
    assert r == pytest.approx(1.0)


def test_a_new_relevant_record_the_arm_misses_lowers_its_recall():
    """The whole point: the denominator grows and the arm does not follow."""
    per_q = {"bees": {"relevant": {"a"}, "new": ["x"]}}
    r = br.recall_at_10({"bees": ["a"]}, per_q, {"bees": {"x"}})
    assert r == pytest.approx(0.5)


def test_an_arm_that_retrieves_the_new_record_is_not_penalised():
    per_q = {"bees": {"relevant": {"a"}, "new": ["x"]}}
    r = br.recall_at_10({"bees": ["a", "x"]}, per_q, {"bees": {"x"}})
    assert r == pytest.approx(1.0)


def test_only_the_top_ten_counts():
    per_q = {"bees": {"relevant": {"a"}, "new": []}}
    ranking = {"bees": [f"pad{i}" for i in range(10)] + ["a"]}
    assert br.recall_at_10(ranking, per_q, {"bees": set()}) == pytest.approx(0.0)


def test_a_query_with_nothing_relevant_is_skipped_not_counted_as_zero():
    """Scoring it as 0.0 would drag every arm down by the same amount and make
    the comparison look closer than it is."""
    per_q = {"bees": {"relevant": {"a"}, "new": []},
             "wasps": {"relevant": set(), "new": []}}
    # One of two queries contributes; the mean is still over both, by design -
    # what must not happen is the empty query contributing a spurious 0 from a
    # division it cannot do.
    r = br.recall_at_10({"bees": ["a"]}, per_q, {"bees": set(), "wasps": set()})
    assert r == pytest.approx(0.5)


def test_recall_is_a_mean_over_queries_not_over_records():
    """Otherwise a query with 35 relevant records would outvote one with 1."""
    per_q = {"big": {"relevant": set("abcdefghij"), "new": []},
             "small": {"relevant": {"z"}, "new": []}}
    extra = {"big": set(), "small": set()}
    r = br.recall_at_10({"big": [], "small": ["z"]}, per_q, extra)
    assert r == pytest.approx(0.5)
