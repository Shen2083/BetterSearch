"""Tests for the catalogue adapter in api/catalogue.py.

This file exists because of a bug that shipped. The adapter cut meaning-based
results at an absolute cosine floor, which looked reasonable on the 74-record
demonstration catalogue and returned 3,665 of 4,000 records on the real one -
presented to a reader as a result count. Nothing failed, nothing errored, and
no test noticed, because until now nothing tested this file at all.

The adapter is presentation code, so these tests are about the contract the page
depends on: how many results come back, that the sidebar counts describe those
results, that paging does not lose or repeat records, and that filtering narrows
the list without rewriting the sidebar. Retrieval quality is not tested here -
the fake provider in conftest hashes text, so similar text gives dissimilar
vectors and any quality assertion would be meaningless.
"""

from __future__ import annotations

from collections import Counter

import pytest

from api.catalogue import (
    SEMANTIC_TOP_K,
    build_facets,
    matches_filters,
    run_search,
    to_card,
)


class StubHit:
    """One retrieval hit, shaped like ScoredChunk as far as run_search reads it."""

    def __init__(self, doc_id: str, score: float) -> None:
        self.chunk = type("Chunk", (), {"doc_id": doc_id, "chunk_id": doc_id})()
        self.score = score


class StubSearcher:
    """Returns a fixed ranking, honouring top_k the way a real Searcher does.

    Scores descend from 1.0 in even steps so a cut-off is visible in the output:
    if the adapter ever reintroduces a score threshold, the result count changes
    and these tests say so.
    """

    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids
        self.keyword_top_k: int | None = None
        self.semantic_top_k: int | None = None

    def _hits(self, top_k: int) -> list[StubHit]:
        n = len(self.doc_ids)
        return [StubHit(d, 1.0 - i / max(n, 1))
                for i, d in enumerate(self.doc_ids[:top_k])]

    def keyword(self, query: str, *, top_k: int = 10) -> list[StubHit]:
        self.keyword_top_k = top_k
        return self._hits(top_k)

    def semantic(self, query: str, *, top_k: int = 10) -> list[StubHit]:
        self.semantic_top_k = top_k
        return self._hits(top_k)


def make_catalogue(n: int) -> dict[str, dict]:
    """A catalogue big enough that a top-k cut is not the whole corpus."""
    formats = ["Book", "Large print", "eAudiobook"]
    branches = ["Northfield Central", "Ashcombe Branch"]
    return {
        f"doc-{i:03d}": {
            "doc_id": f"doc-{i:03d}",
            "title": f"Record {i}",
            "text": f"Author: Writer {i}\nPublished: 19{50 + i % 50}",
            "author": f"Writer {i}",
            "year": str(1950 + i % 50),
            "format": formats[i % 3],
            "language": "English",
            "fiction": i % 2 == 0,
            "subjects": ["Testing"],
            "location": branches[i % 2],
            "available": i % 4,
            "copies": 3,
            "cover": "#2E4A62",
        }
        for i in range(n)
    }


@pytest.fixture
def catalogue() -> dict[str, dict]:
    return make_catalogue(200)


@pytest.fixture
def searcher(catalogue: dict[str, dict]) -> StubSearcher:
    return StubSearcher(list(catalogue))


# --- the cut-off ----------------------------------------------------------


def test_semantic_returns_at_most_top_k(searcher, catalogue):
    """The whole point of the fix: a 200-record catalogue does not return 200."""
    data = run_search(searcher, catalogue, query="anything", mode="semantic",
                      per_page=SEMANTIC_TOP_K * 2)

    assert data["total"] == SEMANTIC_TOP_K
    assert searcher.semantic_top_k == SEMANTIC_TOP_K
    assert len(data["results"]) == SEMANTIC_TOP_K


def test_semantic_asks_the_index_for_top_k_not_the_whole_corpus(searcher, catalogue):
    """Retrieving everything and then discarding it is the old shape; this is not."""
    run_search(searcher, catalogue, query="anything", mode="semantic")

    assert searcher.semantic_top_k == SEMANTIC_TOP_K
    assert searcher.semantic_top_k < len(catalogue)


def test_keyword_keeps_full_depth(searcher, catalogue):
    """The asymmetry is deliberate - BM25 stops on its own, cosine does not."""
    data = run_search(searcher, catalogue, query="anything", mode="keyword",
                      per_page=len(catalogue))

    assert searcher.keyword_top_k == len(catalogue)
    assert data["total"] == len(catalogue)


def test_a_short_ranking_is_not_padded(catalogue):
    """Returning fewer than k results is a real answer, not something to fill."""
    searcher = StubSearcher(["doc-000", "doc-001", "doc-002"])
    data = run_search(searcher, catalogue, query="anything", mode="semantic")

    assert data["total"] == 3


def test_missing_records_are_dropped_not_rendered(catalogue):
    """An id in the index with no catalogue record must not become a blank card."""
    searcher = StubSearcher(["doc-000", "ghost", "doc-001"])
    data = run_search(searcher, catalogue, query="anything", mode="semantic")

    assert [r["doc_id"] for r in data["results"]] == ["doc-000", "doc-001"]


def test_duplicate_doc_ids_collapse_to_one_card(catalogue):
    """Retrieval works on chunks; a catalogue shows records, one card each."""
    searcher = StubSearcher(["doc-000", "doc-000", "doc-001"])
    data = run_search(searcher, catalogue, query="anything", mode="semantic")

    assert [r["doc_id"] for r in data["results"]] == ["doc-000", "doc-001"]


# --- facets describe the results ------------------------------------------


def test_facet_counts_describe_the_returned_results(searcher, catalogue):
    """Under the old floor these counted a set the reader could never page to."""
    data = run_search(searcher, catalogue, query="anything", mode="semantic",
                      per_page=SEMANTIC_TOP_K)

    fiction = next(f for f in data["facets"] if f["key"] == "fiction")
    assert sum(v["count"] for v in fiction["values"]) == data["total"]


def test_facets_are_counted_before_filters_are_applied(searcher, catalogue):
    """The sidebar shows what you could narrow to, not only what is selected."""
    unfiltered = run_search(searcher, catalogue, query="anything", mode="semantic")
    filtered = run_search(searcher, catalogue, query="anything", mode="semantic",
                          filters={"fiction": ["Fiction"]})

    assert filtered["total"] < unfiltered["total"]
    assert filtered["facets"] == unfiltered["facets"]


def test_filters_keep_only_matching_records(searcher, catalogue):
    data = run_search(searcher, catalogue, query="anything", mode="semantic",
                      filters={"format": ["Large print"]}, per_page=SEMANTIC_TOP_K)

    assert data["results"]
    assert all(r["format"] == "Large print" for r in data["results"])


def test_an_impossible_filter_returns_nothing_without_erroring(searcher, catalogue):
    data = run_search(searcher, catalogue, query="anything", mode="semantic",
                      filters={"format": ["Papyrus"]})

    assert data["total"] == 0
    assert data["results"] == []
    assert data["showing"] == [0, 0]


# --- paging ---------------------------------------------------------------


def test_paging_covers_the_result_set_exactly_once(searcher, catalogue):
    per_page = 4
    seen: list[str] = []
    for page in range(1, SEMANTIC_TOP_K // per_page + 1):
        data = run_search(searcher, catalogue, query="anything", mode="semantic",
                          page=page, per_page=per_page)
        seen.extend(r["doc_id"] for r in data["results"])

    assert len(seen) == SEMANTIC_TOP_K
    assert len(set(seen)) == SEMANTIC_TOP_K


def test_ranks_continue_across_pages(searcher, catalogue):
    """Rank is the position in the whole result set, not on the page."""
    page_two = run_search(searcher, catalogue, query="anything", mode="semantic",
                          page=2, per_page=5)

    assert [r["rank"] for r in page_two["results"]] == [6, 7, 8, 9, 10]
    assert page_two["showing"] == [6, 10]


def test_paging_past_the_end_is_empty_rather_than_wrapping(searcher, catalogue):
    data = run_search(searcher, catalogue, query="anything", mode="semantic",
                      page=99, per_page=10)

    assert data["results"] == []
    assert data["showing"] == [0, 0]
    assert data["total"] == SEMANTIC_TOP_K


# --- the helpers the offline build re-implements in JS ---------------------


def test_build_facets_drops_facets_with_nothing_in_them():
    """web/offline-search.js mirrors this; an empty facet must not render."""
    records = [{"doc_id": "a", "title": "A", "fiction": True}]
    keys = {f.key for f in build_facets(records)}

    assert "fiction" in keys
    assert "format" not in keys  # no format on the record, so no heading


def test_build_facets_orders_by_count_then_name():
    records = ([{"doc_id": f"a{i}", "format": "Book"} for i in range(3)]
               + [{"doc_id": "b", "format": "DVD"}])
    formats = next(f for f in build_facets(records) if f.key == "format")

    assert [v.value for v in formats.values] == ["Book", "DVD"]
    assert [v.count for v in formats.values] == [3, 1]


def test_matches_filters_treats_multiple_values_as_or():
    record = {"doc_id": "a", "format": "DVD"}

    assert matches_filters(record, {"format": ["Book", "DVD"]})
    assert not matches_filters(record, {"format": ["Book"]})


def test_matches_filters_treats_multiple_facets_as_and():
    record = {"doc_id": "a", "format": "DVD", "fiction": True}

    assert matches_filters(record, {"format": ["DVD"], "fiction": ["Fiction"]})
    assert not matches_filters(record, {"format": ["DVD"], "fiction": ["Non-fiction"]})


def test_availability_counts_online_and_on_the_shelf_separately():
    records = [
        {"doc_id": "a", "available": 2, "format": "Book"},
        {"doc_id": "b", "available": 0, "format": "Book"},
        {"doc_id": "c", "available": 1, "format": "eAudiobook"},
    ]
    facet = next(f for f in build_facets(records) if f.key == "availability")
    counts = Counter({v.value: v.count for v in facet.values})

    assert counts["Available now"] == 2
    assert counts["Available online"] == 1


def test_to_card_returns_the_record_not_a_rewrite():
    """Enrichment text is a retrieval bridge; a card must show what is catalogued."""
    record = {
        "doc_id": "a", "title": "A Real Title", "author": "A Writer",
        "text": "Author: A Writer", "available": 2, "copies": 3,
    }
    card = to_card(record, 0.42, rank=1)

    assert card["title"] == "A Real Title"
    assert card["record_text"] == "Author: A Writer"
    assert card["score"] == 0.42
    assert card["rank"] == 1


# --- events as a second collection ----------------------------------------


def event(doc_id: str, title: str, series: str, **extra) -> dict:
    return {
        "doc_id": doc_id, "title": title, "series_id": series,
        "record_type": "event", "format": "Event", "location": "Rowley Green",
        "available": 6, "copies": 6, "audience": "Adults",
        "when": "Tuesdays, 2.00pm", "booking": "Just turn up", "cost": "Free",
        **extra,
    }


@pytest.fixture
def mixed() -> dict[str, dict]:
    """Books, plus one weekly session that exists as several dated records."""
    catalogue = make_catalogue(10)
    for i in range(4):
        catalogue[f"ev-knit-{i}"] = event(f"ev-knit-{i}", "Knit and Natter",
                                          "s-knit")
    catalogue["ev-repair"] = event("ev-repair", "Repair Cafe", "s-repair")
    return catalogue


def test_repeat_occurrences_of_one_session_collapse_to_one_card(mixed):
    """Seven copies of a weekly drop-in is not seven results."""
    searcher = StubSearcher([f"ev-knit-{i}" for i in range(4)] + ["ev-repair"])
    data = run_search(searcher, mixed, query="knitting", mode="semantic")

    titles = [r["title"] for r in data["results"]]
    assert titles == ["Knit and Natter", "Repair Cafe"]
    assert data["total"] == 2


def test_a_collapsed_series_says_how_many_dates_it_has(mixed):
    searcher = StubSearcher([f"ev-knit-{i}" for i in range(4)])
    data = run_search(searcher, mixed, query="knitting", mode="semantic")

    assert data["results"][0]["repeats"] == 4


def test_collapsing_keeps_the_best_ranked_occurrence(mixed):
    """The one that matched best is the one to show, not whichever came first."""
    searcher = StubSearcher(["ev-knit-2", "ev-knit-0", "ev-knit-1"])
    data = run_search(searcher, mixed, query="knitting", mode="semantic")

    assert data["results"][0]["doc_id"] == "ev-knit-2"


def test_books_are_never_collapsed(mixed):
    """Books carry no series_id, so the collapse must not touch them."""
    searcher = StubSearcher([f"doc-{i:03d}" for i in range(10)])
    data = run_search(searcher, mixed, query="anything", mode="semantic")

    assert data["total"] == 10


def test_records_are_typed_so_a_card_can_say_which_it_is(mixed):
    searcher = StubSearcher(["ev-repair", "doc-000"])
    data = run_search(searcher, mixed, query="anything", mode="semantic")

    assert [r["record_type"] for r in data["results"]] == ["event", "book"]


def test_the_type_facet_separates_the_two_collections(mixed):
    searcher = StubSearcher(["ev-repair"] + [f"doc-{i:03d}" for i in range(10)])
    data = run_search(searcher, mixed, query="anything", mode="semantic")

    facet = next(f for f in data["facets"] if f["key"] == "record_type")
    counts = {v["value"]: v["count"] for v in facet["values"]}
    assert counts == {"Books": 10, "Events": 1}


def test_filtering_to_events_leaves_only_events(mixed):
    searcher = StubSearcher(["ev-repair"] + [f"doc-{i:03d}" for i in range(10)])
    data = run_search(searcher, mixed, query="anything", mode="semantic",
                      filters={"record_type": ["Events"]})

    assert data["total"] == 1
    assert data["results"][0]["record_type"] == "event"


# --- the page describes the corpus it is actually serving ------------------


def test_meta_comes_from_the_corpus_not_the_page(tmp_path, monkeypatch):
    """The bug this prevents: serving real records under "records are invented",
    with a button for a surname collision that only exists in another corpus."""
    import json

    import api.catalogue as mod

    books = tmp_path / "books.json"
    books.write_text(json.dumps({
        "description": "Records are real.", "presets": ["how to keep bees in a small garden"],
        "documents": [],
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "CATALOGUE_PATHS", [books])

    meta = mod.load_meta()
    assert meta["presets"] == ["how to keep bees in a small garden"]
    assert meta["description"] == "Records are real."


def test_several_collections_each_contribute_presets_and_caveats(tmp_path, monkeypatch):
    import json

    import api.catalogue as mod

    books = tmp_path / "books.json"
    books.write_text(json.dumps({
        "description": "Records are real.", "presets": ["Agatha Christie"],
        "documents": [],
    }), encoding="utf-8")
    events = tmp_path / "events.json"
    events.write_text(json.dumps({
        "description": "Events are invented.", "presets": ["what's on for toddlers"],
        "documents": [],
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "CATALOGUE_PATHS", [books, events])

    meta = mod.load_meta()
    assert meta["presets"] == ["Agatha Christie", "what's on for toddlers"]
    assert meta["description"] == "Records are real. Events are invented."


def test_a_corpus_without_presets_yields_none_rather_than_borrowing(tmp_path, monkeypatch):
    """No presets is an honest empty row; someone else's presets are a lie."""
    import json

    import api.catalogue as mod

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"documents": []}), encoding="utf-8")
    monkeypatch.setattr(mod, "CATALOGUE_PATHS", [bare])

    assert mod.load_meta() == {"presets": [], "description": ""}


def test_every_shipped_corpus_declares_its_own_presets():
    """A corpus with no presets renders a page with nothing to press."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("catalogue_library.json", "catalogue_real.json",
                 "events_northfield.json"):
        raw = json.loads((root / "data" / name).read_text(encoding="utf-8"))
        assert raw.get("presets"), f"{name} declares no example queries"
        assert raw.get("description"), f"{name} does not describe itself"
