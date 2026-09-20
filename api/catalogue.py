"""Catalogue adapter for the demonstration search page.

Everything here is presentation: loading the records, computing facet counts and
joining display fields onto retrieval results. The library itself is untouched -
``Searcher.keyword`` and ``Searcher.semantic`` are used exactly as they are, and
nothing in ``src/bettersearch`` knows this file exists.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DATA = Path(__file__).resolve().parent.parent / "data"
#: Which catalogue the page serves. Settable so the 74-record demonstration
#: catalogue and the 4,000-record real one can both be served without a code
#: change - they share this schema exactly.
#:
#: Comma-separated for several collections at once - books and events, which a
#: library maintains separately and a reader searches together. Whatever is
#: listed here has to match what was ingested, or the page will rank records it
#: cannot then display.
CATALOGUE_PATHS = [
    Path(p) for p in
    os.environ.get("BETTERSEARCH_CATALOGUE", str(_DATA / "catalogue_library.json")).split(",")
    if p.strip()
]
#: Kept for callers that want the primary corpus - notably the standalone build,
#: which reads its footnote out of the corpus description.
CATALOGUE_PATH = CATALOGUE_PATHS[0]

#: Meaning-based retrieval always returns its top_k, however weak the match, so
#: a catalogue has to stop the list somewhere or the result count stops meaning
#: anything. This was an absolute cosine floor of 0.15, eyeballed on 74 records.
#: On 4,000 it fell apart: a fixed threshold admits a roughly constant *fraction*
#: of a corpus, not a constant number, so `books like Agatha Christie` came back
#: with 3,665 of 4,000 records above the floor and the page said "3,665 results".
#:
#: Plain top-k wins instead - precision 0.398 against the floor's 0.035 on the
#: judged set. It also beat every threshold variant, including the relative cut
#: that looked right from result counts alone. Reproduce with
#: `python scripts/tune_cutoff.py`.
#:
#: 20 is where the evidence stops rather than where it peaks: the eval pooled
#: each lane to depth 20, so nothing deeper is judged. F1 was still rising there.
#:
#: Settable, because 20 is measured *for the 4,000-record corpus* and is not a
#: universal constant. On the 74-record demonstration catalogue it is a quarter
#: of the shelf, and results 11-20 for the headline query are a Punjabi book
#: with no title and a DVD about a lighthouse. That is the same error as the
#: floor it replaced - a number tuned on one corpus applied to another - and the
#: honest response is to expose it rather than to hard-code a second guess. Any
#: value set here is eyeballed unless it came out of scripts/tune_cutoff.py
#: against judgements for the corpus being served.
SEMANTIC_TOP_K = int(os.environ.get("BETTERSEARCH_TOP_K", "20"))


@dataclass(frozen=True, slots=True)
class FacetValue:
    value: str
    count: int


@dataclass(frozen=True, slots=True)
class Facet:
    key: str
    label: str
    values: list[FacetValue]


def load_meta() -> dict[str, Any]:
    """What the page needs to describe the corpus it is actually serving.

    The example queries and the footnote used to be written into
    web/catalogue.html. That silently tied the page to one corpus: serving the
    real 4,000 Open Library records still offered a button for `grewal`, a
    surname collision that only exists in the invented catalogue and returns
    nothing, under a footnote stating the records were invented - which is true
    of the demonstration catalogue and false of real bibliographic data.

    Both now come from the corpus file, so a corpus cannot be served with
    another corpus's examples or another corpus's caveat. Several collections
    contribute in order: books bring their queries, events bring theirs.
    """
    presets: list[str] = []
    described: list[str] = []
    for path in CATALOGUE_PATHS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for query in raw.get("presets", []):
            if query not in presets:
                presets.append(query)
        if description := (raw.get("description") or "").strip():
            described.append(description)
    return {"presets": presets, "description": " ".join(described)}


def load_catalogue() -> dict[str, dict[str, Any]]:
    catalogue: dict[str, dict[str, Any]] = {}
    for path in CATALOGUE_PATHS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw["documents"] if isinstance(raw, dict) else raw
        for record in records:
            catalogue[record["doc_id"]] = record
    return catalogue


def _decade(year: str) -> str | None:
    if not year or not year[:3].isdigit():
        return None
    return f"{year[:3]}0s"


def _availability(record: dict) -> list[str]:
    tags = []
    if record.get("available", 0) > 0:
        tags.append("Available now")
    if record.get("format") == "eAudiobook":
        tags.append("Available online")
    return tags


def _record_type(record: dict) -> list[str]:
    """Books unless a record says otherwise.

    Absent on every book record, because the books were catalogued long before
    there was anything else to tell them apart from. Defaulting here rather than
    backfilling 4,000 records keeps the two collections independent: adding a
    third needs a new value, not a migration.
    """
    return ["Events" if record.get("record_type") == "event" else "Books"]


#: (key, label, how to read the value(s) off a record)
_FACET_SPEC: list[tuple[str, str, Any]] = [
    ("record_type", "Books / Events", _record_type),
    ("availability", "Availability", _availability),
    ("format", "Format", lambda r: [r.get("format")] if r.get("format") else []),
    ("fiction", "Fiction / Non-fiction",
     lambda r: ["Fiction" if r.get("fiction") else "Non-fiction"]),
    ("decade", "Publication date",
     lambda r: [d] if (d := _decade(r.get("year", ""))) else []),
    ("location", "Location", lambda r: [r.get("location")] if r.get("location") else []),
    ("language", "Language", lambda r: [r.get("language")] if r.get("language") else []),
]


def build_facets(records: list[dict]) -> list[Facet]:
    """Facet counts computed from the actual result set, as a catalogue does."""
    facets: list[Facet] = []
    for key, label, reader in _FACET_SPEC:
        counter: Counter[str] = Counter()
        for record in records:
            counter.update(reader(record))
        if not counter:
            continue
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        facets.append(
            Facet(key=key, label=label,
                  values=[FacetValue(value=v, count=c) for v, c in ordered])
        )
    return facets


def matches_filters(record: dict, filters: dict[str, list[str]]) -> bool:
    for key, label, reader in _FACET_SPEC:
        wanted = filters.get(key)
        if not wanted:
            continue
        if not set(reader(record)) & set(wanted):
            return False
    return True


def to_card(record: dict, score: float, rank: int) -> dict[str, Any]:
    """One result card. Deliberately returns the real record, never a rewrite."""
    return {
        "rank": rank,
        "score": round(float(score), 4),
        "doc_id": record["doc_id"],
        # Books and events are ranked together, so a card has to say which it is
        # before it says anything else - a result list where a Tuesday drop-in
        # looks like a book is worse than not returning the drop-in at all.
        "record_type": record.get("record_type", "book"),
        "audience": record.get("audience") or "",
        "when": record.get("when") or "",
        "booking": record.get("booking") or "",
        "cost": record.get("cost") or "",
        "repeats": int(record.get("repeats", 1)),
        "title": record["title"],
        "author": record.get("author") or "",
        "author_dates": record.get("author_dates") or "",
        "year": record.get("year") or "",
        "publisher": record.get("publisher") or "",
        "format": record.get("format") or "",
        "subjects": record.get("subjects") or [],
        "blurb": record.get("blurb"),
        "location": record.get("location") or "",
        "available": int(record.get("available", 0)),
        "copies": int(record.get("copies", 0)),
        "cover": record.get("cover") or "#2E4A62",
        "record_text": record.get("text", ""),
    }


def run_search(
    searcher,
    catalogue: dict[str, dict],
    *,
    query: str,
    mode: str,
    filters: dict[str, list[str]] | None = None,
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    filters = filters or {}
    depth = len(catalogue)

    if mode == "keyword":
        # Keyword keeps full depth on purpose, which makes the two modes
        # asymmetric. BM25 has a cut-off built in - a record sharing no term with
        # the query simply does not match - so its result count is already a
        # statement about the query. Cosine similarity has no such floor: every
        # record scores against every query, and something has to impose the end
        # of the list.
        hits = searcher.keyword(query, top_k=depth)
    else:
        hits = searcher.semantic(query, top_k=SEMANTIC_TOP_K)

    # Retrieval works on chunks; a catalogue shows records. One chunk per record
    # here, but de-duplicate by doc_id so this stays correct if that changes.
    #
    # A weekly event is several records - one per date, each with its own branch
    # and place count - so a query matching the session matches all of them.
    # Left alone, "something short I can finish in one sitting" filled its first
    # five slots with five copies of the same chair exercise class. Occurrences
    # of one series therefore collapse to the best-ranked one, which is also how
    # a reader thinks about it: one thing that happens on Tuesdays, not seven
    # things. `repeats` carries the rest so the card can say so.
    seen: set[str] = set()
    series_at: dict[str, int] = {}
    ranked: list[tuple[dict, float]] = []
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        record = catalogue.get(doc_id)
        if record is None:
            continue
        if (series := record.get("series_id")) is not None:
            if (position := series_at.get(series)) is not None:
                first, score = ranked[position]
                ranked[position] = ({**first,
                                     "repeats": first.get("repeats", 1) + 1}, score)
                continue
            series_at[series] = len(ranked)
        ranked.append((record, hit.score))

    # Facets are counted before filtering, so the sidebar shows what you could
    # narrow to rather than only what is already selected. Under the old floor
    # this counted over a 763-record shadow set the reader could never page to;
    # it now describes the results actually returned, which is what a reader
    # takes the numbers to mean.
    facets = build_facets([r for r, _ in ranked])

    kept = [(r, s) for r, s in ranked if matches_filters(r, filters)]
    total = len(kept)
    start = (page - 1) * per_page
    window = kept[start : start + per_page]

    return {
        "query": query,
        "mode": mode,
        "total": total,
        "page": page,
        "per_page": per_page,
        "showing": [start + 1, start + len(window)] if window else [0, 0],
        "results": [to_card(r, s, start + i + 1) for i, (r, s) in enumerate(window)],
        "facets": [
            {"key": f.key, "label": f.label,
             "values": [{"value": v.value, "count": v.count} for v in f.values]}
            for f in facets
        ],
    }
