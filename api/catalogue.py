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
CATALOGUE_PATH = Path(
    os.environ.get("BETTERSEARCH_CATALOGUE", _DATA / "catalogue_library.json")
)

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
SEMANTIC_TOP_K = 20


@dataclass(frozen=True, slots=True)
class FacetValue:
    value: str
    count: int


@dataclass(frozen=True, slots=True)
class Facet:
    key: str
    label: str
    values: list[FacetValue]


def load_catalogue() -> dict[str, dict[str, Any]]:
    raw = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    records = raw["documents"] if isinstance(raw, dict) else raw
    return {r["doc_id"]: r for r in records}


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


#: (key, label, how to read the value(s) off a record)
_FACET_SPEC: list[tuple[str, str, Any]] = [
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
    seen: set[str] = set()
    ranked: list[tuple[dict, float]] = []
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        record = catalogue.get(doc_id)
        if record is not None:
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
