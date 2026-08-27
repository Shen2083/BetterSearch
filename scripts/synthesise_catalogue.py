#!/usr/bin/env python3
"""Degrade a full-text corpus into thin catalogue records.

This exists to make the enrichment question measurable. We already know how the
retrieval scores when the full article text is indexed; this script throws that
text away and leaves only what a library catalogue actually holds - title,
author, and controlled-vocabulary subject headings - so the same labelled
queries can be run against a realistically impoverished corpus.

Deliberately **no body text is carried over**, not even an opening line. Adding
a publisher blurb would quietly turn this into the "records plus partial text"
case and make the baseline look better than a real thin-record catalogue is.

    python scripts/synthesise_catalogue.py \
        --corpus data/corpus_seed.json --out data/catalogue_thin.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# Broad tag -> plausible controlled-vocabulary heading stems.
_TAG_HEADINGS = {
    "history": "History",
    "england": "Great Britain",
    "britain": "Great Britain",
    "science": "Science",
    "biology": "Life sciences",
    "medicine": "Medicine",
    "space": "Astronautics",
    "astronomy": "Astronomy",
    "engineering": "Engineering",
    "technology": "Technology",
    "computing": "Computer science",
    "war": "Military history",
    "art": "Art",
    "music": "Music",
    "food": "Food and cookery",
    "climate": "Climatology",
    "weather": "Meteorology",
    "economics": "Economics",
    "finance": "Finance",
    "law": "Law",
    "france": "France",
    "egypt": "Egypt",
    "exploration": "Discoveries in geography",
    "maritime": "Navigation",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "mathematics": "Mathematics",
    "language": "Language and languages",
    "politics": "Political science",
    "society": "Social history",
    "architecture": "Architecture",
    "infrastructure": "Public works",
    "energy": "Power resources",
    "theory": "Theory",
    "geography": "Geography",
    "cryptography": "Cryptography",
}

_SURNAMES = [
    "Ashworth", "Brennan", "Calloway", "Doyle", "Ellington", "Fairbairn",
    "Gallagher", "Harkness", "Iremonger", "Jarrow", "Kenwright", "Lockhart",
    "Merrick", "Newbold", "Ormerod", "Pemberton", "Quested", "Rutherford",
    "Sandiford", "Thirlwell", "Underhill", "Vaughan", "Whitmore", "Yardley",
]
_INITIALS = ["A.", "C.", "E. M.", "H.", "J. R.", "K.", "M. L.", "P.", "R. D.", "S."]

_STOP = {
    "the", "of", "and", "a", "an", "in", "on", "to", "for", "how", "why",
    "what", "at", "by", "first",
}


def _deterministic(seed: str, options: list[str]) -> str:
    """Stable pseudo-random pick, so re-running produces an identical file."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _author(doc_id: str) -> str:
    surname = _deterministic(doc_id + "s", _SURNAMES)
    initials = _deterministic(doc_id + "i", _INITIALS)
    return f"{surname}, {initials}"


def _year(doc_id: str) -> int:
    digest = hashlib.sha256((doc_id + "y").encode("utf-8")).digest()
    return 1962 + (digest[1] % 62)


def _title_terms(title: str) -> list[str]:
    """Significant words from the title.

    Real catalogue headings genuinely do echo title terms, so this is not
    stacking the deck - it is what a cataloguer would assign.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", title)
    return [w for w in words if w.lower() not in _STOP and len(w) > 3]


def _subjects(title: str, tags: list[str]) -> list[str]:
    heads: list[str] = []
    for tag in tags:
        heading = _TAG_HEADINGS.get(tag.lower())
        if heading and heading not in heads:
            heads.append(heading)
    terms = _title_terms(title)
    if terms:
        # A subdivided heading, in the style of LCSH.
        heads.append(" -- ".join(terms[:2]))
    return heads or ["General works"]


def to_record(doc: dict) -> dict:
    """Render one full document as a thin catalogue record."""
    title = doc["title"]
    subjects = _subjects(title, list(doc.get("tags", [])))
    record = (
        f"Author: {_author(doc['doc_id'])}\n"
        f"Published: {_year(doc['doc_id'])}\n"
        f"Subjects: {'; '.join(subjects)}"
    )
    return {
        "doc_id": doc["doc_id"],
        "title": title,
        "text": record,
        "url": doc.get("url"),
        "tags": list(doc.get("tags", ())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus_seed.json")
    parser.add_argument("--out", default="data/catalogue_thin.json")
    args = parser.parse_args()

    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    documents = raw["documents"] if isinstance(raw, dict) else raw
    records = [to_record(d) for d in documents]

    words = [len((r["title"] + " " + r["text"]).split()) for r in records]
    Path(args.out).write_text(
        json.dumps(
            {
                "name": "catalogue-thin",
                "description": (
                    "Thin catalogue records derived from corpus_seed.json. Title, "
                    "author and subject headings only - no body text. Synthetic "
                    "authors and dates; the titles and subjects are real."
                ),
                "documents": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {args.out}")
    print(f"Words per record: min {min(words)}  mean {sum(words)/len(words):.1f}  max {max(words)}")
    print("\nExample:\n")
    print(records[0]["title"])
    print(records[0]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
