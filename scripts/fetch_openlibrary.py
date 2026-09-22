#!/usr/bin/env python3
"""Build a catalogue of ~4,000 real bibliographic records from Open Library.

    python scripts/fetch_openlibrary.py --target 4000

Why this exists: every number measured so far rests on records I wrote, and the
invented titles are unusually descriptive - "The Battle of Hastings" gives the
game away, which is why thin records already score 0.789 against a 0.890
ceiling. Real catalogues are not like that. A probe of subject:beekeeping
returns *The Secret Life of Bees*, *Palimpsest* and *Chalice* - novels whose
titles say nothing about their subject. That is the case the product exists for,
and a synthetic corpus cannot produce it honestly.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real, from Open Library: title, author, publication year, publisher, subject
headings, language.

Invented here: availability, number of copies, branch, format and cover colour.
Open Library has no holdings data, and the demonstration page needs those fields
to render. They are derived deterministically from the record id so they are
stable across runs - but they are decoration, and the README and the page footer
say so. Never quote an availability figure from this corpus as if it were real.

The mess is kept on purpose. Open Library folds call numbers and Dewey codes
into `subject` ("Tx715 .r7499 1998", "641.5973"); those are an export artefact
rather than headings a cataloguer typed, so they are filtered. Everything else
stays - inconsistent near-duplicates ("American Cookery" / "Cooking, American"),
missing authors, missing dates. That inconsistency is the thing under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "catalogue_real.json"

API = "https://openlibrary.org/search.json"
FIELDS = "key,title,author_name,first_publish_year,publisher,subject,language"
PAGE = 100

BRANCHES = ["Northfield Central", "Ashcombe Branch", "Rowley Green",
            "Marsden Park", "Fenwick Road"]
PALETTE = ["#2E4A62", "#3A5A40", "#6B4226", "#4A3B52", "#1F3A3D",
           "#5C3A3A", "#33475B", "#264653"]
FORMATS = ["Book", "Book", "Book", "Book", "Large print", "eAudiobook", "DVD"]

#: Subject areas a public library actually shelves. Breadth matters more than
#: depth - a corpus of 4,000 cookery books would not test anything.
SUBJECTS: list[tuple[str, bool]] = [
    # (open library subject, is_fiction)
    ("detective and mystery stories", True),
    ("science fiction", True),
    ("historical fiction", True),
    ("romance", True),
    ("thriller", True),
    ("fantasy", True),
    ("horror tales", True),
    ("humorous stories", True),
    ("short stories", True),
    ("graphic novels", True),
    ("picture books", True),
    ("juvenile fiction", True),
    ("young adult fiction", True),
    ("cookery", False),
    ("gardening", False),
    ("beekeeping", False),
    ("knitting", False),
    ("woodwork", False),
    ("parenting", False),
    ("pregnancy", False),
    ("child rearing", False),
    ("mental health", False),
    ("mindfulness", False),
    ("bereavement", False),
    ("sleep", False),
    ("nutrition", False),
    ("physical fitness", False),
    ("biography", False),
    ("autobiography", False),
    ("local history", False),
    ("world war, 1939-1945", False),
    ("british history", False),
    ("travel", False),
    ("natural history", False),
    ("birds", False),
    ("astronomy", False),
    ("popular science", False),
    ("mathematics", False),
    ("computers", False),
    ("personal finance", False),
    ("small business", False),
    ("job hunting", False),
    ("english language", False),
    ("poetry", False),
    ("art", False),
    ("photography", False),
    ("music", False),
    ("football", False),
    ("cycling", False),
    ("dogs", False),
]

#: Call numbers, Dewey codes and other export artefacts masquerading as subject
#: headings. A cataloguer never typed "Tx715 .r7499 1998" into a subject field.
_JUNK_SUBJECT = re.compile(
    r"""^(
        [\d.]+$                       # bare Dewey: 641.5973
      | [A-Z]{1,3}\d+(\s|\.|$).*      # LC call number: Tx715 .r7499 1998
      | .*\b(nyt|new york times):.*   # bestseller-list tags
      | (accessible_book|protected_daisy|in_library|internet_archive|overdrive)
      | lending_library|popular_print_disabled_books|large_type_books
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
MAX_SUBJECTS = 8

#: Deliberately none of them heavy. These sit on the page as buttons for a
#: visitor to press, and a demonstration is not the place to open with
#: bereavement - the grief query stays in the eval set, where it is evidence
#: rather than a greeting.
PRESETS = [
    "learning to take better photographs",
    "something gentle to read before bed",
    "how to keep bees in a small garden",
    "books about the night sky for beginners",
    "Agatha Christie",
]


def clean_subjects(raw: list[str]) -> list[str]:
    """Drop export artefacts, keep real headings including inconsistent ones."""
    out: list[str] = []
    seen: set[str] = set()
    for s in raw or []:
        s = " ".join(str(s).split())
        if not s or len(s) > 60 or _JUNK_SUBJECT.match(s):
            continue
        # Case-insensitive dedupe only; "American Cookery" vs "Cooking, American"
        # survive on purpose - that inconsistency is what we are testing.
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= MAX_SUBJECTS:
            break
    return out


def fetch_subject(subject: str, want: int, *, retries: int = 3) -> list[dict]:
    """One subject, paged until `want` records or the subject runs out."""
    got: list[dict] = []
    for offset in range(0, want, PAGE):
        params = urllib.parse.urlencode({
            "q": f'subject:"{subject}"',
            "fields": FIELDS,
            "limit": min(PAGE, want - offset),
            "offset": offset,
        })
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(f"{API}?{params}", timeout=60) as r:
                    docs = json.loads(r.read()).get("docs", [])
                break
            except Exception as exc:  # network wobble, not a reason to lose the run
                if attempt == retries - 1:
                    print(f"    ! {subject} offset {offset}: {exc}", file=sys.stderr)
                    docs = []
                else:
                    time.sleep(2 ** attempt)
        if not docs:
            break
        got.extend(docs)
        if len(docs) < PAGE:
            break
    return got


def render_record(author, year, publisher, subjects) -> str:
    """The record text as a catalogue stores it - and all that gets indexed.

    Deliberately the same shape as scripts/build_catalogue.py, so the two
    catalogues are directly comparable. No Summary line: Open Library has no
    synopsis field, which makes this corpus uniformly thin - exactly the
    condition the enrichment pipeline exists to address.
    """
    lines = [f"Author: {author}" if author else "Author: [unknown]"]
    pub = ", ".join(x for x in (publisher, year) if x)
    lines.append(f"Published: {pub}" if pub else "Published: [no date]")
    if subjects:
        lines.append("Subjects: " + "; ".join(subjects))
    return "\n".join(lines)


def to_record(doc: dict, *, fiction: bool) -> dict | None:
    title = " ".join(str(doc.get("title", "")).split())
    subjects = clean_subjects(doc.get("subject", []))
    if not title or not subjects:
        return None  # unusable as a catalogue record

    doc_id = "ol-" + str(doc["key"]).rsplit("/", 1)[-1]
    digest = hashlib.sha256(doc_id.encode()).digest()

    author = (doc.get("author_name") or [""])[0]
    year = str(doc.get("first_publish_year") or "")
    publisher = (doc.get("publisher") or [""])[0]
    if len(publisher) > 40:
        publisher = publisher[:40].rsplit(" ", 1)[0]
    langs = doc.get("language") or []
    language = "English" if (not langs or "eng" in langs) else langs[0]

    fmt = FORMATS[digest[1] % len(FORMATS)]
    available = 0 if digest[2] % 5 == 0 else 1 + digest[3] % 3
    copies = available + digest[4] % 3

    return {
        # --- corpus fields: what BetterSearch indexes ----------------------
        "doc_id": doc_id,
        "title": title,
        "text": render_record(author, year, publisher, subjects),
        "tags": [fmt.lower()],
        # --- display fields: ignored by the library, used by the page ------
        "author": author,
        "author_dates": "",
        "year": year,
        "publisher": publisher,
        "format": fmt,
        "language": language,
        "fiction": fiction,
        "subjects": subjects,
        # --- invented: Open Library has no holdings ------------------------
        "location": "Online" if fmt == "eAudiobook" else BRANCHES[digest[5] % len(BRANCHES)],
        "available": max(copies, 1) if fmt == "eAudiobook" else available,
        "copies": max(copies, 1),
        "blurb": None,
        "cover": PALETTE[digest[6] % len(PALETTE)],
    }


def build(target: int, out: Path) -> dict:
    """Fair share per subject first, then round-robin top-up.

    Taking as much as possible from each subject in turn skews the corpus badly:
    an earlier version filled its 4,000 from the first 24 of 50 subjects, leaving
    astronomy, personal finance and job hunting with nothing at all. Every
    subject gets an equal share first; only then do the richer subjects make up
    any shortfall.
    """
    share = max(target // len(SUBJECTS), 1)
    records: dict[str, dict] = {}
    leftovers: list[list[dict]] = []
    duplicates = 0

    for subject, fiction in SUBJECTS:
        docs = fetch_subject(subject, share * 2)
        added, spare = 0, []
        for doc in docs:
            record = to_record(doc, fiction=fiction)
            if record is None:
                continue
            if record["doc_id"] in records:
                duplicates += 1
                continue
            if added < share:
                records[record["doc_id"]] = record
                added += 1
            else:
                spare.append(record)
        leftovers.append(spare)
        print(f"  {subject:<32} +{added:<4} (total {len(records)}, {len(spare)} spare)")

    # Top up round-robin so no single subject dominates the remainder.
    while len(records) < target and any(leftovers):
        progressed = False
        for spare in leftovers:
            if len(records) >= target:
                break
            while spare:
                record = spare.pop(0)
                if record["doc_id"] in records:
                    duplicates += 1
                    continue
                records[record["doc_id"]] = record
                progressed = True
                break
        if not progressed:
            break
    print(f"\n  after top-up: {len(records)}")

    ordered = list(records.values())
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "name": "northfield-libraries-real",
        # The page reads its example queries from here rather than carrying its
        # own, so a corpus can never be served with another corpus's buttons.
        # These are drawn from the reviewed eval set, so each one is judged.
        "presets": PRESETS,
        "description": (
            "Bibliographic records are real, from Open Library (openlibrary.org, "
            "public domain). Availability, copies, branch, format and cover "
            "colour are invented - Open Library holds no holdings data - and "
            "must not be quoted as real."
        ),
        "source": "https://openlibrary.org/developers/api",
        "documents": ordered,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"records": ordered, "duplicates": duplicates}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    print(f"fetching ~{args.target} records across {len(SUBJECTS)} subjects\n")
    result = build(args.target, args.out)
    records = result["records"]

    words = sorted(len(r["text"].split()) + len(r["title"].split()) for r in records)
    subs = sorted(len(r["subjects"]) for r in records)
    print(f"\nWrote {len(records)} records to {args.out}"
          f"  ({args.out.stat().st_size // 1024} KB)")
    print(f"  duplicates skipped   {result['duplicates']}")
    print(f"  words per record     min {words[0]}  median {words[len(words)//2]}  max {words[-1]}")
    print(f"  subjects per record  min {subs[0]}  median {subs[len(subs)//2]}  max {subs[-1]}")
    print(f"  missing author       {sum(1 for r in records if not r['author'])}"
          f" ({sum(1 for r in records if not r['author'])/len(records):.0%})")
    print(f"  missing year         {sum(1 for r in records if not r['year'])}")
    print(f"  fiction / non-fiction {sum(1 for r in records if r['fiction'])}"
          f" / {sum(1 for r in records if not r['fiction'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
