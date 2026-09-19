#!/usr/bin/env python3
"""Check whether events are findable alongside books, and not in the way.

    python scripts/check_events.py --index .bettersearch/real-events

Books and events go into one index and compete for the same result slots, which
can fail in two opposite directions. The arithmetic is lopsided - 4,000 books
against 114 events - so the obvious risk is that no event ever reaches the top
20 and the feature is invisible. The opposite is worse: a bereavement cafe
turning up under `Joy of Cooking` is more damaging than never showing events at
all, because it makes the whole result list look unserious.

So both are measured here:

    reach       event-shaped queries that surface at least one event
    intrusion   book queries that surface events they have no business showing

THE DECISION THIS FEEDS
-----------------------
Written down before running it, so the result cannot be read to suit whatever
came out: **if events are invisible on queries an event should answer, retrieve
each collection separately and fuse by rank** with `reciprocal_rank_fusion` in
`src/bettersearch/fusion.py`. Rank fusion ignores corpus size, which is exactly
the problem. If events do surface on their own, the single index is enough and
the fusion path is complexity nobody needs.

WHAT THIS IS NOT
----------------
Not a relevance measurement. The events are invented, so an event surfacing for
an event-shaped query partly reflects the fact that the same person wrote both.
This checks that the plumbing puts events where a reader would look, and that it
does not put them where a reader would resent them. Quality would need a real
programme and someone other than the author writing the queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TOP_K = 20

#: Queries a reader would type wanting something to go to, not something to
#: borrow. Several deliberately share no vocabulary with the listing that should
#: answer them - "help using my phone" against a session called "Digital Help
#: Drop-in" - because a title-matching lane would find those anyway.
EVENT_QUERIES = [
    "what's on for toddlers",
    "help with my CV",
    "somewhere to go on a Tuesday afternoon",
    "help using my phone",
    "I have lost my husband and need to talk to someone",
    "free things to do with the children in the holidays",
    "learn to knit",
    "getting help with debt",
    "practise speaking English",
    "exercise class for older people",
    "fix my broken toaster",
    "find out about my family history",
]

#: Book queries where an event in the results would be a bug, not a bonus. These
#: are known-item lookups: someone typing a title or an author wants that thing.
NO_EVENT_EXPECTED = [
    "Joy of Cooking",
    "Agatha Christie",
    "The Secret Life of Bees",
    "Betty Crocker",
    "Sue Monk Kidd",
]


def is_event(doc_id: str) -> bool:
    return doc_id.startswith("ev-")


def ranking(searcher, catalogue, query: str, mode: str) -> list[dict]:
    """What the page would show - not the raw index.

    Going through run_search rather than the Searcher matters: it is where
    repeated occurrences of one session collapse, and measuring before that
    would count seven copies of a weekly drop-in as seven findings.
    """
    from api.catalogue import run_search

    data = run_search(searcher, catalogue, query=query, mode=mode,
                      page=1, per_page=TOP_K)
    return data["results"]


def report(searcher, catalogue, queries: list[str],
           mode: str) -> list[tuple[str, int, int | None, str]]:
    rows = []
    for query in queries:
        results = ranking(searcher, catalogue, query, mode)
        events = [r for r in results if is_event(r["doc_id"])]
        best = events[0]["rank"] if events else None
        top = events[0]["title"] if events else ""
        rows.append((query, len(events), best, top))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".bettersearch/real-events")
    ap.add_argument("--mode", default="semantic")
    ap.add_argument("--book-queries",
                    default="data/eval_real_queries_draft.json")
    ap.add_argument("--events", default="data/events_northfield.json")
    ap.add_argument("--books", default="data/catalogue_real.json")
    args = ap.parse_args()

    import os

    os.environ["BETTERSEARCH_CATALOGUE"] = f"{args.books},{args.events}"
    os.environ["BETTERSEARCH_INDEX_PATH"] = args.index
    from api.catalogue import load_catalogue
    from bettersearch.config import load_settings
    from bettersearch.index.numpy_index import NumpyVectorIndex
    from bettersearch.search import Searcher

    searcher = Searcher(index=NumpyVectorIndex(args.index),
                        settings=load_settings())
    catalogue = load_catalogue()
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))["documents"]

    raw = json.loads(Path(args.book_queries).read_text(encoding="utf-8"))
    book_queries = [q["query"] for q in raw["queries"]]

    print(f"\n{len(events)} events among the books · mode={args.mode} · "
          f"top {TOP_K}\n")

    # --- 1. can an event be found at all? ---------------------------------
    print("REACH - queries a reader would ask wanting somewhere to go\n")
    header = f"{'query':<52}{'events':>7}{'best':>6}  first event"
    print(header)
    print("-" * 96)
    found = 0
    for query, count, best, top in report(searcher, catalogue, EVENT_QUERIES, args.mode):
        found += bool(count)
        print(f"{query[:50]:<52}{count:>7}{(best or '-'):>6}  {top}")
    print(f"\n  {found} of {len(EVENT_QUERIES)} event-shaped queries surface an event")

    # --- 2. do events turn up where they are not wanted? ------------------
    print("\n\nINTRUSION - known-item lookups, where an event is a bug\n")
    print(header)
    print("-" * 96)
    intruded = 0
    for query, count, best, top in report(searcher, catalogue, NO_EVENT_EXPECTED, args.mode):
        intruded += bool(count)
        print(f"{query[:50]:<52}{count:>7}{(best or '-'):>6}  {top}")
    print(f"\n  {intruded} of {len(NO_EVENT_EXPECTED)} known-item queries show an event")

    # --- 3. the whole book set, for the overall rate ----------------------
    rows = report(searcher, catalogue, book_queries, args.mode)
    with_events = [r for r in rows if r[1]]
    print(f"\n\nACROSS THE {len(book_queries)} BOOK QUERIES\n")
    print(f"  {len(with_events)} return at least one event in the top {TOP_K}")
    print(f"  {sum(r[1] for r in rows)} event results in total out of "
          f"{len(book_queries) * TOP_K} slots")
    if with_events:
        print("\n  where they appear:\n")
        for query, count, best, top in sorted(with_events, key=lambda r: r[2]):
            print(f"    {query[:46]:<48}{count:>3} at rank {best:<4} {top}")

    # --- the decision, stated before the numbers were known ---------------
    print("\n" + "=" * 96)
    if found < len(EVENT_QUERIES) * 0.6:
        print(f"INVISIBLE: only {found}/{len(EVENT_QUERIES)} event-shaped queries "
              f"reach an event.\nOne index is not enough - retrieve the "
              f"collections separately and fuse by rank.")
        return 1
    print(f"One index is enough: {found}/{len(EVENT_QUERIES)} event-shaped queries "
          f"reach an event\nwithout any help, so rank fusion would be complexity "
          f"for its own sake.")
    if intruded:
        print(f"\nBut {intruded} known-item lookup(s) show an event, which is the "
              f"failure that\nmatters more. Worth looking at before this ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
