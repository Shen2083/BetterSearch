#!/usr/bin/env python3
"""Can the page tell that it has nothing useful for you?

    python scripts/check_answerability.py --index .bettersearch/real-bge

A reader who searches for something this catalogue simply does not hold still
gets twenty results, because cosine similarity has no floor. The question is
whether anything in the output distinguishes "here you go" from "we have
nothing, but here are twenty records anyway".

Two candidate signals are measured here against the same two query sets. The
answer for the first is a flat no, and it is worth knowing why.

**The score cannot do it.** Top-1 similarity for queries the catalogue can
answer runs 0.553-0.797; for queries it verifiably cannot, 0.556-0.748. Every
absent query scores above the weakest answerable one. `harry potter and the
chamber of secrets` scores 0.748 - above the median *answerable* query -
because the corpus holds a film score by John Williams and that is a real
match. This is the 0.15 relevance floor's lesson arriving from a second
direction: an absolute cosine value does not mean what a reader would want it
to mean, and no threshold on it separates these two populations.

**Agreement between the results can.** If the catalogue really holds a topic,
the top few results tend to share subject headings with each other; when it is
reaching, they scatter. On mean pairwise Jaccard overlap across the top 5, all
eleven absent queries fall below the answerable median, and `harry potter` -
the one the score liked most - sits near the bottom at 0.030.

WHY THIS IS MEASURED AND NOT SHIPPED
------------------------------------
The threshold curve below has a knee around 0.04: about two-thirds of absent
queries caught for roughly one good query in ten wrongly flagged. That is a
usable trade, and it is still **eleven queries**. Reading a constant off eleven
examples and wiring it into the page is exactly how the 0.15 floor happened -
a number that looked right on a small sample, tuned to one corpus, admitting
3,665 of 4,000 records on the real one.

So the mechanism is recorded here and nothing in the page depends on it, in the
same spirit as TITLE_WEIGHT in src/bettersearch/keyword.py. To take it further,
grow CANDIDATES to forty or fifty verified-absent queries and re-run: either
the knee holds, or it moves, and either way that is the answer.

A NOTE ON THE QUERY SETS
------------------------
The absent topics are verified against the corpus mechanically - a topic is
kept only if its distinctive term appears in at most one record. Choosing them
by taste would manufacture the separation this is meant to test for. `pilates`
was dropped by that check on the current corpus, which is the check working.

The answerable set is data/eval_real.json, whose queries were written to be
answerable. That is a real limitation: it means "answerable" here also means
"written by someone who knew the corpus", and a genuine reader's answerable
queries may be messier than these.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TOP_N = 5

#: (query a reader might type, the term that proves the corpus has the topic)
CANDIDATES = (
    ("harry potter and the chamber of secrets", "harry potter"),
    ("books about taylor swift", "taylor swift"),
    ("learning to play the ukulele", "ukulele"),
    ("the life of nelson mandela", "mandela"),
    ("how to code in rust", "rust programming"),
    ("cryptocurrency and blockchain explained", "blockchain"),
    ("k-pop and korean music", "k-pop"),
    ("the story of the falklands war", "falklands"),
    ("beginners guide to pilates", "pilates"),
    ("books by sally rooney", "sally rooney"),
    ("how to use tiktok safely", "tiktok"),
    ("the history of formula one racing", "formula one"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".bettersearch/real-bge")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/catalogue_real.json")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real.json")
    args = ap.parse_args()

    raw = json.loads(args.corpus.read_text(encoding="utf-8"))
    records = raw["documents"] if isinstance(raw, dict) else raw
    haystack = [f"{r.get('title', '')} {r.get('text', '')}".lower() for r in records]
    subjects = {r["doc_id"]: set(r.get("subjects") or []) for r in records}

    print(f"\nverifying the absent set against {len(records):,} records")
    absent = []
    for query, probe in CANDIDATES:
        hits = sum(1 for text in haystack if probe in text)
        verdict = "kept" if hits <= 1 else "DROPPED - the corpus has this"
        print(f"  {probe:<22}{hits:>4} record(s)   {verdict}")
        if hits <= 1:
            absent.append(query)

    os.environ["BETTERSEARCH_INDEX_PATH"] = args.index
    os.environ["BETTERSEARCH_LOCAL_MODEL"] = args.model
    from bettersearch import Searcher, load_settings

    searcher = Searcher(settings=load_settings())
    eval_raw = json.loads(args.queries.read_text(encoding="utf-8"))
    answerable = [q["query"] for q in
                  (eval_raw["queries"] if isinstance(eval_raw, dict) else eval_raw)]

    def top_documents(query: str, k: int) -> tuple[list[str], float]:
        docs, seen = [], set()
        best = 0.0
        for hit in searcher.semantic(query, top_k=k * 3):
            if not docs:
                best = float(hit.score)
            if hit.chunk.doc_id not in seen:
                seen.add(hit.chunk.doc_id)
                docs.append(hit.chunk.doc_id)
            if len(docs) == k:
                break
        return docs, best

    def coherence(docs: list[str]) -> float:
        sets = [subjects.get(d, set()) for d in docs]
        pairs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
        return statistics.mean(pairs) if pairs else 0.0

    measured = {}
    for name, queries in (("answerable", answerable), ("absent", absent)):
        rows = []
        for query in queries:
            docs, best = top_documents(query, TOP_N)
            rows.append((query, best, coherence(docs)))
        measured[name] = rows

    def spread(values: list[float]) -> str:
        values = sorted(values)
        return (f"{values[0]:>8.3f}{statistics.median(values):>9.3f}"
                f"{values[-1]:>8.3f}")

    for label, column, index in (("top-1 similarity - the signal that does NOT work", "", 1),
                                 ("top-5 subject agreement - the one that does", "", 2)):
        print(f"\n{label}")
        print(f"  {'set':<26}{'min':>8}{'median':>9}{'max':>8}")
        for name in ("answerable", "absent"):
            rows = measured[name]
            print(f"  {name + f' ({len(rows)})':<26}"
                  f"{spread([r[index] for r in rows])}")

    good = sorted(r[2] for r in measured["answerable"])
    bad = sorted((r[2], r[0]) for r in measured["absent"])
    print(f"\nwhere a line on subject agreement would fall")
    print(f"  {'threshold':<12}{'absent caught':>16}{'good wrongly flagged':>24}")
    for t in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        caught = sum(1 for score, _ in bad if score < t)
        false = sum(1 for score in good if score < t)
        print(f"  {t:<12.2f}{f'{caught}/{len(bad)}':>16}"
              f"{f'{false}/{len(good)}  ({false / len(good):.0%})':>24}")

    print(f"\nevery absent query by subject agreement, least coherent first")
    for score, query in bad:
        print(f"  {score:.3f}  {query}")
    print("\nNothing in the page reads these numbers. See the module docstring "
          "for why.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
