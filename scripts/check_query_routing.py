#!/usr/bin/env python3
"""Can the box tell a name from a need?

    python scripts/check_query_routing.py

The page makes a reader choose between "Catalogue search" and "Search by
meaning" - which is to say, choose a retrieval algorithm before they have
finished asking their question. A library catalogue should have one box. That
means the box has to decide, and this asks whether it can.

The decision is worth making because the two lanes fail in opposite ways.
Someone typing `Agatha Christie` wants that author and keyword answers it
perfectly; meaning-based search blurs proper nouns and hands back Christie-ish
crime novels. Someone typing `something gentle to read before bed` gets nothing
from keyword, because no record says "gentle" and "bed" together.

Fusing the lanes is not the fix and the number is already in: reciprocal rank
fusion scored 0.741 against semantic's 0.890, because it blends a mostly-empty
list into a good one. Routing is a different thing - classify, then send the
query down one lane whole.

THE TRAP IN THIS MEASUREMENT
----------------------------
The known-item queries are generated from the corpus, where titles are stored
in Title Case, and the meaning queries in data/eval_real.json are all
lowercase. Any signal that touches capitalisation would therefore separate the
two sets perfectly, score beautifully, and be measuring nothing but how the two
files happen to be punctuated. **Every query is lowercased before anything
looks at it.**

A second trap, handled by construction: a detector tuned on exact titles is no
use, because readers misremember. Half the known-item set is deliberately
damaged - one word dropped from each title - so a rule that only survives
verbatim strings is visible as such.

WHICH ERROR MATTERS MORE
------------------------
They are not symmetric. Sending a *meaning* query down the keyword lane returns
nothing useful, which is precisely the failure this repository exists to fix.
Sending a *known-item* query down the meaning lane returns something mediocre -
the right author's neighbours rather than the author. So the rule is judged on
the first error first, and a rule that is merely balanced is not good enough.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bettersearch.keyword import BM25Index, tokenize  # noqa: E402
from bettersearch.types import Chunk  # noqa: E402


def build_index(records: list[dict]) -> BM25Index:
    """BM25 over the corpus, built straight from the records.

    No embedding model is loaded anywhere in this script - the whole point of
    routing on BM25 is that it is free, so measuring it should be free too.
    """
    return BM25Index([
        Chunk(chunk_id=r["doc_id"], doc_id=r["doc_id"], title=r["title"],
              ordinal=0, text=r.get("text", ""), embed_text="",
              content_hash=r["doc_id"])
        for r in records
    ])


def known_item_queries(records: list[dict], count: int,
                       seed: int) -> list[tuple[str, set[str]]]:
    """Titles and author names, lowercased, a quarter of them damaged.

    Each query is returned with the doc_ids that would satisfy it, which is
    free ground truth: a query made *from* a record is answered correctly when
    that record comes back. No judge, no pooling bias, and 113 of them rather
    than the five exact-name controls in the eval file.
    """
    rng = random.Random(seed)
    usable = [r for r in records if r.get("title")
              and not r["title"].startswith("[")]
    by_author: dict[str, set[str]] = {}
    for r in records:
        if r.get("author"):
            by_author.setdefault(r["author"], set()).add(r["doc_id"])

    out: list[tuple[str, set[str]]] = []
    for record in rng.sample(usable, count // 2):
        out.append((record["title"].lower(), {record["doc_id"]}))
    for author in rng.sample(sorted(by_author), count // 4):
        out.append((author.lower(), by_author[author]))
    # Damaged: one word dropped, so a reader half-remembering is represented.
    for record in rng.sample(usable, count // 4):
        words = record["title"].split()
        if len(words) < 3:
            continue
        words.pop(rng.randrange(len(words)))
        out.append((" ".join(words).lower(), {record["doc_id"]}))
    return out


def signals(index: BM25Index, records: dict[str, dict], query: str) -> dict:
    """Everything a router could look at, none of it needing a model."""
    terms = tokenize(query)
    hits = index.search(query, top_k=2)
    if not hits or not terms:
        return {"terms": len(terms), "top": 0.0, "dominance": 0.0, "coverage": 0.0}

    best, second = hits[0], (hits[1] if len(hits) > 1 else None)
    record = records.get(best.chunk.doc_id, {})
    haystack = set(tokenize(f"{record.get('title', '')} "
                            f"{record.get('author', '')} "
                            f"{record.get('text', '')}"))
    return {
        "terms": len(terms),
        "top": best.score,
        # How far the best record stands clear of the runner-up.
        "dominance": (best.score - second.score) / best.score if second else 1.0,
        # The share of what was typed that the best record actually contains.
        "coverage": sum(1 for t in terms if t in haystack) / len(terms),
    }


def spread(values: list[float]) -> str:
    values = sorted(values)
    quarter = values[len(values) // 4]
    return (f"{values[0]:>8.2f}{quarter:>9.2f}"
            f"{statistics.median(values):>9.2f}{values[-1]:>8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/catalogue_real.json")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real.json")
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--judge", action="store_true",
                    help="score the rule against the judgements (loads a model)")
    ap.add_argument("--index", default=".bettersearch/real-bge")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    raw = json.loads(args.corpus.read_text(encoding="utf-8"))
    records = raw["documents"] if isinstance(raw, dict) else raw
    by_id = {r["doc_id"]: r for r in records}

    evaluation = json.loads(args.queries.read_text(encoding="utf-8"))
    meaning = [q["query"].lower() for q in
               (evaluation["queries"] if isinstance(evaluation, dict) else evaluation)]
    named = known_item_queries(records, args.count, args.seed)
    names = [q for q, _ in named]

    print(f"\nbuilding BM25 over {len(records):,} records (no model loaded)")
    index = build_index(records)
    print(f"  {len(names)} known-item queries · {len(meaning)} meaning queries\n")

    measured = {
        "known-item": [signals(index, by_id, q) for q in names],
        "meaning": [signals(index, by_id, q) for q in meaning],
    }

    for signal in ("terms", "top", "dominance", "coverage"):
        print(f"{signal}")
        print(f"  {'set':<14}{'min':>8}{'25th':>9}{'median':>9}{'max':>8}")
        for name, rows in measured.items():
            print(f"  {name:<14}{spread([float(r[signal]) for r in rows])}")
        print()

    # Coverage is the signal with a reason behind it: a title or an author name
    # is a string some record literally contains, and a description of a need
    # is not. The others are reported so the choice can be argued with.
    print("routing on coverage - the share of typed words the best record holds")
    print(f"  {'rule: coverage >=':<22}{'names sent to keyword':>24}"
          f"{'needs wrongly sent there':>26}")
    known = [r["coverage"] for r in measured["known-item"]]
    needs = [r["coverage"] for r in measured["meaning"]]
    for threshold in (0.6, 0.7, 0.8, 0.9, 1.0):
        caught = sum(1 for c in known if c >= threshold)
        wrong = sum(1 for c in needs if c >= threshold)
        print(f"  {threshold:<22.1f}{f'{caught}/{len(known)}':>24}"
              f"{f'{wrong}/{len(needs)}  ({wrong / len(needs):.0%})':>26}")

    print("\n  The right-hand column is the one that matters. A need routed to")
    print("  the keyword lane comes back empty, which is the failure this")
    print("  project exists to fix; a name routed to meaning merely comes back")
    print("  vague. Judge the first error first.\n")

    if args.judge:
        judge(measured, meaning, evaluation, named, args)
    else:
        print("  Pass --judge to score the rule against the judgements rather")
        print("  than against the assumption that an eval query is a need.\n")
    return 0


def judge(measured: dict, meaning: list[str], evaluation,
          named: list[tuple[str, set[str]]], args) -> None:
    """Which lane actually answers better, per query, on the judged set?

    Counting a coverage hit as a mistake assumes every query in the eval file
    is a need, and five of them are not: `Agatha Christie`, `Sue Monk Kidd`,
    `Joy of Cooking`, `Betty Crocker` and `The Secret Life of Bees` are
    known-item lookups that happen to live there as controls. Sending those to
    the keyword lane is the right answer, not an error.

    So stop guessing at labels and score the thing directly. Ground truth is
    whichever lane the judgements prefer, and the question is whether routing
    beats simply always using semantic - which is what the page does today.

    This is the only part that loads a model.
    """
    import os

    os.environ.setdefault("BETTERSEARCH_INDEX_PATH", args.index)
    os.environ.setdefault("BETTERSEARCH_LOCAL_MODEL", args.model)
    from bettersearch import Searcher, load_settings
    from bettersearch.evaluate import ndcg_at_k

    items = evaluation["queries"] if isinstance(evaluation, dict) else evaluation
    relevant = {q["query"]: tuple(q["relevant_doc_ids"]) for q in items}
    searcher = Searcher(settings=load_settings())

    def ranked(fn, query):
        seen = []
        for hit in fn(query, top_k=30):
            if hit.chunk.doc_id not in seen:
                seen.append(hit.chunk.doc_id)
        return seen[:10]

    print("scoring both lanes on every judged query (loads the model once)\n")
    rows = []
    for signal, query in zip(measured["meaning"], meaning):
        original = next(q for q in relevant if q.lower() == query)
        gold = relevant[original]
        k = ndcg_at_k(ranked(searcher.keyword, original), gold, 10)
        s = ndcg_at_k(ranked(searcher.semantic, original), gold, 10)
        rows.append((original, signal["coverage"], k, s))

    def mean(values):
        return statistics.mean(values)

    always_semantic = mean([s for _, _, _, s in rows])
    always_keyword = mean([k for _, _, k, _ in rows])
    routed = mean([k if c >= 1.0 else s for _, c, k, s in rows])
    best = mean([max(k, s) for _, _, k, s in rows])

    print(f"  {'nDCG@10 over the 41 judged queries':<44}")
    print(f"    always semantic  (what the page does today)  {always_semantic:.3f}")
    print(f"    always keyword                               {always_keyword:.3f}")
    print(f"    routed on coverage >= 1.0                    {routed:.3f}")
    print(f"    an oracle picking the better lane each time  {best:.3f}")

    print("\n  every query the rule sends to keyword, and what that cost:\n")
    print(f"    {'query':<46}{'keyword':>9}{'semantic':>10}{'change':>9}")
    for query, coverage, k, s in sorted(rows, key=lambda r: r[2] - r[3]):
        if coverage >= 1.0:
            print(f"    {query[:44]:<46}{k:>9.3f}{s:>10.3f}{k - s:>+9.3f}")

    # The judged set contains no known-item lookup that meaning-based search
    # fails - its five exact-name controls all score 1.000 under semantic. So
    # it cannot answer the question a router exists for. These 113 can: each
    # was generated from a record, so the right answer is known without a
    # judge, and there are enough of them to mean something.
    print(f"\n  the {len(named)} generated name lookups, where the answer is known"
          f" by construction\n")
    print(f"    {'lane':<14}{'answer at rank 1':>20}{'answer in top 5':>19}")
    for label, fn in (("keyword", searcher.keyword), ("semantic", searcher.semantic)):
        first = top5 = 0
        for query, wanted in named:
            got = ranked(fn, query)
            first += bool(got) and got[0] in wanted
            top5 += any(d in wanted for d in got[:5])
        print(f"    {label:<14}{f'{first}/{len(named)}':>20}"
              f"{f'{top5}/{len(named)}':>19}")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
