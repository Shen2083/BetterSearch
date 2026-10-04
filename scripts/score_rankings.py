#!/usr/bin/env python3
"""Score any search engine's rankings against this project's judgements.

    python scripts/score_rankings.py vespa-semantic.json
    python scripts/score_rankings.py ours.json vespa.json      # side by side

The judgements in `data/eval_real.json` are the most expensive thing this
project produced: 2,378 relevance labels over 41 queries, pooled from six
retrieval lanes and judged blind. They describe *records*, not an index, so they
outlive the engine that produced them. This script exists so a rebuild - in
Vespa or anything else - can be held to the same bar rather than to a new one.

INPUT
-----
A JSON object mapping each query string to that engine's ranked doc_ids, best
first. Nothing else; any engine can emit this.

    {
      "how to keep bees in a small garden": ["ol-OL10387420W", "ol-OL5591162W"],
      "Agatha Christie": ["ol-OL27695W", "..."]
    }

Query strings must match `data/eval_real.json` exactly. Missing queries are
reported rather than skipped silently, because a quietly shorter query set is
an easy way to post a better number.

WHY COVERAGE IS PRINTED NEXT TO THE SCORE
-----------------------------------------
**This is the part to read before believing any number here, including ours.**

The judgements were pooled from four lanes - BM25, MiniLM over thin records,
MiniLM over enriched records, and bge-base - by judging the top 20 of each. A
record no lane retrieved was never shown to a judge, and an unjudged record
counts as irrelevant.

A new engine is a *fifth lane*. Every good record it finds that those four
missed scores zero. So a system that is genuinely better can score worse here,
and the size of that effect is not small: re-pooling with bge-base as a fourth
lane moved it by 0.080 nDCG and reversed the conclusion. The same bias made
title weighting look harmful. Twice, from two directions.

So every score is printed with the share of the submitted top 10 that carries a
judgement at all. High coverage means the comparison is fair. Low coverage means
the number is measuring the pool rather than the engine, and the honest response
is to re-pool and re-judge - `scripts/build_eval_set.py` does that, and takes a
`--lane` per system - not to accept it.

WHAT THIS CANNOT TELL YOU
-------------------------
Nothing about latency, cost, freshness, or behaviour at a scale this corpus
cannot reach. It is a relevance bar and only that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bettersearch.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank

DEFAULT_QUERIES = "data/eval_real.json"

#: Known-item lookups: someone typing a title or an author wants that thing.
#: Kept in step with `NO_EVENT_EXPECTED` in scripts/check_events.py, where an
#: event surfacing on one of these is a bug rather than a bonus.
KNOWN_ITEM = [
    "Joy of Cooking",
    "Agatha Christie",
    "The Secret Life of Bees",
    "Betty Crocker",
    "Sue Monk Kidd",
]

#: Queries whose `note` begins with this are the deliberate exact-name controls.
#: Read from the data rather than hard-coded, so adding a control to the eval
#: set is enough to have it counted here.
CONTROL_PREFIX = "control"


def load_eval(path: Path) -> tuple[list[dict], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["queries"], raw


def is_event(doc_id: str) -> bool:
    return doc_id.startswith("ev-")


def score(rankings: dict[str, list[str]], queries: list[dict]) -> dict:
    """Metrics for one engine's rankings, plus the coverage that qualifies them."""
    rows: list[dict] = []
    missing: list[str] = []
    judged_seen = judged_total = 0

    for item in queries:
        q = item["query"]
        if q not in rankings:
            missing.append(q)
            continue
        ranked = list(dict.fromkeys(rankings[q]))     # a repeated doc_id is one record
        relevant = item["relevant_doc_ids"]
        # `grades` holds every record a judge actually looked at for this query,
        # relevant or not. That is what coverage is measured against - not
        # `relevant_doc_ids`, which would make coverage and precision the same
        # number and the check worthless.
        graded = set(item.get("grades") or {})
        top = ranked[:10]
        judged_seen += sum(1 for d in top if d in graded)
        judged_total += len(top)

        rows.append({
            "query": q,
            "note": item.get("note", ""),
            "ndcg@10": ndcg_at_k(ranked, relevant, 10),
            "recall@5": recall_at_k(ranked, relevant, 5),
            "mrr@10": reciprocal_rank(ranked, relevant, 10),
            "returned": len(ranked),
            "events_in_top10": sum(1 for d in top if is_event(d)),
        })

    n = len(rows) or 1
    mean = lambda key: sum(r[key] for r in rows) / n
    controls = [r for r in rows if r["note"].startswith(CONTROL_PREFIX)]
    intruded = [r for r in rows if r["query"] in KNOWN_ITEM and r["events_in_top10"]]

    return {
        "queries": len(rows),
        "missing": missing,
        "ndcg@10": mean("ndcg@10"),
        "recall@5": mean("recall@5"),
        "mrr@10": mean("mrr@10"),
        "coverage": judged_seen / judged_total if judged_total else 0.0,
        "controls": sum(r["ndcg@10"] for r in controls) / (len(controls) or 1),
        "control_count": len(controls),
        "events_intruded": [r["query"] for r in intruded],
        "rows": rows,
    }


def report(name: str, s: dict, *, show: int) -> None:
    print(f"\n{name}")
    print(f"  queries scored      {s['queries']}")
    if s["missing"]:
        print(f"  MISSING             {len(s['missing'])} query/queries absent from this file:")
        for q in s["missing"][:5]:
            print(f"                        {q!r}")
        print("                      scored on what was submitted; the mean is "
              "over fewer queries and is not comparable")
    print(f"  nDCG@10             {s['ndcg@10']:.4f}")
    print(f"  Recall@5            {s['recall@5']:.4f}")
    print(f"  MRR@10              {s['mrr@10']:.4f}")
    print(f"  known-item controls {s['controls']:.4f}  (nDCG@10 over {s['control_count']})")

    cov = s["coverage"]
    verdict = ("comparable" if cov >= 0.9 else
               "treat with caution" if cov >= 0.75 else
               "NOT comparable - re-pool before believing the score above")
    print(f"  judged coverage     {cov:.1%}  of the top 10 was seen by a judge - {verdict}")

    if s["events_intruded"]:
        print(f"  EVENTS INTRUSION    {len(s['events_intruded'])} known-item lookup(s) "
              f"return an event in the top 10, which is a bug:")
        for q in s["events_intruded"]:
            print(f"                        {q!r}")

    worst = sorted(s["rows"], key=lambda r: r["ndcg@10"])[:show]
    if worst:
        print(f"\n  weakest {len(worst)} queries:")
        for r in worst:
            print(f"    {r['ndcg@10']:.3f}  {r['query'][:58]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rankings", nargs="+",
                    help="JSON file(s) of {query: [doc_id, ...]}; several are compared")
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--show", type=int, default=5,
                    help="how many of the weakest queries to list (default: 5)")
    ap.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = ap.parse_args()

    queries, meta = load_eval(Path(args.queries))
    print(f"{len(queries)} judged queries from {args.queries}")
    print(f"judged by {meta.get('judge_model')}, pooled from "
          f"{len(meta.get('lanes_pooled', []))} lanes at "
          f"{meta.get('pool_per_lane')} per lane")

    results = {}
    for path in args.rankings:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}: expected an object of "
                             f"{{query: [doc_id, ...]}}, got {type(raw).__name__}")
        results[path] = score(raw, queries)
        report(path, results[path], show=args.show)

    if len(results) > 1:
        print("\nside by side")
        print(f"  {'file':<34} {'nDCG@10':>9} {'Recall@5':>9} {'MRR@10':>9} {'coverage':>9}")
        for path, s in results.items():
            print(f"  {path[-34:]:<34} {s['ndcg@10']:>9.4f} {s['recall@5']:>9.4f} "
                  f"{s['mrr@10']:>9.4f} {s['coverage']:>8.1%}")
        print("\n  A difference in nDCG is only meaningful between files with "
              "comparable coverage.")

    if args.json:
        print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
