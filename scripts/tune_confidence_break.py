#!/usr/bin/env python3
"""Where should the results list stop looking confident?

    python scripts/tune_confidence_break.py --index .bettersearch/real-bge

Meaning-based search returns its top 20 whatever the query, because cosine
similarity has no floor - every record scores against every query. That is the
right call for *ranking* (`scripts/tune_cutoff.py` settled it: fixed top-k beat
every threshold), but it leaves the page telling a reader that twenty records
are the answer when sometimes four are and sixteen are padding.

This asks whether a **presentation** break can be drawn honestly: "closest
matches" above it, "more loosely related" below. Retrieval is untouched; the
same twenty records come back in the same order. The only question is whether
a rule exists that puts the relevant ones above the line often enough to be
worth drawing.

WHAT WOULD MAKE THIS MEASUREMENT A LIE
--------------------------------------
Unjudged counts as irrelevant. If the judge saw rank 3 but never saw rank 17,
then "precision below the break" is low *by construction* and the break looks
brilliant no matter where it is drawn. That is the same pooling bias that made
bge-base look worse than MiniLM and that made title weighting look harmful -
twice now, from two directions.

So every rule reports **judged coverage by depth** alongside its score. If
coverage falls away with rank, the lift is measuring the pool, not the rule,
and the honest answer is that this cannot be settled on the current judgements.

WHAT IT SAID, AND WHY NOTHING SHIPPED
-------------------------------------
No break is drawn on the page, because this says one is not justified.

Coverage came back **100% at every depth**, so for once the pooling bias above
is not in play and the numbers can be read at face value. They do not support
the feature: precision *below* the break runs 62-68% whatever rule is used. The
records that would be labelled "more loosely related" are relevant about two
thirds of the time, so the premise - that the tail of the list is padding - is
simply false for these queries.

The lift looks respectable at around +20 points, but that is 87% against 67%,
and the rules with the best lift leave **79-83% of a query's relevant records
below the line**. Drawing it would demote four good results in five to make the
page look decisive.

One real gap remains, and it is not fixable here: every query in the eval was
written to be answerable, so this cannot speak to the case the feature was
imagined for - a query the catalogue simply cannot serve, where the twenty
results really are padding. That question is asked separately, and answered,
in scripts/check_answerability.py.

NO ABSOLUTE FLOORS
------------------
An absolute cosine threshold is not a candidate here and will not be added
later. 0.15 was eyeballed on 74 records and admitted 3,665 of 4,000 on the real
corpus, because a fixed threshold admits a roughly constant *fraction* of a
corpus rather than a constant number. Every rule below is relative to the
query's own score distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DEPTH = 20


# ------------------------------------------------------------------ the rules
#
# Each takes the descending score list for one query and returns how many
# results sit *above* the break. Returning len(scores) means "no break drawn".
def rule_fraction_of_top(scores: list[float], alpha: float) -> int:
    """Keep while a score is at least `alpha` of the best one."""
    top = scores[0]
    return sum(1 for s in scores if s >= alpha * top) or 1


def rule_largest_gap(scores: list[float], _: float) -> int:
    """Break at the biggest consecutive drop anywhere in the list."""
    if len(scores) < 2:
        return len(scores)
    gaps = [(scores[i] - scores[i + 1], i + 1) for i in range(len(scores) - 1)]
    return max(gaps)[1]


def rule_share_of_spread(scores: list[float], beta: float) -> int:
    """Keep while a score is within `beta` of the way down to the worst one."""
    top, bottom = scores[0], scores[-1]
    if top == bottom:
        return len(scores)
    return sum(1 for s in scores if s >= top - beta * (top - bottom)) or 1


RULES = (
    ("fraction of top   a=0.98", rule_fraction_of_top, 0.98),
    ("fraction of top   a=0.95", rule_fraction_of_top, 0.95),
    ("fraction of top   a=0.92", rule_fraction_of_top, 0.92),
    ("fraction of top   a=0.90", rule_fraction_of_top, 0.90),
    ("share of spread   b=0.25", rule_share_of_spread, 0.25),
    ("share of spread   b=0.50", rule_share_of_spread, 0.50),
    ("largest gap", rule_largest_gap, 0.0),
    ("fixed cut at 5", lambda s, _: min(5, len(s)), 0.0),
    ("fixed cut at 10", lambda s, _: min(10, len(s)), 0.0),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".bettersearch/real-bge")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real.json")
    args = ap.parse_args()

    os.environ["BETTERSEARCH_INDEX_PATH"] = args.index
    os.environ["BETTERSEARCH_LOCAL_MODEL"] = args.model
    from bettersearch import Searcher, load_settings

    raw = json.loads(args.queries.read_text(encoding="utf-8"))
    items = raw["queries"] if isinstance(raw, dict) else raw
    searcher = Searcher(settings=load_settings())

    # One retrieval per query, reused by every rule - the rules differ only in
    # where they draw the line through an identical list.
    runs = []
    coverage_at_rank: dict[int, list[int]] = defaultdict(list)
    for item in items:
        grades = {k: int(v) for k, v in (item.get("grades") or {}).items()}
        seen, scores, relevant, judged = set(), [], [], []
        for hit in searcher.semantic(item["query"], top_k=DEPTH * 3):
            doc = hit.chunk.doc_id
            if doc in seen:
                continue
            seen.add(doc)
            scores.append(float(hit.score))
            relevant.append(grades.get(doc, 0) >= 1)
            judged.append(doc in grades)
            if len(scores) == DEPTH:
                break
        for rank, was_judged in enumerate(judged, start=1):
            coverage_at_rank[rank].append(was_judged)
        runs.append((item["query"], scores, relevant))

    print(f"\n{len(runs)} judged queries · top {DEPTH} · {args.index}\n")

    print("judged coverage by depth - read this first")
    print("-" * 52)
    for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20)):
        flags = [f for r in range(lo, hi + 1) for f in coverage_at_rank[r]]
        print(f"  ranks {lo:>2}-{hi:<2}  {statistics.mean(flags):>6.1%} judged")
    overall = [f for flags in coverage_at_rank.values() for f in flags]
    print(f"  overall     {statistics.mean(overall):>6.1%} judged")

    header = (f"\n{'rule':<26}{'above':>7}{'P(above)':>10}{'P(below)':>10}"
              f"{'lift':>8}{'missed':>8}")
    print(header)
    print("-" * len(header.strip("\n")))

    for name, rule, param in RULES:
        sizes, above_p, below_p, missed = [], [], [], []
        for _, scores, relevant in runs:
            if not scores:
                continue
            k = max(1, min(rule(scores, param), len(scores)))
            sizes.append(k)
            above, below = relevant[:k], relevant[k:]
            above_p.append(sum(above) / len(above))
            if below:
                below_p.append(sum(below) / len(below))
            # The cost of drawing a line: relevant records pushed under it.
            total_relevant = sum(relevant)
            missed.append(sum(below) / total_relevant if total_relevant else 0.0)
        a, b = statistics.mean(above_p), statistics.mean(below_p or [0.0])
        print(f"{name:<26}{statistics.mean(sizes):>7.1f}{a:>10.1%}{b:>10.1%}"
              f"{a - b:>+8.1%}{statistics.mean(missed):>8.1%}")

    print("\n  above   mean number of results above the break")
    print("  lift    P(above) - P(below). A break worth drawing separates them.")
    print("  missed  share of a query's relevant records left below the line -")
    print("          the cost of drawing it, and the reason a big lift alone")
    print("          does not justify a rule.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
