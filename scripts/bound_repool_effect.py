#!/usr/bin/env python3
"""Before paying a judge: what can re-pooling actually change?

    python scripts/bound_repool_effect.py \
        --lane data/rankings/rerank-bge-200.json \
        --lane data/rankings/rerank-bge-200-promoted.json \
        --arm "baseline=/tmp/lane200/d200-baseline--no-reranking-.json"

Adding a lane to the judged pool adds records to `relevant_doc_ids`, which is
the denominator every published recall figure was measured against. So
re-pooling moves every arm's number, not just the new lane's, and it is worth
knowing the size of that move before spending anything - partly to budget, but
mostly because if the answer is "nothing can change" there is no reason to run
the batch at all.

The bounds are exact. Every newly-pooled record is either judged relevant or
not, so:

  * **none relevant** - `relevant_doc_ids` is untouched and every published
    figure stays exactly where it is. The run still pays for itself, because
    the new lane's judged coverage goes to 100% and its number becomes
    comparable for the first time.
  * **all relevant** - the largest possible denominator: the worst case for
    arms that do not retrieve the new records and the best case for the lane
    that found them.

Between those, the fraction that come back relevant decides the ordering, and
this script sweeps it. The sweep is a sensitivity analysis, **not** a
prediction: it says which outcomes are reachable and roughly where the
break-even sits, so a surprise afterwards can be recognised as a surprise.

TWO THINGS IT CANNOT TELL YOU
-----------------------------
The records being judged are the new lane's own top picks. So "does the new
lane win" is partly "does the judge agree with the new lane", and a reranker
that approaches the ceiling here is increasingly agreeing with Claude Haiku
rather than demonstrably serving readers better. Re-pooling removes the
penalty for finding records the pool missed; it does not make an LLM judge into
ground truth. Human judgements, or click data from a real service, are the only
things that would.

And the base rate is borrowed. The share of records that come back relevant is
taken from the *previous* pool, whose records came from four retrieval lanes. A
cross-encoder's picks are not a random sample of those, so treat the sweep as a
range and read the real number off the batch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_rankings(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_at_10(rankings: dict, per_q: dict, extra: dict) -> float:
    """recall@10 of judged-relevant records - the metric with a fixed denominator.

    The headline for this question precisely because the denominator does not
    depend on the arm: nDCG@10 is not comparable while judged coverage differs
    between arms, which at these depths it does.
    """
    total = 0.0
    for query, v in per_q.items():
        relevant = v["relevant"] | extra[query]
        if not relevant:
            continue
        top = rankings.get(query, [])[:10]
        total += len(set(top) & relevant) / len(relevant)
    return total / len(per_q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=ROOT / "data/eval_real.json")
    ap.add_argument("--lane", action="append", default=[], required=True,
                    help="a lane about to be pooled, {query: [doc_id, ...]}; "
                         "repeatable. Its records not already judged are the "
                         "ones whose grades are unknown.")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH.json",
                    help="an arm to re-measure under each assumption; repeatable. "
                         "Pass the arms whose published figures you care about, "
                         "not only the new lane.")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--rates", default="0,0.1,0.2,0.32,0.4,0.5,0.75,1",
                    help="fractions of newly-pooled records to assume relevant")
    args = ap.parse_args()

    evaluation = json.loads(args.eval.read_text(encoding="utf-8"))
    lanes = [load_rankings(p) for p in args.lane]

    per_q = {}
    for item in evaluation["queries"]:
        query = item["query"]
        judged = set(item.get("grades") or {})
        pooled: set[str] = set()
        for lane in lanes:
            pooled |= set(lane.get(query, []))
        per_q[query] = {"relevant": set(item["relevant_doc_ids"]),
                        # Sorted, so a given seed samples the same records on
                        # every run and two invocations are comparable.
                        "new": sorted(pooled - judged)}

    counts = sorted(len(v["new"]) for v in per_q.values())
    rel = sorted(len(v["relevant"]) for v in per_q.values())
    print(f"{sum(counts)} newly-pooled (query, record) pairs over "
          f"{len(per_q)} queries")
    print(f"  new per query:      min {counts[0]} median {counts[len(counts)//2]} "
          f"max {counts[-1]}")
    print(f"  relevant per query: min {rel[0]} median {rel[len(rel)//2]} "
          f"max {rel[-1]}  (today's denominator)")
    if not sum(counts):
        print("\nNothing new is pooled, so no figure can move and the batch "
              "would buy nothing.")
        return 0

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_rankings(path)
    if not arms:
        print("\nNo --arm given, so there is nothing to re-measure.")
        return 0

    first = next(iter(arms))
    rates = [float(r) for r in args.rates.split(",")]
    print(f"\nrecall@10 of judged-relevant records, by the share of newly-pooled"
          f"\nrecords that come back relevant. '{first}' is the comparison.\n")
    head = f"  {'share relevant':<22}" + "".join(f"{n[:15]:>17}" for n in arms)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for rate in rates:
        # 0 and 1 are deterministic - there is nothing to sample - so they are
        # exact bounds rather than estimates, and are not averaged.
        trials = 1 if rate in (0.0, 1.0) else args.trials
        rng = random.Random(3)
        totals = {n: 0.0 for n in arms}
        beat = 0
        for _ in range(trials):
            extra = {q: {d for d in v["new"] if rng.random() < rate}
                     for q, v in per_q.items()}
            got = {n: recall_at_10(r, per_q, extra) for n, r in arms.items()}
            for n in arms:
                totals[n] += got[n]
            beat += any(got[n] > got[first] for n in arms if n != first)
        note = ""
        if trials > 1:
            note = f"   beats {first} in {beat}/{trials}"
        elif rate == 0.0:
            note = "   exact: every published figure unmoved"
        else:
            note = "   exact: the largest denominator possible"
        print(f"  {rate:<22.2f}" + "".join(f"{totals[n]/trials:>17.4f}" for n in arms)
              + note)

    print("\nRead this as a range, not a forecast. The share is borrowed from the")
    print("previous pool, whose records came from four retrieval lanes - a")
    print("cross-encoder's picks are not a random sample of those. And the records")
    print("being judged are the new lane's own choices, so a lane that wins here")
    print("has partly won by agreeing with the judge. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
