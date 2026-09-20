#!/usr/bin/env python3
"""Choose where to cut a meaning-based result list, using the judgements.

    python scripts/tune_cutoff.py thin=.bettersearch/real

Meaning-based retrieval returns its top_k however weak the match, so a catalogue
page has to cut the list somewhere. `api/catalogue.py` cut it at an absolute
cosine floor tuned by eye on 74 records; on 4,000 that floor returns a median of
763 results and calls them a result set.

This script exists because the table that replaced it was produced ad hoc and
then lost. A number in the README that nothing regenerates is a number nobody can
check, so the strategies are all implemented here and measured the same way:

    precision  of the records a strategy returns, how many are judged relevant
    recall     of the judged-relevant records, how many it returns
    F1         the two together, which is what picking a cut is trading off

WHAT THE POOL CAN AND CANNOT TELL YOU
-------------------------------------
`data/eval_real.json` was built by pooling the top 20 of each retrieval lane and
judging that pool, so **no record below rank 20 in its own lane has a judgement**
and every one of them counts as irrelevant here. A strategy is therefore never
rewarded for depth it does have, only punished for it.

Within a lane that makes ranks 1-20 fully judged and everything deeper only
partly judged - a record at rank 25 has a grade if some *other* lane pooled it,
and otherwise counts as irrelevant whatever it is. So a cut that returns more
than 20 has its precision pushed down by construction, and the resulting fall in
F1 is not evidence against it.

The sweep is printed past the boundary anyway, marked, because hiding it would
invite someone to assume the curve keeps climbing. But the only comparison this
eval can actually make is between strategies that stay inside the pool. Deeper
cuts are unmeasured, not worse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bettersearch.config import load_settings  # noqa: E402
from bettersearch.evaluate import load_eval_queries  # noqa: E402
from bettersearch.index.numpy_index import NumpyVectorIndex  # noqa: E402
from bettersearch.search import Searcher  # noqa: E402

#: How deep to retrieve before a strategy cuts. Has to exceed anything the
#: absolute floor admits, or the floor's damage is hidden by the retrieval depth
#: rather than measured.
DEPTH = 4000

TOP_K_SWEEP = (5, 10, 15, 20, 25, 30, 50)


def build(path: str) -> Searcher:
    return Searcher(index=NumpyVectorIndex(path), settings=load_settings())


# --- strategies -----------------------------------------------------------
# Each takes the full ranking as (doc_id, score) and returns the kept prefix.
# They are prefix cuts by construction: none reorders, they only decide where
# the list stops, which is the only decision the page is making.


def absolute_floor(ranked, floor: float):
    return [r for r in ranked if r[1] >= floor]


def relative_to_best(ranked, fraction: float):
    if not ranked:
        return []
    bar = ranked[0][1] * fraction
    return [r for r in ranked if r[1] >= bar]


def largest_gap(ranked, *, search_within: int = 50):
    """Cut at the biggest score drop near the top.

    The intuition is that a relevant cluster ends somewhere and the scores fall
    off a step. It is only looked for in the first `search_within` results: over
    a whole 4,000-record ranking the largest single gap is usually far down the
    tail, between two records nobody would ever see.
    """
    if len(ranked) < 2:
        return list(ranked)
    window = min(search_within, len(ranked))
    gaps = [(ranked[i][1] - ranked[i + 1][1], i + 1) for i in range(window - 1)]
    return list(ranked[: max(gaps)[1]])


def fixed_top_k(ranked, k: int):
    return list(ranked[:k])


def strategies() -> list[tuple[str, callable]]:
    rows: list[tuple[str, callable]] = [
        ("absolute 0.15 (current)", lambda r: absolute_floor(r, 0.15)),
        ("relative >= 0.90 x best", lambda r: relative_to_best(r, 0.90)),
        ("relative >= 0.85 x best", lambda r: relative_to_best(r, 0.85)),
        ("largest-gap cut", largest_gap),
    ]
    rows += [(f"fixed top-{k}", lambda r, k=k: fixed_top_k(r, k)) for k in TOP_K_SWEEP]
    return rows


# --- measurement ----------------------------------------------------------


def score(kept_ids: list[str], relevant: set[str]) -> tuple[float, float]:
    if not kept_ids:
        # Returning nothing is perfectly precise and useless. Scoring it 1.0
        # would let a strategy win by refusing to answer.
        return 0.0, 0.0
    hits = sum(1 for d in kept_ids if d in relevant)
    return hits / len(kept_ids), (hits / len(relevant) if relevant else 0.0)


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate(searcher: Searcher, queries, mode: str) -> list[dict]:
    # Retrieve once per query; every strategy then cuts the same ranking, so
    # differences are the cut and nothing else.
    rankings: list[tuple[list[tuple[str, float]], set[str]]] = []
    for item in queries:
        hits = searcher.search(item.query, mode=mode, top_k=DEPTH).results
        seen: dict[str, float] = {}
        for h in hits:
            seen.setdefault(h.chunk.doc_id, h.score)
        rankings.append((list(seen.items()), set(item.relevant_doc_ids)))

    rows = []
    for name, cut in strategies():
        precisions, recalls, sizes = [], [], []
        for ranked, relevant in rankings:
            kept = cut(ranked)
            p, r = score([d for d, _ in kept], relevant)
            precisions.append(p)
            recalls.append(r)
            sizes.append(len(kept))
        p = statistics.mean(precisions)
        r = statistics.mean(recalls)
        rows.append({
            "strategy": name,
            "precision": p,
            "recall": r,
            "f1": f1(p, r),
            "median_results": statistics.median(sizes),
        })
    return rows


def pool_depth(path: str) -> int:
    """How deep each lane was pooled - the rank past which judgements thin out."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return int(raw.get("pool_per_lane", 20)) if isinstance(raw, dict) else 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", default=["thin=.bettersearch/real"],
                    help="name=index_path (default: thin=.bettersearch/real)")
    ap.add_argument("--queries", default="data/eval_real.json")
    ap.add_argument("--mode", default="semantic")
    args = ap.parse_args()

    queries = load_eval_queries(args.queries)
    depth = pool_depth(args.queries)
    print(f"\n{len(queries)} judged queries from {args.queries} · mode={args.mode}")
    print(f"Each lane was pooled to depth {depth}, so a cut returning more than "
          f"{depth}\nresults is scored partly against records no judge ever saw. "
          f"Those rows are\nmarked 'blind' and cannot be compared with the rest.\n")

    for spec in args.arms:
        if "=" not in spec:
            print(f"Expected name=path, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        rows = evaluate(build(path), queries, args.mode)
        for row in rows:
            row["blind"] = row["median_results"] > depth

        measured = [r for r in rows if not r["blind"]]
        best = max(measured, key=lambda r: r["f1"])

        header = (f"{'strategy':<26}{'precision':>11}{'recall':>9}{'F1':>8}"
                  f"{'median':>9}")
        print(f"arm: {name}  ({path})")
        print(header)
        print("-" * len(header))
        for row in rows:
            if row["blind"]:
                mark = "  blind - past the pool"
            elif row is best:
                mark = "  <-- best measured F1"
            else:
                mark = ""
            print(f"{row['strategy']:<26}{row['precision']:>11.3f}"
                  f"{row['recall']:>9.3f}{row['f1']:>8.3f}"
                  f"{row['median_results']:>9.0f}{mark}")

        # Within the pool the judgements are complete, so this comparison stands.
        # Past it they are not, so the script says nothing about deeper cuts -
        # in particular it does not read the fall after the boundary as evidence,
        # because unjudged records drag precision down whatever they contain.
        sweep = [r for r in measured if r["strategy"].startswith("fixed top-")]
        climbing = all(a["f1"] <= b["f1"] for a, b in zip(sweep, sweep[1:]))
        print(f"\n  best measured: {best['strategy']} (F1 {best['f1']:.3f})")
        if climbing and sweep and sweep[-1] is best:
            print(f"  F1 is still rising at the pool boundary. {best['strategy']} "
                  f"is the deepest\n  cut this eval can vouch for, not a peak - "
                  f"whether deeper is better is unmeasured.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
