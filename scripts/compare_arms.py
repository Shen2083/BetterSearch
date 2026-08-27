#!/usr/bin/env python3
"""Compare retrieval quality across experiment arms on one labelled query set.

An arm is just an index built over a different rendering of the same items -
full text, thin catalogue records, enriched records - evaluated with identical
queries and labels. Sorting by the drop from the first arm shows exactly which
queries a later arm has to rescue, which is more actionable than a single
aggregate number.

    python scripts/compare_arms.py \
        ceiling=.bettersearch/index \
        thin=.bettersearch/catalogue-thin \
        enriched=.bettersearch/catalogue-enriched
"""

from __future__ import annotations

import argparse
import sys

from bettersearch.config import load_settings
from bettersearch.evaluate import evaluate_mode, load_eval_queries, ndcg_at_k
from bettersearch.evaluate import _ranked_doc_ids
from bettersearch.index.numpy_index import NumpyVectorIndex
from bettersearch.search import Searcher


def build(path: str) -> Searcher:
    settings = load_settings()
    return Searcher(index=NumpyVectorIndex(path), settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", help="name=index_path")
    parser.add_argument("--queries", default="data/eval_queries.json")
    parser.add_argument("--mode", default="semantic")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show", type=int, default=12, help="worst N queries to list")
    args = parser.parse_args()

    arms: list[tuple[str, Searcher]] = []
    for spec in args.arms:
        if "=" not in spec:
            print(f"Expected name=path, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        arms.append((name, build(path)))

    queries = load_eval_queries(args.queries)

    print(f"\n{len(queries)} queries · mode={args.mode}\n")
    header = f"{'arm':<14}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@10':>10}{'chunks':>9}"
    print(header)
    print("-" * len(header))
    for name, searcher in arms:
        m = evaluate_mode(searcher, queries, mode=args.mode, top_k=args.top_k)
        print(
            f"{name:<14}{m.recall_at_5:>10.3f}{m.mrr_at_10:>10.3f}"
            f"{m.ndcg_at_10:>10.3f}{searcher.stats().chunk_count:>9}"
        )

    # Per-query, sorted by how far each arm falls behind the first.
    rows = []
    for item in queries:
        scores = []
        for _, searcher in arms:
            response = searcher.search(item.query, mode=args.mode, top_k=args.top_k)
            ranked = _ranked_doc_ids(response.results)
            scores.append(ndcg_at_k(ranked, item.relevant_doc_ids, args.top_k))
        rows.append((item.query, scores))

    rows.sort(key=lambda r: r[1][1] - r[1][0] if len(r[1]) > 1 else 0)

    print(f"\nWorst {args.show} queries by drop from '{arms[0][0]}' (nDCG@10)\n")
    head = f"{'query':<46}" + "".join(f"{n:>12}" for n, _ in arms) + f"{'drop':>9}"
    print(head)
    print("-" * len(head))
    for query, scores in rows[: args.show]:
        label = query if len(query) <= 44 else query[:41] + "..."
        drop = scores[1] - scores[0] if len(scores) > 1 else 0.0
        print(f"{label:<46}" + "".join(f"{s:>12.3f}" for s in scores) + f"{drop:>9.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
