"""What HNSW costs, measured against the exact search over the same vectors.

This is the only unbiased number in the proof of concept. Every other figure
here is qualified by judged coverage, because the Vespa lanes were never pooled
and an unjudged record counts as irrelevant. This one needs no judgements at
all: it asks the same index the same question twice, exactly and
approximately, and reports how much of the exact answer the approximation
found. Engine against itself.

THE LIMIT, BECAUSE IT MATTERS MORE THAN THE NUMBER

4,000 records is far too few for HNSW to behave the way it will in production.
The interesting regime is 10^5 to 10^8 documents; at 4,000 a graph search
touches a large fraction of the corpus anyway. Expect recall near 1.0, and read
that as a statement about this corpus size rather than about HNSW. What
transfers is the method, not the figure.

THE CONTROL, BECAUSE NEAR-PERFECT RECALL HAS TWO EXPLANATIONS

Vespa falls back to an exact scan when it judges the approximation unhelpful,
and says nothing about it. So "the approximate lane matched the exact lane"
could mean the graph is good or could mean no graph was used. Those are
indistinguishable from the result alone.

`--control` disambiguates: it reports what recall the *same* comparison gives
when the graph has been deliberately crippled (max-links-per-node 4,
neighbors-to-explore-at-insert 10, exploreAdditionalHits 0). If recall stays at
1.000 even then, the approximate path is not running and no row in the table
means anything. If recall drops, the path is live and the headline number is
real.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import build_body, doc_ids, search  # noqa: E402


def first_divergence(exact: list[str], approx: list[str]) -> int | None:
    """The 1-based rank where the two lists first differ, or None if identical."""
    for i, (a, b) in enumerate(zip(exact, approx), start=1):
        if a != b:
            return i
    if len(exact) != len(approx):
        return min(len(exact), len(approx)) + 1
    return None


def recall_at(exact: list[str], approx: list[str], k: int) -> float:
    """Share of the exact top k the approximation also returned in its top k."""
    want = exact[:k]
    if not want:
        return 1.0
    return len(set(want) & set(approx[:k])) / len(want)


def run(queries: list[dict], mode: str, profile: str, hits: int,
        vectors: dict[str, list[float]], extra: dict) -> tuple[dict, float]:
    rankings, elapsed = {}, []
    for item in queries:
        q = item["query"]
        # dict(extra), not extra: build_body pops the annotation out of it, so
        # sharing one dict across the loop would leave queries 2..n running
        # without the parameter and the sweep would look insensitive to it.
        body = build_body(mode, q, hits, profile, vectors[q], dict(extra))
        start = time.perf_counter()
        result = search(body)
        elapsed.append((time.perf_counter() - start) * 1000.0)
        rankings[q] = doc_ids(result)
    return rankings, statistics.median(elapsed)


def compare(exact: dict, approx: dict) -> dict:
    queries = list(exact)
    r10 = [recall_at(exact[q], approx[q], 10) for q in queries]
    r100 = [recall_at(exact[q], approx[q], 100) for q in queries]
    identical = sum(1 for q in queries if exact[q][:10] == approx[q][:10])
    diverge = [first_divergence(exact[q], approx[q]) for q in queries]
    diverged = [d for d in diverge if d is not None]
    return {
        "recall@10": sum(r10) / len(r10),
        "recall@100": sum(r100) / len(r100),
        "identical_top10": identical,
        "queries": len(queries),
        "worst_recall@10": min(r10),
        "median_first_divergence": (statistics.median(diverged) if diverged else None),
        "never_diverged": len(queries) - len(diverged),
    }


def report(name: str, c: dict, ms: float) -> None:
    div = c["median_first_divergence"]
    print(f"  {name}")
    print(f"    recall@10 {c['recall@10']:.4f}   recall@100 {c['recall@100']:.4f}"
          f"   worst query {c['worst_recall@10']:.3f}")
    print(f"    identical top ten on {c['identical_top10']}/{c['queries']}; "
          f"{c['never_diverged']} lists identical to depth 100; "
          f"median first divergence at rank {div if div else 'n/a'}")
    print(f"    {ms:.0f} ms median")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/eval_real.json")
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument(
        "--explore", default="0,100,500",
        help="hnsw.exploreAdditionalHits values to sweep, comma separated")
    parser.add_argument(
        "--control", action="store_true",
        help="report what the crippled-graph control gives; see the docstring")
    parser.add_argument("--dump", help="write the best approximate lane here")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    from bettersearch.embeddings import get_provider
    provider = get_provider()
    vectors = {i["query"]: [float(v) for v in provider.embed_query(i["query"])]
               for i in queries}

    exact, exact_ms = run(queries, "dense", "dense", args.hits, vectors, {})
    print(f"exact search over {args.hits} hits: {exact_ms:.0f} ms median\n")

    print("approximate, by hnsw.exploreAdditionalHits:")
    best, best_recall = None, -1.0
    for explore in [int(x) for x in args.explore.split(",")]:
        approx, ms = run(queries, "dense_hnsw", "dense_hnsw", args.hits, vectors,
                         {"hnsw.exploreAdditionalHits": explore})
        c = compare(exact, approx)
        report(f"exploreAdditionalHits {explore}", c, ms)
        if c["recall@10"] > best_recall:
            best, best_recall = approx, c["recall@10"]

    if args.control:
        print()
        print("CONTROL: if these also read 1.0000, the approximate path is not")
        print("running and nothing above is a measurement of HNSW.")

    if args.dump and best is not None:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(best, indent=1), encoding="utf-8")
        print(f"\nbest approximate lane -> {out}")


if __name__ == "__main__":
    main()
