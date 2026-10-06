"""Sweep the fielded BM25 weights, and report them as fitted.

`src/bettersearch/keyword.py:35-38` records that our BM25 cannot weight the
author or the subject headings at all, because neither is a field: both exist
only as lines of prose inside one `text` blob. Vespa indexes them separately,
so this is the first time the question can be asked of this corpus.

Two cautions, both load-bearing:

* Equal weights are not a neutral starting point, they are a bad one. The
  `text` blob already contains the author line and the subject headings, so
  bm25(author) and bm25(subjects) double and triple count terms that bm25(text)
  has already scored. Fielding can only help once the weights are tuned.

* These weights are fitted on the same 41 queries they are scored on, exactly
  as `scripts/tune_reranking.py` reports its own BM25 second-phase sweep. The
  best row is an upper bound on what fielding is worth, not an estimate of what
  it would be worth on queries nobody has seen. `docs/VESPA-HANDOVER.md` flags
  this trap for alpha-weighted hybrid tuning; it applies here unchanged.

Judged coverage is printed for every row because these lanes were never pooled.
A row scoring higher with lower coverage has not necessarily done better.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from score_rankings import score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import build_body, doc_ids, search  # noqa: E402


def run(queries: list[dict], weights: dict[str, float], hits: int) -> dict[str, list[str]]:
    rankings = {}
    for item in queries:
        body = build_body("bm25", item["query"], hits, "bm25_fielded", None,
                          {"model.defaultIndex": "fielded",
                           **{f"input.query({k})": str(v) for k, v in weights.items()}})
        rankings[item["query"]] = doc_ids(search(body))
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/eval_real.json")
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--dump", help="write the best lane here")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    grid = {
        "w_title": [1.0, 2.0, 3.0],
        "w_author": [0.0, 0.5, 1.0],
        "w_subjects": [0.0, 0.5, 1.0],
        "w_text": [1.0],
    }
    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[n] for n in names)):
        weights = dict(zip(names, combo))
        s = score(run(queries, weights, args.hits), queries)
        rows.append((weights, s))
        print(f"  title {weights['w_title']:.1f}  author {weights['w_author']:.1f}  "
              f"subjects {weights['w_subjects']:.1f}  ->  "
              f"nDCG@10 {s['ndcg@10']:.4f}  controls {s['controls']:.4f}  "
              f"coverage {s['coverage']:.1%}")

    rows.sort(key=lambda r: -r[1]["ndcg@10"])
    best, best_score = rows[0]
    print()
    print("best fitted row (not an out-of-sample estimate):")
    print(f"  {best}")
    print(f"  nDCG@10 {best_score['ndcg@10']:.4f}  recall@5 {best_score['recall@5']:.4f}  "
          f"MRR@10 {best_score['mrr@10']:.4f}  controls {best_score['controls']:.4f}  "
          f"coverage {best_score['coverage']:.1%}")

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run(queries, best, args.hits), indent=1), encoding="utf-8")
        print(f"  best lane -> {out}")


if __name__ == "__main__":
    main()
