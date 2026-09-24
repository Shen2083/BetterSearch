#!/usr/bin/env python3
"""Choose how much a title match is worth, using the judgements.

    python scripts/tune_title_weight.py --index .bettersearch/real

`src/bettersearch/keyword.py` used to score a record as one flat bag of words,
so a query for `Harry Potter` put *Woodworking* by Harry Zarchy above anything
about wizards - the author's first name counted exactly as much as a title. The
fix is the BM25F idea: weight the title. The question this answers is how much.

WHY THE NUMBERS BELOW NEED READING CAREFULLY
--------------------------------------------
The keyword lane is **one of the four lanes that built the judged pool**. Change
what it retrieves and the records it newly finds were never shown to a judge -
and unjudged scores as irrelevant. That is exactly the bias that made bge-base
look worse than MiniLM until the pool was rebuilt, and it was worth 0.080 nDCG
there.

So every row reports **pool coverage**: the share of the lane's top 10 that the
judge actually saw. If coverage falls as the weight rises, the score is partly
measuring the pool rather than the change, and a drop in nDCG is not evidence of
harm. Read the two columns together or not at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bettersearch.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from bettersearch.keyword import BM25Index  # noqa: E402
from bettersearch.types import Chunk  # noqa: E402

WEIGHTS = (1.0, 2.0, 3.0, 5.0, 8.0)

#: Exact-name lookups - the queries this change is meant to help. A reader
#: typing an author's name wants that author, and nothing else is a success.
CONTROLS = ("Joy of Cooking", "Agatha Christie", "The Secret Life of Bees",
            "Betty Crocker", "Sue Monk Kidd")


def load_chunks(index_path: Path) -> list[Chunk]:
    meta = json.loads(Path(f"{index_path}.json").read_text(encoding="utf-8"))
    return [Chunk.from_dict(c) for c in meta["chunks"]]


def ranked_docs(index: BM25Index, query: str, top_k: int) -> list[str]:
    seen: list[str] = []
    for hit in index.search(query, top_k=top_k):
        if hit.chunk.doc_id not in seen:
            seen.append(hit.chunk.doc_id)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=ROOT / ".bettersearch/real")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real.json")
    args = ap.parse_args()

    raw = json.loads(args.queries.read_text(encoding="utf-8"))
    items = raw["queries"] if isinstance(raw, dict) else raw
    relevant = {q["query"]: tuple(q["relevant_doc_ids"]) for q in items}
    judged = {q["query"]: set(q.get("grades", {})) for q in items}

    chunks = load_chunks(args.index)
    print(f"\n{len(items)} judged queries · {len(chunks)} chunks · "
          f"{args.index.name}\n")

    header = (f"{'title weight':<14}{'Recall@5':>10}{'MRR@10':>9}{'nDCG@10':>9}"
              f"{'pool seen':>11}")
    print(header)
    print("-" * len(header))

    control_rows: dict[float, list[float]] = {}
    for weight in WEIGHTS:
        index = BM25Index(chunks, title_weight=weight)
        recalls, reciprocals, gains, coverage = [], [], [], []
        for item in items:
            query = item["query"]
            ranked = ranked_docs(index, query, 10)
            recalls.append(recall_at_k(ranked, relevant[query], 5))
            reciprocals.append(reciprocal_rank(ranked, relevant[query], 10))
            gains.append(ndcg_at_k(ranked, relevant[query], 10))
            if ranked:
                coverage.append(sum(1 for d in ranked if d in judged[query]) / len(ranked))
        control_rows[weight] = [
            ndcg_at_k(ranked_docs(index, c, 10), relevant[c], 10)
            for c in CONTROLS if c in relevant
        ]
        mark = "   <- today" if weight == 1.0 else ""
        print(f"{weight:<14.0f}{statistics.mean(recalls):>10.3f}"
              f"{statistics.mean(reciprocals):>9.3f}{statistics.mean(gains):>9.3f}"
              f"{statistics.mean(coverage):>10.0%}{mark}")

    print(f"\nthe five exact-name lookups, nDCG@10 each "
          f"({', '.join(c[:18] for c in CONTROLS)}):\n")
    for weight in WEIGHTS:
        scores = control_rows[weight]
        print(f"  weight {weight:<4.0f} mean {statistics.mean(scores):.3f}   "
              + "  ".join(f"{s:.2f}" for s in scores))

    print("\nRead the nDCG column against pool coverage. If coverage drops as")
    print("the weight rises, the lane is finding records nobody judged and the")
    print("score is measuring the pool, not the change.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
