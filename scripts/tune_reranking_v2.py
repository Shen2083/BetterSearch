#!/usr/bin/env python3
"""Second pass at reranking: how much more of the ceiling can be reached?

    python scripts/tune_reranking_v2.py --index .bettersearch/real-books-base

`tune_reranking.py` established the shape of the problem and the best arm so
far: baseline nDCG@10 0.6582, oracle reordering of the same 20 candidates
0.9207, and bge-reranker-base over enriched records plus exact-match promotion
at 0.7340 - 29% of the headroom, with 0.19 unexplained.

This script keeps that harness (same candidates, same depth, same
reorder-never-substitute guard, same control tripwire) and asks what is left.

WHAT IT FOUND
-------------
    baseline (as shipped)                            0.6582   ctrl 1.000
    best arm from the first pass                     0.7340   ctrl 1.000
    + ms-marco-MiniLM-L-12-v2 over enriched          0.7127   ctrl 0.871
    + bge-reranker-base over both                    0.7352   ctrl 1.000
    + bge-reranker-large over asked                  0.7626   ctrl 1.000
    + bge-reranker-large over enriched               0.7610   ctrl 1.000
    + bge-reranker-large over both                   0.7761   ctrl 1.000
      ... ensembled with bge-reranker-base           0.7822   ctrl 1.000
    oracle                                           0.9207   ctrl 1.000

Three findings, in order of how much they change what to do next.

**The identity regression is an input problem, not a scoring problem.** Every
enriched arm in the first pass broke the known-item controls, and exact.py
repaired it by rule. Concatenate the thin record in front of the enriched text
- the `both` view - and the controls hold at 1.000 *by construction*, because
the title the reader typed is in what the model reads. It also scores higher
than the rule did, on every model tried.

**Capacity helps, but model size is not what capacity means.** MiniLM L-6 to
L-12 is worth -0.002, and on thin records the bigger one is worse.
bge-reranker-large is worth +0.042 over the first pass's best arm, and that
one is distinguishable from noise on a paired bootstrap (p = 0.009).

**It is the enrichment's questions that earn the reranking gain, not its
synopsis.** The `asked` view - questions and topics, no synopsis - scores
0.7626 against 0.7610 for the full enriched text. Synopsis length is the
primary cost lever on the enrichment bill, so the cheap field is the one doing
the work.

What did not work: keeping the retriever's cosine in a blend (best weight is
1.0 on every strong arm), reciprocal rank fusion instead of score fusion
(0.7223 against 0.7610 on the same two signals), BM25 as a third fused lane
(0.7197), max(thin, enriched) instead of concatenation (0.7409 against
0.7761), and ensembling two cross-encoders (+0.006, inside the noise).

THE FOUR RECORD VIEWS
---------------------
    thin      the record as it ships, 24 words median
    asked     thin + the enrichment's questions and topics, 77 words
    enriched  the prose the first pass used, 95 words
    both      thin + the full enriched text, 120 words

HOW SWEPT WEIGHTS ARE REPORTED
------------------------------
Every blend here has a weight, and 41 queries will fit one happily - the first
script's BM25 sweep is the cautionary example, reported as an upper bound
because `w` was chosen on the same queries it scored on. So each swept arm is
reported twice:

    fitted      the best w on all 41 queries - an upper bound on the idea
    LOO-CV      w chosen on the other 40 queries, scored on the held-out one,
                41 times - an honest estimate of what the idea is worth

Where those two numbers differ a lot, the gain was the sweep rather than the
idea. The best arm here needs no weight at all.

WHAT IS STILL NOT REACHED
-------------------------
0.1445 of nDCG over 27 of the 41 queries, and reading the worst of them says
the rest is not an ordering problem. `something short I can finish in one
sitting` needs a page count the catalogue does not hold. `gripping but not too
violent` and `an uplifting story after a hard year` are constraints to satisfy
rather than topics to match, and a cross-encoder trained on passage relevance
scores resemblance. A listwise LLM reranker over the same 20 candidates is the
obvious untested arm; it is untested because this container has no LLM
credentials, not because it was judged unpromising.

COST
----
bge-reranker-large is 8.5 seconds per query on CPU for 20 candidates, four
times the base model and unservable as it stands. Nothing here is a shipping
recommendation; it is a measurement of what the ordering is worth.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bettersearch.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank
from bettersearch.keyword import BM25Index
from bettersearch.types import Chunk

from tune_reranking import (  # the first pass's harness, reused verbatim
    CONTROL_PREFIX,
    candidates,
    enriched_text,
    load_index,
    measure,
    minmax,
    oracle_order,
    promote_named,
)

DEPTH = 20

#: Cross-encoders, smallest first. `bge` is the first script's best; `minilm12`
#: and `bge-large` are the capacity question - the same architectures with more
#: layers, nothing else changed.
MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge": "BAAI/bge-reranker-base",
    "bge-large": "BAAI/bge-reranker-large",
}

#: Weight grids. Coarse on purpose: a finer grid buys only a better fit.
W_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def zscore(a: np.ndarray) -> np.ndarray:
    sd = float(a.std())
    return (a - float(a.mean())) / sd if sd > 1e-9 else np.zeros_like(a)


def rrf(ranks: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal rank fusion contribution for 0-based ranks."""
    return 1.0 / (k + 1.0 + ranks)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each position under a descending sort of `scores`."""
    order = np.argsort(-scores)
    out = np.empty(len(scores), dtype=np.float32)
    out[order] = np.arange(len(scores), dtype=np.float32)
    return out


def views(chunks, enrichment: Path | None, titles: dict[str, str]) -> dict[str, dict[str, str]]:
    """The several texts a reranker could be shown for the same record."""
    thin = {c.doc_id: f"{c.title}\n\n{c.text}" for c in chunks}
    out = {"thin": thin}
    if enrichment and enrichment.exists():
        rich = enriched_text(enrichment, titles)
        # Questions and topics only, no synopsis. The enriched view's gain could
        # be the questions - which read like the queries people type - or the
        # synopsis, and the losses look like prose generalising a specific
        # record away. Dropping the synopsis separates the two.
        out["asked"] = {}
        for line in enrichment.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            out["asked"][e["item_id"]] = "\n".join(filter(None, [
                thin.get(e["item_id"], titles.get(e["item_id"], "")),
                " ".join(e.get("questions") or []),
                "Topics: " + ", ".join(e.get("topics") or []),
            ]))
        out["asked"] = {d: out["asked"].get(d, thin[d]) for d in thin}
        out["enriched"] = {d: rich.get(d, thin[d]) for d in thin}
        # Identity first, then the invented prose. The whole recorded failure of
        # the enriched arms is that the title someone typed is buried; this view
        # puts the real record back at the front of what the model reads.
        out["both"] = {d: f"{thin[d]}\n\n{out['enriched'][d]}" for d in thin}
    return out


def ce_scores(cands, corpus: dict[str, str], repo: str, cache: Path) -> np.ndarray:
    """A (queries x depth) matrix of cross-encoder scores, cached on disk.

    Cached because every arm below is a different combination of the same
    matrices, and re-running the models to try a new weight would make the
    experiment cost the models rather than the ideas.
    """
    if cache.exists():
        return np.load(cache)["scores"]
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(repo, max_length=512)
    rows, elapsed = [], 0.0
    for c in cands:
        pairs = [(c["query"], corpus[d]) for d in c["doc_ids"]]
        start = time.perf_counter()
        rows.append(np.asarray(model.predict(pairs, show_progress_bar=False), dtype=np.float32))
        elapsed += time.perf_counter() - start
    del model
    scores = np.vstack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, scores=scores, ms_per_query=1000 * elapsed / len(cands))
    print(f"    {repo.split('/')[-1]} over this view: "
          f"{1000 * elapsed / len(cands):.0f} ms/query", flush=True)
    return scores


def order_by(cands, matrix: np.ndarray):
    """An order_fn that sorts each query's candidates by a precomputed row."""
    def fn(c):
        i = next(j for j, x in enumerate(cands) if x["query"] == c["query"])
        return [c["doc_ids"][k] for k in np.argsort(-matrix[i])]
    return fn


def blended(cands, parts: dict[str, np.ndarray], weights: dict[str, float],
            *, normalise=minmax) -> np.ndarray:
    """One (queries x depth) matrix: a weighted sum of normalised signals."""
    out = np.zeros_like(next(iter(parts.values())))
    for name, w in weights.items():
        if w:
            out += w * np.vstack([normalise(row) for row in parts[name]])
    return out


def sweep(cands, parts: dict[str, np.ndarray], keys: tuple[str, ...], *,
          name: str, normalise=minmax, grid=W_GRID) -> tuple[dict, dict]:
    """Fit a one-parameter blend of two signals, and cross-validate it.

    Returns (fitted arm, loo arm). `fitted` picks w on all 41 queries, which
    overstates. `loo` picks w on the other 40 and scores the held-out query,
    which does not.
    """
    a, b = keys
    per_w = {}
    for w in grid:
        matrix = blended(cands, parts, {a: 1 - w, b: w}, normalise=normalise)
        per_w[w] = measure(cands, order_by(cands, matrix), name=f"{name} w={w}")

    best_w = max(grid, key=lambda w: per_w[w]["ndcg@10"])
    fitted = dict(per_w[best_w])
    fitted["name"] = f"{name} (w={best_w}, fitted)"

    # Leave-one-query-out: for each query, choose w on the other 40, then take
    # that query's row from the arm scored with that w.
    rows = []
    picks = []
    for i in range(len(cands)):
        held = max(grid, key=lambda w: float(np.mean(
            [r["ndcg@10"] for j, r in enumerate(per_w[w]["rows"]) if j != i])))
        picks.append(held)
        rows.append(per_w[held]["rows"][i])
    loo = _summarise(rows, name=f"{name} (LOO-CV, w in {sorted(set(picks))})")
    loo["ms_per_query"] = fitted["ms_per_query"]
    return fitted, loo


def _summarise(rows: list[dict], *, name: str) -> dict:
    """Re-aggregate per-query rows into the same shape `measure` returns."""
    controls = [r for r in rows if r["note"].startswith(CONTROL_PREFIX)]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "name": name,
        "ndcg@10": mean("ndcg@10"),
        "recall@5": mean("recall@5"),
        "mrr@10": mean("mrr@10"),
        "coverage": mean("coverage"),
        "controls": float(np.mean([r["ndcg@10"] for r in controls])) if controls else 0.0,
        "control_count": len(controls),
        "ms_per_query": 0.0,
        "rows": rows,
    }


def with_promotion(cands, arm: dict, catalogue: dict[str, dict]) -> dict:
    """The same arm with exact-match promotion on top."""
    by_query = {r["query"]: r["ranking"] for r in arm["rows"]}
    return measure(cands,
                   lambda c: promote_named(c, by_query[c["query"]], catalogue),
                   name=f"  … {arm['name'].strip()[:40]} + exact promotion")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".bettersearch/real-books-base")
    ap.add_argument("--queries", default="data/eval_real.json")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--models", default="minilm,minilm12,bge",
                    help=f"comma-separated keys from {sorted(MODELS)}")
    ap.add_argument("--catalogue", default="data/catalogue_real.json")
    ap.add_argument("--enrichment", default="data/enrichment_real_haiku.jsonl")
    ap.add_argument("--cache", default=".bettersearch/rerank-cache",
                    help="where cross-encoder score matrices are kept")
    ap.add_argument("--dump", help="write each arm's rankings here, for score_rankings.py")
    args = ap.parse_args()

    os.environ.setdefault("BETTERSEARCH_LOCAL_MODEL", args.model)
    from bettersearch.embeddings import get_provider

    chunks, vectors, model_id = load_index(Path(args.index))
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    print(f"{len(chunks)} records · {model_id} · {len(queries)} judged queries · "
          f"reranking the top {args.depth}\n")

    provider = get_provider()
    cands = candidates(chunks, vectors, queries, provider, args.depth)
    titles = {c.doc_id: c.title for c in chunks}
    corpora = views(chunks, Path(args.enrichment) if args.enrichment else None, titles)
    words = lambda c: int(np.median([len(t.split()) for t in c.values()]))
    print("  record views: " + ", ".join(
        f"{k} {words(v)} words median" for k, v in corpora.items()) + "\n")

    catalogue = {}
    if args.catalogue and Path(args.catalogue).exists():
        catalogue = {r["doc_id"]: r
                     for r in json.loads(Path(args.catalogue).read_text())["documents"]}

    # ---- the signals, each a (queries x depth) matrix ----------------------
    parts: dict[str, np.ndarray] = {
        "cosine": np.vstack([c["cosine"] for c in cands]),
    }
    bm25 = BM25Index(chunks)
    bm25_rows = []
    for c in cands:
        hits = {h.chunk.doc_id: h.score for h in bm25.search(c["query"], top_k=len(chunks))}
        bm25_rows.append(np.array([hits.get(d, 0.0) for d in c["doc_ids"]], dtype=np.float32))
    parts["bm25"] = np.vstack(bm25_rows)

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    cache_dir = Path(args.cache)
    for key in keys:
        repo = MODELS[key]
        for view in corpora:
            print(f"  scoring {key} over {view} records …", flush=True)
            parts[f"{key}:{view}"] = ce_scores(
                cands, corpora[view], repo, cache_dir / f"{key}-{view}.npz")

    # ---- the arms ----------------------------------------------------------
    arms = [measure(cands, lambda c: list(c["doc_ids"]), name="baseline (as shipped)")]
    swept: list[tuple[dict, dict]] = []

    # Reproduce the first pass's best arm, so the two scripts are comparable.
    for key in keys:
        for view in ("thin", "enriched", "both", "asked"):
            if f"{key}:{view}" not in parts:
                continue
            arm = measure(cands, order_by(cands, parts[f"{key}:{view}"]),
                          name=f"+ {key} over {view} records")
            arms.append(arm)
            if catalogue and arm["controls"] < 0.999:
                arms.append(with_promotion(cands, arm, catalogue))

    # Reading the record twice: max of the thin and enriched scores.
    for key in keys:
        if f"{key}:enriched" not in parts:
            continue
        both = np.maximum(parts[f"{key}:thin"], parts[f"{key}:enriched"])
        arm = measure(cands, order_by(cands, both),
                      name=f"+ {key}, max(thin, enriched) score")
        arms.append(arm)
        if catalogue and arm["controls"] < 0.999:
            arms.append(with_promotion(cands, arm, catalogue))
        swept.append(sweep(cands, parts, (f"{key}:thin", f"{key}:enriched"),
                           name=f"+ {key}, thin/enriched blend"))

    # Keeping the retriever's opinion: cosine blended with the reranker.
    for key in keys:
        for view in ("enriched", "both", "asked"):
            if f"{key}:{view}" not in parts:
                continue
            swept.append(sweep(cands, parts, ("cosine", f"{key}:{view}"),
                               name=f"+ cosine/{key}-{view} blend"))

    # Ensembling the two rerankers, z-scored so one model's wider spread does
    # not silently become its weight.
    if len(keys) > 1:
        for a, b in itertools.combinations(keys, 2):
            for view in ("enriched", "both", "asked"):
                if f"{a}:{view}" not in parts or f"{b}:{view}" not in parts:
                    continue
                swept.append(sweep(cands, parts, (f"{a}:{view}", f"{b}:{view}"),
                                   name=f"+ {a}+{b} over {view}", normalise=zscore))

    # Rank fusion rather than score fusion.
    def fuse(names: list[str]) -> np.ndarray:
        return sum(np.vstack([rrf(ranks_of(row)) for row in parts[n]]) for n in names)

    for key in keys:
        combos = {
            f"+ RRF(cosine, {key}-enriched)": ["cosine", f"{key}:enriched"],
            f"+ RRF(cosine, bm25, {key}-enriched)": ["cosine", "bm25", f"{key}:enriched"],
            f"+ RRF(cosine, {key}-thin, {key}-enriched)":
                ["cosine", f"{key}:thin", f"{key}:enriched"],
        }
        for name, names in combos.items():
            if any(n not in parts for n in names):
                continue
            arm = measure(cands, order_by(cands, fuse(names)), name=name)
            arms.append(arm)
            if catalogue and arm["controls"] < 0.999:
                arms.append(with_promotion(cands, arm, catalogue))

    # Promotion on top of the swept arms too, since that is where the controls
    # are most likely to have moved.
    for fitted, loo in swept:
        arms.append(fitted)
        arms.append(loo)
        if catalogue and fitted["controls"] < 0.999:
            arms.append(with_promotion(cands, fitted, catalogue))

    arms.append(measure(cands, oracle_order, name="oracle (the ceiling)"))

    # ---- the table ---------------------------------------------------------
    base = arms[0]["ndcg@10"]
    ceiling = arms[-1]["ndcg@10"]
    print(f"\n  {'arm':<58} {'nDCG@10':>8} {'vs base':>8} {'R@5':>6} "
          f"{'MRR':>6} {'ctrl':>6} {'cover':>6} {'head%':>6}")
    for a in arms:
        flag = "" if a["controls"] >= 0.999 else "  ← CONTROLS REGRESSED"
        share = (a["ndcg@10"] - base) / (ceiling - base) if ceiling > base else 0.0
        print(f"  {a['name'][:58]:<58} {a['ndcg@10']:>8.4f} "
              f"{a['ndcg@10'] - base:>+8.4f} {a['recall@5']:>6.3f} {a['mrr@10']:>6.3f} "
              f"{a['controls']:>6.3f} {a['coverage']:>6.1%} {share:>5.0%}{flag}")

    for a in arms:
        if abs(a["coverage"] - 1.0) > 1e-9:
            raise SystemExit(
                f"{a['name']}: judged coverage is {a['coverage']:.1%}, not 100%. "
                f"Reranking cannot change coverage, so this is a bug in the arm.")
    print("\n  judged coverage is 100% on every arm, as reranking requires.")

    keep = [a for a in arms[1:-1] if a["controls"] >= 0.999]
    if keep:
        winner = max(keep, key=lambda a: a["ndcg@10"])
        print(f"\n  best arm that keeps the controls at 1.000: {winner['name']}")
        print(f"  {winner['ndcg@10']:.4f} against {base:.4f} shipped and "
              f"{ceiling:.4f} oracle - "
              f"{(winner['ndcg@10'] - base) / (ceiling - base):.0%} of the headroom")
        deltas = sorted(((a["ndcg@10"] - b["ndcg@10"], a["query"])
                         for a, b in zip(winner["rows"], arms[0]["rows"])), reverse=True)
        print("  biggest gains:")
        for d, q in deltas[:8]:
            if d > 0:
                print(f"    {d:>+7.3f}  {q[:58]}")
        losses = [x for x in deltas if x[0] < 0]
        print(f"  biggest losses ({len(losses)} queries got worse):")
        for d, q in losses[-8:]:
            print(f"    {d:>+7.3f}  {q[:58]}")

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        for a in arms:
            slug = "".join(ch if ch.isalnum() else "-" for ch in a["name"].strip())[:70]
            (out / f"{slug}.json").write_text(json.dumps(
                {r["query"]: r["ranking"] for r in a["rows"]}, indent=1))
        print(f"\n  rankings written to {out}/ - check them with scripts/score_rankings.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
