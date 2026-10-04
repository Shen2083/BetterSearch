#!/usr/bin/env python3
"""Is the problem retrieval, or ordering? Measure the ceiling, then try to reach it.

    python scripts/tune_reranking.py --index .bettersearch/real-books-base

Meaning-based search here scores nDCG@10 0.658. The question this answers is
where the remaining quality is: in records we never retrieve, or in records we
retrieve and rank badly.

It is asked because the engineering team is moving to Vespa, and Vespa's
contribution to *quality* - as opposed to serving - is its reranking phase. A
number here says whether that phase is worth building.

THE CEILING, MEASURED FIRST
---------------------------
Reordering only the candidates we already fetch, with every relevant one moved
to the top:

    depth   ours    oracle   headroom   judged coverage
       20   0.6582  0.9124     +0.254      100%
       50   0.6582  0.9703     +0.312     92.7%
      100   0.6582  0.9850     +0.327     98.8%
      200   0.6582  1.0000     +0.342      100%

**The retrieval is fine; the ordering is the problem.** 161 records across the
41 queries are judged relevant, retrieved, and sitting at ranks 11-20 - on page
two. For scale, +0.25 is three times what the bge-base upgrade bought (+0.079)
and nearly three times what LLM enrichment bought (+0.092).

WHAT THE DEPTH SWEEP SAYS
-------------------------
"Retrieve wide, rank deep" is what Vespa's phased ranking exists for, so it was
swept. It does not work. The gain decays monotonically with depth:

    rerank depth      20       50      100      200
    recall@10     +0.066   +0.047   +0.035   +0.010
    coverage        100%    92.7%    98.8%     100%
    ms/query       2,809    5,548   10,093   19,937

Seven times the latency for a seventh of the gain. **Rerank the twenty you
display, and stop there.** For a Vespa `rerank-count`, that is the number.

IT TOOK TWO WRONG ANSWERS TO GET THERE
--------------------------------------
Earlier versions of this docstring published both of them, so they are kept
here rather than quietly deleted - the way they were wrong is the transferable
part.

**First wrong answer.** Pooled from four lanes, the sweep read `+0.079 / +0.029
/ -0.007 / -0.062` with coverage falling to 59.5%. That is the pooling penalty:
reranking 200 candidates promotes records no judge ever saw, and every one
scores zero however good it is.

**Second wrong answer, and the larger error.** Re-run with `--judged-only`,
which discards unjudged candidates and pins coverage at 100%, it read `+0.079 /
+0.072 / +0.091 / +0.087` - apparently no decay at all, strongest at 100. That
was written up as the true shape the eval could not see. It was not. What
`--judged-only` discards is precisely the junk the reranker promoted and nobody
graded, so the arm is scored only on candidates where it was already known to
be doing well. It flattered the reranker roughly ninefold at depth 200:
`+0.087` against a judged truth of `+0.010`.

**Neither estimator is safe; only judgements are.** The fix was to pay for them.
`data/rankings/` holds the depth-200 reranked lane, plain and
exact-match-promoted; both were pooled and the 399 records they surfaced that no
judge had seen were graded for $0.06. 81 came back relevant, coverage at depth
200 went 59.5% -> 100%, and the number stopped moving.

So: a coverage *correction* is not the same as having the judgements. If you
reach for `--judged-only` to compare arms, read it as an upper bound on the arm
that surfaced the unjudged records, never as the answer.

WHY RECALL@10 IS THE HEADLINE
-----------------------------
nDCG@10 is barely sensitive to any of this, which is a trap of its own: 28 of
the 41 queries already have ten or more relevant records, so `IDCG@10` is
saturated and growing the pool cannot move the metric. Every nDCG figure in this
project survived re-pooling unchanged for that reason - including the ones that
*should* have moved.

recall@10 of judged-relevant records has no such protection, because its
denominator is every relevant record. It is the metric that noticed.

But it is not pooling-safe either, and this script once claimed it was. The
denominator is fixed; the **numerator** is not, since an unjudged record
entering the top 10 displaces a judged-relevant one. Coverage is therefore
reported beside every arm, and it is reported rather than asserted at depths
past 20 because a reranker there genuinely can reach records the pooled lanes
missed.

WHAT WOULD MAKE THIS A LIE
--------------------------
A reranker that lifts the mean while breaking exact-name lookup is not an
improvement - finding a named book is the thing library users do most. The five
known-item controls already score 1.000, so they can only move down. They are
reported per arm, and an arm that drops them is reported as rejected whatever
its mean does.

Forty-one queries is also few. A +0.02 result is noise; the weighted-BM25 sweep
below shows exactly how easy it is to fit it. Per-query output is always printed
so a mean moved by two queries is visible as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bettersearch.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank
from bettersearch.keyword import BM25Index
from bettersearch.types import Chunk

DEPTH = 20
CONTROL_PREFIX = "control"

#: Cross-encoders score a (query, record) pair directly rather than comparing
#: two independently-made vectors, which is why they can order better and why
#: they cost more: one model call per candidate per query, not one per query.
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}


def enriched_text(path: Path, titles: dict[str, str]) -> dict[str, str]:
    """The record as prose, from the enrichment store.

    Same shape `Enrichment.embed_text` builds for indexing. Reranking reads it
    at query time rather than embedding it, so this needs no re-ingest.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        out[e["item_id"]] = "\n".join(filter(None, [
            titles.get(e["item_id"], ""),
            " ".join(e.get("questions") or []),
            e.get("synopsis") or "",
            "Topics: " + ", ".join(e.get("topics") or []),
            "Related: " + ", ".join(e.get("entities") or []),
        ]))
    return out


def load_index(path: Path):
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    vectors = np.load(path.with_suffix(".npz"))["vectors"].astype(np.float32)
    chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
    return chunks, vectors, meta.get("model_id")


def candidates(chunks, vectors, queries, provider, depth, judged_only=False):
    """The top `depth` per query, once. Every arm reorders exactly this.

    `judged_only` drops candidates no judge ever saw, pinning judged coverage at
    100% at every depth. It was added to separate "does a reranker get worse
    with more candidates" from "is it punished for finding records the pool
    missed", which are otherwise confounded and point opposite ways.

    **It is a biased estimator, not a correction, and it over-reads the
    reranker.** What it discards is precisely the candidates the reranker
    promoted and nobody graded, so the arm is scored only where it was already
    known to be doing well. Measured against judgements that were then paid for,
    it overstated the gain at depth 200 roughly ninefold - +0.087 against
    +0.010. Treat it as an upper bound on whichever arm surfaced the unjudged
    records, and when the answer matters, pool that arm and judge it instead.
    """
    out = []
    for item in queries:
        scores = vectors @ provider.embed_query(item["query"]).astype(np.float32)
        top = list(np.argsort(-scores)[:depth])
        if judged_only:
            graded = set(item.get("grades") or {})
            top = [i for i in top if chunks[i].doc_id in graded]
        out.append({
            "query": item["query"],
            "note": item.get("note", ""),
            "relevant": item["relevant_doc_ids"],
            "graded": set(item.get("grades") or {}),
            "idx": top,
            "doc_ids": [chunks[i].doc_id for i in top],
            "cosine": np.array([scores[i] for i in top], dtype=np.float32),
        })
    return out


def minmax(a: np.ndarray) -> np.ndarray:
    span = float(a.max() - a.min())
    return (a - a.min()) / span if span > 1e-9 else np.zeros_like(a)


def measure(cands, order_fn, *, name: str) -> dict:
    """Score one arm. `order_fn(c) -> list[doc_id]`, a permutation of c['doc_ids']."""
    rows, elapsed = [], 0.0
    for c in cands:
        start = time.perf_counter()
        ranked = order_fn(c)
        elapsed += time.perf_counter() - start
        # The guard that makes this comparable: an arm may reorder, never
        # substitute. Anything else has changed the candidate set and its
        # coverage - and so its score - is not the same measurement.
        if set(ranked) != set(c["doc_ids"]):
            raise SystemExit(
                f"{name}: arm changed the candidate set on {c['query']!r}; "
                f"its score would not be comparable to the others"
            )
        rows.append({
            "query": c["query"],
            "note": c["note"],
            "ndcg@10": ndcg_at_k(ranked, c["relevant"], 10),
            "recall@5": recall_at_k(ranked, c["relevant"], 5),
            # The headline when comparing depths, and **only trustworthy while
            # judged coverage is 100%**. The denominator is fixed - the records
            # a judge saw - but the numerator is not safe: an unjudged record
            # promoted into the top 10 displaces a judged-relevant one and
            # recall falls, however good the newcomer was. I claimed this metric
            # was pooling-safe when proposing the sweep. It is not - and
            # --judged-only, which seemed to prove it, was itself biased the
            # other way. Only judging the candidates settled it.
            "recall@10": recall_at_k(ranked, c["relevant"], 10),
            "mrr@10": reciprocal_rank(ranked, c["relevant"], 10),
            "coverage": sum(1 for d in ranked[:10] if d in c["graded"]) / 10,
            "ranking": ranked,
        })
    controls = [r for r in rows if r["note"].startswith(CONTROL_PREFIX)]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "name": name,
        "ndcg@10": mean("ndcg@10"),
        "recall@5": mean("recall@5"),
        "recall@10": mean("recall@10"),
        "mrr@10": mean("mrr@10"),
        "coverage": mean("coverage"),
        "controls": float(np.mean([r["ndcg@10"] for r in controls])) if controls else 0.0,
        "control_count": len(controls),
        "ms_per_query": 1000 * elapsed / (len(rows) or 1),
        "rows": rows,
    }


def _ranking(arm: dict, query: str) -> list[str]:
    return next(r["ranking"] for r in arm["rows"] if r["query"] == query)


def promote_named(c, ranked: list[str], catalogue: dict[str, dict]) -> list[str]:
    """Lift records the reader arguably named, by rule rather than by score."""
    from bettersearch import exact
    sub = {d: catalogue[d] for d in c["doc_ids"] if d in catalogue}
    rows = [(catalogue[d], 0.0) for d in ranked if d in catalogue]
    out, _ = exact.promote(rows, exact.find(c["query"], sub.values()), sub)
    moved = [r[0]["doc_id"] for r in out]
    # measure() requires a permutation; records absent from the catalogue file
    # keep their place at the end rather than vanishing.
    return moved + [d for d in ranked if d not in catalogue]


def oracle_order(c):
    rel = set(c["relevant"])
    return [d for d in c["doc_ids"] if d in rel] + [d for d in c["doc_ids"] if d not in rel]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".bettersearch/real-books-base")
    ap.add_argument("--queries", default="data/eval_real.json")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--depths", default="20,50,100,200",
                    help="rerank depths to sweep; Vespa's rerank-count")
    ap.add_argument("--rerankers", default="minilm,bge",
                    help="comma-separated keys from RERANKERS, or 'none'")
    ap.add_argument("--catalogue", default="data/catalogue_real.json",
                    help="records with title/author, for exact-match promotion")
    ap.add_argument("--enrichment", default="data/enrichment_real_haiku.jsonl",
                    help="records as prose; reranking reads it, so no re-ingest")
    ap.add_argument("--judged-only", action="store_true",
                    help="keep only judged candidates, pinning coverage at "
                         "100%. Separates depth from pooling bias, but is "
                         "itself biased towards whichever arm found the "
                         "unjudged records - read it as an upper bound")
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--dump", help="write each arm's rankings here, for score_rankings.py")
    args = ap.parse_args()

    os.environ.setdefault("BETTERSEARCH_LOCAL_MODEL", args.model)
    from bettersearch.embeddings import get_provider

    chunks, vectors, model_id = load_index(Path(args.index))
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    print(f"{len(chunks)} records · {model_id} · {len(queries)} judged queries")

    provider = get_provider()
    catalogue = {}
    if args.catalogue and Path(args.catalogue).exists():
        catalogue = {r["doc_id"]: r
                     for r in json.loads(Path(args.catalogue).read_text())["documents"]}
    titles = {c.doc_id: c.title for c in chunks}
    corpora = {"thin": {c.doc_id: f"{c.title}\n\n{c.text}" for c in chunks}}
    if args.enrichment and Path(args.enrichment).exists():
        rich = enriched_text(Path(args.enrichment), titles)
        corpora["enriched"] = {d: rich.get(d, corpora["thin"][d]) for d in titles}
        words = lambda c: int(np.median([len(t.split()) for t in c.values()]))
        print(f"  record text: thin {words(corpora['thin'])} words median, "
              f"enriched {words(corpora['enriched'])}")

    depths = [int(d) for d in args.depths.split(",")]
    bm25 = BM25Index(chunks)

    # Loaded once for the whole sweep. Reloading per depth would put model
    # construction into the latency column and make deep reranking look worse
    # than it is.
    models = {}
    if args.rerankers != "none":
        from sentence_transformers import CrossEncoder
        for key in args.rerankers.split(","):
            repo = RERANKERS[key.strip()]
            print(f"  loading {repo} …", flush=True)
            models[repo] = CrossEncoder(repo, max_length=512)

    everything = {}
    for depth in depths:
        print(f"\n{'=' * 96}\nrerank depth {depth}\n{'=' * 96}", flush=True)
        cands = candidates(chunks, vectors, queries, provider, depth,
                           judged_only=args.judged_only)
        for c in cands:
            hits = {h.chunk.doc_id: h.score
                    for h in bm25.search(c["query"], top_k=len(chunks))}
            c["bm25"] = np.array([hits.get(d, 0.0) for d in c["doc_ids"]],
                                 dtype=np.float32)

        arms = [measure(cands, lambda c: list(c["doc_ids"]),
                        name="baseline (no reranking)")]

        best_w, best_arm = None, None
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            arm = measure(
                cands,
                lambda c, w=w: [c["doc_ids"][i] for i in
                                np.argsort(-((1 - w) * minmax(c["cosine"])
                                             + w * minmax(c["bm25"])))],
                name=f"+ BM25 second phase (w={w})")
            if best_arm is None or arm["recall@10"] > best_arm["recall@10"]:
                best_w, best_arm = w, arm
        # Reported as fitted: w was chosen on the same 41 queries it is scored
        # on, so this is an upper bound on the idea, not an estimate of it.
        best_arm["name"] = f"+ BM25 second phase (w={best_w}, fitted)"
        arms.append(best_arm)

        for repo, model in models.items():
            def order(c, model=model, corpus=None):
                # Timed with the model call inside, because the model is the
                # cost. Vespa pays it on every search, at this depth.
                pairs = [(c["query"], corpus[d]) for d in c["doc_ids"]]
                scores = np.asarray(model.predict(pairs, show_progress_bar=False))
                return [c["doc_ids"][i] for i in np.argsort(-scores)]

            short = repo.split("/")[-1]
            for label, corpus in corpora.items():
                # Thin records were already measured at nothing (+0.005 and
                # -0.001 at depth 20). Carried at the shallowest depth only, to
                # tie this sweep to that result, and not re-paid at every depth.
                if label == "thin" and depth != depths[0]:
                    continue
                arm = measure(cands, lambda c, k=corpus: order(c, corpus=k),
                              name=f"+ {short} over {label} records")
                arms.append(arm)
                if catalogue and arm["controls"] < 0.999:
                    arms.append(measure(
                        cands,
                        lambda c, a=arm: promote_named(c, _ranking(a, c["query"]),
                                                       catalogue),
                        name=f"  … {short}/{label} + exact-match promotion"))

        arms.append(measure(cands, oracle_order, name="oracle (the ceiling)"))
        everything[depth] = arms

        base = arms[0]
        print(f"\n  {'arm':<48} {'R@10':>7} {'vs base':>8} {'nDCG@10':>8} "
              f"{'ctrl':>6} {'cover':>6} {'ms/q':>8}")
        for a in arms:
            flag = "" if a["controls"] >= 0.999 else "  ← CONTROLS REGRESSED"
            print(f"  {a['name'][:48]:<48} {a['recall@10']:>7.4f} "
                  f"{a['recall@10'] - base['recall@10']:>+8.4f} {a['ndcg@10']:>8.4f} "
                  f"{a['controls']:>6.3f} {a['coverage']:>6.1%} "
                  f"{a['ms_per_query']:>8.1f}{flag}")
        # Coverage is reported, not asserted. At depth 20 every candidate was
        # judged, so it stays at 100% and nDCG is comparable. Deeper, a
        # reranker pulls records the four pooled lanes never surfaced into the
        # top 10, coverage falls, and **nDCG stops being comparable** - the
        # unjudged ones score zero however good they are. That is the whole
        # reason recall@10 of judged-relevant is the headline instead.
        #
        # The invariant that *is* enforced lives in measure(): an arm may
        # reorder the candidates, never substitute them.
        worst = min(a["coverage"] for a in arms)
        if worst < 0.999:
            print(f"\n  judged coverage falls to {worst:.1%} at this depth - "
                  f"the nDCG column is NOT comparable across depths.\n"
                  f"  Read recall@10, whose denominator is fixed.")
        else:
            print("\n  judged coverage 100% on every arm - nDCG is comparable here.")

    # ---------------------------------------------------------------- summary
    print(f"\n{'=' * 96}\nrecall@10 of judged-relevant records, by rerank depth")
    print("Comparable ONLY between arms whose judged coverage is 100%. The")
    print("denominator is fixed, but an unjudged record promoted into the top 10")
    print("displaces a judged-relevant one, so a system that finds something the")
    print("pool missed is penalised for it. --judged-only does NOT fix that; it")
    print("reverses the bias. Pool the arm and judge what it found instead:")
    print("build_eval_set.py --lane-file, which judges only what is new.")
    print(f"{'=' * 96}\n")
    names = []
    for arms in everything.values():
        for a in arms:
            if a["name"] not in names:
                names.append(a["name"])
    print(f"  {'arm':<48} " + " ".join(f"{d:>9}" for d in depths))
    for n in names:
        cells = []
        for d in depths:
            a = next((x for x in everything[d] if x["name"] == n), None)
            cells.append(f"{a['recall@10']:>9.4f}" if a else f"{'-':>9}")
        print(f"  {n[:48]:<48} " + " ".join(cells))
    print(f"\n  {'latency, ms/query':<48} " +
          " ".join(f"{max(x['ms_per_query'] for x in everything[d]):>9.0f}"
                   for d in depths))

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        for depth, arms in everything.items():
            for a in arms:
                slug = "".join(ch if ch.isalnum() else "-" for ch in a["name"])[:50]
                (out / f"d{depth}-{slug}.json").write_text(json.dumps(
                    {r["query"]: r["ranking"] for r in a["rows"]}, indent=1))
        print(f"\n  rankings written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
