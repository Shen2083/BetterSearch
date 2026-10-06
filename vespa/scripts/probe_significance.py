"""Test the per-node IDF warning in docs/VESPA-HANDOVER.md.

The warning says: Vespa builds BM25 significance per content node from the
documents that node happens to hold, ours is global over the whole corpus, so a
multi-node port would miss the published 0.443 bar for reasons unrelated to the
port. It was never tested. This tests it.

Not with a significance model file. The Vespa CLI that matches this server
(8.763.13) has no `significance` subcommand, and a model file written by hand
would be keyed on our guess at Vespa's stemming: if the terms did not match,
the model would silently do nothing and a null result would be
indistinguishable from a working null. So instead:

1. Read the significance Vespa is really using, per query term, from
   `term(i).significance` in match-features. This needs no guess about tokens.
2. Invert the measured significance-to-document-frequency relation to recover
   each term's document frequency.
3. Simulate one content node holding half the corpus by resampling each term's
   document frequency from Binomial(df, 0.5) -- which is what a random split
   does, and the reason the skew is larger for rare terms than common ones.
4. Re-issue every query with those significances set explicitly, using Vespa's
   own per-term `{significance: x}` annotation.
5. Score the result against the global-significance ranking.

The honest caveat, stated because it bounds the answer: step 2 rests on a
log-linear fit whose residuals are about 0.02, which is the same size as the
shift a half corpus produces. So this measures whether the effect is large, and
cannot resolve whether it is small or zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from score_rankings import score  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ENDPOINT = "http://localhost:8080/search/"
MAX_TERMS = 10


def ask(body: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fit_significance(terms: list[str]) -> tuple[float, float, float]:
    """Fit significance against log(document frequency), and report the fit."""
    rows = []
    for term in terms:
        df = ask({"yql": f'select doc_id from record where text contains "{term}"',
                  "hits": 0, "ranking.profile": "unranked", "timeout": "20s"}
                 )["root"]["fields"]["totalCount"]
        got = ask({"yql": "select doc_id from record where userQuery()", "query": term,
                   "type": "any", "ranking.profile": "bm25_flat_explain", "hits": 1,
                   "timeout": "20s"})
        kids = got["root"].get("children", [])
        if df and kids:
            rows.append((df, kids[0]["fields"]["matchfeatures"]["term(0).significance"]))

    df = np.array([r[0] for r in rows], float)
    sig = np.array([r[1] for r in rows], float)
    x = np.log(df)
    slope, intercept = np.linalg.lstsq(
        np.vstack([x, np.ones_like(x)]).T, sig, rcond=None)[0]
    pred = intercept + slope * x
    residual = float(np.abs(sig - pred).max())
    r2 = float(1 - ((sig - pred) ** 2).sum() / ((sig - sig.mean()) ** 2).sum())
    return float(slope), float(intercept), residual if r2 else residual


def term_significances(query: str) -> list[float]:
    got = ask({"yql": "select doc_id from record where userQuery()", "query": query,
               "type": "any", "ranking.profile": "bm25_flat_explain", "hits": 1,
               "timeout": "20s"})
    kids = got["root"].get("children", [])
    if not kids:
        return []
    features = kids[0]["fields"]["matchfeatures"]
    out = []
    for i in range(MAX_TERMS):
        value = features.get(f"term({i}).significance", 0.0)
        if value <= 0.0:
            break
        out.append(float(value))
    return out


def annotated_yql(query: str, significances: list[float]) -> str:
    """Vespa's own per-term significance annotation, one weakAnd-free OR."""
    words = [w for w in query.replace('"', " ").split() if w]
    parts = []
    for word, sig in zip(words, significances):
        parts.append(f'default contains ({{significance:{sig:.6f}}}"{word}")')
    return "select doc_id from record where " + " or ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/eval_real.json")
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    probe = ["psychology", "beekeeping", "garden", "history", "london", "poems",
             "cookery", "mystery", "children", "science", "gardening", "astronomy",
             "bees", "novel", "poetry", "cooking", "biography", "travel", "music", "art"]
    slope, intercept, residual = fit_significance(probe)
    print(f"significance ~ {intercept:.4f} + {slope:.5f} * log(df)")
    print(f"  worst residual over {len(probe)} probe terms: {residual:.4f}")
    halving = abs(slope) * math.log(2)
    print(f"  a term whose df halves moves by {halving:.4f}, "
          f"which is {'smaller' if halving < residual else 'larger'} than that residual")
    print()

    # The reference ranking: Vespa's own significance, annotated explicitly so
    # that the only difference from the perturbed runs is the numbers, not the
    # query form.
    observed = {item["query"]: term_significances(item["query"]) for item in queries}

    def run(perturb) -> dict[str, list[str]]:
        rankings = {}
        for item in queries:
            q = item["query"]
            sigs = perturb(observed[q])
            if not sigs:
                rankings[q] = []
                continue
            got = ask({"yql": annotated_yql(q, sigs), "hits": args.hits,
                       "ranking.profile": "bm25_flat", "timeout": "20s"})
            rankings[q] = [c["fields"]["doc_id"]
                           for c in got["root"].get("children", [])
                           if "doc_id" in c.get("fields", {})]
        return rankings

    base = run(lambda s: s)
    base_score = score(base, queries)
    print(f"global significance (as Vespa computed it)")
    print(f"  nDCG@10 {base_score['ndcg@10']:.4f}  controls {base_score['controls']:.4f}"
          f"  coverage {base_score['coverage']:.1%}")
    print()

    rng = np.random.default_rng(args.seed)

    def half_node(sigs: list[float]) -> list[float]:
        out = []
        for sig in sigs:
            df = math.exp((intercept - sig) / abs(slope))
            df = max(1.0, df)
            drawn = max(1, int(rng.binomial(int(round(df)), 0.5)))
            out.append(intercept + slope * math.log(drawn))
        return out

    print(f"one node holding half the corpus, {args.trials} random splits")
    deltas = []
    overlaps = []
    for trial in range(args.trials):
        perturbed = run(half_node)
        s = score(perturbed, queries)
        delta = s["ndcg@10"] - base_score["ndcg@10"]
        deltas.append(delta)
        same = [len(set(base[q][:10]) & set(perturbed[q][:10])) / 10
                for q in base if base[q]]
        overlap = float(np.mean(same))
        overlaps.append(overlap)
        print(f"  split {trial + 1}: nDCG@10 {s['ndcg@10']:.4f} "
              f"({delta:+.4f})  top-10 overlap with global {overlap:.1%}")

    print()
    print(f"mean shift {np.mean(deltas):+.4f}, worst {min(deltas):+.4f} / "
          f"{max(deltas):+.4f}")
    print(f"mean top-10 overlap {np.mean(overlaps):.1%}")


if __name__ == "__main__":
    main()
