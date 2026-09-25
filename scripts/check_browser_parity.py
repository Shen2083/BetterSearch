#!/usr/bin/env python3
"""Does the browser return what the API returns?

    python scripts/check_browser_parity.py \
        --page docs/catalogue-standalone.html \
        --index .bettersearch/real-events-small \
        --model BAAI/bge-small-en-v1.5

The standalone page does its own retrieval now: it embeds the query with
transformers.js and scores it against int8 corpus vectors. That is a second
implementation of something already implemented in Python, and the only honest
way to ship it is to check the two agree on every query we have.

So this drives the real built file in a real browser, runs each eval query
through `window.OFFLINE.search`, and compares the ranking against `run_search`
on the same corpus and the same model.

WHAT WOULD MAKE THEM DIFFER, AND WHICH OF IT IS FIXABLE
-------------------------------------------------------
* **Corpus vectors** are lifted from the index rather than recomputed, so they
  are the same numbers on both sides. Not a source of difference.
* **int8 storage.** The page stores each vector as int8 with a per-vector
  scale. That is lossy, and it is the cost of not shipping 6 MB of float32.
* **The model itself.** The page loads the *quantised* ONNX build - the whole
  point is that a reader downloads 35 MB, not 130 MB - while Python runs fp32
  weights. The same sentence therefore gets slightly different query vectors:
  measured at about 0.01 per component. This one cannot be removed without
  making the page far heavier, so it is measured instead.

A near-identical top 20 with a little churn at the bottom is the expected
result. Wholesale disagreement means something is actually wrong - most likely
the page and the index no longer share a model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def server_rankings(queries: list[str], catalogue_paths: str, index: str,
                    model: str, top_k: int, mode: str,
                    explained: dict) -> dict[str, list[str]]:
    os.environ["BETTERSEARCH_CATALOGUE"] = catalogue_paths
    os.environ["BETTERSEARCH_INDEX_PATH"] = index
    os.environ["BETTERSEARCH_LOCAL_MODEL"] = model
    from api.catalogue import load_catalogue, run_search
    from bettersearch import Searcher, load_settings

    catalogue = load_catalogue()
    searcher = Searcher(settings=load_settings())
    out = {}
    for query in queries:
        data = run_search(searcher, catalogue, query=query, mode=mode,
                          page=1, per_page=top_k)
        out[query] = [r["doc_id"] for r in data["results"]]
        explained[query] = {r["doc_id"]: r.get("closest_headings")
                            or r.get("missing_terms") or []
                            for r in data["results"]}
    return out


async def browser_rankings(page_path: Path, queries: list[str], top_k: int,
                           mode: str, explained: dict) -> dict[str, list[str]]:
    from playwright.async_api import async_playwright

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    out: dict[str, list[str]] = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=CHROMIUM,
            proxy={"server": proxy} if proxy else None,
            # The model is fetched through this environment's TLS-terminating
            # proxy, which the bundled Chromium has no root for. Only ever for
            # this local check - nothing about the shipped page depends on it.
            args=["--ignore-certificate-errors"] if proxy else [],
        )
        page = await browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(page_path.resolve().as_uri())
        await page.wait_for_function("window.OFFLINE !== undefined", timeout=30_000)

        # First call downloads the model; give it room, then the rest are fast.
        for i, query in enumerate(queries):
            rows = await page.evaluate(
                """async ([q, k, mode]) => {
                    const d = await window.OFFLINE.search(q, mode, {}, 1, k);
                    return d.results.map(r => [r.doc_id,
                        r.closest_headings || r.missing_terms || []]);
                }""",
                [query, top_k, mode],
            )
            ids = [r[0] for r in rows]
            explained[query] = {r[0]: r[1] for r in rows}
            out[query] = ids
            if i == 0:
                print(f"  model loaded, first query returned {len(ids)} records")
        await browser.close()
    if errors:
        print(f"  ! {len(errors)} page errors: {errors[:3]}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, default=ROOT / "docs/catalogue-standalone.html")
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real.json")
    ap.add_argument("--catalogue",
                    default="data/catalogue_real.json,data/events_northfield.json")
    ap.add_argument("--index", default=".bettersearch/real-events-small")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--show", type=int, default=8, help="worst N queries to list")
    ap.add_argument("--mode", default="semantic", choices=("semantic", "keyword"))
    args = ap.parse_args()

    raw = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = [q["query"] for q in (raw["queries"] if isinstance(raw, dict) else raw)]
    print(f"\n{len(queries)} queries · top-{args.top_k} · {args.page.name}\n")

    browser_why: dict[str, dict] = {}
    server_why: dict[str, dict] = {}
    print("browser:")
    browser = asyncio.run(browser_rankings(args.page, queries, args.top_k,
                                           args.mode, browser_why))
    print("server:")
    server = server_rankings(queries, args.catalogue, args.index, args.model,
                             args.top_k, args.mode, server_why)
    print(f"  {len(server)} rankings computed\n")

    overlaps, identical, first_same = [], 0, 0
    rows = []
    for query in queries:
        b, s = browser[query], server[query]
        shared = len(set(b) & set(s))
        denominator = max(len(s), 1)
        overlaps.append(shared / denominator)
        identical += b == s
        first_same += bool(b and s and b[0] == s[0])
        rows.append((shared / denominator, query, shared, len(s), b, s))

    print(f"  top-{args.top_k} overlap      mean {statistics.mean(overlaps):.1%}   "
          f"min {min(overlaps):.0%}")

    # Set overlap says how much the lists differ. It does not say whether the
    # difference matters, and that is the question: a page that swaps two
    # irrelevant records at rank 18 is fine, one that loses the right answer is
    # not. Where the query file carries judgements, score both rankings.
    graded = {q["query"]: set(q.get("relevant_doc_ids", []))
              for q in (raw["queries"] if isinstance(raw, dict) else raw)}
    if any(graded.values()):
        import math

        def ndcg(ranked, relevant, k=10):
            gain = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k])
                       if d in relevant)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
            return gain / ideal if ideal else 0.0

        b = statistics.mean(ndcg(browser[q], graded[q]) for q in queries)
        srv = statistics.mean(ndcg(server[q], graded[q]) for q in queries)
        print(f"  nDCG@10              browser {b:.3f}   server {srv:.3f}   "
              f"difference {b - srv:+.3f}")
    # The page now also says *why* a record is here, and that is a second
    # implementation of a second thing - so it needs holding to the same
    # standard as the ranking. Compared only on records both sides returned:
    # a record one side never retrieved has no explanation to disagree about.
    #
    # Do not expect 100%, even with both sides on the same model. Which two of
    # a record's seven headings sit closest to a query is often a near tie, and
    # the page reads those distances off q8 ONNX weights while Python reads
    # them off fp32 - the same difference that moves the ranking. Measured at
    # 89% on the eval set. A sharp fall from there means the port has drifted;
    # a few points either way is quantisation.
    shared_records = agree = 0
    for query in queries:
        for doc, why in server_why.get(query, {}).items():
            if doc in browser_why.get(query, {}):
                shared_records += 1
                agree += browser_why[query][doc] == why
    if shared_records:
        print(f"  same explanation     {agree}/{shared_records} shared records "
              f"({agree / shared_records:.1%})")
    print(f"  identical ordering   {identical}/{len(queries)}")
    print(f"  same first result    {first_same}/{len(queries)}")

    rows.sort()
    worst = [r for r in rows if r[0] < 1.0][: args.show]
    if worst:
        print(f"\n  queries where the sets differ:\n")
        for score, query, shared, total, b, s in worst:
            print(f"    {query[:44]:<46} {shared}/{total} shared")
            only_b = [d for d in b if d not in set(s)]
            only_s = [d for d in s if d not in set(b)]
            if only_b:
                print(f"      browser only: {', '.join(only_b[:4])}")
            if only_s:
                print(f"      server only:  {', '.join(only_s[:4])}")
    else:
        print("\n  every query returns the same set of records.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
