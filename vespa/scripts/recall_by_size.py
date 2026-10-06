"""How HNSW recall moves as the corpus grows, over real records only.

One measurement at one corpus size says nothing about a corpus a hundred times
larger. Three points over a 4x range do not say much more, but a flat line and
a falling line are different claims, and this is the only honest scale signal
available here: the alternative would be inventing catalogue records, which
would make every number downstream unfalsifiable.

It feeds in stages rather than redeploying, so the graph grows the way it would
in service rather than being rebuilt from scratch at each size. At each stage it
asks the same 41 queries exactly and approximately and reports the recall of
the approximation against the exact answer. No judgements are involved, so
nothing here is affected by the unpooled lanes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_approximation import compare, run  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def count() -> int:
    body = {"yql": "select doc_id from record where true", "hits": 0,
            "ranking.profile": "unranked", "timeout": "30s"}
    request = urllib.request.Request(
        "http://localhost:8080/search/", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(request, timeout=60) as response:
        return json.loads(response.read())["root"]["fields"]["totalCount"]


def feed(path: Path) -> None:
    # Extend the environment rather than replace it. Replacing it drops HOME
    # and everything else the CLI reads, and the failure is silent: the command
    # exits with no stdout at all, which looks like an empty feed rather than a
    # broken invocation.
    env = dict(os.environ)
    env.update({"VESPA_CLI_HOME": "/tmp/.vespa", "HTTPS_PROXY": "",
                "https_proxy": "", "NO_PROXY": "*", "no_proxy": "*"})
    out = subprocess.run(["vespa", "feed", str(path)], capture_output=True,
                         text=True, env=env, timeout=1800)
    if '"feeder.error.count": 0' not in out.stdout:
        raise SystemExit(
            f"feed did not come back clean (exit {out.returncode})\n"
            f"stdout: {out.stdout[-1200:]!r}\nstderr: {out.stderr[-1200:]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", default="vespa/feed/records.jsonl")
    parser.add_argument("--queries", default="data/eval_real.json")
    parser.add_argument("--sizes", default="1000,2000,4000")
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--explore", type=int, default=0,
                        help="hnsw.exploreAdditionalHits; 0 is the honest "
                             "setting here because a generous one hides the "
                             "trend behind a perfect score")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    lines = Path(args.feed).read_text(encoding="utf-8").splitlines()
    sizes = [int(x) for x in args.sizes.split(",")]

    from bettersearch.embeddings import get_provider
    provider = get_provider()
    vectors = {i["query"]: [float(v) for v in provider.embed_query(i["query"])]
               for i in queries}

    print(f"{'documents':>10} {'recall@10':>10} {'recall@100':>11} "
          f"{'identical top10':>16} {'exact ms':>9} {'hnsw ms':>8}")

    staged = Path("/tmp/stage.jsonl")
    previous = 0
    for size in sizes:
        staged.write_text("\n".join(lines[previous:size]) + "\n", encoding="utf-8")
        feed(staged)
        previous = size
        indexed = count()

        exact, exact_ms = run(queries, "dense", "dense", args.hits, vectors, {})
        approx, hnsw_ms = run(queries, "dense_hnsw", "dense_hnsw", args.hits,
                              vectors, {"hnsw.exploreAdditionalHits": args.explore})
        c = compare(exact, approx)
        print(f"{indexed:>10} {c['recall@10']:>10.4f} {c['recall@100']:>11.4f} "
              f"{c['identical_top10']:>13}/{c['queries']} "
              f"{exact_ms:>8.0f} {hnsw_ms:>8.0f}")

    print()
    print("Read the direction, not the values. A 4x range cannot tell you what")
    print("happens at 100x, and every row here is far below the size at which")
    print("the approximation starts to earn its place.")


if __name__ == "__main__":
    main()
