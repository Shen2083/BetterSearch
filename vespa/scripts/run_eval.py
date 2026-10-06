"""Ask Vespa the 41 eval queries and write lane files the harness can score.

Output is `{query: [doc_id, ...]}` and nothing else, which is the only shape
`scripts/score_rankings.py` reads and the shape
`scripts/build_eval_set.py --lane-file` pools. RUNBOOK.md:155 anticipated this:
"Any future system - a Vespa ranking profile included - becomes poolable by
dumping its rankings in that shape, with no code change here."

Query strings are copied out of data/eval_real.json untouched, because the
scorer matches on the query text character for character and silently reports
a mismatch as a missing query rather than failing.

Two deliberate choices about faithfulness:

* The `dense` lane's query vector comes from `bettersearch.embeddings.get_provider`,
  the same encoder that built the index, so the only variable under test is
  Vespa's retrieval and ranking. Letting Vespa embed the query is the separate
  `dense_native` lane, because it changes two things at once: ONNX against torch
  numerics, and Vespa's tokenizer against sentence-transformers'. Both have now
  been run and they agree on every one of the 41 top tens, so the separation
  turned out to be unnecessary - but it is the reason the agreement means
  anything.

* The keyword lane uses `type=any`. Our BM25 scans every record and scores
  whatever terms are present, which is an OR. Vespa defaults to weakAnd, an OR
  with early termination, and that approximation would show up as a scoring
  difference that has nothing to do with BM25.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8080/search/"
QUERIES = "data/eval_real.json"

# No proxy: Vespa is on localhost, and the agent proxy would neither route to
# it nor be reachable from it.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def search(body: dict, *, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Vespa returned {exc.code}: {exc.read().decode()[:600]}") from exc


def doc_ids(result: dict) -> list[str]:
    children = result.get("root", {}).get("children", [])
    ids = []
    for child in children:
        fields = child.get("fields", {})
        if "doc_id" in fields:
            ids.append(fields["doc_id"])
    return ids


def build_body(mode: str, query: str, hits: int, profile: str,
               vector: list[float] | None, extra: dict) -> dict:
    body: dict = {"hits": hits, "ranking.profile": profile, "timeout": "20s"}

    if mode in {"bm25", "hybrid"}:
        body["query"] = query
        body["type"] = "any"

    if mode == "bm25":
        body["yql"] = "select doc_id from record where userQuery()"
    elif mode == "dense":
        body["yql"] = (
            f"select doc_id from record where "
            f"{{targetHits:{hits}}}nearestNeighbor(embedding, q)"
        )
    elif mode == "hybrid":
        body["yql"] = (
            f"select doc_id from record where userQuery() or "
            f"({{targetHits:{hits}}}nearestNeighbor(embedding, q))"
        )
    elif mode == "dense_hnsw":
        # approximate:true is not decoration. Vespa falls back to an exact scan
        # when it judges the approximation unhelpful - a restrictive filter, or
        # targetHits large against the corpus - and says nothing about it. A
        # lane that silently ran exactly would read as perfect recall, which is
        # the same failure shape as closeness(field, embedding) scoring 0.0837
        # while retrieving the right candidates. compare_approximation.py
        # checks the fallback did not happen rather than assuming.
        # exploreAdditionalHits is an annotation on the operator, not a query
        # property. Sent as a property it is accepted and ignored, so the sweep
        # would have produced identical rows and read as insensitivity to the
        # parameter rather than as the parameter never arriving.
        explore = extra.pop("hnsw.exploreAdditionalHits", None)
        annotations = f"targetHits:{hits}, approximate:true"
        if explore is not None:
            annotations += f", hnsw.exploreAdditionalHits:{int(explore)}"
        body["yql"] = (
            f"select doc_id from record where "
            f"{{{annotations}}}nearestNeighbor(embedding_hnsw, q)"
        )
    elif mode == "dense_native":
        # No vector in the request. Vespa tokenises and embeds the query with
        # the same ONNX model it embedded the records with.
        body["yql"] = (
            f"select doc_id from record where "
            f"{{targetHits:{hits}}}nearestNeighbor(embedding_native, qn)"
        )
        body["input.query(qn)"] = "embed(bge, @qtext)"
        body["qtext"] = query
    elif mode == "binary":
        # Hamming over the packed vectors. The query is packed the same way:
        # one bit per dimension by sign, eight per byte, most significant bit
        # first, which is what Vespa's pack_bits does. Getting the bit order
        # wrong does not error, it returns nonsense, so the dense lane is the
        # control: a correct packing tracks it closely.
        body["yql"] = (
            f"select doc_id from record where "
            f"{{targetHits:{hits}}}nearestNeighbor(embedding_binary, qb)"
        )
    elif mode == "colbert":
        # Retrieval is the same dense lane; the second phase is what differs.
        # The query's token embeddings are produced by Vespa's own embedder,
        # not by us, which is the point: our 0.7407 rested on a MaxSim written
        # by hand against pylate.
        body["yql"] = (
            f"select doc_id from record where "
            f"{{targetHits:{hits}}}nearestNeighbor(embedding, q)"
        )
        body["input.query(qt)"] = "embed(colbert, @qtext)"
        body["qtext"] = query
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    if vector is not None:
        body["input.query(q)"] = {"values": vector}
        if mode == "binary":
            import numpy as np
            bits = (np.asarray(vector, dtype=np.float32) > 0).astype(np.uint8)
            packed = np.packbits(bits, bitorder="big").view(np.int8)
            body["input.query(qb)"] = {"values": [int(v) for v in packed]}

    body.update(extra)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["bm25", "dense", "hybrid", "colbert",
                                 "binary", "dense_native", "dense_hnsw"])
    parser.add_argument("--profile", required=True, help="Vespa rank-profile name")
    parser.add_argument("--out", required=True, help="lane file to write")
    parser.add_argument("--queries", default=QUERIES)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument(
        "--set", action="append", default=[],
        help="extra query property, KEY=VALUE, repeatable "
             "(e.g. --set 'model.defaultIndex=fielded')",
    )
    args = parser.parse_args()

    extra: dict = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra[key] = value

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    provider = None
    if args.mode in {"dense", "hybrid", "colbert", "binary", "dense_hnsw"}:
        from bettersearch.embeddings import get_provider
        provider = get_provider()
        print(f"query encoder: {provider.model_id}")

    rankings: dict[str, list[str]] = {}
    empty: list[str] = []
    elapsed: list[float] = []

    for item in queries:
        query = item["query"]
        vector = None
        if provider is not None:
            vector = [float(v) for v in provider.embed_query(query)]
        body = build_body(args.mode, query, args.hits, args.profile, vector, extra)
        start = time.perf_counter()
        result = search(body)
        elapsed.append((time.perf_counter() - start) * 1000.0)
        ids = doc_ids(result)
        if not ids:
            empty.append(query)
        rankings[query] = ids

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rankings, indent=1), encoding="utf-8")

    depths = [len(v) for v in rankings.values()]
    print(f"{len(rankings)} queries -> {out}")
    print(f"profile {args.profile}, mode {args.mode}"
          + (f", {extra}" if extra else ""))
    print(f"hits returned: min {min(depths)}, median "
          f"{sorted(depths)[len(depths) // 2]}, max {max(depths)}")
    print(f"vespa round trip: {sum(elapsed) / len(elapsed):.0f} ms mean, "
          f"{max(elapsed):.0f} ms max")
    if empty:
        print(f"! {len(empty)} queries returned nothing: {empty[:3]}")


if __name__ == "__main__":
    main()
