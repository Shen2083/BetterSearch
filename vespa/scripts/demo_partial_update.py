"""Show availability changing in place, with no reindexing.

This is the operational argument for Vespa rather than a quality one, and it is
the finding in docs/VESPA-HANDOVER.md with no number attached because it does
not produce one. `available` and `copies` change on every loan and every
return. In our system those live in the catalogue JSON that the index was built
from, so moving them means rebuilding, re-embedding and reloading.

Here they are attributes declared fast-access. An `assign` update rewrites two
integers on the content node and nothing else: the 768-float vector and the
ColBERT token tensors are untouched. The test is whether the change is visible
to a *filter* on the very next query, because a stale availability filter is
worse than no filter.

The record is chosen so the filter genuinely flips: one that is currently
unavailable, made available. Changing 2 copies to 3 would not move a count of
records with any copies at all.

Availability and copy counts in this corpus are invented for the proof of
concept. They are not real library data.
"""

from __future__ import annotations

import json
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SEARCH = "http://localhost:8080/search/"
DOCUMENT = "http://localhost:8080/document/v1/northfield/record/docid"


def ask(body: dict) -> dict:
    request = urllib.request.Request(
        SEARCH, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def update(doc_id: str, fields: dict) -> float:
    body = {"fields": {k: {"assign": v} for k, v in fields.items()}}
    request = urllib.request.Request(
        f"{DOCUMENT}/{doc_id}?create=false", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="PUT")
    start = time.perf_counter()
    with OPENER.open(request, timeout=30) as response:
        response.read()
    return (time.perf_counter() - start) * 1000.0


def count(where: str) -> int:
    return ask({"yql": f"select doc_id from record where {where}", "hits": 0,
                "ranking.profile": "unranked", "timeout": "20s"}
               )["root"]["fields"]["totalCount"]


def read(doc_id: str) -> dict:
    got = ask({"yql": f'select doc_id, title, available, copies from record '
                       f'where doc_id contains "{doc_id}"',
               "hits": 1, "ranking.profile": "unranked", "timeout": "20s"})
    return got["root"]["children"][0]["fields"]


def main() -> None:
    # A record nobody can borrow right now, so making it available moves the
    # filter rather than leaving the count where it was.
    got = ask({"yql": "select doc_id, title, available, copies from record "
                      "where available = 0",
               "hits": 1, "ranking.profile": "unranked", "timeout": "20s"})
    record = got["root"]["children"][0]["fields"]
    doc_id = record["doc_id"]

    before = count("available > 0")
    print(f"{doc_id}  {record['title'][:50]!r}")
    print(f"  available {record['available']}, copies {record['copies']}")
    print(f"  {before} of 4000 records are borrowable")

    ms = update(doc_id, {"available": 1, "copies": max(1, record["copies"])})
    now = read(doc_id)
    after = count("available > 0")

    print()
    print(f"  one returned copy, fed as a partial update: {ms:.0f} ms")
    print(f"  available {now['available']}, copies {now['copies']}")
    print(f"  {after} borrowable, a change of {after - before}, "
          f"on the query issued immediately after")

    restored_ms = update(doc_id, {"available": record["available"],
                                  "copies": record["copies"]})
    final = count("available > 0")
    print()
    print(f"  put back in {restored_ms:.0f} ms; {final} borrowable again")
    if final != before:
        raise SystemExit(f"failed to restore: {final} borrowable, expected {before}")
    print("  no reindexing, no re-embedding, no redeploy")


if __name__ == "__main__":
    main()
