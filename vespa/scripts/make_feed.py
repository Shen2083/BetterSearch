"""Turn the catalogue into a Vespa feed, and prove the port is faithful.

The vectors are not recomputed. They are lifted straight out of
`.bettersearch/real-books-base.npz`, the index behind the published 0.6582, so
that dense retrieval in Vespa starts from bit-identical input. Any difference
in the measured score is then Vespa's retrieval and ranking, which is the thing
under test, rather than a difference in what was embedded.

The script refuses to write a feed it cannot vouch for. Before emitting
anything it checks, for every one of the 4,000 records, that the text the
Python encoder saw is exactly `f"{title}\n\n{text}"` (src/bettersearch/
chunking.py:125) reconstructed from the corpus. If a single record disagrees it
exits, because a feed built on a guess about what was embedded would make every
number downstream unfalsifiable.

Availability, copies, branch, format and cover colour in the corpus are
invented for the proof of concept. The bibliographic fields are real, from
Open Library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORPUS = "data/catalogue_real.json"
INDEX = ".bettersearch/real-books-base"
NAMESPACE = "northfield"
DOCTYPE = "record"


def load_index(prefix: str) -> tuple[list[dict], np.ndarray, str]:
    meta = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
    vectors = np.load(prefix + ".npz")["vectors"]
    return meta["chunks"], vectors, meta["model_id"]


def year_int(year: str) -> int:
    """The corpus stores year as a string. Keep both forms rather than guess."""
    digits = "".join(c for c in year if c.isdigit())[:4]
    return int(digits) if digits else 0


def feed_document(record: dict, vector: np.ndarray) -> dict:
    return {
        "put": f"id:{NAMESPACE}:{DOCTYPE}::{record['doc_id']}",
        "fields": {
            "doc_id": record["doc_id"],
            "title": record["title"],
            "author": record["author"],
            "author_dates": record["author_dates"],
            "subjects": record["subjects"],
            "text": record["text"],
            "embed_text": f"{record['title']}\n\n{record['text']}",
            "year": record["year"],
            "year_int": year_int(record["year"]),
            "publisher": record["publisher"],
            "format": record["format"],
            "language": record["language"],
            "fiction": record["fiction"],
            "location": record["location"],
            "available": record["available"],
            "copies": record["copies"],
            "tags": record["tags"],
            "cover": record["cover"],
            "embedding": {"values": [float(v) for v in vector]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--out", default="vespa/feed/records.jsonl")
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))["documents"]
    by_id = {r["doc_id"]: r for r in corpus}
    chunks, vectors, model_id = load_index(args.index)

    if len(chunks) != len(vectors):
        raise SystemExit(f"{args.index}: {len(chunks)} chunks but {len(vectors)} vectors")

    # The audit. One chunk per record is an assumption the published figures
    # rest on; check it rather than inherit it.
    seen: set[str] = set()
    mismatched: list[str] = []
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id in seen:
            raise SystemExit(
                f"{doc_id} has more than one chunk; the feed assumes one vector "
                f"per record and would silently drop the rest"
            )
        seen.add(doc_id)
        record = by_id.get(doc_id)
        if record is None:
            raise SystemExit(f"{doc_id} is in the index but not in {args.corpus}")
        if chunk["embed_text"] != f"{record['title']}\n\n{record['text']}":
            mismatched.append(doc_id)

    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} records were embedded from text that does not "
            f"match the corpus (first: {mismatched[0]}). The feed would not be "
            f"a faithful port; refusing to write it."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for chunk, vector in zip(chunks, vectors):
            handle.write(json.dumps(feed_document(by_id[chunk["doc_id"]], vector)) + "\n")

    print(f"{len(chunks)} documents -> {out}")
    print(f"vectors from {args.index} ({model_id}), {vectors.shape[1]} dimensions, {vectors.dtype}")
    print(f"embed_text verified against {args.corpus} for all {len(chunks)} records")


if __name__ == "__main__":
    main()
