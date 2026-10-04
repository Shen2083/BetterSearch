#!/usr/bin/env python3
"""Rebuild the three indexes the published comparison arms were measured on.

    python scripts/rebuild_eval_indexes.py

`.bettersearch/` is gitignored, so a fresh clone - or a container restart, which
is how this script came to exist - has none of these. Without them the arms in
README's measured tables cannot be re-scored after the judged pool changes, and
"re-run every arm, not only the new one" becomes impossible to honour.

Everything here is deterministic and offline: the same corpus and the same
encoder produce the same vectors, so a rebuilt index ranks identically to the
one the figures were taken from. No API key is needed - the enriched arm reads
the enrichment already in `data/enrichment_real_haiku.jsonl` rather than
generating any, which is also why it must not go through `bettersearch enrich`:
that constructs an API client before it discovers it has nothing to ask.

    .bettersearch/real          all-MiniLM-L6-v2, record text as-is
    .bettersearch/real-enriched all-MiniLM-L6-v2, record text plus enrichment
    .bettersearch/real-bge      bge-base-en-v1.5, record text as-is

The first serves two lanes: BM25 reads the same chunks the vectors were built
from, so `keyword` and `semantic-thin` share an index and differ only in mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-base-en-v1.5"

#: (index path, encoder, fold enrichment in)
ARMS = [
    (".bettersearch/real", MINILM, False),
    (".bettersearch/real-enriched", MINILM, True),
    (".bettersearch/real-bge", BGE, False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/catalogue_real.json")
    ap.add_argument("--enrichment", default="data/enrichment_real_haiku.jsonl")
    ap.add_argument("--only", help="rebuild just this index path")
    args = ap.parse_args()

    from bettersearch.config import load_settings
    from bettersearch.enrichment import EnrichmentStore, enriched_documents
    from bettersearch.index.numpy_index import NumpyVectorIndex
    from bettersearch.ingest import ingest_documents, load_corpus

    documents = load_corpus(args.corpus)
    store = EnrichmentStore(Path(args.enrichment))
    base = load_settings()
    print(f"{len(documents)} records from {args.corpus}")

    for path, model, enrich in ARMS:
        if args.only and args.only != path:
            continue
        docs = documents
        if enrich:
            # The model argument selects which generation of enrichment to fold
            # in - the store is keyed on (prompt_version, model), so asking for
            # the wrong one silently returns nothing and builds a thin index
            # under an enriched name.
            docs = enriched_documents(documents, store=store, model="claude-haiku-4-5")
            folded = sum(1 for a, b in zip(docs, documents) if a.text != b.text)
            print(f"  enrichment folded into {folded}/{len(documents)} records")
            if not folded:
                print("  ! no enrichment matched - refusing to write a thin "
                      "index under an enriched name", file=sys.stderr)
                return 1

        # Set per-arm rather than exported, so one run can build indexes under
        # different encoders without the second inheriting the first's.
        os.environ["BETTERSEARCH_LOCAL_MODEL"] = model
        settings = replace(base, local_model_name=model, index_path=path)
        for suffix in (".json", ".npz"):
            Path(path + suffix).unlink(missing_ok=True)   # a model change invalidates it
        report = ingest_documents(docs, settings=settings,
                                  index=NumpyVectorIndex(path), progress=True)
        print(f"  {path}  {model.rsplit('/', 1)[-1]}  "
              f"{report.chunks_embedded} chunks, "
              f"{report.truncated_chunks} truncated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
