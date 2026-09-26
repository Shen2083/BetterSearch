#!/usr/bin/env python3
"""Build a self-contained catalogue that runs real search in the browser.

    python scripts/build_standalone.py \
        --catalogue "data/catalogue_real.json,data/events_northfield.json" \
        --index .bettersearch/real-events-small \
        --model BAAI/bge-small-en-v1.5

The served page is only a front end: every search POSTs to /catalogue/search,
so opening web/catalogue.html from disk shows "Failed to fetch" (Safari says
"Load failed"). That makes the one thing the catalogue prototype exists for -
showing it to someone - the one thing it cannot do without a laptop running
Python.

This build removes the service. The whole corpus, its embedding vectors and the
search itself go into one HTML file, and the browser answers **any** query the
reader types, not a list decided here.

HOW IT WORKS
------------
Embedding a query needs the model; scoring against a corpus needs the corpus
vectors. So:

* The corpus vectors are lifted straight out of the index this script is
  pointed at - not re-derived. That matters: re-embedding here would mean two
  code paths producing "the same" vectors, and the day they diverge the page
  would quietly rank differently from the API. Reading the index makes them the
  same numbers by construction.
* The query is embedded in the browser by transformers.js, loading the ONNX
  build of the **same model** from the Hugging Face hub on first search.

That last point is the constraint that shapes everything: **the browser model
and the corpus vectors must be the same model.** Vectors from one encoder
scored against a query from another are not similar, they are noise. So this
build uses `bge-small-en-v1.5` (384 dims, ~35 MB quantised) while the server
default stays `bge-base-en-v1.5` (768 dims) - a browser cannot reasonably be
asked to download the base model. `--model` records which was used and the page
says so; nothing about the API changes.

The server does not prefix queries with bge's retrieval instruction, so neither
does the page. Matching the server matters more than the instruction, which was
measured at 0.006 nDCG either way.

VECTOR STORAGE
--------------
float32 would be 6 MB for this corpus before base64. The vectors are
L2-normalised, so each component is small and int8 with a per-vector scale
loses almost nothing: a dot product over 384 dimensions averages the error out.
Stored as base64 in a <script type="application/json"> block, that is about
2 MB. `scripts/check_browser_parity.py` measures what the quantisation costs
against the server's own ranking rather than assuming it is nothing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SOURCE = ROOT / "web" / "catalogue.html"
ENGINE = ROOT / "web" / "offline-search.js"
DEFAULT_OUT = ROOT / "docs" / "catalogue-standalone.html"

#: The ONNX build transformers.js loads. Must be the same weights as the index
#: the vectors came from, or the query and the corpus live in different spaces.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
ONNX_REPO = {"BAAI/bge-small-en-v1.5": "Xenova/bge-small-en-v1.5"}


def quantise(vectors: np.ndarray) -> tuple[bytes, bytes]:
    """int8 codes plus a per-vector scale.

    A single corpus-wide scale would be cheaper to store and worse: these are
    L2-normalised vectors, so a row whose weight sits in a few components has a
    much larger maximum than a diffuse one, and one shared divisor would flatten
    the diffuse rows into a handful of levels.
    """
    scales = np.abs(vectors).max(axis=1)
    scales[scales == 0] = 1.0  # a zero vector scores zero either way
    codes = np.rint(vectors / scales[:, None] * 127.0).astype(np.int8)
    return codes.tobytes(), (scales / 127.0).astype(np.float32).tobytes()


def read_index(index_path: Path) -> tuple[list[str], np.ndarray, str]:
    """Chunk doc_ids and their vectors, as the index stores them on disk.

    Deliberately reads the files rather than going through NumpyVectorIndex:
    the index exposes search, not "give me every vector", and reaching into its
    privates would be worse than reading the two-file format it documents.
    """
    meta = json.loads(Path(f"{index_path}.json").read_text(encoding="utf-8"))
    with np.load(f"{index_path}.npz") as payload:
        vectors = payload["vectors"].astype(np.float32, copy=False)
    doc_ids = [c["doc_id"] for c in meta["chunks"]]
    if len(doc_ids) != len(vectors):
        raise SystemExit(
            f"{index_path}: {len(doc_ids)} chunks but {len(vectors)} vectors"
        )
    return doc_ids, vectors, meta.get("model_id", "")


def json_block(name: str, payload: object) -> str:
    """A JSON island the page reads by id.

    `<` is escaped even though JSON does not require it: a record containing
    "</script>" would otherwise end the block early and take the rest of the
    page with it. Real catalogue text is not supposed to contain markup, which
    is exactly the sort of assumption that holds until it does not.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("<", "\\u003c")
    return f'<script type="application/json" id="{name}">{text}</script>\n'


def build(out: Path, model: str, interaction: str = "toggle") -> Path:
    from api.catalogue import (EXPLAIN_HEADINGS, SEMANTIC_TOP_K,
                              load_catalogue, load_meta)
    from bettersearch.exact import MAX_PROMOTED, PHRASE_TOKENS

    html = SOURCE.read_text(encoding="utf-8")
    if "window.OFFLINE" not in html:
        raise SystemExit(
            "catalogue.html has no window.OFFLINE hook - the standalone build "
            "cannot take over from the fetch. Re-add it before building."
        )

    catalogue = load_catalogue()
    meta = load_meta()
    index_path = Path(os.environ.get("BETTERSEARCH_INDEX_PATH", ".bettersearch/index"))
    doc_ids, vectors, index_model = read_index(index_path)

    if index_model and not index_model.startswith(model):
        raise SystemExit(
            f"index was built with {index_model!r} but --model says {model!r}. "
            f"The page would embed queries with one model and score them "
            f"against another, which returns plausible nonsense rather than "
            f"failing, so this is refused."
        )
    if model not in ONNX_REPO:
        raise SystemExit(
            f"no ONNX build recorded for {model!r}. Add it to ONNX_REPO once "
            f"you have checked transformers.js can load it."
        )

    # Every record ships now: the reader can type anything, so any record can
    # be a result. The old build shipped only records some pre-baked ranking
    # reached, which was the right call when the answers were decided here.
    unknown = set(doc_ids) - set(catalogue)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} indexed records are missing from the corpus - the "
            f"index and --catalogue disagree; rebuild one of them"
        )

    codes, scales = quantise(vectors)
    print(f"  corpus     {len(catalogue)} records, {len(doc_ids)} chunks")
    print(f"  vectors    {vectors.shape[1]} dims, int8 + per-vector scale, "
          f"{len(codes) / 1e6:.2f} MB raw")

    blob = (
        json_block("catalogue-meta", meta)
        + json_block("catalogue-records", catalogue)
        + json_block("catalogue-vectors", {
            "model": model,
            "onnx_repo": ONNX_REPO[model],
            "dims": int(vectors.shape[1]),
            "doc_ids": doc_ids,
            # Both base64 of little-endian buffers: int8 codes row-major, and
            # one float32 scale per row already divided by 127.
            "codes": base64.b64encode(codes).decode("ascii"),
            "scales": base64.b64encode(scales).decode("ascii"),
        })
        # The cut is read from api/catalogue.py rather than repeated here, so
        # the page and the API cannot disagree about how long a result list is.
        + f"<script>window.__SEMANTIC_TOP_K__ = {SEMANTIC_TOP_K};"
          f"window.__EXPLAIN_HEADINGS__ = {EXPLAIN_HEADINGS};"
          f"window.__PHRASE_TOKENS__ = {PHRASE_TOKENS};"
          f"window.__MAX_PROMOTED__ = {MAX_PROMOTED};"
          f'window.__INTERACTION__ = "{interaction}";</script>\n' 
        + "<script>\n" + ENGINE.read_text(encoding="utf-8") + "</script>\n"
    )

    marker = "<script>\nconst $ = (id) => document.getElementById(id);"
    if marker not in html:
        raise SystemExit("could not find the page script to inject before")
    html = html.replace(marker, blob + marker, 1)
    html = html.replace(
        "<title>", "<!-- Self-contained build: the whole corpus, its vectors "
        "and the search itself.\n     Regenerate with scripts/build_standalone.py "
        "-->\n<title>", 1
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--catalogue", type=Path,
                        help="corpus to serve (default: whatever "
                             "BETTERSEARCH_CATALOGUE or api/catalogue.py picks)")
    parser.add_argument("--index", type=Path,
                        help="index to take corpus vectors from "
                             "(default: BETTERSEARCH_INDEX_PATH)")
    parser.add_argument("--interaction", default="toggle",
                        choices=("toggle", "blended"),
                        help="toggle keeps the two modes; blended is one box "
                             "with named records promoted into the ranking")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"the model the index was built with, which the "
                             f"browser must also load (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Set before build() imports api.catalogue, which reads the env at import.
    if args.catalogue:
        os.environ["BETTERSEARCH_CATALOGUE"] = str(args.catalogue)
    if args.index:
        os.environ["BETTERSEARCH_INDEX_PATH"] = str(args.index)

    out = build(args.out, args.model, args.interaction)
    size = out.stat().st_size
    print(f"\n{SOURCE.relative_to(ROOT)} -> {out}  ({size / 1e6:.2f} MB)")
    if size > 8_000_000:
        print(f"  ! over the 8 MB budget by {(size - 8_000_000) / 1e6:.2f} MB",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
