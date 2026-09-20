#!/usr/bin/env python3
"""Build a self-contained catalogue demo that works from file://.

    python scripts/build_standalone.py

The served page is only a front end: every search POSTs to /catalogue/search,
so opening web/catalogue.html from disk shows "Failed to fetch" (Safari says
"Load failed"). That makes the one thing the catalogue prototype exists for -
showing it to someone - the one thing it cannot do without a laptop running
Python.

Only retrieval actually needs Python. So the example searches are run here, at
build time, through the *real* Searcher and the *real* run_search, and the
resulting rankings are baked into a copy of the page. Facets, filtering and
pagination still run live in the browser, ported in web/offline-search.js.

The example queries come from the corpus file, the same place the served page
gets them from, so the built file cannot offer a button it has no ranking for.
--query overrides them when you want to bake a different set.

Only the records those rankings reach are shipped. Against the real 4,000-record
catalogue six example queries reach 446 records, so the file stays around 300 KB
rather than carrying 3.2 MB of catalogue nobody can navigate to. That only works
because the meaning lane now stops at SEMANTIC_TOP_K - under the old relevance
floor a single query reached 3,665 records on its own.

Needs an index built for the corpus being baked:

    export BETTERSEARCH_INDEX_PATH=.bettersearch/catalogue
    bettersearch ingest --corpus data/catalogue_library.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SOURCE = ROOT / "web" / "catalogue.html"
ENGINE = ROOT / "web" / "offline-search.js"
DEFAULT_OUT = ROOT / "docs" / "catalogue-standalone.html"

MODES = ("keyword", "semantic")

#: Must match the normaliser in offline-search.js.
#: Keys are "<normalised query>|<mode>" - see the note there on why the
#: separator is a plain pipe and not something more exotic.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(query: str) -> str:
    return _NON_ALNUM.sub(" ", query.lower()).strip()


def build_rankings(queries: list[str]) -> tuple[dict, dict]:
    """Run each query through the real search path and keep the ordering."""
    from api.catalogue import load_catalogue, run_search
    from bettersearch import Searcher, load_settings

    catalogue = load_catalogue()
    searcher = Searcher(settings=load_settings())

    rankings: dict[str, list] = {}
    for query in queries:
        for mode in MODES:
            # per_page = whole catalogue so the page is not what truncates this.
            # The ranking still ends where run_search ends it - top-k for the
            # meaning lane, full BM25 depth for keyword - which is the point:
            # the offline copy inherits the served page's cut rather than
            # choosing its own, so the two cannot disagree about a result count.
            data = run_search(
                searcher, catalogue, query=query, mode=mode,
                page=1, per_page=len(catalogue),
            )
            key = f"{normalise(query)}|{mode}"
            # Third element is the occurrence count of a collapsed event
            # series. It is carried rather than re-derived offline because
            # run_search has already collapsed the list by this point - the
            # baked ranking holds one record per series and the count with it.
            rankings[key] = [[r["doc_id"], r["score"], r.get("repeats", 1)]
                             for r in data["results"]]
            print(f"  {mode:<9} {query!r} -> {len(rankings[key])} records")
    return catalogue, rankings


def reachable(catalogue: dict, rankings: dict) -> dict:
    """Only the records some baked ranking can reach.

    Shipping the whole catalogue would put every record a query never returns
    into the file - on the real corpus that is 3.2 MB of dead weight. The
    offline engine looks records up by doc_id and drops misses, so a subset is
    safe; it is only unsafe if a ranking references a record that is not here,
    which the caller checks.
    """
    wanted = {doc_id for ranking in rankings.values() for doc_id, *_ in ranking}
    return {doc_id: catalogue[doc_id] for doc_id in wanted if doc_id in catalogue}


def build(out: Path, queries: list[str] | None = None) -> Path:
    from api.catalogue import load_meta

    html = SOURCE.read_text(encoding="utf-8")
    meta = load_meta()
    # The page renders its buttons and its footnote from this, exactly as the
    # served page does from /catalogue/meta - so the offline copy cannot end up
    # offering a query it has no ranking for, or describing a different corpus.
    if queries:
        meta["presets"] = queries
    else:
        queries = meta["presets"]
    if not queries:
        raise SystemExit(
            "the corpus declares no presets and none were given with --query; "
            "the built file would open with no examples to press"
        )
    print(f"{len(queries)} example queries x {len(MODES)} modes:")
    catalogue, rankings = build_rankings(queries)

    full = len(catalogue)
    catalogue = reachable(catalogue, rankings)
    missing = {doc_id for r in rankings.values() for doc_id, *_ in r} - set(catalogue)
    if missing:
        raise SystemExit(f"{len(missing)} ranked records are not in the "
                         f"catalogue - the index and the corpus disagree")
    print(f"\n  {len(catalogue)} of {full} records reachable from these queries")

    if "window.OFFLINE" not in html:
        raise SystemExit(
            "catalogue.html has no window.OFFLINE hook - the standalone build "
            "cannot take over from the fetch. Re-add it before building."
        )

    blob = (
        "<script>\n"
        "window.__OFFLINE_META__ = "
        + json.dumps(meta, ensure_ascii=False)
        + ";\n"
        "window.__OFFLINE_RECORDS__ = "
        + json.dumps(catalogue, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
        "window.__OFFLINE_RANKINGS__ = "
        + json.dumps(rankings, ensure_ascii=False, separators=(",", ":"))
        + ";\n</script>\n<script>\n"
        + ENGINE.read_text(encoding="utf-8")
        + "</script>\n"
    )

    # Inject immediately before the page's own script, which calls run() at its
    # end - window.OFFLINE has to exist by then. The footnote is above this
    # point in the document, so the engine can annotate it straight away.
    marker = "<script>\nconst $ = (id) => document.getElementById(id);"
    if marker not in html:
        raise SystemExit("could not find the page script to inject before")
    html = html.replace(marker, blob + marker, 1)

    html = html.replace(
        "<title>", "<!-- Self-contained offline build. Regenerate with "
        "scripts/build_standalone.py -->\n<title>", 1
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--query", action="append", metavar="TEXT",
                        help="example query to bake, repeatable; overrides the "
                             "page's own preset buttons")
    parser.add_argument("--catalogue", type=Path,
                        help="corpus to serve (default: whatever "
                             "BETTERSEARCH_CATALOGUE or api/catalogue.py picks)")
    parser.add_argument("--index", type=Path,
                        help="index to search (default: BETTERSEARCH_INDEX_PATH)")
    args = parser.parse_args()

    # Set before build() imports api.catalogue, which reads the env at import.
    if args.catalogue:
        os.environ["BETTERSEARCH_CATALOGUE"] = str(args.catalogue)
    if args.index:
        os.environ["BETTERSEARCH_INDEX_PATH"] = str(args.index)
    out = build(args.out, args.query or None)
    print(f"\n{SOURCE.relative_to(ROOT)} -> {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
