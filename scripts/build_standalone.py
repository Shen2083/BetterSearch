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

The example queries are read out of catalogue.html's own preset buttons rather
than listed here, so the two cannot drift apart.

Needs an index built for the catalogue corpus:

    export BETTERSEARCH_INDEX_PATH=.bettersearch/catalogue
    bettersearch ingest --corpus data/catalogue_library.json
"""

from __future__ import annotations

import argparse
import json
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


def preset_queries(html: str) -> list[str]:
    """The example queries, taken from the page's own preset buttons."""
    queries = re.findall(r'<button type="button" data-q="([^"]+)"', html)
    if not queries:
        raise SystemExit("no data-q preset buttons found in catalogue.html")
    # The default query in the search box must work too, or the file opens broken.
    if match := re.search(r'<input id="q"[^>]*value="([^"]*)"', html):
        if (default := match.group(1).strip()) and default not in queries:
            queries.insert(0, default)
    return queries


def build_rankings(queries: list[str]) -> tuple[dict, dict]:
    """Run each query through the real search path and keep the ordering."""
    from api.catalogue import load_catalogue, run_search
    from bettersearch import Searcher, load_settings

    catalogue = load_catalogue()
    searcher = Searcher(settings=load_settings())

    rankings: dict[str, list] = {}
    for query in queries:
        for mode in MODES:
            # per_page = whole catalogue: we want the full ranking, not a page.
            data = run_search(
                searcher, catalogue, query=query, mode=mode,
                page=1, per_page=len(catalogue),
            )
            key = f"{normalise(query)}|{mode}"
            rankings[key] = [[r["doc_id"], r["score"]] for r in data["results"]]
            print(f"  {mode:<9} {query!r} -> {len(rankings[key])} records")
    return catalogue, rankings


def build(out: Path) -> Path:
    html = SOURCE.read_text(encoding="utf-8")
    queries = preset_queries(html)
    print(f"{len(queries)} example queries x {len(MODES)} modes:")
    catalogue, rankings = build_rankings(queries)

    if "window.OFFLINE" not in html:
        raise SystemExit(
            "catalogue.html has no window.OFFLINE hook - the standalone build "
            "cannot take over from the fetch. Re-add it before building."
        )

    blob = (
        "<script>\n"
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
    args = parser.parse_args()
    out = build(args.out)
    print(f"\n{SOURCE.relative_to(ROOT)} -> {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
