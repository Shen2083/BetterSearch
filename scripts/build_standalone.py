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
than listed here, so the two cannot drift apart. Passing --query overrides them,
for building against a corpus the page's own presets were not written for; the
preset row and the default search box in the output are rewritten to match, so
every button in the built file is one the file can actually answer.

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


def rewrite_presets(html: str, queries: list[str]) -> str:
    """Point the page's preset buttons and search box at the baked queries.

    Without this a build against a different corpus ships buttons the file
    cannot answer: the page's own presets name a surname collision and a title
    that exist only in the demonstration catalogue. A dead preset button reads
    as a broken page, which is worse than no button.
    """
    buttons = "\n      ".join(
        f'<button type="button" data-q="{html_escape(q)}">{html_escape(q)}</button>'
        for q in queries
    )
    html = re.sub(
        r'(<p class="presets">\s*\n\s*Try:\n)(.*?)(\n\s*</p>)',
        lambda m: m.group(1) + "      " + buttons + m.group(3),
        html, count=1, flags=re.DOTALL,
    )
    return re.sub(
        r'(<input id="q"[^>]*value=")[^"]*(")',
        lambda m: m.group(1) + html_escape(queries[0]) + m.group(2),
        html, count=1,
    )


def rewrite_footnote(html: str) -> str:
    """Replace the page footnote with the corpus's own description of itself.

    The served page describes a catalogue of invented records. That is a false
    statement about the real corpus, where the bibliographic data is genuine
    Open Library material and only the holdings - availability, copies, branch,
    format, cover colour - are invented. Getting that the wrong way round in a
    file meant to be sent to a library is exactly the claim not to get wrong, so
    it is taken from the corpus file rather than written here.
    """
    from api.catalogue import CATALOGUE_PATHS

    # Every collection served, not just the first. With books and events the
    # books file says the bibliographic data is real and the events file says
    # the whole programme is invented; printing only one of those is precisely
    # the misstatement this function exists to avoid.
    parts = []
    for path in CATALOGUE_PATHS:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("description"):
            parts.append(raw["description"].strip())
    description = " ".join(parts)
    if not description:
        return html
    return re.sub(
        r'(<p class="footnote">\s*\n)(.*?)(\n</p>)',
        lambda m: m.group(1) + "  " + html_escape(description) + m.group(3),
        html, count=1, flags=re.DOTALL,
    )


def html_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def build(out: Path, queries: list[str] | None = None) -> Path:
    html = SOURCE.read_text(encoding="utf-8")
    if queries:
        html = rewrite_presets(html, queries)
        if set(preset_queries(html)) != set(queries):
            raise SystemExit("preset rewrite did not take - the page markup "
                             "has moved; fix rewrite_presets before shipping")
    else:
        queries = preset_queries(html)
    print(f"{len(queries)} example queries x {len(MODES)} modes:")
    catalogue, rankings = build_rankings(queries)

    full = len(catalogue)
    catalogue = reachable(catalogue, rankings)
    missing = {doc_id for r in rankings.values() for doc_id, *_ in r} - set(catalogue)
    if missing:
        raise SystemExit(f"{len(missing)} ranked records are not in the "
                         f"catalogue - the index and the corpus disagree")
    print(f"\n  {len(catalogue)} of {full} records reachable from these queries")
    html = rewrite_footnote(html)

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
