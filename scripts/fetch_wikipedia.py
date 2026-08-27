#!/usr/bin/env python3
"""Build a corpus from real Wikipedia article extracts.

The seed corpus in data/corpus_seed.json was written by the same author as the
eval labels, so good scores on it are partly circular. This script produces a
corpus nobody involved wrote, in the identical JSON schema, so the evaluation
can be repeated against content that was not shaped to fit the questions.

    python scripts/fetch_wikipedia.py --count 2000 --out data/corpus_wikipedia.json

Uses only the standard library. Be polite: the API asks for a descriptive
User-Agent and this script batches twenty titles per request.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BetterSearch-PoC/0.1 (corpus builder; standard library urllib)"

# Wikipedia caps extract-bearing queries at 20 titles per request.
_TITLE_BATCH = 20
_MIN_CHARS = 400


def _request(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def random_titles(count: int) -> list[str]:
    """Collect random article titles, de-duplicated."""
    titles: list[str] = []
    seen: set[str] = set()
    while len(titles) < count:
        payload = _request(
            {
                "action": "query",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": min(500, count - len(titles) + 50),
                "format": "json",
            }
        )
        for item in payload["query"]["random"]:
            title = item["title"]
            if title not in seen:
                seen.add(title)
                titles.append(title)
        time.sleep(0.2)
    return titles[:count]


def search_titles(term: str, count: int) -> list[str]:
    """Collect titles matching a search term, for a topic-focused corpus."""
    titles: list[str] = []
    offset = 0
    while len(titles) < count:
        payload = _request(
            {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srlimit": min(500, count - len(titles)),
                "sroffset": offset,
                "srnamespace": 0,
                "format": "json",
            }
        )
        hits = payload.get("query", {}).get("search", [])
        if not hits:
            break
        titles.extend(hit["title"] for hit in hits)
        offset += len(hits)
        time.sleep(0.2)
    return titles[:count]


def fetch_extracts(titles: list[str]) -> list[dict]:
    """Fetch plain-text intro extracts for the given titles."""
    documents: list[dict] = []
    for start in range(0, len(titles), _TITLE_BATCH):
        batch = titles[start : start + _TITLE_BATCH]
        try:
            payload = _request(
                {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "titles": "|".join(batch),
                    "format": "json",
                }
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  batch failed ({exc}); skipping", file=sys.stderr)
            continue

        for page in payload.get("query", {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            title = page.get("title", "")
            # Skip stubs and disambiguation pages - they add noise, not signal.
            if len(text) < _MIN_CHARS or "may refer to:" in text[:200]:
                continue
            documents.append(
                {
                    "doc_id": urllib.parse.quote(title.replace(" ", "_")).lower(),
                    "title": title,
                    "text": text,
                    "url": "https://en.wikipedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_")),
                    "tags": ["wikipedia"],
                }
            )

        print(f"  {len(documents)} usable documents so far", file=sys.stderr)
        time.sleep(0.2)
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="articles to fetch")
    parser.add_argument("--out", default="data/corpus_wikipedia.json")
    parser.add_argument(
        "--search",
        help="fetch articles matching this term instead of random ones",
    )
    args = parser.parse_args()

    if args.search:
        print(f"Searching for {args.search!r}...", file=sys.stderr)
        titles = search_titles(args.search, args.count)
    else:
        print(f"Collecting {args.count} random titles...", file=sys.stderr)
        titles = random_titles(args.count)

    print(f"Fetching extracts for {len(titles)} titles...", file=sys.stderr)
    documents = fetch_extracts(titles)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": "wikipedia",
                "description": (
                    "Wikipedia intro extracts. Text is CC BY-SA 4.0; see each "
                    "document's url for attribution."
                ),
                "documents": documents,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(documents)} documents to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
