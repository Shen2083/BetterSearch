#!/usr/bin/env python3
"""Sanity-check a catalogue corpus before anyone measures anything with it.

    python scripts/check_corpus.py --corpus data/catalogue_real.json \
        --queries data/eval_real_queries_draft.json

This exists because of a real miss. The first 4,000-record fetch looked healthy
by every summary statistic it printed - record count, word counts, 1% missing
authors - and was badly skewed: per-subject allocation let the first 24 of 50
subjects consume the whole budget, so astronomy had 8 records and job hunting,
local history and football had none at all. Those are areas the eval queries
target, so the eval would have been measuring an absence and reporting it as a
retrieval failure.

No aggregate statistic catches that. The only check that does is asking whether
the corpus can plausibly answer the questions we intend to ask it, which is what
this does: it probes each query's key terms against titles and subject headings
and reports the ones with nothing behind them.

It is a cheap structural check, not a relevance judgement - a query with plenty
of term matches can still retrieve badly, which is what the eval set is for.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Words too common to say anything about coverage.
STOP = {
    "a", "about", "after", "again", "an", "and", "are", "as", "at", "be", "before",
    "book", "books", "but", "by", "can", "do", "for", "from", "get", "getting",
    "good", "he", "help", "how", "i", "if", "in", "is", "it", "like", "make", "me",
    "my", "no", "not", "of", "on", "one", "or", "out", "read", "she", "so",
    "some", "something", "that", "the", "their", "them", "then", "there", "they",
    "this", "to", "too", "up", "was", "what", "when", "where", "which", "who",
    "why", "will", "with", "without", "would", "you", "your",
}
MIN_HITS = 10


def terms(query: str) -> list[str]:
    words = re.findall(r"[a-z']+", query.lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def stem(word: str) -> str:
    """Crudest possible stem - enough to match 'gardening' against 'garden'."""
    for suffix in ("ing", "ers", "er", "ies", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/catalogue_real.json")
    ap.add_argument("--queries", type=Path,
                    default=ROOT / "data/eval_real_queries_draft.json")
    ap.add_argument("--min-hits", type=int, default=MIN_HITS)
    args = ap.parse_args()

    raw = json.loads(args.corpus.read_text(encoding="utf-8"))
    docs = raw["documents"] if isinstance(raw, dict) else raw

    # ---- structural integrity ------------------------------------------------
    problems = []
    for d in docs:
        if not d.get("title", "").strip():
            problems.append((d.get("doc_id", "?"), "empty title"))
        if not d.get("subjects"):
            problems.append((d.get("doc_id", "?"), "no subjects"))
        if not d.get("text", "").strip():
            problems.append((d.get("doc_id", "?"), "empty text"))
    ids = [d["doc_id"] for d in docs]
    dupes = [i for i, n in Counter(ids).items() if n > 1]

    print(f"{len(docs)} records from {args.corpus.name}")
    print(f"  duplicate doc_ids   {len(dupes)}")
    print(f"  structural problems {len(problems)}")
    for doc_id, why in problems[:5]:
        print(f"     {doc_id}: {why}")

    words = sorted(len(d["text"].split()) + len(d["title"].split()) for d in docs)
    print(f"  words per record    min {words[0]} median {words[len(words)//2]} max {words[-1]}")
    print(f"  missing author      {sum(1 for d in docs if not d.get('author'))}")
    print(f"  fiction split       {sum(1 for d in docs if d.get('fiction'))}"
          f" / {sum(1 for d in docs if not d.get('fiction'))}")

    # ---- can the corpus plausibly answer the queries? ------------------------
    # Author belongs in the haystack: "Agatha Christie" is a legitimate query and
    # the name lives in its own field, not in the title or the subject headings.
    haystack = [
        (d["title"] + " " + (d.get("author") or "") + " "
         + " ".join(d.get("subjects", []))).lower()
        for d in docs
    ]
    stemmed = [" ".join(stem(w) for w in re.findall(r"[a-z']+", h)) for h in haystack]

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = queries["queries"] if isinstance(queries, dict) else queries

    # Mood, read-alike and half-remembered queries are *designed* to share no
    # vocabulary with the records - that is the thing semantic retrieval is for.
    # Flagging them as corpus gaps would be exactly backwards, so they are
    # reported separately and never counted as a failure.
    EXPECT_NO_OVERLAP = ("mood", "read-alike", "half-remembered")

    print(f"\nterm coverage for {len(queries)} queries "
          f"(records whose title, author or subjects contain each term)\n")
    weak, by_design = [], []
    for item in queries:
        q = item["query"]
        ts = terms(q)
        if not ts:
            continue
        counts = {t: sum(1 for h in stemmed if stem(t) in h) for t in ts}
        if max(counts.values()) >= args.min_hits:
            continue
        cat = item.get("category", "")
        (by_design if cat.startswith(EXPECT_NO_OVERLAP) else weak).append((q, counts, cat))

    if weak:
        print(f"{len(weak)} GAPS — queries whose subject area is missing or thin.")
        print("These would look like retrieval failures but are corpus holes:\n")
        for q, counts, cat in weak:
            top = ", ".join(f"{t}={n}" for t, n in sorted(counts.items(),
                                                          key=lambda kv: -kv[1])[:4])
            print(f"  {q[:44]:<46} [{cat[:22]:<22}] {top}")
    else:
        print(f"no corpus gaps: every query expecting term overlap has one "
              f">= {args.min_hits} records")

    if by_design:
        print(f"\n{len(by_design)} queries with low overlap BY DESIGN "
              f"(mood / read-alike / half-remembered) — not gaps, these are the")
        print("ones that can only be answered by meaning:\n")
        for q, counts, cat in by_design:
            top = ", ".join(f"{t}={n}" for t, n in sorted(counts.items(),
                                                          key=lambda kv: -kv[1])[:3])
            print(f"  {q[:44]:<46} [{cat[:16]:<16}] {top}")

    print()
    # Only structural damage is a hard failure. A gap is a judgement call - some
    # queries are deliberately unanswerable, and finding that out is the point -
    # so it is reported loudly and left for a person to weigh.
    broken = bool(problems or dupes)
    if broken:
        print(f"FAIL: {len(problems)} structural problems, {len(dupes)} duplicate "
              f"ids. Do not index this corpus.", file=sys.stderr)
    elif weak:
        print(f"OK structurally, but {len(weak)} query gaps above are worth a look "
              f"before you trust any score they produce.")
    else:
        print("OK: structurally sound, and every query has something to find.")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
