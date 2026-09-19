#!/usr/bin/env python3
"""Build relevance judgements for the real catalogue by pooling and judging.

    python scripts/build_eval_set.py --queries data/eval_real_queries_draft.json

Standard TREC-style pooling. For each query, take the top-K from *both* lanes,
union them, and judge every (query, record) pair. Three properties make the
result usable as ground truth:

1. **The judge never learns which lane retrieved a record.** It sees a query and
   a record, nothing else, and the pool is shuffled, so it cannot systematically
   favour keyword or semantic.
2. **Judging is independent of the thing under test.** Retrieval is MiniLM
   embeddings and BM25; the judge is Claude. A system cannot mark its own work.
3. **Grades are 0/1/2, not binary**, so "exactly what I asked for" and "adjacent,
   might do" are distinguishable later even though the current harness reads
   binary.

The honest limit, recorded in the output and the README: **records that neither
lane retrieves are never judged.** Recall is therefore relative to the pool, not
absolute. That is the standard pooling caveat and it does not go away by being
ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

JUDGE_MODEL = "claude-haiku-4-5"
POOL_PER_LANE = 20
#: Grade at or above this counts as relevant for the binary harness. Strict on
#: purpose: the previous eval set saturated, and a lenient threshold saturates
#: sooner. The lenient figure is reported alongside so both are visible.
STRICT_GRADE = 2

RUBRIC = """You are judging search results for a public library catalogue.

You will be given a reader's query and one catalogue record. Decide how well the \
record answers what the reader is actually looking for.

Grade:
  2 - Clearly what they asked for. A librarian would hand them this.
  1 - Plausibly useful. Adjacent subject, or right topic but wrong level/audience.
  0 - Not useful. Different subject, or matches only on an incidental word.

Judge the reader's INTENT, not word overlap. "Something gentle to read before \
bed" is asking about tone, so a calm novel or a book of soothing essays scores \
2 even with no shared words, while a thriller scores 0 however often it says \
"bed" or "night".

For a query naming an exact title or author, the named work scores 2; other \
works by that author score 1; unrelated works score 0.

Some queries carry emotional weight, and getting those wrong harms the reader \
rather than mildly annoying them. "Coping after someone dies" is asked by \
someone who has been bereaved. Books that help a person grieve score 2. Fiction \
in which death is a plot device, a horror premise, or played for comedy scores \
**0, not 1** - however prominently "Death" sits in its subject headings. \
Adjacent by keyword is not adjacent by need. Apply the same reasoning to any \
query about illness, loss, addiction or mental health: a novel that features \
the subject is not a book that helps with it.

Catalogue records are thin - often just a title, an author and subject \
headings, with no summary. Judge on what a reader could reasonably infer from \
that, as they would in a real catalogue. If the record is too sparse to tell, \
grade 0 rather than guessing generously.

Reply with a JSON object only: {"grade": 0|1|2, "why": "<at most 12 words>"}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer", "enum": [0, 1, 2]},
        "why": {"type": "string"},
    },
    "required": ["grade", "why"],
    "additionalProperties": False,
}


def load_queries(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["queries"] if isinstance(raw, dict) else raw


def build_pools(lanes: list[tuple[str, str, str]], queries: list[dict],
                *, per_lane: int) -> dict:
    """top-K from every lane that will later be measured, unioned and shuffled.

    Every arm under measurement must contribute to the pool. Pooling only judges
    what the pooled lanes retrieve, so if the enriched arm is measured but never
    pooled, any record it surfaces that the thin lanes missed goes unjudged and
    scores as irrelevant - penalising enrichment exactly where it works. The
    number would look defensible and be wrong.

    `lanes` is (label, index_path, mode). The label is used for reporting only;
    it is discarded before judging so the judge cannot tell lanes apart.
    """
    from bettersearch.config import load_settings
    from bettersearch.index.numpy_index import NumpyVectorIndex
    from bettersearch.search import Searcher

    settings = load_settings()
    searchers = {path: Searcher(index=NumpyVectorIndex(path), settings=settings)
                 for _, path, _ in lanes}
    rng = random.Random(20260919)

    pools: dict[str, list[str]] = {}
    contributed: dict[str, int] = {label: 0 for label, _, _ in lanes}
    unique_to: dict[str, int] = {label: 0 for label, _, _ in lanes}
    sizes: list[int] = []

    for item in queries:
        q = item["query"]
        per_lane_ids: dict[str, list[str]] = {}
        for label, path, mode in lanes:
            got = [r.chunk.doc_id
                   for r in searchers[path].search(q, mode=mode, top_k=per_lane).results]
            per_lane_ids[label] = got
            contributed[label] += len(got)

        # Records only one lane found - the ones that would be lost if that lane
        # were left out of the pool.
        for label in per_lane_ids:
            others = set().union(*(set(v) for k, v in per_lane_ids.items() if k != label)) \
                     if len(per_lane_ids) > 1 else set()
            unique_to[label] += len(set(per_lane_ids[label]) - others)

        ids = [i for label, _, _ in lanes for i in per_lane_ids[label]]
        seen: set[str] = set()
        pool = [i for i in ids if not (i in seen or seen.add(i))]
        rng.shuffle(pool)  # pool position must not encode which lane found it
        pools[q] = pool
        sizes.append(len(pool))

    return {"pools": pools, "sizes": sizes,
            "contributed": contributed, "unique_to": unique_to}


def judge_pairs(pairs: list[tuple[str, str, str]], *, model: str, wait: int = 30) -> dict:
    """Judge (key, query, record_text) triples via the Batch API. Returns key -> grade dict."""
    import anthropic

    client = anthropic.Anthropic()
    requests = [
        {
            "custom_id": key,
            "params": {
                "model": model,
                "max_tokens": 200,
                "system": [{"type": "text", "text": RUBRIC,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user",
                              "content": f"Reader's query:\n{query}\n\nCatalogue record:\n{record}"}],
                "output_config": {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            },
        }
        for key, query, record in pairs
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted ({len(requests)} judgements); polling")
    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status in {"ended", "canceled", "expired"}:
            break
        time.sleep(wait)

    out, failed, usage = {}, [], {"in": 0, "out": 0, "cw": 0, "cr": 0}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            failed.append(entry.custom_id)
            continue
        u = entry.result.message.usage
        usage["in"] += u.input_tokens; usage["out"] += u.output_tokens
        usage["cw"] += u.cache_creation_input_tokens or 0
        usage["cr"] += u.cache_read_input_tokens or 0
        try:
            text = "".join(b.text for b in entry.result.message.content if b.type == "text")
            parsed = json.loads(text)
            out[entry.custom_id] = {"grade": int(parsed["grade"]), "why": parsed.get("why", "")}
        except (json.JSONDecodeError, KeyError, ValueError):
            failed.append(entry.custom_id)
    return {"grades": out, "failed": failed, "usage": usage, "batch_id": batch.id}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=ROOT / "data/eval_real_queries_draft.json")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/catalogue_real.json")
    ap.add_argument(
        "--lane", action="append", default=None,
        help="label=index_path:mode, repeatable. Every arm that will be measured "
             "must be a lane, or its unique finds go unjudged and score as "
             "irrelevant. Default: all three.")
    ap.add_argument("--validate", type=int, default=100,
                    help="re-judge this many random pairs to measure the judge's "
                         "self-consistency (0 to skip)")
    ap.add_argument("--spot-check-out", type=Path,
                    default=ROOT / "data/eval_real_spotcheck.md")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval_real.json")
    ap.add_argument("--judgements-out", type=Path, default=ROOT / "data/eval_real_judgements.json")
    ap.add_argument("--per-lane", type=int, default=POOL_PER_LANE)
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    corpus = {d["doc_id"]: d for d in json.loads(
        args.corpus.read_text(encoding="utf-8"))["documents"]}
    print(f"{len(queries)} queries · {len(corpus)} records · pooling top-{args.per_lane} per lane\n")

    if args.lane:
        lanes = []
        for spec in args.lane:
            label, rest = spec.split("=", 1)
            path, mode = rest.rsplit(":", 1)
            lanes.append((label, path, mode))
    else:
        lanes = [
            ("keyword", ".bettersearch/real", "keyword"),
            ("semantic-thin", ".bettersearch/real", "semantic"),
            ("semantic-enriched", ".bettersearch/real-enriched", "semantic"),
        ]
    print("lanes pooled: " + ", ".join(f"{l}({m})" for l, _, m in lanes))

    built = build_pools(lanes, queries, per_lane=args.per_lane)
    pools = built["pools"]
    sizes = sorted(built["sizes"])
    print(f"pool size: min {sizes[0]} median {sizes[len(sizes)//2]} max {sizes[-1]}")
    print("records only one lane found (these would be lost if it were "
          "left out of the pool):")
    for label, n in built["unique_to"].items():
        print(f"   {label:<20} {n}")

    pairs = []
    for qi, item in enumerate(queries):
        for doc_id in pools[item["query"]]:
            record = corpus[doc_id]
            text = f"{record['title']}\n{record['text']}"
            pairs.append((f"q{qi}--{doc_id}", item["query"], text))
    print(f"\n{len(pairs)} judgements to make\n")

    result = judge_pairs(pairs, model=args.model)
    grades = result["grades"]
    if result["failed"]:
        print(f"  ! {len(result['failed'])} judgements failed", file=sys.stderr)

    u = result["usage"]
    # Haiku 4.5: $1 in / $5 out per MTok; batch halves it; cache write 1.25x, read 0.1x
    usd = ((u["in"] + u["cw"] * 1.25 + u["cr"] * 0.1) / 1e6 * 1.0 + u["out"] / 1e6 * 5.0) * 0.5
    print(f"  judged {len(grades)}  ·  spend ${usd:.2f}  ·  "
          f"cache hit {u['cr']/(u['cr']+u['cw'])*100:.0f}%" if (u["cr"] + u["cw"]) else "")

    # ---- validate the judge before anyone believes it ----------------------
    consistency = None
    if args.validate and len(pairs) > args.validate:
        rng = random.Random(7)
        sample = rng.sample([p for p in pairs if f"{p[0]}" in grades], args.validate)
        recheck = judge_pairs([(k + "--recheck", q, r) for k, q, r in sample],
                              model=args.model)["grades"]
        agree = exact = 0
        for key, _, _ in sample:
            a = grades[key]["grade"]
            b = recheck.get(key + "--recheck", {}).get("grade")
            if b is None:
                continue
            agree += 1
            exact += (a == b)
        consistency = exact / agree if agree else None
        print(f"\n  judge self-consistency: {exact}/{agree} identical "
              f"({consistency:.0%})" if consistency is not None else "")

    out_queries, dist = [], {0: 0, 1: 0, 2: 0}
    for qi, item in enumerate(queries):
        per_doc = {}
        for doc_id in pools[item["query"]]:
            g = grades.get(f"q{qi}--{doc_id}")
            if g is None:
                continue
            per_doc[doc_id] = g["grade"]
            dist[g["grade"]] += 1
        strict = sorted([d for d, g in per_doc.items() if g >= STRICT_GRADE])
        out_queries.append({
            "query": item["query"],
            "relevant_doc_ids": strict,
            "note": f"{item.get('category','')} · {item.get('note','')}".strip(" ·"),
            "grades": per_doc,
        })

    args.out.write_text(json.dumps({
        "name": "real-catalogue-eval",
        "description": (
            f"Relevance judgements over {len(corpus)} real Open Library records. "
            f"Pooled top-{args.per_lane} from each retrieval lane, judged by "
            f"{args.model} blind to which lane retrieved each record. "
            f"relevant_doc_ids uses grade >= {STRICT_GRADE} (clearly relevant); "
            "per-document grades are kept in `grades`."
        ),
        "caveat": (
            "POOLING BIAS: records retrieved by neither lane were never judged, "
            "so recall is relative to the pool rather than absolute. The judge is "
            "an LLM validated by sampling, not human ground truth. Queries were "
            "drafted by the system's author and edited by the reviewer."
        ),
        "judge_model": args.model,
        "judge_self_consistency": consistency,
        "lanes_pooled": [l for l, _, _ in lanes],
        "pool_per_lane": args.per_lane,
        "strict_grade": STRICT_GRADE,
        "queries": out_queries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    args.judgements_out.write_text(json.dumps({
        "batch_id": result["batch_id"],
        "judge_model": args.model,
        "judgements": {k: v for k, v in sorted(grades.items())},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(dist.values()) or 1
    print(f"\ngrade distribution: "
          f"0={dist[0]} ({dist[0]/total:.0%})  "
          f"1={dist[1]} ({dist[1]/total:.0%})  "
          f"2={dist[2]} ({dist[2]/total:.0%})")
    rel = [len(q["relevant_doc_ids"]) for q in out_queries]
    print(f"relevant per query (grade>={STRICT_GRADE}): "
          f"min {min(rel)} median {sorted(rel)[len(rel)//2]} max {max(rel)}")
    zero = [q["query"] for q in out_queries if not q["relevant_doc_ids"]]
    if zero:
        print(f"\n{len(zero)} queries with NO relevant record - drop or rewrite these:")
        for q in zero:
            print(f"   {q!r}")
    # ---- 50 pairs for a person to check by eye -----------------------------
    rng = random.Random(11)
    by_key = {k: (q, r) for k, q, r in pairs}
    sample = rng.sample(sorted(grades), min(50, len(grades)))
    lines = ["# Judge spot check", "",
             f"50 random judgements from `{args.model}`, for you to sanity-check by eye.",
             "The question is not whether you agree with every one - it is whether the",
             "judge is reading **intent** or just matching words. If it is grading a",
             "thriller as relevant to \"something gentle to read before bed\" because the",
             "record says \"night\", the eval is not measuring what we think it is.", ""]
    if consistency is not None:
        lines += [f"Measured self-consistency on a re-judged sample: **{consistency:.0%}** "
                  f"identical.", ""]
    for key in sample:
        q, record = by_key[key]
        g = grades[key]
        lines += [f"**{g['grade']}** — _{q}_",
                  "```", record.strip()[:260], "```",
                  f"judge: {g['why']}", ""]
    args.spot_check_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {args.out}, {args.judgements_out} and {args.spot_check_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
