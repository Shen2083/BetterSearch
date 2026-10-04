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


def build_pools(lanes: list[dict], queries: list[dict],
                *, per_lane: int) -> dict:
    """top-K from every lane that will later be measured, unioned and shuffled.

    Every arm under measurement must contribute to the pool. Pooling only judges
    what the pooled lanes retrieve, so if the enriched arm is measured but never
    pooled, any record it surfaces that the thin lanes missed goes unjudged and
    scores as irrelevant - penalising enrichment exactly where it works. The
    number would look defensible and be wrong.

    A lane is a dict: either `{kind: "index", label, path, mode, model}` or
    `{kind: "file", label, path}`. The label is used for reporting only; it is
    discarded before judging so the judge cannot tell lanes apart. `model` may
    be None to use the configured default - it exists because an index built
    with a different encoder has to be queried with that same encoder, and a
    lane using a larger model is exactly the case this pool has to cover.

    A **file** lane reads `{query: [doc_id, ...]}` - the shape
    `scripts/tune_reranking.py --dump` writes and `scripts/score_rankings.py`
    reads. It exists because the arms worth pooling are no longer all
    expressible as an index and a mode: a reranked lane is a retrieval plus a
    cross-encoder plus a depth, and the pool has to be able to contain it or
    everything that lane alone surfaces goes unjudged and scores as irrelevant.
    """
    # Imported inside, and only when an index lane exists: a pool made entirely
    # of file lanes is just a union of rankings, and should not need an encoder,
    # an index on disk, or a loadable config to compute.
    searchers = {}
    if any(l["kind"] == "index" for l in lanes):
        from dataclasses import replace

        from bettersearch.config import load_settings
        from bettersearch.index.numpy_index import NumpyVectorIndex
        from bettersearch.search import Searcher

        base = load_settings()
        for lane in lanes:
            if lane["kind"] != "index":
                continue
            key = (lane["path"], lane["model"])
            if key not in searchers:
                settings = (replace(base, local_model_name=lane["model"])
                            if lane["model"] else base)
                searchers[key] = Searcher(index=NumpyVectorIndex(lane["path"]),
                                          settings=settings)

    files = {}
    for lane in lanes:
        if lane["kind"] == "file":
            files[lane["label"]] = json.loads(
                Path(lane["path"]).read_text(encoding="utf-8"))
    rng = random.Random(20260919)

    pools: dict[str, list[str]] = {}
    contributed: dict[str, int] = {l["label"]: 0 for l in lanes}
    unique_to: dict[str, int] = {l["label"]: 0 for l in lanes}
    missing_from_file: dict[str, int] = {l["label"]: 0 for l in lanes
                                         if l["kind"] == "file"}
    sizes: list[int] = []

    for item in queries:
        q = item["query"]
        per_lane_ids: dict[str, list[str]] = {}
        for lane in lanes:
            label = lane["label"]
            if lane["kind"] == "file":
                ranked = files[label].get(q)
                if ranked is None:
                    # A lane that cannot answer a query would silently shrink
                    # the pool for it. Counted and reported rather than skipped.
                    missing_from_file[label] += 1
                    got = []
                else:
                    got = list(dict.fromkeys(ranked))[:per_lane]
            else:
                got = [r.chunk.doc_id
                       for r in searchers[(lane["path"], lane["model"])].search(
                           q, mode=lane["mode"], top_k=per_lane).results]
            per_lane_ids[label] = got
            contributed[label] += len(got)

        # Records only one lane found - the ones that would be lost if that lane
        # were left out of the pool.
        for label in per_lane_ids:
            others = set().union(*(set(v) for k, v in per_lane_ids.items() if k != label)) \
                     if len(per_lane_ids) > 1 else set()
            unique_to[label] += len(set(per_lane_ids[label]) - others)

        ids = [i for lane in lanes for i in per_lane_ids[lane["label"]]]
        seen: set[str] = set()
        pool = [i for i in ids if not (i in seen or seen.add(i))]
        rng.shuffle(pool)  # pool position must not encode which lane found it
        pools[q] = pool
        sizes.append(len(pool))

    return {"pools": pools, "sizes": sizes, "contributed": contributed,
            "unique_to": unique_to, "missing_from_file": missing_from_file}


def load_reuse(eval_path: Path, judgements_path: Path) -> dict:
    """Grades already paid for, keyed on (query text, doc_id).

    Read from the eval file's per-query `grades`, **not** from the judgements
    file's `q{i}--{doc_id}` keys. Those carry the query's *index*: reorder or
    insert a query and every key downstream silently attaches to the wrong one,
    which would corrupt the ground truth invisibly. The `grades` dict sits
    beside its own query text and cannot drift that way. The judgements file is
    consulted only to carry the judge's reason across, positionally, from the
    ordering that produced it.
    """
    raw = eval_path.read_text(encoding="utf-8") if eval_path.exists() else ""
    if not raw.strip():
        # Covers both a first run and `--reuse /dev/null`, the documented way
        # to re-judge everything from scratch.
        return {}
    data = json.loads(raw)
    whys = {}
    if judgements_path.exists():
        whys = json.loads(judgements_path.read_text(encoding="utf-8")).get("judgements", {})
    out = {}
    for i, item in enumerate(data.get("queries", [])):
        for doc_id, grade in (item.get("grades") or {}).items():
            prior = whys.get(f"q{i}--{doc_id}", {})
            out[(item["query"], doc_id)] = {
                "grade": int(grade),
                "why": prior.get("why", ""),
            }
    return out


def keep_judged(pools: dict, queries: list[dict], reuse: dict,
                seed: int = 20260919) -> int:
    """Union every already-judged record back into the pool it belongs to.

    Re-pooling adds a lane, and the danger is subtler than it looks: the new
    pool is built from whatever lanes *this* run has, so a record the old lanes
    found and the new ones do not would silently drop out. Its grade would still
    be carried forward by `split_pairs`, but it would vanish from
    `relevant_doc_ids`, shrinking the denominator every published figure was
    measured against. Arms would appear to improve because the bar moved.

    So the pool only ever grows. A judgement is paid for and permanent; keeping
    it costs nothing and discarding it quietly rewrites history. This also means
    the lanes that built the previous pool do not have to still exist on disk to
    keep contributing to it - which is just as well, since three of them did not
    survive a container restart.

    Returns how many (query, record) pairs were restored.
    """
    by_query: dict[str, list[str]] = {}
    for (query, doc_id) in reuse:
        by_query.setdefault(query, []).append(doc_id)

    rng = random.Random(seed)
    restored = 0
    for item in queries:
        q = item["query"]
        pool = pools.setdefault(q, [])
        have = set(pool)
        extra = [d for d in by_query.get(q, []) if d not in have]
        restored += len(extra)
        pool.extend(extra)
        # Re-shuffled so a record's position still cannot tell the judge which
        # lane found it - or, now, whether it is new this run at all.
        rng.shuffle(pool)
    return restored


def split_pairs(queries: list[dict], pools: dict, corpus: dict,
                reuse: dict) -> tuple[list[tuple[str, str, str]], dict]:
    """Divide the pool into grades already paid for and pairs still to judge.

    This is the function that decides what the run spends, so it errs towards
    judging: a pair is carried forward only on an exact (query text, doc_id)
    hit in `reuse`. Anything else - a record the pool has newly surfaced, a
    query whose wording changed by a character - is new, and gets looked at.
    The opposite default would silently record an unjudged record as whatever
    `reuse` happened to return.

    Keys stay `q{index}--{doc_id}` because that is what the Batch API round
    trip carries and what the judgements file has always used. They are built
    here from this run's query order and consumed in it, so the index never
    outlives the ordering that produced it - which is exactly why `reuse` is
    keyed on the query's text instead.
    """
    pairs: list[tuple[str, str, str]] = []
    grades: dict[str, dict] = {}
    for qi, item in enumerate(queries):
        for doc_id in pools[item["query"]]:
            key = f"q{qi}--{doc_id}"
            if (prior := reuse.get((item["query"], doc_id))) is not None:
                grades[key] = prior
                continue
            record = corpus[doc_id]
            pairs.append((key, item["query"],
                          f"{record['title']}\n{record['text']}"))
    return pairs, grades


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
        help="label=index_path:mode[@model], repeatable. Every arm that will be "
             "measured must be a lane, or its unique finds go unjudged and "
             "score as irrelevant. Append @model when the index was built with "
             "an encoder other than the configured default.")
    ap.add_argument("--validate", type=int, default=100,
                    help="re-judge this many random pairs to measure the judge's "
                         "self-consistency (0 to skip)")
    ap.add_argument("--spot-check-out", type=Path,
                    default=ROOT / "data/eval_real_spotcheck.md")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval_real.json")
    ap.add_argument("--judgements-out", type=Path, default=ROOT / "data/eval_real_judgements.json")
    ap.add_argument(
        "--lane-file", action="append", default=None, metavar="LABEL=PATH.json",
        help="a lane supplied as precomputed rankings, {query: [doc_id, ...]} - "
             "the shape tune_reranking.py --dump writes. Use this for arms that "
             "are not just an index and a mode, such as a reranked lane.")
    ap.add_argument(
        "--reuse", type=Path, default=ROOT / "data/eval_real.json",
        help="carry grades forward from this eval file and judge only unseen "
             "pairs. Keeps published figures comparable and is far cheaper; "
             "pass /dev/null to re-judge everything.")
    ap.add_argument("--no-keep-judged", dest="keep_judged", action="store_false",
                    help="let records drop out of the pool when no current lane "
                         "retrieves them. Shrinks the denominator every "
                         "published figure was measured against - only pass it "
                         "to build a pool from scratch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="pool, report what is new and what it would cost, and "
                         "stop before spending anything")
    ap.add_argument("--per-lane", type=int, default=POOL_PER_LANE)
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    corpus = {d["doc_id"]: d for d in json.loads(
        args.corpus.read_text(encoding="utf-8"))["documents"]}
    print(f"{len(queries)} queries · {len(corpus)} records · pooling top-{args.per_lane} per lane\n")

    lanes = []
    for spec in args.lane or []:
        label, rest = spec.split("=", 1)
        rest, model = rest.rsplit("@", 1) if "@" in rest else (rest, None)
        path, mode = rest.rsplit(":", 1)
        lanes.append({"kind": "index", "label": label, "path": path,
                      "mode": mode, "model": model})
    for spec in args.lane_file or []:
        label, path = spec.split("=", 1)
        lanes.append({"kind": "file", "label": label, "path": path})
    if not lanes and not args.keep_judged:
        lanes = [
            {"kind": "index", "label": "keyword", "path": ".bettersearch/real",
             "mode": "keyword", "model": None},
            {"kind": "index", "label": "semantic-thin", "path": ".bettersearch/real",
             "mode": "semantic", "model": None},
            {"kind": "index", "label": "semantic-enriched",
             "path": ".bettersearch/real-enriched", "mode": "semantic", "model": None},
        ]
    print("lanes pooled: " + (", ".join(
        l["label"] + ("(file)" if l["kind"] == "file" else
                      f"({l['mode']}" + ("" if not l["model"]
                                         else ", " + l["model"].rsplit("/", 1)[-1]) + ")")
        for l in lanes) or "none this run - the pool is what has already been judged"))

    built = build_pools(lanes, queries, per_lane=args.per_lane)
    pools = built["pools"]
    sizes = sorted(built["sizes"])
    print(f"pool size: min {sizes[0]} median {sizes[len(sizes)//2]} max {sizes[-1]}")
    print("records only one lane found (these would be lost if it were "
          "left out of the pool):")
    for label, n in built["unique_to"].items():
        print(f"   {label:<20} {n}")

    for label, n in built["missing_from_file"].items():
        if n:
            print(f"   ! {label}: no ranking for {n} quer{'y' if n == 1 else 'ies'} "
                  f"- the pool is thinner there")

    reuse = load_reuse(args.reuse, args.judgements_out) if args.reuse else {}
    inherited, prior_consistency = [], None
    if reuse and args.keep_judged:
        restored = keep_judged(pools, queries, reuse)
        prior = json.loads(args.reuse.read_text(encoding="utf-8"))
        prior_consistency = prior.get("judge_self_consistency")
        inherited = [l for l in prior.get("lanes_pooled") or []
                     if l not in {x["label"] for x in lanes}]
        sizes = sorted(len(pools[q["query"]]) for q in queries)
        print(f"kept {len(reuse)} judged records in the pool "
              f"({restored} no current lane retrieved) · "
              f"pool size now min {sizes[0]} median {sizes[len(sizes)//2]} "
              f"max {sizes[-1]}")
        if inherited:
            print("   inherited pool support from: " + ", ".join(inherited))

    pairs, grades = split_pairs(queries, pools, corpus, reuse)
    carried = len(grades)

    # An estimate, not a quote: the real figure is printed from the API's own
    # usage after the batch. Judging is small - a cached rubric, a one-line
    # record, a two-field answer - so this is pounds-and-pence territory and the
    # point of printing it is to show that, not to budget against it.
    est = len(pairs) * ((50 + 400 * 0.1) * 1.0 + 40 * 5.0) / 1e6 * 0.5
    print(f"\n{carried} grades carried forward · {len(pairs)} new to judge "
          f"· estimated ${est:.2f} at {args.model} batch rates")
    if not pairs:
        print("  nothing new in the pool - the lanes add no records this eval "
              "has not already seen")
    if args.dry_run:
        print("\n--dry-run: stopping before spending anything")
        return 0

    result = {"grades": {}, "failed": [], "batch_id": None,
              "usage": {"in": 0, "out": 0, "cw": 0, "cr": 0}}
    if pairs:
        result = judge_pairs(pairs, model=args.model)
        grades.update(result["grades"])
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
        # Only pairs judged in this run - re-judging a carried-forward grade
        # would measure consistency across runs, which is a different number.
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
            + (f"The pool is cumulative: {len(reuse)} records judged for earlier "
               f"lanes are retained, so adding a lane can only widen it. " if inherited
               else "")
            + f"relevant_doc_ids uses grade >= {STRICT_GRADE} (clearly relevant); "
            "per-document grades are kept in `grades`."
        ),
        "caveat": (
            "POOLING BIAS: records retrieved by neither lane were never judged, "
            "so recall is relative to the pool rather than absolute. The judge is "
            "an LLM validated by sampling, not human ground truth. Queries were "
            "drafted by the system's author and edited by the reviewer."
        ),
        "judge_model": args.model,
        # Measured on pairs judged in *this* run, so a run that added few pairs
        # measures it on few and a run that added none cannot measure it at all.
        # The previous run's figure is kept beside it rather than overwritten:
        # most grades in this file were made under that measurement, and
        # dropping it would leave the file with no stated judge reliability.
        "judge_self_consistency": consistency,
        "judge_self_consistency_prior": prior_consistency,
        # Every lane whose retrievals this pool contains - the ones run now,
        # plus the ones whose contribution was inherited with the judgements.
        # Dropping the inherited names would make the pool look narrower than
        # it is and invite someone to re-judge what is already paid for.
        "lanes_pooled": [l["label"] for l in lanes] + inherited,
        "lanes_run": [l["label"] for l in lanes],
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
    # Sampled from pairs judged in *this* run. Carried-forward grades were
    # eyeballed when they were made; re-presenting them would pad the file with
    # work already reviewed and hide whatever the new lane actually brought in.
    by_key = {k: (q, r) for k, q, r in pairs}
    judged_now = sorted(k for k in by_key if k in grades)
    sample = rng.sample(judged_now, min(50, len(judged_now)))
    lines = ["# Judge spot check", "",
             f"50 random judgements from `{args.model}`, for you to sanity-check by eye.",
             "The question is not whether you agree with every one - it is whether the",
             "judge is reading **intent** or just matching words. If it is grading a",
             "thriller as relevant to \"something gentle to read before bed\" because the",
             "record says \"night\", the eval is not measuring what we think it is.", ""]
    if not judged_now:
        lines = ["# Judge spot check", "",
                 "Nothing new was judged in this run - every pair in the pool "
                 "already had a grade, carried forward. The previous spot check "
                 "still describes the judgements in use."]
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
