# A Vespa-only proof of concept

`docs/VESPA-HANDOVER.md` hands the engineering team eight findings about Vespa.
Every one of them was reasoned from the outside: read off Vespa's documentation
and argued against our Python measurements. None had been run.

This directory runs them. One Vespa node, the same 4,000 book records, the same
41 judged queries, scored through the same harness that produced every
published figure in this repository.

Four of the eight now have a measured number against them. Late interaction and
phased ranking came out **better** than the handover expected; RRF came out
exactly as it predicted, in Vespa's own implementation; the per-node IDF
warning is **overstated** at this corpus size. A fifth, `sameElement`, is
**wrong** and needed no measurement to settle: it does not apply to this schema
at all. Three are not done, and are listed at the end with reasons.

One finding that is not in the handover came out of the port, and it is a
negative: fielding the record properly, which our Python provably cannot do,
does not help this corpus.

Nothing on the serving path changes. This is a parallel branch.

## Reproducing it

```bash
vespa/scripts/up.sh                       # dockerd, pull, run, deploy
python3 vespa/scripts/fetch_models.py     # 133 MB of ONNX, on the host
python3 vespa/scripts/make_feed.py        # 4,000 docs + the existing vectors
vespa feed vespa/feed/records.jsonl       # 41 s, including ColBERT inference

PYTHONPATH=src python3 vespa/scripts/run_eval.py \
    --mode dense --profile dense --out vespa/rankings/vespa-dense.json
PYTHONPATH=src python3 scripts/score_rankings.py vespa/rankings/vespa-dense.json
```

Three things about this machine, none of them about Vespa:

* **The daemon is not running at session start and there is no systemd.**
  `up.sh` starts `dockerd` by hand.
* **Docker Hub rate-limits anonymous pulls from this egress IP.**
  The image comes from `ghcr.io/vespa-engine/vespa`, pinned by digest.
* **Processes inside containers cannot reach the outbound proxy.** So Vespa
  cannot fetch a model by `url` at deploy time. `fetch_models.py` downloads on
  the host and the package references the files with `path=`.

Two choices in the application package exist only because of this box, and
should be deleted in production: the raised `resource-limits` in
`services.xml` (the container filesystem reports a 252 GB size with 15 GB
available, so Vespa computes 94% full and blocks every feed with HTTP 507,
while `df` reports 62% used) and `validation-overrides.xml`.

## How a Vespa lane gets scored

`RUNBOOK.md:155` anticipated this exactly: *"Any future system — a Vespa
ranking profile included — becomes poolable by dumping its rankings in that
shape, with no code change here."* It was right. `run_eval.py` writes
`{query: [doc_id, ...]}` and `scripts/score_rankings.py` reads it. No change to
the harness.

**Every number below is a floor, not an estimate**, except where coverage is
100%. No Anthropic key is available, so no new judgements could be bought, and
these lanes were never pooled: anything a Vespa profile surfaced that our six
pooled lanes never found counts as irrelevant. The judged-coverage column is
the qualifier, and `score_rankings.py` prints its own verdict — at or above 90%
comparable, 75 to 90% treat with caution, below that not comparable. The lane
files are committed so they can be pooled the moment a key exists.

The comparisons worth trusting are **Vespa against Vespa**, where both sides
are equally unjudged and the difference is close to unbiased.

## 1. The port is faithful, to four decimal places

| lane | nDCG@10 | coverage | controls | ours |
|---|---|---|---|---|
| dense, fed vectors | **0.6582** | 100.0% | 1.000 | **0.6582** |

The vectors are not recomputed. They are lifted out of
`.bettersearch/real-books-base.npz`, the index behind the published figure
(MD5-identical to `real-bge.npz`), so dense retrieval starts from bit-identical
input and the only variable is Vespa's retrieval and ranking. Vespa's exact
`nearestNeighbor` over an attribute with no `index` block reproduces what
`src/bettersearch/index/numpy_index.py` does by brute force, exactly.

`make_feed.py` refuses to write a feed it cannot vouch for: before emitting
anything it checks that all 4,000 records were embedded from exactly
`title + blank line + text`, and exits if one disagrees.

This is the gate. Everything below rests on it.

## 2. Fielding the record does not help — and the reason is in our data

`src/bettersearch/keyword.py:35-38` records that our BM25 cannot weight the
author or the subject headings at all, because neither is a field: both exist
only as lines of prose inside one `text` blob. Vespa indexes them separately,
so this is the first time the question could be asked.

| lane | nDCG@10 | coverage | controls |
|---|---|---|---|
| BM25, flat title + text | **0.4120** | 75.4% | 0.886 |
| BM25F, best of 27 weightings, **fitted** | 0.3952 | 75.1% | 0.726 |
| BM25F, equal weights | 0.3734 | 72.2% | 0.600 |
| ours, Python | 0.4433 | — | — |

All 27 combinations score below flat, and the best is fitted on the same 41
queries it is scored on — so the honest out-of-sample number is worse still.

The reason is not Vespa. The `text` field is a *derived* concatenation:
`Author: …` / `Published: <publisher>, <year>` / `Subjects: a; b; c`. Weighting
author and subjects separately double- and triple-counts terms `bm25(text)` has
already scored. Fielding would pay on a corpus where the fields carry distinct
content. Here they carry the same content twice.

**The sweep did reproduce one of our findings independently.** Raising the
title weight trades nDCG for known-item controls, in the same direction and
roughly the same magnitude as `keyword.py:47` records for our own sweep
(0.443 → 0.403 at weight 3). At title weight 2 and above, controls reach
**1.000 natively** — which is what `src/bettersearch/exact.py` exists to repair
in our system. Vespa gets there with a rank-profile weight instead of a
promotion stage.

## 3. RRF confirmed, in Vespa's own implementation

| lane | nDCG@10 | coverage | controls |
|---|---|---|---|
| dense only | 0.6582 | 100.0% | 1.000 |
| `reciprocal_rank_fusion()`, global-phase | **0.5620** | 90.5% | 1.000 |
| ours, Python RRF | 0.5490 | — | — |

Both at coverage the harness calls comparable. Fusing costs **0.096 nDCG**
against dense alone. `src/bettersearch/exact.py:52-60` says blending "drags a
good list down with a mostly-empty one"; Vespa's own `reciprocal_rank_fusion`
in a `global-phase` does the same thing by the same margin. The finding
survives contact.

## 4. Native late interaction is the result worth acting on

`rerank-count` swept over one deployment, retrieval depth 100:

| rerank-count | nDCG@10 | coverage | MRR@10 | round trip |
|---|---|---|---|---|
| none (dense) | 0.6582 | 100.0% | 0.7820 | 23 ms |
| **20** | **0.7131** | **100.0%** | 0.8460 | 45 ms |
| 50 | 0.6557 | 89.5% | 0.8240 | 48 ms |
| 100 | 0.6449 | 87.3% | 0.8106 | 58 ms |
| 200 | 0.6473 | 85.6% | 0.8190 | 70 ms |

At rerank-count 20 **both arms sit at 100% coverage**, so +0.0549 is a clean
within-Vespa A/B, and controls stay at 1.000 with no exact-match repair.

The decay below that is **confounded**, and the coverage column says how.
Reranking the top 20 of the dense list can only reorder records that were
already pooled. Past 20 it promotes records from deeper, which no judge ever
saw, and those count as irrelevant. Our Python sweep measured a monotonic decay
over the same depths (+0.066 / +0.047 / +0.035 / +0.010) and read it as a real
effect; the same shape here comes with coverage falling from 100% to 85.6%, so
part of what we published as decay may be the pool thinning rather than the
reranker weakening. That is a caution about our own number, not Vespa's.

**Two things make this the finding to act on.**

It works on records as they ship. Vespa's embedder read the thin record — a
title, an author, a semicolon list of subject headings, 24 words median — and
got +0.0549. Our own measurements on the same thin records:

| reranker, thin records | ours | Vespa, native |
|---|---|---|
| ms-marco-MiniLM | +0.005 | — |
| bge-reranker-base | −0.001 | — |
| answerai-colbert-small-v1 | +0.036 | **+0.0549** |

Vespa's native embedder beats the MaxSim we wrote by hand against `pylate`, and
lands within 0.002 of what our MiniLM achieves **only after** every one of the
4,000 records has been enriched by an LLM (+0.0567, `rerank.py:9-16`). That is
a budget question for a library, not a config value: late interaction gets
most of the enrichment win without the enrichment bill.

And **rerank-count 200 completes**. In our implementation depth 200 did not
finish — twice, alone, on an idle box, with no theory ever offered
(`scripts/tune_reranking.py`). Here it is a 70 ms query. The handover's advice
to let the Vespa team evaluate ColBERT properly, because for them it is a
supported path rather than an extra dependency and someone else's MaxSim, is
borne out.

## 5. The per-node IDF warning is overstated

This is the one the handover says it most wants read, and it does not hold at
this corpus size.

`vespa/scripts/probe_significance.py` reads the significance Vespa is really
using for each query term out of `term(i).significance` in match-features,
recovers each term's document frequency, resamples it from `Binomial(df, 0.5)`
to simulate a node holding half the corpus, and re-issues every query with
those values set through Vespa's own per-term `{significance: x}` annotation.

```
significance ~ 0.7454 + -0.02617 * log(df)
  worst residual over 20 probe terms: 0.0254
  a term whose df halves moves by 0.0181, smaller than that residual

global significance                     nDCG@10 0.4002
half-corpus node, 5 random splits       -0.0009 -0.0051 -0.0012 -0.0011 +0.0002
mean shift -0.0016, mean top-10 overlap with global 98.0%
```

Significance spans only 0.574 to 0.687 across a **124-fold** range of document
frequency — Vespa compresses it hard. Halving the documents a node sees moves a
term by about 0.018, which is less than the scatter in the relationship itself.
A multi-node port will not miss the 0.443 bar for this reason.

Two caveats, both real. The absolute 0.4002 is not comparable to the 0.4120
flat lane, because the annotated query form differs; only the within-form A/B
is sound. And the document-frequency recovery rests on a fit whose residuals
are the same size as the effect, so this establishes that the effect is *not
large* and cannot resolve whether it is small or zero.

## Not done, and why

* **`sameElement` does not apply.** It needs an array of structs. `subjects` is
  a flat `array<string>`, so the version of that finding which is true here is
  `matched-elements-only`, declared on the field in `record.sd`. Vespa can
  return which subject headings matched; `api/catalogue.py:306-383` runs an
  embedding pass at query time to *guess* that.
* **`grouping` cannot be measured.** None of the 41 eval queries carries a
  facet filter, so there is no nDCG to move. It would remove the
  `SEMANTIC_POOL = 200` approximation `api/catalogue.py` admits is "judgement,
  not measurement", but that is a demonstration, not a number.
* **Binary quantisation and partial attribute updates** are not yet run.
* **The query embedder is still ours**, deliberately. Letting Vespa embed the
  query too changes two things at once — ONNX against torch numerics, and
  Vespa's tokenizer against sentence-transformers' — and would have muddied the
  parity gate. Worth doing next, now that the gate has held.
