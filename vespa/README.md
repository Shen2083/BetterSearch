# A Vespa-only proof of concept

Eight claims about Vespa were worked out for this project by reading Vespa's
documentation and arguing it against our Python measurements. **None had been
run, and none had been written down** — they lived in conversation, offered for
`docs/VESPA-HANDOVER.md` and never added to it. Only two of the eight touch
anything that document actually says: grouping over the matched set
(`VESPA-HANDOVER.md:171`) and `rerank-count` as the phase boundary (`:200`).
This directory is where they are recorded for the first time, with numbers.

One Vespa node, the same 4,000 book records, the same 41 judged queries, scored
through the same harness that produced every published figure in this
repository.

**Six of the eight now have evidence**, one is wrong, and one cannot be
measured with this eval set. That is everything answerable here.

Late interaction and phased ranking came out **better** than predicted. RRF
came out exactly as predicted, and confirms `VESPA-HANDOVER.md` §3.1 in Vespa's
own implementation. Partial attribute updates do what was claimed. The per-node
IDF warning is **overstated** at this corpus size, and binary quantisation is
**oversold**: the 32-fold reduction and the quality are not available at the
same time. `sameElement` is **wrong** and needed no measurement to settle — it
does not apply to this schema at all. `grouping` has no number because none of
the 41 queries carries a facet filter.

A ninth question, nobody's prediction, came out of the port and is a negative:
fielding the record properly, which our Python provably cannot do, does not
help this corpus.

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

The decay below that is **confounded, and it is my measurement that is
confounded, not ours.** Reranking the top 20 of the dense list can only reorder
records that were already pooled. Past 20 it promotes records from deeper, which
no judge ever saw, and those count as irrelevant.

I first wrote this up the other way round, as a caution about our own published
depth sweep. That was wrong, and the handover says so on its own page.
`VESPA-HANDOVER.md:391-394` already reports coverage per depth alongside the
+0.066 / +0.047 / +0.035 / +0.010 figures — 100% / 92.7% / 98.8% / 100% — and
§5a documents at length how an earlier draft got the depth question wrong by
discarding unjudged candidates. Our sweep never dropped below 92.7%. Mine falls
to 85.6%. So the decay I measured past rerank-count 20 is **less** trustworthy
than the one already published, not more, and the only row here that supports a
conclusion is rerank-count 20, where both arms sit at 100%.

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
(`scripts/tune_reranking.py`). Here it is a 70 ms query. The reason given for
handing late interaction to the Vespa team rather than shipping it ourselves —
that for them it is a supported path rather than an extra dependency and
someone else's MaxSim — is borne out.

## 5. The per-node IDF warning is overstated

This was the prediction I most wanted checked, and it does not hold at this
corpus size.

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

## 6. Binary quantisation is oversold, and there is a trap next to it

768 floats at 4 bytes each become 96 bytes: a 32-fold reduction, Hamming
instead of cosine, computed by Vespa from the float field at indexing time.

| lane | nDCG@10 | coverage | top-10 same as float | round trip |
|---|---|---|---|---|
| float, exact | 0.6582 | 100.0% | — | 23 ms |
| packed, Hamming only | 0.5847 | 83.2% | 59.0% | 10 ms |
| packed, then rescored on the floats | **0.6580** | 99.8% | **94.4%** | 10 ms |

Retrieve on the packed vectors and rescore the survivors with the float ones,
and essentially all the quality comes back: 0.6580 against 0.6582, an identical
top ten on 32 of 41 queries, at **less than half the latency**.

**But the two benefits are not simultaneous.** The rescore needs the float
vectors, so they must still be stored — that profile buys query cost, not
memory. Drop the floats and take the real 32-fold saving, and you are at
0.5847, down 0.07 with 59% of the top ten changed. The prediction was that the
compression is close to free. On this corpus it is free of *latency*, not of
memory.

**The trap.** The obvious second-phase expression, `closeness(field, embedding)`,
scores **0.0837**. `closeness` only has a distance for the field
`nearestNeighbor` actually ran against, which here is `embedding_binary`; asked
about any other field it returns a meaningless value rather than failing. The
profile retrieved almost exactly the right candidates — 63.9% of them matched
the float lane — while its top ten matched 4.1% of the time. A quality
measurement that only looked at nDCG would have read it as binary quantisation
being catastrophic. Writing the dot product out, `sum(query(q) *
attribute(embedding))`, is what the table above measures. Worth a line in the
handover: this one fails silently and plausibly.

## 7. Availability really does change in place

The one finding with no number, because it does not produce one.
`available` and `copies` change on every loan. In our system they live in the
catalogue JSON the index was built from, so moving one means rebuilding,
re-embedding and reloading. Here, `vespa/scripts/demo_partial_update.py`:

```
ol-OL15844725W  'Psychology'      available 0, copies 1
  3312 of 4000 records are borrowable
  one returned copy, fed as a partial update: 20 ms
  3313 borrowable, a change of 1, on the query issued immediately after
  no reindexing, no re-embedding, no redeploy
```

The test is a *filter*, not a summary field, because a stale availability
filter is worse than no filter. The two integers are rewritten on the content
node; the 768-float vector and the ColBERT token tensors are untouched.

## 8. Vespa's own encoder changes nothing measurable

The parity gate was deliberately fed vectors from our encoder, so the only
variable was Vespa's retrieval. That left the other half of the question open:
does Vespa's **own** embedding of the same text agree with ours? It changes two
things at once — ONNX against torch numerics, and Vespa's tokenizer against
sentence-transformers' — which is why it was kept out of the gate.

With `hugging-face-embedder` over the ONNX export of `bge-base-en-v1.5`, Vespa
embedding both the records and the query:

| lane | nDCG@10 | Recall@5 | MRR@10 | coverage | round trip |
|---|---|---|---|---|---|
| vectors fed from our encoder | 0.6582 | 0.2819 | 0.7820 | 100.0% | 23 ms |
| **Vespa embeds everything** | **0.6582** | **0.2819** | **0.7820** | 100.0% | 72 ms |

Not merely equal in aggregate — **the same documents in the same order**:

```
identical top-10 ordering    41/41 queries
identical top-100 ordering   37/41 queries
set overlap at 10 and at 100     100.0%
```

The four queries that differ at all differ only past position ten, where
near-tied records swap. So the port is faithful end to end, encoder included,
and no caveat about ONNX numerics is needed.

Two settings carried that result and neither is a default:

- **`pooling-strategy: cls`.** bge pools the CLS token; Vespa's embedder
  defaults to mean. Mean-pooling a model trained for CLS would have measured
  the pooling rather than the port.
- **no `<prepend>`.** bge's model card suggests an instruction prefix for
  queries. `bettersearch.embeddings.local` does not add one, and a prefix on one
  side only would have measured the prefix.

The cost is latency: 72 ms against 23 ms, the difference being one bge-base
forward pass per query on CPU. For a service that already has the query vector
from somewhere else, feeding it in is three times cheaper. For one that does
not, this is the simpler architecture and it costs nothing in quality.

## 9. What HNSW costs — and why this corpus cannot tell you

Every dense figure above used **exact** search: `nearestNeighbor` over an
attribute with no `index` block, which is what made parity with
`numpy_index.py` possible. A library running 500,000 holdings will use the
approximation instead. This measures it.

**Read the limit before the table.** 4,000 records is far too few for HNSW to
behave the way it will in production — the interesting regime is 10⁵ to 10⁸,
and at this size a graph search touches a large share of the corpus anyway.
What follows is evidence the approximation is configured correctly and a method
that transfers. It is not a scale answer.

This is, though, **the only unbiased number in the whole proof of concept.**
Everything else here is qualified by judged coverage, because the lanes were
never pooled. This asks the same index the same question twice, exactly and
approximately, and reports how much of the exact answer the approximation
found. No judgements are involved at all.

| `hnsw.exploreAdditionalHits` | recall@10 | recall@100 | identical top ten | median latency |
|---|---|---|---|---|
| exact (the reference) | — | — | — | 17 ms |
| 0 | 0.9951 | 0.9654 | 40/41 | 18 ms |
| 100 | 0.9976 | 0.9944 | 40/41 | 18 ms |
| 500 | **1.0000** | **1.0000** | **41/41** | 17 ms |

Two things follow, and the second is the one worth carrying.

**The approximate path is really running.** Near-perfect recall has two
explanations — a good graph, or Vespa quietly falling back to an exact scan,
which it does when it judges the approximation unhelpful and does not say so.
The sweep separates them: recall *moves* with the parameter, from 0.9654 to
1.0000 at depth 100. A fallback would be insensitive to it. (This nearly went
wrong twice. `hnsw.exploreAdditionalHits` is an annotation on the operator, not
a query property, and sent as a property it is accepted and ignored. And
`build_body` consumes it from the dict it is handed, so a caller reusing one
dict across the loop would have set it on the first query only. Either bug
produces a flat sweep that reads as "the parameter does nothing".)

**At this size the approximation buys nothing.** 17 ms exact against 17 ms
approximate. HNSW earns its place by making a scan cheaper, and a scan over
4,000 × 768 floats is not expensive. Anyone quoting a latency win from a
corpus this size is quoting noise.

### The direction, over the only honest lever available

Recall as the corpus grows, fed in stages so the graph grows rather than being
rebuilt, `exploreAdditionalHits` at 0 so the trend is not hidden behind a
perfect score. Real records only — no catalogue records were invented to pad
the corpus, because fabricated bibliographic data would make everything
downstream unfalsifiable.

| documents | recall@10 | recall@100 | identical top ten | exact ms | hnsw ms |
|---|---|---|---|---|---|
| 1,000 | 0.9976 | 0.9793 | 40/41 | 11 | 12 |
| 2,000 | 0.9854 | 0.9702 | 35/41 | 12 | 13 |
| 4,000 | 0.9976 | **0.9639** | 40/41 | 14 | 12 |

`recall@100` falls monotonically — 0.9793, 0.9702, 0.9639 — losing about 0.015
over a fourfold increase. `recall@10` does not move monotonically and should be
read as noise: 41 queries is too few to resolve a difference that small.

The latency columns show the expected *shape* — exact rises with the corpus,
11 to 14 ms, while HNSW stays flat — but 3 ms across three points is not a
measurement of scaling and should not be quoted as one. It is the right shape,
not a result.

**What to do with this.** Run `vespa/scripts/compare_approximation.py` against
your own corpus at your own scale; it needs no judgements, so it works on day
one, before any relevance work exists. Tune `exploreAdditionalHits` until
recall against exact is where you want it, and check that recall *moves* when
you change it — if it does not, you are measuring a fallback.

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
* Nothing else. The query embedder was the last item here, and it has since
  been measured — see §8.
