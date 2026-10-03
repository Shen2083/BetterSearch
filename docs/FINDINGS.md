# BetterSearch PoC — findings so far

A summary of what this proof of concept has established, for someone who will
not read the README end to end. Every number below was produced by code in this
repository and can be regenerated; where a result is weak or contested, it says
so. The README is the long version, and each section here names the part of it
that carries the detail.

Last updated 3 October 2026, at commit `aa5421e`.

---

## The question

Can retrieval by **meaning** beat retrieval by **shared words** on a library
catalogue of thin records — title, author, subject headings, around 15 words and
usually no summary at all?

Yes, and by a margin that is not close. The rest is about how much of that
margin comes from where.

## How it is measured

41 queries over **4,000 real Open Library records**, with 1,979 relevance
judgements pooled from four retrieval lanes and judged blind by Claude Haiku.
The headline metric is nDCG@10. Five of the 41 queries are deliberate
known-item controls — exact titles and author names — which score **1.000** and
exist as a tripwire rather than a target: a change that drops them has broken
exact-name lookup, the thing library users do most.

Queries are author-drafted and reviewer-edited, grades are an LLM's validated by
sampling (94% judge self-consistency), and holdings data is invented while the
bibliographic data is real. See *How much to trust this* in the README.

## Baseline

| arm | nDCG@10 |
|---|---|
| BM25 only | 0.443 |
| MiniLM 384d, semantic | 0.579 |
| RRF hybrid, BM25 + MiniLM | 0.549 |
| **bge-base 768d, semantic — what ships** | **0.658** |
| bge-base + LLM enrichment of thin records | 0.743 |

Three things in that table are worth more than the ordering:

- **Hybrid lost.** Reciprocal rank fusion of the two lanes scored *below* pure
  semantic. The first time this happened it was explained away as an artefact of
  keyword-hostile query phrasing; re-tested on real records with exact-name
  controls, it lost again. Blending two full rankings drags a good list down
  with a mostly empty one, because BM25 returns nothing at all for a query that
  shares no term with any record and rank fusion still weights those empty
  positions.
- **A bigger encoder buys most of what LLM enrichment buys, for free.** bge-base
  on thin records (0.658) nearly matches MiniLM on enriched records (0.671), with
  no API calls and no per-record cost. Enrichment is still worth its £7 on top
  (+0.085 over bge-base), but it is the second thing to try, not the first.
- **Enrichment helps aboutness and hurts identity.** It improved 26 of 41
  queries and regressed 5. The worst, `Sue Monk Kidd`, fell 1.000 → 0.387,
  because topical prose about bees and the American South crowded out the
  author's name in that record's vector.

## Reranking: the ordering is the problem, not the retrieval

Commit `aa5421e`. The ceiling was measured before anything was built.

| | nDCG@10 |
|---|---|
| what we ship | 0.6582 |
| perfect reordering of the same 20 candidates | **0.9207** |

**The retrieval is already finding the right records; it is putting them on page
two.** 161 judged-relevant records across the 41 queries are retrieved and
sitting at ranks 11–20. Median per-query headroom is 0.274, so the problem is
broad rather than a handful of outliers — the ten queries with nothing to gain
are the exact-name controls, already at 1.000.

Reranking is also **the only change this eval can measure without re-pooling**,
because it never changes the candidate set, so judged coverage stays at 100% by
construction. `scripts/tune_reranking.py` asserts that rather than trusting it.

| arm | nDCG@10 | vs base | controls | ms/query |
|---|---|---|---|---|
| baseline (bge-base, as shipped) | 0.6582 | — | 1.000 | 0 |
| + BM25 as a weighted second phase | 0.6782 | +0.020 | 1.000 | 0 |
| + MiniLM cross-encoder over **thin** records | 0.6632 | +0.005 | 1.000 | 314 |
| + MiniLM cross-encoder over **enriched** records | 0.7149 | +0.057 | 0.877 | 314 |
| &nbsp;&nbsp;… + exact-match promotion | 0.7280 | +0.070 | 1.000 | 314 |
| + `bge-reranker-base` over **thin** records | 0.6573 | −0.001 | 1.000 | 2,391 |
| + `bge-reranker-base` over **enriched** records | 0.7289 | +0.071 | 0.926 | 2,391 |
| &nbsp;&nbsp;**… + exact-match promotion** | **0.7340** | **+0.076** | **1.000** | 2,391 |
| oracle (the ceiling) | 0.9207 | +0.263 | 1.000 | — |

- **A cross-encoder is worth nothing on a bare catalogue record** — +0.005 and
  −0.001. That is the data, not the model: on a written sentence
  `bge-reranker-base` separates cleanly (0.643 for a beekeeping passage against
  0.000 for a calculus textbook), but the record it is handed is a 24-word
  metadata stub with nothing to read. Give it the enriched text at 95 words and
  the same model is worth +0.071.
- **So reranking does not replace enrichment; it is unlocked by it.** Both are
  the same bet — that these records are too thin — arriving from different ends,
  and they compound.
- **It broke the controls, and the tripwire caught it.** Both enriched arms
  regressed known-item lookup, `Sue Monk Kidd` worst at −0.369, inheriting
  enrichment's own failure because the reranker reads the same prose.
  `src/bettersearch/exact.py` — which lifts named records by rule rather than by
  score, and whose own docstring called it "a guarantee rather than an
  improvement" worth 0.658 → 0.658 — restores the controls to 1.000 *and* raises
  the mean, because the queries it rescues are the ones being broken. A component
  measured as worthless became valuable the moment something upstream was strong
  enough to be unpredictable.

**What is not claimed.** The best arm captures 29% of the available headroom;
0.19 of nDCG is still on the table and nobody here knows what reaches it. That is
the most interesting open question in the project. `bge-reranker-base` costs 2.4
seconds per query on CPU for 20 candidates — unusable without ONNX or a GPU —
and the MiniLM cross-encoder gets +0.070 of the +0.076 for an eighth of that.
The BM25 second phase at +0.020 is reported and not shipped, because `w` was
swept on the same 41 queries it is scored on, making it an upper bound on the
idea rather than an estimate of it.

## Vespa

The engineering team is moving to Vespa, which prompted the reranking work
above: Vespa's contribution to *quality*, as opposed to serving, is its
reranking phase.

**Nothing in this prototype was ported to Vespa, and nothing needed to be.**
Vespa is a reasonable choice for a real service — hundreds of thousands of
holdings, multiple branches, live circulation — and it replaces a good deal of
machinery this PoC had to hand-roll: BM25, ANN with HNSW, facet grouping, phased
ranking, an in-cluster embedder. But all of that is *serving*, and none of it
decides quality. The quality question Vespa raised turned out to be measurable
here, without it, which is what `aa5421e` did. A naive port would also have
scored slightly worse than the prototype rather than better, because Vespa's ANN
is approximate where our brute-force cosine over 4,000 records is exhaustive.

What the PoC hands over instead is the measurement, in
**[docs/VESPA-HANDOVER.md](VESPA-HANDOVER.md)**: an acceptance bar (0.658
semantic, 0.443 BM25 on the same records — below that, the port is broken), four
traps including the bge pooling-strategy default that silently halves retrieval
quality, and `scripts/score_rankings.py`, which scores any engine's ranked
`doc_id`s with the same arithmetic as everything here.

**The trap in comparing engines at all.** The judgements were pooled from four
lanes, and an unjudged record counts as irrelevant. A Vespa implementation is a
*fifth lane*, so every good record it finds that the four missed scores zero —
**a better system can score worse.** That is not hypothetical: bge-base first
measured worse than MiniLM, 0.578 against 0.588, and was nearly rejected; 100%
of MiniLM's top 10 had been judged against 71% of bge's, and re-pooling with bge
as a fourth lane moved it +0.080. Hence coverage is reported beside every score,
and below 75% the comparison is not a comparison.

## Where this leaves the PoC

Settled:

- Meaning-based retrieval beats keyword retrieval on this corpus, 0.658 against
  0.443, and wins on exact-name queries too once `exact.py` is in the path.
- Flat hybrid fusion is not the free win the engine offers it as.
- Thin records are the binding constraint, and the two things that address them
  — a 768d encoder and LLM enrichment — stack almost additively, +0.164 together.
- Retrieval is not the bottleneck at depth 20; ordering is.
- Do not cut a vector result list by score. Fixed top-20 beat every similarity
  threshold tried, because cosine has no absolute floor.

Open:

1. **The remaining 0.19 of headroom.** The first thing worth a week.
2. **Reranking latency.** 2.4 s/query is not shippable; ONNX, a GPU, or the
   MiniLM cross-encoder, and rerank depth is the lever.
3. **41 queries remains few**, and two of them score 0.000 on both the thin and enriched arms.
4. **Whether the top model tier buys better retrieval** is unanswered — the 289
   Opus enrichments support a quality comparison, not a retrieval one, and the
   real answer costs about £37.
5. **Routing by query type** rather than enriching everything and hoping.
