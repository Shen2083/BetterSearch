# Vespa lane files

Each file is `{query: [doc_id, ...]}`, best first, 100 deep, over the 41 queries
in `data/eval_real.json`. That is the only shape `scripts/score_rankings.py`
reads and the shape `scripts/build_eval_set.py --lane-file` pools, so each of
these can be scored and judged with no change to the harness.

They are committed for the same reason `data/rankings/*.json` are: a judgement
is only auditable if the pool it was made over can be reconstructed, and a
Vespa rank profile has no index path that re-derives it.

**None of these lanes has been pooled.** No Anthropic key is available, so no
new judgements could be bought. Read the coverage line in
`score_rankings.py`'s output before the nDCG line: anything a profile surfaced
that our six pooled lanes never found counts as irrelevant, so these scores are
floors except where coverage is 100%.

| file | profile | what it is | nDCG@10 | coverage |
|---|---|---|---|---|
| `vespa-dense.json` | `dense` | exact `nearestNeighbor` over the vectors from `.bettersearch/real-books-base.npz`. The parity gate. | 0.6582 | 100.0% |
| `vespa-bm25-flat.json` | `bm25_flat` | `bm25(title) + bm25(text)`, the two fields our Python scores, `type=any` to match its OR | 0.4120 | 75.4% |
| `vespa-bm25-fielded.json` | `bm25_fielded` | title, author, subjects, text as separate fields, all weights 1.0 | 0.3734 | 72.2% |
| `vespa-bm25-fielded-best.json` | `bm25_fielded` | best of 27 weightings, **fitted on these same queries**: title 1.0, author 0.0, subjects 0.5, text 1.0 | 0.3952 | 75.1% |
| `vespa-hybrid-rrf.json` | `hybrid_rrf` | Vespa's `reciprocal_rank_fusion()` in a `global-phase` | 0.5620 | 90.5% |
| `vespa-colbert-rc20.json` | `colbert` | native `colbert-embedder`, MaxSim in `second-phase`, `rerank-count` 20 | 0.7131 | 100.0% |
| `vespa-colbert-rc50.json` | `colbert` | same, `rerank-count` 50 | 0.6557 | 89.5% |
| `vespa-colbert-rc100.json` | `colbert` | same, `rerank-count` 100 | 0.6449 | 87.3% |
| `vespa-colbert-rc200.json` | `colbert` | same, `rerank-count` 200 | 0.6473 | 85.6% |

Regenerate any of them with `vespa/scripts/run_eval.py`; see `vespa/README.md`
for the sequence from a clean clone.
