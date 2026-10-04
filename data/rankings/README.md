# Pooled lane rankings

`{query: [doc_id, ...]}`, best first — the shape `scripts/build_eval_set.py
--lane-file` pools and `scripts/score_rankings.py` scores.

These are here for provenance, not convenience. `data/eval_real.json` records
*which* records a judge was shown, and a reranked lane is a retrieval plus a
cross-encoder plus a depth — there is no index path that re-derives it. Without
these files, nobody could reconstruct the pool the judgements were made over,
and the eval would be a set of numbers with no auditable origin.

| file | arm |
|---|---|
| `rerank-bge-200.json` | `BAAI/bge-reranker-base` over enriched record text, rerank depth 200, top 20 |
| `rerank-bge-200-promoted.json` | the same, with `exact.py` named-record promotion applied |

Both are pooled. The plain arm regresses the known-item controls to 0.800 —
reranking by topical similarity pushes an exactly-named book off the top — so
the promoted arm is the one that would ship. They differ by 2 records per query,
so pooling both costs 0 extra judgements over pooling either.

Trimmed to the top 20, which is what the pool consumes. To regenerate the full
200-deep dumps (about 25 minutes of CPU, no network or API key):

```bash
python scripts/tune_reranking.py --depths 200 --rerankers bge --dump /tmp/lane200
```

The sweep is deterministic — same index, same encoder, same cross-encoder — so a
regenerated dump is identical to the trim here on its first 20 entries.
