"""Reciprocal rank fusion.

Combines two ranked lists using only *rank*, never raw score. That matters:
BM25 scores are unbounded term-overlap sums and cosine scores live in [-1, 1],
so any attempt to blend the numbers directly needs normalisation constants that
have to be re-tuned every time the corpus or embedding model changes. Ranks are
comparable by construction.

    score(d) = sum over lists of 1 / (k + rank(d))

``k`` (conventionally 60) damps the influence of the very top ranks so a single
list cannot dominate the fused order on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import ScoredChunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    *,
    top_k: int = 10,
    k: int = DEFAULT_K,
) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    seen: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for position, scored in enumerate(results, start=1):
            # Trust the list's own ordering rather than a possibly-unset rank.
            chunk_id = scored.chunk.chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)
            seen.setdefault(chunk_id, scored)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        ScoredChunk(chunk=seen[chunk_id].chunk, score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]
