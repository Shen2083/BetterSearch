"""The public search entry point.

Three retrieval modes over the same indexed chunks:

``keyword``
    BM25 term overlap. The baseline.
``semantic``
    Cosine similarity over embeddings. The thesis: matches meaning, not words.
``hybrid``
    Reciprocal rank fusion of the two. What you would actually ship - semantics
    find the concept, keywords pin the exact names, dates and figures that
    embeddings blur together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .config import Settings, load_settings
from .embeddings import EmbeddingProvider, get_provider
from .fusion import reciprocal_rank_fusion
from .index import VectorIndex, get_index
from .keyword import BM25Index
from .types import IndexStats, ScoredChunk

Mode = Literal["keyword", "semantic", "hybrid"]
MODES: tuple[Mode, ...] = ("keyword", "semantic", "hybrid")

# Each arm of a hybrid search retrieves deeper than the final cut so that fusion
# has enough candidates to reorder.
_FUSION_DEPTH_MULTIPLIER = 4


@dataclass(slots=True)
class SearchResponse:
    query: str
    mode: str
    results: list[ScoredChunk]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
        }


class Searcher:
    """Holds a provider and an index, and answers queries in any of the modes.

    The BM25 index is built lazily from the chunks already in the vector index,
    so both modes always search exactly the same corpus - otherwise the
    comparison would be measuring corpus differences rather than retrieval.
    """

    def __init__(
        self,
        *,
        provider: EmbeddingProvider | None = None,
        index: VectorIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._index = index if index is not None else get_index(settings=self._settings)
        self._provider = provider
        self._bm25: BM25Index | None = None

    @property
    def provider(self) -> EmbeddingProvider:
        # Deferred so that a keyword-only search never pays to load a model.
        if self._provider is None:
            self._provider = get_provider(settings=self._settings)
        return self._provider

    @property
    def index(self) -> VectorIndex:
        return self._index

    def _keyword_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self._index.all_chunks())
        return self._bm25

    # ----------------------------------------------------------------- search
    def keyword(self, query: str, *, top_k: int = 10) -> list[ScoredChunk]:
        return self._keyword_index().search(query, top_k=top_k)

    def semantic(self, query: str, *, top_k: int = 10) -> list[ScoredChunk]:
        provider = self.provider
        vector = provider.embed_query(query)
        return self._index.query(vector, top_k=top_k, model_id=provider.model_id)

    def hybrid(self, query: str, *, top_k: int = 10) -> list[ScoredChunk]:
        depth = top_k * _FUSION_DEPTH_MULTIPLIER
        return reciprocal_rank_fusion(
            [self.keyword(query, top_k=depth), self.semantic(query, top_k=depth)],
            top_k=top_k,
        )

    def search(
        self, query: str, *, mode: Mode = "semantic", top_k: int = 10
    ) -> SearchResponse:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: {MODES}")
        results = getattr(self, mode)(query, top_k=top_k)
        return SearchResponse(query=query, mode=mode, results=results)

    def compare(
        self, query: str, *, top_k: int = 10, modes: Sequence[Mode] = MODES
    ) -> dict[str, SearchResponse]:
        """Run every mode on one query - this is what the demo page renders."""
        return {mode: self.search(query, mode=mode, top_k=top_k) for mode in modes}

    def stats(self) -> IndexStats:
        return self._index.stats()
