"""The vector index contract.

Both backends implement the same four operations, so the demo can run entirely
in memory while production runs on Postgres with no change above this line.

Two invariants every backend must uphold:

1. ``existing_hashes()`` reports what is already indexed, so ingestion can skip
   unchanged chunks instead of re-embedding a whole library.
2. ``query()`` raises ``ModelMismatchError`` if the caller's ``model_id`` differs
   from the indexed vectors'. Vectors from different models occupy different
   spaces; comparing them returns confident nonsense.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ..types import EmbeddedChunk, IndexStats, ScoredChunk


@runtime_checkable
class VectorIndex(Protocol):
    def upsert(self, chunks: Sequence[EmbeddedChunk], *, model_id: str) -> int:
        """Insert or replace chunks. Returns the number written."""
        ...

    def query(
        self, vector: np.ndarray, *, top_k: int, model_id: str
    ) -> list[ScoredChunk]:
        """Return the top_k most similar chunks by cosine similarity."""
        ...

    def existing_hashes(self) -> set[str]:
        """Content hashes already present, for incremental ingestion."""
        ...

    def all_chunks(self) -> list:
        """Every indexed chunk, used to build the keyword baseline."""
        ...

    def stats(self) -> IndexStats:
        ...

    def clear(self) -> None:
        ...
