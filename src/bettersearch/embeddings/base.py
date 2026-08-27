"""The embedding provider contract.

Every provider returns **L2-normalised float32** vectors. That single guarantee
means cosine similarity is a plain dot product and no index ever needs to know
which provider produced its vectors.

``model_id`` is the important field. It is stored next to every vector so the
index can refuse a query embedded by a different model, and so a rolling
re-embed is possible without taking search offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    #: Stable identifier including output width, e.g. "voyage-3.5-lite@512".
    model_id: str
    #: Width of the returned vectors.
    dimensions: int
    #: Provider's input limit. Text longer than this is silently truncated by
    #: the provider, which is a real quality ceiling worth surfacing.
    max_input_tokens: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed content for indexing. Returns shape (len(texts), dimensions)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns shape (dimensions,)."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise rows (or a single vector) to unit length, as float32."""
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    # Guard against a zero vector producing NaNs.
    np.maximum(norms, 1e-12, out=norms)
    return (array / norms).astype(np.float32, copy=False)


class MissingCredentialError(RuntimeError):
    """Raised when a provider is selected but its API key is not configured.

    Deliberately fatal rather than falling back to another provider: a silent
    fallback would write vectors from the wrong model into the index.
    """
