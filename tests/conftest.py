"""Shared test fixtures.

The fake embedding provider is the important piece: it produces deterministic,
L2-normalised vectors from a hash of the text, so the whole pipeline can be
tested without a model, a network call or an API key. Similar text does not
produce similar vectors, so it cannot be used to test retrieval *quality* - only
plumbing, which is what these tests are for.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from bettersearch.config import Settings
from bettersearch.embeddings.base import l2_normalize
from bettersearch.index.numpy_index import NumpyVectorIndex
from bettersearch.types import Document

DIMENSIONS = 16


class FakeEmbeddingProvider:
    """Deterministic hash-based embeddings."""

    def __init__(self, model_id: str = "fake@16", dimensions: int = DIMENSIONS) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.max_input_tokens = 10_000
        self.embed_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest[: self.dimensions], dtype=np.uint8)
        return raw.astype(np.float32) - 128.0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.embed_calls += len(texts)
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def index(tmp_path) -> NumpyVectorIndex:
    return NumpyVectorIndex(tmp_path / "index")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(index_path=str(tmp_path / "index"))


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            doc_id="hastings",
            title="The Battle of Hastings",
            text=(
                "An invading army met the English shield wall on Senlac Hill "
                "in October.\n\n"
                "By dusk the English king was dead and the army had broken."
            ),
        ),
        Document(
            doc_id="yeast",
            title="Baker's Yeast",
            text=(
                "A single-celled fungus that consumes sugars and releases "
                "carbon dioxide.\n\n"
                "The gas inflates pockets in the dough so the mass expands."
            ),
        ),
    ]
