"""OpenAI embeddings backend.

``text-embedding-3-small`` supports Matryoshka truncation via the ``dimensions``
parameter, so we ask for 512 rather than the native 1536. That is a 3x smaller
index for a small quality cost - see the storage-cost note in the README.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .base import MissingCredentialError, l2_normalize

DEFAULT_MODEL = "text-embedding-3-small"
MAX_INPUT_TOKENS = 8191
_BATCH_SIZE = 128


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimensions: int = 512,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The OpenAI embedding backend needs the openai package. "
                'Install it with: pip install "bettersearch[openai]"'
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingCredentialError(
                "OPENAI_API_KEY is not set. Export it, or select another "
                "provider with BETTERSEARCH_EMBEDDING_PROVIDER=local."
            )

        self._client = OpenAI(api_key=key)
        self._model_name = model_name
        self.dimensions = dimensions
        self.model_id = f"{model_name}@{dimensions}"
        self.max_input_tokens = MAX_INPUT_TOKENS

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        collected: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
                dimensions=self.dimensions,
            )
            # The API does not guarantee ordering; key by the returned index.
            ordered = sorted(response.data, key=lambda item: item.index)
            collected.extend(item.embedding for item in ordered)
        return l2_normalize(np.asarray(collected, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]
