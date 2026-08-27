"""Voyage AI embeddings backend.

Two things this backend does that the others cannot:

* **Asymmetric encoding.** Voyage takes an ``input_type`` of ``"document"`` or
  ``"query"`` and encodes each differently. Queries and documents are not the
  same kind of text - a query is short and interrogative, a document is long and
  declarative - and telling the model which it is measurably improves recall.
  Getting this wrong is a silent quality loss, so it is wired in explicitly.
* **Matryoshka truncation** via ``output_dimension``, same idea as OpenAI's
  ``dimensions``: 512 instead of the native width, for a proportionally
  smaller index.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .base import MissingCredentialError, l2_normalize

DEFAULT_MODEL = "voyage-3.5-lite"
MAX_INPUT_TOKENS = 32_000
_BATCH_SIZE = 128


class VoyageEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimensions: int = 512,
        api_key: str | None = None,
    ) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The Voyage embedding backend needs the voyageai package. "
                'Install it with: pip install "bettersearch[voyage]"'
            ) from exc

        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise MissingCredentialError(
                "VOYAGE_API_KEY is not set. Export it, or select another "
                "provider with BETTERSEARCH_EMBEDDING_PROVIDER=local."
            )

        self._client = voyageai.Client(api_key=key)
        self._model_name = model_name
        self.dimensions = dimensions
        self.model_id = f"{model_name}@{dimensions}"
        self.max_input_tokens = MAX_INPUT_TOKENS

    def _encode(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        collected: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            response = self._client.embed(
                batch,
                model=self._model_name,
                input_type=input_type,
                output_dimension=self.dimensions,
            )
            collected.extend(response.embeddings)
        return l2_normalize(np.asarray(collected, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], "query")[0]
