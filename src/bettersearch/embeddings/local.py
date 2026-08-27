"""Local sentence-transformers backend - the default, and the only one that
runs with no API key and no network after the first model download.

Quality ceiling, stated plainly: all-MiniLM-L6-v2 accepts **256 tokens** of
input and truncates the rest. The chunker targets 450 tokens, so this backend
silently drops the tail of most chunks. It is fine for a PoC and for a small
library; it is the wrong choice for a large content library, where one of the
hosted providers should be used instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import l2_normalize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Input cap of the default model. Longer text is truncated by the encoder.
MAX_INPUT_TOKENS = 256


class LocalEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The local embedding backend needs sentence-transformers. "
                'Install it with: pip install "bettersearch[local]"'
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.model_id = f"{model_name}@{self.dimensions}"
        self.max_input_tokens = MAX_INPUT_TOKENS

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]
