"""Local sentence-transformers backend - the default, and the only one that
runs with no API key and no network after the first model download.

Any sentence-transformers model works here; set ``BETTERSEARCH_LOCAL_MODEL``.
Dimensions and the input limit are read from the loaded model rather than
hardcoded, so swapping to a larger encoder needs no code change.

**The input limit is architectural, not a throughput limit.** bge-base accepts
512 tokens, all-MiniLM-L6-v2 only 256, because their positional embeddings stop
there. More CPU or a bigger GPU makes it faster, never able to read more - which
is why enriched records truncated 79 chunks under MiniLM and none under bge.

Whether that cap costs anything depends entirely on the content, so measure
rather than assume. The chunker targets 450 tokens, but a chunk only gets near
that if the source paragraphs are long enough to fill it; on both corpora in
this repo nothing reaches even half the cap. ``bettersearch ingest`` counts and
reports ``truncated_chunks`` for exactly this reason - check it against your own
content. If it is not zero, lower ``BETTERSEARCH_CHUNK_TOKENS`` (free) or move
to a longer-context model - see SUGGESTED_MODELS below.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import l2_normalize

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

#: Fallback only, for the rare model that does not report its own limit.
FALLBACK_MAX_INPUT_TOKENS = 512

#: Self-hosted upgrade path, all drop-in via BETTERSEARCH_LOCAL_MODEL.
#: (input tokens, dimensions, approximate download size)
SUGGESTED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": (256, 384, "80MB"),
    "BAAI/bge-small-en-v1.5": (512, 384, "130MB"),
    "BAAI/bge-base-en-v1.5": (512, 768, "440MB"),
    "intfloat/e5-large-v2": (512, 1024, "1.3GB"),
    "nomic-ai/nomic-embed-text-v1.5": (8192, 768, "550MB"),
    "Alibaba-NLP/gte-large-en-v1.5": (8192, 1024, "1.7GB"),
    "BAAI/bge-m3": (8192, 1024, "2.3GB"),
}


class LocalEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The local embedding backend needs sentence-transformers. "
                'Install it with: pip install "bettersearch[local]"'
            ) from exc

        self._model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        self._batch_size = batch_size
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.model_id = f"{model_name}@{self.dimensions}"
        # Read the real limit off the model. Hardcoding it would silently
        # misreport truncation the moment anyone swaps to a longer-context
        # encoder, which is the whole upgrade path for a self-hosted setup.
        self.max_input_tokens = int(
            getattr(self._model, "max_seq_length", None) or FALLBACK_MAX_INPUT_TOKENS
        )

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
