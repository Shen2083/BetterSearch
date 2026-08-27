"""Embedding provider registry.

Selection is by name (``BETTERSEARCH_EMBEDDING_PROVIDER``). A missing API key
raises rather than falling back to another provider - a silent fallback would
write vectors from the wrong model into the index, which is exactly the failure
the ``model_id`` bookkeeping exists to prevent.
"""

from __future__ import annotations

from ..config import Settings, load_settings
from .base import EmbeddingProvider, MissingCredentialError, l2_normalize

PROVIDER_NAMES = ("local", "openai", "voyage")


def get_provider(
    name: str | None = None,
    *,
    settings: Settings | None = None,
) -> EmbeddingProvider:
    settings = settings or load_settings()
    name = (name or settings.embedding_provider).lower().strip()

    if name == "local":
        from .local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            settings.local_model_name,
            trust_remote_code=settings.local_trust_remote_code,
        )

    if name == "openai":
        from .openai_ import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(dimensions=settings.embedding_dimensions)

    if name == "voyage":
        from .voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(dimensions=settings.embedding_dimensions)

    raise ValueError(
        f"Unknown embedding provider {name!r}. "
        f"Expected one of: {', '.join(PROVIDER_NAMES)}"
    )


__all__ = [
    "EmbeddingProvider",
    "MissingCredentialError",
    "PROVIDER_NAMES",
    "get_provider",
    "l2_normalize",
]
