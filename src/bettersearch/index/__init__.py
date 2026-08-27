"""Vector index registry."""

from __future__ import annotations

from ..config import Settings, load_settings
from .base import VectorIndex

BACKEND_NAMES = ("numpy", "pgvector")


def get_index(
    name: str | None = None,
    *,
    settings: Settings | None = None,
) -> VectorIndex:
    settings = settings or load_settings()
    name = (name or settings.index_backend).lower().strip()

    if name == "numpy":
        from .numpy_index import NumpyVectorIndex

        return NumpyVectorIndex(settings.index_path)

    if name == "pgvector":
        from .pgvector_index import PgVectorIndex

        return PgVectorIndex(settings.database_url or "")

    raise ValueError(
        f"Unknown index backend {name!r}. Expected one of: {', '.join(BACKEND_NAMES)}"
    )


__all__ = ["BACKEND_NAMES", "VectorIndex", "get_index"]
