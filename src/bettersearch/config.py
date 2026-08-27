"""Environment-driven settings.

Every knob is a ``BETTERSEARCH_*`` environment variable so that switching
embedding provider or storage backend is a config change, never a code change.
No secrets are stored here - API keys are read by the provider that needs them,
at the point of use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_INDEX_PATH = ".bettersearch/index"


@dataclass(frozen=True, slots=True)
class Settings:
    embedding_provider: str = "local"
    index_backend: str = "numpy"
    index_path: str = DEFAULT_INDEX_PATH
    database_url: str | None = None
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 60
    # Matryoshka output width for providers that support truncation. Smaller
    # vectors mean a proportionally smaller index; see README on storage cost.
    embedding_dimensions: int = 512
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Some long-context encoders (nomic-embed, gte-*-v1.5) ship custom modelling
    # code that transformers will only execute with this opted into explicitly.
    # Off by default: it runs arbitrary code from the model repo.
    local_trust_remote_code: bool = False


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_settings() -> Settings:
    return Settings(
        embedding_provider=os.environ.get("BETTERSEARCH_EMBEDDING_PROVIDER", "local"),
        index_backend=os.environ.get("BETTERSEARCH_INDEX", "numpy"),
        index_path=os.environ.get("BETTERSEARCH_INDEX_PATH", DEFAULT_INDEX_PATH),
        database_url=os.environ.get("BETTERSEARCH_DATABASE_URL")
        or os.environ.get("DATABASE_URL"),
        chunk_target_tokens=_int_env("BETTERSEARCH_CHUNK_TOKENS", 450),
        chunk_overlap_tokens=_int_env("BETTERSEARCH_CHUNK_OVERLAP", 60),
        embedding_dimensions=_int_env("BETTERSEARCH_EMBEDDING_DIMENSIONS", 512),
        local_model_name=os.environ.get(
            "BETTERSEARCH_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        local_trust_remote_code=os.environ.get(
            "BETTERSEARCH_LOCAL_TRUST_REMOTE_CODE", ""
        ).lower()
        in {"1", "true", "yes"},
    )
