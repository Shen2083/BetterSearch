"""Core data types shared across the pipeline.

Two fields deserve explanation because they carry the whole scale story:

``Chunk.content_hash``
    Lets ingestion skip work that is already indexed. Without it, re-ingesting a
    large content library means re-embedding every chunk on every content update.

``IndexStats.model_id`` / ``dimensions``
    Recorded alongside every stored vector. Vectors from different embedding
    models are not comparable, so an index must be able to say which model
    produced it and refuse queries embedded with a different one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Document:
    """A single piece of source content, before chunking."""

    doc_id: str
    title: str
    text: str
    url: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(raw["doc_id"]),
            title=str(raw["title"]),
            text=str(raw["text"]),
            url=raw.get("url"),
            tags=tuple(raw.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text.

    ``text`` is what we show the user. ``embed_text`` is what we embed - it is
    the chunk prefixed with its document title, which meaningfully improves
    retrieval because a mid-article chunk usually does not restate its subject.
    """

    chunk_id: str
    doc_id: str
    title: str
    ordinal: int
    text: str
    embed_text: str
    content_hash: str
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "ordinal": self.ordinal,
            "text": self.text,
            "embed_text": self.embed_text,
            "content_hash": self.content_hash,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=raw["chunk_id"],
            doc_id=raw["doc_id"],
            title=raw["title"],
            ordinal=int(raw["ordinal"]),
            text=raw["text"],
            embed_text=raw["embed_text"],
            content_hash=raw["content_hash"],
            url=raw.get("url"),
        )


@dataclass(slots=True)
class EmbeddedChunk:
    """A chunk paired with its (L2-normalised) embedding."""

    chunk: Chunk
    vector: np.ndarray


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieval hit. ``rank`` is 1-based within its result list."""

    chunk: Chunk
    score: float
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "url": self.chunk.url,
            "snippet": self.chunk.text,
            "score": round(float(self.score), 6),
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class IndexStats:
    backend: str
    chunk_count: int
    model_id: str | None = None
    dimensions: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "chunk_count": self.chunk_count,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            **self.extra,
        }


class ModelMismatchError(RuntimeError):
    """Raised when a query vector's model does not match the indexed vectors.

    This is deliberately fatal. Silently comparing vectors from two different
    embedding models returns plausible-looking nonsense, which is far worse than
    an error - it is the failure mode that survives all the way to production.
    """


class EmptyIndexError(RuntimeError):
    """Raised when querying an index that has no vectors in it yet."""
