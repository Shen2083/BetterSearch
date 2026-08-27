"""Ingestion: documents -> chunks -> embeddings -> index.

The important behaviour here is **incremental**: chunks whose content hash is
already in the index are not re-embedded. On a large content library this is the
difference between a content update costing pennies and costing the full corpus
price every time. Re-running an unchanged ingest should embed nothing at all,
and the report says so.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_documents
from .config import Settings, load_settings
from .embeddings import EmbeddingProvider, get_provider
from .index import VectorIndex, get_index
from .types import Document, EmbeddedChunk

_EMBED_BATCH = 64


@dataclass(slots=True)
class IngestReport:
    documents: int
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int
    model_id: str
    dimensions: int
    truncated_chunks: int

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chunks_total": self.chunks_total,
            "chunks_embedded": self.chunks_embedded,
            "chunks_skipped": self.chunks_skipped,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "truncated_chunks": self.truncated_chunks,
        }


def load_corpus(path: str | Path) -> list[Document]:
    """Read a corpus JSON file: either a list, or {"documents": [...]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw["documents"] if isinstance(raw, dict) else raw
    return [Document.from_dict(record) for record in records]


def ingest_documents(
    documents: Sequence[Document],
    *,
    provider: EmbeddingProvider | None = None,
    index: VectorIndex | None = None,
    settings: Settings | None = None,
    force: bool = False,
    progress: bool = False,
) -> IngestReport:
    settings = settings or load_settings()
    provider = provider or get_provider(settings=settings)
    index = index if index is not None else get_index(settings=settings)

    chunks = chunk_documents(
        documents,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    known = set() if force else index.existing_hashes()
    pending = [c for c in chunks if c.content_hash not in known]

    # Surface silent truncation rather than letting it degrade recall unnoticed.
    from .chunking import estimate_tokens

    truncated = sum(
        1 for c in pending if estimate_tokens(c.embed_text) > provider.max_input_tokens
    )

    embedded_count = 0
    for start in range(0, len(pending), _EMBED_BATCH):
        batch = pending[start : start + _EMBED_BATCH]
        vectors = provider.embed_documents([c.embed_text for c in batch])
        index.upsert(
            [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(batch, vectors)],
            model_id=provider.model_id,
        )
        embedded_count += len(batch)
        if progress:
            print(f"  embedded {embedded_count}/{len(pending)} chunks", flush=True)

    return IngestReport(
        documents=len(documents),
        chunks_total=len(chunks),
        chunks_embedded=embedded_count,
        chunks_skipped=len(chunks) - len(pending),
        model_id=provider.model_id,
        dimensions=provider.dimensions,
        truncated_chunks=truncated,
    )


def ingest_corpus(
    path: str | Path,
    *,
    provider: EmbeddingProvider | None = None,
    index: VectorIndex | None = None,
    settings: Settings | None = None,
    force: bool = False,
    progress: bool = False,
) -> IngestReport:
    return ingest_documents(
        load_corpus(path),
        provider=provider,
        index=index,
        settings=settings,
        force=force,
        progress=progress,
    )
