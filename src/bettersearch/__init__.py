"""BetterSearch - semantic content search that retrieves by meaning.

Quick start::

    from bettersearch import Searcher, ingest_corpus

    ingest_corpus("data/corpus_seed.json")
    Searcher().search("Norman conquest", mode="semantic").results
"""

from __future__ import annotations

from .chunking import chunk_document, chunk_documents
from .config import Settings, load_settings
from .embeddings import get_provider
from .index import get_index
from .ingest import IngestReport, ingest_corpus, ingest_documents, load_corpus
from .search import MODES, SearchResponse, Searcher
from .types import (
    Chunk,
    Document,
    EmptyIndexError,
    IndexStats,
    ModelMismatchError,
    ScoredChunk,
)

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Document",
    "EmptyIndexError",
    "IndexStats",
    "IngestReport",
    "MODES",
    "ModelMismatchError",
    "ScoredChunk",
    "SearchResponse",
    "Searcher",
    "Settings",
    "chunk_document",
    "chunk_documents",
    "get_index",
    "get_provider",
    "ingest_corpus",
    "ingest_documents",
    "load_corpus",
    "load_settings",
]
