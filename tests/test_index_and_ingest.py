from __future__ import annotations

import numpy as np
import pytest

from bettersearch.chunking import chunk_documents
from bettersearch.index.numpy_index import NumpyVectorIndex
from bettersearch.ingest import ingest_documents
from bettersearch.types import (
    Chunk,
    EmbeddedChunk,
    EmptyIndexError,
    ModelMismatchError,
)

from .conftest import FakeEmbeddingProvider


def _embedded(chunks, provider) -> list[EmbeddedChunk]:
    vectors = provider.embed_documents([c.embed_text for c in chunks])
    return [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)]


# ------------------------------------------------------------------ the index
def test_query_returns_exact_top_k_by_cosine(index, provider):
    """Check the ranking against hand-computed dot products."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    chunks = [
        Chunk(
            chunk_id=f"d:{i}",
            doc_id="d",
            title="t",
            ordinal=i,
            text=f"text {i}",
            embed_text=f"t\n\ntext {i}",
            content_hash=f"h{i}",
        )
        for i in range(3)
    ]
    index.upsert(
        [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)],
        model_id="hand@3",
    )

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.query(query, top_k=3, model_id="hand@3")

    assert [r.chunk.chunk_id for r in results] == ["d:0", "d:2", "d:1"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.6)
    assert results[2].score == pytest.approx(0.0)
    assert [r.rank for r in results] == [1, 2, 3]


def test_top_k_larger_than_index_is_clamped(index, provider, documents):
    chunks = chunk_documents(documents)
    index.upsert(_embedded(chunks, provider), model_id=provider.model_id)
    results = index.query(provider.embed_query("anything"), top_k=999, model_id=provider.model_id)
    assert len(results) == len(chunks)


def test_querying_an_empty_index_raises(index, provider):
    with pytest.raises(EmptyIndexError):
        index.query(provider.embed_query("q"), top_k=5, model_id=provider.model_id)


def test_query_with_a_different_model_is_refused(index, provider, documents):
    """The core scale invariant: vectors from two models are not comparable."""
    chunks = chunk_documents(documents)
    index.upsert(_embedded(chunks, provider), model_id="model-a@16")

    with pytest.raises(ModelMismatchError, match="model-a@16"):
        index.query(provider.embed_query("q"), top_k=5, model_id="model-b@16")


def test_writing_a_different_model_is_refused(index, provider, documents):
    chunks = chunk_documents(documents)
    index.upsert(_embedded(chunks, provider), model_id="model-a@16")

    with pytest.raises(ModelMismatchError):
        index.upsert(_embedded(chunks, provider), model_id="model-b@16")


def test_upsert_replaces_rather_than_duplicates(index, provider, documents):
    chunks = chunk_documents(documents)
    index.upsert(_embedded(chunks, provider), model_id=provider.model_id)
    before = index.stats().chunk_count

    index.upsert(_embedded(chunks, provider), model_id=provider.model_id)
    assert index.stats().chunk_count == before


def test_index_persists_across_instances(tmp_path, provider, documents):
    path = tmp_path / "index"
    chunks = chunk_documents(documents)
    NumpyVectorIndex(path).upsert(_embedded(chunks, provider), model_id=provider.model_id)

    reopened = NumpyVectorIndex(path)
    assert reopened.stats().chunk_count == len(chunks)
    assert reopened.stats().model_id == provider.model_id
    assert reopened.existing_hashes() == {c.content_hash for c in chunks}


def test_clear_empties_the_index(index, provider, documents):
    index.upsert(_embedded(chunk_documents(documents), provider), model_id=provider.model_id)
    index.clear()
    assert index.stats().chunk_count == 0
    assert index.existing_hashes() == set()


# --------------------------------------------------------------- ingestion
def test_ingest_embeds_every_chunk_first_time(index, provider, documents, settings):
    report = ingest_documents(
        documents, provider=provider, index=index, settings=settings
    )
    assert report.documents == 2
    assert report.chunks_embedded == report.chunks_total
    assert report.chunks_skipped == 0
    assert provider.embed_calls == report.chunks_total


def test_reingest_embeds_nothing_when_content_is_unchanged(
    index, provider, documents, settings
):
    """The incremental-ingest guarantee: a no-op update must cost nothing."""
    first = ingest_documents(documents, provider=provider, index=index, settings=settings)
    calls_after_first = provider.embed_calls

    second = ingest_documents(documents, provider=provider, index=index, settings=settings)

    assert second.chunks_embedded == 0
    assert second.chunks_skipped == first.chunks_total
    assert provider.embed_calls == calls_after_first


def test_changed_document_reembeds_only_its_own_chunks(
    index, provider, documents, settings
):
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    calls_after_first = provider.embed_calls

    edited = list(documents)
    edited[0] = type(documents[0])(
        doc_id=documents[0].doc_id,
        title=documents[0].title,
        text=documents[0].text + "\n\nA newly appended paragraph.",
    )
    report = ingest_documents(edited, provider=provider, index=index, settings=settings)

    assert 0 < report.chunks_embedded < report.chunks_total
    assert provider.embed_calls == calls_after_first + report.chunks_embedded


def test_force_reembeds_everything(index, provider, documents, settings):
    ingest_documents(documents, provider=provider, index=index, settings=settings)
    report = ingest_documents(
        documents, provider=provider, index=index, settings=settings, force=True
    )
    assert report.chunks_embedded == report.chunks_total
    assert report.chunks_skipped == 0


def test_truncation_is_counted_and_reported(index, documents, settings):
    """A provider that truncates should say so rather than silently lose recall."""
    tiny = FakeEmbeddingProvider()
    tiny.max_input_tokens = 1

    report = ingest_documents(documents, provider=tiny, index=index, settings=settings)
    assert report.truncated_chunks == report.chunks_total
