from __future__ import annotations

from bettersearch.chunking import (
    chunk_document,
    chunk_documents,
    content_hash,
    estimate_tokens,
)
from bettersearch.types import Document


def _doc(text: str, *, title: str = "Title", doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, title=title, text=text)


def test_short_document_becomes_one_chunk():
    chunks = chunk_document(_doc("One short paragraph."))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1:0"
    assert chunks[0].ordinal == 0


def test_embed_text_is_prefixed_with_the_title():
    # This prefix is the single largest retrieval-quality lever, so it is
    # asserted rather than left as an implementation detail.
    chunks = chunk_document(_doc("His army crossed the Channel.", title="Hastings"))
    assert chunks[0].embed_text.startswith("Hastings\n\n")
    assert "His army crossed the Channel." in chunks[0].embed_text
    # The displayed text stays clean - the title is only for embedding.
    assert chunks[0].text == "His army crossed the Channel."


def test_paragraphs_are_grouped_up_to_the_target():
    paragraphs = ["word " * 60 for _ in range(6)]
    chunks = chunk_document(_doc("\n\n".join(paragraphs)), target_tokens=200)
    assert len(chunks) > 1
    for chunk in chunks:
        # Allow one unit of overshoot: a group is closed *after* the unit that
        # crosses the target, never mid-paragraph.
        assert estimate_tokens(chunk.text) <= 200 + estimate_tokens(paragraphs[0])


def test_chunks_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    chunks = chunk_document(
        _doc("\n\n".join(paragraphs)), target_tokens=150, overlap_tokens=60
    )
    assert len(chunks) >= 2
    tail = chunks[0].text.split("\n\n")[-1]
    assert tail in chunks[1].text


def test_overlap_never_loops_forever():
    # A pathological case: overlap large enough that a naive carry would
    # reproduce the whole chunk and never advance.
    paragraphs = ["word " * 100 for _ in range(5)]
    chunks = chunk_document(
        _doc("\n\n".join(paragraphs)), target_tokens=140, overlap_tokens=130
    )
    assert 0 < len(chunks) < 50


def test_oversized_paragraph_splits_on_sentences():
    paragraph = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = chunk_document(_doc(paragraph), target_tokens=100)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_ordinals_are_sequential_and_ids_unique():
    paragraphs = ["word " * 80 for _ in range(8)]
    chunks = chunk_document(_doc("\n\n".join(paragraphs)), target_tokens=150)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_content_hash_is_stable_and_sensitive():
    a = chunk_document(_doc("Identical body text."))[0]
    b = chunk_document(_doc("Identical body text."))[0]
    assert a.content_hash == b.content_hash

    # A title change must invalidate the chunk, because the title is embedded.
    c = chunk_document(_doc("Identical body text.", title="Different"))[0]
    assert c.content_hash != a.content_hash


def test_content_hash_helper_matches_chunk_hash():
    chunk = chunk_document(_doc("Body."))[0]
    assert content_hash(chunk.embed_text) == chunk.content_hash


def test_empty_document_produces_no_chunks():
    assert chunk_document(_doc("   \n\n  ")) == []


def test_chunk_documents_preserves_document_order():
    docs = [_doc("First body.", doc_id="a"), _doc("Second body.", doc_id="b")]
    chunks = chunk_documents(docs)
    assert [c.doc_id for c in chunks] == ["a", "b"]
