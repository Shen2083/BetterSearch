"""Paragraph-aware chunking with overlap and content hashing.

Design notes:

* We split on paragraph boundaries and never mid-paragraph unless a single
  paragraph exceeds the target on its own, in which case we fall back to
  sentence boundaries. Chunks that end mid-thought embed badly.
* Each chunk's *embed text* is prefixed with the document title. A chunk that
  reads "his army crossed the Channel in September" is nearly useless on its
  own; with "Battle of Hastings" attached it carries its subject into the
  vector. This one line is the largest single quality lever in the pipeline.
* The content hash is taken over the embed text, so a title change correctly
  invalidates every chunk of that document.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .types import Chunk, Document

# Rough token estimate. We only need this to be consistent, not exact - it
# decides chunk sizes, and every provider tokenises differently anyway.
_TOKENS_PER_WORD = 1.3

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count for sizing decisions."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _paragraphs(text: str) -> list[str]:
    parts = (p.strip() for p in _PARAGRAPH_SPLIT.split(text))
    return [p for p in parts if p]


def _split_long_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
    if not sentences:
        return [paragraph]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing units of a chunk that should carry into the next one."""
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if tail and total + unit_tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit_tokens
    # Never carry the whole chunk forward - that would loop forever.
    if len(tail) >= len(units):
        tail = tail[1:]
    return tail


def chunk_document(
    document: Document,
    *,
    target_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Split one document into overlapping, title-prefixed chunks."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    units: list[str] = []
    for paragraph in _paragraphs(document.text):
        if estimate_tokens(paragraph) > target_tokens:
            units.extend(_split_long_paragraph(paragraph, target_tokens))
        else:
            units.append(paragraph)

    if not units:
        return []

    grouped: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            grouped.append(current)
            carry = _overlap_tail(current, overlap_tokens)
            current = list(carry)
            current_tokens = sum(estimate_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        grouped.append(current)

    chunks: list[Chunk] = []
    for ordinal, group in enumerate(grouped):
        text = "\n\n".join(group)
        embed_text = f"{document.title}\n\n{text}"
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{ordinal}",
                doc_id=document.doc_id,
                title=document.title,
                ordinal=ordinal,
                text=text,
                embed_text=embed_text,
                content_hash=content_hash(embed_text),
                url=document.url,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    target_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks
