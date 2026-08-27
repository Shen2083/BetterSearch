"""Brute-force in-memory index. The default, and the one the demo runs on.

Exact cosine similarity over a single float32 matrix. No approximation, no
index build step, no database. Comfortable to roughly 100k chunks on a laptop;
past that, switch ``BETTERSEARCH_INDEX`` to ``pgvector``.

Because every provider returns L2-normalised vectors, cosine similarity is just
``matrix @ query``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..types import (
    Chunk,
    EmbeddedChunk,
    EmptyIndexError,
    IndexStats,
    ModelMismatchError,
    ScoredChunk,
)

BACKEND_NAME = "numpy"


class NumpyVectorIndex:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._model_id: str | None = None
        self._position: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ paths
    @property
    def _vectors_file(self) -> Path:
        return self._path.with_suffix(".npz")

    @property
    def _meta_file(self) -> Path:
        return self._path.with_suffix(".json")

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not (self._vectors_file.exists() and self._meta_file.exists()):
            return
        meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
        self._model_id = meta.get("model_id")
        self._chunks = [Chunk.from_dict(c) for c in meta.get("chunks", [])]
        with np.load(self._vectors_file) as payload:
            self._vectors = payload["vectors"].astype(np.float32, copy=False)
        self._reindex_positions()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        vectors = (
            self._vectors
            if self._vectors is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.savez_compressed(self._vectors_file, vectors=vectors)
        self._meta_file.write_text(
            json.dumps(
                {
                    "model_id": self._model_id,
                    "dimensions": int(vectors.shape[1]) if vectors.size else None,
                    "chunks": [c.to_dict() for c in self._chunks],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _reindex_positions(self) -> None:
        self._position = {c.chunk_id: i for i, c in enumerate(self._chunks)}

    # ---------------------------------------------------------------- writing
    def upsert(self, chunks: Sequence[EmbeddedChunk], *, model_id: str) -> int:
        if not chunks:
            return 0

        # Changing embedding model invalidates every existing vector. Refuse to
        # mix rather than silently corrupting the index.
        if self._model_id is not None and self._model_id != model_id:
            raise ModelMismatchError(
                f"Index was built with {self._model_id!r} but this write uses "
                f"{model_id!r}. Re-ingest into a fresh index path, or clear this one."
            )
        self._model_id = model_id

        new_vectors = np.stack([c.vector for c in chunks]).astype(
            np.float32, copy=False
        )

        if self._vectors is None or self._vectors.size == 0:
            self._vectors = new_vectors
            self._chunks = [c.chunk for c in chunks]
        else:
            replacements: list[tuple[int, EmbeddedChunk]] = []
            additions: list[EmbeddedChunk] = []
            for embedded in chunks:
                existing = self._position.get(embedded.chunk.chunk_id)
                if existing is None:
                    additions.append(embedded)
                else:
                    replacements.append((existing, embedded))

            for position, embedded in replacements:
                self._vectors[position] = embedded.vector
                self._chunks[position] = embedded.chunk

            if additions:
                self._vectors = np.vstack(
                    [self._vectors, np.stack([c.vector for c in additions])]
                ).astype(np.float32, copy=False)
                self._chunks.extend(c.chunk for c in additions)

        self._reindex_positions()
        self._save()
        return len(chunks)

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []
        self._model_id = None
        self._position = {}
        for path in (self._vectors_file, self._meta_file):
            path.unlink(missing_ok=True)

    # ---------------------------------------------------------------- reading
    def query(
        self, vector: np.ndarray, *, top_k: int, model_id: str
    ) -> list[ScoredChunk]:
        if self._vectors is None or self._vectors.size == 0:
            raise EmptyIndexError("Index is empty - run `bettersearch ingest` first.")
        if self._model_id != model_id:
            raise ModelMismatchError(
                f"Index was built with {self._model_id!r} but the query was "
                f"embedded with {model_id!r}. These vectors are not comparable."
            )

        query_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query_vector

        k = min(top_k, scores.shape[0])
        # argpartition is O(n) vs argsort's O(n log n); we only sort the top k.
        candidates = np.argpartition(-scores, k - 1)[:k]
        ordered = candidates[np.argsort(-scores[candidates])]

        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ordered, start=1)
        ]

    def existing_hashes(self) -> set[str]:
        return {c.content_hash for c in self._chunks}

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def stats(self) -> IndexStats:
        dimensions = (
            int(self._vectors.shape[1])
            if self._vectors is not None and self._vectors.size
            else None
        )
        return IndexStats(
            backend=BACKEND_NAME,
            chunk_count=len(self._chunks),
            model_id=self._model_id,
            dimensions=dimensions,
            extra={"path": str(self._path)},
        )
