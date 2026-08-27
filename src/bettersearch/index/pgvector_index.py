"""PostgreSQL + pgvector index - the production path.

Three storage decisions here are the whole reason this backend exists, and they
are what make a large content library affordable:

``halfvec`` instead of ``vector``
    fp16 rather than fp32 storage. Exactly half the bytes, negligible recall
    loss at these dimensions.

Truncated (Matryoshka) dimensions
    Set by the embedding provider, not here - 512 instead of a native 1536 is
    another 3x. Combined with halfvec, 1M chunks goes from ~6GB of raw vectors
    to ~500MB, which is the difference between needing a large managed Postgres
    and fitting on a small one.

HNSW over IVFFlat
    Better recall/latency, and no need to pick a list count up front or rebuild
    when the corpus grows.

The generated ``tsvector`` column with a GIN index lets the keyword baseline run
against the same rows. Note that the ``Searcher`` deliberately uses the Python
BM25 implementation for both backends so that scoring is identical and the
keyword-vs-semantic comparison is apples to apples; ``keyword_query`` below is
what you would switch to at scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import (
    Chunk,
    EmbeddedChunk,
    EmptyIndexError,
    IndexStats,
    ModelMismatchError,
    ScoredChunk,
)

BACKEND_NAME = "pgvector"
TABLE = "bettersearch_chunk"


def _to_pg_vector(vector: np.ndarray) -> str:
    """Render a vector as a pgvector literal.

    Passed as text and cast in SQL rather than using pgvector's Python type
    adapters, so this works across pgvector-python versions without pinning.
    """
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{v:.7g}" for v in values.tolist()) + "]"


def _row_to_chunk(row: tuple) -> Chunk:
    chunk_id, doc_id, title, ordinal, text, embed_text, content_hash, url = row[:8]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        ordinal=ordinal,
        text=text,
        embed_text=embed_text,
        content_hash=content_hash,
        url=url,
    )


_CHUNK_COLUMNS = "chunk_id, doc_id, title, ordinal, text, embed_text, content_hash, url"


class PgVectorIndex:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The pgvector backend needs psycopg. "
                'Install it with: pip install "bettersearch[pgvector]"'
            ) from exc

        if not database_url:
            raise ValueError(
                "BETTERSEARCH_DATABASE_URL (or DATABASE_URL) must be set to use "
                "the pgvector backend."
            )
        self._psycopg = psycopg
        self._database_url = database_url

    def _connect(self):
        return self._psycopg.connect(self._database_url, autocommit=True)

    # ----------------------------------------------------------------- schema
    def _ensure_schema(self, conn, dimensions: int) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    chunk_id     text PRIMARY KEY,
                    doc_id       text NOT NULL,
                    title        text NOT NULL,
                    ordinal      integer NOT NULL,
                    text         text NOT NULL,
                    embed_text   text NOT NULL,
                    content_hash text NOT NULL,
                    url          text,
                    model_id     text NOT NULL,
                    dims         integer NOT NULL,
                    embedding    halfvec({dimensions}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english',
                            coalesce(title, '') || ' ' || coalesce(text, ''))
                    ) STORED
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw "
                f"ON {TABLE} USING hnsw (embedding halfvec_cosine_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_tsv_gin "
                f"ON {TABLE} USING gin (tsv)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_content_hash "
                f"ON {TABLE} (content_hash)"
            )

    def _table_exists(self, conn) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (TABLE,))
            return cur.fetchone()[0] is not None

    def _indexed_model_id(self, conn) -> str | None:
        if not self._table_exists(conn):
            return None
        with conn.cursor() as cur:
            cur.execute(f"SELECT model_id FROM {TABLE} LIMIT 1")
            row = cur.fetchone()
        return row[0] if row else None

    # ---------------------------------------------------------------- writing
    def upsert(self, chunks: Sequence[EmbeddedChunk], *, model_id: str) -> int:
        if not chunks:
            return 0
        dimensions = int(np.asarray(chunks[0].vector).reshape(-1).shape[0])

        with self._connect() as conn:
            existing_model = self._indexed_model_id(conn)
            if existing_model is not None and existing_model != model_id:
                raise ModelMismatchError(
                    f"Table {TABLE} holds vectors from {existing_model!r} but this "
                    f"write uses {model_id!r}. Clear the table or use a new one."
                )
            self._ensure_schema(conn, dimensions)

            rows = [
                (
                    c.chunk.chunk_id,
                    c.chunk.doc_id,
                    c.chunk.title,
                    c.chunk.ordinal,
                    c.chunk.text,
                    c.chunk.embed_text,
                    c.chunk.content_hash,
                    c.chunk.url,
                    model_id,
                    dimensions,
                    _to_pg_vector(c.vector),
                )
                for c in chunks
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {TABLE} (
                        chunk_id, doc_id, title, ordinal, text, embed_text,
                        content_hash, url, model_id, dims, embedding
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::halfvec)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        title = EXCLUDED.title,
                        ordinal = EXCLUDED.ordinal,
                        text = EXCLUDED.text,
                        embed_text = EXCLUDED.embed_text,
                        content_hash = EXCLUDED.content_hash,
                        url = EXCLUDED.url,
                        model_id = EXCLUDED.model_id,
                        dims = EXCLUDED.dims,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )
        return len(chunks)

    def clear(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {TABLE}")

    # ---------------------------------------------------------------- reading
    def query(
        self, vector: np.ndarray, *, top_k: int, model_id: str
    ) -> list[ScoredChunk]:
        with self._connect() as conn:
            indexed_model = self._indexed_model_id(conn)
            if indexed_model is None:
                raise EmptyIndexError(
                    "Index is empty - run `bettersearch ingest` first."
                )
            if indexed_model != model_id:
                raise ModelMismatchError(
                    f"Table {TABLE} holds vectors from {indexed_model!r} but the "
                    f"query was embedded with {model_id!r}. Not comparable."
                )
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {_CHUNK_COLUMNS}, 1 - (embedding <=> %s::halfvec) AS score
                    FROM {TABLE}
                    ORDER BY embedding <=> %s::halfvec
                    LIMIT %s
                    """,
                    (_to_pg_vector(vector), _to_pg_vector(vector), top_k),
                )
                rows = cur.fetchall()

        return [
            ScoredChunk(chunk=_row_to_chunk(row), score=float(row[8]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def keyword_query(self, text: str, *, top_k: int) -> list[ScoredChunk]:
        """Full-text search over the generated tsvector column.

        Not used by ``Searcher`` (which uses the Python BM25 for both backends so
        the comparison is like-for-like), but this is the query you would move to
        once the corpus outgrows loading every chunk into memory.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {_CHUNK_COLUMNS},
                           ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
                    FROM {TABLE}
                    WHERE tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (text, text, top_k),
                )
                rows = cur.fetchall()
        return [
            ScoredChunk(chunk=_row_to_chunk(row), score=float(row[8]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def existing_hashes(self) -> set[str]:
        with self._connect() as conn:
            if not self._table_exists(conn):
                return set()
            with conn.cursor() as cur:
                cur.execute(f"SELECT content_hash FROM {TABLE}")
                return {row[0] for row in cur.fetchall()}

    def all_chunks(self) -> list[Chunk]:
        """Every chunk in the table.

        Fine at PoC scale. At library scale this is the call to replace with
        ``keyword_query`` above rather than pulling the corpus into memory.
        """
        with self._connect() as conn:
            if not self._table_exists(conn):
                return []
            with conn.cursor() as cur:
                cur.execute(f"SELECT {_CHUNK_COLUMNS} FROM {TABLE} ORDER BY chunk_id")
                return [_row_to_chunk(row) for row in cur.fetchall()]

    def stats(self) -> IndexStats:
        with self._connect() as conn:
            if not self._table_exists(conn):
                return IndexStats(backend=BACKEND_NAME, chunk_count=0)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*), min(model_id), min(dims) FROM {TABLE}"
                )
                count, model_id, dims = cur.fetchone()
        return IndexStats(
            backend=BACKEND_NAME,
            chunk_count=int(count),
            model_id=model_id,
            dimensions=int(dims) if dims is not None else None,
            extra={"table": TABLE, "storage": "halfvec + hnsw"},
        )
