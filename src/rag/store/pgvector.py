"""Postgres + pgvector store.

Every query below was run against the project's Supabase database (Postgres 17 with pgvector) on
2026-11-17, with results checked: keyword and vector search, access-level filtering, the document list
and duplicate detection. `tests/test_pgvector_store.py` runs the class itself against a real database
when TEST_DATABASE_URL is set.
"""
from __future__ import annotations

import re
from typing import Optional

from rag.models import Chunk, SourceMeta
from rag.text_utils import content_tokens

EXTENSION_SQL = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'extensions') THEN "
    "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions; "   # Supabase convention
    "ELSE CREATE EXTENSION IF NOT EXISTS vector; END IF; END $$;"
)

SCHEMA = """

CREATE TABLE IF NOT EXISTS documents (
    doc_id       text PRIMARY KEY,
    title        text NOT NULL,
    publisher    text NOT NULL DEFAULT '',
    published    text NOT NULL DEFAULT '',
    url          text NOT NULL DEFAULT '',
    retrieved    text NOT NULL DEFAULT '',
    file_name    text NOT NULL DEFAULT '',
    access_level integer NOT NULL DEFAULT 0,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page         integer NOT NULL,
    idx          integer NOT NULL,
    text         text NOT NULL,
    text_hash    text NOT NULL,
    access_level integer NOT NULL DEFAULT 0,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx  ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_vec_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks (text_hash);
CREATE INDEX IF NOT EXISTS chunks_doc_idx  ON chunks (doc_id);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;
"""


def _vec(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _or_query(query: str) -> str:
    """Build an OR tsquery from the question's words, so one missing word does not zero the match."""
    terms = [re.sub(r"[^a-z0-9]", "", t) for t in content_tokens(query)]
    terms = sorted({t for t in terms if t})
    return " | ".join(terms)


class PgVectorStore:
    def __init__(self, database_url: str, dim: int):
        import psycopg  # imported lazily so the package works without the postgres extra

        self._psycopg = psycopg
        self.url = database_url
        self.dim = dim

    def _connect(self):
        return self._psycopg.connect(self.url, autocommit=True)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(EXTENSION_SQL)
            conn.execute(SCHEMA.format(dim=self.dim))

    # --- writes -----------------------------------------------------------
    def has_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        return row is not None

    def add_document(self, meta: SourceMeta) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents (doc_id, title, publisher, published, url, retrieved, file_name, access_level)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (doc_id) DO NOTHING""",
                (meta.doc_id, meta.title, meta.publisher, meta.published, meta.url,
                 meta.retrieved, meta.file_name, meta.access_level),
            )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            with conn.transaction():
                for c in chunks:
                    if c.embedding is None:
                        raise ValueError("chunks must be embedded before they are stored")
                    conn.execute(
                        """INSERT INTO chunks (chunk_id, doc_id, page, idx, text, text_hash, access_level, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector) ON CONFLICT (chunk_id) DO NOTHING""",
                        (c.chunk_id, c.doc_id, c.page, c.index, c.text, c.text_hash,
                         c.access_level, _vec(c.embedding)),
                    )

    def existing_hashes(self, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT text_hash FROM chunks WHERE text_hash = ANY(%s)", (hashes,)
            ).fetchall()
        return {r[0] for r in rows}

    # --- reads ------------------------------------------------------------
    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        return Chunk(chunk_id=row[0], doc_id=row[1], page=row[2], index=row[3],
                     text=row[4], text_hash=row[5], access_level=row[6])

    _COLS = "chunk_id, doc_id, page, idx, text, text_hash, access_level"

    def keyword_search(self, query: str, k: int, access_level: int) -> list[tuple[Chunk, float]]:
        tsq = _or_query(query)
        if not tsq:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {self._COLS}, ts_rank_cd(tsv, to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE tsv @@ to_tsquery('english', %s) AND access_level <= %s
                    ORDER BY score DESC LIMIT %s""",
                (tsq, tsq, access_level, k),
            ).fetchall()
        return [(self._row_to_chunk(r), float(r[7])) for r in rows]

    def vector_search(self, vector: list[float], k: int, access_level: int) -> list[tuple[Chunk, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {self._COLS}, 1 - (embedding <=> %s::vector) AS score
                    FROM chunks WHERE access_level <= %s
                    ORDER BY embedding <=> %s::vector LIMIT %s""",
                (_vec(vector), access_level, _vec(vector), k),
            ).fetchall()
        return [(self._row_to_chunk(r), float(r[7])) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {self._COLS} FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_document(self, doc_id: str) -> Optional[SourceMeta]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT doc_id, title, publisher, published, url, retrieved, file_name, access_level
                   FROM documents WHERE doc_id = %s""", (doc_id,)
            ).fetchone()
        return SourceMeta(*row) if row else None

    def list_documents(self, access_level: int) -> list[tuple[SourceMeta, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT d.doc_id, d.title, d.publisher, d.published, d.url, d.retrieved, d.file_name,
                          d.access_level, count(c.chunk_id)
                   FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
                   WHERE d.access_level <= %s
                   GROUP BY d.doc_id ORDER BY d.title""", (access_level,)
            ).fetchall()
        return [(SourceMeta(*r[:8]), int(r[8])) for r in rows]

    def is_healthy(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False
