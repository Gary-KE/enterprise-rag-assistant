"""Runs PgVectorStore against a real Postgres with pgvector. Skipped unless TEST_DATABASE_URL is set.

Use a throwaway database: the test creates the tables and deletes its own rows afterwards.
"""
import os

import pytest

from rag.models import Chunk, SourceMeta

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
DIM = 768


def vec(i):
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def test_round_trip_search_and_access_levels():
    pytest.importorskip("psycopg")
    from rag.store.pgvector import PgVectorStore
    store = PgVectorStore(URL, DIM)
    store.init_schema()
    try:
        store.add_document(SourceMeta("pgtest-pub", "Public Act", url="https://example.org"))
        store.add_document(SourceMeta("pgtest-sec", "Secret", access_level=2))
        store.add_chunks([
            Chunk("pgtest-pub-p3-c0", "pgtest-pub", 3, 0, "notify the Commissioner within seventy-two hours of a breach", "pgh1", 0, vec(0)),
            Chunk("pgtest-pub-p5-c0", "pgtest-pub", 5, 0, "complaints are concluded within ninety days", "pgh2", 0, vec(1)),
            Chunk("pgtest-sec-p1-c0", "pgtest-sec", 1, 0, "vault breach procedure within seventy-two hours", "pgh3", 2, vec(0)),
        ])
        kw = store.keyword_search("breach hours", 5, 0)
        assert [c.chunk_id for c, _ in kw] == ["pgtest-pub-p3-c0"]
        vs = store.vector_search(vec(0), 5, 0)
        assert vs[0][0].chunk_id == "pgtest-pub-p3-c0" and all(c.access_level == 0 for c, _ in vs)
        assert any(c.chunk_id == "pgtest-sec-p1-c0" for c, _ in store.vector_search(vec(0), 5, 2))
        assert store.existing_hashes(["pgh1", "nope"]) == {"pgh1"}
        assert store.get_chunk("pgtest-pub-p5-c0").page == 5
        assert store.has_document("pgtest-pub") and store.is_healthy()
        titles = [d.title for d, n in store.list_documents(0)]
        assert "Public Act" in titles and "Secret" not in titles
    finally:
        with store._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id LIKE 'pgtest-%'")
