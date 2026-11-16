"""Embed the downloaded corpus and store it, ready for the app.

    python scripts/build_index.py --target pgvector     # into Postgres (DATABASE_URL), e.g. Supabase
    python scripts/build_index.py --target jsonl        # into data/index.jsonl for the memory store

Uses the real embedding model from the RAG_EMBED_* variables. Safe to rerun: documents already stored
are skipped, and duplicate passages are not stored twice.
"""
from __future__ import annotations

import argparse
import sys

from common import DATA, build_indexed_pipeline

from rag.config import Settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["pgvector", "jsonl"], required=True)
    args = ap.parse_args()
    settings = Settings.from_env()
    if settings.providers != "http":
        raise SystemExit("set RAG_PROVIDERS=http and the RAG_EMBED_* variables: the index needs real embeddings")
    settings.store = "pgvector" if args.target == "pgvector" else "memory"
    if args.target == "pgvector":
        from rag.store.pgvector import PgVectorStore
        PgVectorStore(settings.database_url, settings.embed_dim).init_schema()
    pipeline = build_indexed_pipeline(settings)
    if args.target == "jsonl":
        from rag.index_io import save_index
        store = pipeline.store
        docs = [d for d, _ in store.list_documents(99)]
        chunks = [store.get_chunk(cid) for cid in store._chunks]
        save_index(DATA / "index.jsonl", docs, chunks)
        print(f"wrote {len(docs)} documents and {len(chunks)} chunks to {DATA / 'index.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
