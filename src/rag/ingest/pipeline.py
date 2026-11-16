"""Ingestion: file in, searchable chunks out. Safe to run twice on the same file."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.config import Settings
from rag.ingest.chunking import chunk_pages
from rag.ingest.extract import extract_pages
from rag.models import SourceMeta
from rag.providers.base import Embedder
from rag.store.base import Store

log = logging.getLogger("rag.ingest")


@dataclass
class IngestResult:
    doc_id: str = ""
    status: str = "failed"                  # ingested | skipped_duplicate | failed
    chunks_added: int = 0
    duplicate_chunks_skipped: int = 0
    scanned_pages: list[int] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def file_doc_id(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def ingest_file(
    path: Path,
    meta: dict,
    store: Store,
    embedder: Embedder,
    settings: Settings,
) -> IngestResult:
    """meta keys: title, publisher, published, url, retrieved, file_name, access_level."""
    result = IngestResult()
    try:
        doc_id = file_doc_id(path)
        result.doc_id = doc_id
        if store.has_document(doc_id):
            result.status = "skipped_duplicate"
            return result

        pages, scanned = extract_pages(path)
        result.scanned_pages = scanned
        access_level = int(meta.get("access_level", 0))
        chunks = chunk_pages(pages, doc_id, settings.chunk_words, settings.chunk_overlap, access_level)
        if not chunks:
            result.error = "no extractable text (a scanned PDF needs OCR first)"
            return result

        known = store.existing_hashes([c.text_hash for c in chunks])
        seen: set[str] = set()
        fresh = []
        for c in chunks:
            if c.text_hash in known or c.text_hash in seen:
                continue
            seen.add(c.text_hash)
            fresh.append(c)
        result.duplicate_chunks_skipped = len(chunks) - len(fresh)

        for start in range(0, len(fresh), settings.embed_batch):
            batch = fresh[start:start + settings.embed_batch]
            vectors = embedder.embed([c.text for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedder returned the wrong number of vectors")
            for c, v in zip(batch, vectors):
                c.embedding = v

        store.add_document(
            SourceMeta(
                doc_id=doc_id,
                title=meta.get("title") or path.stem,
                publisher=meta.get("publisher", ""),
                published=meta.get("published", ""),
                url=meta.get("url", ""),
                retrieved=meta.get("retrieved", ""),
                file_name=meta.get("file_name") or path.name,
                access_level=access_level,
            )
        )
        store.add_chunks(fresh)
        result.chunks_added = len(fresh)
        result.status = "ingested"
        log.info("ingested doc=%s chunks=%d scanned_pages=%d", doc_id, len(fresh), len(scanned))
    except Exception as exc:  # report the failure instead of crashing a batch
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        log.warning("ingest failed for %s: %s", path.name, result.error)
    return result
