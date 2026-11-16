"""Split pages into overlapping word windows. Chunks never cross a page, so a citation page is exact."""
from __future__ import annotations

import hashlib
import re

from rag.models import Chunk


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    """Stable fingerprint used to spot duplicate chunks."""
    return hashlib.sha256(normalise(text).lower().encode("utf-8")).hexdigest()[:16]


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    chunk_words: int,
    overlap: int,
    access_level: int = 0,
    min_words: int = 5,
) -> list[Chunk]:
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    step = chunk_words - overlap
    chunks: list[Chunk] = []
    for page_no, raw in pages:
        text = normalise(raw)
        if not text:
            continue
        words = text.split(" ")
        n = len(words)
        if n < min_words:
            continue
        # Window starts: every `step` words, and a final window aligned to the page end so the
        # last words are always covered by a full-size chunk (never dropped, never a tiny tail).
        starts = list(range(0, max(n - chunk_words, 0) + 1, step))
        if starts[-1] + chunk_words < n:
            starts.append(n - chunk_words)
        for index, start in enumerate(starts):
            body = " ".join(words[start:start + chunk_words])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page_no}-c{index}",
                    doc_id=doc_id,
                    page=page_no,
                    index=index,
                    text=body,
                    text_hash=text_hash(body),
                    access_level=access_level,
                )
            )
    return chunks
