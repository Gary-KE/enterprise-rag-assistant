"""Check the golden set against the actual downloaded documents.

For every answerable question, the verified quote must appear in its source document. The script also
reports whether the quote fits inside a single chunk with the current chunk settings: a quote split
across two chunks (or across a page break) can never count as a retrieval hit, which caps the
measured scores, so those are listed for fixing.

    python scripts/verify_golden.py            # needs data/raw/*.pdf
"""
from __future__ import annotations

import sys

from common import GOLDEN, RAW, load_sources

from rag.config import Settings
from rag.evaluation.golden import load_golden, squash
from rag.ingest.chunking import chunk_pages
from rag.ingest.extract import extract_pages


def main() -> int:
    items = load_golden(GOLDEN)
    settings = Settings.from_env()
    docs = {}
    for src in load_sources():
        pages, scanned = extract_pages(RAW / f"{src['key']}.pdf")
        chunks = chunk_pages(pages, src["key"], settings.chunk_words, settings.chunk_overlap)
        docs[src["key"]] = {
            "full": squash(" ".join(text for _, text in pages)),
            "pages": [(no, squash(text)) for no, text in pages],
            "chunks": [squash(c.text) for c in chunks],
            "scanned": scanned,
        }
        print(f"[{src['key']}] {len(pages)} pages, {len(chunks)} chunks, {len(scanned)} pages without text")

    missing, split = [], []
    answerable = [i for i in items if i.answerable]
    for item in answerable:
        if item.doc not in docs:
            missing.append(f"{item.id}: unknown doc '{item.doc}'")
            continue
        d = docs[item.doc]
        q = squash(item.quote)
        if q not in d["full"]:
            missing.append(f"{item.id} ({item.section}): quote not found in document")
            continue
        page = next((no for no, text in d["pages"] if q in text), None)
        if page is None or not any(q in c for c in d["chunks"]):
            split.append(f"{item.id} ({item.section}): quote crosses a {'page break' if page is None else 'chunk boundary'}")

    print(f"\n{len(answerable)} answerable questions, {len(items) - len(answerable)} unanswerable")
    print(f"quotes found in their document: {len(answerable) - len(missing)}/{len(answerable)}")
    print(f"quotes inside a single chunk:   {len(answerable) - len(missing) - len(split)}/{len(answerable)}")
    for line in missing:
        print("  MISSING  " + line)
    for line in split:
        print("  SPLIT    " + line)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
