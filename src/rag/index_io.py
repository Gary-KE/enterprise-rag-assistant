"""Save and load a built index (documents, chunks and their embeddings) as JSON Lines.

Lets the memory store start with a corpus that was embedded once, elsewhere, without re-embedding.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag.models import Chunk, SourceMeta


def save_index(path: Path, docs: list[SourceMeta], chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"type": "document", **asdict(d)}) + "\n")
        for c in chunks:
            fh.write(json.dumps({"type": "chunk", **asdict(c)}) + "\n")


def load_index(path: Path, store) -> tuple[int, int]:
    docs: list[SourceMeta] = []
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.pop("type")
            if kind == "document":
                docs.append(SourceMeta(**row))
            elif kind == "chunk":
                chunks.append(Chunk(**row))
            else:
                raise ValueError(f"unknown record type: {kind}")
    for d in docs:
        store.add_document(d)
    store.add_chunks(chunks)
    return len(docs), len(chunks)
