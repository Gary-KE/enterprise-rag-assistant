"""In-memory store. Used for tests, the CI evaluation and quick local runs. Not durable."""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from rag.models import Chunk, SourceMeta
from rag.text_utils import content_tokens


class MemoryStore:
    def __init__(self) -> None:
        self._docs: dict[str, SourceMeta] = {}
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._lengths: list[int] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    # --- writes -----------------------------------------------------------
    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def add_document(self, meta: SourceMeta) -> None:
        self._docs[meta.doc_id] = meta

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError("chunks must be embedded before they are stored")
            self._chunks[c.chunk_id] = c
        self._dirty = True

    def existing_hashes(self, hashes: list[str]) -> set[str]:
        known = {c.text_hash for c in self._chunks.values()}
        return {h for h in hashes if h in known}

    # --- reads ------------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_document(self, doc_id: str) -> Optional[SourceMeta]:
        return self._docs.get(doc_id)

    def is_healthy(self) -> bool:
        return True

    def list_documents(self, access_level: int) -> list[tuple[SourceMeta, int]]:
        counts: dict[str, int] = {}
        for c in self._chunks.values():
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        docs = [d for d in self._docs.values() if d.access_level <= access_level]
        return [(d, counts.get(d.doc_id, 0)) for d in sorted(docs, key=lambda d: d.title)]

    def _rebuild(self) -> None:
        self._order = list(self._chunks)
        self._tf = [Counter(content_tokens(self._chunks[i].text)) for i in self._order]
        self._lengths = [sum(t.values()) for t in self._tf]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        if self._order:
            m = np.array([self._chunks[i].embedding for i in self._order], dtype=float)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = m / norms
        else:
            self._matrix = None
        self._dirty = False

    def keyword_search(self, query: str, k: int, access_level: int) -> list[tuple[Chunk, float]]:
        if self._dirty:
            self._rebuild()
        q = content_tokens(query)
        n = len(self._order)
        if not q or n == 0:
            return []
        avg_len = (sum(self._lengths) / n) or 1.0
        k1, b = 1.5, 0.75
        scored: list[tuple[Chunk, float]] = []
        for i, chunk_id in enumerate(self._order):
            chunk = self._chunks[chunk_id]
            if chunk.access_level > access_level:
                continue
            score = 0.0
            for term in set(q):
                f = self._tf[i].get(term, 0)
                if not f:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self._lengths[i] / avg_len))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    def vector_search(self, vector: list[float], k: int, access_level: int) -> list[tuple[Chunk, float]]:
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return []
        v = np.array(vector, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            return []
        sims = self._matrix @ (v / norm)
        order = np.argsort(-sims)
        out: list[tuple[Chunk, float]] = []
        for idx in order:
            chunk = self._chunks[self._order[int(idx)]]
            if chunk.access_level > access_level:
                continue
            out.append((chunk, float(sims[int(idx)])))
            if len(out) >= k:
                break
        return out
