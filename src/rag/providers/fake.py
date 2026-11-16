"""Deterministic stand-ins for the models. For tests and the CI smoke evaluation only.

They prove the wiring works. They say nothing about how well real models will perform.
"""
from __future__ import annotations

import hashlib
import html
import math
import re

from rag.models import Hit
from rag.text_utils import content_tokens


class HashingEmbedder:
    """Bag-of-words hashing into a fixed-size vector. Word overlap only, not real meaning."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in content_tokens(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OverlapReranker:
    """Baseline reranker: weighted share of the query's content words found in the chunk.

    A word found in every candidate carries little weight, so a question made of company-name
    words alone does not look relevant. A real cross-encoder should replace this.
    """

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q_tokens = set(content_tokens(query))
        n = len(hits)
        if not q_tokens or n == 0:
            for h in hits:
                h.relevance = 0.0
            return hits
        chunk_sets = [set(content_tokens(h.chunk.text)) for h in hits]
        weights: dict[str, float] = {}
        for tok in q_tokens:
            df = sum(tok in s for s in chunk_sets)
            weights[tok] = max(0.1, 1.0 - df / n) if n >= 3 else 1.0
        total = sum(weights.values())
        for h, tokens in zip(hits, chunk_sets):
            h.relevance = sum(w for tok, w in weights.items() if tok in tokens) / total
        return sorted(hits, key=lambda h: (h.relevance, h.fused_score), reverse=True)


_SOURCE = re.compile(r'<source id="(S\d+)">(.*?)</source>', re.S)
_QUESTION = re.compile(r"<question>(.*?)</question>", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class FakeLLM:
    """Extractive stand-in: copies the best-matching sentence from the sources and cites it."""

    def __init__(self, min_overlap: float = 0.4):
        self.min_overlap = min_overlap

    def complete(self, system: str, user: str) -> str:
        q = _QUESTION.search(user)
        question = html.unescape(q.group(1)) if q else ""
        q_tokens = set(content_tokens(question))
        best: tuple[float, str, str] = (0.0, "", "")
        for marker, body in _SOURCE.findall(user):
            for sentence in _SENTENCE.split(html.unescape(body)):
                s_tokens = set(content_tokens(sentence))
                if not q_tokens or not s_tokens:
                    continue
                score = len(q_tokens & s_tokens) / len(q_tokens)
                if score > best[0]:
                    best = (score, sentence.strip(), marker)
        if best[0] < self.min_overlap:
            return "NOT_FOUND"
        return f"{best[1]} [{best[2]}]"
