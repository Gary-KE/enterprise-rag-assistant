"""Hybrid retrieval: keyword search and vector search, merged by rank, then reranked."""
from __future__ import annotations

from rag.config import Settings
from rag.models import Hit
from rag.providers.base import Embedder, Reranker
from rag.store.base import Store


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: an item scores 1/(k+rank) in each list it appears in."""
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, reranker: Reranker, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings

    def retrieve(self, query: str, access_level: int = 0) -> list[Hit]:
        qvec = self.embedder.embed([query])[0]
        keyword = self.store.keyword_search(query, self.s.top_k_each, access_level)
        vector = self.store.vector_search(qvec, self.s.top_k_each, access_level)

        fused = rrf_fuse(
            [[c.chunk_id for c, _ in keyword], [c.chunk_id for c, _ in vector]], self.s.rrf_k
        )
        chunks = {c.chunk_id: c for c, _ in keyword + vector}
        kw_rank = {c.chunk_id: r for r, (c, _) in enumerate(keyword, start=1)}
        vec_rank = {c.chunk_id: r for r, (c, _) in enumerate(vector, start=1)}

        top_ids = sorted(fused, key=fused.get, reverse=True)[: self.s.fused_k]
        hits = [
            Hit(
                chunk=chunks[i],
                fused_score=fused[i],
                rank_keyword=kw_rank.get(i),
                rank_vector=vec_rank.get(i),
            )
            for i in top_ids
        ]
        if not hits:
            return []
        reranked = self.reranker.rerank(query, hits)
        return reranked[: self.s.final_k]
