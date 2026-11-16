"""Builds the whole pipeline from Settings. One place decides which implementations are used."""
from __future__ import annotations

from dataclasses import dataclass

from rag.answer import AnswerService
from rag.config import Settings
from rag.providers.fake import FakeLLM, HashingEmbedder, OverlapReranker
from rag.retrieval import Retriever


@dataclass
class Pipeline:
    settings: Settings
    store: object
    embedder: object
    reranker: object
    llm: object
    retriever: Retriever
    answers: AnswerService


def build_pipeline(settings: Settings) -> Pipeline:
    settings.validate()

    if settings.store == "pgvector":
        from rag.store.pgvector import PgVectorStore
        store = PgVectorStore(settings.database_url, settings.embed_dim)
    else:
        from rag.store.memory import MemoryStore
        store = MemoryStore()

    if settings.providers == "http":
        from rag.providers.http import OpenAICompatibleEmbedder, OpenAICompatibleLLM
        embedder = OpenAICompatibleEmbedder(
            settings.embed_base_url, settings.embed_api_key, settings.embed_model,
            dimensions=settings.embed_dim,
        )
        llm = OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model, timeout=60.0)
    else:
        embedder = HashingEmbedder(settings.embed_dim)
        llm = FakeLLM()

    if settings.rerank_base_url:
        from rag.providers.rerank_http import FallbackReranker, HttpReranker
        reranker = FallbackReranker(
            HttpReranker(settings.rerank_base_url, settings.rerank_api_key, settings.rerank_model),
            OverlapReranker(),
        )
    else:
        reranker = OverlapReranker()

    retriever = Retriever(store, embedder, reranker, settings)
    answers = AnswerService(retriever, llm, store, settings)
    return Pipeline(settings, store, embedder, reranker, llm, retriever, answers)
