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
        from rag.providers.http import FallbackLLM
        names = [settings.llm_model] + [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]
        clients = [OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, name, timeout=60.0,
                                       min_interval_s=settings.llm_min_interval_s,
                                       reasoning_effort=settings.llm_reasoning_effort) for name in names]
        if settings.llm_backup_base_url and settings.llm_backup_model:
            clients.append(OpenAICompatibleLLM(
                settings.llm_backup_base_url, settings.llm_backup_api_key or settings.embed_api_key,
                settings.llm_backup_model, timeout=60.0))
        llm = clients[0] if len(clients) == 1 else FallbackLLM(clients)
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
