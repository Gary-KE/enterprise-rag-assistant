"""Central settings. Everything tunable lives here and can be set from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # chunking (words, roughly 1.4 tokens each)
    chunk_words: int = 350
    chunk_overlap: int = 50
    embed_batch: int = 64

    # retrieval
    top_k_each: int = 20      # candidates taken from each of keyword and vector search
    fused_k: int = 20         # candidates kept after rank fusion, sent to the reranker
    final_k: int = 5          # chunks given to the LLM
    rrf_k: int = 60           # reciprocal rank fusion constant
    # Chunks whose best reranker score is below this make the system decline.
    # The value depends on the reranker: calibrate it on the golden set.
    min_relevance: float = 0.4
    # Threshold used when the hosted reranker fails and the baseline reranker answers instead.
    min_relevance_fallback: float = 0.4

    # answering
    strict_sentence_citations: bool = False   # True: every sentence must carry a citation
    max_answer_retries: int = 1

    # api
    jwt_secret: str = ""
    jwt_ttl_minutes: int = 60
    allow_dev_token: bool = False             # never enable in a real deployment
    rate_limit_per_min: int = 30
    max_upload_mb: int = 25
    trusted_proxy_hops: int = 0               # set to 1 behind a single reverse proxy such as Render
    preload_index: str = ""                   # memory store only: JSONL index to load at start-up

    # providers and storage
    providers: str = "fake"                   # fake | http
    store: str = "memory"                     # memory | pgvector
    database_url: str = ""
    embed_dim: int = 256
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    rerank_base_url: str = ""                 # optional: leave empty to use the baseline reranker
    rerank_api_key: str = ""
    rerank_model: str = ""
    admin_key: str = ""                       # exchanged for an admin token that may upload documents

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()

        def grab(env: str, attr: str, cast=str):
            val = os.environ.get(env)
            if val is not None and val != "":
                if cast is bool:
                    setattr(s, attr, val.strip().lower() in {"1", "true", "yes"})
                else:
                    setattr(s, attr, cast(val))

        grab("RAG_PROVIDERS", "providers")
        grab("RAG_STORE", "store")
        grab("DATABASE_URL", "database_url")
        grab("RAG_JWT_SECRET", "jwt_secret")
        grab("RAG_ALLOW_DEV_TOKEN", "allow_dev_token", bool)
        grab("RAG_EMBED_DIM", "embed_dim", int)
        grab("RAG_MIN_RELEVANCE", "min_relevance", float)
        grab("RAG_STRICT_CITATIONS", "strict_sentence_citations", bool)
        grab("RAG_RATE_LIMIT_PER_MIN", "rate_limit_per_min", int)
        grab("RAG_LLM_BASE_URL", "llm_base_url")
        grab("RAG_LLM_API_KEY", "llm_api_key")
        grab("RAG_LLM_MODEL", "llm_model")
        grab("RAG_EMBED_BASE_URL", "embed_base_url")
        grab("RAG_EMBED_API_KEY", "embed_api_key")
        grab("RAG_EMBED_MODEL", "embed_model")
        grab("RAG_RERANK_BASE_URL", "rerank_base_url")
        grab("RAG_RERANK_API_KEY", "rerank_api_key")
        grab("RAG_RERANK_MODEL", "rerank_model")
        grab("RAG_ADMIN_KEY", "admin_key")
        grab("RAG_TRUSTED_PROXY_HOPS", "trusted_proxy_hops", int)
        grab("RAG_PRELOAD_INDEX", "preload_index")
        return s

    def validate(self) -> None:
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise ValueError("chunk_overlap must be smaller than chunk_words")
        if self.final_k > self.fused_k:
            raise ValueError("final_k cannot exceed fused_k")
        if self.providers not in {"fake", "http"}:
            raise ValueError("providers must be 'fake' or 'http'")
        if self.store not in {"memory", "pgvector"}:
            raise ValueError("store must be 'memory' or 'pgvector'")
        if self.providers == "http":
            for name in ("llm_base_url", "llm_api_key", "llm_model", "embed_base_url", "embed_api_key", "embed_model"):
                if not getattr(self, name):
                    raise ValueError(f"providers=http requires {name} to be set")
        if self.store == "pgvector" and not self.database_url:
            raise ValueError("store=pgvector requires DATABASE_URL")
        if bool(self.rerank_base_url) != bool(self.rerank_api_key and self.rerank_model):
            raise ValueError("set RAG_RERANK_BASE_URL, RAG_RERANK_API_KEY and RAG_RERANK_MODEL together, or none")
        if self.admin_key and len(self.admin_key) < 24:
            raise ValueError("RAG_ADMIN_KEY must be at least 24 characters")
        if self.admin_key and len(self.jwt_secret) < 32:
            raise ValueError("RAG_JWT_SECRET must be at least 32 characters when uploads are enabled")
