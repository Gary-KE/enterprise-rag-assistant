"""Interfaces for the three model-backed parts. Swap implementations without touching the pipeline."""
from __future__ import annotations

from typing import Protocol

from rag.models import Hit


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Set hit.relevance (0 to 1) on every hit and return them best first."""
        ...


class ProviderError(RuntimeError):
    """A model provider failed after retries."""
