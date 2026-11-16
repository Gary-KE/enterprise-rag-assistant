"""Hosted cross-encoder reranker using the Jina/Cohere-style /rerank API.

Request:  {"model", "query", "documents": [str], "top_n"}
Response: {"results": [{"index": int, "relevance_score": float}]}
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from rag.models import Hit
from rag.providers.base import ProviderError
from rag.providers.http import RETRY_STATUS


class HttpReranker:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        attempts: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.attempts = attempts
        self.sleep = sleep
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        payload = {
            "model": self.model,
            "query": query,
            "documents": [h.chunk.text for h in hits],
            "top_n": len(hits),
        }
        last = "unknown error"
        for attempt in range(self.attempts):
            try:
                resp = self.http.post("/rerank", json=payload)
                if resp.status_code in RETRY_STATUS:
                    last = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise ProviderError(f"rerank failed with HTTP {resp.status_code}")
                else:
                    results = resp.json()["results"]
                    for h in hits:
                        h.relevance = 0.0
                    for r in results:
                        idx = int(r["index"])
                        if 0 <= idx < len(hits):
                            hits[idx].relevance = float(r["relevance_score"])
                    return sorted(hits, key=lambda h: (h.relevance, h.fused_score), reverse=True)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderError("unexpected rerank response shape") from exc
            if attempt < self.attempts - 1:
                self.sleep(0.5 * (2 ** attempt))
        raise ProviderError(f"rerank failed after {self.attempts} attempts ({last})")


class FallbackReranker:
    """Use the hosted reranker; if it fails, fall back to the baseline so answers keep working.

    The fallback's scores are on a different scale, so `last_used` is exposed for logging.
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.last_used = "primary"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        try:
            out = self.primary.rerank(query, hits)
            self.last_used = "primary"
            return out
        except ProviderError:
            self.last_used = "fallback"
            return self.fallback.rerank(query, hits)
