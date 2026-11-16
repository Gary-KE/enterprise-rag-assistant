"""Clients for any provider that offers an OpenAI-compatible HTTP API (chat completions and embeddings).

Retries with backoff on rate limits and server errors. Times out instead of hanging.
Check your provider's documentation for its base URL and model names.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from rag.providers.base import ProviderError

RETRY_STATUS = {429, 500, 502, 503, 504}


class _Client:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        attempts: int = 4,
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

    def _post(self, path: str, payload: dict) -> dict:
        last_error = "unknown error"
        for attempt in range(self.attempts):
            try:
                resp = self.http.post(path, json=payload)
                if resp.status_code in RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise ProviderError(f"{path} failed with HTTP {resp.status_code}")
                else:
                    return resp.json()
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}"
            if attempt < self.attempts - 1:
                self.sleep(0.5 * (2 ** attempt))
        raise ProviderError(f"{path} failed after {self.attempts} attempts ({last_error})")


class OpenAICompatibleLLM(_Client):
    def complete(self, system: str, user: str) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected chat completion response shape") from exc


class OpenAICompatibleEmbedder(_Client):
    def __init__(self, *args, dimensions: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = self._post("/embeddings", payload)
        try:
            rows = data["data"]
            # Some providers (e.g. Gemini's compatibility layer) omit "index"; then list order is the order.
            if all("index" in r for r in rows):
                rows = sorted(rows, key=lambda r: r["index"])
            vectors = [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            shape = {k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else type(data).__name__
            first = data.get("data", [None])[0] if isinstance(data, dict) and data.get("data") else None
            first_keys = sorted(first.keys()) if isinstance(first, dict) else None
            raise ProviderError(f"unexpected embeddings response shape: top-level {shape}, first item keys {first_keys}") from exc
        if len(vectors) != len(texts):
            raise ProviderError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        if self.dimensions and any(len(v) != self.dimensions for v in vectors):
            raise ProviderError(
                f"embedding size {len(vectors[0])} does not match RAG_EMBED_DIM={self.dimensions}; "
                "set RAG_EMBED_DIM to the model's size (and the database column to match)")
        return vectors
