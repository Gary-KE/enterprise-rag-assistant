"""Clients for any provider that offers an OpenAI-compatible HTTP API (chat completions and embeddings).

Retries with backoff on rate limits and server errors. Times out instead of hanging.
Check your provider's documentation for its base URL and model names.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

import httpx

from rag.providers.base import ProviderError

RETRY_STATUS = {429, 500, 502, 503, 504}
# "retry in 7s" (Gemini), "Please try again in 7.66s" / "in 1m30.5s" / "in 450ms" (Groq, OpenAI).
_RETRY_IN = re.compile(r"(?:retry|try again) in ((?:\d+h)?(?:\d+m(?!s))?(?:[0-9]+(?:\.[0-9]+)?(?:ms|s))?)", re.I)
_PART = re.compile(r"([0-9]+(?:\.[0-9]+)?)(h|ms|m|s)")
WAITED = {"seconds": 0.0}   # total time spent deliberately waiting (pacing, rate limits, back-off)
MAX_RATE_WAITS = 3          # per-minute limits are normal on free tiers; wait for them a few times per call
_DAILY = re.compile(r"per ?day|perday|daily", re.I)
_TOKENISH = re.compile(r"[A-Za-z0-9_.-]{32,}")


def retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    """Seconds the provider asks us to wait, from its message or the Retry-After header; None if unknown."""
    m = _RETRY_IN.search(resp.text)
    if m and m.group(1):
        units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
        total = sum(float(n) * units[u] for n, u in _PART.findall(m.group(1)))
        if total > 0:
            return total
    header = resp.headers.get("retry-after", "")
    try:
        return float(header) if header else None
    except ValueError:
        return None


def _reason(resp: httpx.Response) -> str:
    """The provider's own error message, shortened, with any long token-like string masked."""
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    except ValueError:
        msg = resp.text
    return _TOKENISH.sub("[redacted]", " ".join(str(msg).split()))[:240]


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
        min_interval_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.attempts = attempts
        self.sleep = sleep
        self.min_interval_s = min_interval_s   # pacing for per-minute limits (free Gemini: ~5 per minute)
        self.clock = clock
        self._last_call: Optional[float] = None
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def _wait(self, seconds: float) -> None:
        WAITED["seconds"] += seconds
        self.sleep(seconds)

    def _pace(self) -> None:
        if self.min_interval_s > 0 and self._last_call is not None:
            wait = self.min_interval_s - (self.clock() - self._last_call)
            if wait > 0:
                self._wait(wait)
        self._last_call = self.clock()

    def _post(self, path: str, payload: dict) -> dict:
        last_error = "unknown error"
        rate_waits = 0
        for attempt in range(self.attempts):
            try:
                self._pace()
                resp = self.http.post(path, json=payload)
                if resp.status_code == 429:
                    reason = _reason(resp)
                    wait = retry_after_seconds(resp)
                    if _DAILY.search(resp.text):
                        # A daily limit will not clear by waiting: give up on this model for now.
                        raise QuotaExhausted(f"{path} failed with HTTP 429 (daily limit reached): {reason}")
                    if wait is not None and rate_waits < MAX_RATE_WAITS:
                        # A per-minute limit (requests or tokens) says how long to wait: wait and try again.
                        rate_waits += 1
                        self._wait(min(wait + 1.0, 70.0))
                        continue
                    if "quota" in reason.lower() or wait is not None:
                        raise QuotaExhausted(f"{path} failed with HTTP 429 (quota exhausted): {reason}")
                if resp.status_code in RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}: {_reason(resp)}"
                elif resp.status_code >= 400:
                    raise ProviderError(f"{path} failed with HTTP {resp.status_code}: {_reason(resp)}")
                else:
                    return resp.json()
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}"
            if attempt < self.attempts - 1:
                self._wait(0.5 * (2 ** attempt))
        raise ProviderError(f"{path} failed after {self.attempts} attempts ({last_error})")


class QuotaExhausted(ProviderError):
    """The provider's quota for this model is used up; another model may still work."""


class OpenAICompatibleLLM(_Client):
    def __init__(self, *args, reasoning_effort: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort   # e.g. "low" for reasoning models such as gpt-oss on Groq

    def complete(self, system: str, user: str) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        data = self._post("/chat/completions", payload)
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


class FallbackLLM:
    """Try the primary model; when its quota is exhausted, move to the next model and stay there.

    Only QuotaExhausted triggers a switch: other errors (bad request, outage) are raised as they are.
    """

    def __init__(self, models: list):
        if not models:
            raise ValueError("FallbackLLM needs at least one model")
        self.models = models
        self.current = 0

    @property
    def model(self) -> str:
        return self.models[self.current].model

    def complete(self, system: str, user: str) -> str:
        while True:
            try:
                return self.models[self.current].complete(system, user)
            except QuotaExhausted:
                if self.current + 1 >= len(self.models):
                    raise
                self.current += 1
