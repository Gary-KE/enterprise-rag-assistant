"""Authentication, rate limiting and HTTP security headers."""
from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from typing import Optional

import jwt

ROLE_LEVELS = {"public": 0, "staff": 1, "admin": 2}


def issue_token(secret: str, role: str, ttl_minutes: int, subject: str = "admin") -> str:
    if role not in ROLE_LEVELS:
        raise ValueError("unknown role")
    now = int(time.time())
    payload = {"sub": subject, "role": role, "iat": now, "exp": now + ttl_minutes * 60}
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(secret: str, token: str) -> Optional[dict]:
    """Return the claims of a valid token, or None. Only HS256 is accepted."""
    if not secret:
        return None
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "iat", "role"]})
    except jwt.PyJWTError:
        return None
    if claims.get("role") not in ROLE_LEVELS:
        return None
    return claims


def keys_match(given: str, expected: str) -> bool:
    """Constant-time comparison, so response timing does not leak the key."""
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


class RateLimiter:
    """Sliding one-minute window per client key. In-memory: one process only."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self._hits) > 10_000:        # bound memory under many distinct clients
                for k in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[k]
            return True


SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# The interactive API docs page loads Swagger UI from a CDN, so it gets its own, narrower exception.
DOCS_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)
