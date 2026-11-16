"""HTTP API and web UI.

Public:  GET /, GET /api/health, GET /api/documents, POST /api/ask
Admin:   POST /api/auth/token (admin key -> token), POST /api/documents (upload)
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.factory import Pipeline, build_pipeline
from rag.ingest.pipeline import ingest_file
from rag.security import DOCS_CSP, ROLE_LEVELS, SECURITY_HEADERS, RateLimiter, decode_token, issue_token, keys_match

log = logging.getLogger("rag.api")
WEB_DIR = Path(__file__).parent / "web"
PDF_MAGIC = b"%PDF-"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    lang: Literal["en", "sw"] = "en"


class TokenRequest(BaseModel):
    admin_key: str = Field(min_length=1, max_length=512)


def _client_key(request: Request, trusted_hops: int) -> str:
    """Client address for rate limiting.

    X-Forwarded-For can be forged by the client, so it is used only when we sit behind a known number
    of proxies (`trusted_hops`). Each trusted proxy appends one entry, so the client is that many from
    the right; anything to its left was supplied by the client and is ignored.
    """
    if trusted_hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            return parts[-trusted_hops]
    return request.client.host if request.client else "unknown"


def create_app(settings: Optional[Settings] = None, pipeline: Optional[Pipeline] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = pipeline or build_pipeline(settings)
    if settings.preload_index and settings.store == "memory":
        from rag.index_io import load_index
        load_index(Path(settings.preload_index), pipeline.store)
    limiter = RateLimiter(settings.rate_limit_per_min)
    app = FastAPI(title="Dondoo: Enterprise RAG Assistant", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.pipeline = pipeline

    @app.middleware("http")
    async def headers_and_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error request_id=%s", request_id)
            response = JSONResponse({"detail": "internal error", "request_id": request_id}, status_code=500)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path == "/api/docs":
            response.headers["Content-Security-Policy"] = DOCS_CSP
        response.headers["X-Request-ID"] = request_id
        log.info(json.dumps({
            "request_id": request_id, "method": request.method, "path": request.url.path,
            "status": response.status_code, "ms": round((time.perf_counter() - start) * 1000, 1),
        }))
        return response

    def access_level(authorization: Optional[str] = Header(default=None)) -> int:
        if not authorization:
            return ROLE_LEVELS["public"]
        scheme, _, token = authorization.partition(" ")
        claims = decode_token(settings.jwt_secret, token.strip()) if scheme.lower() == "bearer" else None
        if claims is None:
            raise HTTPException(status_code=401, detail="invalid or expired token")
        return ROLE_LEVELS[claims["role"]]

    def rate_limited(request: Request) -> None:
        if not limiter.allow(_client_key(request, settings.trusted_proxy_hops)):
            raise HTTPException(status_code=429, detail="too many requests, try again in a minute")

    @app.get("/api/health")
    def health():
        ok = pipeline.store.is_healthy()
        body = {"status": "ok" if ok else "degraded", "store": settings.store, "providers": settings.providers}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/api/documents")
    def documents(level: int = Depends(access_level)):
        return [
            {"doc_id": d.doc_id, "title": d.title, "publisher": d.publisher, "published": d.published,
             "url": d.url, "retrieved": d.retrieved, "chunks": n}
            for d, n in pipeline.store.list_documents(level)
        ]

    @app.post("/api/ask", dependencies=[Depends(rate_limited)])
    def ask(body: AskRequest, level: int = Depends(access_level)):
        result = pipeline.answers.answer(body.question.strip(), access_level=level, lang=body.lang)
        log.info(json.dumps({
            "event": "answer", "declined": result.declined, "reason": result.decline_reason,
            "citations": len(result.citations), "ms": round(result.timings_ms.get("total", 0), 1),
        }))
        return result.to_dict()

    @app.post("/api/auth/token", dependencies=[Depends(rate_limited)])
    def token(body: TokenRequest):
        if not settings.admin_key or not keys_match(body.admin_key, settings.admin_key):
            raise HTTPException(status_code=401, detail="invalid admin key")
        return {"access_token": issue_token(settings.jwt_secret, "admin", settings.jwt_ttl_minutes),
                "token_type": "bearer", "expires_in": settings.jwt_ttl_minutes * 60}

    @app.post("/api/documents", dependencies=[Depends(rate_limited)])
    async def upload(
        file: UploadFile = File(...),
        title: str = Form(..., min_length=2, max_length=300),
        publisher: str = Form("", max_length=200),
        published: str = Form("", max_length=40),
        url: str = Form("", max_length=500),
        doc_access_level: int = Form(0, ge=0, le=2),
        level: int = Depends(access_level),
    ):
        if level < ROLE_LEVELS["admin"]:
            raise HTTPException(status_code=403, detail="admin token required")
        if url and not url.startswith(("https://", "http://")):
            raise HTTPException(status_code=422, detail="url must start with http:// or https://")
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise HTTPException(status_code=415, detail="only PDF, TXT and MD files are accepted")
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"file larger than {settings.max_upload_mb} MB")
        if suffix == ".pdf" and not data.startswith(PDF_MAGIC):
            raise HTTPException(status_code=415, detail="file is not a valid PDF")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            result = ingest_file(
                path,
                {"title": title, "publisher": publisher, "published": published, "url": url,
                 "retrieved": date.today().isoformat(), "file_name": name, "access_level": doc_access_level},
                pipeline.store, pipeline.embedder, settings,
            )
        status = 201 if result.status == "ingested" else 200 if result.status == "skipped_duplicate" else 422
        return JSONResponse(result.to_dict(), status_code=status)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
