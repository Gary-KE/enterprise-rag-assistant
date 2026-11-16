"""Shared helpers for the scripts: corpus manifest, building an indexed pipeline."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings  # noqa: E402
from rag.factory import build_pipeline  # noqa: E402
from rag.ingest.pipeline import ingest_file  # noqa: E402

DATA = Path(os.environ.get("RAG_DATA_DIR", ROOT / "data"))
MANIFEST = DATA / "sources.json"
RAW = DATA / "raw"
GOLDEN = Path(os.environ.get("RAG_GOLDEN", ROOT / "eval" / "golden.jsonl"))
RESULTS = Path(os.environ.get("RAG_RESULTS_DIR", ROOT / "eval" / "results"))
THRESHOLDS = Path(os.environ.get("RAG_THRESHOLDS", ROOT / "eval" / "thresholds.json"))


def load_sources() -> list[dict]:
    return json.loads(MANIFEST.read_text())["sources"]


def source_meta(src: dict, retrieved: str | None = None) -> dict:
    return {
        "title": src["title"],
        "publisher": src.get("publisher", ""),
        "published": src.get("published", ""),
        "url": src.get("page_url") or src["url"],
        "retrieved": retrieved or date.today().isoformat(),
        "file_name": f"{src['key']}.pdf",
        "access_level": src.get("access_level", 0),
    }


def build_indexed_pipeline(settings: Settings):
    """Build a pipeline and ingest every downloaded source into its store."""
    pipeline = build_pipeline(settings)
    for src in load_sources():
        path = RAW / f"{src['key']}.pdf"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/fetch_corpus.py first")
        res = ingest_file(path, source_meta(src), pipeline.store, pipeline.embedder, settings)
        print(f"[{src['key']}] {res.status}: {res.chunks_added} chunks, "
              f"{len(res.scanned_pages)} pages without text {res.error}")
        if res.status == "failed":
            raise SystemExit(f"ingestion failed for {src['key']}: {res.error}")
    return pipeline
