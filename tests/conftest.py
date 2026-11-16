from pathlib import Path

import pytest

from rag.config import Settings
from rag.factory import build_pipeline
from rag.ingest.pipeline import ingest_file

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_META = {
    "title": "Sample Data Protection Act",
    "publisher": "Test Publisher",
    "published": "2019-11-08",
    "url": "https://example.org/sample-act",
    "retrieved": "2026-11-17",
}


@pytest.fixture
def settings():
    s = Settings()
    s.chunk_words = 60
    s.chunk_overlap = 10
    return s


@pytest.fixture
def pipeline(settings):
    p = build_pipeline(settings)
    res = ingest_file(FIXTURES / "sample_act.txt", SAMPLE_META, p.store, p.embedder, settings)
    assert res.status == "ingested", res.error
    return p
