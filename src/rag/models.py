"""Plain data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SourceMeta:
    """Where a document came from. Shown to the user next to every citation."""
    doc_id: str
    title: str
    publisher: str = ""
    published: str = ""        # publication date as given by the publisher
    url: str = ""
    retrieved: str = ""        # the date we collected the document
    file_name: str = ""
    access_level: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    index: int
    text: str
    text_hash: str
    access_level: int = 0
    embedding: Optional[list[float]] = None


@dataclass
class Hit:
    chunk: Chunk
    fused_score: float = 0.0
    rank_keyword: Optional[int] = None
    rank_vector: Optional[int] = None
    relevance: float = 0.0     # reranker score, 0 to 1


@dataclass
class Citation:
    marker: str
    chunk_id: str
    doc_id: str
    title: str
    publisher: str
    published: str
    page: int
    url: str
    retrieved: str
    quote: str


@dataclass
class Answer:
    question: str
    answer: str
    declined: bool
    decline_reason: str = ""
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
