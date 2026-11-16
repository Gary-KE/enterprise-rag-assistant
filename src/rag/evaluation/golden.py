"""Golden test set: loading, validation and the quote matching used by the metrics."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REQUIRED = {"id", "question", "answerable", "reference", "doc", "section", "quote", "category"}


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    answerable: bool
    reference: str
    doc: str
    section: str
    quote: str
    category: str


def squash(text: str) -> str:
    """Lower-case letters and digits only. Makes quote matching immune to spacing, line breaks,
    hyphenation across lines, curly versus straight quotes and punctuation."""
    text = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def contains_quote(text: str, quote: str) -> bool:
    q = squash(quote)
    return bool(q) and q in squash(text)


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"line {lineno}: missing fields {sorted(missing)}")
            item = GoldenItem(**{k: row[k] for k in REQUIRED})
            if item.id in seen:
                raise ValueError(f"line {lineno}: duplicate id {item.id}")
            seen.add(item.id)
            if item.answerable and (not item.quote.strip() or not item.reference.strip() or not item.doc):
                raise ValueError(f"{item.id}: answerable items need doc, quote and reference")
            if item.answerable and len(squash(item.quote)) < 12:
                raise ValueError(f"{item.id}: quote too short to identify a passage")
            if not item.answerable and item.quote:
                raise ValueError(f"{item.id}: unanswerable items must not have a quote")
            items.append(item)
    if not items:
        raise ValueError("golden set is empty")
    return items
