"""Turn a file into a list of (page_number, text). Page numbers are what citations point to."""
from __future__ import annotations

import unicodedata
from pathlib import Path

from pypdf import PdfReader

SCAN_MIN_CHARS = 20          # a page with less text than this is treated as a scan with no text layer
SUPPORTED = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    """NFKC turns typographic ligatures (\ufb01 -> "fi") and no-break spaces into plain characters.
    Without it, PDF words like "notification" are split by the ligature and never match a search."""
    return unicodedata.normalize("NFKC", text)


def extract_pages(path: Path) -> tuple[list[tuple[int, str]], list[int]]:
    """Return (pages, scanned_pages). Pages without a usable text layer come back with empty text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        scanned: list[int] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = clean_text(text)
            if len(text.strip()) < SCAN_MIN_CHARS:
                scanned.append(number)
                text = ""
            pages.append((number, text))
        return pages, scanned
    if suffix in {".txt", ".md"}:
        raw = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        # a form feed character starts a new page, so plain-text fixtures can have pages
        return [(i, part) for i, part in enumerate(raw.split("\f"), start=1)], []
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")
