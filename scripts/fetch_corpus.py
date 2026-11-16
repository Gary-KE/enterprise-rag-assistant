"""Download every source in data/sources.json into data/raw/.

Each file's SHA-256 is compared with data/corpus.lock.json. A new source is added to the lock file;
a changed one fails the run (pass --update-lock after checking the change), so the golden set is
never silently evaluated against a different document.

    python scripts/fetch_corpus.py [--update-lock]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RAG_DATA_DIR", ROOT / "data"))
MANIFEST = DATA / "sources.json"
LOCK = DATA / "corpus.lock.json"
RAW = DATA / "raw"
MAX_BYTES = 50 * 1024 * 1024
HEADERS = {"User-Agent": "dondoo-rag-corpus-fetcher/0.1 (+https://github.com/Gary-KE/enterprise-rag-assistant)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-lock", action="store_true")
    args = ap.parse_args()

    sources = json.loads(MANIFEST.read_text())["sources"]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    RAW.mkdir(parents=True, exist_ok=True)
    failed = False

    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as http:
        for src in sources:
            key, url = src["key"], src["url"]
            if not url.startswith("https://"):
                print(f"[{key}] refusing non-HTTPS url")
                failed = True
                continue
            resp = http.get(url)
            if resp.status_code != 200:
                print(f"[{key}] HTTP {resp.status_code} for {url}")
                failed = True
                continue
            data = resp.content
            if len(data) > MAX_BYTES:
                print(f"[{key}] file larger than {MAX_BYTES} bytes")
                failed = True
                continue
            if not data.startswith(b"%PDF-"):
                print(f"[{key}] response is not a PDF (content-type {resp.headers.get('content-type')})")
                failed = True
                continue
            digest = hashlib.sha256(data).hexdigest()
            known = lock.get(key, {}).get("sha256")
            if known and known != digest and not args.update_lock:
                print(f"[{key}] CHANGED since it was locked: {known[:12]} -> {digest[:12]}. "
                      "Check the golden set still matches, then rerun with --update-lock.")
                failed = True
                continue
            (RAW / f"{key}.pdf").write_bytes(data)
            lock[key] = {"sha256": digest, "bytes": len(data), "url": url}
            print(f"[{key}] ok  {len(data):,} bytes  sha256 {digest[:12]}")

    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
