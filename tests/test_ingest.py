from pathlib import Path

import pytest

from rag.ingest.chunking import chunk_pages, text_hash
from rag.ingest.extract import extract_pages
from rag.ingest.pipeline import ingest_file
from tests.conftest import FIXTURES, SAMPLE_META


def test_chunks_overlap_and_stay_on_their_page():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_pages([(3, words)], "doc", chunk_words=10, overlap=3)
    assert [c.page for c in chunks] == [3, 3, 3, 3]          # starts 0, 7, 14 and a final window at 15
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w24"          # nothing lost at the end
    covered = set(w for c in chunks for w in c.text.split())
    assert covered == set(words.split())
    assert all(len(c.text.split()) == 10 for c in chunks)


def test_every_word_is_covered_for_many_lengths():
    for n in range(1, 80):
        words = [f"w{i}" for i in range(n)]
        chunks = chunk_pages([(1, " ".join(words))], "d", chunk_words=10, overlap=3, min_words=1)
        assert set(w for c in chunks for w in c.text.split()) == set(words), n
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_no_tail_chunk_that_is_only_overlap():
    words = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_pages([(1, words)], "doc", chunk_words=10, overlap=3)
    assert len(chunks) == 1


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([(1, "a b c")], "doc", chunk_words=5, overlap=5)


def test_hash_ignores_case_and_spacing():
    assert text_hash("Hello   World") == text_hash("hello world")


def test_text_pages_split_on_form_feed():
    pages, scanned = extract_pages(FIXTURES / "sample_act.txt")
    assert len(pages) == 4 and scanned == []
    assert "seventy-two hours" in pages[2][1]


def test_pdf_pages_and_scanned_page_detection(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    pdf = tmp_path / "t.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Page one says the reporting deadline is seventy-two hours.")
    c.showPage()
    c.showPage()                                          # blank page, like a scan without text
    c.drawString(72, 720, "Page three covers the rights of a data subject.")
    c.showPage()
    c.save()
    pages, scanned = extract_pages(pdf)
    assert [p for p, _ in pages] == [1, 2, 3]
    assert scanned == [2]
    assert "seventy-two" in pages[0][1]


def test_unsupported_type_fails_cleanly(tmp_path, pipeline, settings):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    res = ingest_file(bad, {}, pipeline.store, pipeline.embedder, settings)
    assert res.status == "failed" and "Unsupported" in res.error


def test_reingesting_same_file_is_skipped(pipeline, settings):
    res = ingest_file(FIXTURES / "sample_act.txt", SAMPLE_META, pipeline.store, pipeline.embedder, settings)
    assert res.status == "skipped_duplicate"


def test_duplicate_passages_in_another_file_are_not_stored_twice(tmp_path, pipeline, settings):
    copy = tmp_path / "copy.txt"
    copy.write_text((FIXTURES / "sample_act.txt").read_text() + "\f One extra page with new words here today.")
    res = ingest_file(copy, {"title": "Copy"}, pipeline.store, pipeline.embedder, settings)
    assert res.status == "ingested"
    assert res.duplicate_chunks_skipped > 0
    assert res.chunks_added == 1


def test_empty_document_reports_why(tmp_path, pipeline, settings):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    res = ingest_file(empty, {}, pipeline.store, pipeline.embedder, settings)
    assert res.status == "failed" and "no extractable text" in res.error


def test_ligatures_are_normalised_for_search():
    from rag.ingest.extract import clean_text
    from rag.text_utils import content_tokens
    assert clean_text("Noti\ufb01cation of the O\ufb03ce\u00a0now") == "Notification of the Office now"
    assert "notification" in content_tokens("Noti\ufb01cation")
    assert content_tokens("con\ufb01dential") == content_tokens("confidential")


def test_real_corpus_has_no_ligatures_after_extraction():
    from pathlib import Path
    pdf = Path(__file__).resolve().parents[1] / "data" / "corpus" / "dpa.pdf"
    if not pdf.exists():
        pytest.skip("corpus copy not present")
    pages, scanned = extract_pages(pdf)
    text = " ".join(t for _, t in pages)
    assert scanned == [] and len(pages) == 32
    assert not any("\ufb00" <= ch <= "\ufb06" for ch in text)
    assert "notification" in text.lower() and "Data Commissioner" in text


def test_running_headers_and_page_numbers_are_removed():
    from rag.ingest.extract import strip_running_lines
    pages = [(i, f"Act (Cap. 1) Kenya\nBody text {i} here.\n{i}") for i in range(1, 6)]
    out = strip_running_lines(pages)
    assert all(t == f"Body text {n} here." for n, t in out)


def test_real_corpus_headers_removed():
    from pathlib import Path
    pdf = Path(__file__).resolve().parents[1] / "data" / "corpus" / "dpa.pdf"
    if not pdf.exists():
        pytest.skip("corpus copy not present")
    pages, _ = extract_pages(pdf)
    assert sum("Data Protection Act (Cap. 411C)  Kenya" in t for _, t in pages[5:]) == 0
