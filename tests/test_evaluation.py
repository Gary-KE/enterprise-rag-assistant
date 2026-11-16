import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag.evaluation.golden import contains_quote, load_golden, squash
from rag.evaluation.metrics import ItemResult, check_thresholds, percentile, summarise

ROOT = Path(__file__).resolve().parents[1]


def write_golden(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def row(**kw):
    base = {"id": "A01", "question": "q?", "answerable": True, "reference": "r", "doc": "dpa",
            "section": "s1", "quote": "a quote long enough", "category": "c"}
    base.update(kw)
    return base


def test_squash_ignores_layout_noise():
    assert squash("seventy-two\nhours") == squash("seventy two  hours") == "seventytwohours"
    assert contains_quote("the data subject’s consent", "data subject's consent")


def test_real_golden_set_is_valid():
    items = load_golden(ROOT / "eval" / "golden.jsonl")
    assert len(items) >= 50
    assert sum(i.answerable for i in items) >= 40
    assert sum(not i.answerable for i in items) >= 10
    assert len({squash(i.quote) for i in items if i.answerable}) == sum(i.answerable for i in items)


@pytest.mark.parametrize("bad, msg", [
    ([row(), row()], "duplicate"),
    ([row(quote="")], "need doc"),
    ([row(quote="too short")], "too short"),
    ([row(answerable=False, quote="something here")], "must not have a quote"),
    ([{"id": "x"}], "missing fields"),
])
def test_golden_validation_errors(tmp_path, bad, msg):
    p = tmp_path / "g.jsonl"
    write_golden(p, bad)
    with pytest.raises(ValueError, match=msg):
        load_golden(p)


def result(id_, answerable, declined=False, rank=0, correct=False, ms=100.0, ragas=None):
    return ItemResult(id=id_, question="q", answerable=answerable, category="c", declined=declined,
                      decline_reason="", answer="a", reference="r", retrieved_texts=[], cited_texts=[],
                      cited_pages=[], latency_ms=ms, first_hit_rank=rank, citation_correct=correct,
                      ragas=ragas or {})


def test_summary_maths():
    rs = [
        result("A1", True, rank=1, correct=True, ms=100, ragas={"faithfulness": 1.0}),
        result("A2", True, rank=2, correct=False, ms=200, ragas={"faithfulness": 0.5}),
        result("A3", True, declined=True, rank=0, ms=300, ragas={"faithfulness": None}),
        result("U1", False, declined=True, ms=400),
        result("U2", False, declined=False, ms=500),
    ]
    s = summarise(rs)
    assert s["hit_rate@k"] == round(2 / 3, 4)
    assert s["mrr@k"] == round((1 + 0.5 + 0) / 3, 4)
    assert s["answer_rate"] == round(2 / 3, 4)
    assert s["citation_accuracy"] == 0.5
    assert s["decline_accuracy"] == 0.5
    assert s["latency_p50_ms"] == 300.0
    assert s["ragas_faithfulness"] == 0.75          # the missing score is skipped, not counted as 0


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([10.0], 95) == 10.0
    assert percentile([0, 100], 50) == 50.0


def test_thresholds():
    s = {"hit_rate@k": 0.9, "latency_p95_ms": 900, "ragas_faithfulness": None}
    assert check_thresholds(s, {"hit_rate@k": 0.8, "latency_p95_ms": 1000}) == []
    fails = check_thresholds(s, {"hit_rate@k": 0.95, "latency_p95_ms": 500, "ragas_faithfulness": 0.8})
    assert len(fails) == 3 and any("not measured" in f for f in fails)


def test_ragas_runner_skips_declined_and_survives_judge_errors(monkeypatch):
    from rag.evaluation import ragas_runner

    class Ok:
        async def ascore(self, **kw):
            return type("R", (), {"value": 0.9})()

    class Broken:
        async def ascore(self, **kw):
            raise RuntimeError("judge down")

    monkeypatch.setattr(ragas_runner, "build_metrics", lambda *a: {
        "faithfulness": Ok(), "context_precision": Broken(), "context_recall": Ok()})
    answered = result("A1", True)
    answered.retrieved_texts = ["passage"]
    declined = result("A2", True, declined=True)
    declined.retrieved_texts = ["passage"]
    ragas_runner.score_with_ragas([answered, declined], "u", "k", "m", pause_s=0)
    assert answered.ragas == {"faithfulness": 0.9, "context_precision": None, "context_recall": 0.9}
    assert "faithfulness" not in declined.ragas


def test_real_ragas_metrics_can_be_constructed():
    pytest.importorskip("ragas")
    pytest.importorskip("openai")
    from rag.evaluation.ragas_runner import build_metrics
    m = build_metrics("https://example.invalid/v1", "not-a-real-key", "some-model")
    assert set(m) == {"faithfulness", "context_precision", "context_recall"}


def test_scripts_end_to_end_on_a_generated_pdf(tmp_path):
    """verify_golden.py and run_eval.py --mode offline, on a small PDF made here."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    pages = [
        ["Breach notification. A data controller shall notify the Data Commissioner",
         "without delay, within seventy-two hours of becoming aware of such breach."],
        ["Complaints. A complaint made to the Data Commissioner shall be investigated",
         "and concluded within ninety days."],
    ]
    c = canvas.Canvas(str(data / "raw" / "dpa.pdf"), pagesize=A4)
    for lines in pages:
        for i, line in enumerate(lines):
            c.drawString(50, 780 - 16 * i, line)
        c.showPage()
    c.save()
    (data / "sources.json").write_text(json.dumps({"sources": [{
        "key": "dpa", "title": "Test Act", "publisher": "Test", "published": "2026",
        "url": "https://example.org/dpa.pdf", "access_level": 0}]}))
    golden = tmp_path / "golden.jsonl"
    write_golden(golden, [
        row(id="A01", question="Within how many hours must a data controller notify the Data Commissioner of a breach?",
            quote="within seventy-two hours of becoming aware of such breach"),
        row(id="A02", question="Within how many days must a complaint to the Data Commissioner be investigated and concluded?",
            quote="shall be investigated and concluded within ninety days"),
        row(id="U01", question="What is the capital of France?", answerable=False, doc="", section="", quote="", reference=""),
    ])
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"offline": {"hit_rate@k": 1.0, "citation_accuracy": 1.0, "decline_accuracy": 1.0}}))
    env = {**os.environ, "RAG_DATA_DIR": str(data), "RAG_GOLDEN": str(golden),
           "RAG_RESULTS_DIR": str(tmp_path / "results"), "RAG_THRESHOLDS": str(thresholds)}

    v = subprocess.run([sys.executable, "scripts/verify_golden.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert v.returncode == 0, v.stdout + v.stderr
    assert "quotes found in their document: 2/2" in v.stdout

    e = subprocess.run([sys.executable, "scripts/run_eval.py", "--mode", "offline"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert e.returncode == 0, e.stdout + e.stderr
    assert "GATE PASSED" in e.stdout
    report = (tmp_path / "results" / "offline" / "report.md").read_text()
    assert "Gate: PASSED" in report and "Successful examples" in report


def test_verify_golden_fails_on_a_wrong_quote(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    c = canvas.Canvas(str(data / "raw" / "dpa.pdf"))
    c.drawString(50, 780, "The complaint shall be concluded within ninety days.")
    c.showPage()
    c.save()
    (data / "sources.json").write_text(json.dumps({"sources": [{"key": "dpa", "title": "T", "url": "https://x/y.pdf"}]}))
    golden = tmp_path / "g.jsonl"
    write_golden(golden, [row(quote="concluded within thirty days of filing")])
    env = {**os.environ, "RAG_DATA_DIR": str(data), "RAG_GOLDEN": str(golden)}
    v = subprocess.run([sys.executable, "scripts/verify_golden.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert v.returncode == 1 and "MISSING" in v.stdout
