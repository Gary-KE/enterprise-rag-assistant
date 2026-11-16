"""Run the golden set through the full pipeline, score it, write a report, and enforce the gate.

    python scripts/run_eval.py --mode offline           # stand-in models, no API keys (CI smoke test)
    python scripts/run_eval.py --mode live --ragas      # real models from RAG_* env vars, plus RAGAS

Exit code 1 if any threshold in eval/thresholds.json for the chosen mode is missed.
Writes eval/results/<mode>/summary.json, results.jsonl and report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from common import GOLDEN, RESULTS, THRESHOLDS, build_indexed_pipeline

from rag.config import Settings
from rag.evaluation.golden import load_golden
from rag.evaluation.metrics import check_thresholds, score_item, summarise


def write_report(path, mode, summary, failures, results, settings):
    lines = [
        f"# Evaluation report ({mode})",
        "",
        f"Run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Models: {'stand-in (offline)' if mode == 'offline' else settings.llm_model + ' / ' + settings.embed_model}. "
        f"Reranker: {settings.rerank_model or 'baseline'}.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"**Gate: {'PASSED' if not failures else 'FAILED'}**", ""]
    lines += [f"- {f}" for f in failures]

    def example(r):
        cite = f" (cited page {', '.join(map(str, r.cited_pages))})" if r.cited_pages else ""
        return f"- **{r.id}** {r.question}\n  - Answer: {r.answer[:300]}{cite}"

    good = [r for r in results if (r.answerable and r.citation_correct) or (not r.answerable and r.declined)]
    bad = [r for r in results if (r.answerable and not r.citation_correct) or (not r.answerable and not r.declined)]
    lines += ["", "## Successful examples", ""] + [example(r) for r in good[:6]]
    lines += ["", f"## Failed examples ({len(bad)})", ""]
    for r in bad:
        why = ("answered an out-of-scope question" if not r.answerable else
               f"declined ({r.decline_reason})" if r.declined else
               "retrieved the right passage but cited another" if r.first_hit_rank else
               "the right passage was not retrieved")
        lines.append(example(r) + f"\n  - Why it failed: {why}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "live"], required=True)
    ap.add_argument("--ragas", action="store_true", help="also score with RAGAS (live mode only)")
    ap.add_argument("--ragas-limit", type=int, default=None, help="score only the first N answerable items")
    ap.add_argument("--no-gate", action="store_true", help="report only; never fail")
    args = ap.parse_args()

    settings = Settings.from_env()
    if args.mode == "offline":
        settings.providers = "fake"
        settings.rerank_base_url = settings.rerank_api_key = settings.rerank_model = ""
    elif settings.providers != "http":
        raise SystemExit("live mode needs RAG_PROVIDERS=http and the RAG_LLM_* / RAG_EMBED_* variables")
    settings.store = "memory"

    pipeline = build_indexed_pipeline(settings)
    items = load_golden(GOLDEN)
    results = []
    for n, item in enumerate(items, start=1):
        ans = pipeline.answers.answer(item.question)
        r = score_item(item, ans, pipeline.store)
        results.append(r)
        mark = "ok " if ((item.answerable and r.citation_correct) or (not item.answerable and r.declined)) else "BAD"
        print(f"{mark} {n:>3}/{len(items)} {item.id} rank={r.first_hit_rank} declined={r.declined} {r.latency_ms:.0f}ms")

    if args.ragas:
        if args.mode != "live":
            raise SystemExit("--ragas needs --mode live (it uses a judge model)")
        from rag.evaluation.ragas_runner import score_with_ragas
        judge_model = os.environ.get("RAG_JUDGE_MODEL") or settings.llm_model
        score_with_ragas(results, settings.llm_base_url, settings.llm_api_key, judge_model, limit=args.ragas_limit)

    summary = summarise(results)
    thresholds = json.loads(THRESHOLDS.read_text())[args.mode]
    if not args.ragas:
        thresholds = {k: v for k, v in thresholds.items() if not k.startswith("ragas_")}
    failures = check_thresholds(summary, thresholds)

    out = RESULTS / args.mode
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "thresholds": thresholds,
                                                  "failures": failures}, indent=2) + "\n")
    with (out / "results.jsonl").open("w") as fh:
        for r in results:
            row = r.to_dict()
            row.pop("retrieved_texts")      # keep the artefact small; the passages are in the corpus
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(out / "report.md", args.mode, summary, failures, results, settings)

    print(json.dumps(summary, indent=2))
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        return 0 if args.no_gate else 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
