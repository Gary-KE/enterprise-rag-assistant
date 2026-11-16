"""Our own evaluation metrics. These need no judge model, so they are exact and cheap to run.

For each golden question we record what the system did, then score:
  hit_rate@k        answerable questions where a retrieved passage contains the verified quote
  mrr@k             1 / rank of the first retrieved passage containing the quote (0 if none)
  answer_rate       answerable questions that were answered rather than declined
  citation_accuracy answered questions where at least one cited passage contains the quote
  decline_accuracy  unanswerable questions that were correctly declined
  latency p50/p95   end-to-end time per question
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from rag.evaluation.golden import GoldenItem, contains_quote


@dataclass
class ItemResult:
    id: str
    question: str
    answerable: bool
    category: str
    declined: bool
    decline_reason: str
    answer: str
    reference: str
    retrieved_texts: list[str]
    cited_texts: list[str]
    cited_pages: list[int]
    latency_ms: float
    first_hit_rank: int = 0          # 1-based rank of the first passage containing the quote; 0 = none
    citation_correct: bool = False
    ragas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def score_item(item: GoldenItem, answer, store) -> ItemResult:
    retrieved = [store.get_chunk(cid) for cid in answer.retrieved_chunk_ids]
    retrieved_texts = [c.text for c in retrieved if c is not None]
    cited_texts = []
    for c in answer.citations:
        chunk = store.get_chunk(c.chunk_id)
        cited_texts.append(chunk.text if chunk else c.quote)
    r = ItemResult(
        id=item.id, question=item.question, answerable=item.answerable, category=item.category,
        declined=answer.declined, decline_reason=answer.decline_reason, answer=answer.answer,
        reference=item.reference, retrieved_texts=retrieved_texts, cited_texts=cited_texts,
        cited_pages=[c.page for c in answer.citations],
        latency_ms=float(answer.timings_ms.get("total", 0.0)),
    )
    if item.answerable:
        for rank, text in enumerate(retrieved_texts, start=1):
            if contains_quote(text, item.quote):
                r.first_hit_rank = rank
                break
        r.citation_correct = (not answer.declined) and any(contains_quote(t, item.quote) for t in cited_texts)
    return r


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(results: list[ItemResult]) -> dict:
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    answered = [r for r in ans if not r.declined]
    lat = [r.latency_ms for r in results]

    def ratio(n, d):
        return round(n / d, 4) if d else None

    out = {
        "questions": len(results),
        "answerable": len(ans),
        "unanswerable": len(unans),
        "hit_rate@k": ratio(sum(r.first_hit_rank > 0 for r in ans), len(ans)),
        "mrr@k": round(statistics.fmean([1 / r.first_hit_rank if r.first_hit_rank else 0 for r in ans]), 4) if ans else None,
        "answer_rate": ratio(len(answered), len(ans)),
        "citation_accuracy": ratio(sum(r.citation_correct for r in answered), len(answered)),
        # A decline caused by a provider failure is not a correct decline.
        "decline_accuracy": ratio(sum(r.declined and r.decline_reason != "llm_unavailable" for r in unans), len(unans)),
        "provider_errors": sum(r.decline_reason == "llm_unavailable" for r in results),
        "latency_p50_ms": round(percentile(lat, 50), 1),
        "latency_p95_ms": round(percentile(lat, 95), 1),
    }
    ragas_keys = sorted({k for r in results for k in r.ragas})
    for k in ragas_keys:
        vals = [r.ragas[k] for r in results if k in r.ragas and r.ragas[k] is not None]
        out[f"ragas_{k}"] = round(statistics.fmean(vals), 4) if vals else None
    return out


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Return a list of failures. Keys ending in _ms, and provider_errors, are maximums; all others are minimums."""
    failures = []
    for key, limit in thresholds.items():
        value = summary.get(key)
        is_max = key.endswith("_ms") or key == "provider_errors"
        if value is None:
            failures.append(f"{key}: not measured (threshold {limit})")
        elif is_max and value > limit:
            failures.append(f"{key}: {value} is above the maximum {limit}")
        elif not is_max and value < limit:
            failures.append(f"{key}: {value} is below the minimum {limit}")
    return failures
