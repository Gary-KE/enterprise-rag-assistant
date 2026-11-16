"""RAGAS scoring: faithfulness, context precision and context recall.

Uses the ragas collections metrics (ragas >= 0.4) with any OpenAI-compatible judge model. Each metric
call costs judge-model requests, so calls run one at a time with a pause, to stay inside free-tier rate
limits, and a failed call is recorded as missing rather than as a zero.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("rag.ragas")


def build_metrics(base_url: str, api_key: str, model: str):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=4, timeout=90)
    llm = llm_factory(model, provider="openai", client=client)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def _score_one(metrics: dict, r, pause_s: float) -> dict:
    scores: dict = {}
    calls = []
    if not r.declined and r.retrieved_texts:
        calls.append(("faithfulness", dict(user_input=r.question, response=r.answer, retrieved_contexts=r.retrieved_texts)))
    if r.retrieved_texts:
        calls.append(("context_precision", dict(user_input=r.question, reference=r.reference, retrieved_contexts=r.retrieved_texts)))
        calls.append(("context_recall", dict(user_input=r.question, retrieved_contexts=r.retrieved_texts, reference=r.reference)))
    for name, kwargs in calls:
        try:
            res = await metrics[name].ascore(**kwargs)
            scores[name] = float(res.value)
        except Exception as exc:  # judge errors must not look like a perfect or a zero score
            log.warning("ragas %s failed for %s: %s", name, r.id, type(exc).__name__)
            scores[name] = None
        await asyncio.sleep(pause_s)
    return scores


def score_with_ragas(results: list, base_url: str, api_key: str, model: str,
                     limit: int | None = None, pause_s: float = 2.0) -> None:
    """Adds r.ragas scores in place for answerable items (up to `limit` of them)."""
    metrics = build_metrics(base_url, api_key, model)
    targets = [r for r in results if r.answerable]
    if limit:
        targets = targets[:limit]

    async def run():
        for r in targets:
            r.ragas = await _score_one(metrics, r, pause_s)

    asyncio.run(run())
