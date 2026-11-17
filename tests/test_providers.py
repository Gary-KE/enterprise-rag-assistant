import json

import httpx
import pytest

from rag.config import Settings
from rag.index_io import load_index, save_index
from rag.models import Chunk, Hit
from rag.providers.base import ProviderError
from rag.providers.fake import OverlapReranker
from rag.providers.http import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from rag.providers.rerank_http import FallbackReranker, HttpReranker
from rag.store.memory import MemoryStore


def transport(responses):
    """Serve the given (status, body) pairs in order and record the requests."""
    seen = []

    def handler(request):
        seen.append(request)
        status, body = responses[min(len(seen) - 1, len(responses) - 1)]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), seen


def no_sleep(_):
    pass


def test_llm_sends_openai_shape_and_parses_reply():
    t, seen = transport([(200, {"choices": [{"message": {"content": "Answer [S1]."}}]})])
    llm = OpenAICompatibleLLM("https://x.test/v1/", "key", "m", transport=t, sleep=no_sleep)
    assert llm.complete("sys", "user") == "Answer [S1]."
    req = seen[0]
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer key"
    body = json.loads(req.content)
    assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"


def test_llm_retries_rate_limits_then_succeeds():
    t, seen = transport([(429, {}), (503, {}), (200, {"choices": [{"message": {"content": "ok"}}]})])
    llm = OpenAICompatibleLLM("https://x.test/v1", "k", "m", transport=t, sleep=no_sleep)
    assert llm.complete("s", "u") == "ok" and len(seen) == 3


def test_llm_gives_up_after_attempts_and_does_not_retry_client_errors():
    t, seen = transport([(500, {})])
    with pytest.raises(ProviderError):
        OpenAICompatibleLLM("https://x.test", "k", "m", attempts=3, transport=t, sleep=no_sleep).complete("s", "u")
    assert len(seen) == 3
    t2, seen2 = transport([(401, {"error": "bad key"})])
    with pytest.raises(ProviderError):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t2, sleep=no_sleep).complete("s", "u")
    assert len(seen2) == 1


def test_error_messages_never_contain_the_api_key():
    t, _ = transport([(401, {})])
    with pytest.raises(ProviderError) as err:
        OpenAICompatibleLLM("https://x.test", "SECRET-KEY-123", "m", transport=t, sleep=no_sleep).complete("s", "u")
    assert "SECRET-KEY-123" not in str(err.value)


def test_embedder_orders_by_index_and_sends_dimensions():
    t, seen = transport([(200, {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})])
    emb = OpenAICompatibleEmbedder("https://x.test", "k", "e", dimensions=2, transport=t, sleep=no_sleep)
    assert emb.embed(["a", "b"]) == [[1, 0], [0, 1]]
    assert json.loads(seen[0].content)["dimensions"] == 2
    assert emb.embed([]) == []


def test_embedder_rejects_wrong_size_vectors():
    t, _ = transport([(200, {"data": [{"index": 0, "embedding": [0.1] * 3072}]})])
    emb = OpenAICompatibleEmbedder("https://x.test", "k", "e", dimensions=768, transport=t, sleep=no_sleep)
    with pytest.raises(ProviderError, match="does not match RAG_EMBED_DIM"):
        emb.embed(["a"])


def hits(*texts):
    return [Hit(chunk=Chunk(f"c{i}", "d", 1, i, t, "h"), fused_score=1 / (i + 1)) for i, t in enumerate(texts)]


def test_http_reranker_applies_scores():
    t, seen = transport([(200, {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]})])
    r = HttpReranker("https://r.test/v1", "k", "jina-reranker", transport=t, sleep=no_sleep)
    out = r.rerank("q", hits("first", "second"))
    assert [h.chunk.text for h in out] == ["second", "first"]
    assert out[0].relevance == 0.9
    body = json.loads(seen[0].content)
    assert body["documents"] == ["first", "second"] and body["top_n"] == 2


def test_fallback_reranker_used_when_hosted_one_fails():
    t, _ = transport([(500, {})])
    fb = FallbackReranker(HttpReranker("https://r.test", "k", "m", attempts=1, transport=t, sleep=no_sleep), OverlapReranker())
    out = fb.rerank("breach hours", hits("nothing here", "breach notice within hours"))
    assert fb.last_used == "fallback" and out[0].chunk.text == "breach notice within hours"


def test_config_validation():
    s = Settings(providers="http")
    with pytest.raises(ValueError):
        s.validate()
    s = Settings(chunk_words=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        s.validate()
    s = Settings(admin_key="short")
    with pytest.raises(ValueError):
        s.validate()
    s = Settings(admin_key="k" * 30, jwt_secret="short")
    with pytest.raises(ValueError):
        s.validate()
    s = Settings(rerank_base_url="https://r.test")
    with pytest.raises(ValueError):
        s.validate()
    Settings().validate()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RAG_MIN_RELEVANCE", "0.25")
    monkeypatch.setenv("RAG_STRICT_CITATIONS", "true")
    monkeypatch.setenv("RAG_TRUSTED_PROXY_HOPS", "1")
    s = Settings.from_env()
    assert s.min_relevance == 0.25 and s.strict_sentence_citations is True and s.trusted_proxy_hops == 1


def test_index_round_trip(pipeline, tmp_path):
    store = pipeline.store
    docs = [d for d, _ in store.list_documents(9)]
    chunks = [store.get_chunk(cid) for cid in store._chunks]
    path = tmp_path / "index.jsonl"
    save_index(path, docs, chunks)
    fresh = MemoryStore()
    assert load_index(path, fresh) == (len(docs), len(chunks))
    assert fresh.keyword_search("seventy-two hours breach", 3, 0)[0][0].page == 3


def test_embedder_accepts_items_without_index():
    t, _ = transport([(200, {"object": "list", "data": [{"object": "embedding", "embedding": [1, 0]},
                                                         {"object": "embedding", "embedding": [0, 1]}]})])
    emb = OpenAICompatibleEmbedder("https://x.test", "k", "e", dimensions=2, transport=t, sleep=no_sleep)
    assert emb.embed(["a", "b"]) == [[1, 0], [0, 1]]


def test_embedder_error_names_the_shape_but_not_the_data():
    t, _ = transport([(200, {"embeddings": [{"values": [0.123456]}]})])
    emb = OpenAICompatibleEmbedder("https://x.test", "SECRET", "e", transport=t, sleep=no_sleep)
    with pytest.raises(ProviderError) as err:
        emb.embed(["a"])
    assert "embeddings" in str(err.value) and "0.123456" not in str(err.value) and "SECRET" not in str(err.value)


def test_error_includes_provider_reason_but_masks_tokens():
    t, _ = transport([(404, {"error": {"message": "models/x is not found for API version v1beta"}})])
    with pytest.raises(ProviderError, match="is not found"):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep).complete("s", "u")
    token = "FAKExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # made-up placeholder
    t2, _ = transport([(400, {"error": {"message": "bad credential " + token}})])
    with pytest.raises(ProviderError) as err:
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t2, sleep=no_sleep).complete("s", "u")
    assert token not in str(err.value) and "[redacted]" in str(err.value)


def test_quota_exhausted_is_not_retried():
    t, seen = transport([(429, {"error": {"message": "You exceeded your current quota, please check your plan"}})])
    with pytest.raises(ProviderError, match="quota exhausted"):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep).complete("s", "u")
    assert len(seen) == 1


def test_plain_rate_limit_is_still_retried():
    t, seen = transport([(429, {"error": {"message": "Too many requests"}}), (200, {"choices": [{"message": {"content": "ok"}}]})])
    assert OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep).complete("s", "u") == "ok"
    assert len(seen) == 2


def test_per_minute_quota_waits_once_then_retries():
    slept = []
    t, seen = transport([
        (429, {"error": {"message": "You exceeded your current quota. Please retry in 12.5s."}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=slept.append)
    assert llm.complete("s", "u") == "ok" and len(seen) == 2 and slept == [13.5]


def test_daily_quota_is_not_waited_for():
    from rag.providers.http import QuotaExhausted
    t, seen = transport([(429, {"error": {"message": "You exceeded your current quota: GenerateRequestsPerDay. Please retry in 9s."}})])
    with pytest.raises(QuotaExhausted):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep).complete("s", "u")
    assert len(seen) == 1


def test_pacing_spaces_out_calls():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(round(s, 2))
        now[0] += s

    t, _ = transport([(200, {"choices": [{"message": {"content": "ok"}}]})])
    llm = OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=sleep,
                              min_interval_s=13, clock=lambda: now[0])
    llm.complete("s", "u")
    now[0] += 3
    llm.complete("s", "u")
    assert slept == [10.0]


def test_fallback_switches_on_quota_and_stays():
    from rag.providers.http import FallbackLLM
    quota = {"error": {"message": "You exceeded your current quota (per day)"}}
    t1, seen1 = transport([(429, quota)])
    t2, seen2 = transport([(200, {"choices": [{"message": {"content": "from lite"}}]})])
    fb = FallbackLLM([OpenAICompatibleLLM("https://x.test", "k", "flash", transport=t1, sleep=no_sleep),
                      OpenAICompatibleLLM("https://x.test", "k", "lite", transport=t2, sleep=no_sleep)])
    assert fb.complete("s", "u") == "from lite" and fb.model == "lite"
    assert fb.complete("s", "u") == "from lite"
    assert len(seen1) == 1 and len(seen2) == 2


def test_fallback_does_not_hide_other_errors():
    from rag.providers.http import FallbackLLM
    t1, _ = transport([(400, {"error": {"message": "bad request"}})])
    t2, seen2 = transport([(200, {"choices": [{"message": {"content": "x"}}]})])
    fb = FallbackLLM([OpenAICompatibleLLM("https://x.test", "k", "a", transport=t1, sleep=no_sleep),
                      OpenAICompatibleLLM("https://x.test", "k", "b", transport=t2, sleep=no_sleep)])
    with pytest.raises(ProviderError, match="bad request"):
        fb.complete("s", "u")
    assert len(seen2) == 0


def test_factory_builds_fallback_chain():
    from rag.factory import build_pipeline
    from rag.providers.http import FallbackLLM
    s = Settings(providers="http", llm_base_url="https://x.test", llm_api_key="k", llm_model="flash",
                 llm_fallback_models="lite, other ", embed_base_url="https://x.test", embed_api_key="k",
                 embed_model="e")
    p = build_pipeline(s)
    assert isinstance(p.llm, FallbackLLM) and [m.model for m in p.llm.models] == ["flash", "lite", "other"]


# Groq (and OpenAI) phrase per-minute limits as "Please try again in 7.66s", with no word "quota".
GROQ_TPM = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
            "on tokens per minute (TPM): Limit 8000, Used 6100, Requested 2500. Please try again in 4.5s.")
GROQ_RPD = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
            "on requests per day (RPD): Limit 1000, Used 1000, Requested 1. Please try again in 1m26.4s.")


def test_groq_per_minute_limit_waits_and_retries():
    slept = []
    t, seen = transport([(429, {"error": {"message": GROQ_TPM}}), (429, {"error": {"message": GROQ_TPM}}),
                         (200, {"choices": [{"message": {"content": "ok"}}]})])
    llm = OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=slept.append)
    assert llm.complete("s", "u") == "ok" and len(seen) == 3 and slept == [5.5, 5.5]


def test_groq_daily_limit_is_exhausted_not_waited_for():
    from rag.providers.http import QuotaExhausted
    t, seen = transport([(429, {"error": {"message": GROQ_RPD}})])
    with pytest.raises(QuotaExhausted, match="daily limit"):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep).complete("s", "u")
    assert len(seen) == 1


def test_endless_per_minute_limit_stops_after_a_few_waits():
    from rag.providers.http import MAX_RATE_WAITS, QuotaExhausted
    slept = []
    t, seen = transport([(429, {"error": {"message": GROQ_TPM}})])
    with pytest.raises(QuotaExhausted):
        OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=slept.append).complete("s", "u")
    assert len(slept) == MAX_RATE_WAITS and len(seen) == MAX_RATE_WAITS + 1


@pytest.mark.parametrize("text,expected", [
    ("Please try again in 7.66s.", 7.66),
    ("Please try again in 1m26.4s.", 86.4),
    ("Please try again in 450ms.", 0.45),
    ("Please retry in 12s", 12.0),
])
def test_retry_after_parsing(text, expected):
    from rag.providers.http import retry_after_seconds
    assert retry_after_seconds(httpx.Response(429, json={"error": {"message": text}})) == pytest.approx(expected)


def test_retry_after_header_is_used_when_message_has_none():
    from rag.providers.http import retry_after_seconds
    resp = httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "3"})
    assert retry_after_seconds(resp) == 3.0


def test_reasoning_effort_is_sent_only_when_set():
    ok = (200, {"choices": [{"message": {"content": "ok"}}]})
    t, seen = transport([ok])
    OpenAICompatibleLLM("https://x.test", "k", "m", transport=t, sleep=no_sleep, reasoning_effort="low").complete("s", "u")
    assert json.loads(seen[0].content)["reasoning_effort"] == "low"
    t2, seen2 = transport([ok])
    OpenAICompatibleLLM("https://x.test", "k", "m", transport=t2, sleep=no_sleep).complete("s", "u")
    assert "reasoning_effort" not in json.loads(seen2[0].content)



def test_fallback_returns_to_the_primary_model_after_cooldown():
    from rag.providers.http import FallbackLLM, QuotaExhausted

    class M:
        def __init__(self, name, fail):
            self.model, self.fail, self.calls = name, fail, 0
        def complete(self, s, u):
            self.calls += 1
            if self.fail[0]:
                raise QuotaExhausted("x")
            return self.model

    now = [0.0]
    primary_fails = [True]
    a, b = M("a", primary_fails), M("b", [False])
    llm = FallbackLLM([a, b], clock=lambda: now[0])
    assert llm.complete("s", "u") == "b"
    primary_fails[0] = False
    now[0] = 100
    assert llm.complete("s", "u") == "b"          # still within the cooldown
    now[0] = 700
    assert llm.complete("s", "u") == "a"          # back on the primary


def test_backup_provider_is_appended_with_the_embedding_key():
    from rag.config import Settings
    from rag.factory import build_pipeline
    s = Settings(providers="http", llm_base_url="https://groq.test", llm_api_key="g", llm_model="m1",
                 embed_base_url="https://gem.test", embed_api_key="e", embed_model="emb",
                 llm_backup_base_url="https://gem.test", llm_backup_model="flash-lite")
    p = build_pipeline(s)
    assert [m.model for m in p.llm.models] == ["m1", "flash-lite"]
    assert p.llm.models[-1].http.headers["Authorization"] == "Bearer e"
