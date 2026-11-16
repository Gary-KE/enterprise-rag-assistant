import io

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.security import decode_token, issue_token

ADMIN_KEY = "a" * 32
SECRET = "s" * 40


@pytest.fixture
def client(pipeline, settings):
    settings.admin_key = ADMIN_KEY
    settings.jwt_secret = SECRET
    settings.rate_limit_per_min = 1000
    return TestClient(create_app(settings, pipeline))


def admin_headers(client):
    r = client.post("/api/auth/token", json={"admin_key": ADMIN_KEY})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_and_security_headers(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert len(r.headers["x-request-id"]) == 12


def test_ui_and_static_files_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Dondoo" in r.text and "<script>" not in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_ask_returns_cited_answer(client):
    r = client.post("/api/ask", json={"question": "Within how many hours must the controller notify the Commissioner of a breach?"})
    body = r.json()
    assert r.status_code == 200 and body["declined"] is False
    assert body["citations"][0]["page"] == 3


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/ask", json={}).status_code == 422


def test_library_lists_public_documents(client):
    docs = client.get("/api/documents").json()
    assert [d["title"] for d in docs] == ["Sample Data Protection Act"]
    assert docs[0]["chunks"] > 0


def test_bad_tokens_are_rejected(client):
    assert client.get("/api/documents", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Basic abc"}).status_code == 401
    forged = issue_token("x" * 40, "admin", 5)                     # signed with the wrong secret
    assert client.get("/api/documents", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    expired = issue_token(SECRET, "admin", -1)
    assert client.get("/api/documents", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_unsigned_token_is_rejected():
    import jwt
    none_token = jwt.encode({"role": "admin", "iat": 0, "exp": 9999999999}, key=None, algorithm="none")
    assert decode_token(SECRET, none_token) is None


def test_wrong_admin_key_is_rejected(client):
    assert client.post("/api/auth/token", json={"admin_key": "wrong"}).status_code == 401


def test_admin_login_disabled_when_no_key_set(pipeline, settings):
    settings.admin_key = ""
    c = TestClient(create_app(settings, pipeline))
    assert c.post("/api/auth/token", json={"admin_key": "anything"}).status_code == 401


def test_upload_requires_admin(client):
    files = {"file": ("doc.txt", b"Some text about the branch network expansion plans.", "text/plain")}
    assert client.post("/api/documents", files=files, data={"title": "Doc"}).status_code == 403


def test_admin_can_upload_then_ask(client):
    h = admin_headers(client)
    files = {"file": ("plan.txt", b"The bank plans to open fifteen agency outlets in Kisumu during the next financial year.", "text/plain")}
    r = client.post("/api/documents", files=files, data={"title": "Branch plan", "url": "https://example.org/plan"}, headers=h)
    assert r.status_code == 201, r.text
    again = client.post("/api/documents", files=files, data={"title": "Branch plan"}, headers=h)
    assert again.status_code == 200 and again.json()["status"] == "skipped_duplicate"
    ans = client.post("/api/ask", json={"question": "How many agency outlets will the bank open in Kisumu?"}).json()
    assert not ans["declined"] and ans["citations"][0]["title"] == "Branch plan"


def test_upload_rejects_wrong_types_fake_pdfs_and_big_files(client, settings):
    h = admin_headers(client)
    exe = {"file": ("x.exe", b"MZ....", "application/octet-stream")}
    assert client.post("/api/documents", files=exe, data={"title": "Test file"}, headers=h).status_code == 415
    fake_pdf = {"file": ("x.pdf", b"<html>not a pdf</html>", "application/pdf")}
    assert client.post("/api/documents", files=fake_pdf, data={"title": "Test file"}, headers=h).status_code == 415
    settings.max_upload_mb = 1
    big = {"file": ("big.txt", io.BytesIO(b"a " * (1024 * 1024)), "text/plain")}
    assert client.post("/api/documents", files=big, data={"title": "Test file"}, headers=h).status_code == 413
    js_url = {"file": ("ok.txt", b"Some useful words for the test here.", "text/plain")}
    r = client.post("/api/documents", files=js_url, data={"title": "Test file", "url": "javascript:alert(1)"}, headers=h)
    assert r.status_code == 422


def test_path_traversal_filename_is_neutralised(client):
    h = admin_headers(client)
    files = {"file": ("../../etc/evil.txt", b"Harmless text about savings account interest rates and fees.", "text/plain")}
    r = client.post("/api/documents", files=files, data={"title": "Evil"}, headers=h)
    assert r.status_code == 201
    docs = client.get("/api/documents").json()
    assert all(".." not in d["title"] for d in docs)


def test_rate_limit(pipeline, settings):
    settings.rate_limit_per_min = 2
    c = TestClient(create_app(settings, pipeline))
    q = {"question": "What is the capital of France?"}
    codes = [c.post("/api/ask", json=q).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_forged_forwarded_for_does_not_bypass_rate_limit(pipeline, settings):
    settings.rate_limit_per_min = 1
    c = TestClient(create_app(settings, pipeline))
    q = {"question": "What is the capital of France?"}
    assert c.post("/api/ask", json=q, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.post("/api/ask", json=q, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_trusted_proxy_uses_rightmost_address(pipeline, settings):
    settings.rate_limit_per_min = 1
    settings.trusted_proxy_hops = 1
    c = TestClient(create_app(settings, pipeline))
    q = {"question": "What is the capital of France?"}
    assert c.post("/api/ask", json=q, headers={"X-Forwarded-For": "9.9.9.9, 5.5.5.5"}).status_code == 200
    # client forges a different left-most value, but the proxy-added right-most one is the same
    assert c.post("/api/ask", json=q, headers={"X-Forwarded-For": "7.7.7.7, 5.5.5.5"}).status_code == 429
    assert c.post("/api/ask", json=q, headers={"X-Forwarded-For": "6.6.6.6"}).status_code == 200


def test_ask_accepts_language_and_rejects_others(client):
    ok = client.post("/api/ask", json={"question": "Mji mkuu wa Ufaransa ni upi?", "lang": "sw"})
    assert ok.status_code == 200 and ok.json()["answer"].startswith("Sikupata")
    assert client.post("/api/ask", json={"question": "anything here", "lang": "fr"}).status_code == 422


def test_api_docs_page_may_load_swagger_but_app_pages_stay_strict(client):
    docs = client.get("/api/docs")
    assert docs.status_code == 200 and "cdn.jsdelivr.net" in docs.headers["content-security-policy"]
    assert "cdn.jsdelivr.net" not in client.get("/").headers["content-security-policy"]
    assert "unsafe-inline" not in client.get("/").headers["content-security-policy"]
