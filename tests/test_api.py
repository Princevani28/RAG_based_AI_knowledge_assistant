import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.assistant import get_assistant


@pytest.fixture()
def client(assistant):
    app.dependency_overrides[get_assistant] = lambda: assistant
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_and_history(client):
    r = client.post("/chat", json={"question": "How much does the Pro plan cost per month?", "session_id": "api1"})
    assert r.status_code == 200 and "$99" in r.json()["answer"]
    assert len(client.get("/history/api1").json()["history"]) == 1


def test_chat_validation(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_upload_ingest_then_query(client):
    files = [("files", ("kb.txt", b"The company mascot is a friendly otter named Ollie.", "text/plain"))]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200 and r.json()["ingested"][0]["source"] == "kb.txt"
    ans = client.post("/chat", json={"question": "What is the company mascot?", "session_id": "api2"}).json()
    assert "otter" in ans["answer"]
    assert any(d["source"] == "kb.txt" for d in client.get("/documents").json()["documents"])


def test_rejects_unsupported_upload(client):
    r = client.post("/ingest", files=[("files", ("x.exe", b"bin", "application/octet-stream"))])
    assert r.status_code == 400
