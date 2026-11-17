from app.assistant import KnowledgeAssistant
from tests.conftest import DOCS


def test_ingest_creates_chunks_with_metadata(assistant):
    assert assistant.store.count() > 3
    docs = assistant.store.similarity_search_with_score("annual leave", k=1)
    meta = docs[0][0].metadata
    assert meta["source"] == "hr_policy.md" and "chunk_index" in meta and "doc_hash" in meta


def test_reingest_is_idempotent(assistant):
    before = assistant.store.count()
    report = assistant.ingest([DOCS])
    assert not report["ingested"] and len(report["skipped"]) == 3
    assert assistant.store.count() == before


def test_changed_file_replaces_old_vectors(settings, tmp_path):
    a = KnowledgeAssistant(settings)
    f = tmp_path / "note.txt"
    f.write_text("The office wifi password is rotated weekly by the facilities team.")
    a.ingest([f])
    n1 = a.store.count()
    f.write_text("The cafeteria opens at 8am and closes at 4pm every weekday.")
    r = a.ingest([f])
    assert r["ingested"] and a.store.count() == n1
    hit = a.store.similarity_search_with_score("cafeteria opens", k=1)[0][0]
    assert "cafeteria" in hit.page_content


def test_empty_and_unsupported_files_are_skipped(settings, tmp_path):
    a = KnowledgeAssistant(settings)
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "image.png").write_bytes(b"x")
    r = a.ingest([tmp_path])
    assert r["ingested"] == [] and r["skipped"][0]["reason"] == "no extractable text"
