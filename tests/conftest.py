from dataclasses import replace
from pathlib import Path

import pytest

from app.assistant import KnowledgeAssistant
from app.config import ROOT, Settings

GOLDEN = ROOT / "tests" / "golden" / "golden_qa.json"
DOCS = ROOT / "data" / "documents"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return replace(Settings(), data_dir=tmp_path, sqlite_path=tmp_path / "t.db", memory_store_path=tmp_path / "m.json",
                   chunk_size=400, chunk_overlap=50, min_relevance=0.15)


@pytest.fixture()
def assistant(settings) -> KnowledgeAssistant:
    a = KnowledgeAssistant(settings)
    a.ingest([DOCS])
    return a
