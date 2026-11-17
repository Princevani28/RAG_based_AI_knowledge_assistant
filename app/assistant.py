"""High-level facade used by the API, CLI, UI, MCP server and evaluation."""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from app.config import Settings, get_settings
from app.db import Database
from app.embeddings import get_embeddings
from app.graph.workflow import build_graph
from app.ingestion.pipeline import ingest_paths
from app.llm import get_llm
from app.vectorstore import VectorStore, get_vector_store


class KnowledgeAssistant:
    def __init__(self, settings: Settings | None = None, llm: Runnable | None = None,
                 store: VectorStore | None = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings.sqlite_path)
        self.store = store or get_vector_store(self.settings, get_embeddings(self.settings))
        self.llm = llm or get_llm(self.settings)
        self.graph = build_graph(self.llm, self.store, self.settings)

    def ingest(self, paths: list[Path | str] | None = None) -> dict[str, Any]:
        return ingest_paths(paths or [self.settings.documents_dir], self.store, self.db, self.settings)

    def ask(self, question: str, session_id: str = "default") -> dict[str, Any]:
        start = time.perf_counter()
        state = self.graph.invoke(
            {"question": question, "steps": [], "retries": 0, "documents": [], "sources": [],
             "answer": "", "grounded": True},
            config={"configurable": {"thread_id": session_id}},
        )
        latency = round((time.perf_counter() - start) * 1000, 1)
        result = {"session_id": session_id, "answer": state["answer"], "sources": state.get("sources", []),
                  "route": state.get("route"), "grounded": state.get("grounded", True),
                  "steps": state.get("steps", []), "latency_ms": latency}
        self.db.log_interaction(session_id, question, result["answer"], result["route"] or "",
                                result["grounded"], result["sources"], result["steps"], latency)
        return result


@lru_cache(maxsize=1)
def get_assistant() -> KnowledgeAssistant:
    assistant = KnowledgeAssistant()
    assistant.ingest()  # idempotent: unchanged files are skipped
    return assistant
