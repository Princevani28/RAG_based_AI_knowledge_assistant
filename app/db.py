"""SQLite persistence: document registry (ingestion idempotency) + interaction audit log."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    route TEXT,
    grounded INTEGER,
    sources TEXT,
    steps TEXT,
    latency_ms REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- document registry ----
    def get_document(self, source: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE source=?", (source,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["chunk_ids"] = json.loads(d["chunk_ids"])
        return d

    def upsert_document(self, source: str, content_hash: str, chunk_ids: list[str]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents(source, content_hash, chunk_ids, chunk_count) VALUES(?,?,?,?) "
                "ON CONFLICT(source) DO UPDATE SET content_hash=excluded.content_hash, "
                "chunk_ids=excluded.chunk_ids, chunk_count=excluded.chunk_count, "
                "ingested_at=CURRENT_TIMESTAMP",
                (source, content_hash, json.dumps(chunk_ids), len(chunk_ids)),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT source, content_hash, chunk_count, ingested_at FROM documents ORDER BY source"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- interaction log ----
    def log_interaction(self, session_id: str, question: str, answer: str, route: str,
                        grounded: bool, sources: list[dict], steps: list[str],
                        latency_ms: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO interactions(session_id, question, answer, route, grounded, sources, steps, latency_ms) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (session_id, question, answer, route, int(grounded), json.dumps(sources),
                 json.dumps(steps), latency_ms),
            )

    def history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT question, answer, route, grounded, latency_ms, created_at FROM interactions "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]
