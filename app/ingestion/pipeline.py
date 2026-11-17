"""Document processing: load -> chunk -> embed -> upsert into the vector DB (idempotent via SQL registry)."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.db import Database
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".txt", ".md"}


def collect_files(path: Path | str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED else []
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)


def load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [Document(page_content=(pg.extract_text() or ""), metadata={"source": path.name, "page": i + 1})
                for i, pg in enumerate(reader.pages)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                              chunk_overlap=settings.chunk_overlap)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def ingest_paths(paths: list[Path | str], store: VectorStore, db: Database, settings: Settings) -> dict[str, Any]:
    report: dict[str, Any] = {"ingested": [], "skipped": [], "total_chunks": 0}
    files: list[Path] = []
    for p in paths:
        files.extend(collect_files(p))
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        existing = db.get_document(f.name)
        if existing and existing["content_hash"] == digest:
            report["skipped"].append({"source": f.name, "reason": "unchanged"})
            continue
        chunks = split_documents(load_file(f), settings)
        if not chunks:
            report["skipped"].append({"source": f.name, "reason": "no extractable text"})
            continue
        if existing:  # document changed -> remove stale vectors
            store.delete(existing["chunk_ids"])
        ids = [f"{digest[:16]}-{i}" for i in range(len(chunks))]
        for c in chunks:
            c.metadata["doc_hash"] = digest[:16]
        store.add_documents(chunks, ids=ids)
        db.upsert_document(f.name, digest, ids)
        report["ingested"].append({"source": f.name, "chunks": len(chunks)})
        report["total_chunks"] += len(chunks)
        log.info("Ingested %s (%d chunks)", f.name, len(chunks))
    return report
