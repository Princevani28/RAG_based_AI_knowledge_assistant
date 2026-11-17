"""FastAPI service: /health, /chat, /ingest, /documents, /history/{session_id}."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.assistant import KnowledgeAssistant, get_assistant
from app.config import setup_logging
from app.ingestion.pipeline import SUPPORTED

setup_logging()
app = FastAPI(title="AI-Powered Knowledge Assistant", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, assistant: KnowledgeAssistant = Depends(get_assistant)):
    return assistant.ask(req.question, req.session_id)


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...), assistant: KnowledgeAssistant = Depends(get_assistant)):
    upload_dir = assistant.settings.documents_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file type: {name}. Allowed: {sorted(SUPPORTED)}")
        dest = upload_dir / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    return assistant.ingest(saved)


@app.get("/documents")
def documents(assistant: KnowledgeAssistant = Depends(get_assistant)):
    return {"documents": assistant.db.list_documents()}


@app.get("/history/{session_id}")
def history(session_id: str, assistant: KnowledgeAssistant = Depends(get_assistant)):
    return {"session_id": session_id, "history": assistant.db.history(session_id)}
