"""Central configuration. Defaults run fully offline (mock LLM, local embeddings, in-memory store)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_backend: str = "mock"          # mock | openai
    embedding_backend: str = "local"   # local | openai
    vector_backend: str = "memory"     # memory | pinecone

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    pinecone_api_key: str = ""
    pinecone_index: str = "knowledge-assistant"
    pinecone_namespace: str = "enterprise-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_relevance: float = 0.15
    max_retries: int = 1
    max_history: int = 6

    data_dir: Path = ROOT / "data"
    sqlite_path: Path = ROOT / "data" / "assistant.db"
    memory_store_path: Path = ROOT / "data" / "memory_store.json"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def dimension(self) -> int:
        """Embedding dimension; must match the Pinecone index."""
        return 1536 if self.embedding_backend == "openai" else 512

    def validate(self) -> None:
        if self.llm_backend not in {"mock", "openai"}:
            raise ConfigError(f"Invalid LLM_BACKEND={self.llm_backend}")
        if self.embedding_backend not in {"local", "openai"}:
            raise ConfigError(f"Invalid EMBEDDING_BACKEND={self.embedding_backend}")
        if self.vector_backend not in {"memory", "pinecone"}:
            raise ConfigError(f"Invalid VECTOR_BACKEND={self.vector_backend}")
        if "openai" in (self.llm_backend, self.embedding_backend) and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when using the openai backend.")
        if self.vector_backend == "pinecone" and not self.pinecone_api_key:
            raise ConfigError("PINECONE_API_KEY is required when VECTOR_BACKEND=pinecone.")

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.getenv
        return cls(
            llm_backend=e("LLM_BACKEND", "mock").split("#")[0].strip(),
            embedding_backend=e("EMBEDDING_BACKEND", "local").split("#")[0].strip(),
            vector_backend=e("VECTOR_BACKEND", "memory").split("#")[0].strip(),
            openai_api_key=e("OPENAI_API_KEY", ""),
            llm_model=e("LLM_MODEL", "gpt-4o-mini"),
            embedding_model=e("EMBEDDING_MODEL", "text-embedding-3-small"),
            pinecone_api_key=e("PINECONE_API_KEY", ""),
            pinecone_index=e("PINECONE_INDEX", "knowledge-assistant"),
            pinecone_namespace=e("PINECONE_NAMESPACE", "enterprise-docs"),
            pinecone_cloud=e("PINECONE_CLOUD", "aws"),
            pinecone_region=e("PINECONE_REGION", "us-east-1"),
            chunk_size=int(e("CHUNK_SIZE", "800").split("#")[0]),
            chunk_overlap=int(e("CHUNK_OVERLAP", "120").split("#")[0]),
            top_k=int(e("TOP_K", "4").split("#")[0]),
            min_relevance=float(e("MIN_RELEVANCE", "0.15").split("#")[0]),
            max_retries=int(e("MAX_RETRIES", "1").split("#")[0]),
            max_history=int(e("MAX_HISTORY", "6").split("#")[0]),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings.from_env()
    s.validate()
    return s


def setup_logging() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
