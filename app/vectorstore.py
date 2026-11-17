"""Vector store abstraction: Pinecone (production) and a persistent in-memory cosine store (offline)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings

log = logging.getLogger(__name__)


class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], ids: list[str]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def similarity_search_with_score(self, query: str, k: int = 4,
                                     filter: dict | None = None) -> list[tuple[Document, float]]: ...
    def count(self) -> int: ...


class MemoryVectorStore:
    def __init__(self, embeddings: Embeddings, persist_path: Path | None = None):
        self.embeddings = embeddings
        self.persist_path = Path(persist_path) if persist_path else None
        self._items: dict[str, dict[str, Any]] = {}
        if self.persist_path and self.persist_path.exists():
            self._items = json.loads(self.persist_path.read_text())

    def _save(self) -> None:
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(self._items))

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        for _id, doc, vec in zip(ids, documents, vectors):
            self._items[_id] = {"text": doc.page_content, "metadata": doc.metadata, "vector": vec}
        self._save()

    def delete(self, ids: list[str]) -> None:
        for _id in ids:
            self._items.pop(_id, None)
        self._save()

    def similarity_search_with_score(self, query, k=4, filter=None):
        items = [it for it in self._items.values()
                 if not filter or all(it["metadata"].get(f) == v for f, v in filter.items())]
        if not items:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float64)
        mat = np.array([it["vector"] for it in items], dtype=np.float64)
        denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0))
        denom[denom == 0] = 1.0
        scores = (mat @ q) / denom
        order = np.argsort(-scores)[:k]
        return [(Document(page_content=items[i]["text"], metadata=items[i]["metadata"]), float(scores[i]))
                for i in order]

    def count(self) -> int:
        return len(self._items)


class PineconeStore:
    def __init__(self, embeddings: Embeddings, settings: Settings):
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        self.namespace = settings.pinecone_namespace
        pc = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index
        if name not in pc.list_indexes().names():
            log.info("Creating Pinecone index %s (dim=%d)", name, settings.dimension)
            pc.create_index(name=name, dimension=settings.dimension, metric="cosine",
                            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region))
            while not pc.describe_index(name).status["ready"]:
                time.sleep(1)
        self.index = pc.Index(name)
        self.store = PineconeVectorStore(index=self.index, embedding=embeddings, namespace=self.namespace)

    def add_documents(self, documents, ids):
        self.store.add_documents(documents, ids=ids)

    def delete(self, ids):
        if ids:
            self.store.delete(ids=ids)

    def similarity_search_with_score(self, query, k=4, filter=None):
        return self.store.similarity_search_with_score(query, k=k, filter=filter)

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        ns = (stats.get("namespaces") or {}).get(self.namespace)
        return int(ns["vector_count"]) if ns else 0


def get_vector_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    if settings.vector_backend == "pinecone":
        return PineconeStore(embeddings, settings)
    return MemoryVectorStore(embeddings, settings.memory_store_path)
