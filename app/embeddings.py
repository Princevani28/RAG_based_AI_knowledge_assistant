"""Embedding backends: OpenAI (production) and a deterministic local hashing embedder (offline/tests)."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import Settings

_STOP = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or", "do", "does",
         "how", "what", "who", "which", "can", "be", "by", "at", "it", "with", "i", "my", "me", "get"}


class LocalHashEmbeddings(Embeddings):
    """Bag-of-words + bigram feature hashing. Deterministic, no network; good enough for demos/tests."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _stem(w: str) -> str:
        for suffix in ("ing", "es", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                return w[: -len(suffix)]
        return w

    def _tokens(self, text: str) -> list[str]:
        words = [self._stem(w) for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in _STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return LocalHashEmbeddings(dim=settings.dimension)
