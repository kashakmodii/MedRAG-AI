"""Retrieval helpers that connect embeddings to FAISS search."""

from __future__ import annotations

from typing import Any

from models.embedding_model import EmbeddingModel
from rag.faiss_index import FaissVectorStore


class Retriever:
    """High-level retriever that accepts a query string and returns relevant chunks."""

    def __init__(self, vector_store: FaissVectorStore, embedding_model: EmbeddingModel) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the most relevant chunks for a query."""

        if not query.strip() or not self.vector_store.is_ready:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.search(query_embedding, top_k=top_k)