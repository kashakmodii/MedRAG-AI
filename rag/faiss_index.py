"""FAISS vector store wrapper for chunk retrieval."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Iterable

import faiss
import numpy as np


@dataclass(slots=True)
class ChunkRecord:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FaissVectorStore:
    """Simple FAISS index with companion chunk metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[ChunkRecord] = []

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def build(self, embeddings: np.ndarray, chunks: Iterable[str], metadata: Iterable[dict[str, Any]] | None = None) -> None:
        """Create a new index from scratch."""

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        chunk_list = list(chunks)
        metadata_list = list(metadata or [{} for _ in chunk_list])
        self.chunks = [ChunkRecord(text=chunk, metadata=meta) for chunk, meta in zip(chunk_list, metadata_list, strict=False)]

    def add(self, embeddings: np.ndarray, chunks: Iterable[str], metadata: Iterable[dict[str, Any]] | None = None) -> None:
        """Append chunks and vectors to an existing index or build if needed."""

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        chunk_list = list(chunks)
        metadata_list = list(metadata or [{} for _ in chunk_list])

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(
            ChunkRecord(text=chunk, metadata=meta) for chunk, meta in zip(chunk_list, metadata_list, strict=False)
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve the closest chunks for a query vector."""

        if self.index is None or self.index.ntotal == 0 or not self.chunks:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        results: list[dict[str, Any]] = []
        for distance, index in zip(distances[0], indices[0], strict=False):
            if index < 0 or index >= len(self.chunks):
                continue
            record = self.chunks[index]
            results.append(
                {
                    "text": record.text,
                    "metadata": record.metadata,
                    "distance": float(distance),
                    "index": int(index),
                }
            )
        return results

    def save(self, directory: str) -> None:
        """Persist the index and metadata to disk."""

        if self.index is None:
            raise ValueError("No FAISS index is available to save.")

        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "faiss_chunks.pkl"), "wb") as file_handle:
            pickle.dump(self.chunks, file_handle)

    @classmethod
    def load(cls, directory: str) -> "FaissVectorStore":
        """Load a previously saved FAISS index from disk."""

        index_path = os.path.join(directory, "faiss.index")
        chunks_path = os.path.join(directory, "faiss_chunks.pkl")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Saved FAISS index files were not found.")

        store = cls()
        store.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as file_handle:
            store.chunks = pickle.load(file_handle)
        return store