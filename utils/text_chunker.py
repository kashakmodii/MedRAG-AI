"""Chunk text into overlapping word windows for retrieval."""

from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size_words: int = 400, overlap_words: int = 80) -> List[str]:
    """Split text into roughly equal-sized chunks with overlap."""

    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunk_size_words = max(1, chunk_size_words)
    overlap_words = max(0, min(overlap_words, chunk_size_words - 1))
    step = max(1, chunk_size_words - overlap_words)

    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size_words]
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks