"""SentenceTransformer embedding wrapper used across the app."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


# Reduce noisy third-party model load logs in app output.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    token = os.getenv("HF_TOKEN") or None
    return SentenceTransformer(model_name, token=token)


class EmbeddingModel:
    """Load the embedding model once and reuse it for all requests."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = _load_model(model_name)

    def get_embeddings(self, text_list: Iterable[str]) -> np.ndarray:
        """Return float32 embeddings for a list of texts."""

        texts = list(text_list)
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single embedding vector for a query string."""

        return self.get_embeddings([text])[0]