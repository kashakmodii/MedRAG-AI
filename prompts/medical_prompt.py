"""Prompt helpers for the medical RAG assistant."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Use the context below to answer. "
    "If the context is insufficient, say so clearly. "
    "Do not provide a diagnosis as a certainty. "
    "Encourage professional medical care for urgent or severe concerns."
)


DISCLAIMER = "This is not medical advice."


def build_medical_prompt(query: str, retrieved_chunks: str | None = None) -> str:
    """Build the structured prompt used for grounded answering."""

    context = retrieved_chunks.strip() if retrieved_chunks else "No document context provided."
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Answer clearly and safely. {DISCLAIMER}"
    )


def build_general_prompt(query: str) -> str:
    """Build a general medical prompt for fallback mode without PDF context."""

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Answer clearly and safely. {DISCLAIMER}"
    )