"""Text normalization helpers for medical PDF content."""

from __future__ import annotations

import re
from html import escape
from typing import Iterable


MEDICAL_KEYWORDS = {
    "pain",
    "fever",
    "cough",
    "dose",
    "dosage",
    "blood",
    "pressure",
    "diabetes",
    "hypertension",
    "infection",
    "symptom",
    "diagnosis",
    "treatment",
    "medication",
    "nausea",
    "headache",
    "fatigue",
    "allergy",
    "asthma",
    "imaging",
    "lab",
    "result",
    "results",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious extraction artifacts."""

    if not text:
        return ""

    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    return normalized.strip()


def get_medical_keywords(text: str, keywords: Iterable[str] | None = None) -> list[str]:
    """Return a small set of medical terms found in the text."""

    keyword_set = set(keywords or MEDICAL_KEYWORDS)
    lower_text = text.lower()
    return sorted({keyword for keyword in keyword_set if keyword in lower_text})


def highlight_keywords(text: str, keywords: Iterable[str] | None = None) -> str:
    """Highlight medical keywords using basic HTML markup for Streamlit."""

    if not text:
        return ""

    keyword_set = sorted(set(keywords or MEDICAL_KEYWORDS), key=len, reverse=True)
    escaped = escape(text)

    for keyword in keyword_set:
        pattern = re.compile(rf"(?<!\w)({re.escape(escape(keyword))})(?!\w)", re.IGNORECASE)
        escaped = pattern.sub(r'<mark style="background:#fff2a8;padding:0 0.2rem;border-radius:0.2rem;">\1</mark>', escaped)

    return escaped