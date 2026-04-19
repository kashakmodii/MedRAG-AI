"""PDF extraction helpers for local medical documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO, List

import pdfplumber

from utils.text_cleaner import clean_text


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str


def extract_pdf_pages(pdf_file: bytes | BinaryIO) -> List[ExtractedPage]:
    """Extract text from each page in a PDF file-like object or raw bytes."""

    if isinstance(pdf_file, (bytes, bytearray)):
        pdf_stream = BytesIO(pdf_file)
    else:
        pdf_stream = pdf_file

    pages: List[ExtractedPage] = []
    with pdfplumber.open(pdf_stream) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            pages.append(ExtractedPage(page_number=page_number, text=page_text))

    return pages


def extract_pdf_text(pdf_file: bytes | BinaryIO) -> str:
    """Extract and concatenate all text from a PDF."""

    pages = extract_pdf_pages(pdf_file)
    return clean_text(" ".join(page.text for page in pages if page.text))