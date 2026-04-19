"""Optional dataset helpers for indexing local PDFs from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.pdf_loader import extract_pdf_pages


def load_pdf_corpus(data_dir: str | Path) -> list[dict[str, str | int]]:
    """Load every PDF page from a directory into a simple corpus list."""

    corpus: list[dict[str, str | int]] = []
    for pdf_path in Path(data_dir).glob("*.pdf"):
        for page in extract_pdf_pages(pdf_path.read_bytes()):
            if page.text:
                corpus.append(
                    {
                        "source": pdf_path.name,
                        "page_number": page.page_number,
                        "text": page.text,
                    }
                )
    return corpus


def iter_pdf_paths(data_dir: str | Path) -> Iterable[Path]:
    """Yield local PDF files from a directory."""

    yield from Path(data_dir).glob("*.pdf")