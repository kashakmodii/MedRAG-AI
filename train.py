"""Optional local indexing script for PDFs stored in the data directory."""

from __future__ import annotations

from models.embedding_model import EmbeddingModel
from rag.faiss_index import FaissVectorStore
from data.dataset_loader import load_pdf_corpus
from utils.text_chunker import chunk_text
from utils.text_cleaner import clean_text


def build_local_index(data_dir: str = "data", output_dir: str = "data/index") -> None:
    """Create a reusable FAISS index from PDFs placed in the data folder."""

    corpus = load_pdf_corpus(data_dir)
    if not corpus:
        raise ValueError("No PDF pages were found in the data directory.")

    chunks: list[str] = []
    metadata: list[dict[str, object]] = []
    for item in corpus:
        text = clean_text(str(item["text"]))
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metadata.append({"source": item["source"], "page_number": item["page_number"]})

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.get_embeddings(chunks)

    vector_store = FaissVectorStore()
    vector_store.build(embeddings, chunks, metadata)
    vector_store.save(output_dir)


if __name__ == "__main__":
    build_local_index()