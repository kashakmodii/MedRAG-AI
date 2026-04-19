"""Streamlit application for the local MedRAG-AI medical chatbot."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st

from models.embedding_model import EmbeddingModel
from models.llm_model import OllamaLLM
from prompts.medical_prompt import build_general_prompt, build_medical_prompt
from rag.faiss_index import FaissVectorStore
from rag.retriever import Retriever
from utils.pdf_loader import extract_pdf_text
from utils.text_chunker import chunk_text
from utils.text_cleaner import clean_text, get_medical_keywords, highlight_keywords


APP_TITLE = "MedRAG-AI 🏥"
INDEX_DIR = Path("data") / "index"


st.set_page_config(page_title=APP_TITLE, page_icon="🏥", layout="wide")


def initialize_session_state() -> None:
    """Create all session-scoped state used by the chat UI."""

    defaults = {
        "messages": [],
        "vector_store": None,
        "document_name": None,
        "document_text": "",
        "document_chunks": [],
        "processing_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_document_state() -> None:
    """Remove the active document index and cached content."""

    st.session_state.vector_store = None
    st.session_state.document_name = None
    st.session_state.document_text = ""
    st.session_state.document_chunks = []
    st.session_state.processing_error = None


def load_uploaded_pdf(uploaded_file) -> tuple[FaissVectorStore, str, list[str]]:
    """Extract text from the uploaded PDF and build a fresh FAISS store."""

    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    text = extract_pdf_text(BytesIO(pdf_bytes))
    if not text.strip():
        raise ValueError("No readable text could be extracted from the PDF.")

    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text, chunk_size_words=400, overlap_words=80)
    if not chunks:
        raise ValueError("The PDF did not produce any chunks for retrieval.")

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.get_embeddings(chunks)

    vector_store = FaissVectorStore()
    metadata = [
        {"source": uploaded_file.name, "chunk_number": index + 1}
        for index in range(len(chunks))
    ]
    vector_store.build(embeddings, chunks, metadata)
    return vector_store, cleaned_text, chunks


def format_retrieved_chunks(results: list[dict]) -> str:
    """Combine retrieved chunks into a prompt-ready context block."""

    context_blocks: list[str] = []
    for result in results:
        metadata = result.get("metadata", {}) or {}
        source = metadata.get("source", "Uploaded PDF")
        chunk_number = metadata.get("chunk_number", result.get("index", 0) + 1)
        context_blocks.append(f"Source: {source} | Chunk: {chunk_number}\n{result['text']}")
    return "\n\n---\n\n".join(context_blocks)


def render_sidebar() -> None:
    """Display app instructions and safety guidance."""

    with st.sidebar:
        st.title("Instructions")
        st.markdown(
            """
            1. Install and start Ollama locally.
            2. Pull the `mistral` model, or keep `phi` available as fallback.
            3. Upload a medical PDF to enable grounded answers.
            4. Ask a question in the chat box.

            The app works without a PDF in fallback mode for general medical questions.
            """
        )
        st.divider()
        st.warning("This is not medical advice.")
        if st.session_state.vector_store is not None and st.button("Save FAISS index"):
            try:
                st.session_state.vector_store.save(str(INDEX_DIR))
                st.success(f"Saved FAISS index to {INDEX_DIR}.")
            except Exception as exc:
                st.error(f"Could not save index: {exc}")

        if st.button("Load saved FAISS index"):
            try:
                st.session_state.vector_store = FaissVectorStore.load(str(INDEX_DIR))
                st.session_state.document_name = "Saved FAISS index"
                st.session_state.document_text = ""
                st.session_state.document_chunks = [chunk.text for chunk in st.session_state.vector_store.chunks]
                st.session_state.processing_error = None
                st.success(f"Loaded saved FAISS index from {INDEX_DIR}.")
            except Exception as exc:
                st.error(f"Could not load saved index: {exc}")
        if st.button("Clear chat and document"):
            st.session_state.messages = []
            clear_document_state()
            st.rerun()


def render_chat_history() -> None:
    """Replay previous conversation turns."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)


def handle_query(user_query: str, retriever: Retriever | None, llm: OllamaLLM) -> None:
    """Generate and append an answer for the current user query."""

    if not user_query.strip():
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            try:
                if retriever is not None and st.session_state.vector_store is not None:
                    retrieved_chunks = retriever.retrieve(user_query, top_k=3)
                    context = format_retrieved_chunks(retrieved_chunks)
                    prompt = build_medical_prompt(user_query, context)
                else:
                    retrieved_chunks = []
                    context = ""
                    prompt = build_general_prompt(user_query)

                if not llm.is_running():
                    st.warning("Ollama is not running. Start it locally with `ollama serve`.")
                    fallback_response = (
                        "I cannot generate a local answer because Ollama is not running. "
                        "Please start Ollama and retry. This is not medical advice."
                    )
                    st.markdown(highlight_keywords(fallback_response), unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_response})
                    return

                result = llm.generate(prompt)
                highlighted_response = highlight_keywords(result.response)
                st.markdown(highlighted_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": highlighted_response})

                if retrieved_chunks:
                    with st.expander("Retrieved chunks"):
                        for index, chunk in enumerate(retrieved_chunks, start=1):
                            metadata = chunk.get("metadata", {}) or {}
                            source = metadata.get("source", "Uploaded PDF")
                            st.markdown(f"**Chunk {index}** | Source: {source} | Distance: {chunk['distance']:.4f}")
                            st.markdown(highlight_keywords(chunk["text"]), unsafe_allow_html=True)
                            st.divider()
                elif context == "":
                    with st.expander("Retrieved chunks"):
                        st.info("No PDF has been uploaded, so the assistant used fallback general medical mode.")

            except Exception as exc:
                fallback_message = (
                    "I am temporarily unable to generate a local model response. "
                    "Please retry in a few seconds. If this continues, confirm Ollama is running and the model is loaded. "
                    f"Technical details: {str(exc)}"
                )
                st.warning("Local generation had a temporary issue. Returned a fallback assistant message.")
                st.markdown(highlight_keywords(fallback_message), unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": fallback_message})


def main() -> None:
    """Render the Streamlit app."""

    initialize_session_state()
    render_sidebar()

    st.title(APP_TITLE)
    st.caption("Local medical RAG chatbot with FAISS retrieval and Ollama generation.")

    uploaded_file = st.file_uploader("Upload a PDF medical report", type=["pdf"])
    process_clicked = st.button("Process PDF")

    if uploaded_file is not None and (process_clicked or st.session_state.document_name != uploaded_file.name):
        with st.spinner("Extracting and indexing the PDF..."):
            try:
                vector_store, extracted_text, chunks = load_uploaded_pdf(uploaded_file)
                st.session_state.vector_store = vector_store
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_text = extracted_text
                st.session_state.document_chunks = chunks
                st.session_state.processing_error = None
                st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}.")
            except Exception as exc:
                clear_document_state()
                st.session_state.processing_error = str(exc)
                st.error(f"Could not process PDF: {exc}")

    if st.session_state.document_name:
        keyword_hits = get_medical_keywords(st.session_state.document_text)
        if keyword_hits:
            st.info("Detected keywords: " + ", ".join(keyword_hits))

    render_chat_history()

    retriever = None
    if st.session_state.vector_store is not None:
        retriever = Retriever(st.session_state.vector_store, EmbeddingModel())

    user_query = st.chat_input("Ask a medical question")
    if user_query:
        handle_query(user_query, retriever, OllamaLLM())


if __name__ == "__main__":
    main()