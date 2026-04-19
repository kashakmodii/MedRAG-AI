# MedRAG-AI: Local Medical RAG Chatbot

MedRAG-AI is a fully local, offline medical retrieval-augmented generation chatbot built with Streamlit, FAISS, Sentence Transformers, and Ollama. It lets you upload PDF medical reports, extract and chunk the text, retrieve relevant passages, and generate grounded answers without using paid APIs.

## Features

- ChatGPT-style Streamlit chat interface
- PDF upload and text extraction with `pdfplumber`
- Text cleaning, chunking, and embedding generation
- FAISS CPU vector search with save/load support
- Local Ollama inference with `mistral` and `phi` fallback
- Session-based chat history
- Retrieved chunk inspection in expandable UI
- Basic medical keyword highlighting
- Offline fallback mode for general medical questions when no PDF is loaded
- Clear medical disclaimer in the UI

## Project Structure

```text
med-rag-ai/
├── app.py
├── requirements.txt
├── README.md
├── utils/
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   └── text_cleaner.py
├── models/
│   ├── embedding_model.py
│   └── llm_model.py
├── rag/
│   ├── faiss_index.py
│   └── retriever.py
├── prompts/
│   └── medical_prompt.py
└── data/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Ollama: https://ollama.com

3. Pull the recommended model:

```bash
ollama pull mistral
```

4. Run the app:

```bash
streamlit run app.py
```

## Screenshots

Add screenshots of the chat interface, PDF upload flow, and retrieved chunk panel here.

## Resume-Ready Summary

Built a production-ready offline medical RAG chatbot that combines PDF ingestion, semantic retrieval with FAISS, local embeddings via Sentence Transformers, and low-latency generation through Ollama. The system is modular, CPU-friendly, and designed to support secure local use without external APIs.

## Disclaimer

This is not medical advice. The application is intended for informational purposes only and should not replace a qualified clinician.
