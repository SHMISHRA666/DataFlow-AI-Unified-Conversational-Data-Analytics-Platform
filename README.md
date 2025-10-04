# 📄 RAG Ingestion Pipeline with Gemini + FAISS

This project is a **Retrieval-Augmented Generation (RAG) pipeline** that ingests documents, extracts text/captions from multiple formats, generates embeddings, and builds a FAISS index for semantic search and chat.  
It supports **PDF, JSON, HTML, and images** (with OCR + Gemini multimodal fallback).  

---

## 🚀 Features

- **Multi-format ingestion**:
  - 📑 PDF → Markdown (via `pymupdf4llm`)
  - 🧾 JSON → Pretty-printed text
  - 🌐 HTML → Text extraction (with Playwright screenshot fallback)
  - 🖼️ Images → OCR first, Gemini fallback for pictorial content
- **Image captioning**:
  - OCR with preprocessing
  - Gemini fallback → if no text is detected
- **Embeddings**:
  - Uses `nomic-embed-text` (via Ollama or any embedding API)
- **Vector search**:
  - FAISS backend for fast nearest-neighbor search
- **Chat with docs**:
  - Context-aware answers using Gemini

---

## ⚙️ Installation

### 1. Clone & Setup
```bash
git clone <your-repo>
cd project
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Install Extras
- **FAISS**:  
  ```bash
  pip install faiss-cpu
  ```
- **Playwright (for HTML rendering)**:  
  ```bash
  pip install playwright
  playwright install chromium
  ```
- **OCR (optional, for images)**:  
  ```bash
  pip install pillow pytesseract
  ```
  > Requires [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) installed on system.

---

## 🔑 Environment Setup

Create a `.env` file in the project root:

```ini
# Directories
DOCS_DIR=./documents
FAISS_DIR=./faiss_index

# Embeddings
EMBED_API_URL=http://localhost:11434/api/embeddings
EMBED_MODEL=nomic-embed-text
EMBED_API_KEY=

# Gemini
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🛠️ Usage

### 1. Ingest documents
Drop your files (`.pdf`, `.json`, `.html`, `.png`, `.jpg`) into the `documents/` folder and run:

```bash
python rag_ingest.py ingest
```

This will:
- Extract text/captions
- Chunk content
- Generate embeddings
- Build FAISS index (`faiss_index/index.bin` + `metadata.json`)

---

### 2. Search
```bash
python rag_ingest.py search "What are the key points?"
```

Searches the Index directly and returns JSON of matching chunks.

---

### 3. Chat with your docs
```bash
python rag_ingest.py chat "Summarize the findings of the report"
```

Searches the FAISS and uses Gemini to generate context-aware answers.

---

### 4. Streamlit UI (optional)
Launch a simple UI to upload files and chat:

```bash
streamlit run app.py
```

---

## 📌 Notes

- **Gemini fallback is always enabled** if `GOOGLE_API_KEY` is set.  
  OCR runs first, then Gemini is used for non-textual images.  
- **Replace mode**: Each ingestion run **replaces** the old FAISS index with a new one.  
  (No accumulation of old docs unless you modify the pipeline).  
- **Keep Images**: By default, extracted images are deleted after captioning.  
  You can adjust in `process_documents()` with `keep_images=True`.

---

## 🧪 Testing

- `test_embed.py` → Tests embedding API connectivity.  
- `test_gemma.py` → Tests Gemini API connection.

---

## 🛡️ Requirements

- Python 3.10+  
- FAISS (`faiss-cpu`)  
- Playwright (`playwright`)  
- Tesseract OCR (optional, for image text extraction)

---

## 📖 Example Workflow

```bash
# Step 1: Put "GOIReport1.pdf" into ./documents
# Step 2: Ingest
python rag_ingest.py ingest

# Step 3: Ask a question
python rag_ingest.py chat "What does the report say about fiscal deficit?"
```

---

## 🔮 Roadmap

- [ ] Add support for incremental indexing  
- [ ] Improve HTML iframe/chart captioning  
- [ ] Add multiple LLM backends for chat  
- [ ] Expand UI with history + multi-doc chat  

---

## 🧑‍💻 Author
Built by **Vibhanshu Ray** with ❤️ — combining OCR, Gemini, and FAISS for a powerful local RAG pipeline.
