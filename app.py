import streamlit as st
from pathlib import Path
import shutil
from rag_ingest import process_documents, search, chat_with_gemini, DOCS_DIR

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📄 RAG Document Assistant")

# --- Upload documents ---
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, HTML, JSON, or Images",
    type=["pdf", "html", "json", "png", "jpg", "jpeg", "svg"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        save_path = DOCS_DIR / file.name
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        st.sidebar.success(f"Saved {file.name}")

    if st.sidebar.button("📥 Ingest Documents"):
        with st.spinner("Building FAISS index..."):
            process_documents(rebuild_index=True, use_gemini=True, keep_images=False)
        st.sidebar.success("Ingestion complete!")

# --- Chat with docs ---
st.subheader("💬 Chat with your documents")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching and generating answer..."):
        results = search(query)
        context_chunks = [r["chunk"] for r in results]
        answer = chat_with_gemini(query, context_chunks)

    st.markdown("### 🔍 Retrieved Chunks")
    for r in results:
        with st.expander(f"{r['doc']} (score={r['score']:.2f})"):
            st.write(r["chunk"])

    st.markdown("### 🤖 Gemini Answer")
    st.success(answer)

