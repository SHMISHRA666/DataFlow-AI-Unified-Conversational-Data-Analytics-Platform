# import streamlit as st
# from pathlib import Path
# import shutil
# from rag_ingest import process_documents, search, chat_with_gemini, DOCS_DIR

# st.set_page_config(page_title="RAG Assistant", layout="wide")

# st.title("📄 RAG Document Assistant")

# # --- Upload documents ---
# st.sidebar.header("Upload Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDF, HTML, JSON, or Images",
#     type=["pdf", "html", "json", "png", "jpg", "jpeg", "svg"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     for file in uploaded_files:
#         save_path = DOCS_DIR / file.name
#         with open(save_path, "wb") as f:
#             f.write(file.getbuffer())
#         st.sidebar.success(f"Saved {file.name}")

#     if st.sidebar.button("📥 Ingest Documents"):
#         with st.spinner("Building FAISS index..."):
#             process_documents(rebuild_index=True, use_gemini=True, keep_images=False)
#         st.sidebar.success("Ingestion complete!")

# # --- Chat with docs ---
# st.subheader("💬 Chat with your documents")

# query = st.text_input("Enter your question:")

# if query:
#     with st.spinner("Searching and generating answer..."):
#         results = search(query)
#         context_chunks = [r["chunk"] for r in results]
#         answer = chat_with_gemini(query, context_chunks)

#     st.markdown("### 🔍 Retrieved Chunks")
#     for r in results:
#         with st.expander(f"{r['doc']} (score={r['score']:.2f})"):
#             st.write(r["chunk"])

#     st.markdown("### 🤖 Gemini Answer")
#     st.success(answer)


import streamlit as st
import os
import shutil
from pathlib import Path
from rag_ingest import process_documents, search, chat_with_gemini, DOCS_DIR

st.set_page_config(page_title="RAG UI", layout="wide")

st.title("📄 RAG Document QA (Replace Mode)")

# Sidebar upload
st.sidebar.header("📂 Upload & Ingest Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, HTML, JSON, or Images",
    type=["pdf", "html", "json", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button("📥 Ingest Documents"):
    if uploaded_files:
        with st.spinner("Clearing old docs and index..."):
            # remove old docs and index
            shutil.rmtree(DOCS_DIR, ignore_errors=True)
            shutil.rmtree("faiss_index", ignore_errors=True)

            # recreate dirs
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            (DOCS_DIR / "images").mkdir(parents=True, exist_ok=True)

            # save new uploads
            for file in uploaded_files:
                save_path = DOCS_DIR / file.name
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

        with st.spinner("Building FAISS index..."):
            process_documents(rebuild_index=True, use_gemini=True, keep_images=False)
        st.sidebar.success("✅ Ingestion complete!")
    else:
        st.sidebar.warning("Please upload at least one document before ingestion.")

# Chat/Q&A
st.markdown("### 💬 Ask Questions About Your Documents")
user_query = st.text_input("Enter your question here:")

if st.button("🔎 Get Answer") and user_query:
    with st.spinner("Searching index..."):
        results = search(user_query)
        context_chunks = [r["chunk"] for r in results]

    with st.spinner("Generating answer with Gemini..."):
        answer = chat_with_gemini(user_query, context_chunks)

    st.markdown("### 🤖 Gemini Answer")
    st.write(answer)