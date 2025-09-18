"""
rag_ingest.py
Single-file RAG ingestion pipeline:
- Extract PDF -> markdown using pymupdf4llm (write_images=True)
- For each image, try OCR (pytesseract) to produce a caption
- If OCR fails, fallback to Gemini multimodal (if --use_gemini is enabled and GOOGLE_API_KEY is set)
- Replace image links with "**Image:** <caption>"
- Chunk text, get embeddings via embedding endpoint (nomic-embed-text)
- Build FAISS index and save metadata JSON
- Simple search + chat with Gemini CLI
"""

import os
import re
import json
import base64
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import google.generativeai as genai
import requests
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# --- Optional imports ---
try:
    import faiss
except Exception as e:
    raise ImportError("faiss not installed. On mac use 'conda install -c conda-forge faiss-cpu' or 'pip install faiss-cpu'") from e

try:
    import pymupdf4llm
except Exception as e:
    raise ImportError("pymupdf4llm is required. pip install pymupdf4llm") from e

# ---- Config ----
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./documents"))
IMAGES_DIR_NAME = "images"
FAISS_DIR = Path(os.getenv("FAISS_DIR", "./faiss_index"))

EMBED_API_URL = os.getenv("EMBED_API_URL", "http://localhost:11434/api/embeddings")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "256"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure folders
DOCS_DIR.mkdir(parents=True, exist_ok=True)
(IMAGES_DIR := DOCS_DIR / IMAGES_DIR_NAME).mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Gemini Chat ----------------
def chat_with_gemini(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant.
Answer the question based only on the context below.

Context:
{context}

User question: {query}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# ---------------- utilities ----------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---- 1) PDF -> markdown (images written) ----
def pdf_to_markdown(pdf_path: Path, images_outdir: Path) -> str:
    log(f"Extracting PDF: {pdf_path}")
    md = pymupdf4llm.to_markdown(str(pdf_path), write_images=True, image_path=str(images_outdir))
    return md or ""


# ---- 2) Caption image (OCR -> Gemini fallback) ----
def caption_image(image_path: Path, use_gemini: bool = False) -> str:
    log(f"Captioning image: {image_path.name}")
    try:
        raw = image_path.read_bytes()
    except Exception as e:
        return f"[Could not read image {image_path.name}: {e}]"

    # ---- OCR with preprocessing ----
    def ocr_with_preprocessing(img_path: Path) -> str:
        try:
            from PIL import Image, ImageOps, ImageEnhance
            import pytesseract

            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.point(lambda x: 0 if x < 128 else 255, "1")

            text = pytesseract.image_to_string(img).strip()
            if text:
                return " ".join(text.split())
        except Exception as e:
            log(f"OCR preprocessing failed: {e}")
        return ""

    # 1) Try OCR first
    text = ocr_with_preprocessing(image_path)
    if text:
        log(f"OCR text extracted (len={len(text)}).")
        return text

    # 2) Gemini fallback
    if use_gemini and os.getenv("GOOGLE_API_KEY"):
        try:
            file_obj = genai.upload_file(path=str(image_path))
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content([
                "If there is text in this image, transcribe it exactly. "
                "Otherwise, describe the image in one concise sentence.",
                file_obj
            ])
            return response.text.strip()
        except Exception as e:
            log(f"Gemini captioning failed: {e}")

    return f"[Could not caption image {image_path.name}]"


# ---- 3) Replace image markdown with caption ----
IMAGE_MD_RE = re.compile(r'!\[.*?\]\((.*?)\)')

def replace_images_with_captions(markdown: str, images_dir: Path) -> str:
    def _repl(match):
        src = match.group(1)
        src_path = (DOCS_DIR / src) if not Path(src).is_absolute() else Path(src)
        if not src_path.exists():
            candidate = images_dir / Path(src).name
            if candidate.exists():
                src_path = candidate
        if not src_path.exists():
            log(f"Image not found for captioning: {src}")
            return f"[Image not found: {src}]"

        try:
            caption = caption_image(src_path)
            return f"**Image:** {caption}"
        except Exception as e:
            log(f"Warning: failed to caption {src}: {e}")
            return f"[Image could not be processed: {src}]"

    return IMAGE_MD_RE.sub(_repl, markdown)


# ---- 4) Chunking ----
def chunk_text(text: str, size_words: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if size_words <= 0:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + size_words]
        chunks.append(" ".join(chunk_words))
        i += size_words - overlap
    return chunks


# ---- 5) Get embedding ----
def get_embedding(text: str) -> np.ndarray:
    payload = {"model": EMBED_MODEL, "prompt": text}
    headers = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {EMBED_API_KEY}"

    r = requests.post(EMBED_API_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    emb = None
    if isinstance(data, dict):
        emb = data.get("embedding")
        if emb is None and "data" in data and isinstance(data["data"], list):
            emb = data["data"][0].get("embedding")
    if emb is None:
        raise RuntimeError(f"Embedding response did not include an embedding: {data}")
    return np.array(emb, dtype=np.float32)


# ---- 6) Build FAISS ----
def build_and_save_faiss(embeddings: List[np.ndarray], metadata: List[Dict[str, Any]], out_dir: Path):
    if not embeddings:
        raise ValueError("No embeddings to index")
    dim = embeddings[0].shape[0]
    mat = np.stack(embeddings).astype(np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    idx_path = out_dir / "index.bin"
    meta_path = out_dir / "metadata.json"
    faiss.write_index(index, str(idx_path))
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    log(f"Saved FAISS index -> {idx_path} and metadata -> {meta_path}")


# ---- 7) Ingest ----
from bs4 import BeautifulSoup   # pip install beautifulsoup4

def process_documents(rebuild_index: bool = True, use_gemini: bool = False, keep_images: bool = False):
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    json_files = sorted(DOCS_DIR.glob("*.json"))
    html_files = sorted(DOCS_DIR.glob("*.html"))
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg"]:
        image_files.extend(sorted(DOCS_DIR.glob(ext)))

    if not pdf_files and not json_files and not html_files and not image_files:
        log("No supported files found in documents/.")
        return

    embeddings, metadata = [], []

    # --- PDFs ---
    for pdf in pdf_files:
        log(f"Processing PDF {pdf.name}")
        md = pdf_to_markdown(pdf, IMAGES_DIR)
        if not md.strip():
            continue
        md = replace_images_with_captions(md, IMAGES_DIR)
        for i, chunk in enumerate(chunk_text(md)):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": pdf.name, "chunk_id": f"{pdf.stem}_{i}", "chunk": chunk[:2000]})
            except Exception as e:
                log(f"Failed to embed chunk {i} of {pdf.name}: {e}")

    # --- JSON ---
    for js in json_files:
        log(f"Processing JSON {js.name}")
        try:
            data = json.loads(js.read_text())
            md = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"Failed to parse {js.name}: {e}")
            continue
        for i, chunk in enumerate(chunk_text(md)):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": js.name, "chunk_id": f"{js.stem}_{i}", "chunk": chunk[:2000]})
            except Exception as e:
                log(f"Failed to embed chunk {i} of {js.name}: {e}")

    # # --- HTML ---
    import tempfile
    from playwright.sync_api import sync_playwright
    for html in html_files:
        log(f"Processing HTML {html.name}")
        md = ""
        try:
            soup = BeautifulSoup(html.read_text(), "html.parser")
            for tag in soup(["script", "style"]): tag.extract()
            md = soup.get_text(separator="\n", strip=True)
            if not md.strip():
                with tempfile.NamedTemporaryFile(suffix=".png", dir=str(IMAGES_DIR), delete=False) as tmp_img:
                    tmp_img_path = Path(tmp_img.name)
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch()
                        page = browser.new_page()
                        page.goto(f"file://{html.resolve()}")
                        page.screenshot(path=str(tmp_img_path), full_page=True)
                        browser.close()
                    caption = caption_image(tmp_img_path, use_gemini=use_gemini)
                    md = caption
                    if not keep_images: tmp_img_path.unlink(missing_ok=True)
                except Exception as e:
                    log(f"Screenshot failed: {e}")
                    continue
        except Exception as e:
            log(f"Failed to parse {html.name}: {e}")
            continue
        if not md.strip():
            continue
        for i, chunk in enumerate(chunk_text(md)):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": html.name, "chunk_id": f"{html.stem}_{i}", "chunk": chunk[:2000]})
            except Exception as e:
                log(f"Failed to embed chunk {i} of {html.name}: {e}")

    
    # --- Images ---
    for img in image_files:
        log(f"Processing Image {img.name}")
        try:
            caption = caption_image(img, use_gemini=use_gemini)
            if not caption or caption.startswith("[Could not caption"):
                continue
            for i, chunk in enumerate(chunk_text(caption)):
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": img.name, "chunk_id": f"{img.stem}_{i}", "chunk": chunk[:2000]})
            if not keep_images:
                img.unlink(missing_ok=True)
        except Exception as e:
            log(f"Failed to process image {img.name}: {e}")

    if embeddings:
        build_and_save_faiss(embeddings, metadata, FAISS_DIR)
    else:
        log("No embeddings were created.")


# ---- 8) Search ----
def load_index_and_metadata(index_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    idx_path = index_dir / "index.bin"
    meta_path = index_dir / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found. Run ingest first.")
    index = faiss.read_index(str(idx_path))
    metadata = json.loads(meta_path.read_text())
    return index, metadata

def search(query: str, k: int = TOP_K):
    q_emb = get_embedding(query).reshape(1, -1)
    index, metadata = load_index_and_metadata(FAISS_DIR)
    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        md = metadata[idx]
        results.append({"score": float(dist), "doc": md.get("doc"), "chunk_id": md.get("chunk_id"), "chunk": md.get("chunk")})
    return results


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="RAG ingest/search/chat")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Process docs and build FAISS index")
    ing.add_argument("--use_gemini", action="store_true", help="Enable Gemini fallback for image captioning")

    s = sub.add_parser("search", help="Search the FAISS index")
    s.add_argument("q", nargs="+", help="Query text to search")

    c = sub.add_parser("chat", help="Chat with your docs using Gemini")
    c.add_argument("q", nargs="+", help="User question")

    args = parser.parse_args()

    if args.cmd == "ingest":
        process_documents(rebuild_index=True, use_gemini=args.use_gemini)

    elif args.cmd == "search":
        query = " ".join(args.q)
        results = search(query)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.cmd == "chat":
        query = " ".join(args.q)
        results = search(query)
        context_chunks = [r["chunk"] for r in results]
        answer = chat_with_gemini(query, context_chunks)
        print("\n=== Gemini Answer ===\n")
        print(answer)


if __name__ == "__main__":
    main()