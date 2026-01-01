"""
rag_ingest.py
Single-file RAG ingestion pipeline:
- Extract PDF -> markdown using pymupdf4llm (write_images=True)
- For each image, try OCR (pytesseract) to produce a caption
- If OCR fails, fallback to Gemini multimodal (if --use_gemini is enabled and GOOGLE_API_KEY is set OR we have made the gemini enabled by default if Goofle API key is set)
- Replace image links with "**Image:** <caption>"
- Chunk text, get embeddings via embedding endpoint (nomic-embed-text)
- Build FAISS index and save metadata JSON
- Simple search + chat with Gemini CLI
"""

import os
import re
import json
import base64
import asyncio
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

# Optional JSON repair for lenient parsing of embedded configs
try:
    from json_repair import repair_json  # type: ignore
except Exception:
    repair_json = None

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

# Configure Gemini API key (compat with env var names)
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

# Resolve model name via models.json key 'gemini'
try:
    from agentLoop.model_manager import MODELS_JSON
    _models_cfg = json.loads(Path(MODELS_JSON).read_text())
    GEMINI_MODEL_NAME = _models_cfg["models"]["gemini"]["model"]
except Exception:
    # Fallback to a reasonable default
    GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Ensure folders
DOCS_DIR.mkdir(parents=True, exist_ok=True)
(IMAGES_DIR := DOCS_DIR / IMAGES_DIR_NAME).mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Gemini Chat ----------------
def execute_data_query(query: str, data_files: List[Path]) -> str:
    """Execute actual data queries on CSV/Excel files for numerical aggregations."""
    try:
        import pandas as pd
        import re
        
        # Try to find CSV or Excel files
        csv_files = [f for f in data_files if f.suffix.lower() == '.csv']
        excel_files = [f for f in data_files if f.suffix.lower() in ['.xlsx', '.xls']]
        
        # Prefer CSV for performance
        data_file = csv_files[0] if csv_files else (excel_files[0] if excel_files else None)
        if not data_file:
            return None
        
        # Load the data
        if data_file.suffix.lower() == '.csv':
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
        
        # Detect query type and parameters
        query_lower = query.lower()
        
        # Extract aggregation type
        agg_type = None
        if any(w in query_lower for w in ['sum', 'total', 'add']):
            agg_type = 'sum'
        elif any(w in query_lower for w in ['average', 'mean', 'avg']):
            agg_type = 'mean'
        elif any(w in query_lower for w in ['count', 'number of', 'how many']):
            agg_type = 'count'
        elif any(w in query_lower for w in ['max', 'maximum', 'highest', 'largest']):
            agg_type = 'max'
        elif any(w in query_lower for w in ['min', 'minimum', 'lowest', 'smallest']):
            agg_type = 'min'
        
        if not agg_type:
            return None
        
        # Try to find the column to aggregate using intelligent word matching
        # This approach is generic and works for any column names
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        target_col = None
        
        # Extract all meaningful words from the query
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        # Remove common stop words that don't help with column matching
        stop_words = {'the', 'is', 'of', 'for', 'in', 'what', 'give', 'me', 'get', 'show', 'find', 'calculate', 'data'}
        query_words = query_words - stop_words
        
        def normalize_word(word):
            """Normalize word to its base form for better matching."""
            # Remove trailing 's' for plural forms (simple heuristic)
            if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
            return word
        
        def words_match(word1, word2):
            """Check if two words match, considering variations."""
            # Exact match
            if word1 == word2:
                return True
            # Normalized match (handles plurals)
            if normalize_word(word1) == normalize_word(word2):
                return True
            # Check if one word starts with the other (handles abbreviations)
            if len(word1) >= 3 and len(word2) >= 3:
                # Check if smaller word is prefix of larger word
                shorter, longer = (word1, word2) if len(word1) < len(word2) else (word2, word1)
                if longer.startswith(shorter):
                    return True
                # Check substring for very short abbreviations (3 chars)
                if len(shorter) == 3 and shorter in longer:
                    return True
            return False
        
        # Score each numeric column based on word overlap with query
        best_score = 0
        best_col = None
        
        for col in numeric_cols:
            score = 0
            col_lower = col.lower()
            
            # Split column name into words (handle underscores and camelCase)
            col_words = set(re.findall(r'[a-z]+', col_lower.replace('_', ' ')))
            
            # Calculate overlap between query words and column words using fuzzy matching
            for q_word in query_words:
                for c_word in col_words:
                    if words_match(q_word, c_word):
                        score += 15  # Higher score for word matches
            
            # Bonus if the entire column name appears in query or vice versa
            if col_lower in query_lower or col_lower.replace('_', ' ') in query_lower:
                score += 50
            
            # Additional bonus for each column word appearing in query (exact or fuzzy)
            for c_word in col_words:
                if len(c_word) > 2:
                    # Check exact presence
                    if c_word in query_lower:
                        score += 10
                    # Check normalized presence
                    elif normalize_word(c_word) in query_lower:
                        score += 8
                    # Check if any query word contains this column word
                    else:
                        for q_word in query_words:
                            if words_match(c_word, q_word):
                                score += 5
                                break
            
            # Track the best match
            if score > best_score:
                best_score = score
                best_col = col
        
        # Use the best matching column if we found any matches
        if best_col and best_score > 0:
            target_col = best_col
        elif numeric_cols:
            # Fallback to first numeric column if no good match found
            target_col = numeric_cols[0]
        else:
            return None
        
        # Try to find filter conditions (e.g., "for central region")
        filter_col = None
        filter_val = None
        
        # Look for common filter patterns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in query_lower or col_lower.replace('_', ' ') in query_lower:
                # Check if there's a value after the column name
                for val in df[col].dropna().unique():
                    if str(val).lower() in query_lower:
                        filter_col = col
                        filter_val = val
                        break
                if filter_col:
                    break
        
        # Apply filter if found
        if filter_col and filter_val is not None:
            filtered_df = df[df[filter_col] == filter_val]
            filter_desc = f"for {filter_col}={filter_val}"
        else:
            filtered_df = df
            filter_desc = "for all data"
        
        # Execute aggregation
        if agg_type == 'count':
            result = len(filtered_df)
        else:
            result = filtered_df[target_col].agg(agg_type)
        
        # Format response
        response = f"Based on the data in {data_file.name}:\n\n"
        response += f"The {agg_type} of '{target_col}' {filter_desc} is: **{result:,.2f}**\n\n"
        response += f"(Calculated from {len(filtered_df)} rows"
        if filter_col:
            response += f" where {filter_col} = '{filter_val}'"
        response += ")"
        
        return response
        
    except Exception as e:
        log(f"Error executing data query: {e}")
        return None


def chat_with_gemini(query: str, context_chunks: List[str], data_files: List[Path] = None) -> str:
    # First, try to execute as a data query if we have data files
    if data_files:
        direct_answer = execute_data_query(query, data_files)
        if direct_answer:
            return direct_answer
    
    # Fall back to LLM-based answer from context
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant.
Answer the question based only on the context below.
If the question requires numerical calculations (sum, average, count, etc.) on structured data,
perform the calculation accurately using all relevant data points shown in the context.

Context:
{context}

User question: {query}
"""
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# ---------------- utilities ----------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def read_text_flexible(path: Path) -> str:
    """Robustly read text from a file with encoding detection and safe fallbacks.

    Avoids hardcoding any single encoding by trying detection libraries first,
    then falling back to permissive decoding.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        pass

    try:
        raw = path.read_bytes()
    except Exception as e:
        return f"[Could not read file: {e}]"

    # Try charset_normalizer
    try:
        try:
            from charset_normalizer import from_bytes as _cn_from_bytes  # type: ignore
            best = _cn_from_bytes(raw).best()
            if best is not None:
                return str(best)
        except Exception:
            pass
        # Try chardet
        try:
            import chardet  # type: ignore
            det = chardet.detect(raw)
            enc = det.get("encoding")
            if enc:
                return raw.decode(enc, errors="replace")
        except Exception:
            pass
    except Exception:
        pass

    # Final fallback: utf-8 with replacement
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        # As a last resort, latin1 rarely fails
        return raw.decode("latin1", errors="replace")


def extract_plotly_summary(html_text: str) -> str:
    """Extract a generic textual summary from embedded Plotly configs in HTML.

    - Searches for Plotly.newPlot(data, layout) patterns
    - Attempts lenient JSON repair without assuming exact schema
    - Produces readable text describing title, axes, and trace names/types
    """
    try:
        mds: list[str] = []
        # More flexible pattern for minified code - use non-greedy matching
        pattern = re.compile(
            r"Plotly\.newPlot\s*\(\s*[^,]+,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*[,\)]", 
            re.DOTALL
        )
        for match in pattern.finditer(html_text or ""):
            raw_data, raw_layout = match.group(1), match.group(2)
            def _to_json(s: str):
                try:
                    return json.loads(s)
                except Exception:
                    if repair_json is not None:
                        try:
                            return json.loads(repair_json(s))
                        except Exception:
                            return None
                    return None
            data = _to_json(raw_data) or []
            layout = _to_json(raw_layout) or {}
            parts = []
            # Title
            try:
                title = layout.get('title')
                if isinstance(title, dict):
                    title_text = title.get('text')
                else:
                    title_text = title
                if isinstance(title_text, str) and title_text.strip():
                    parts.append(f"Title: {title_text.strip()}")
            except Exception:
                pass
            # Axes
            try:
                xt = layout.get('xaxis', {}).get('title')
                yt = layout.get('yaxis', {}).get('title')
                if isinstance(xt, dict): xt = xt.get('text')
                if isinstance(yt, dict): yt = yt.get('text')
                if xt: parts.append(f"X-axis: {str(xt)}")
                if yt: parts.append(f"Y-axis: {str(yt)}")
            except Exception:
                pass
            # Traces
            try:
                if isinstance(data, list):
                    for i, tr in enumerate(data):
                        if not isinstance(tr, dict):
                            continue
                        ttype = tr.get('type') or (tr.get('marker') or {}).get('type')
                        name = tr.get('name')
                        # Estimate points
                        npts = None
                        for key in ('y','x','values','z'):
                            seq = tr.get(key)
                            if isinstance(seq, list):
                                npts = len(seq)
                                break
                        desc = [f"Trace {i+1}"]
                        if ttype: desc.append(f"type={ttype}")
                        if name: desc.append(f"name={name}")
                        if npts is not None: desc.append(f"points={npts}")
                        parts.append(", ".join(desc))
            except Exception:
                pass
            if parts:
                mds.append("\n".join(parts))
        return "\n\n".join(mds)
    except Exception:
        return ""


def extract_plotly_from_json_scripts(html_text: str) -> str:
    """Parse script tags with JSON payloads that include Plotly data/layout.

    Looks for <script type="application/json|application/vnd.plotly.v1+json"> blocks
    and attempts to read keys like 'data' and 'layout' without relying on
    a fixed schema. Returns a human-readable summary string.
    """
    try:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html_text or "", "html.parser")
        summaries: list[str] = []
        for sc in soup.find_all('script'):
            t = (sc.get('type') or '').lower()
            if 'json' not in t:
                # Also try script tags with data embedded as text
                if t and 'javascript' not in t:
                    continue
            raw = sc.string or sc.get_text() or ''
            raw = raw.strip()
            if not raw:
                continue
            # Some blocks are JSON strings; others might be JS objects
            candidates = [raw]
            if repair_json is not None:
                try:
                    candidates.append(repair_json(raw))
                except Exception:
                    pass
            parsed = None
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    break
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue
            data = parsed.get('data') or parsed.get('figure', {}).get('data')
            layout = parsed.get('layout') or parsed.get('figure', {}).get('layout')
            if not data and not layout:
                continue
            # Build summary
            parts = []
            try:
                title = None
                if isinstance(layout, dict):
                    ttl = layout.get('title')
                    if isinstance(ttl, dict):
                        title = ttl.get('text')
                    elif isinstance(ttl, str):
                        title = ttl
                if title:
                    parts.append(f"Title: {title}")
            except Exception:
                pass
            try:
                if isinstance(layout, dict):
                    xt = layout.get('xaxis', {}).get('title')
                    yt = layout.get('yaxis', {}).get('title')
                    if isinstance(xt, dict): xt = xt.get('text')
                    if isinstance(yt, dict): yt = yt.get('text')
                    if xt: parts.append(f"X-axis: {str(xt)}")
                    if yt: parts.append(f"Y-axis: {str(yt)}")
            except Exception:
                pass
            try:
                if isinstance(data, list):
                    for i, tr in enumerate(data):
                        if not isinstance(tr, dict):
                            continue
                        ttype = tr.get('type')
                        name = tr.get('name')
                        npts = None
                        for key in ('y','x','values','z'):
                            seq = tr.get(key)
                            if isinstance(seq, list):
                                npts = len(seq)
                                break
                        desc = [f"Trace {i+1}"]
                        if ttype: desc.append(f"type={ttype}")
                        if name: desc.append(f"name={name}")
                        if npts is not None: desc.append(f"points={npts}")
                        parts.append(", ".join(desc))
            except Exception:
                pass
            if parts:
                summaries.append("\n".join(parts))
        return "\n\n".join(summaries)
    except Exception:
        return ""


def extract_plotly_from_data_attrs(html_text: str) -> str:
    """Extract Plotly figure summaries from elements with data-plotly attributes."""
    try:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html_text or "", "html.parser")
        summaries: list[str] = []
        for el in soup.find_all(True):
            dp = el.get('data-plotly') or el.get('data-figure') or el.get('data-plotly-config')
            if not dp:
                continue
            raw = (dp or '').strip()
            if not raw:
                continue
            candidates = [raw]
            if repair_json is not None:
                try:
                    candidates.append(repair_json(raw))
                except Exception:
                    pass
            parsed = None
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    break
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue
            data = parsed.get('data') or parsed.get('figure', {}).get('data')
            layout = parsed.get('layout') or parsed.get('figure', {}).get('layout')
            parts = []
            try:
                ttl = layout.get('title') if isinstance(layout, dict) else None
                if isinstance(ttl, dict): ttl = ttl.get('text')
                if isinstance(ttl, str) and ttl.strip():
                    parts.append(f"Title: {ttl.strip()}")
            except Exception:
                pass
            try:
                if isinstance(layout, dict):
                    xt = layout.get('xaxis', {}).get('title')
                    yt = layout.get('yaxis', {}).get('title')
                    if isinstance(xt, dict): xt = xt.get('text')
                    if isinstance(yt, dict): yt = yt.get('text')
                    if xt: parts.append(f"X-axis: {str(xt)}")
                    if yt: parts.append(f"Y-axis: {str(yt)}")
            except Exception:
                pass
            try:
                if isinstance(data, list):
                    for i, tr in enumerate(data):
                        if not isinstance(tr, dict):
                            continue
                        ttype = tr.get('type')
                        name = tr.get('name')
                        npts = None
                        for key in ('y','x','values','z'):
                            seq = tr.get(key)
                            if isinstance(seq, list):
                                npts = len(seq)
                                break
                        desc = [f"Trace {i+1}"]
                        if ttype: desc.append(f"type={ttype}")
                        if name: desc.append(f"name={name}")
                        if npts is not None: desc.append(f"points={npts}")
                        parts.append(", ".join(desc))
            except Exception:
                pass
            if parts:
                summaries.append("\n".join(parts))
        return "\n\n".join(summaries)
    except Exception:
        return ""


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
    # if use_gemini and os.getenv("GOOGLE_API_KEY"):   ##Enable it to specifically use gemini for image with no text in it
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):                    ##If Gemini API key is configure, then the solution will use it.
        try:
            file_obj = genai.upload_file(path=str(image_path))
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
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
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved FAISS index -> {idx_path} and metadata -> {meta_path}")


# ---- 7) Ingest ----
from bs4 import BeautifulSoup   # pip install beautifulsoup4

def extract_plotly_from_data_attrs(html_text: str) -> str:
    """Extract Plotly data from data-* attributes in HTML."""
    try:
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html_text or "", "html.parser")
        parts = []
        
        # Look for elements with data-plotly or data-chart attributes
        for el in soup.find_all(attrs=lambda x: x and any(k.startswith(('data-plotly', 'data-chart')) for k in x)):
            for attr, val in (el.attrs or {}).items():
                if attr.lower().startswith(('data-plotly', 'data-chart')):
                    # Try to parse as JSON
                    try:
                        data = json.loads(val)
                        if isinstance(data, dict):
                            if 'title' in data:
                                parts.append(f"Title: {data['title']}")
                            if 'data' in data and isinstance(data['data'], list):
                                for trace in data['data']:
                                    if isinstance(trace, dict) and 'name' in trace:
                                        parts.append(f"Series: {trace['name']}")
                    except:
                        # If not JSON, just add as text if meaningful
                        if len(val) < 200 and val.strip():
                            parts.append(val.strip())
        
        return "\n".join(parts)
    except Exception:
        return ""


def extract_plotly_from_large_html(html_text: str) -> str:
    """Extract Plotly data from very large HTML files by searching for key patterns."""
    try:
        parts = []
        
        # Search for data/layout objects in script tags or Plotly calls
        # Look for patterns that indicate Plotly chart configuration
        
        # Pattern 1: Look for "title":{...} or title:"..." in layout
        title_patterns = [
            r'"title"\s*:\s*\{\s*"text"\s*:\s*"([^"]+)"',
            r'"title"\s*:\s*"([^"]+)"',
            r'title\s*:\s*\{\s*text\s*:\s*"([^"]+)"',
            r'title\s*:\s*"([^"]+)"'
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, html_text[:500000])  # Search first 500KB
            if matches:
                parts.append(f"Title: {matches[0]}")
                break
        
        # Pattern 2: Look for axis titles
        axis_patterns = [
            (r'"xaxis"\s*:\s*\{[^}]*"title"\s*:\s*(?:"([^"]+)"|\{"text"\s*:\s*"([^"]+)")', "X-axis"),
            (r'"yaxis"\s*:\s*\{[^}]*"title"\s*:\s*(?:"([^"]+)"|\{"text"\s*:\s*"([^"]+)")', "Y-axis"),
            (r'xaxis\s*:\s*\{[^}]*title\s*:\s*(?:"([^"]+)"|\{text\s*:\s*"([^"]+)")', "X-axis"),
            (r'yaxis\s*:\s*\{[^}]*title\s*:\s*(?:"([^"]+)"|\{text\s*:\s*"([^"]+)")', "Y-axis")
        ]
        
        for pattern, label in axis_patterns:
            matches = re.findall(pattern, html_text[:500000])
            if matches:
                # matches is a tuple, take the non-empty one
                axis_title = matches[0][0] if matches[0][0] else matches[0][1] if len(matches[0]) > 1 else None
                if axis_title:
                    parts.append(f"{label}: {axis_title}")
        
        # Pattern 3: Look for trace names
        trace_patterns = [
            r'"name"\s*:\s*"([^"]+)"',
            r'name\s*:\s*"([^"]+)"'
        ]
        
        trace_names = set()
        for pattern in trace_patterns:
            matches = re.findall(pattern, html_text[:500000])
            trace_names.update(matches[:10])  # Limit to first 10 unique names
        
        if trace_names:
            parts.append("Data series: " + ", ".join(trace_names))
        
        # Pattern 4: Look for chart type
        type_patterns = [
            r'"type"\s*:\s*"([^"]+)"',
            r'type\s*:\s*"([^"]+)"'
        ]
        
        chart_types = set()
        for pattern in type_patterns:
            matches = re.findall(pattern, html_text[:100000])
            chart_types.update(m for m in matches if m in ['bar', 'scatter', 'line', 'pie', 'heatmap', 'box', 'histogram', 'surface', 'mesh3d'])
        
        if chart_types:
            parts.append("Chart types: " + ", ".join(chart_types))
        
        return "\n".join(parts) if parts else ""
        
    except Exception as e:
        log(f"Error extracting Plotly from large HTML: {e}")
        return ""


def process_documents(rebuild_index: bool = True, use_gemini: bool = False, keep_images: bool = False):
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    json_files = sorted(DOCS_DIR.glob("*.json"))
    html_files = sorted(DOCS_DIR.glob("*.html"))
    csv_files = sorted(DOCS_DIR.glob("*.csv"))
    xlsx_files = []
    for ext in ["*.xlsx", "*.xls"]:
        xlsx_files.extend(sorted(DOCS_DIR.glob(ext)))
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg"]:
        image_files.extend(sorted(DOCS_DIR.glob(ext)))

    if (not pdf_files and not json_files and not html_files and not image_files
        and not csv_files and not xlsx_files):
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

    # --- CSV ---
    for csv_path in csv_files:
        log(f"Processing CSV {csv_path.name}")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # Convert to markdown-like rows without hardcoding columns
            header = " | ".join(map(str, df.columns))
            lines = [header]
            for _, row in df.iterrows():
                vals = ["" if pd.isna(v) else str(v) for v in row.tolist()]
                lines.append(" | ".join(vals))
            md = "\n".join(lines)
        except Exception as e:
            log(f"Failed to read CSV {csv_path.name}: {e}")
            continue
        for i, chunk in enumerate(chunk_text(md)):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": csv_path.name, "chunk_id": f"{csv_path.stem}_{i}", "chunk": chunk[:2000]})
            except Exception as e:
                log(f"Failed to embed chunk {i} of {csv_path.name}: {e}")

    # --- XLS/XLSX ---
    for xl_path in xlsx_files:
        log(f"Processing Excel {xl_path.name}")
        try:
            import pandas as pd
            # Read first sheet by default; avoid hardcoding-specific sheet names
            df = pd.read_excel(xl_path)
            header = " | ".join(map(str, df.columns))
            lines = [header]
            for _, row in df.iterrows():
                vals = ["" if pd.isna(v) else str(v) for v in row.tolist()]
                lines.append(" | ".join(vals))
            md = "\n".join(lines)
        except Exception as e:
            log(f"Failed to read Excel {xl_path.name}: {e}")
            continue
        for i, chunk in enumerate(chunk_text(md)):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"doc": xl_path.name, "chunk_id": f"{xl_path.stem}_{i}", "chunk": chunk[:2000]})
            except Exception as e:
                log(f"Failed to embed chunk {i} of {xl_path.name}: {e}")

    # # --- HTML ---
    for html in html_files:
        log(f"Processing HTML {html.name}")
        md = ""
        try:
            # Robust reading to avoid decode errors
            html_text = read_text_flexible(html)
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style"]): tag.extract()

            # Collect diverse textual signals without hardcoding page structure
            parts = []

            # Document title
            try:
                if soup.title and soup.title.string:
                    parts.append(soup.title.string.strip())
            except Exception:
                pass

            # Meta tags with content
            try:
                for meta in soup.find_all('meta'):
                    content = (meta.get('content') or '').strip()
                    if content:
                        parts.append(content)
            except Exception:
                pass

            # Headings h1..h6
            try:
                for h in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
                    txt = h.get_text(separator=" ", strip=True)
                    if txt:
                        parts.append(txt)
            except Exception:
                pass

            # Visible text
            try:
                visible_text = soup.get_text(separator="\n", strip=True)
                if visible_text:
                    parts.append(visible_text)
            except Exception:
                pass

            # Image alt attributes
            try:
                alts = [img.get("alt", "").strip() for img in soup.find_all("img") if img.get("alt")]
                if alts:
                    parts.append("\n".join(a for a in alts if a))
            except Exception:
                pass

            # aria-label attributes
            try:
                aria = [el.get("aria-label", "").strip() for el in soup.find_all(attrs={"aria-label": True}) if el.get("aria-label")]
                if aria:
                    parts.append("\n".join(a for a in aria if a))
            except Exception:
                pass

            # title attributes on elements
            try:
                etitles = [el.get("title", "").strip() for el in soup.find_all(attrs={"title": True}) if el.get("title")]
                if etitles:
                    parts.append("\n".join(t for t in etitles if t))
            except Exception:
                pass

            # Generic data-* attributes
            try:
                for el in soup.find_all(True):
                    for attr, val in (el.attrs or {}).items():
                        if isinstance(attr, str) and attr.lower().startswith("data-"):
                            sval = " ".join(val) if isinstance(val, list) else str(val)
                            sval = sval.strip()
                            if sval:
                                parts.append(sval)
            except Exception:
                pass

            md = "\n".join(p for p in parts if p)

            # Try extracting tables if general text is sparse
            if not md.strip():
                try:
                    tables = []
                    for table in soup.find_all('table'):
                        rows = []
                        for tr in table.find_all('tr'):
                            cells = [c.get_text(" ", strip=True) for c in tr.find_all(['td','th'])]
                            if cells:
                                rows.append(" | ".join(cells))
                        if rows:
                            tables.append("\n".join(rows))
                    if tables:
                        md = "\n\n".join(tables)
                except Exception:
                    pass

            # Try extracting Plotly figure metadata from scripts and data-* attributes
            if not md.strip():
                try:
                    # For very large HTML files, use specialized extraction
                    if len(html_text) > 500000:  # > 500KB
                        log(f"Using large file extraction for {html.name} ({len(html_text)} bytes)")
                        plotly_md = extract_plotly_from_large_html(html_text)
                        if plotly_md.strip():
                            md = plotly_md
                    else:
                        plotly_md = extract_plotly_summary(html_text)
                        if not plotly_md.strip():
                            plotly_md = extract_plotly_from_json_scripts(html_text)
                        if not plotly_md.strip():
                            plotly_md = extract_plotly_from_data_attrs(html_text)
                        if plotly_md.strip():
                            md = plotly_md
                except Exception as e:
                    log(f"Error extracting Plotly metadata: {e}")
                    pass

            # As a last resort, attempt screenshot only when not inside asyncio loop
            if not md.strip():
                inside_loop = False
                try:
                    asyncio.get_running_loop()
                    inside_loop = True
                except Exception:
                    pass
                if not inside_loop:
                    import tempfile
                    from playwright.sync_api import sync_playwright
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
                        # continue with empty md
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
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata

def search(query: str, k: int = TOP_K, index_dir: Path = None):
    """Search the FAISS index and return relevant chunks."""
    if index_dir is None:
        index_dir = FAISS_DIR
    
    q_emb = get_embedding(query).reshape(1, -1)
    index, metadata = load_index_and_metadata(index_dir)
    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        md = metadata[idx]
        results.append({"score": float(dist), "doc": md.get("doc"), "chunk_id": md.get("chunk_id"), "chunk": md.get("chunk")})
    return results


def search_and_answer(query: str, k: int = TOP_K, index_dir: Path = None, docs_dir: Path = None):
    """Search and generate an answer, with direct data query execution for numerical queries."""
    if index_dir is None:
        index_dir = FAISS_DIR
    if docs_dir is None:
        docs_dir = DOCS_DIR
    
    # Get search results
    results = search(query, k, index_dir)
    context_chunks = [r["chunk"] for r in results]
    
    # Find data files in the docs directory
    data_files = []
    if docs_dir.exists():
        data_files.extend(docs_dir.glob("*.csv"))
        data_files.extend(docs_dir.glob("*.xlsx"))
        data_files.extend(docs_dir.glob("*.xls"))
    
    # Generate answer
    answer = chat_with_gemini(query, context_chunks, data_files)
    return answer


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="RAG ingest/search/chat")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Process docs and build FAISS index") ##Commenting this line and below as use_gemini was commented out above on line 124 and use_gemini ain't required
    # ing.add_argument("--use_gemini", action="store_true", help="Enable Gemini fallback for image captioning")

    s = sub.add_parser("search", help="Search the FAISS index")
    s.add_argument("q", nargs="+", help="Query text to search")

    c = sub.add_parser("chat", help="Chat with your docs using Gemini")
    c.add_argument("q", nargs="+", help="User question")

    args = parser.parse_args()

    if args.cmd == "ingest":
        # process_documents(rebuild_index=True, use_gemini=args.use_gemini). ##Commenting as use_gemini was commented out above on line 124 and use_gemini ain't required
        process_documents(rebuild_index=True)

    elif args.cmd == "search":
        query = " ".join(args.q)
        results = search(query)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.cmd == "chat":
        query = " ".join(args.q)
        answer = search_and_answer(query)
        print("\n=== Gemini Answer ===\n")
        print(answer)


if __name__ == "__main__":
    main()