# main.py – 100% NetworkX Graph-First (FIXED MultiMCP)

from utils.utils import log_step, log_error, load_file_type_config
import asyncio
from dotenv import load_dotenv
from agentLoop.flow import AgentLoop4
from pathlib import Path
import sys
import os
import json

# Ensure UTF-8 console on Windows to avoid 'charmap' codec errors
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

BANNER = """
──────────────────────────────────────────────────────
🔸  Agentic Query Assistant  🔸
Files first, then your question.
Type 'exit' or 'quit' to leave.
──────────────────────────────────────────────────────
"""

"""
Centralized UI parameters
These values define what the frontend UI needs to know (without modifying UI files).
Other backend modules can import get_ui_parameters() from this file.
"""

def _load_models_config():
    try:
        config_path = Path(__file__).resolve().parent / "config" / "models.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "defaults": {
            "text_generation": "gemini",
            "embedding": "nomic"
        },
        "models": {}
    }


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _normalize_api_base(base: str) -> str:
    base = (base or "").strip()
    if not base:
        return ""
    if not base.startswith("/"):
        base = "/" + base
    return base.rstrip("/")


def _join_url_paths(base: str, leaf: str) -> str:
    base_norm = _normalize_api_base(base)
    leaf_norm = (leaf or "").lstrip("/")
    return f"{base_norm}/{leaf_norm}" if base_norm else f"/{leaf_norm}"


def _get_allowed_extensions_from_config():
    try:
        cfg = load_file_type_config(str(Path(__file__).resolve().parent / "config" / "file_types.yaml"))
        exts = set()
        for key in ["fixed_quantitative", "fixed_qualitative", "flexible_types", "qualitative_rag_extensions"]:
            values = cfg.get(key) or set()
            for ext in values:
                try:
                    exts.add(ext.lstrip("."))
                except Exception:
                    pass
        # If config is empty, fallback to safe defaults
        if not exts:
            exts = {"csv", "json", "xlsx", "xls", "html", "htm", "pdf"}
        return sorted(exts)
    except Exception:
        return ["csv", "json", "xlsx", "xls", "html", "htm", "pdf"]


def _parse_dataset_options_env():
    """Parse dataset options from UI_DATASET_OPTIONS env.

    Supported formats:
    - JSON list of objects: [{"value":"sales","label":"Sales Data"}, ...]
    - Comma-separated pairs: "sales:Sales Data,patents:Patents Data"
    """
    raw = os.environ.get("UI_DATASET_OPTIONS")
    options = []
    if not raw:
        return options
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("value") and item.get("label"):
                    options.append({"value": str(item["value"]), "label": str(item["label"])})
            return options
    except Exception:
        pass
    # Fallback: comma-separated pairs value:label
    try:
        pairs = [p for p in (raw.split(",") if raw else []) if p.strip()]
        for p in pairs:
            if ":" in p:
                val, lab = p.split(":", 1)
                val = val.strip()
                lab = lab.strip() or val
                if val:
                    options.append({"value": val, "label": lab})
    except Exception:
        options = []
    return options


def get_ui_parameters():
    """Return a dictionary of parameters intended for the frontend UI.

    Note: This centralizes values only; UI files are unchanged. Backend
    routes can import and expose these values as needed without duplicating
    constants in multiple places.
    """
    models_cfg = _load_models_config()
    default_text_model = (models_cfg.get("defaults", {}) or {}).get("text_generation", "gemini")
    default_embed_model = (models_cfg.get("defaults", {}) or {}).get("embedding", "nomic")

    # Allow override via env for current active text model (used by ConversationPlannerAgent)
    env_text_model = os.environ.get("DATAFLOW_MODEL_NAME")

    # Upload constraints from config and env
    allowed_extensions = _get_allowed_extensions_from_config()
    max_upload_mb = _get_int_env("MAX_UPLOAD_MB", 16)

    # Endpoints with optional base prefix (e.g., /api or /v1)
    api_base = os.environ.get("API_BASE_PATH", "")

    # Dataset options from env (UI not wired yet). Always include "upload".
    builtin_enabled = _get_bool_env("FEATURE_BUILTIN_DATASETS_ENABLED", False)
    dataset_options = _parse_dataset_options_env() if builtin_enabled else []
    upload_option = {"value": "upload", "label": "Upload your own data"}
    # Ensure upload is last and unique
    filtered = [o for o in dataset_options if o.get("value") != "upload"]
    dataset_options = filtered + [upload_option]

    # Feature flags
    history_enabled = _get_bool_env("FEATURE_HISTORY_ENABLED", True)

    params = {
        "app": {
            "name": os.environ.get("APP_NAME", "DataFlow AI"),
            "environment": os.environ.get("APP_ENV", os.environ.get("FLASK_ENV", "development")),
        },
        "endpoints": {
            "base": _normalize_api_base(api_base),
            "upload": _join_url_paths(api_base, "upload"),
            "ask": _join_url_paths(api_base, "ask"),
            "history": _join_url_paths(api_base, "history"),
            "capabilities": _join_url_paths(api_base, "capabilities")
        },
        "upload": {
            "allowed_extensions": allowed_extensions,
            "max_upload_mb": max_upload_mb
        },
        "outputs": {
            "dir": os.environ.get("OUTPUTS_DIR", "generated_charts"),
            "public_base_url": os.environ.get("PUBLIC_BASE_URL", ""),
            "preferred_entries": ["plotly_index.html", "report.html"]
        },
        "datasets": {
            # UI can consume and render these later; default is always "upload"
            "options": dataset_options,
            "default": "upload"
        },
        "models": {
            "default_text_generation": env_text_model or default_text_model,
            "default_embedding": default_embed_model,
            "available": list((models_cfg.get("models") or {}).keys())
        },
        "features": {
            "history_enabled": history_enabled,
            "built_in_datasets_enabled": builtin_enabled
        },
        "ui_text": {
            "title": os.environ.get("UI_TITLE", "DataFlow"),
            "welcome_message": os.environ.get("UI_WELCOME", "Welcome to DataFlow AI")
        }
    }
    return params


def get_ui_parameters_json(indent: int = 2) -> str:
    """Return UI parameters as JSON string (for easy templating or API)."""
    return json.dumps(get_ui_parameters(), ensure_ascii=False, indent=indent)


# -------------------------------------------------------------
# UI Result helpers (general; UI/backend can adopt later)
# -------------------------------------------------------------
def _get_outputs_dir() -> Path:
    return Path(os.environ.get("OUTPUTS_DIR", "generated_charts")).resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _to_project_relative(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(_project_root().resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _build_public_url(relative_path: str) -> str | None:
    base = os.environ.get("PUBLIC_BASE_URL") or ""
    base = base.strip()
    if not base:
        return None
    return f"{base.rstrip('/')}/{relative_path.lstrip('/')}"


def discover_session_artifacts(session_id: str) -> dict:
    """Inspect the per-session outputs directory and list artifacts in a structured way.

    Returns a dict with relative paths (from project root) and optional public URLs.
    """
    out_dir = _get_outputs_dir() / str(session_id)
    artifacts: dict = {
        "session_dir": None,
        "exists": False,
        "files": {
            "html": [],
            "png": [],
            "svg": [],
            "pdf": [],
            "yaml": [],
            "json": [],
            "other": []
        }
    }

    if not out_dir.exists():
        return artifacts

    artifacts["exists"] = True
    artifacts["session_dir"] = _to_project_relative(out_dir)

    def add_file(p: Path):
        rel = _to_project_relative(p)
        entry = {"relative": rel}
        pub = _build_public_url(rel)
        if pub:
            entry["public_url"] = pub
        suffix = p.suffix.lower()
        if suffix == ".html":
            artifacts["files"]["html"].append(entry)
        elif suffix == ".png":
            artifacts["files"]["png"].append(entry)
        elif suffix == ".svg":
            artifacts["files"]["svg"].append(entry)
        elif suffix == ".pdf":
            artifacts["files"]["pdf"].append(entry)
        elif suffix == ".yaml":
            artifacts["files"]["yaml"].append(entry)
        elif suffix == ".json":
            artifacts["files"]["json"].append(entry)
        else:
            artifacts["files"]["other"].append(entry)

    # Scan top-level of session dir and known subdirs
    for child in out_dir.rglob("*"):
        if child.is_file():
            add_file(child)

    # Preferred entries from parameters
    params = get_ui_parameters()
    preferred_names = [n for n in (params.get("outputs", {}).get("preferred_entries") or []) if isinstance(n, str)]
    preferred_entry = None
    for name in preferred_names:
        candidate = out_dir / name
        if candidate.exists():
            rel = _to_project_relative(candidate)
            preferred_entry = {"relative": rel}
            pub = _build_public_url(rel)
            if pub:
                preferred_entry["public_url"] = pub
            break
    artifacts["preferred_entry"] = preferred_entry
    return artifacts


def build_ui_final_payload(session_id: str, rag_answer=None, extra: dict | None = None) -> dict:
    """Compose a general response payload to be sent to the UI later.

    - Includes the session artifacts with relative paths and optional public URLs
    - Optionally includes a RAG answer if provided
    - Accepts extra fields to merge in without hardcoding schema further
    """
    artifacts = discover_session_artifacts(session_id)
    payload = {
        "success": True,
        "session_id": session_id,
        "artifacts": artifacts,
        "rag": {
            "available": rag_answer is not None,
            "answer": rag_answer
        }
    }
    if extra and isinstance(extra, dict):
        try:
            payload.update(extra)
        except Exception:
            pass
    return payload

def get_file_input():
    """Get file paths from user"""
    log_step("📁 File Input (optional):", symbol="")
    print("Enter file paths (one per line), or press Enter to skip:")
    print("Example: /path/to/file.csv")
    print("Press Enter twice when done.")
    
    uploaded_files = []
    file_manifest = []
    
    while True:
        file_path = input("📄 File path: ").strip()
        if not file_path:
            break
        
        # Strip quotes from drag-and-drop paths
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        
        if Path(file_path).exists():
            uploaded_files.append(file_path)
            file_manifest.append({
                "path": file_path,
                "name": Path(file_path).name,
                "size": Path(file_path).stat().st_size
            })
            print(f"✅ Added: {Path(file_path).name}")
        else:
            print(f"❌ File not found: {file_path}")
    
    return uploaded_files, file_manifest

def get_user_query():
    """Get query from user"""
    log_step("📝 Your Question:", symbol="")
    return input().strip()

async def main():
    load_dotenv()
    print(BANNER)
    # Print centralized UI parameters at startup
    log_step("UI Parameters", get_ui_parameters(), symbol="📋")
    
    # Initialize AgentLoop4 without MCP layer
    log_step("🚀 Initializing DataFlow AI")
    agent_loop = AgentLoop4(None)
    
    while True:
        try:
            # Get file input first
            uploaded_files, file_manifest = get_file_input()
            
            # Get user query
            query = get_user_query()
            if query.lower() in ['exit', 'quit']:
                break
            
            # Process with AgentLoop4 - returns ExecutionContextManager object
            log_step("🔄 Processing with AgentLoop4...")
            execution_context = await agent_loop.run(query, file_manifest, uploaded_files)
            
            # Minimal completion message
            print("\n" + "="*60)
            print("✅ DataFlow AI completed.")
            print("="*60)
            
            print("\n😴 Agent Resting now")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            log_error(f"Error: {e}")
            print("Let's try again...")
        
        # Continue prompt
        cont = input("\nContinue? (press Enter) or type 'exit': ").strip()
        if cont.lower() in ['exit', 'quit']:
            break

    # No MCP shutdown required

if __name__ == "__main__":
    asyncio.run(main())
