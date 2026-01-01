# backend_app.py - The Agentic Framework Server

import asyncio
import json
import os
import sys
import traceback # Import traceback
from pathlib import Path

from flask import Flask, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS  # <-- 1. IMPORT CORS

# Assuming agentLoop and utils are in the same directory or Python path
from agentLoop.flow import AgentLoop4
from utils.utils import log_error, log_step, load_file_type_config

# --- START: Supporting Code Integrated from main_old.py ---

# (All the helper functions like _get_outputs_dir, discover_session_artifacts, etc.
# remain unchanged here. They are omitted for brevity but are still in your file.)
# ... (omitting identical helper functions for clarity) ...
def _get_outputs_dir() -> Path:
    """Gets the base directory for generated artifacts."""
    return Path(os.environ.get("OUTPUTS_DIR", "generated_charts")).resolve()

def _project_root() -> Path:
    """Gets the project's root directory."""
    return Path(__file__).resolve().parent

def _to_project_relative(path: Path) -> str:
    """Converts an absolute path to a path relative to the project root."""
    try:
        rel = path.resolve().relative_to(_project_root().resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()

def _build_public_url(relative_path: str, base_url: str) -> str | None:
    """Constructs a full public URL for an artifact."""
    base = (base_url or "").strip()
    if not base:
        return None
    return f"{base.rstrip('/')}/{relative_path.lstrip('/')}"

def discover_session_artifacts(session_id: str, public_base_url: str = "") -> dict:
    """
    Inspects the per-session outputs directory and lists artifacts in a structured way.
    """
    out_dir = _get_outputs_dir() / str(session_id)
    artifacts = {
        "session_dir": None, "exists": False,
        "files": {
            "html": [], "png": [], "svg": [], "pdf": [],
            "yaml": [], "json": [], "other": []
        },
        "preferred_entry": None
    }
    if not out_dir.is_dir():
        return artifacts

    artifacts["exists"] = True
    artifacts["session_dir"] = _to_project_relative(out_dir)
    
    for child in out_dir.rglob("*"):
        if child.is_file():
            rel = _to_project_relative(child)
            # Normalize to the path segment under generated_charts for public URL building
            try:
                rel_under = rel.split("generated_charts/", 1)[1]
            except Exception:
                rel_under = rel
            entry = {"relative": rel}
            # Build public URL like /static/generated_charts/<session>/<file>
            pub = _build_public_url(rel_under, public_base_url)
            if pub:
                entry["public_url"] = pub
            
            suffix = child.suffix.lower().lstrip('.')
            if suffix in artifacts["files"]:
                artifacts["files"][suffix].append(entry)
            else:
                artifacts["files"]["other"].append(entry)
    
    preferred_names = ["plotly_index.html", "report.html"] 
    for name in preferred_names:
        candidate = out_dir / name
        if candidate.exists():
            rel = _to_project_relative(candidate)
            try:
                rel_under = rel.split("generated_charts/", 1)[1]
            except Exception:
                rel_under = rel
            artifacts["preferred_entry"] = {"relative": rel}
            pub = _build_public_url(rel_under, public_base_url)
            if pub:
                artifacts["preferred_entry"]["public_url"] = pub
            break
            
    return artifacts

def extract_final_answer_from_context(execution_context, public_base_url: str = "") -> dict:
    """
    Extracts the final answer and all related metadata from the agent execution context.
    """
    try:
        graph = execution_context.plan_graph.get('graph', {})
        session_id = graph.get('session_id')
        final_answer_text = graph.get('rag_answer')
        
        if not final_answer_text:
             try:
                rag_output = execution_context.get_output('rag_processing')
                if rag_output and isinstance(rag_output, dict):
                    final_answer_text = rag_output.get('answer')
             except Exception:
                 pass

        output_directory = str(_get_outputs_dir() / str(session_id)) if session_id else None
        artifacts = discover_session_artifacts(session_id, public_base_url) if session_id else {}
        
        # Build classification block from conversation_plan if not explicitly present
        classification = graph.get('classification')
        try:
            if not classification:
                conv = graph.get('conversation_plan') or {}
                if isinstance(conv, dict):
                    classification = {
                        'user_query': conv.get('user_query'),
                        'primary_classification': conv.get('primary_classification') or conv.get('primary'),
                        'secondary_classification': conv.get('secondary_classification') or conv.get('secondary') or 'None'
                    }
        except Exception:
            classification = classification or None

        payload = {
            "session_id": session_id,
            "output_directory": output_directory,
            "final_answer_text": final_answer_text,
            "artifacts": artifacts,
            "success": True,
            "classification": classification
        }
        return payload
        
    except Exception as e:
        log_error(f"Failed to extract final answer from context: {e}")
        return {"success": False, "error": str(e), "final_answer_text": None, "session_id": None}

# --- END: Supporting Code ---


# --- Flask App Initialization ---
app = Flask(__name__)
load_dotenv()
CORS(app)  # <-- 2. INITIALIZE CORS (This is the fix)

# Set project root as current working directory so agent relative paths work
os.chdir(str(_project_root()))

# Instantiate the main AgentLoop.
log_step("🚀 Initializing DataFlow AI AgentLoop...")
try:
    agent_loop = AgentLoop4(None)
    log_step("✅ DataFlow AI AgentLoop is ready.")
except Exception as e:
    log_error(f"FATAL: Failed to initialize AgentLoop4. {e}")
    agent_loop = None # Set to None so we can handle it
# ---------------------------------

@app.route('/run_agent_loop', methods=['POST'])
def run_agent_loop():
    """
    API endpoint to run the full agentic framework process.
    """
    # <-- 3. ADD A CHECK
    if agent_loop is None:
        log_error("AgentLoop is not initialized. Cannot process request.")
        return jsonify({"error": "AgentLoop is not initialized. Check backend logs.", "success": False}), 500
        
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('question')
    
    if not session_id or not query:
        return jsonify({"error": "session_id and question are required"}), 400

    results_folder = Path(__file__).resolve().parent / 'results'
    manifest_path = results_folder / f'{session_id}.json'

    if not manifest_path.exists():
        return jsonify({"error": f"Session manifest not found for session_id: {session_id}"}), 404

    try:
        session_data = json.loads(manifest_path.read_text())
        file_path = session_data.get('file_path')
        
        if not file_path or not Path(file_path).exists():
             return jsonify({"error": f"File path not found or invalid: {file_path}"}), 400

        uploaded_files = [file_path] 
        file_manifest = [{"path": file_path, "name": Path(file_path).name, "size": Path(file_path).stat().st_size}]
        
        log_step(f"🚀 [Backend] Received request for session {session_id}")
        
        # Call signature: run(query, file_manifest, uploaded_files)
        # Run the async agent in a fresh event loop (Flask view is sync)
        execution_context = asyncio.run(agent_loop.run(query, file_manifest, uploaded_files))
        
        frontend_base_url = request.host_url.replace('5001', '5000')
        public_artifact_url_base = f"{frontend_base_url.rstrip('/')}/static/generated_charts"
        
        ui_payload = extract_final_answer_from_context(execution_context, public_artifact_url_base)
        
        log_step(f"✅ [Backend] Processing complete for session {session_id}")
        return jsonify(ui_payload)

    except Exception as e:
        log_error(f"An error occurred in /run_agent_loop: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    log_step(f"Starting Backend Agentic Framework Server on http://127.0.0.1:5001 (PID: {os.getpid()})")
    app.run(debug=True, port=5001)

