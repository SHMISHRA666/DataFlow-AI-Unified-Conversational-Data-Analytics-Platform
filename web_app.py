from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import traceback
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import requests  # <-- Ensure this import is present
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
USERS_FILE = BASE_DIR / 'users.json'
GENERATED_CHARTS_DIR = BASE_DIR / 'generated_charts' # For serving artifacts
HISTORY_FILE = BASE_DIR / 'history.json'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'html', 'htm', 'pdf'}

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
GENERATED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
if not USERS_FILE.exists():
    USERS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2))
if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text(json.dumps([], ensure_ascii=False, indent=2))

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Define the URL of our backend API endpoint
BACKEND_API_URL = "http://13.234.11.1:5001/run_agent_loop"
# BACKEND_API_URL = "http://127.0.0.1:5001/run_agent_loop"

# --- NEW: Cache-Busting ---
# This dictionary will hold a timestamp to append to static files
# forcing the browser to reload them on server restart.
app.config['CACHE_BUSTER'] = {'v': str(int(datetime.now().timestamp()))}

@app.context_processor
def inject_cache_buster():
    """Injects the cache-buster version into all templates."""
    return dict(cache_version=app.config['CACHE_BUSTER']['v'])
# --- End of Cache-Busting ---


# --- Helper Functions ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_users():
    return json.loads(USERS_FILE.read_text() or '{}')

def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2))

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return wrapper

# --- Auth Routes (No changes) ---
@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('user_id'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', next=request.args.get('next', '/'))
    username = (request.form.get('username') or '').strip()
    password = (request.form.get('password') or '').strip()
    next_url = request.form.get('next') or '/'
    users = load_users()
    user = users.get(username)
    if not user or not check_password_hash(user.get('password_hash', ''), password):
        return render_template('login.html', error='Invalid credentials', next=next_url)
    session['user_id'] = username
    return redirect(next_url)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    username = (request.form.get('username') or '').strip()
    password = (request.form.get('password') or '').strip()
    if not username or not password:
        return render_template('register.html', error='Username and password are required')
    users = load_users()
    if username in users:
        return render_template('register.html', error='Username already exists')
    users[username] = {
        'password_hash': generate_password_hash(password),
        'created_at': datetime.now().isoformat()
    }
    save_users(users)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# --- Core App Routes ---
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or no file selected'}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = UPLOAD_FOLDER / unique_filename
        file.save(str(file_path))
        
        session_id = str(uuid.uuid4())
        session_data = {
            'file_path': str(file_path),
            'original_filename': filename,
            'upload_time': datetime.now().isoformat(),
        }
        # Save the session manifest
        (RESULTS_FOLDER / f'{session_id}.json').write_text(json.dumps(session_data, ensure_ascii=False))
        
        file_info = None
        try:
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext in ['csv']:
                df = pd.read_csv(str(file_path))
                file_info = {
                    'columns': list(df.columns),
                    'shape': [int(df.shape[0]), int(df.shape[1])]
                }
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(str(file_path))
                file_info = {
                    'columns': list(df.columns),
                    'shape': [int(df.shape[0]), int(df.shape[1])]
                }
            elif ext in ['json']:
                try:
                    data_json = json.loads(Path(file_path).read_text(encoding='utf-8', errors='ignore'))
                    if isinstance(data_json, list) and data_json and isinstance(data_json[0], dict):
                        cols = list({k for item in data_json for k in item.keys()})
                        file_info = {
                            'columns': cols,
                            'shape': [len(data_json), len(cols)]
                        }
                    else:
                        file_info = {
                            'columns': [],
                            'shape': [0, 0]
                        }
                except Exception:
                    file_info = {
                        'columns': [],
                        'shape': [0, 0]
                    }
            else:
                file_info = {
                    'columns': [],
                    'shape': [0, 0]
                }
        except Exception:
            file_info = {
                'columns': [],
                'shape': [0, 0]
            }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'original_filename': filename,
            'file_info': file_info
        })
    except Exception as e:
        print(f'Error in /upload: {e}\n{traceback.format_exc()}')
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
@login_required
def ask_question():
    """
    This endpoint now acts as a proxy to the powerful backend.
    """
    try:
        data = request.get_json(force=True)
        session_id = data.get('session_id')
        user_question = (data.get('question') or '').strip()

        if not session_id or not user_question:
            return jsonify({'error': 'Session ID and question are required'}), 400

        print(f"🖥️ [Frontend] Forwarding request to backend for session: {session_id}")
        
        backend_payload = {
            "session_id": session_id,
            "question": user_question
        }
        
        # Make the request to the backend service
        response = requests.post(BACKEND_API_URL, json=backend_payload, timeout=900)
        response.raise_for_status()

        backend_data = response.json()
        return jsonify(backend_data)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with backend: {e}")
        return jsonify({'error': f'Could not connect to the agentic framework: {e}', 'success': False}), 502
    except Exception as e:
        print(f'Error in /ask: {e}\n{traceback.format_exc()}')
        return jsonify({'error': f'An unexpected error occurred: {str(e)}', 'success': False}), 500

@app.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    try:
        if request.method == 'GET':
            items = json.loads(HISTORY_FILE.read_text() or '[]')
            return jsonify({'success': True, 'history': items})
        data = request.get_json(force=True) or {}
        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({'success': False, 'error': 'question required'}), 400
        items = json.loads(HISTORY_FILE.read_text() or '[]')
        items.append({'question': question, 'timestamp': datetime.now().isoformat()})
        HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/generated_charts/<path:session_id>/<path:filename>')
@login_required
def serve_artifact(session_id, filename):
    """Serves files from the generated_charts directory."""
    return send_from_directory(GENERATED_CHARTS_DIR / session_id, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

