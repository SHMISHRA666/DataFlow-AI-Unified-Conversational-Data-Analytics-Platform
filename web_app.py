from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import numpy as np
import asyncio
from pathlib import Path
import traceback
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# Local imports (we are inside the DataFlow project)
from agentLoop.conversation_planner_agent import ConversationPlannerAgent

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
USERS_FILE = BASE_DIR / 'users.json'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'html', 'htm', 'pdf'}

# Ensure process working directory is project root so relative prompt paths work
try:
    os.chdir(str(BASE_DIR))
except Exception:
    pass

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / 'css').mkdir(parents=True, exist_ok=True)
(STATIC_DIR / 'js').mkdir(parents=True, exist_ok=True)
if not USERS_FILE.exists():
    USERS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2))

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = 'dev-secret-key'  # Replace in production
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(file_path: str):
    """Return basic preview and metadata for supported files.
    For non-tabular (html/pdf), return minimal info instead of failing.
    """
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        elif file_ext in ['.xlsx', '.xls']:
            engine = 'xlrd' if file_ext == '.xls' else 'openpyxl'
            df = pd.read_excel(file_path, nrows=5, engine=engine)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
            if len(df) > 5:
                df = df.head(5)
        elif file_ext in ['.html', '.htm', '.pdf']:
            return {
                'columns': [],
                'preview': [],
                'shape': [0, 0],
                'dtypes': {}
            }
        else:
            return None

        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {
            'columns': list(df.columns),
            'preview': df.to_dict('records'),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'dtypes': dtypes_dict
        }
    except Exception as e:
        print(f"Error in get_file_info: {e}")
        return {
            'columns': [],
            'preview': [],
            'shape': [0, 0],
            'dtypes': {}
        }


def load_users() -> dict:
    try:
        return json.loads(USERS_FILE.read_text() or '{}')
    except Exception:
        return {}


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2))


def is_api_request() -> bool:
    # Treat JSON or specific endpoints as API
    return request.is_json or request.path in ['/upload', '/ask']


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            if is_api_request():
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return wrapper


def run_async_safely(coro):
    """Run an async coroutine safely from a sync Flask route."""
    try:
        loop = asyncio.get_running_loop()
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        return asyncio.run(coro)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('user_id'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    # POST
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


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', next=request.args.get('next', '/'))
    # POST
    username = (request.form.get('username') or '').strip()
    password = (request.form.get('password') or '').strip()
    next_url = request.form.get('next') or '/'
    users = load_users()
    user = users.get(username)
    if not user or not check_password_hash(user.get('password_hash', ''), password):
        return render_template('login.html', error='Invalid credentials', next=next_url)
    session['user_id'] = username
    return redirect(next_url)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV, JSON, Excel, HTML, or PDF files.'}), 400

        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = UPLOAD_FOLDER / unique_filename
        file.save(str(file_path))

        # File info preview (best-effort)
        file_info = get_file_info(str(file_path)) or {
            'columns': [], 'preview': [], 'shape': [0, 0], 'dtypes': {}
        }

        # Persist a session manifest
        session_id = str(uuid.uuid4())
        session_data = {
            'file_path': str(file_path),
            'original_filename': filename,
            'upload_time': datetime.now().isoformat(),
            'file_info': file_info
        }
        (RESULTS_FOLDER / f'{session_id}.json').write_text(json.dumps(session_data, ensure_ascii=False))

        return jsonify({
            'success': True,
            'session_id': session_id,
            'file_info': file_info,
            'original_filename': filename
        })
    except Exception as e:
        print('Error in /upload:', e)
        print(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        data = request.get_json(force=True)
        session_id = data.get('session_id')
        user_question = (data.get('question') or '').strip()

        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        if not user_question:
            return jsonify({'error': 'Question is required'}), 400

        manifest_path = RESULTS_FOLDER / f'{session_id}.json'
        if not manifest_path.exists():
            return jsonify({'error': 'Session not found'}), 404

        session_data = json.loads(manifest_path.read_text())
        file_path = session_data['file_path']
        file_manifest = [{
            'path': file_path,
            'name': Path(file_path).name,
            'size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
        }]

        async def _classify():
            # Allow overriding model via env to align with current ModelManager/Agent setup
            model_name = os.environ.get('DATAFLOW_MODEL_NAME')
            if model_name:
                agent = ConversationPlannerAgent(model_name=model_name)
            else:
                agent = ConversationPlannerAgent()
            return await agent.run_classification({
                'user_query': user_question,
                'files': file_manifest
            })

        result = run_async_safely(_classify())
        if 'error' in result:
            print(f"Error in classification: {result['error']}")
        return jsonify({'success': True, 'classification': result.get('output', {}), 'user_query': user_question, 'raw': result})
    except Exception as e:
        print('Error in /ask:', e)
        print(traceback.format_exc())
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500


@app.route('/capabilities')
def get_capabilities():
    capabilities = {
        'quantitative': {
            'name': 'Quantitative Analysis',
            'description': 'Statistical analysis using data processing agents',
            'features': [
                'Data cleaning and transformation',
                'Trend analysis',
                'Top performers analysis',
                'Distribution analysis',
                'Automated reporting'
            ]
        },
        'qualitative': {
            'name': 'Qualitative Analysis',
            'description': 'Text and pattern analysis for insights',
            'features': [
                'Text analysis',
                'Pattern recognition',
                'Data quality assessment',
                'Insight generation',
                'Data overview'
            ]
        }
    }
    return jsonify(capabilities)


# ---------------------------
# History Endpoints (per user)
# ---------------------------
@app.route('/history', methods=['GET'])
@login_required
def get_history():
    users = load_users()
    uid = session.get('user_id')
    history = users.get(uid, {}).get('history', [])
    return jsonify({'success': True, 'history': history})


@app.route('/history', methods=['POST'])
@login_required
def add_history():
    try:
        payload = request.get_json(force=True)
        question = (payload.get('question') or '').strip()
        if not question:
            return jsonify({'error': 'question required'}), 400
        entry = {
            'question': question,
            'timestamp': datetime.now().isoformat()
        }
        users = load_users()
        uid = session.get('user_id')
        user = users.get(uid) or {}
        hist = user.get('history') or []
        hist.insert(0, entry)
        user['history'] = hist[:100]  # cap
        users[uid] = user
        save_users(users)
        return jsonify({'success': True})
    except Exception as e:
        print('Error in /history POST:', e)
        print(traceback.format_exc())
        return jsonify({'error': 'failed to save history'}), 500


if __name__ == '__main__':
    # Run the app (dev mode)
    app.run(debug=True, host='0.0.0.0', port=5000)