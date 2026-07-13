# Removed eventlet monkey_patch to prevent deadlocks with C-extensions

import sys
import builtins
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Force all print statements to flush immediately to avoid buffering in standard terminals/IDE logs
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from backend.realtime_core import StressStreamProcessor
import numpy as np
import threading
import csv
import subprocess
import sys
from collections import deque
from werkzeug.utils import secure_filename
import tempfile
import cv2
import librosa
from backend.model import MultimodalStressDetector, safe_pickle_load

app = Flask(__name__)
# Secure CORS for production compatibility
FRONTEND_URLS = [
    os.environ.get('FRONTEND_URL', 'http://localhost:3000'),
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URLS}})
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25, max_http_buffer_size=100000000)

# Initialize global stream processor (will be injected after runtime_engine is ready)
stream_processor = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}
ALLOWED_SIGNAL_EXTENSIONS = {'csv', 'txt'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the Phase 7 RuntimeEngine
from backend.runtime.runtime_engine import RuntimeEngine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

runtime_engine = RuntimeEngine.from_registry()
# Keep 'model' for backward compatibility (extractors)
model = MultimodalStressDetector()

# Inject into stream processor
stream_processor = StressStreamProcessor(runtime_engine=runtime_engine)

# --- Load environment variables and configuration ---
import pickle
import time
import json
import urllib.request
import urllib.error
from backend.voice_worker import extract_voice_stress_indicators
from backend.score_buffer import score_buffer

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

def get_env_or_dotenv(key, default=''):
    value = os.getenv(key)
    if value:
        return value

    dotenv_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(dotenv_path):
        return default

    try:
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        return default

    return default

GEMINI_API_KEY = get_env_or_dotenv('GEMINI_API_KEY', '')
GEMINI_MODEL = get_env_or_dotenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Muse EEG stream tracking configuration
MUSE_DEFAULT_FILENAME = os.path.join(BASE_DIR, 'uploads', 'eeg_session.csv')
MUSE_SESSION_LOCK = threading.Lock()
MUSE_SESSION = {
    'process': None,
    'duration_seconds': 0,
    'file_path': MUSE_DEFAULT_FILENAME,
    'started_at': None,
    'completed': False,
    'prediction': None,
    'error': None,
}

MODEL_LOAD_ERRORS = {}

# --- Phase 8: Runtime Observability and Replay ---
from backend.monitoring.runtime_metrics import RuntimeMetrics
from backend.monitoring.drift_monitor import DriftMonitor
from backend.monitoring.golden_replay import GoldenReplay

runtime_metrics = RuntimeMetrics()
drift_monitor = DriftMonitor(window_size=1000)
golden_replay = GoldenReplay(runtime_engine)



# --- Chatbot Helper Functions ---
def local_chat_fallback(user_message, stress_level):
    query = (user_message or '').strip().lower()

    if 'what is stress' in query or (query.startswith('what is') and 'stress' in query):
        return (
            "Stress is your body and mind's response to pressure or challenge. "
            "Short-term stress can improve focus, but prolonged stress may affect sleep, mood, energy, and concentration. "
            "Try: slow breathing, brief movement, hydration, and task prioritization to regulate it."
        )

    if 'symptom' in query or 'sign' in query:
        return (
            "Common stress signs include muscle tension, fast heartbeat, racing thoughts, irritability, shallow breathing, "
            "and poor sleep. If symptoms persist or feel severe, consult a qualified health professional."
        )

    if 'sleep' in query:
        return (
            "For stress-related sleep issues: avoid screens 60 minutes before bed, keep room cool/dark, "
            "and do 2-3 minutes of slow exhale breathing before sleep."
        )

    guidance = {
        'High': "Try this now: 1) inhale for 4s, exhale for 6s for 5 rounds, 2) loosen shoulders/jaw, 3) take a 2-minute screen break.",
        'Moderate': "Try a quick reset: 1) 60 seconds of slow breathing, 2) drink water, 3) switch to one priority task for 10 minutes.",
        'Low': "You are doing well. Maintain momentum with a 1-minute posture check and short breaks every 45-60 minutes.",
    }

    baseline = guidance.get(stress_level, guidance['Moderate'])
    return (
        "I can help with stress support. "
        f"Current stress context: {stress_level}. "
        f"{baseline} You asked: '{user_message}'."
    )


def ask_gemini_stress_assistant(user_message, stress_level, stress_percentage):
    if not GEMINI_API_KEY:
        return local_chat_fallback(user_message, stress_level)

    prompt = (
        "You are a supportive stress-management assistant in a general stress monitoring app. "
        "Give concise, practical, non-medical advice. Do not diagnose. "
        "If user appears in crisis, suggest contacting local emergency services or a mental health professional. "
        f"Current detected stress level: {stress_level}. "
        f"Current detected stress percentage: {stress_percentage}. "
        f"User question: {user_message}"
    )

    payload = {
        'contents': [
            {
                'parts': [
                    {'text': prompt}
                ]
            }
        ],
        'generationConfig': {
            'temperature': 0.5,
            'maxOutputTokens': 800
        }
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    def _fetch():
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode('utf-8'))

    try:
        response_data = _fetch()

        candidates = response_data.get('candidates', [])
        if not candidates:
            return local_chat_fallback(user_message, stress_level)

        parts = candidates[0].get('content', {}).get('parts', [])
        reply = "\n".join(part.get('text', '') for part in parts).strip()
        return reply or local_chat_fallback(user_message, stress_level)
    except Exception as e:
        print(f"[Gemini API Error] {e}")
        return local_chat_fallback(user_message, stress_level)


# --- Muse LSL Helper Functions ---
def _normalize_header(value):
    return (value or '').strip().lower().replace('_', ' ')


def _read_muse_points(file_path, limit=240):
    if not file_path or not os.path.exists(file_path):
        return []

    points = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []

            header_map = {_normalize_header(h): h for h in reader.fieldnames}
            ts_key = header_map.get('timestamps') or header_map.get('timestamp')
            tp9_key = header_map.get('tp9')
            af7_key = header_map.get('af7')
            af8_key = header_map.get('af8')
            tp10_key = header_map.get('tp10')
            aux_key = header_map.get('right aux') or header_map.get('rightaux') or header_map.get('aux')

            if not all([ts_key, tp9_key, af7_key, af8_key, tp10_key, aux_key]):
                return []

            for row in reader:
                try:
                    points.append({
                        'timestamp': float(row[ts_key]),
                        'TP9': float(row[tp9_key]),
                        'AF7': float(row[af7_key]),
                        'AF8': float(row[af8_key]),
                        'TP10': float(row[tp10_key]),
                        'RightAUX': float(row[aux_key]),
                    })
                except (ValueError, TypeError, KeyError):
                    continue
    except Exception:
        return []

    if limit and len(points) > limit:
        return points[-limit:]
    return points


def _read_muse_arrays(file_path):
    points = _read_muse_points(file_path, limit=0)
    if not points:
        return np.array([]), np.array([])

    tp9 = np.array([p['TP9'] for p in points], dtype=float)
    af7 = np.array([p['AF7'] for p in points], dtype=float)
    af8 = np.array([p['AF8'] for p in points], dtype=float)
    tp10 = np.array([p['TP10'] for p in points], dtype=float)
    right_aux = np.array([p['RightAUX'] for p in points], dtype=float)

    eeg_array = np.concatenate([tp9, af7, af8, tp10])
    return eeg_array, right_aux


def _predict_from_muse_csv(file_path):
    eeg_array, gsr_array = _read_muse_arrays(file_path)
    if eeg_array.size == 0:
        return {
            'status': 'error',
            'message': 'No valid Muse channel values found in CSV.',
        }

    phys_features = model.extract_physiological_features(eeg_data=eeg_array, gsr_data=gsr_array)
    result = runtime_engine.predict_fused(physio=phys_features)
    if 'status' not in result and 'error' not in result:
        result['status'] = 'success'
    result['source'] = 'muse_stream'
    return result


def _refresh_muse_session_if_needed():
    with MUSE_SESSION_LOCK:
        proc = MUSE_SESSION.get('process')
        if proc is None:
            return

        if proc.poll() is None:
            return

        if MUSE_SESSION.get('completed'):
            return

        file_path = MUSE_SESSION.get('file_path')
        if not file_path or not os.path.exists(file_path):
            MUSE_SESSION['completed'] = True
            MUSE_SESSION['error'] = 'Recording finished but CSV file was not found.'
            return

        try:
            MUSE_SESSION['prediction'] = _predict_from_muse_csv(file_path)
            MUSE_SESSION['completed'] = True
        except Exception as exc:
            MUSE_SESSION['completed'] = True
            MUSE_SESSION['error'] = f'Failed to analyze Muse recording: {exc}'

def parse_numeric_csv_file(file_storage, signal_type='eeg'):
    """Extract numeric signal values from CSV/TXT, preferring channel columns and skipping timestamp-like fields."""
    from io import StringIO
    try:
        raw_text = file_storage.read().decode('utf-8', errors='ignore')
        file_storage.stream.seek(0)
    except Exception:
        return np.array([])

    rows = list(csv.reader(StringIO(raw_text)))
    if not rows:
        return np.array([])

    first_row = rows[0]

    def _is_numeric(token):
        try:
            float(token)
            return True
        except (ValueError, TypeError):
            return False

    has_header = any(cell and not _is_numeric(cell.strip()) for cell in first_row)
    headers = [cell.strip().lower() for cell in first_row] if has_header else []
    data_rows = rows[1:] if has_header else rows

    if not data_rows:
        return np.array([])

    num_cols = max(len(r) for r in data_rows)
    cols = [[] for _ in range(num_cols)]

    for row in data_rows:
        for idx in range(num_cols):
            token = row[idx].strip() if idx < len(row) else ''
            if not token:
                continue
            try:
                value = float(token)
                if np.isfinite(value):
                    cols[idx].append(value)
            except ValueError:
                continue

    if not any(cols):
        return np.array([])

    keep_col_indices = []
    for idx, values in enumerate(cols):
        if len(values) < 5:
            continue

        header = headers[idx] if idx < len(headers) else ''
        is_timestamp_header = any(word in header for word in ['time', 'timestamp', 'datetime'])
        if is_timestamp_header:
            continue

        arr = np.array(values, dtype=float)
        mostly_increasing = np.mean(np.diff(arr) >= 0) > 0.95 if len(arr) > 10 else False
        looks_like_epoch = np.nanmedian(np.abs(arr)) > 1e6

        if not header and mostly_increasing and looks_like_epoch:
            continue

        keep_col_indices.append(idx)

    if not keep_col_indices:
        flat = [value for values in cols for value in values]
        return np.array(flat, dtype=float)

    merged = []
    for idx in keep_col_indices:
        merged.extend(cols[idx])

    merged_arr = np.array(merged, dtype=float)
    if signal_type == 'eeg' and merged_arr.size > 0:
        p1, p99 = np.percentile(merged_arr, [1, 99])
        merged_arr = np.clip(merged_arr, p1, p99)

    return merged_arr

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/api/runtime/status', methods=['GET'])
def runtime_status():
    return jsonify(runtime_engine.status())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    engine_status = runtime_engine.status()
    is_ready = engine_status.get('ready', False)
    
    return jsonify({
        'status': 'ok' if is_ready else 'degraded',
        'models_loaded': {
            'face_expert': engine_status['models'].get('face', {}).get('loaded', False),
            'voice_expert': engine_status['models'].get('voice', {}).get('loaded', False),
            'physio_expert': engine_status['models'].get('physio', {}).get('loaded', False)
        },
        'explainability_engine': {'loaded': engine_status.get('explainability_bundle_loaded', False)},
        'load_errors': MODEL_LOAD_ERRORS,
        'server': 'eventlet'
    })


@app.route('/api/explainability/status', methods=['GET'])
def explainability_status():
    """Phase 6: Return explainability bundle version and modality coverage."""
    if not runtime_engine or not runtime_engine.expl_engine:
        return jsonify({'loaded': False, 'error': 'ExplainabilityEngine not initialized'}), 503
    return jsonify(runtime_engine.expl_engine.status())


@app.route('/api/multimodal/analyze', methods=['POST'])
def analyze_multimodal():
    """
    Multimodal stress analysis endpoint
    Accepts: image file, audio file, EEG data, GSR data
    """
    print("[HTTP] POST /api/multimodal/analyze - Starting multimodal analysis...")
    try:
        # Initialize feature holders
        facial_features = None
        voice_features = None
        phys_features = None
        
        # Process facial image if provided
        if 'face_image' in request.files:
            face_file = request.files['face_image']
            if face_file and allowed_file(face_file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = secure_filename(face_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_file.save(filepath)
                
                # Extract facial features
                facial_features, _ = model.extract_facial_features(filepath)
                
                # Clean up
                os.remove(filepath)
        
        # Process voice audio if provided
        if 'voice_audio' in request.files:
            audio_file = request.files['voice_audio']
            if audio_file and allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                filename = secure_filename(audio_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(filepath)
                
                # Extract voice features
                voice_features = model.extract_voice_features(filepath)
                
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)

                if voice_features is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to extract voice features. Ensure the audio is not silent and is in a valid format.'
                    }), 400
        
        # Process physiological data if provided
        eeg_data = request.form.get('eeg_data')
        gsr_data = request.form.get('gsr_data')
        
        eeg_array = None
        gsr_array = None

        if 'eeg_file' in request.files:
            eeg_file = request.files['eeg_file']
            if eeg_file and allowed_file(eeg_file.filename, ALLOWED_SIGNAL_EXTENSIONS):
                eeg_array = parse_numeric_csv_file(eeg_file, 'eeg')

        if 'gsr_file' in request.files:
            gsr_file = request.files['gsr_file']
            if gsr_file and allowed_file(gsr_file.filename, ALLOWED_SIGNAL_EXTENSIONS):
                gsr_array = parse_numeric_csv_file(gsr_file, 'gsr')

        if eeg_data and eeg_array is None:
            eeg_array = np.fromstring(eeg_data, sep=',')
            
        if gsr_data and gsr_array is None:
            gsr_array = np.fromstring(gsr_data, sep=',')

        if (eeg_array is not None and eeg_array.size > 0) or (gsr_array is not None and gsr_array.size > 0):
            phys_features = model.extract_physiological_features(eeg_data=eeg_array, gsr_data=gsr_array)
        
        # Check if at least one modality is provided
        if facial_features is None and voice_features is None and phys_features is None:
            return jsonify({
                'status': 'error',
                'message': 'Please provide at least one input (image, audio, or physiological data)'
            }), 400
        
        # Make prediction
        import time
        start_t = time.time()
        
        result = runtime_engine.predict_fused(
            face=facial_features,
            voice=voice_features,
            physio=phys_features
        )
        
        # Phase 8: Record telemetry
        latency = (time.time() - start_t) * 1000
        missing = []
        if facial_features is None: missing.append("face")
        if voice_features is None: missing.append("voice")
        if phys_features is None: missing.append("physio")
        
        runtime_metrics.record_prediction(
            latency_ms=latency,
            missing_modalities=missing,
            stress_probability=result.get("stress_probability", 0)
        )
        drift_monitor.record_features(
            face=facial_features,
            voice=voice_features,
            physio=phys_features
        )
        
        if 'error' in result:
            print(f"[HTTP] POST /api/multimodal/analyze - Failed: {result['error']}")
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
        
        print(f"[HTTP] POST /api/multimodal/analyze - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/runtime/reset', methods=['POST'])
def reset_runtime():
    """Reset the calibration baselines for a new user session."""
    try:
        runtime_engine.reset_calibration()
        return jsonify({
            'status': 'success',
            'message': 'Calibration baselines and temporal history reset.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/upload', methods=['POST'])
def analyze_face():
    """Facial stress analysis endpoint"""
    print("[HTTP] POST /api/face/upload - Starting facial analysis...")
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Please upload an image (PNG, JPG, JPEG)'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        facial_features, _ = model.extract_facial_features(filepath)
        result = runtime_engine.predict_face(raw_features=facial_features)
        
        # Clean up
        os.remove(filepath)
        
        if 'error' in result:
            print(f"[HTTP] POST /api/face/upload - Failed: {result['error']}")
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
            
        print(f"[HTTP] POST /api/face/upload - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/voice/upload', methods=['POST'])
def analyze_voice():
    """Voice stress analysis endpoint"""
    print("[HTTP] POST /api/voice/upload - Starting voice analysis...")
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Please upload an audio file (WAV, MP3)'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        voice_features = model.extract_voice_features(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

        if voice_features is None:
            return jsonify({
                'status': 'error',
                'message': 'Failed to extract voice features. Ensure the audio is not silent and is in a valid format.'
            }), 400
            
        result = runtime_engine.predict_voice(raw_features=voice_features)
        
        if 'error' in result:
            print(f"[HTTP] POST /api/voice/upload - Failed: {result['error']}")
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
            
        print(f"[HTTP] POST /api/voice/upload - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/voice/record', methods=['POST'])
def record_voice():
    """Voice recording endpoint (simulated)"""
    # This is a placeholder - actual recording would be done client-side
    return jsonify({
        'status': 'error',
        'message': 'Please use the upload feature instead of recording'
    }), 501

@app.route('/api/webcam/capture', methods=['POST'])
def capture_webcam():
    """Webcam capture endpoint"""
    try:
        # Get base64 image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_webcam.jpg')
        image.save(temp_path)
        
        # Extract features and predict
        facial_features, _ = model.extract_facial_features(temp_path)
        result = runtime_engine.predict_face(raw_features=facial_features)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# -----------------------------------------------------------------------------
# PHASE 1 REAL-TIME LIGHTWEIGHT MULTIMODAL STRESS DETECTION ENDPOINTS (SSE)
# -----------------------------------------------------------------------------

def fuse_predictions(probs, confs, certainties=None, fusion_mode='reliability'):
    """
    Phase 5: Real-time SSE fusion using Phase 4-validated optimal weights.
    Face=0.30, Voice=0.40, Physio=0.30. Re-normalised when some modalities are absent.
    """
    OPTIMAL_WEIGHTS = {'face': 0.30, 'voice': 0.40, 'physio': 0.30}
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
    
    active_modes = [m for m in active_modes if m in OPTIMAL_WEIGHTS]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
        
    if len(active_modes) == 1:
        score = probs[active_modes[0]]
        level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
        return {'fused_score': score, 'stress_level': level}
        
    # Re-normalise the active subset
    raw_w = {m: OPTIMAL_WEIGHTS[m] for m in active_modes}
    w_sum = sum(raw_w.values())
    norm_weights = {m: raw_w[m] / w_sum for m in active_modes}
        
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': {m: round(norm_weights[m], 3) for m in active_modes}
    }

# Rolling histories and locks for multi-user face score smoothing
_face_histories = {}
_face_hist_lock = threading.Lock()

def get_face_history(user_id):
    with _face_hist_lock:
        if user_id not in _face_histories:
            _face_histories[user_id] = deque(maxlen=4)
        return _face_histories[user_id]

def build_face_feature_vector(indicators):
    return np.array([
        indicators.get('left_ear', 0.3),
        indicators.get('right_ear', 0.3),
        indicators.get('avg_ear', 0.3),
        indicators.get('blink_velocity', 0.0),
        indicators.get('brow_descent_left', 0.1),
        indicators.get('brow_descent_right', 0.1),
        indicators.get('brow_asymmetry', 0.0),
        indicators.get('lip_compression', 0.2),
        indicators.get('jaw_displacement', 1.85),
        indicators.get('mouth_corner_pull', 0.3),
        indicators.get('forehead_tension', 0.1),
        indicators.get('face_height_norm', 1.5),
        indicators.get('head_tilt', 0.0),
        indicators.get('temporal_x_var', 0.0),
        indicators.get('temporal_y_var', 0.0),
        indicators.get('eye_openness_ratio', 0.3),
        indicators.get('landmark_confidence', 0.9),
        indicators.get('nose_wrinkle', 0.1),
    ], dtype=np.float32)

@app.route('/api/stream/face', methods=['POST'])
def stream_face():
    """
    Receive 18 browser-extracted facial indicators and update ScoreBuffer.
    """
    face_expert = runtime_engine._models.get('face')
    face_scaler = runtime_engine._scalers.get('face')
    if face_expert is None or face_scaler is None:
        return jsonify({'error': 'Face expert model not loaded'}), 500
        
    data = request.json or {}
    indicators = data.get('indicators', {})
    user_id = data.get('user_id', 'default')

    if not indicators:
        return jsonify({'score': None}), 200

    try:
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        landmark_conf = indicators.get('landmark_confidence', 1.0)
        # Gate 1: Landmark quality gate
        if landmark_conf < 0.5:
            return jsonify({
                'score': None,
                'reason': 'low_landmark_confidence',
                'confidence': landmark_conf
            })

        # Apply personal baseline normalization if complete and session scaler not used
        # We construct the raw feature vector first
        raw_vec = build_face_feature_vector(indicators)
        
        session_scaled = None
        if cal.is_complete:
            session_scaled = cal.scale_face_features(raw_vec)

        if session_scaled is not None:
            feature_scaled = session_scaled
        else:
            # Fall back to training scaler + personal baseline normalization
            indicators_norm = indicators.copy()
            if cal.is_complete:
                indicators_norm = cal.normalize_face_indicators(indicators)
                cal.add_face_sample(indicators_norm)  # continue updating baseline slowly

            norm_vec = raw_vec.copy()
            if cal.is_complete:
                means = face_scaler.mean_
                scales = face_scaler.scale_
                
                z_ear = indicators_norm.get('avg_ear_normalized', 0.0)
                norm_vec[0] = means[0] + z_ear * scales[0]
                norm_vec[1] = means[1] + z_ear * scales[1]
                norm_vec[2] = means[2] + z_ear * scales[2]
                norm_vec[15] = means[15] + z_ear * scales[15]
                
                z_brow_l = indicators_norm.get('brow_descent_left_normalized', 0.0)
                z_brow_r = indicators_norm.get('brow_descent_right_normalized', 0.0)
                norm_vec[4] = means[4] + z_brow_l * scales[4]
                norm_vec[5] = means[5] + z_brow_r * scales[5]
                
                z_jaw = indicators_norm.get('jaw_displacement_normalized', 0.0)
                norm_vec[8] = means[8] + z_jaw * scales[8]

            feature_scaled = face_scaler.transform(norm_vec.reshape(1, -1))

        raw_score = float(face_expert.predict_proba(feature_scaled)[0][1])

        # Gate 2: Certainty calculation
        certainty = abs(raw_score - 0.5) * 2  # 0.0 at boundary, 1.0 at extremes

        # Smile/laughter dampening
        smile_score = float(indicators.get('smile_score', 0.0))
        if smile_score > 0.3:
            dampening = smile_score * 0.4
            raw_score = max(0.0, raw_score - dampening)

        # Temporal smoothing: 4-sample rolling median per user
        hist = get_face_history(user_id)
        hist.append(raw_score)
        smoothed_score = float(np.median(list(hist)))

        # Write to buffer with certainty and confidence
        score_buffer.write('face', smoothed_score, {
            **indicators,
            'certainty': certainty,
            'landmark_confidence': landmark_conf,
        })
        print(f"[Face Expert] Processed frame. Score: {smoothed_score:.3f} (raw: {raw_score:.3f}, smile: {smile_score:.2f})")
        return jsonify({'score': smoothed_score, 'raw_score': raw_score, 'smile_detected': smile_score > 0.3})
    except Exception as e:
        print(f"Face streaming error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/voice', methods=['POST'])
def stream_voice():
    """
    Receive 2-second audio blob (WAV), extract 12 vocal biomarkers in eventlet's OS thread pool,
    and update ScoreBuffer.
    """
    voice_expert = runtime_engine._models.get('voice')
    voice_scaler = runtime_engine._scalers.get('voice')
    if voice_expert is None or voice_scaler is None:
        return jsonify({'error': 'Voice expert model not loaded'}), 500
        
    audio_bytes = request.data
    if not audio_bytes or len(audio_bytes) < 1000:
        return jsonify({'score': None, 'reason': 'too_short_or_empty'}), 200

    user_id = request.args.get('user_id', 'default')

    # Check silence first — fast, no librosa needed (using scipy.io.wavfile to avoid soundfile/CFFI deadlocks on Windows)
    try:
        import io
        from scipy.io import wavfile
        sr_orig, y_orig = wavfile.read(io.BytesIO(audio_bytes))
        # Convert to float
        if y_orig.dtype == np.int16:
            y = y_orig.astype(np.float32) / 32768.0
        elif y_orig.dtype == np.int32:
            y = y_orig.astype(np.float32) / 2147483648.0
        elif y_orig.dtype == np.uint8:
            y = (y_orig.astype(np.float32) - 128.0) / 128.0
        else:
            y = y_orig.astype(np.float32)
        rms = float(np.sqrt(np.mean(y ** 2)))
        
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        # Adjust silence threshold based on noise floor if calibrated
        threshold = 0.003
        if cal.is_complete and cal.noise_floor is not None:
            threshold = max(0.003, cal.noise_floor * 1.5)
            
        if rms < threshold:
            score_buffer.clear('voice')
            print(f"[Voice Expert] Silence/Ambient noise detected (Intensity: {rms:.4f} < threshold: {threshold:.4f}). Clearing voice buffer.")
            mock_indicators = {
                'f0_mean': 0.0,
                'jitter_percent': 0.0,
                'shimmer_db': 0.0,
                'speaking_rate_proxy': 0.0,
                'voice_intensity': rms
            }
            return jsonify({'score': None, 'reason': 'silence_detected', 'indicators': mock_indicators})
    except Exception as e:
        print(f"Error checking silence: {e}")
        pass

    try:
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        # Phase 7: Set F0 bounds from calibration if available
        if cal.is_complete and cal.f0_mean is not None and cal.f0_mean > 60:
            f0_min = max(60.0, cal.f0_mean * 0.40)   # 40% below personal baseline
            f0_max = min(500.0, cal.f0_mean * 1.80)   # 80% above personal baseline
        else:
            f0_min = 75.0
            f0_max = 400.0

        # Run librosa feature extraction directly to avoid thread deadlocks on Windows
        result = extract_voice_stress_indicators(audio_bytes, 16000, f0_min, f0_max)
        
        if result is None:
            return jsonify({'score': None, 'reason': 'audio_too_short_or_invalid'}), 200

        # Extract features and indicators
        indicators = result['indicators']
        
        # Filter out silence or non-speech hum/noise post-feature-extraction
        if indicators.get('f0_mean', 0.0) == 0.0 or indicators.get('voiced_fraction', 0.0) < 0.05:
            score_buffer.clear('voice')
            print(f"[Voice Expert] Silence/Unvoiced audio detected in features (F0: {indicators.get('f0_mean', 0.0):.1f} Hz, Voiced Frac: {indicators.get('voiced_fraction', 0.0):.3f}). Clearing voice buffer.")
            return jsonify({'score': None, 'reason': 'unvoiced_or_silence', 'indicators': indicators})

        features = result['features']
        
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        session_scaled = None
        if cal.is_complete:
            session_scaled = cal.scale_voice_features(features)

        if session_scaled is not None:
            features_scaled = session_scaled
        else:
            if cal.is_complete:
                features = cal.normalize_voice_features(features, voice_scaler)
                cal.add_voice_sample(indicators)  # continue updating baseline slowly
            features = features.reshape(1, -1)
            features_scaled = voice_scaler.transform(features)

        score = float(voice_expert.predict_proba(features_scaled)[0][1])

        score_buffer.write('voice', score, {
            'landmark_confidence': 0.9  # high confidence for voice if extracted successfully
        })
        print(f"[Voice Expert] Processed audio chunk in OS thread pool. Score: {score:.3f}")
        return jsonify({'score': score, 'indicators': indicators, 'features': result['features'].tolist()})
    except Exception as e:
        print(f"Voice streaming error: {e}")
        return jsonify({'error': str(e)}), 500

# -----------------------------------------------------------------------------
# USER CALIBRATION ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/api/calibrate/silence', methods=['POST'])
def calibrate_silence():
    """Phase 1: Record ambient noise floor from 15 seconds of silence."""
    data        = request.json or {}
    user_id     = data.get('user_id', 'default')
    noise_rms   = float(data.get('noise_rms', 0.01))

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.noise_floor = noise_rms
    cal.phase = 'silence_done'
    print(f"[Calibration] Saved noise floor for user {user_id}: {noise_rms:.4f}")
    return jsonify({'status': 'ok', 'noise_floor': noise_rms})

@app.route('/api/calibrate/voice_sample', methods=['POST'])
def calibrate_voice_sample():
    """Phase 2: Receive voice indicators during neutral speech calibration."""
    data     = request.json or {}
    user_id  = data.get('user_id', 'default')
    indicators = data.get('indicators', {})
    features = data.get('features', None)

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.phase = 'voice_calibrating'
    cal.add_voice_sample(indicators)
    if features is not None:
        cal.add_voice_feature_vector(np.array(features, dtype=np.float32))
    return jsonify({'status': 'ok', 'samples': len(cal.samples_voice)})

@app.route('/api/calibrate/face_sample', methods=['POST'])
def calibrate_face_sample():
    """Phase 3: Receive face indicators during neutral face calibration."""
    data       = request.json or {}
    user_id    = data.get('user_id', 'default')
    indicators = data.get('indicators', {})

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.phase = 'face_calibrating'
    cal.add_face_sample(indicators)
    
    # Extract raw face feature vector for session scaler
    feature_vec = build_face_feature_vector(indicators)
    cal.add_face_feature_vector(feature_vec)
    return jsonify({'status': 'ok', 'samples': len(cal.samples_face)})

@app.route('/api/calibrate/physio_sample', methods=['POST'])
def calibrate_physio_sample():
    """Phase 3b: Receive physio indicators/features during neutral calibration."""
    data = request.json or {}
    user_id = data.get('user_id', 'default')
    indicators = data.get('indicators', {})
    features = data.get('features', None)

    from calibration import get_or_create
    cal = get_or_create(user_id)
    cal.phase = 'physio_calibrating'
    cal.add_physio_sample(indicators)
    if features is not None:
        cal.add_physio_feature_vector(np.array(features, dtype=np.float32))
    else:
        # Default/simulated values
        f_val = [
            float(indicators.get('ecg_rate_mean', 72.0)),
            float(indicators.get('ecg_hrv_rmssd', 45.0)),
            float(indicators.get('ecg_hrv_sdnn', 50.0)),
            float(indicators.get('eda_scl_mean', 1.5)),
            float(indicators.get('resp_rate_mean', 16.0))
        ]
        cal.add_physio_feature_vector(np.array(f_val, dtype=np.float32))
        
    return jsonify({'status': 'ok', 'samples': len(cal.samples_physio)})

@app.route('/api/calibrate/finalize', methods=['POST'])
def calibrate_finalize():
    """Compute final baseline statistics and verify them using the stress models."""
    data    = request.json or {}
    user_id = data.get('user_id', 'default')

    from calibration import get_or_create, get_save_path
    import shutil
    import json
    
    cal = get_or_create(user_id)
    voice_ok = cal.finalize_voice()
    face_ok  = cal.finalize_face()
    
    # Check physiological signals. If missing/insufficient, fill with simulated neutral values
    physio_ok = cal.finalize_physio()
    if not physio_ok:
        print(f"[Calibration] Simulating neutral physiological baseline for user {user_id}")
        for _ in range(10):
            rate = float(np.random.normal(70.0, 1.5))
            rmssd = float(np.random.normal(45.0, 2.0))
            sdnn = float(np.random.normal(50.0, 2.0))
            scl = float(np.random.normal(1.5, 0.1))
            resp = float(np.random.normal(16.0, 0.5))
            indicators = {
                'ecg_rate_mean': rate,
                'ecg_hrv_rmssd': rmssd,
                'ecg_hrv_sdnn': sdnn,
                'eda_scl_mean': scl,
                'resp_rate_mean': resp
            }
            cal.add_physio_sample(indicators)
            cal.add_physio_feature_vector(np.array([rate, rmssd, sdnn, scl, resp], dtype=np.float32))
        physio_ok = cal.finalize_physio()
        
    session_scalers_built = cal.build_session_scalers()

    status = 'complete' if (voice_ok and face_ok and session_scalers_built) else 'partial'
    verification_results = {}
    
    if voice_ok and face_ok:
        # 1. Compute average features from collected matrices
        mean_raw_face = np.mean(cal._face_baseline_matrix, axis=0)
        mean_raw_voice = np.mean(cal._voice_baseline_matrix, axis=0)
        mean_raw_physio = np.mean(cal._physio_baseline_matrix, axis=0)
        
        # 2. Run uncalibrated predictions to verify if baseline window is neutral
        # Temporarily disable completed status so inference runs in population/uncalibrated space
        was_complete = cal.is_complete
        cal.is_complete = False
        try:
            res_fused = runtime_engine.predict_fused(
                face=mean_raw_face, 
                voice=mean_raw_voice, 
                physio=mean_raw_physio
            )
        finally:
            cal.is_complete = was_complete
            
        stress_prob = res_fused.get('stress_probability', 0.0)
        biomarker_scores = {
            'face': float(res_fused.get('individual_predictions', {}).get('face', 0.0)),
            'voice': float(res_fused.get('individual_predictions', {}).get('voice', 0.0)),
            'physio': float(res_fused.get('individual_predictions', {}).get('physio', 0.0))
        }
        
        # 3. Compute population baseline deviations (average absolute Z-score of window features)
        pop_deviations = {}
        for modality, mean_feats, scaler_key in [('face', mean_raw_face, 'face'), 
                                                 ('voice', mean_raw_voice, 'voice'), 
                                                 ('physio', mean_raw_physio, 'physio')]:
            scaler = runtime_engine._scalers.get(scaler_key)
            if scaler is not None:
                # Preprocess without personal baseline normalization
                if modality == 'face':
                    locked = runtime_engine.feature_lock.process_face_features(mean_feats, scaler=None)
                elif modality == 'voice':
                    locked = runtime_engine.feature_lock.process_voice_features(mean_feats, scaler=None)
                else:
                    locked = runtime_engine.feature_lock.process_physio_features(mean_feats, scaler=None)
                pop_deviations[modality] = float(np.mean(np.abs(scaler.transform(locked))))
            else:
                pop_deviations[modality] = 0.0
                
        # 4. Compute deviation from prior user baseline if available
        prior_devs = {}
        save_path = get_save_path(user_id)
        accepted_path = save_path.replace(".json", "_accepted.json")
        if os.path.exists(accepted_path):
            try:
                with open(accepted_path, 'r') as f:
                    prior_data = json.load(f)
                if prior_data.get('ear_baseline') and cal.ear_baseline:
                    prior_devs['face'] = float(abs(cal.ear_baseline - prior_data['ear_baseline']) / prior_data['ear_baseline'])
                if prior_data.get('f0_mean') and cal.f0_mean:
                    prior_devs['voice'] = float(abs(cal.f0_mean - prior_data['f0_mean']) / prior_data['f0_mean'])
                if prior_data.get('physio_mean') and cal.physio_mean:
                    prior_devs['physio'] = float(abs(cal.physio_mean[0] - prior_data['physio_mean'][0]) / prior_data['physio_mean'][0])
            except Exception as e:
                print(f"Error loading prior baseline: {e}")
                
        # 5. Generate explanation summary using top drivers
        explanation_summary = "All biomarkers are within normal resting parameters."
        top_drivers = []
        if runtime_engine.expl_engine and runtime_engine.expl_engine.is_loaded:
            expl_payload = runtime_engine.expl_engine.build_full_payload(
                face_features=mean_raw_face,
                voice_features=mean_raw_voice,
                physio_features=mean_raw_physio
            )
            top_drivers = expl_payload.get('top_drivers', [])
            drivers_desc = []
            for d in top_drivers[:3]:
                feat_name = d.get('feature', 'Unknown')
                mod_name = d.get('modality', '')
                drivers_desc.append(f"{feat_name} ({mod_name})")
            if drivers_desc:
                explanation_summary = f"Baseline features show primary contributions from: {', '.join(drivers_desc)}."
                
        # 6. Recommendation: Auto-accept if stress probability < 0.40 and no modality score > 0.50
        max_modality_prob = max(biomarker_scores.values())
        if stress_prob < 0.40 and max_modality_prob < 0.50:
            recommendation = 'ACCEPT_BASELINE'
            cal.is_complete = True
            cal.save_to_file(user_id)
            try:
                shutil.copyfile(save_path, accepted_path)
                print(f"[Calibration] Saved automatically accepted baseline to {accepted_path}")
            except Exception as e:
                print(f"Error copying accepted baseline file: {e}")
        else:
            recommendation = 'NEEDS_CONFIRMATION'
            cal.is_complete = False
            cal.save_to_file(user_id)
            
        verification_results = {
            'recommendation': recommendation,
            'stress_probability': float(stress_prob),
            'biomarker_scores': biomarker_scores,
            'pop_deviations': pop_deviations,
            'prior_deviations': prior_devs,
            'explanation_summary': explanation_summary,
            'top_features': top_drivers[:3]
        }
        
        cal.verification_results = verification_results
        cal.save_to_file(user_id)

    print(f"[Calibration] Finalized for user {user_id}. Voice ok: {voice_ok}, Face ok: {face_ok}, Physio ok: {physio_ok}, Scalers built: {session_scalers_built}")
    return jsonify({
        'status':      status,
        'calibration': cal.to_dict(),
        'session_scalers_built': session_scalers_built,
        'verification': verification_results
    })

@app.route('/api/calibrate/confirm', methods=['POST'])
def calibrate_confirm():
    """Consent loop: user reviews the baseline verification and chooses action."""
    data = request.json or {}
    user_id = data.get('user_id', 'default')
    action = data.get('action') # 'accept_low_confidence', 'recalibrate', 'discard'
    notes = data.get('notes', '')

    from calibration import get_or_create, clear, get_save_path
    import shutil
    cal = get_or_create(user_id)

    if action == 'accept_low_confidence':
        cal.is_complete = True
        cal.is_low_confidence = True
        cal.confidence_notes = notes
        cal.save_to_file(user_id)
        
        # Save as accepted baseline
        save_path = get_save_path(user_id)
        accepted_path = save_path.replace(".json", "_accepted.json")
        try:
            shutil.copyfile(save_path, accepted_path)
            print(f"[Calibration] Manually accepted low-confidence baseline saved to {accepted_path}")
        except Exception as e:
            print(f"Error copying accepted calibration: {e}")
            
        return jsonify({
            'status': 'ok',
            'state': 'ACCEPT_WITH_LOW_CONFIDENCE',
            'calibration': cal.to_dict()
        })
    elif action in ('recalibrate', 'discard'):
        clear(user_id)
        return jsonify({
            'status': 'ok',
            'state': 'RECALIBRATE',
            'calibration': None
        })
    else:
        return jsonify({'error': 'Invalid action'}), 400

@app.route('/api/calibrate/status', methods=['GET'])
def calibrate_status():
    user_id = request.args.get('user_id', 'default')
    from calibration import get_or_create
    cal = get_or_create(user_id)
    return jsonify(cal.to_dict())

@app.route('/api/stream/fused', methods=['GET'])
def stream_fused():
    """
    SSE stream emitting fused predictions every 2 seconds.
    """
    request.environ['eventlet.minimum_write_chunk_size'] = 1
    request.environ['eventlet.minimum_chunk_size'] = 1
    print("[Fusion Engine] Client connected to fused SSE stream.")

    def generate():
        yield ":" + " " * 4096 + "\n\n"
        while True:
            all_scores = score_buffer.read_all()

            if not all_scores:
                data = json.dumps({'status': 'waiting', 'modalities_active': 0})
                print("[Fusion Engine] Waiting for active modalities...")
            else:
                probs = {k: v['ema_score'] for k, v in all_scores.items()}
                confs = {k: v['indicators'].get('landmark_confidence', 0.7)
                         for k, v in all_scores.items()}
                certainties = {k: v['indicators'].get('certainty', 1.0)
                               for k, v in all_scores.items()}

                fused = fuse_predictions(probs, confs, certainties=certainties, fusion_mode='reliability')
                fused['status'] = 'active'
                fused['modalities_active'] = len(all_scores)
                fused['per_modality'] = {
                    k: {'score': round(v['ema_score'], 3)} for k, v in all_scores.items()
                }
                
                if runtime_engine.expl_engine and runtime_engine.expl_engine.is_loaded:
                    fused['explainability'] = runtime_engine.expl_engine.build_full_payload(
                        face_features=[] if 'face' in probs else None,
                        voice_features=[] if 'voice' in probs else None,
                        physio_features=[] if 'physio' in probs else None,
                    )
                
                data = json.dumps(fused)
                print(f"[Fusion Engine] Fused Level: {fused['stress_level']} ({fused['fused_score']:.3f}) | Active: {list(probs.keys())}")

            yield f"data: {data}\n\n"
            eventlet.sleep(1)

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )

# ---------------------------
# Real-Time WebSocket Events
# ---------------------------

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    stream_processor.initialize_session(request.sid)
    emit('status', {'msg': 'Connected to streaming server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    stream_processor.remove_session(request.sid)

@socketio.on('stream_audio')
def handle_audio_stream(data):
    """
    Handle incoming audio buffer stream.
    Expects data = {'audio': [float array]}
    """
    if 'audio' not in data:
        return
        
    session_id = request.sid
    # Get sensitivity from data (default 0.5)
    sensitivity = data.get('sensitivity', 0.5)
    
    result = stream_processor.process_audio_chunk(
        session_id, 
        data['audio'], 
        data.get('sr', 44100),
        sensitivity=sensitivity
    )
    
    emit('stress_update', {'type': 'audio', 'result': result})

@socketio.on('stream_video')
def handle_video_stream(data):
    """
    Handle incoming video frame.
    Expects data = {'image': 'base64string'}
    """
    try:
        if 'image' not in data:
            return
            
        # Get sensitivity from data (default 0.5)
        sensitivity = data.get('sensitivity', 0.5)
            
        result = stream_processor.process_video_frame(
            data['image'],
            sensitivity=sensitivity
        )
        
        if 'error' not in result:
            emit('stress_update', {
                'type': 'video', 
                'result': result
            })
    except Exception as e:
        print(f"Stream Video Error: {e}")

@app.route('/api/chat/stress', methods=['POST'])
def stress_chat():
    """Stress assistant chat endpoint backed by Gemini API with local fallback."""
    try:
        payload = request.get_json(silent=True) or {}
        message = (payload.get('message') or '').strip()
        stress_level = payload.get('stress_level', 'Moderate')
        stress_percentage = payload.get('stress_percentage', None)

        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message is required.'
            }), 400

        # Crisis Detection Gate
        import re
        crisis_keywords = [
            r'\bsuicide\b', r'\bkill myself\b', r'\bwant to die\b', r'\bend it all\b',
            r'\bhurt myself\b', r'\bno reason to live\b', r'\bdon\'t want to be here anymore\b'
        ]
        message_lower = message.lower()
        if any(re.search(keyword, message_lower) for keyword in crisis_keywords):
            return jsonify({
                'status': 'success',
                'reply': "It sounds like you are going through a very difficult time. Please know that you are not alone and help is available right now. Please reach out to a local emergency service or a crisis hotline immediately (e.g., dial 988 in the US/Canada or your local emergency number).",
                'provider': 'crisis-gate'
            })

        reply = ask_gemini_stress_assistant(message, stress_level, stress_percentage)

        return jsonify({
            'status': 'success',
            'reply': reply,
            'provider': 'gemini' if GEMINI_API_KEY else 'local-fallback'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/muse/start', methods=['POST'])
def start_muse_stream():
    """Start Muse LSL CSV recording for a fixed duration."""
    payload = request.get_json(silent=True) or {}

    try:
        duration = int(payload.get('duration', 20))
    except (ValueError, TypeError):
        duration = 20

    duration = max(5, min(duration, 1800))
    file_path = (payload.get('filename') or MUSE_DEFAULT_FILENAME).strip()

    if not file_path:
        return jsonify({'status': 'error', 'message': 'filename is required'}), 400

    # Resolve relative path to absolute path relative to BASE_DIR
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(os.path.join(BASE_DIR, file_path))

    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with MUSE_SESSION_LOCK:
        current_proc = MUSE_SESSION.get('process')
        if current_proc is not None and current_proc.poll() is None:
            return jsonify({
                'status': 'error',
                'message': 'A Muse recording session is already in progress.'
            }), 409

        cmd = [
            sys.executable,
            '-m',
            'muselsl',
            'record',
            '--duration',
            str(duration),
            '--filename',
            file_path,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            return jsonify({
                'status': 'error',
                'message': f'Could not start muselsl recording: {exc}'
            }), 500

        MUSE_SESSION['process'] = proc
        MUSE_SESSION['duration_seconds'] = duration
        MUSE_SESSION['file_path'] = file_path
        MUSE_SESSION['started_at'] = time.time()
        MUSE_SESSION['completed'] = False
        MUSE_SESSION['prediction'] = None
        MUSE_SESSION['error'] = None

    return jsonify({
        'status': 'success',
        'message': 'Muse recording started',
        'duration_seconds': duration,
        'file_path': file_path,
        'command': f'python -m muselsl record --duration {duration} --filename {file_path}',
    })


@app.route('/api/muse/stop', methods=['POST'])
def stop_muse_stream():
    """Stop active Muse recording session."""
    with MUSE_SESSION_LOCK:
        proc = MUSE_SESSION.get('process')
        if proc is None or proc.poll() is not None:
            return jsonify({'status': 'success', 'message': 'No active Muse recording.'})

        proc.terminate()
        MUSE_SESSION['completed'] = True
        MUSE_SESSION['error'] = 'Recording stopped by user.'

    return jsonify({'status': 'success', 'message': 'Muse recording stopped.'})


@app.route('/api/muse/status', methods=['GET'])
def muse_stream_status():
    """Return live Muse points and final prediction when available."""
    _refresh_muse_session_if_needed()

    try:
        limit = int(request.args.get('limit', 240))
    except (ValueError, TypeError):
        limit = 240

    limit = max(30, min(limit, 2000))

    with MUSE_SESSION_LOCK:
        proc = MUSE_SESSION.get('process')
        collecting = proc is not None and proc.poll() is None
        file_path = MUSE_SESSION.get('file_path')
        started_at = MUSE_SESSION.get('started_at')
        duration_seconds = MUSE_SESSION.get('duration_seconds')
        completed = MUSE_SESSION.get('completed', False)
        prediction = MUSE_SESSION.get('prediction')
        error = MUSE_SESSION.get('error')

    points = _read_muse_points(file_path, limit=limit)
    elapsed_seconds = int(max(0, time.time() - started_at)) if started_at else 0

    return jsonify({
        'status': 'success',
        'collecting': collecting,
        'completed': completed,
        'duration_seconds': duration_seconds,
        'elapsed_seconds': elapsed_seconds,
        'file_path': file_path,
        'points': points,
        'prediction': prediction,
        'error': error,
    })

# -----------------------------------------------------------------------------
# SYSTEM SHUTDOWN ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/api/restart/backend', methods=['POST'])
def restart_backend():
    if request.remote_addr != '127.0.0.1':
        return jsonify({'status': 'error', 'message': 'Forbidden'}), 403
    print("[Shutdown] Restarting backend server...")
    def restart_self():
        import time, os, sys, subprocess
        time.sleep(1)
        # On Windows, ping localhost for a few seconds to let the port free up, then restart
        if os.name == 'nt':
            cmd = f'ping 127.0.0.1 -n 3 > nul && "{sys.executable}" "{sys.argv[0]}"'
            subprocess.Popen(cmd, shell=True)
        else:
            cmd = f'sleep 2 && "{sys.executable}" "{sys.argv[0]}"'
            subprocess.Popen(cmd, shell=True)
        os._exit(0)
    import threading
    threading.Thread(target=restart_self).start()
    return jsonify({'status': 'success', 'message': 'Backend is restarting...'})

@app.route('/api/shutdown/backend', methods=['POST'])
def shutdown_backend():
    if request.remote_addr != '127.0.0.1':
        return jsonify({'status': 'error', 'message': 'Forbidden'}), 403
    print("[Shutdown] Shutting down backend server...")
    def kill_self():
        import time, os
        time.sleep(1)
        os._exit(0)
    import threading
    threading.Thread(target=kill_self).start()
    return jsonify({'status': 'success', 'message': 'Backend is shutting down...'})

if __name__ == '__main__':
    print("Starting Multimodal Stress Detection API...")
    
    # Waitress does not support WebSockets/Socket.IO, so we use eventlet via socketio.run
    print("Starting SocketIO server on http://localhost:5000...")
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, use_reloader=False, minimum_chunk_size=1)


# --- Phase 8: Admin & Monitoring Endpoints ---

@app.route('/api/admin/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        "status": "success",
        "metrics": runtime_metrics.get_metrics(),
        "drift": drift_monitor.get_drift_report()
    })

@app.route('/api/admin/rollback', methods=['POST'])
def rollback():
    global runtime_engine, stream_processor
    
    data = request.json or {}
    model_key = data.get('model_key', 'face')
    version = data.get('version')
    if not version:
        return jsonify({"status": "error", "message": "version required"}), 400
        
    registry = runtime_engine.registry
    if registry.rollback_model(model_key, version):
        # Reload runtime engine
        runtime_engine = RuntimeEngine.from_registry()
        stream_processor = StressStreamProcessor(runtime_engine=runtime_engine)
        return jsonify({"status": "success", "message": f"Rolled back {model_key} to {version}"})
    return jsonify({"status": "error", "message": "Rollback failed"}), 400

@app.route('/api/admin/golden_replay', methods=['POST'])
def run_golden_replay():
    data = request.json or {}
    rows = data.get('rows', [])
    if not rows:
        return jsonify({"status": "error", "message": "rows required"}), 400
    
    res = golden_replay.run_replay(rows)
    return jsonify(res)
