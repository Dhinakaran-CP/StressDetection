import eventlet
eventlet.monkey_patch()

import sys
import builtins

# Force all print statements to flush immediately to avoid buffering in standard terminals/IDE logs
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from realtime_core import StressStreamProcessor
import os
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
from model import MultimodalStressDetector, extract_physiological_features, safe_pickle_load

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25, max_http_buffer_size=100000000)

# Initialize global stream processor
stream_processor = StressStreamProcessor()

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

# Initialize the model
model = MultimodalStressDetector()

# Try to load pre-trained expert models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The load_model method now expects a directory containing the 3 expert .pkl files
if model.load_model(BASE_DIR):
    print("Pre-trained expert models loaded successfully!")
else:
    print("Could not load one or more expert models.")
    print("Please ensure 'face_expert_lightweight.pkl', 'voice_expert_lightweight.pkl', and 'physio_expert.pkl' exist in the 'backend/expert_models/' folder.")
    if hasattr(model, 'load_errors') and model.load_errors:
        print("Detailed load errors:")
        for name, err in model.load_errors.items():
            print(f"  - {name}: {err}")

# --- Load environment variables and configuration ---
import pickle
import time
import json
import urllib.request
import urllib.error
from voice_worker import extract_voice_stress_indicators
from score_buffer import score_buffer

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

def load_expert(filename):
    # Try backend/expert_models/ first, then backend/
    path1 = os.path.join(BASE_DIR, 'expert_models', filename)
    path2 = os.path.join(BASE_DIR, filename)
    path = path1 if os.path.exists(path1) else path2
    if not os.path.exists(path):
        err = f"File not found at {path1} or {path2}"
        print(f"Warning: {filename} not found at {path1} or {path2}")
        MODEL_LOAD_ERRORS[filename] = err
        return None
    try:
        with open(path, 'rb') as f:
            return safe_pickle_load(f)
    except Exception as e:
        err = f"Pickle load error: {e}"
        print(f"Error loading {filename}: {e}")
        MODEL_LOAD_ERRORS[filename] = err
        return None

face_expert = load_expert('face_expert_lightweight.pkl')
face_scaler = load_expert('face_scaler_lightweight.pkl')
voice_expert = load_expert('voice_expert_lightweight.pkl')
voice_scaler = load_expert('voice_scaler_lightweight.pkl')

# --- Explainability SHAP Helper Functions ---
def _extract_class1_shap_values(shap_values):
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
        
    if isinstance(shap_values, list):
        return np.array(shap_values[1][0], dtype=float)

    arr = np.array(shap_values)
    if arr.ndim == 3:
        return np.array(arr[0, :, 1], dtype=float)
    if arr.ndim == 2:
        return np.array(arr[0], dtype=float)
    return np.array(arr, dtype=float).flatten()


def _extract_class1_expected_value(expected_value):
    if isinstance(expected_value, list):
        return float(expected_value[1])

    arr = np.array(expected_value)
    if arr.ndim == 1 and arr.size >= 2:
        return float(arr[1])
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flatten()[0])


def _modality_shap_explanation(modality_name, estimator, scaler, raw_features, feature_prefix):
    if raw_features is None or estimator is None:
        return None

    x_raw = np.array(raw_features, dtype=float).reshape(1, -1)
    
    # Scale features if scaler is present
    if scaler is not None:
        x_scaled = scaler.transform(x_raw)
    else:
        x_scaled = x_raw

    try:
        stress_prob = float(estimator.predict_proba(x_scaled)[0][1])
    except Exception:
        stress_prob = 0.5

    if not SHAP_AVAILABLE:
        return {
            'modality': modality_name,
            'status': 'unavailable',
            'reason': 'SHAP package is not installed in the backend environment.',
            'stress_probability': stress_prob,
            'top_features': [],
        }

    try:
        # If the estimator is a VotingClassifier, pick its RandomForest or GradientBoosting sub-estimator for tree explanation
        actual_estimator = estimator
        
        # Unwrap CalibratedClassifierCV if present
        if hasattr(actual_estimator, 'calibrated_classifiers_') and len(actual_estimator.calibrated_classifiers_) > 0:
            actual_estimator = actual_estimator.calibrated_classifiers_[0].estimator
        elif hasattr(actual_estimator, 'estimator'):
            actual_estimator = actual_estimator.estimator

        if hasattr(actual_estimator, 'estimators_') and len(actual_estimator.estimators_) > 0:
            for sub in actual_estimator.estimators_:
                sub_name = type(sub).__name__
                if 'Forest' in sub_name or 'Boosting' in sub_name or 'Tree' in sub_name:
                    actual_estimator = sub
                    break
        
        explainer = shap.TreeExplainer(actual_estimator)
        shap_values = explainer.shap_values(x_scaled)
        class1_values = _extract_class1_shap_values(shap_values)
        base_value = _extract_class1_expected_value(explainer.expected_value)

        top_count = min(6, class1_values.shape[0])
        top_indices = np.argsort(np.abs(class1_values))[::-1][:top_count]

        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': f'{feature_prefix}_{int(idx)}',
                'feature_index': int(idx),
                'feature_value': float(x_raw[0, idx]),
                'shap_value': float(class1_values[idx]),
                'direction': 'increase' if class1_values[idx] >= 0 else 'decrease',
            })

        return {
            'modality': modality_name,
            'status': 'ok',
            'base_value': base_value,
            'stress_probability': stress_prob,
            'top_features': top_features,
        }
    except Exception as exc:
        return {
            'modality': modality_name,
            'status': 'error',
            'reason': f'SHAP computation failed: {exc}',
            'stress_probability': stress_prob,
            'top_features': [],
        }


def build_explainability_payload(facial_features=None, voice_features=None, phys_features=None):
    modalities = []

    facial_expl = _modality_shap_explanation(
        modality_name='facial',
        estimator=face_expert,
        scaler=face_scaler,
        raw_features=facial_features,
        feature_prefix='facial',
    )
    if facial_expl:
        modalities.append(facial_expl)

    voice_expl = _modality_shap_explanation(
        modality_name='voice',
        estimator=voice_expert,
        scaler=voice_scaler,
        raw_features=voice_features,
        feature_prefix='voice',
    )
    if voice_expl:
        modalities.append(voice_expl)

    phys_expl = _modality_shap_explanation(
        modality_name='physiological',
        estimator=model.phys_model,
        scaler=model.phys_scaler,
        raw_features=phys_features,
        feature_prefix='phys',
    )
    if phys_expl:
        modalities.append(phys_expl)

    top_drivers = []
    for modality in modalities:
        for feat in modality.get('top_features', []):
            top_drivers.append({
                'modality': modality['modality'],
                **feat,
            })

    top_drivers = sorted(top_drivers, key=lambda item: abs(item['shap_value']), reverse=True)[:8]

    return {
        'engine': 'shap',
        'available': SHAP_AVAILABLE,
        'modalities': modalities,
        'top_drivers': top_drivers,
        'message': None if SHAP_AVAILABLE else 'Install shap in backend environment to enable SHAP values.',
    }


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
        import eventlet.tpool
        response_data = eventlet.tpool.execute(_fetch)

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

    phys_features = extract_physiological_features(eeg_array, gsr_array)
    result = model.predict(phys_features=phys_features)
    if result.get('status') == 'success':
        result['source'] = 'muse_stream'
        result['explainability'] = build_explainability_payload(phys_features=phys_features)
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    detector_load_errors = getattr(model, 'load_errors', {})
    
    # Merge load errors from both sources
    all_errors = {**MODEL_LOAD_ERRORS}
    for k, v in detector_load_errors.items():
        all_errors[f"detector_{k}"] = v
        
    status = 'ok' if (face_expert is not None and model.is_trained) else 'degraded'
    return jsonify({
        'status': status,
        'models_loaded': {
            'face_expert': face_expert is not None,
            'voice_expert': model.voice_model is not None,
            'physio_expert': (model.phys_model is not None if hasattr(model, 'phys_model') else False)
        },
        'load_errors': all_errors,
        'server': 'eventlet'
    })

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
            phys_features = extract_physiological_features(eeg_array, gsr_array)
        
        # Check if at least one modality is provided
        if facial_features is None and voice_features is None and phys_features is None:
            return jsonify({
                'status': 'error',
                'message': 'Please provide at least one input (image, audio, or physiological data)'
            }), 400
        
        # Make prediction
        result = model.predict(
            facial_features=facial_features,
            voice_features=voice_features,
            phys_features=phys_features
        )
        
        if 'error' in result:
            print(f"[HTTP] POST /api/multimodal/analyze - Failed: {result['error']}")
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
        
        result['explainability'] = build_explainability_payload(
            facial_features=facial_features,
            voice_features=voice_features,
            phys_features=phys_features,
        )
        
        print(f"[HTTP] POST /api/multimodal/analyze - Success: stress_level={result.get('stress_level')}, percentage={result.get('percentage')}%")
        return jsonify(result)
    
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
        result = model.predict(facial_features=facial_features)
        
        # Clean up
        os.remove(filepath)
        
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
            
        result = model.predict(voice_features=voice_features)
        
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
        result = model.predict(facial_features=facial_features)
        
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
    Fuse predictions from active modalities.
    If only one modality is active, returns that score.
    If multiple, fuses them using the reliability weight algorithm.
    """
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
        
    if len(active_modes) == 1:
        score = probs[active_modes[0]]
        level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
        return {'fused_score': score, 'stress_level': level}
        
    # Reliability weights: base confidence on MediaPipe landmark confidence or fixed defaults
    # face: 0.5, voice: 0.5
    base_weights = {'face': 0.5, 'voice': 0.5}
    
    # Filter active modalities to only include face and voice
    active_modes = [m for m in active_modes if m in base_weights]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low'}
        
    # Calculate reliability weights: weight = base_weight * confidence * certainty
    weights = {}
    for m in active_modes:
        conf = confs.get(m, 1.0)
        cert = certainties.get(m, 1.0) if certainties else 1.0
        weights[m] = base_weights[m] * conf * cert

    w_sum = sum(weights.values())
    if w_sum > 0:
        norm_weights = {m: w / w_sum for m, w in weights.items()}
    else:
        norm_weights = {m: 1.0 / len(active_modes) for m in active_modes}
        
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': norm_weights
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
    if voice_expert is None or voice_scaler is None:
        return jsonify({'error': 'Voice expert model not loaded'}), 500
        
    audio_bytes = request.data
    if not audio_bytes or len(audio_bytes) < 1000:
        return jsonify({'score': None, 'reason': 'too_short_or_empty'}), 200

    user_id = request.args.get('user_id', 'default')

    # Check silence first — fast, no librosa needed (using numpy from audio_bytes)
    try:
        import io
        import soundfile as sf
        y, _ = sf.read(io.BytesIO(audio_bytes))
        rms = float(np.sqrt(np.mean(y ** 2)))
        
        from calibration import get_or_create
        cal = get_or_create(user_id)
        
        # Adjust silence threshold based on noise floor if calibrated
        threshold = 0.015
        if cal.is_complete and cal.noise_floor is not None:
            threshold = max(0.015, cal.noise_floor * 1.5)
            
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

        # Run CPU-heavy librosa processing in eventlet's built-in OS thread pool
        # This keeps the main greenlet loop completely free without spawning subprocesses on Windows
        import eventlet.tpool
        result = eventlet.tpool.execute(extract_voice_stress_indicators, audio_bytes, 16000, f0_min, f0_max)
        
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

@app.route('/api/calibrate/finalize', methods=['POST'])
def calibrate_finalize():
    """Compute final baseline statistics from all collected samples."""
    data    = request.json or {}
    user_id = data.get('user_id', 'default')

    from calibration import get_or_create
    cal = get_or_create(user_id)
    voice_ok = cal.finalize_voice()
    face_ok  = cal.finalize_face()
    session_scalers_built = cal.build_session_scalers()

    print(f"[Calibration] Finalized for user {user_id}. Voice ok: {voice_ok}, Face ok: {face_ok}, Session Scalers built: {session_scalers_built}")
    return jsonify({
        'status':      'complete' if (voice_ok and face_ok and session_scalers_built) else 'partial',
        'calibration': cal.to_dict(),
        'session_scalers_built': session_scalers_built,
    })

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
    print("[Shutdown] Shutting down backend server...")
    def kill_self():
        import time, os
        time.sleep(1)
        os._exit(0)
    import threading
    threading.Thread(target=kill_self).start()
    return jsonify({'status': 'success', 'message': 'Backend is shutting down...'})

@app.route('/api/shutdown/all', methods=['POST'])
def shutdown_all():
    print("[Shutdown] Shutting down entire application (frontend + backend)...")
    def kill_all():
        import time, os, subprocess
        time.sleep(1)
        try:
            if os.name == 'nt':
                # Find and kill the process listening on port 3000 (React frontend)
                cmd = 'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :3000\') do taskkill /F /PID %a'
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run('kill -9 $(lsof -t -i:3000)', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[Shutdown] Failed to kill frontend: {e}")
        # Kill backend
        os._exit(0)
    import threading
    threading.Thread(target=kill_all).start()
    return jsonify({'status': 'success', 'message': 'Entire app is shutting down...'})

if __name__ == '__main__':
    print("Starting Multimodal Stress Detection API...")
    print(f"Model trained: {model.is_trained}")
    
    # Waitress does not support WebSockets/Socket.IO, so we use eventlet via socketio.run
    print("Starting SocketIO server on http://localhost:5000...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False, minimum_chunk_size=1)
