# -*- coding: utf-8 -*-
"""
run_and_verify_all.py
Programmatic test runner that executes all test cases from TEST_GUIDE.md
and outputs validation statuses and reliability analysis data.
"""
import sys
import os
import time
import json
import pickle
import threading
import io
import numpy as np
import requests
import soundfile as sf
from scipy import signal

# Set directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

print("=" * 70)
print("  STRESS DETECTION MULTIMODAL SYSTEM - AUTOMATED VERIFICATION SUITE")
print("=" * 70)

results = {}

# ---------------------------------------------------------
# TEST P1: Environment Check
# ---------------------------------------------------------
print("\n--- TEST P1: Environment Packages ---")
packages = {
    'flask':          'flask',
    'flask_socketio': 'flask_socketio',
    'eventlet':       'eventlet',
    'numpy':          'numpy',
    'sklearn':        'sklearn',
    'librosa':        'librosa',
    'soundfile':      'soundfile',
    'scipy':          'scipy',
    'mediapipe':      'mediapipe',
    'cv2':            'cv2',
    'shap':           'shap',
    'parselmouth':    'parselmouth',
    'imblearn':       'imblearn',
}
missing = []
for name, mod in packages.items():
    try:
        m = __import__(mod)
        missing.append(False)
    except ImportError:
        missing.append(name)
if any(missing):
    p1_status = "FAIL"
    p1_msg = f"Missing: {[m for m in missing if m]}"
else:
    p1_status = "PASS"
    p1_msg = "All packages present."
results['P1'] = (p1_status, p1_msg)
print(f"[{p1_status}] {p1_msg}")

# ---------------------------------------------------------
# TEST P2: Frontend dependencies
# ---------------------------------------------------------
print("\n--- TEST P2: Frontend Package.json Configuration ---")
try:
    pkg_path = os.path.join(BASE_DIR, '..', 'frontend', 'package.json')
    with open(pkg_path, 'r', encoding='utf-8') as f:
        pkg = json.load(f)
    deps = {**(pkg.get('dependencies', {})), **(pkg.get('devDependencies', {}))}
    required = ['react', 'react-dom', 'react-router-dom', 'recharts']
    ok = []
    for r in required:
        if r in deps:
            ok.append(f"{r}: {deps[r]}")
        else:
            ok.append(None)
    if None in ok:
        p2_status = "FAIL"
        p2_msg = f"Missing deps in package.json"
    else:
        p2_status = "PASS"
        p2_msg = "Required dependencies " + ", ".join([x for x in ok if x])
except Exception as e:
    p2_status = "FAIL"
    p2_msg = f"Could not read frontend package.json: {e}"
results['P2'] = (p2_status, p2_msg)
print(f"[{p2_status}] {p2_msg}")

# ---------------------------------------------------------
# TEST P3: Expert model files existence & loadability
# ---------------------------------------------------------
print("\n--- TEST P3: Expert Model Files loadability ---")
required_models = [
    'face_expert_lightweight.pkl',
    'face_scaler_lightweight.pkl',
    'voice_expert_lightweight.pkl',
    'voice_scaler_lightweight.pkl',
    'physio_expert.pkl',
    'physio_scaler.pkl',
]
loaded = []
for fname in required_models:
    path = os.path.join(BASE_DIR, 'expert_models', fname)
    if not os.path.exists(path):
        loaded.append(None)
        print(f"  MISSING: {fname}")
    else:
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            loaded.append(type(obj).__name__)
            print(f"  OK {fname}: type={type(obj).__name__}")
        except Exception as e:
            loaded.append(f"ERROR: {e}")
if None in loaded or any("ERROR" in str(x) for x in loaded):
    p3_status = "FAIL"
    p3_msg = "One or more models failed to load or were missing"
else:
    p3_status = "PASS"
    p3_msg = f"Loaded models: {[x for x in loaded]}"
results['P3'] = (p3_status, p3_msg)
print(f"[{p3_status}] {p3_msg}")

# ---------------------------------------------------------
# TEST P4: Score buffer and calibration imports
# ---------------------------------------------------------
print("\n--- TEST P4: Score Buffer & Calibration Imports ---")
try:
    from score_buffer import ScoreBuffer
    from calibration import UserCalibration
    sb = ScoreBuffer()
    sb.write('face', 0.72, {'avg_ear': 0.28})
    sb.write('voice', 0.61, {'f0_mean': 180.0})
    r_face = sb.read('face')
    r_all = sb.read_all()
    assert r_face is not None and r_face['score'] == 0.72
    assert len(r_all) == 2
    
    cal = UserCalibration()
    for pitch in [165.0, 170.0, 168.0, 172.0, 167.0]:
        cal.add_voice_sample({'f0_mean': pitch, 'voice_intensity': 0.08, 'hnr': 12.0})
    ok = cal.finalize_voice()
    assert ok and 160 < cal.f0_mean < 200
    p4_status = "PASS"
    p4_msg = f"ScoreBuffer & UserCalibration working. Calibrated f0: {cal.f0_mean:.1f}Hz"
except Exception as e:
    p4_status = "FAIL"
    p4_msg = f"ScoreBuffer or UserCalibration failed: {e}"
results['P4'] = (p4_status, p4_msg)
print(f"[{p4_status}] {p4_msg}")

# ---------------------------------------------------------
# TEST F1: MediaPipe 18 real indicator values
# ---------------------------------------------------------
print("\n--- TEST F1: Face MediaPipe Indicator Extraction ---")
test_face_img = "C:/Users/KISHO/.gemini/antigravity-ide/brain/89cd562a-b3e7-462b-8eb8-c4521ae831f5/test_face_1780641154038.png"
if not os.path.exists(test_face_img):
    f1_status = "FAIL"
    f1_msg = "test_face_1780641154038.png not found in scratch folder"
else:
    try:
        import sys as _sys
        _training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')
        if _training_dir not in _sys.path:
            _sys.path.insert(0, _training_dir)
        # pyrefly: ignore [missing-import]
        from extract_face_indicators_offline import compute_18_indicators
        res = compute_18_indicators(test_face_img)
        assert res is not None and len(res) == 18
        arr = np.array(res)
        assert not np.any(np.isnan(arr))
        assert np.max(arr) < 10.0
        f1_status = "PASS"
        f1_msg = f"Got 18 normalized face indicators. Max val: {np.max(arr):.4f}, avg_ear: {res[2]:.4f}"
    except Exception as e:
        f1_status = "FAIL"
        f1_msg = f"Indicator extraction failed: {e}"
results['F1'] = (f1_status, f1_msg)
print(f"[{f1_status}] {f1_msg}")

# Pre-generate shared audio_bytes at module scope so downstream tests (A3, A4, R1) never crash.
_sr = 16000
_t = np.linspace(0, 2.0, _sr * 2)
np.random.seed(42)
_freqs = 150.0 * (1 + 0.005 * np.cumsum(np.random.randn(len(_t)) * 0.01))
_phase = 2 * np.pi * np.cumsum(_freqs / _sr)
_wave = 0.5 * np.sin(_phase) * (1 + 0.03 * np.sin(2 * np.pi * 5 * _t))
_wave += 0.005 * np.random.randn(len(_wave))
_buf = io.BytesIO()
sf.write(_buf, _wave.astype(np.float32), _sr, format='WAV')
audio_bytes = _buf.getvalue()  # module-level: available to all tests

# ---------------------------------------------------------
# TEST F2: Voice Jitter non-zero
# ---------------------------------------------------------
print("\n--- TEST F2: Voice Parabolic Interpolation Jitter ---")
try:
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    np.random.seed(42)
    f0_base = 150.0
    jitter_amount = 0.005
    freqs = f0_base * (1 + jitter_amount * np.cumsum(np.random.randn(len(t)) * 0.01))
    phase = 2 * np.pi * np.cumsum(freqs / sr)
    signal_wave = 0.5 * np.sin(phase) * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
    signal_wave += 0.005 * np.random.randn(len(signal_wave))
    
    buf = io.BytesIO()
    sf.write(buf, signal_wave.astype(np.float32), sr, format='WAV')
    audio_bytes = buf.getvalue()  # also reassign local copy for this test's assertions
    
    from voice_worker import extract_voice_stress_indicators
    res = extract_voice_stress_indicators(audio_bytes, sr_target=sr)
    assert res is not None
    ind = res['indicators']
    jitter = ind.get('jitter_percent', 0.0)
    f0 = ind.get('f0_mean', 0.0)
    hnr = ind.get('hnr', 0.0)
    
    assert jitter > 0.001 and jitter < 5.0
    assert 100 < f0 < 250
    assert hnr > 0
    f2_status = "PASS"
    f2_msg = f"Jitter percent: {jitter:.4f}%, f0_mean: {f0:.1f} Hz, hnr: {hnr:.2f} dB"
except Exception as e:
    f2_status = "FAIL"
    f2_msg = f"Voice extraction failed: {e}"
results['F2'] = (f2_status, f2_msg)
print(f"[{f2_status}] {f2_msg}")

# ---------------------------------------------------------
# TEST F3: Silence returns low intensity
# ---------------------------------------------------------
print("\n--- TEST F3: Voice Silence Gate ---")
try:
    sr = 16000
    noise = np.random.randn(sr * 2) * 0.005
    buf = io.BytesIO()
    sf.write(buf, noise.astype(np.float32), sr, format='WAV')
    res = extract_voice_stress_indicators(buf.getvalue(), sr_target=sr)
    assert res is not None
    intensity = res['indicators'].get('voice_intensity', 0)
    assert intensity < 0.04
    f3_status = "PASS"
    f3_msg = f"Silence RMS intensity: {intensity:.4f} (under 0.04 gate)"
except Exception as e:
    f3_status = "FAIL"
    f3_msg = f"Silence test failed: {e}"
results['F3'] = (f3_status, f3_msg)
print(f"[{f3_status}] {f3_msg}")

# ---------------------------------------------------------
# TEST F4: EEG band powers determinism and GSR peaks
# ---------------------------------------------------------
print("\n--- TEST F4: Physiological Feature Determinism ---")
try:
    from model import extract_eeg_features, extract_gsr_features, extract_physiological_features
    sr = 256
    t = np.linspace(0, 10.0, sr * 10)
    eeg_relax = (2.0 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))).tolist()
    feats1 = extract_eeg_features(eeg_relax, fs=sr)
    feats2 = extract_eeg_features(eeg_relax, fs=sr)
    assert np.allclose(feats1, feats2, atol=1e-6)
    
    alpha_p = feats1[4]
    beta_p = feats1[6]
    theta_p = feats1[2]
    assert alpha_p > beta_p and alpha_p > theta_p
    stress_idx = feats1[11]
    assert stress_idx < 0.5
    
    gsr_calm = (0.5 + 0.02 * np.random.randn(120)).tolist()
    gsr_feats = extract_gsr_features(gsr_calm, fs=4)
    scr_count = gsr_feats[3]
    assert scr_count < 3
    
    f4_status = "PASS"
    f4_msg = f"EEG is deterministic, alpha={alpha_p:.4f} dominates beta={beta_p:.4f}. GSR scr_count={scr_count:.0f}"
except Exception as e:
    f4_status = "FAIL"
    f4_msg = f"Physiological verification failed: {e}"
results['F4'] = (f4_status, f4_msg)
print(f"[{f4_status}] {f4_msg}")

# Load lightweight models for section 2 inference verification
face_expert = pickle.load(open('expert_models/face_expert_lightweight.pkl','rb'))
face_scaler = pickle.load(open('expert_models/face_scaler_lightweight.pkl','rb'))
voice_expert = pickle.load(open('expert_models/voice_expert_lightweight.pkl','rb'))
voice_scaler = pickle.load(open('expert_models/voice_scaler_lightweight.pkl','rb'))
physio_expert = pickle.load(open('expert_models/physio_expert.pkl','rb'))
physio_scaler = pickle.load(open('expert_models/physio_scaler.pkl','rb'))

# ---------------------------------------------------------
# TEST M1: Face discrimination
# ---------------------------------------------------------
print("\n--- TEST M1: Face Model Discrimination ---")
try:
    calm_face = np.array([
        0.31, 0.30, 0.305, 0.0, 0.11, 0.11, 0.0, 0.001, 0.81, 0.23, 0.08, 1.90, 0.0, 0.0, 0.0, 0.305, 0.95, 0.26
    ], dtype=np.float32).reshape(1, -1)
    stress_face = np.array([
        0.16, 0.15, 0.155, 0.09, 0.25, 0.26, 0.02, 0.16, 0.98, 0.29, 0.16, 1.55, 0.09, 0.02, 0.02, 0.155, 0.90, 0.39
    ], dtype=np.float32).reshape(1, -1)
    
    calm_prob = float(face_expert.predict_proba(face_scaler.transform(calm_face))[0][1])
    stress_prob = float(face_expert.predict_proba(face_scaler.transform(stress_face))[0][1])
    
    assert calm_prob < 0.50
    assert stress_prob > 0.50
    assert stress_prob > calm_prob + 0.15
    m1_status = "PASS"
    m1_msg = f"Calm: {calm_prob:.4f}, Stressed: {stress_prob:.4f}. Separation: {stress_prob-calm_prob:.4f}"
except Exception as e:
    m1_status = "FAIL"
    m1_msg = f"Face model check failed: {e}"
results['M1'] = (m1_status, m1_msg)
print(f"[{m1_status}] {m1_msg}")

# ---------------------------------------------------------
# TEST M2: Voice discrimination
# ---------------------------------------------------------
print("\n--- TEST M2: Voice Model Discrimination ---")
try:
    calm_voice = np.array([
        160.0, 8.0, 30.0, 0.30, 0.15, 14.0, 0.05, 0.08, 0.12, 0.002, 0.15, 0.85
    ], dtype=np.float32).reshape(1, -1)
    stress_voice = np.array([
        250.0, 45.0, 120.0, 4.50, 1.80, 1.0, 0.22, 0.45, 0.40, 0.04, 0.01, 0.55
    ], dtype=np.float32).reshape(1, -1)
    
    calm_prob = float(voice_expert.predict_proba(voice_scaler.transform(calm_voice))[0][1])
    stress_prob = float(voice_expert.predict_proba(voice_scaler.transform(stress_voice))[0][1])
    
    assert calm_prob < 0.55
    assert stress_prob > 0.50
    assert stress_prob > calm_prob
    m2_status = "PASS"
    m2_msg = f"Calm: {calm_prob:.4f}, Stressed: {stress_prob:.4f}. Separation: {stress_prob-calm_prob:.4f}"
except Exception as e:
    m2_status = "FAIL"
    m2_msg = f"Voice model check failed: {e}"
results['M2'] = (m2_status, m2_msg)
print(f"[{m2_status}] {m2_msg}")

# ---------------------------------------------------------
# TEST M3: Physiological discrimination
# ---------------------------------------------------------
print("\n--- TEST M3: Physiological Model Discrimination ---")
try:
    calm_phys = np.zeros(51)
    stress_phys = np.ones(51) * 2.0
    
    calm_prob = float(physio_expert.predict_proba(physio_scaler.transform(calm_phys.reshape(1, -1)))[0][1])
    stress_prob = float(physio_expert.predict_proba(physio_scaler.transform(stress_phys.reshape(1, -1)))[0][1])
    
    assert stress_prob > calm_prob
    m3_status = "PASS"
    m3_msg = f"Calm: {calm_prob:.4f}, Stressed: {stress_prob:.4f}. Separation: {stress_prob-calm_prob:.4f}"
except Exception as e:
    m3_status = "FAIL"
    m3_msg = f"Physio model check failed: {e}"
results['M3'] = (m3_status, m3_msg)
print(f"[{m3_status}] {m3_msg}")

# ---------------------------------------------------------
# TEST C1: Calibration absolute pitch normalization
# ---------------------------------------------------------
print("\n--- TEST C1: Voice Baseline Calibration Normalization ---")
try:
    from calibration import UserCalibration
    cal = UserCalibration()
    for pitch in [210.0, 218.0, 205.0, 215.0, 212.0, 220.0, 208.0, 214.0]:
        cal.add_voice_sample({'f0_mean': pitch, 'voice_intensity': 0.06, 'hnr': 13.5})
    cal.finalize_voice()
    assert 205 < cal.f0_mean < 225
    
    calm_voice_feats = np.array([
        213.0, 10.0, 35.0, 0.28, 0.14, 13.5, 0.05, 0.065, 0.11, 0.002, 0.12, 0.88
    ], dtype=np.float32)
    norm_calm = cal.normalize_voice_features(calm_voice_feats)
    f0_norm_calm = norm_calm[0]
    assert abs(f0_norm_calm) < 0.5
    
    stressed_voice_feats = calm_voice_feats.copy()
    stressed_voice_feats[0] = 260.0
    norm_stress = cal.normalize_voice_features(stressed_voice_feats)
    f0_norm_stress = norm_stress[0]
    assert f0_norm_stress > f0_norm_calm
    
    c1_status = "PASS"
    c1_msg = f"F0 normalized to {f0_norm_calm:.4f} (calm at baseline), and {f0_norm_stress:.4f} (stressed pitch)"
except Exception as e:
    c1_status = "FAIL"
    c1_msg = f"Pitch calibration failed: {e}"
results['C1'] = (c1_status, c1_msg)
print(f"[{c1_status}] {c1_msg}")

# ---------------------------------------------------------
# TEST C2: Face calibration jaw displacement
# ---------------------------------------------------------
print("\n--- TEST C2: Face Baseline Calibration Normalization ---")
try:
    cal = UserCalibration()
    for i in range(15):
        cal.add_face_sample({'avg_ear': 0.30, 'jaw_displacement': 1.85, 'brow_descent_left': 0.12})
    cal.finalize_face()
    assert abs(cal.ear_baseline - 0.3) < 1e-6
    assert abs(cal.jaw_baseline - 1.85) < 1e-6
    
    rest_face = {'avg_ear': 0.30, 'jaw_displacement': 1.85, 'brow_descent_left': 0.12, 'brow_descent_right': 0.12}
    norm_rest = cal.normalize_face_indicators(rest_face)
    jaw_rest = norm_rest['jaw_displacement_normalized']
    assert abs(jaw_rest) < 0.1
    
    clenched_face = rest_face.copy()
    clenched_face['jaw_displacement'] = 1.55
    norm_clenched = cal.normalize_face_indicators(clenched_face)
    jaw_clenched = norm_clenched['jaw_displacement_normalized']
    assert jaw_clenched > 0.05
    
    c2_status = "PASS"
    c2_msg = f"Jaw rest Z-score: {jaw_rest:.4f}, Clenched Z-score: {jaw_clenched:.4f} (>0)"
except Exception as e:
    c2_status = "FAIL"
    c2_msg = f"Face calibration normalization failed: {e}"
results['C2'] = (c2_status, c2_msg)
print(f"[{c2_status}] {c2_msg}")

# ---------------------------------------------------------
# API Endpoints (TEST A1 - A4, A6, S1, R1)
# Auto-start the backend server if it is not already running.
# ---------------------------------------------------------
import subprocess as _sp

API_BASE = "http://127.0.0.1:5000"
_server_proc = None
_server_started_by_us = False

def _server_alive():
    try:
        requests.get(f"{API_BASE}/api/health", timeout=2)
        return True
    except Exception:
        return False

if not _server_alive():
    print("\n[INFO] Backend server not detected — starting it automatically...")
    _server_proc = _sp.Popen(
        [sys.executable, "app.py"],
        cwd=BASE_DIR,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    _server_started_by_us = True
    # Wait up to 25 seconds for server to become ready
    for _i in range(25):
        time.sleep(1)
        if _server_alive():
            print(f"[INFO] Server ready after {_i+1}s.")
            break
    else:
        print("[WARN] Server did not start in 25s — API tests will fail.")
else:
    print("\n[INFO] Backend server already running — skipping auto-start.")

print("\n--- TEST A1: Health API Endpoint ---")
try:
    r = requests.get(f"{API_BASE}/api/health")
    data = r.json()
    assert r.status_code == 200
    assert data.get('status') == 'ok'
    a1_status = "PASS"
    a1_msg = f"Health check returned OK. Models: {data.get('models_loaded')}"
except Exception as e:
    a1_status = "FAIL"
    a1_msg = f"Health API failed: {e}"
results['A1'] = (a1_status, a1_msg)
print(f"[{a1_status}] {a1_msg}")

# Pre-define payload at module scope so A4, R1 can reference it even if A2's try block fails.
payload = {
    'indicators': {
        'left_ear': 0.18, 'right_ear': 0.17, 'avg_ear': 0.175, 'blink_velocity': 0.08,
        'brow_descent_left': 0.22, 'brow_descent_right': 0.23, 'brow_asymmetry': 0.018,
        'lip_compression': 0.10, 'jaw_displacement': 1.60, 'mouth_corner_pull': 0.24,
        'forehead_tension': 0.24, 'face_height_norm': 2.10, 'head_tilt': 4.20,
        'temporal_x_var': 0.009, 'temporal_y_var': 0.007, 'eye_openness_ratio': 0.175,
        'landmark_confidence': 0.90, 'nose_wrinkle': 0.18, 'smile_score': 0.0, 'smile_detected': False
    },
    'user_id': 'test_user'
}

print("\n--- TEST A2: Stream Face API Endpoint ---")
try:
    url = f"{API_BASE}/api/stream/face"
    r = requests.post(url, json=payload)
    data = r.json()
    assert r.status_code == 200
    assert 'score' in data
    a2_status = "PASS"
    a2_msg = f"Face stream output score: {data.get('score'):.4f}"
except Exception as e:
    a2_status = "FAIL"
    a2_msg = f"Face stream API failed: {e}"
results['A2'] = (a2_status, a2_msg)
print(f"[{a2_status}] {a2_msg}")

print("\n--- TEST A3: Stream Voice API Endpoint ---")
try:
    url = f"{API_BASE}/api/stream/voice"
    r = requests.post(url, data=audio_bytes, headers={'Content-Type': 'audio/wav'})
    data = r.json()
    assert r.status_code == 200
    assert 'score' in data
    a3_status = "PASS"
    a3_msg = f"Voice stream output score: {data.get('score'):.4f}"
except Exception as e:
    a3_status = "FAIL"
    a3_msg = f"Voice stream API failed: {e}"
results['A3'] = (a3_status, a3_msg)
print(f"[{a3_status}] {a3_msg}")

print("\n--- TEST A4: Fused SSE Stream Endpoint ---")
try:
    # First write face and voice to buffer to seed it
    requests.post(f"{API_BASE}/api/stream/face", json=payload)
    requests.post(f"{API_BASE}/api/stream/voice", data=audio_bytes, headers={'Content-Type': 'audio/wav'})
    
    r = requests.get(f"{API_BASE}/api/stream/fused", stream=True, timeout=5)
    assert r.status_code == 200
    assert 'text/event-stream' in r.headers.get('Content-Type', '')
    
    events = []
    for line in r.iter_lines():
        if line and line.startswith(b'data: '):
            evt = json.loads(line[6:].decode('utf-8'))
            events.append(evt)
            if len(events) >= 1:
                break
    evt = events[0]
    assert 'fused_score' in evt or evt.get('status') == 'waiting'
    a4_status = "PASS"
    a4_msg = f"Fused SSE stream active. Emitted event: {evt}"
except Exception as e:
    a4_status = "FAIL"
    a4_msg = f"Fused SSE failed: {e}"
results['A4'] = (a4_status, a4_msg)
print(f"[{a4_status}] {a4_msg}")

print("\n--- TEST A6: 3-Phase Calibration Endpoint Flow ---")
try:
    UID = 'calibration_api_test_user'
    # Phase 1
    r = requests.post(f"{API_BASE}/api/calibrate/silence", json={'user_id': UID, 'noise_rms': 0.0048})
    assert r.json()['status'] == 'ok', f"Phase 1 failed: {r.json()}"
    # Phase 2
    for i in range(8):
        fake_features = [170.0 + i*2.0, 5.0, 10.0, 0.2, 0.1, 13.0, 0.04, 0.07, 0.1, 0.001, 0.12, 0.88]
        r = requests.post(f"{API_BASE}/api/calibrate/voice_sample", json={
            'user_id': UID,
            'indicators': {'f0_mean': 170 + i*2.0, 'voice_intensity': 0.07, 'hnr': 13.0},
            'features': fake_features
        })
        assert r.json()['status'] == 'ok', f"Phase 2 failed at step {i}: {r.json()}"
    # Phase 3
    for i in range(15):
        r = requests.post(f"{API_BASE}/api/calibrate/face_sample", json={'user_id': UID, 'indicators': {'avg_ear': 0.30, 'jaw_displacement': 1.85, 'brow_descent_left': 0.12}})
        assert r.json()['status'] == 'ok', f"Phase 3 failed at step {i}: {r.json()}"
    # Finalize
    r = requests.post(f"{API_BASE}/api/calibrate/finalize", json={'user_id': UID})
    data = r.json()
    assert data['status'] == 'complete', f"Finalize status was not complete: {data}"
    cal_data = data['calibration']
    assert abs(cal_data['ear_baseline'] - 0.3) < 1e-6, f"ear_baseline mismatch: {cal_data['ear_baseline']}"
    a6_status = "PASS"
    a6_msg = f"Calibration complete. ear: {cal_data['ear_baseline']:.4f}, f0: {cal_data['f0_mean']:.1f}Hz"
except Exception as e:
    a6_status = "FAIL"
    a6_msg = f"Calibration API flow failed: {e}"
results['A6'] = (a6_status, a6_msg)
print(f"[{a6_status}] {a6_msg}")

print("\n--- TEST S1: Laughter Dampening API check ---")
try:
    laugh_payload = {
        'indicators': {
            'left_ear': 0.16, 'right_ear': 0.15, 'avg_ear': 0.155, 'blink_velocity': 0.04,
            'brow_descent_left': 0.10, 'brow_descent_right': 0.10, 'brow_asymmetry': 0.005,
            'lip_compression': 0.30, 'jaw_displacement': 1.40, 'mouth_corner_pull': 0.38,
            'forehead_tension': 0.10, 'face_height_norm': 2.20, 'head_tilt': 2.0,
            'temporal_x_var': 0.003, 'temporal_y_var': 0.003, 'eye_openness_ratio': 0.155,
            'landmark_confidence': 0.92, 'nose_wrinkle': 0.06, 'smile_score': 0.75,
            'smile_detected': True, 'corner_elevation': 0.045
        },
        'user_id': 'laugh_user'
    }
    r = requests.post(f"{API_BASE}/api/stream/face", json=laugh_payload)
    data = r.json()
    score = data['score']
    raw_score = data['raw_score']
    assert data['smile_detected']
    assert score < 0.65
    s1_status = "PASS"
    s1_msg = f"Laughter correctly dampened. Raw: {raw_score:.4f}, Final: {score:.4f}"
except Exception as e:
    s1_status = "FAIL"
    s1_msg = f"Laughter dampening test failed: {e}"
results['S1'] = (s1_status, s1_msg)
print(f"[{s1_status}] {s1_msg}")

print("\n--- TEST R1: Concurrency and Low Latency ---")
try:
    results_r1 = {}
    def s_face():
        start = time.time()
        r = requests.post(f"{API_BASE}/api/stream/face", json=payload)
        results_r1['face'] = {'time': time.time()-start, 'status': r.status_code}
    def s_voice():
        start = time.time()
        r = requests.post(f"{API_BASE}/api/stream/voice", data=audio_bytes, headers={'Content-Type': 'audio/wav'})
        results_r1['voice'] = {'time': time.time()-start, 'status': r.status_code}
        
    t1 = threading.Thread(target=s_face)
    t2 = threading.Thread(target=s_voice)
    t0 = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    wall = time.time() - t0
    
    face_ms = results_r1.get('face', {}).get('time', 0) * 1000
    voice_ms = results_r1.get('voice', {}).get('time', 0) * 1000
    assert results_r1.get('face', {}).get('status') == 200, f"Face thread status: {results_r1.get('face')}"
    assert results_r1.get('voice', {}).get('status') == 200, f"Voice thread status: {results_r1.get('voice')}"
    assert face_ms < 500
    assert voice_ms < 3000
    r1_status = "PASS"
    r1_msg = f"Concurrency verified. Face: {face_ms:.1f}ms, Voice: {voice_ms:.1f}ms, Wall: {wall*1000:.1f}ms"
except Exception as e:
    r1_status = "FAIL"
    r1_msg = f"Concurrency check failed: {e}"
results['R1'] = (r1_status, r1_msg)
print(f"[{r1_status}] {r1_msg}")

# Shutdown server if we started it
if _server_started_by_us and _server_proc is not None:
    print("\n[INFO] Shutting down auto-started backend server...")
    _server_proc.terminate()
    try:
        _server_proc.wait(timeout=5)
    except Exception:
        _server_proc.kill()

# ---------------------------------------------------------
# Integrity & Verification (TEST D1, D2, I1)
# ---------------------------------------------------------
print("\n--- TEST D1: Production Code Randomness Scan ---")
try:
    found = False
    for filename in ['model.py', 'voice_worker.py', 'app.py']:
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                stripped = line.split('#')[0]
                if 'np.random' in stripped or 'random.randn' in stripped:
                    found = True
                    print(f"  FLAG: {filename}:{idx}: {line.strip()}")
    if found:
        d1_status = "FAIL"
        d1_msg = "Found np.random code inside production pipeline files"
    else:
        d1_status = "PASS"
        d1_msg = "No random fallback code detected in pipeline modules"
except Exception as e:
    d1_status = "FAIL"
    d1_msg = f"Integrity scan crashed: {e}"
results['D1'] = (d1_status, d1_msg)
print(f"[{d1_status}] {d1_msg}")

print("\n--- TEST D2: Model Input Shape Consistency ---")
try:
    face_in = face_scaler.n_features_in_
    voice_in = voice_scaler.n_features_in_
    phys_in = physio_scaler.n_features_in_
    assert face_in == 18
    assert voice_in == 12
    assert phys_in == 51
    d2_status = "PASS"
    d2_msg = f"Shape consistency matched. Face: {face_in}, Voice: {voice_in}, Physio: {phys_in}"
except Exception as e:
    d2_status = "FAIL"
    d2_msg = f"Model shapes do not match feature outputs: {e}"
results['D2'] = (d2_status, d2_msg)
print(f"[{d2_status}] {d2_msg}")

print("\n--- TEST I1: Multimodal Fusion Integration Scenarios ---")
# Safe fallbacks for variables defined inside M1/M2/M3 try blocks
try:
    calm_face
except NameError:
    calm_face = np.zeros((1, 18), dtype=np.float32)
try:
    stress_face
except NameError:
    stress_face = np.ones((1, 18), dtype=np.float32) * 0.5
try:
    calm_voice
except NameError:
    calm_voice = np.zeros((1, 12), dtype=np.float32)
try:
    stress_voice
except NameError:
    stress_voice = np.ones((1, 12), dtype=np.float32) * 0.5
try:
    calm_phys
except NameError:
    calm_phys = np.zeros(51)
try:
    stress_phys
except NameError:
    stress_phys = np.ones(51) * 0.5
try:
    # 1. Calm User
    probs_calm = {
        'face': float(face_expert.predict_proba(face_scaler.transform(calm_face))[0][1]),
        'voice': float(voice_expert.predict_proba(voice_scaler.transform(calm_voice))[0][1]),
        'physio': float(physio_expert.predict_proba(physio_scaler.transform(calm_phys.reshape(1,-1)))[0][1]),
    }
    # 2. Stressed User
    probs_stress = {
        'face': float(face_expert.predict_proba(face_scaler.transform(stress_face))[0][1]),
        'voice': float(voice_expert.predict_proba(voice_scaler.transform(stress_voice))[0][1]),
        'physio': float(physio_expert.predict_proba(physio_scaler.transform(stress_phys.reshape(1,-1)))[0][1]),
    }
    
    from model import fuse_predictions
    confs = {'face': 0.90, 'voice': 0.80, 'physio': 0.85}
    fused_calm = fuse_predictions(probs_calm, confs)
    fused_stress = fuse_predictions(probs_stress, confs)
    
    assert fused_calm['fused_score'] < 0.50 and fused_calm['stress_level'] == 'Low'
    assert fused_stress['fused_score'] > 0.55 and fused_stress['stress_level'] in ['Moderate', 'High']
    margin = fused_stress['fused_score'] - fused_calm['fused_score']
    assert margin > 0.30
    
    i1_status = "PASS"
    i1_msg = f"Calm: {fused_calm['fused_score']:.4f} ({fused_calm['stress_level']}), Stressed: {fused_stress['fused_score']:.4f} ({fused_stress['stress_level']}). Separation: {margin:.4f}"
except Exception as e:
    i1_status = "FAIL"
    i1_msg = f"Integration scenario test failed: {e}"
results['I1'] = (i1_status, i1_msg)
print(f"[{i1_status}] {i1_msg}")

print("\n" + "=" * 70)
print("  VERIFICATION COMPLETED")
print("=" * 70)
passed_count = sum(1 for v in results.values() if v[0] == "PASS")
total_count = len(results)
print(f"Passed: {passed_count}/{total_count} cases.")
print("=" * 70)
