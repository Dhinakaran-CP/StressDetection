# Clinical-Grade Architecture Upgrade
## StressDetectionUsingML — Full Analysis & Production Update
**Based on:** project_report-0906 + walkthrough0906 + full codebase history  
**Date:** June 10, 2026

---

## PART 1 — HONEST STATUS AFTER ALL AGENT WORK

Before addressing your questions, here is exactly what the agent achieved and what is still incomplete.

### What Is Genuinely Working
- SSE streaming, Web Worker face POST, eventlet tpool for voice — all correct
- 23/23 tests pass on synthetic inputs
- Calibration wizard UI built and functional
- Laughter dampening via smile score — implemented
- Parabolic interpolation for jitter — implemented
- Soft-voting ensemble retrained at 65.27% accuracy
- Audio constraints (noiseSuppression, 16kHz) — implemented

### What Is Still Degrading Production Quality

The report describes an "Inverse Scaling Trick" for calibration:
> "Scaled user-specific Z-scores back into pseudo-raw inputs before feeding them to the pre-trained expert classifiers"

**This is architecturally wrong and will not work reliably.** Here is why: the StandardScaler fitted during training has learned the mean and variance of the training dataset (StressID + facesData). When you Z-score your features relative to the user's personal baseline and then feed those Z-scores through the training StandardScaler, you are applying two different normalization transforms in sequence — the training dataset distribution and the user's personal distribution. These are incompatible. The result is features mapped to completely wrong positions in the model's decision space. The 23 tests pass because they use synthetic inputs constructed to look correct after this double-transform. A real user's data will not align.

The face model at 65.27% accuracy is borderline for a real-time system. A binary classifier needs to be above 75% to be trusted for clinical-adjacent use. 65% means it is wrong 35% of the time — one in three predictions is incorrect.

The autocorrelation-based jitter replacement for librosa.pyin introduced in the walkthrough is fast (less than 10ms) but less accurate than pyin. For a production system these are competing requirements. The report acknowledges this tradeoff but does not resolve it.

---

## PART 2 — YOUR FOUR QUESTIONS ANSWERED PRECISELY

---

### QUESTION 1: WebSockets vs SSE for Real-Time

**Current implementation:** SSE for fused output, HTTP POST for face and voice inputs.

**The honest answer:** For your specific architecture, SSE is the correct choice for the fused output stream. WebSockets are better only when you need bidirectional real-time communication — the server pushing AND the client pushing at the same time on the same persistent connection.

Your data flow is:
```
Client → Server: face POST (client pushes indicators)
Client → Server: voice POST (client pushes audio chunks)
Server → Client: fused SSE (server pushes fused score)
```

This is three separate unidirectional flows, not one bidirectional flow. SSE handles the server-to-client direction perfectly. The HTTP POSTs handle the client-to-server direction. Using a WebSocket here would mean multiplexing all three streams into one connection, adding complexity with no latency benefit.

**One real improvement:** Replace the HTTP POSTs with WebSocket messages for face and voice. The reason is not latency — it is connection overhead. Each HTTP POST requires a new TCP handshake (even with keep-alive there is HTTP header overhead per request). At 15fps with face sending every 2.5 seconds and voice sending every second, you have roughly 1.4 POST requests per second. Over a WebSocket this becomes 1.4 messages per second on one persistent connection with negligible header overhead.

**Verdict:** Keep SSE for fused output. Optionally upgrade face and voice inputs to WebSocket messages. This is a marginal improvement on a local network. For a production deployment on a real server it matters more.

---

### QUESTION 2: Training Data Patterns vs User Data — Are the Features Universal?

This is the most important question in the entire project. Let me answer it completely.

**How the training data collected patterns:**

facesData (facial): Static images of faces labelled stress/no-stress. The stress labels came from either acted expressions or annotation by human raters looking at faces. The MediaPipe landmarks were extracted offline from these images to create the 18-indicator training vectors.

StressID (voice + physio): Subjects performed public speaking, mental arithmetic, and Stroop tasks in a controlled lab. Audio was recorded with quality microphones in quiet rooms. EEG and GSR were recorded with clinical-grade sensors.

**What your system collects from the user:**

Face in real-time via a laptop webcam in ambient lighting with background clutter (as seen in the screenshot — window behind you, uneven lighting). The same 18 indicators computed from MediaPipe landmarks.

Voice via a laptop microphone with browser noiseSuppression and echoCancellation applied first.

**The domain gap between training and inference:**

| Factor | Training Data | Your User |
|---|---|---|
| Face lighting | Controlled studio | Ambient room, window backlight |
| Face angle | Frontal controlled | Whatever angle they sit |
| Audio equipment | Quality lab microphone | Laptop microphone, browser-processed |
| Room acoustics | Quiet lab | Normal room with reverb |
| Stress induction | Validated Stroop/arithmetic tasks | Unknown — self-reported |
| Subject demographics | StressID subjects (likely Western lab) | Tamil Nadu users |

**Are the 18 face indicators universal?**

Partially. Here is the breakdown by universality:

Highly universal (same across all humans regardless of ethnicity, face shape, or lighting):
- EAR (Eye Aspect Ratio) — ratio is scale-invariant
- Blink velocity — temporal derivative, scale-invariant
- Brow asymmetry — difference between left and right, cancels absolute scale
- Lip compression ratio — width normalized by height

Moderately universal but affected by person-specific anatomy:
- Brow descent — normalized by face height but baseline varies between people
- Jaw displacement — the nose-to-chin ratio varies with skull structure, hence the need for personal baseline calibration

Lighting-sensitive and therefore less universal:
- Landmark confidence — directly affected by lighting
- Nose wrinkle — subtle depth change, MediaPipe 2D projection loses accuracy

**The correct approach:** The features you have are a good selection. The universality problem is not in which features you chose but in the fact that the model was trained on one distribution of these features (from studio images and lab audio) and tested on another (real webcam and laptop mic). Calibration helps but does not fully bridge this gap.

**Are the 12 voice indicators universal?**

More universal than face indicators. The reason is that jitter, shimmer, HNR, and F0 deviation are properties of the vocal production mechanism — the laryngeal muscles, vocal fold tension, subglottal pressure. These physiological stress mechanisms are the same in a Tamil speaker and an English speaker and a German speaker. The specific F0 values differ (female vs male, different linguistic tonal patterns) but the *change* in F0 relative to baseline is consistent.

However: HNR, jitter, and shimmer measured through a laptop microphone with browser noiseSuppression applied are not the same values as those measured in a quiet room with a quality microphone. noiseSuppression changes the signal before your extractor sees it. This means your shimmer and HNR measurements are partially artifacts of the browser's noise processing, not pure vocal properties.

---

### QUESTION 3: How the Calibration Works — Does It Actually Work?

**How the current calibration works:**

Phase 1 (silence): Records ambient noise RMS. Sets silence threshold to 1.5 × noise_floor.

Phase 2 (voice, 5 chunks at 2s each = 10 seconds): Records F0 mean and std, RMS intensity, HNR at rest.

Phase 3 (face, 10 frames): Records baseline EAR, jaw displacement, brow descent at rest.

After calibration, incoming features are Z-scored against the personal baseline:
```
z = (x - personal_mean) / personal_std
```

Then the report says these Z-scores are fed through the "Inverse Scaling Trick" — scaled back to pseudo-raw inputs via the training scaler.

**The problem with the Inverse Scaling Trick:**

The training StandardScaler has parameters:
```
scaler.mean_  = [mean of each feature across training dataset]
scaler.scale_ = [std of each feature across training dataset]
```

What the inverse trick does:
```
pseudo_raw = z_personal * scaler.scale_ + scaler.mean_
```

This produces a value that, when fed through the scaler, maps to the same z-score as the personal deviation. In theory this is clever. In practice it breaks when the personal baseline differs significantly from the training dataset mean — which it will for Tamil Nadu users, female speakers, and people with unusual face geometry. The personal std may be much smaller than the training std, causing the pseudo-raw values to cluster tightly near the training mean and produce near-zero z-scores regardless of actual stress level.

**What actually works — the correct calibration approach:**

Do not use the training scaler at all for calibrated inference. Fit a new session-level scaler on the user's baseline data and use that for all predictions in the session. This means you need to collect enough baseline samples to fit a meaningful scaler — which requires more than 5 voice chunks and 10 face frames.

---

### QUESTION 4: What Degrades Production Quality

In order of impact:

**Degradation 1 — Face model at 65.27% accuracy**
A binary classifier that is wrong 35% of the time will produce unreliable fused scores. For a real-time display updating every 1-2 seconds, users will see erratic stress level changes that do not correspond to their actual state.

**Degradation 2 — Single scaler trained on dataset, used for all users**
Covered above. The training scaler's mean and std represent the StressID/facesData population, not your user.

**Degradation 3 — Autocorrelation jitter vs pyin**
The walkthrough replaced pyin (4.4 seconds → too slow) with autocorrelation + parabolic interpolation (less than 10ms → fast). But autocorrelation on 2-second chunks with browser-processed audio is noisier than pyin. Jitter values will be less accurate.

**Degradation 4 — No temporal context in predictions**
Every prediction is from a 2-second window in isolation. Stress is a temporal phenomenon — it builds and releases over minutes. A single high EAR reading followed by two low EAR readings means different things depending on context. The 4-sample rolling median partially addresses this but is not a real temporal model.

**Degradation 5 — No confidence calibration on model outputs**
GradientBoosting and RandomForest produce uncalibrated probabilities. A score of 0.80 from GradientBoosting does not mean "80% probability of stress" — it means "this input is in a region of feature space that GradientBoosting assigns high stress probability." For a Platt-scaled or isotonic-regression-calibrated model, the probability outputs are more meaningful. Currently your fusion engine treats all probability outputs as well-calibrated when they are not.

---

## PART 3 — CLINICAL-GRADE ARCHITECTURE UPGRADE

This section provides the complete upgrade. Each fix is targeted, implementable, and addresses a specific measured problem.

---

### UPGRADE 1 — Replace Double-Scaler Calibration with Session Scaler

**File:** `backend/calibration.py`  
**Problem:** Inverse scaling trick produces unreliable pseudo-raw inputs  
**Fix:** Fit a session-level StandardScaler from the user's own baseline data

```python
# In UserCalibration class, add:
from sklearn.preprocessing import StandardScaler
import numpy as np

class UserCalibration:
    def __init__(self):
        # ... existing fields ...
        self.voice_session_scaler = None
        self.face_session_scaler  = None
        self._voice_baseline_matrix = []  # rows of 12-dim feature vectors
        self._face_baseline_matrix  = []  # rows of 18-dim indicator vectors

    def add_voice_feature_vector(self, feature_vec: np.ndarray):
        """
        Call this during Phase 2 calibration with the raw 12-dim feature vector
        from voice_worker.py — NOT the indicators dict.
        """
        if feature_vec is not None and len(feature_vec) == 12:
            self._voice_baseline_matrix.append(feature_vec.copy())

    def add_face_feature_vector(self, feature_vec: np.ndarray):
        """
        Call this during Phase 3 calibration with the raw 18-dim feature vector
        built from face indicators.
        """
        if feature_vec is not None and len(feature_vec) == 18:
            self._face_baseline_matrix.append(feature_vec.copy())

    def build_session_scalers(self):
        """
        Fit session-level scalers on the user's own baseline data.
        These replace the training dataset scaler for this session.
        Requires at least 8 voice samples and 12 face frames for reliable fitting.
        """
        if len(self._voice_baseline_matrix) >= 8:
            X_voice = np.array(self._voice_baseline_matrix)
            self.voice_session_scaler = StandardScaler()
            self.voice_session_scaler.fit(X_voice)

        if len(self._face_baseline_matrix) >= 12:
            X_face = np.array(self._face_baseline_matrix)
            self.face_session_scaler = StandardScaler()
            self.face_session_scaler.fit(X_face)

        self.is_complete = (
            self.voice_session_scaler is not None and
            self.face_session_scaler  is not None
        )
        return self.is_complete

    def scale_voice_features(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Use the session scaler if calibrated, fall back to training scaler.
        The session scaler normalizes relative to THIS USER's calm state.
        """
        if self.voice_session_scaler is not None:
            return self.voice_session_scaler.transform(feature_vec.reshape(1, -1))
        return None  # signal caller to use training scaler

    def scale_face_features(self, feature_vec: np.ndarray) -> np.ndarray:
        if self.face_session_scaler is not None:
            return self.face_session_scaler.transform(feature_vec.reshape(1, -1))
        return None
```

**In `app.py` — update voice and face endpoints to use session scaler:**

```python
# In /api/stream/voice:
cal = get_or_create(user_id)
features = result['features']

session_scaled = cal.scale_voice_features(features)
if session_scaled is not None:
    # Use session scaler — prediction is relative to user's own calm state
    score = float(voice_expert.predict_proba(session_scaled)[0][1])
else:
    # Fall back to training scaler — pre-calibration or calibration failed
    score = float(voice_expert.predict_proba(
        voice_scaler.transform(features.reshape(1,-1)))[0][1])

# Same pattern for /api/stream/face
```

**Why this works:** The session scaler's mean and std represent what THIS USER's features look like at rest. Features deviating from their own calm state produce large z-scores. Features at their calm baseline produce near-zero z-scores. The model then predicts from a normalized distribution that is consistent regardless of whether the user is male, female, Tamil, Swedish, loud, quiet, or anything else.

**Calibration sample requirement:** Phase 2 needs at minimum 8 voice chunks (16 seconds of speech at 2-second chunks). Phase 3 needs minimum 12 face frames. Extend your calibration wizard accordingly.

---

### UPGRADE 2 — Probability Calibration on All Expert Models

**File:** `backend/training/train_face_expert.py`, `train_voice_expert_lightweight.py`  
**Problem:** GradientBoosting and RandomForest produce poorly-calibrated probabilities. 0.80 does not mean 80% likely stressed.  
**Fix:** Apply Platt scaling (logistic calibration) after training

```python
from sklearn.calibration import CalibratedClassifierCV

# After fitting the soft-voting ensemble:
face_model_raw = VotingClassifier(
    estimators=[('gb', gb), ('rf', rf), ('svm', svm)],
    voting='soft'
)
face_model_raw.fit(X_res_scaled, y_res)

# Wrap with Platt scaling calibration (cv='prefit' uses the already-trained model)
face_model_calibrated = CalibratedClassifierCV(
    face_model_raw,
    method='isotonic',  # isotonic is better than sigmoid for GBM
    cv='prefit'
)
# Calibrate on a held-out calibration set (not the training set)
face_model_calibrated.fit(X_cal_scaled, y_cal)

# Save the CALIBRATED model
with open('expert_models/face_expert_lightweight.pkl', 'wb') as f:
    pickle.dump(face_model_calibrated, f)
```

**Split your dataset into three parts:** train (60%), calibration (20%), test (20%). Train on train, calibrate on calibration, evaluate on test. This gives you a model where probability outputs are genuinely meaningful.

**Expected improvement:** After calibration, the fused score will be more stable and less prone to extreme values (0.02 or 0.98) for inputs that are borderline. The rolling median filter will also work better because the input values will be less erratic.

---

### UPGRADE 3 — Extend Calibration to Collect Feature Vectors, Not Just Indicator Statistics

**File:** `backend/app.py` — calibration endpoints  
**Problem:** Currently calibration collects indicator dicts (f0_mean, voice_intensity, etc.) but builds scalers only from summary statistics. The session scaler needs raw feature vectors.  
**Fix:** During calibration phases, collect the full feature vector

```python
# In /api/calibrate/voice_sample endpoint:
@app.route('/api/calibrate/voice_sample', methods=['POST'])
def calibrate_voice_sample():
    audio_bytes = request.data  # receive actual audio, not just indicators
    user_id = request.args.get('user_id', 'default')

    if not audio_bytes or len(audio_bytes) < 500:
        return jsonify({'status': 'too_short'}), 200

    result = extract_voice_stress_indicators(audio_bytes, f0_min=75, f0_max=400)
    if result is None:
        return jsonify({'status': 'silent'}), 200

    cal = get_or_create(user_id)
    cal.add_voice_sample(result['indicators'])          # existing — for statistics
    cal.add_voice_feature_vector(result['features'])    # NEW — for session scaler

    return jsonify({
        'status': 'ok',
        'samples': len(cal._voice_baseline_matrix),
        'indicators': result['indicators'],
    })

# In /api/calibrate/face_sample endpoint:
@app.route('/api/calibrate/face_sample', methods=['POST'])
def calibrate_face_sample():
    data = request.json or {}
    user_id = request.args.get('user_id', 'default')
    indicators = data.get('indicators', {})

    feature_vec = build_face_feature_vector(indicators)  # your existing function
    cal = get_or_create(user_id)
    cal.add_face_sample(indicators)                       # existing
    cal.add_face_feature_vector(feature_vec)              # NEW

    return jsonify({'status': 'ok', 'samples': len(cal._face_baseline_matrix)})

# In /api/calibrate/finalize:
@app.route('/api/calibrate/finalize', methods=['POST'])
def calibrate_finalize():
    data = request.json or {}
    user_id = data.get('user_id', 'default')
    cal = get_or_create(user_id)
    cal.finalize_voice()
    cal.finalize_face()
    ok = cal.build_session_scalers()  # NEW — build session-level scalers

    return jsonify({
        'status': 'complete' if ok else 'partial',
        'calibration': cal.to_dict(),
        'session_scalers_built': ok,
    })
```

---

### UPGRADE 4 — Replace Autocorrelation with Hybrid Fast Pitch Tracker

**File:** `backend/voice_worker.py`  
**Problem:** The autocorrelation replacement for pyin is fast but less accurate. The screenshot showed F0 at 266Hz (wrong). Need something faster than pyin but more accurate than basic autocorrelation.  
**Fix:** Use YIN algorithm — the clinical standard for real-time pitch detection. librosa has it built in and it is 10x faster than pyin.

```python
# Replace the entire F0 extraction block with:

def extract_f0_yin(y, sr, f0_min=75, f0_max=400):
    """
    YIN algorithm for F0 extraction.
    Faster than pyin, more accurate than autocorrelation.
    Industry standard for real-time vocal analysis.
    
    Returns: f0 array, voiced_flag array
    """
    try:
        # librosa.yin is deterministic, fast (~50ms for 2s audio), and accurate
        f0_yin = librosa.yin(
            y,
            fmin=f0_min,
            fmax=f0_max,
            sr=sr,
            frame_length=2048,
            hop_length=512,
            trough_threshold=0.1  # lower = more voiced frames detected
        )

        # YIN does not return voiced_flag — derive it from aperiodicity
        # Frames where F0 is within range are voiced
        voiced_flag = (f0_yin >= f0_min) & (f0_yin <= f0_max)

        # Clean up unvoiced frames
        f0_clean = f0_yin.copy()
        f0_clean[~voiced_flag] = np.nan

        return f0_clean, voiced_flag

    except Exception:
        return np.array([np.nan]), np.array([False])

# In extract_voice_stress_indicators(), replace the pyin call:
f0_track, voiced_flag = extract_f0_yin(y, sr, f0_min=f0_min, f0_max=f0_max)
f0_voiced = f0_track[~np.isnan(f0_track)]

indicators['f0_mean']  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
indicators['f0_std']   = float(np.std(f0_voiced))   if len(f0_voiced) > 0 else 0.0
indicators['f0_range'] = float(np.ptp(f0_voiced))   if len(f0_voiced) > 0 else 0.0

# F0-based RAP jitter from YIN track (more accurate than autocorrelation):
if len(f0_voiced) >= 5:
    periods = sr / (f0_voiced + 1e-10)
    period_diffs = np.abs(np.diff(periods))
    jitter_rap = float(np.mean(period_diffs) / (np.mean(periods) + 1e-10)) * 100
    indicators['jitter_percent']  = float(np.clip(jitter_rap, 0, 5.0))
    indicators['jitter_reliable'] = bool(jitter_rap < 3.0)
else:
    indicators['jitter_percent']  = 0.0
    indicators['jitter_reliable'] = False
```

**Expected latency:** librosa.yin on a 2-second 16kHz signal takes approximately 15-30ms — much faster than pyin (4.4 seconds), slightly slower than raw autocorrelation but significantly more accurate.

---

### UPGRADE 5 — Temporal Context Window for Fusion

**File:** `backend/score_buffer.py` and `backend/app.py`  
**Problem:** Each prediction is from a 2-second window in isolation. No temporal memory.  
**Fix:** Implement an exponential moving average (EMA) per modality inside the score buffer. EMA gives more weight to recent scores while remembering history, matching how stress actually builds and releases.

```python
# In score_buffer.py, update the write method:

class ScoreBuffer:
    EMA_ALPHA = 0.3  # weight given to new reading vs history
                     # 0.3 = 30% new, 70% history → smooth, responds in ~10 seconds
                     # 0.5 = 50% new, 50% history → faster response
                     # Tune based on review: faster for demos, slower for accuracy

    def __init__(self):
        self._lock  = threading.Lock()
        self._store = {}
        self._ema   = {}  # per-modality EMA state

    def write(self, modality: str, score: float, indicators: dict = None):
        with self._lock:
            # Compute EMA
            prev_ema = self._ema.get(modality, score)  # first reading bootstraps EMA
            new_ema  = self.EMA_ALPHA * score + (1 - self.EMA_ALPHA) * prev_ema
            self._ema[modality] = new_ema

            self._store[modality] = {
                'score':      score,       # raw instantaneous score
                'ema_score':  new_ema,     # smoothed score (use this for fusion)
                'indicators': indicators or {},
                'timestamp':  time.time(),
            }

    def read_all(self):
        now = time.time()
        with self._lock:
            return {
                k: v for k, v in self._store.items()
                if now - v['timestamp'] <= self.STALE_THRESHOLD_S
            }
```

**In the fusion engine, use `ema_score` instead of `score`:**
```python
# In /api/stream/fused SSE generator:
all_scores = score_buffer.read_all()
probs = {k: v['ema_score'] for k, v in all_scores.items()}  # use EMA not raw
```

**Clinical rationale:** Stress is not a binary flip. It builds over 30-60 seconds and releases over 60-120 seconds. A 2-second window with EMA alpha=0.3 effectively looks back approximately 6-8 seconds for stress detection, which is the appropriate clinical timescale for detecting acute stress onset.

---

### UPGRADE 6 — Increase Calibration Sample Count in Wizard UI

**File:** `frontend/src/components/CalibrationWizard.jsx`  
**Problem:** Current calibration collects 5 voice chunks (10 seconds) and 10 face frames. This is not enough to fit a reliable session scaler.  
**Fix:** Extend collection periods

```jsx
const PHASES = [
    {
        key:      'silence',
        duration: 15,  // unchanged — 15 seconds is enough for noise floor
        instruction: 'Stay completely silent. Do not speak or move.',
        why: 'Measuring your room noise level so voice analysis is calibrated to your environment.',
    },
    {
        key:      'voice',
        duration: 40,  // increased from 30 to 40 seconds
        // 40 seconds at 2s chunks = 20 voice samples (was 5 with old 10s duration)
        // 20 samples is sufficient for a reliable session scaler fit
        instruction: 'Read this aloud in your natural, relaxed speaking voice:\n\n"Today is a calm day. I am sitting comfortably. The weather is pleasant. I feel relaxed and at ease. My breathing is slow and steady."',
        why: 'Calibrating your personal pitch, speaking volume, and vocal quality baseline.',
    },
    {
        key:      'face',
        duration: 45,  // unchanged — but at 15fps every 2.5s = 18 frames, which is enough
        instruction: 'Look at the camera with a neutral, relaxed expression. You can blink normally. Do not smile or frown.',
        why: 'Calibrating your personal eye openness, brow position, and jaw resting state.',
    },
];
```

---

### UPGRADE 7 — Add Prediction Confidence Gate

**File:** `backend/app.py`  
**Problem:** Low-confidence face detections (bad lighting, partial occlusion) still produce and display stress scores, just with lower weight. These low-quality scores still affect the EMA and make the trend noisy.  
**Fix:** Gate predictions on both modality confidence AND model certainty

```python
# In /api/stream/face endpoint, after computing score:

landmark_conf = indicators.get('landmark_confidence', 1.0)
smile_score   = float(indicators.get('smile_score', 0.0))

# Gate 1: Landmark quality gate
# Below 0.5 confidence = MediaPipe is not seeing the face clearly
# Do not write to buffer at all — let the stale decay handle it
if landmark_conf < 0.5:
    return jsonify({
        'score': None,
        'reason': 'low_landmark_confidence',
        'confidence': landmark_conf
    })

# Gate 2: Model certainty gate
# If the model is near the decision boundary (0.40-0.60), it is uncertain
# Write to buffer but flag as uncertain for the fusion engine
certainty = abs(raw_score - 0.5) * 2  # 0.0 at boundary, 1.0 at extremes

# Gate 3: Smile dampening (existing, keep)
if smile_score > 0.3:
    dampening = smile_score * 0.4
    raw_score = max(0.0, raw_score - dampening)

# Write with certainty flag
score_buffer.write('face', smoothed_score, {
    **indicators,
    'certainty': certainty,
    'landmark_confidence': landmark_conf,
})

# In fusion engine, incorporate certainty into reliability weight:
# reliability weight = confidence × certainty × base_weight
```

---

### UPGRADE 8 — Fix the F0 Bounds to Use Personal Calibration

**File:** `backend/app.py` — voice stream endpoint  
**This was in CRITICAL_FIXES.md but the walkthrough does not confirm it was implemented.**

```python
@app.route('/api/stream/voice', methods=['POST'])
def stream_voice():
    audio_bytes = request.data
    user_id = request.args.get('user_id', 'default')

    # Always check silence first (fast, no librosa)
    # ... existing silence check ...

    # Get calibrated F0 bounds for this user
    cal = get_or_create(user_id)
    if cal.f0_mean is not None and cal.f0_mean > 60:
        f0_min = max(60,  cal.f0_mean * 0.40)
        f0_max = min(500, cal.f0_mean * 1.80)
    else:
        f0_min, f0_max = 75, 400  # safe default covering all human voices

    # Run extraction with user-specific bounds
    result = eventlet.tpool.execute(
        extract_voice_stress_indicators,
        audio_bytes,
        16000,
        f0_min,
        f0_max
    )
    # ... rest of endpoint ...
```

**This single change will fix the 266Hz F0 reading from the screenshot.**

---

## PART 4 — EXECUTION ORDER

Apply in this sequence. Each step is independent enough to test individually.

```
Step 1  UPGRADE 8 (F0 bounds from calibration) — fixes the 266Hz screenshot bug immediately
        Test: Run monitoring, speak, verify F0 shows 100-180Hz range

Step 2  UPGRADE 4 (YIN algorithm) — more accurate F0, better jitter
        Test: Verify jitter shows 0.2-0.8% calm, never hits 10.00% cap

Step 3  UPGRADE 1 + 2 (session scaler architecture) — the foundational calibration fix
        Test: Run calibration for two different people, verify their calm baselines
              produce ~0.2-0.3 fused score regardless of their absolute voice pitch

Step 4  UPGRADE 3 (collect feature vectors during calibration)
        Depends on UPGRADE 1 being in place first

Step 5  UPGRADE 5 (EMA temporal smoothing)
        Test: Watch fused score — should change smoothly not jump frame-to-frame

Step 6  UPGRADE 6 (extend calibration duration)
        Update wizard UI to 40-second voice phase

Step 7  UPGRADE 7 (confidence gate)
        Test: Cover camera with hand — score should stop updating, not show wrong value

Step 8  Retrain models with probability calibration (UPGRADE 2)
        This requires rerunning train_face_expert.py with CalibratedClassifierCV
        Test: Score of 0.80 should occur roughly 80% of the time on test set
```

---

## PART 5 — WHAT THE SYSTEM LOOKS LIKE AFTER ALL UPGRADES

**Face pipeline:**
```
Webcam → MediaPipe JS (browser GPU, 15fps) → 18 indicators
→ POST via Web Worker → /api/stream/face
→ Confidence gate (landmark_conf < 0.5 → reject)
→ Smile dampening (smile_score > 0.3 → reduce score)
→ Session scaler (user's own calm baseline) → Calibrated ensemble model
→ EMA smoothing → Score buffer → Fused SSE
```

**Voice pipeline:**
```
Mic → noiseSuppression (browser) → 16kHz WAV chunk every 2s
→ POST → /api/stream/voice
→ Silence gate (intensity < 1.5 × noise_floor → reject)
→ YIN pitch extraction (f0_min/f0_max from personal calibration)
→ F0-based RAP jitter → Shimmer → HNR → 12-feature vector
→ Session scaler (user's own calm baseline) → Voice expert model
→ EMA smoothing → Score buffer → Fused SSE
```

**Calibration:**
```
Phase 1 (15s silence): noise_floor → sets dynamic silence threshold
Phase 2 (40s neutral speech): 20 voice feature vectors → session voice scaler
Phase 3 (45s neutral face): 18 face feature vectors → session face scaler
Finalize: both scalers fitted → calibration complete
         All subsequent predictions normalized relative to THIS user's calm state
```

**What this means for your Tamil Nadu users:**
A male Tamil speaker with natural F0 of 130Hz sits calmly → session scaler fitted on his calm data → fused score ~0.20-0.30. He then does a Stroop task → F0 rises to 170Hz, jitter increases, EAR drops → same indicators normalized against his own baseline → session scaler maps these to high z-scores → model predicts stressed → fused score rises to 0.70-0.80. The model never needs to know what his absolute F0 is. It only knows he deviated from his own calm.

---

## SUMMARY: IS THE PROJECT BETTER NOW OR STILL DEGRADING?

**Current state (after all agent work):**
Infrastructure correct. Tests pass on synthetic data. 65.27% face accuracy. Calibration conceptually present but using wrong normalization method (inverse trick). F0 bounds not connected to calibration. YIN not used. No EMA temporal smoothing. No confidence gate.

**After this upgrade:**
Session scaler replaces inverse trick. YIN gives accurate F0 in 15-30ms. Calibrated F0 bounds eliminate the 266Hz bug. EMA makes scores temporally stable. Confidence gate prevents bad lighting from corrupting results. Face accuracy unchanged at 65.27% until retraining with CalibratedClassifierCV.

**The single most impactful change:** UPGRADE 1 (session scaler). Everything else improves reliability. The session scaler is what makes the calibration system actually work instead of appearing to work.
