# Phase 1 — Real-Time Lightweight Multimodal Stress Detection
## Architecture, Implementation Guide, and AI Agent Instructions

**Goal:** Run face + voice + physio stress detection simultaneously on 8GB RAM  
**Constraint:** No blocking, no crashes, no dataset-dependent raw feature learning  
**Target:** Real-time inference that generalizes to Tamil Nadu users without retraining

---

## WHY THE CURRENT APPROACH FAILS ON 8GB RAM

The current architecture is **synchronous and monolithic**:

```
User hits Analyze → Flask loads face model → runs MediaPipe → runs RF inference
                  → THEN loads voice model → runs librosa → runs RF inference
                  → THEN loads physio model → runs scipy → runs RF inference
                  → Returns result
```

Problems:
- All three models loaded into memory simultaneously on a single request thread
- librosa audio processing is CPU-heavy and blocks everything else
- MediaPipe loads its full graph on every request if not cached
- Flask handles one request at a time in dev mode
- React polls the backend — if backend is blocked, UI freezes

**Memory breakdown on current stack (approximate):**
```
Flask + Python base          ~150 MB
MediaPipe Face Mesh graph    ~180 MB
3x RandomForest models       ~80–200 MB each
librosa + numpy audio        ~120 MB per call
React dev server + Node      ~400 MB
Browser (Chrome)             ~600 MB
OS + background              ~1.5 GB
─────────────────────────────────────
Total worst case             ~4–5 GB → hits swap on 8GB, causes lag/crash
```

Adding simultaneous webcam + mic streams pushes this over 6GB easily.

---

## THE SOLUTION — 4 CORE PRINCIPLES

### Principle 1: Independent Async Workers Per Modality
Each modality runs in its own background thread. They never wait for each other.

### Principle 2: Score Buffer with Timeout
Each worker writes its latest score to a shared dict. Fusion reads from this dict every 2 seconds. If a modality has not updated in 10 seconds, it is marked stale and excluded from fusion.

### Principle 3: Computed Stress Indicators, Not Raw Features
Replace raw pixel/audio features with computed geometric and acoustic stress biomarkers. These are:
- Faster to compute (no large model inference for feature extraction)
- Smaller vectors (20–30 features instead of 80–140)
- Generalizable across ethnicity, skin tone, accent, and geography
- Directly interpretable for SHAP

### Principle 4: Frontend Drives Its Own Stream
Instead of the frontend sending video frames to the backend, the frontend runs lightweight face geometry extraction in-browser using MediaPipe JS (runs on GPU via WebGL), sends only the computed 20 numbers to Flask. This eliminates the biggest memory bottleneck — MediaPipe Python never runs on the server.

---

## NEW LIGHTWEIGHT ARCHITECTURE

```
Browser Side:
  Webcam → MediaPipe JS FaceMesh (WebGL, GPU) → 20 geometric values → POST /stream/face

  Mic → Web Audio API → compute RMS/ZCR in JS every 2s → POST /stream/voice/chunk
      → Backend receives audio blob → lightweight librosa (10 features only) → score

Server Side:
  /stream/face   → FaceWorker  → stress_score → ScoreBuffer['face']
  /stream/voice  → VoiceWorker → stress_score → ScoreBuffer['voice']
  /stream/physio → PhysioWorker→ stress_score → ScoreBuffer['physio']

  /stream/fused  → reads ScoreBuffer → returns fused result (SSE stream)

Frontend:
  EventSource('/stream/fused') → updates UI every 2 seconds automatically
```

---

## FEATURE REDESIGN — STRESS INDICATORS NOT RAW FEATURES

### Face: 18 Computed Geometric Stress Indicators

These are computed from MediaPipe JS landmark coordinates in the browser.
No image pixels are sent to the server. Only 18 float numbers per frame.

```
Compute in browser JavaScript from MediaPipe FaceMesh landmarks:

1.  left_EAR              Eye Aspect Ratio left  = (|p159-p145| + |p158-p153|) / (2 * |p33-p133|)
2.  right_EAR             Eye Aspect Ratio right = (|p386-p374| + |p385-p380|) / (2 * |p362-p263|)
3.  avg_EAR               Mean of left and right EAR
4.  blink_velocity         Rate of EAR change between last 3 frames (stress = reduced, irregular blinks)
5.  brow_descent_left     Vertical distance: inner left brow (55) to left eye center (159) — normalized
6.  brow_descent_right    Vertical distance: inner right brow (285) to right eye center (386) — normalized
7.  brow_asymmetry        |brow_descent_left - brow_descent_right| — tension marker
8.  lip_compression       Vertical lip gap (13→14) / lip width (61→291) — lower = more compressed
9.  jaw_tension           Jaw width (234→454) / face height (10→152) — higher = clenched
10. mouth_corner_pull     Distance of lip corners from nose tip — lower = stress-related retraction
11. nose_wrinkle          Distance between nose bridge and cheek — drops when nose wrinkles
12. face_height_norm      Face height / inter-ocular distance — normalization anchor
13. head_tilt             Angle of line between eye centers relative to horizontal
14. temporal_x_var        Variance of nose tip X position over last 5 frames — head movement/restlessness
15. temporal_y_var        Variance of nose tip Y position over last 5 frames
16. forehead_tension      Distance from inner brows to forehead midpoint — decreases when furrowed
17. eye_openness_ratio    avg_EAR normalized by that user's calibration baseline EAR
18. landmark_confidence   MediaPipe detection confidence score (0–1)
```

**Why these are generalizable:**
Geometric ratios between landmarks are scale-invariant and independent of skin tone, facial hair, or ethnicity. A Tamil face and a Swedish face both show brow descent and reduced EAR under stress. The normalization by face dimensions removes absolute size differences.

### Voice: 12 Computed Acoustic Stress Biomarkers

Computed server-side from 2-second audio chunks. librosa processes only 2 seconds at a time — fast and lightweight.

```
1.  f0_mean               Mean fundamental frequency (pitch) — rises under stress
2.  f0_std                Pitch variability — increases under stress
3.  f0_range              Max - min F0 in the chunk — compressed range = fatigue
4.  jitter_percent        Cycle-to-cycle F0 variation % — elevated under vocal stress
5.  shimmer_db            Cycle-to-cycle amplitude variation dB — elevated under stress
6.  hnr                   Harmonics-to-Noise Ratio — decreases under stress
7.  speaking_rate_proxy   Zero crossing rate normalized — proxy for speaking pace
8.  voice_intensity       RMS energy normalized — elevated under stress
9.  high_freq_ratio       Energy above 3kHz / total energy — stress elevates high freq content
10. spectral_flux          Frame-to-frame spectral change rate — elevated under stress
11. pause_ratio           Proportion of near-silent frames in chunk — increases under stress
12. voiced_fraction        Proportion of voiced frames — decreases in fragmented stress speech
```

**Why these are generalizable:**
These are properties of the vocal production mechanism (laryngeal muscle tension, subglottal pressure, vocal fold vibration irregularity). They manifest the same way in Tamil, English, Hindi, or any other language under stress because they are biomechanical, not linguistic.

### Physio: 12 EEG + GSR Stress Biomarkers (unchanged, already universal)

```
EEG (8 features):
1.  theta_alpha_ratio      Theta(4-8Hz)/Alpha(8-13Hz) — elevated under cognitive stress
2.  beta_alpha_ratio       Beta(13-30Hz)/Alpha — elevated under arousal/stress
3.  stress_index           (Beta+Gamma)/(Alpha+Theta) — composite stress index
4.  alpha_power_relative   Relative alpha power — decreases under stress
5.  frontal_asymmetry      AF7/AF8 alpha ratio — left-right frontal asymmetry
6.  theta_power_relative   Relative theta — increases under mental load
7.  gamma_relative         Relative gamma — attention/anxiety marker
8.  hjorth_mobility        Signal complexity measure

GSR (4 features):
9.  scl_mean              Skin conductance level — elevated baseline = stress
10. scr_rate              SCR peaks per minute — direct sympathetic activation count
11. scr_amplitude_mean    Mean SCR peak height — intensity of stress responses
12. scl_drift             Slope of SCL over window — rising = increasing stress
```

---

## IMPLEMENTATION — STEP BY STEP

---

### STAGE 1 — MediaPipe JS in Browser (Eliminates Python MediaPipe entirely)

**WHERE:** `frontend/src/components/FaceStream.jsx` (new file)  
**WHAT:** Run MediaPipe FaceMesh in browser, compute 18 stress indicators in JS, POST to backend  
**MEMORY SAVED:** ~180 MB (MediaPipe Python model never loads)

**HOW:**

Install MediaPipe JS in frontend:
```bash
cd frontend
npm install @mediapipe/face_mesh @mediapipe/camera_utils
```

Create `frontend/src/components/FaceStream.jsx`:

```jsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

// Compute 18 stress indicators from MediaPipe landmarks in browser
function computeStressIndicators(landmarks, imageWidth, imageHeight, history) {
  const pt = (i) => ({
    x: landmarks[i].x * imageWidth,
    y: landmarks[i].y * imageHeight,
  });
  const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

  const faceH = dist(pt(10), pt(152)) + 1e-6;
  const faceW = dist(pt(234), pt(454)) + 1e-6;
  const iod   = dist(pt(33), pt(263)) + 1e-6; // inter-ocular distance

  // EAR
  const earL = (dist(pt(159), pt(145)) + dist(pt(158), pt(153))) / (2 * dist(pt(33), pt(133)) + 1e-6);
  const earR = (dist(pt(386), pt(374)) + dist(pt(385), pt(380))) / (2 * dist(pt(362), pt(263)) + 1e-6);
  const avgEAR = (earL + earR) / 2;

  // Blink velocity from history
  const blinkVelocity = history.length >= 3
    ? Math.abs(avgEAR - history[history.length - 1]) / 0.033  // per second at 30fps
    : 0;

  // Brow descent (normalized by face height)
  const browDescL = dist(pt(55), pt(159)) / faceH;
  const browDescR = dist(pt(285), pt(386)) / faceH;
  const browAsym  = Math.abs(browDescL - browDescR);

  // Lip compression
  const lipGap   = dist(pt(13), pt(14));
  const lipWidth = dist(pt(61), pt(291)) + 1e-6;
  const lipCompression = lipGap / lipWidth;

  // Jaw tension
  const jawTension = faceW / faceH;

  // Mouth corner pull
  const noseTip = pt(4);
  const mcPull  = (dist(pt(61), noseTip) + dist(pt(291), noseTip)) / (2 * faceH);

  // Forehead tension
  const foreheadTension = dist(pt(10), pt(151)) / faceH;

  // Head tilt
  const eyeL = pt(33); const eyeR = pt(263);
  const headTilt = Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x) * (180 / Math.PI);

  // Temporal variance (nose tip movement)
  const nosePt = pt(4);
  const xVar = history.length >= 5
    ? history.slice(-5).reduce((acc, h) => acc + (h.nx - nosePt.x / imageWidth) ** 2, 0) / 5
    : 0;
  const yVar = history.length >= 5
    ? history.slice(-5).reduce((acc, h) => acc + (h.ny - nosePt.y / imageHeight) ** 2, 0) / 5
    : 0;

  const detection_confidence = landmarks[0]?.visibility ?? 0.9;

  return {
    left_ear: earL,
    right_ear: earR,
    avg_ear: avgEAR,
    blink_velocity: blinkVelocity,
    brow_descent_left: browDescL,
    brow_descent_right: browDescR,
    brow_asymmetry: browAsym,
    lip_compression: lipCompression,
    jaw_tension: jawTension,
    mouth_corner_pull: mcPull,
    forehead_tension: foreheadTension,
    face_height_norm: faceH / iod,
    head_tilt: Math.abs(headTilt),
    temporal_x_var: xVar,
    temporal_y_var: yVar,
    eye_openness_ratio: avgEAR,   // calibration applied server-side
    landmark_confidence: detection_confidence,
    nose_wrinkle: dist(pt(4), pt(50)) / faceH,
  };
}

const SEND_INTERVAL_MS = 2000;  // send every 2 seconds

export default function FaceStream({ onResult, active }) {
  const videoRef  = useRef(null);
  const camRef    = useRef(null);
  const meshRef   = useRef(null);
  const histRef   = useRef([]);
  const lastSend  = useRef(0);
  const [fps, setFps]     = useState(0);
  const frameCount = useRef(0);
  const fpsTimer   = useRef(null);

  const handleResults = useCallback((results) => {
    frameCount.current++;
    if (!results.multiFaceLandmarks?.length) return;

    const lm = results.multiFaceLandmarks[0];
    const iw = videoRef.current?.videoWidth  || 640;
    const ih = videoRef.current?.videoHeight || 480;

    const indicators = computeStressIndicators(lm, iw, ih, histRef.current);

    // Update history (keep last 10 frames)
    histRef.current = [
      ...histRef.current.slice(-9),
      { nx: lm[4].x, ny: lm[4].y, ear: indicators.avg_ear },
    ];

    // Send to backend every SEND_INTERVAL_MS
    const now = Date.now();
    if (now - lastSend.current > SEND_INTERVAL_MS) {
      lastSend.current = now;
      fetch('/api/stream/face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indicators, timestamp: now }),
      })
        .then(r => r.json())
        .then(data => onResult && onResult(data))
        .catch(() => {});
    }
  }, [onResult]);

  useEffect(() => {
    if (!active) {
      camRef.current?.stop();
      return;
    }

    const mesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });
    mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: false,  // false = faster, saves ~20ms per frame
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    mesh.onResults(handleResults);
    meshRef.current = mesh;

    const cam = new Camera(videoRef.current, {
      onFrame: async () => {
        await mesh.send({ image: videoRef.current });
      },
      width: 320,   // lower resolution = much less memory and CPU
      height: 240,
    });
    cam.start();
    camRef.current = cam;

    // FPS counter
    fpsTimer.current = setInterval(() => {
      setFps(frameCount.current);
      frameCount.current = 0;
    }, 1000);

    return () => {
      cam.stop();
      clearInterval(fpsTimer.current);
    };
  }, [active, handleResults]);

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <video ref={videoRef} style={{ width: 320, height: 240, borderRadius: 8,
                                      background: '#000', display: 'block' }} />
      {active && (
        <div style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,0,0,0.6)',
                       color: '#4CAF50', borderRadius: 4, padding: '2px 8px',
                       fontSize: '0.75rem', fontFamily: 'monospace' }}>
          {fps} fps · Face
        </div>
      )}
    </div>
  );
}
```

**KEY DECISIONS EXPLAINED:**
- `refineLandmarks: false` — saves ~20ms per frame, still gives all 468 points
- `width: 320, height: 240` — half resolution, cuts memory by 4x, still plenty for landmarks
- Sends only 18 numbers per POST, not a full image — eliminates image transfer overhead
- Computes everything in JS on browser GPU (WebGL), zero Python memory for face geometry

---

### STAGE 2 — Lightweight Voice Chunk Processing

**WHERE:** `backend/voice_worker.py` (new), `backend/app.py` (new endpoint)  
**WHAT:** Process 2-second audio chunks with a minimal 12-feature extractor instead of full 140-feature librosa pipeline  
**MEMORY SAVED:** ~80 MB (smaller audio buffers, no full spectrogram)

**HOW:**

Create `backend/voice_worker.py`:

```python
import numpy as np
import librosa
import io
import soundfile as sf

def extract_voice_stress_indicators(audio_bytes, sr_target=16000):
    """
    Extract 12 acoustic stress biomarkers from a raw audio chunk.
    Designed for 1-3 second chunks. Fast, lightweight, generalizable.
    
    Returns: dict with 12 named indicators + numpy array for model input
    """
    try:
        # Load audio — accepts webm, wav, ogg, mp3
        audio_buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buf, sr=sr_target, mono=True, duration=3.0)
    except Exception:
        return None

    if len(y) < sr_target * 0.5:  # less than 0.5 seconds — skip
        return None

    indicators = {}
    EPS = 1e-10

    # 1-3: F0 (fundamental frequency / pitch)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=2048
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        indicators['f0_mean']  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        indicators['f0_std']   = float(np.std(f0_voiced))   if len(f0_voiced) > 0 else 0.0
        indicators['f0_range'] = float(np.ptp(f0_voiced))   if len(f0_voiced) > 0 else 0.0
    except Exception:
        indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0

    # 4-5: Jitter and Shimmer via autocorrelation approximation
    # Fast approximation — avoids praat dependency for real-time use
    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len   = int(sr * 0.010)  # 10ms hop
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)

    # Period estimation per frame using autocorrelation peak
    periods = []
    amplitudes = []
    for frame in frames.T:
        ac = np.correlate(frame, frame, mode='full')[frame_len - 1:]
        ac = ac / (ac[0] + EPS)
        # Find first peak after lag 0
        min_lag = int(sr / 500)  # 500 Hz max
        max_lag = int(sr / 60)   # 60 Hz min
        if max_lag < len(ac):
            peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
            periods.append(peak_idx)
        amplitudes.append(np.sqrt(np.mean(frame ** 2)))

    periods = np.array(periods, dtype=float)
    amplitudes = np.array(amplitudes, dtype=float)

    jitter  = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS)) if len(periods) > 1 else 0.0
    shimmer = float(np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + EPS)) if len(amplitudes) > 1 else 0.0

    indicators['jitter_percent'] = min(jitter * 100, 10.0)  # cap at 10%
    indicators['shimmer_db']     = min(shimmer * 20, 5.0)   # approximate dB scale

    # 6: HNR approximation via autocorrelation
    ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
    ac_norm = ac_full / (ac_full[0] + EPS)
    min_period = int(sr / 400)
    max_period = int(sr / 80)
    if max_period < len(ac_norm):
        peak_val = np.max(ac_norm[min_period:max_period])
        hnr = 10 * np.log10(peak_val / (1 - peak_val + EPS) + EPS)
    else:
        hnr = 0.0
    indicators['hnr'] = float(np.clip(hnr, -20, 30))

    # 7: Speaking rate proxy (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    indicators['speaking_rate_proxy'] = float(np.mean(zcr))

    # 8: Voice intensity
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    indicators['voice_intensity'] = float(np.mean(rms))

    # 9: High frequency ratio (stress elevates high-freq content)
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_len))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    high_mask = freqs >= 3000
    total_energy = np.sum(stft) + EPS
    indicators['high_freq_ratio'] = float(np.sum(stft[high_mask]) / total_energy)

    # 10: Spectral flux
    spectral_flux = np.mean(np.diff(stft, axis=1) ** 2)
    indicators['spectral_flux'] = float(np.clip(spectral_flux, 0, 1))

    # 11: Pause ratio (near-silent frames)
    silence_thresh = 0.01 * np.max(np.abs(y))
    pause_frames = np.sum(rms < silence_thresh)
    indicators['pause_ratio'] = float(pause_frames / (len(rms) + EPS))

    # 12: Voiced fraction
    try:
        voiced_frac = float(np.sum(voiced_flag) / (len(voiced_flag) + EPS))
    except Exception:
        voiced_frac = 0.5
    indicators['voiced_fraction'] = voiced_frac

    # Feature vector for model (fixed order, 12 features)
    feature_vec = np.array([
        indicators['f0_mean'],
        indicators['f0_std'],
        indicators['f0_range'],
        indicators['jitter_percent'],
        indicators['shimmer_db'],
        indicators['hnr'],
        indicators['speaking_rate_proxy'],
        indicators['voice_intensity'],
        indicators['high_freq_ratio'],
        indicators['spectral_flux'],
        indicators['pause_ratio'],
        indicators['voiced_fraction'],
    ], dtype=np.float32)

    return {'indicators': indicators, 'features': feature_vec}
```

Add to `backend/app.py`:

```python
from voice_worker import extract_voice_stress_indicators
import threading

@app.route('/api/stream/voice', methods=['POST'])
def stream_voice():
    audio_bytes = request.data  # raw audio blob from browser MediaRecorder
    if not audio_bytes:
        return jsonify({'error': 'no audio'}), 400

    result = extract_voice_stress_indicators(audio_bytes)
    if result is None:
        return jsonify({'score': None, 'reason': 'too_short'}), 200

    features = result['features'].reshape(1, -1)
    score = float(voice_expert.predict_proba(features)[0][1])

    score_buffer['voice'] = {
        'score': score,
        'indicators': result['indicators'],
        'timestamp': time.time(),
    }

    return jsonify({'score': score, 'indicators': result['indicators']})
```

---

### STAGE 3 — Score Buffer + Async Fusion Engine

**WHERE:** `backend/score_buffer.py` (new), `backend/app.py` (new endpoints)  
**WHAT:** Thread-safe shared buffer that holds latest score per modality, fusion reads from it independently

**HOW:**

Create `backend/score_buffer.py`:

```python
import threading
import time

class ScoreBuffer:
    """
    Thread-safe buffer holding the latest stress score per modality.
    Each modality writes independently. Fusion reads from all simultaneously.
    Scores older than STALE_THRESHOLD_S are excluded from fusion.
    """
    STALE_THRESHOLD_S = 15  # seconds before a score is considered stale

    def __init__(self):
        self._lock  = threading.Lock()
        self._store = {}  # modality -> {score, indicators, timestamp}

    def write(self, modality: str, score: float, indicators: dict = None):
        with self._lock:
            self._store[modality] = {
                'score':      score,
                'indicators': indicators or {},
                'timestamp':  time.time(),
            }

    def read(self, modality: str):
        with self._lock:
            entry = self._store.get(modality)
            if entry is None:
                return None
            if time.time() - entry['timestamp'] > self.STALE_THRESHOLD_S:
                return None  # stale
            return entry

    def read_all(self):
        now = time.time()
        with self._lock:
            return {
                k: v for k, v in self._store.items()
                if now - v['timestamp'] <= self.STALE_THRESHOLD_S
            }

    def clear(self, modality: str = None):
        with self._lock:
            if modality:
                self._store.pop(modality, None)
            else:
                self._store.clear()

# Singleton instance shared across all Flask routes
score_buffer = ScoreBuffer()
```

Add to `backend/app.py`:

```python
from score_buffer import score_buffer
import json

# ── Face stream endpoint ─────────────────────────────────────────────────────
@app.route('/api/stream/face', methods=['POST'])
def stream_face():
    data = request.json or {}
    indicators = data.get('indicators', {})

    if not indicators:
        return jsonify({'score': None}), 200

    # Build 18-feature vector from browser-computed indicators
    feature_vec = np.array([
        indicators.get('left_ear', 0.3),
        indicators.get('right_ear', 0.3),
        indicators.get('avg_ear', 0.3),
        indicators.get('blink_velocity', 0),
        indicators.get('brow_descent_left', 0.1),
        indicators.get('brow_descent_right', 0.1),
        indicators.get('brow_asymmetry', 0),
        indicators.get('lip_compression', 0.2),
        indicators.get('jaw_tension', 0.7),
        indicators.get('mouth_corner_pull', 0.3),
        indicators.get('forehead_tension', 0.1),
        indicators.get('face_height_norm', 1.5),
        indicators.get('head_tilt', 2.0),
        indicators.get('temporal_x_var', 0),
        indicators.get('temporal_y_var', 0),
        indicators.get('eye_openness_ratio', 0.3),
        indicators.get('landmark_confidence', 0.9),
        indicators.get('nose_wrinkle', 0.1),
    ], dtype=np.float32).reshape(1, -1)

    score = float(face_expert.predict_proba(feature_vec)[0][1])

    score_buffer.write('face', score, indicators)

    return jsonify({'score': score})


# ── Fused score endpoint (Server-Sent Events) ─────────────────────────────────
@app.route('/api/stream/fused')
def stream_fused():
    """
    SSE stream — browser connects once, receives updated fused score every 2 seconds.
    No polling needed. Much lower overhead than repeated POST requests.
    """
    def generate():
        while True:
            all_scores = score_buffer.read_all()

            if not all_scores:
                data = json.dumps({'status': 'waiting', 'modalities_active': 0})
            else:
                probs = {k: v['score'] for k, v in all_scores.items()}
                confs = {k: v['indicators'].get('landmark_confidence', 0.7)
                         for k, v in all_scores.items()}

                fused = fuse_predictions(probs, confs, fusion_mode='reliability')
                fused['modalities_active'] = len(all_scores)
                fused['per_modality'] = {
                    k: {'score': round(v['score'], 3)} for k, v in all_scores.items()
                }
                data = json.dumps(fused)

            yield f'data: {data}\n\n'
            time.sleep(2)

    return app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )
```

---

### STAGE 4 — Retrain Face Expert on 18-Feature Stress Indicators

**WHERE:** `backend/train_face_expert_lightweight.py` (new file)  
**WHAT:** Since face features are now 18 numbers (not MediaPipe Python extracted from images), you need a dataset of 18-indicator vectors with stress/no-stress labels.

**The challenge and solution:**

You cannot run the browser JS computation offline on facesData images. Instead, use a hybrid approach:

**Step A — Convert facesData using Python MediaPipe (one-time offline):**
Run MediaPipe Python on all 12,275 facesData images once, extract the same 18 geometric values. This gives you a training CSV.

Create `backend/extract_face_indicators_offline.py`:

```python
"""
One-time script. Run once to create face_indicators_train.csv and face_indicators_test.csv.
These are used to train the lightweight 18-feature face expert.
After running, delete this script — it is not needed at runtime.
"""
import mediapipe as mp
import cv2
import numpy as np
import os
import csv

mp_face_mesh = mp.solutions.face_mesh

def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))

def compute_18_indicators(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=False,
                                min_detection_confidence=0.5) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm  = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm])

    faceH = dist(pts, 10, 152) + 1e-6
    faceW = dist(pts, 234, 454) + 1e-6
    iod   = dist(pts, 33, 263) + 1e-6

    earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
    earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
    avgEAR = (earL + earR) / 2

    return [
        earL,
        earR,
        avgEAR,
        0.0,                                          # blink_velocity — static image, set 0
        dist(pts, 55, 159) / faceH,                  # brow_descent_left
        dist(pts, 285, 386) / faceH,                 # brow_descent_right
        abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,  # brow_asymmetry
        dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
        faceW / faceH,                                # jaw_tension
        (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
        dist(pts, 10, 151) / faceH,                  # forehead_tension
        faceH / iod,                                  # face_height_norm
        0.0,                                          # head_tilt — compute if needed
        0.0,                                          # temporal_x_var — static image
        0.0,                                          # temporal_y_var — static image
        avgEAR,                                       # eye_openness_ratio
        0.9,                                          # landmark_confidence (detected)
        dist(pts, 4, 50) / faceH,                    # nose_wrinkle
    ]

def process_split(root_dir, split, output_csv):
    rows = []
    for label_name, label_val in [('stress', 1), ('nostress', 0)]:
        folder = os.path.join(root_dir, split, label_name)
        if not os.path.exists(folder):
            print(f'WARNING: {folder} not found')
            continue
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f'Processing {len(images)} images from {folder}...')
        for img_name in images:
            path = os.path.join(folder, img_name)
            indicators = compute_18_indicators(path)
            if indicators is not None:
                rows.append(indicators + [label_val])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
            'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
            'lip_compression', 'jaw_tension', 'mouth_corner_pull',
            'forehead_tension', 'face_height_norm', 'head_tilt',
            'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
            'landmark_confidence', 'nose_wrinkle', 'label',
        ]
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {output_csv}')

if __name__ == '__main__':
    FACES_ROOT = r'f:\Multimodal_stress_Detection\facesData'  # update path
    process_split(FACES_ROOT, 'train', 'face_indicators_train.csv')
    process_split(FACES_ROOT, 'test',  'face_indicators_test.csv')
    print('Done. Run train_face_expert_lightweight.py next.')
```

Create `backend/train_face_expert_lightweight.py`:

```python
"""
Train an 18-feature lightweight face stress expert.
Requires: face_indicators_train.csv (generated by extract_face_indicators_offline.py)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os

TRAIN_CSV = 'face_indicators_train.csv'
TEST_CSV  = 'face_indicators_test.csv'
OUT_MODEL = os.path.join('expert_models', 'face_expert_lightweight.pkl')
OUT_SCALER = os.path.join('expert_models', 'face_scaler_lightweight.pkl')

FEATURES = [
    'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
    'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
    'lip_compression', 'jaw_tension', 'mouth_corner_pull',
    'forehead_tension', 'face_height_norm', 'head_tilt',
    'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
    'landmark_confidence', 'nose_wrinkle',
]

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

X_train = df_train[FEATURES].values
y_train = df_train['label'].values
X_test  = df_test[FEATURES].values
y_test  = df_test['label'].values

print(f'Train: {len(X_train)} samples  |  Test: {len(X_test)} samples')
print(f'Class balance — Train: {dict(zip(*np.unique(y_train, return_counts=True)))}')

# SMOTE for class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# Model — GradientBoosting is slightly better than RF on small feature sets
# and is fast at inference time
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train_scaled, y_train_res)

# Evaluate
y_pred = model.predict(X_test_scaled)
print('\nTest Results:')
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Cross-validation on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train_res, cv=cv, scoring='f1')
print(f'\n5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# Save
os.makedirs('expert_models', exist_ok=True)
with open(OUT_MODEL,  'wb') as f: pickle.dump(model, f)
with open(OUT_SCALER, 'wb') as f: pickle.dump(scaler, f)
print(f'\nSaved: {OUT_MODEL}  {OUT_SCALER}')
```

---

### STAGE 5 — Retrain Voice Expert on 12-Feature Indicators

Create `backend/train_voice_expert_lightweight.py`:

```python
"""
Extract 12 stress indicators from StressID audio files and train voice expert.
Handles the full StressID audio directory structure.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle

from voice_worker import extract_voice_stress_indicators

STRESSID_AUDIO = r'f:\Multimodal_stress_Detection\StressID\StressID Dataset\Audio'
OUT_MODEL  = os.path.join('expert_models', 'voice_expert_lightweight.pkl')
OUT_SCALER = os.path.join('expert_models', 'voice_scaler_lightweight.pkl')

# StressID label convention — adjust to match actual folder names
STRESS_CONDITIONS = ['public_speaking', 'mental_math', 'stroop', 'stress']
CALM_CONDITIONS   = ['rest', 'baseline', 'relax', 'calm', 'nostress']

rows = []
for root, dirs, files in os.walk(STRESSID_AUDIO):
    for fname in files:
        if not fname.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            continue
        fpath = os.path.join(root, fname)
        # Determine label from folder name
        folder_lower = root.lower()
        label = None
        for sc in STRESS_CONDITIONS:
            if sc in folder_lower:
                label = 1
                break
        if label is None:
            for cc in CALM_CONDITIONS:
                if cc in folder_lower:
                    label = 0
                    break
        if label is None:
            continue  # skip unknown labels

        with open(fpath, 'rb') as f:
            audio_bytes = f.read()
        result = extract_voice_stress_indicators(audio_bytes)
        if result is not None:
            rows.append(list(result['features']) + [label])
            print(f'  Processed: {fname} → label={label}')

FEATURE_NAMES = [
    'f0_mean', 'f0_std', 'f0_range', 'jitter_percent', 'shimmer_db',
    'hnr', 'speaking_rate_proxy', 'voice_intensity', 'high_freq_ratio',
    'spectral_flux', 'pause_ratio', 'voiced_fraction', 'label',
]
df = pd.DataFrame(rows, columns=FEATURE_NAMES)
print(f'\nTotal samples: {len(df)}')
print(df['label'].value_counts())

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_res_s  = scaler.fit_transform(X_res)
X_test_s = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=42)
model.fit(X_res_s, y_res)

y_pred = model.predict(X_test_s)
print('\nTest Results:')
print(classification_report(y_test, y_pred, target_names=['Calm', 'Stress']))

os.makedirs('expert_models', exist_ok=True)
with open(OUT_MODEL,  'wb') as f: pickle.dump(model, f)
with open(OUT_SCALER, 'wb') as f: pickle.dump(scaler, f)
print(f'Saved: {OUT_MODEL}  {OUT_SCALER}')
```

---

### STAGE 6 — Frontend SSE Consumer + Real-Time UI

**WHERE:** `frontend/src/components/RealtimeMonitor.jsx` (new)  
**WHAT:** Connects to `/api/stream/fused` via SSE, updates stress display in real time without polling

**HOW:**

Create `frontend/src/components/RealtimeMonitor.jsx`:

```jsx
import React, { useEffect, useState, useRef } from 'react';
import FaceStream from './FaceStream';
import WaveformRecorder from './WaveformRecorder';

const LEVEL_COLOR = { Low: '#4CAF50', Moderate: '#FF9800', High: '#F44336' };

export default function RealtimeMonitor() {
  const [active,  setActive]  = useState(false);
  const [result,  setResult]  = useState(null);
  const [history, setHistory] = useState([]);
  const esRef = useRef(null);

  const startMonitoring = () => {
    setActive(true);

    // Connect to SSE fused stream
    const es = new EventSource('/api/stream/fused');
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.fused_score !== undefined) {
          setResult(data);
          setHistory(h => [...h.slice(-29), {
            t: new Date().toLocaleTimeString(),
            score: Math.round(data.fused_score * 100),
          }]);
        }
      } catch (_) {}
    };
    es.onerror = () => es.close();
    esRef.current = es;
  };

  const stopMonitoring = () => {
    setActive(false);
    esRef.current?.close();
  };

  // Send voice chunks to backend every 2 seconds
  const handleVoiceChunk = async (blob) => {
    try {
      await fetch('/api/stream/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/webm' },
        body: blob,
      });
    } catch (_) {}
  };

  useEffect(() => () => esRef.current?.close(), []);

  const levelColor = LEVEL_COLOR[result?.stress_level] || '#888';

  return (
    <div style={{ padding: 16, fontFamily: 'Segoe UI, sans-serif' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        {!active ? (
          <button onClick={startMonitoring} style={{
            background: '#2E75B6', color: '#fff', border: 'none',
            padding: '12px 28px', borderRadius: 8, fontSize: '1rem', cursor: 'pointer',
          }}>
            ▶ Start Monitoring
          </button>
        ) : (
          <button onClick={stopMonitoring} style={{
            background: '#D32F2F', color: '#fff', border: 'none',
            padding: '12px 28px', borderRadius: 8, fontSize: '1rem', cursor: 'pointer',
          }}>
            ⏹ Stop
          </button>
        )}
        {result && (
          <div style={{
            background: levelColor + '22', border: `2px solid ${levelColor}`,
            borderRadius: 8, padding: '10px 20px', fontWeight: 700,
            color: levelColor, fontSize: '1.1rem',
          }}>
            {result.stress_level} — {Math.round((result.fused_score || 0) * 100)}%
          </div>
        )}
      </div>

      {/* Streams */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 20 }}>
        <div>
          <FaceStream active={active} onResult={() => {}} />
          <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 4, textAlign: 'center' }}>
            Face (browser GPU)
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <WaveformRecorder
            continuous={active}
            chunkIntervalMs={2000}
            onChunk={handleVoiceChunk}
          />
        </div>
      </div>

      {/* Per-modality scores */}
      {result?.per_modality && (
        <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
          {Object.entries(result.per_modality).map(([mod, v]) => (
            <div key={mod} style={{
              background: '#EEF4FB', borderRadius: 8, padding: '8px 14px',
              border: '1px solid #C8DCEF', minWidth: 100, textAlign: 'center',
            }}>
              <div style={{ fontSize: '0.75rem', color: '#666', textTransform: 'capitalize' }}>{mod}</div>
              <div style={{ fontWeight: 700, color: '#1F3864', fontSize: '1.1rem' }}>
                {Math.round(v.score * 100)}%
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Mini trend chart — last 30 readings */}
      {history.length > 1 && (
        <svg width="100%" height={60} style={{ display: 'block', marginTop: 8 }}>
          {history.map((h, i) => {
            const x = (i / (history.length - 1)) * 100;
            const y = 55 - (h.score / 100) * 50;
            return i === 0 ? null : (
              <line key={i}
                x1={`${(( i - 1) / (history.length - 1)) * 100}%`}
                y1={55 - (history[i - 1].score / 100) * 50}
                x2={`${x}%`} y2={y}
                stroke={h.score > 65 ? '#F44336' : h.score > 35 ? '#FF9800' : '#4CAF50'}
                strokeWidth={2}
              />
            );
          })}
        </svg>
      )}

      {result?.status === 'waiting' && (
        <p style={{ color: '#888', fontStyle: 'italic' }}>
          Waiting for modality data...
        </p>
      )}
    </div>
  );
}
```

Update `WaveformRecorder` to support continuous chunked recording:

In `WaveformRecorder.jsx`, add a `continuous` prop mode that creates a new MediaRecorder every `chunkIntervalMs` and calls `onChunk` with each blob, instead of recording until stopped.

---

### STAGE 7 — Memory Optimization Checklist for 8GB RAM

Apply these configuration changes:

**`backend/app.py`:**
```python
# Load models once at startup, not per-request
import pickle, os

def load_expert(filename):
    path = os.path.join('expert_models', filename)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load at module level — once, cached for all requests
face_expert   = load_expert('face_expert_lightweight.pkl')
face_scaler   = load_expert('face_scaler_lightweight.pkl')
voice_expert  = load_expert('voice_expert_lightweight.pkl')
voice_scaler  = load_expert('voice_scaler_lightweight.pkl')
physio_expert = load_expert('physio_expert.pkl')
physio_scaler = load_expert('physio_scaler.pkl')
```

**`backend/app.py` — production server:**
```python
# Replace Flask dev server with waitress for better threading
# pip install waitress
from waitress import serve
if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000, threads=4)
```

Add to `backend/requirements.txt`:
```
waitress>=2.1.0
```

**`frontend/package.json` — reduce Node memory:**
Add to scripts:
```json
"start": "cross-env NODE_OPTIONS=--max_old_space_size=512 react-scripts start"
```
```bash
npm install cross-env
```

**Estimated memory after optimization:**
```
Flask + waitress (4 threads)     ~100 MB
3x GradientBoosting models       ~15–30 MB each (much lighter than RF)
MediaPipe JS in browser (GPU)    ~0 MB server RAM
React (capped at 512 MB)         ~300 MB
Browser                          ~400 MB
OS                               ~1.5 GB
─────────────────────────────────────────
Total                            ~2.5–3 GB  ← fits comfortably in 8GB
```

---

## TRAINING EXECUTION ORDER

```
Step 1  Run extract_face_indicators_offline.py
        → Produces: face_indicators_train.csv, face_indicators_test.csv
        → Time: ~30–60 min on 12,275 images

Step 2  Run train_face_expert_lightweight.py
        → Produces: expert_models/face_expert_lightweight.pkl
        →           expert_models/face_scaler_lightweight.pkl
        → Time: ~5 min

Step 3  Run train_voice_expert_lightweight.py
        → Produces: expert_models/voice_expert_lightweight.pkl
        →           expert_models/voice_scaler_lightweight.pkl
        → Time: ~10–20 min depending on audio file count

Step 4  Existing physio expert from train_model.py remains usable
        (physio features are already EEG/GSR numbers, not images)
        Retrain only if you change physiological feature extraction.

Step 5  Run app.py → confirm all 3 experts load at startup
Step 6  Open browser → Start Monitoring → verify face + voice run simultaneously
```

---

## VERIFY PHASE 1 IS COMPLETE

```
[ ] FaceStream component renders webcam at 320x240 with FPS indicator
[ ] Face landmarks computed in browser — Chrome DevTools shows no image sent to server
[ ] POST /api/stream/face returns score in <50ms
[ ] POST /api/stream/voice returns score in <500ms for 2s audio chunk
[ ] GET /api/stream/fused SSE stream emits events every 2 seconds
[ ] Both face and voice running simultaneously: browser RAM stays below 2GB total
[ ] face_expert_lightweight.pkl trained on 18 indicators — inference shape (1,18)
[ ] voice_expert_lightweight.pkl trained on 12 indicators — inference shape (1,12)
[ ] Score buffer correctly returns None for stale scores (>15 seconds old)
[ ] Fused score updates in RealtimeMonitor UI without page reload
[ ] Mini trend line chart updates in real time
[ ] Stopping monitoring closes SSE connection cleanly
[ ] waitress server handles 4 concurrent requests without blocking
```

---

## WHY THIS GENERALIZES TO TAMIL NADU USERS

| Old Approach | New Approach |
|---|---|
| Trains on raw pixels from facesData (Western lab faces) | Trains on geometric ratios — scale/ethnicity/lighting independent |
| MFCC fingerprints of StressID audio (specific speakers) | F0 jitter/shimmer/HNR — physiological vocal stress markers, not speaker identity |
| Model learns "this texture = stress" | Model learns "brow descent + reduced EAR + elevated jitter = stress" |
| Fails when user's face looks different from training set | Generalizes because the stress mechanism is universal |
| Accent-dependent voice features | Accent-independent vocal biomarkers |

A Tamil Nadu user showing brow furrowing, reduced blinking, compressed lips, elevated vocal pitch, and increased jitter will score high stress on this system — the same way a user from any other population would — because these are the physiological expressions of stress, not cultural ones.
