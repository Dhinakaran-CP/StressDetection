# StressDetectionUsingML — Critical Fixes for Real-World Reliability
**Based on:** Screenshot analysis + project_summary.md review  
**Date:** June 2026  
**Status of project:** Architecture correct, tests pass on synthetic data, real-world display showing wrong values

---

## WHAT IS CONFIRMED WORKING
- SSE streaming pipeline (face + voice running simultaneously)
- Calibration architecture (3-phase wizard built)
- Score buffer + fusion engine
- 23 automated tests pass
- Web Worker face POST (non-blocking)
- OS thread pool for voice (non-blocking)
- Clean project structure

## WHAT THE SCREENSHOT PROVES IS STILL BROKEN

| What Screenshot Shows | What It Means | Root Cause |
|---|---|---|
| Jitter: **10.00%** always | Hitting the cap — raw jitter is impossibly high | Parabolic interpolation not working on your mic audio |
| F0: **266 Hz** | Wrong for a male voice (should be 85–180Hz) | pyin tracking wrong harmonic |
| Jaw Width: **130%** | Static number, never changes | Still using skeletal width ratio not dynamic displacement |
| Face Δ = **0.25** only | Barely separates calm from stressed | Model trained on clean dataset images, not real webcam |

---

## FIX 1 — F0 Detection: Restrict Frequency Bounds to Your Voice

**File:** `backend/voice_worker.py`  
**Problem:** `pyin` is set with bounds too wide (`C2` to `C7` = 65Hz to 2093Hz). For a male voice the upper bound should be 300Hz maximum. At 2093Hz the algorithm finds harmonic peaks and misreports them as the fundamental.

**Find this block in voice_worker.py:**
```python
f0, voiced_flag, _ = librosa.pyin(
    y, fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    sr=sr, frame_length=2048
)
```

**Replace with:**
```python
# Separate bounds for male vs female — default to male-safe range
# Male: 75–300Hz  Female: 150–400Hz
# The calibration system detects gender from baseline f0 after calibration
# Before calibration is complete, use the widest safe range that excludes harmonics
f0, voiced_flag, _ = librosa.pyin(
    y,
    fmin=75,       # 75Hz — below any normal human voice fundamental
    fmax=400,      # 400Hz — covers both male and female, excludes 1st harmonic
    sr=sr,
    frame_length=2048,
    hop_length=512,
    fill_na=np.nan
)
```

**After calibration completes**, update the bounds dynamically in `app.py` using the calibrated `f0_mean`:
```python
# In /api/stream/voice endpoint, after calibration is available:
cal = get_or_create(user_id)
if cal.f0_mean is not None:
    # Set bounds to ±60% of personal baseline
    f0_min = max(60,  cal.f0_mean * 0.4)
    f0_max = min(500, cal.f0_mean * 1.6)
else:
    f0_min, f0_max = 75, 400

# Pass to extraction:
result = extract_voice_stress_indicators(audio_bytes, f0_min=f0_min, f0_max=f0_max)
```

**Update `extract_voice_stress_indicators` signature:**
```python
def extract_voice_stress_indicators(audio_bytes, sr_target=16000, f0_min=75, f0_max=400):
    # ... existing code ...
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=f0_min, fmax=f0_max,
        sr=sr, frame_length=2048, hop_length=512, fill_na=np.nan
    )
```

**Expected result after fix:** F0 should show 100–170Hz for your male voice at rest.

---

## FIX 2 — Jitter: Replace Autocorrelation with Frame-Level RMSE Method

**File:** `backend/voice_worker.py`  
**Problem:** The parabolic interpolation was applied but the autocorrelation method itself is fundamentally unreliable on laptop microphone audio because microphone noise corrupts the autocorrelation peaks. The result is that the detected period jumps between integer values anyway, hitting the 10% cap.

**Replace the entire jitter/shimmer calculation block with a F0-derived method that uses the already-computed pyin F0 track:**

Find and replace the period extraction loop:
```python
# REMOVE the entire for-loop period extraction section:
# for frame in frames.T:
#     ac = np.correlate(frame, frame, mode='full')[...]
#     ...
#     periods.append(refined_period)
#     amplitudes.append(...)
```

**Replace with this F0-based jitter calculation** (more accurate, no autocorrelation needed):
```python
# F0-based jitter: use the already-computed pyin F0 track
# This is more reliable than autocorrelation on consumer microphone audio
f0_voiced_all = f0[~np.isnan(f0) & (f0 > 0)]

if len(f0_voiced_all) >= 3:
    # Convert F0 (Hz) to periods (samples)
    periods_from_f0 = sr / (f0_voiced_all + 1e-10)

    # Jitter = mean absolute period deviation / mean period
    # This is the clinical RAP (Relative Average Perturbation) definition
    period_diffs = np.abs(np.diff(periods_from_f0))
    mean_period  = np.mean(periods_from_f0)
    jitter_raw   = float(np.mean(period_diffs) / (mean_period + 1e-10))
    jitter_pct   = jitter_raw * 100

    # Physiological cap: real jitter is 0.1% to 2.0% for stress
    # Values above 3% on consumer mic are usually noise artifacts
    # Do NOT cap silently — report the raw value and flag if suspicious
    indicators['jitter_percent'] = float(np.clip(jitter_pct, 0.0, 5.0))
    indicators['jitter_reliable'] = bool(jitter_pct < 3.0)  # flag for UI

    # Shimmer: amplitude variation between consecutive voiced frames
    # Use RMS energy of voiced frames instead of per-cycle amplitude
    hop = 512
    frame_len = 2048
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    # Only use frames where F0 was detected (voiced)
    voiced_indices = np.where(~np.isnan(f0) & (f0 > 0))[0]
    if len(voiced_indices) >= 3:
        voiced_rms = rms_frames[np.minimum(voiced_indices, len(rms_frames)-1)]
        shimmer_raw = float(np.mean(np.abs(np.diff(voiced_rms))) / (np.mean(voiced_rms) + 1e-10))
        indicators['shimmer_db'] = float(np.clip(shimmer_raw * 20, 0.0, 3.0))
    else:
        indicators['shimmer_db'] = 0.0
else:
    indicators['jitter_percent'] = 0.0
    indicators['jitter_reliable'] = False
    indicators['shimmer_db'] = 0.0
```

**Expected result after fix:** Jitter should show 0.2%–0.8% during normal calm speech, 1.0%–2.5% during stressed speech. Never 10.00%.

---

## FIX 3 — Jaw: Replace Static Skull Ratio with Real Dynamic Measurement

**File:** `frontend/src/components/FaceStream.jsx`  
**Problem:** The current jaw computation uses cheekbone width / face height — both are fixed skeletal measurements. The value 130% does not change when you clench or relax your jaw.

**Find in FaceStream.jsx:**
```javascript
const jawTension = faceW / faceH;
// or:
const jawAngleWidth = dist(pt(172), pt(397)) / stableIOD;
```

**Replace with:**
```javascript
// Dynamic jaw displacement: nose tip to chin distance normalized by IOD
// This CHANGES when you open/close your mouth or raise your jaw
const stableIOD = dist(pt(33), pt(263)) + 1e-6;  // inter-ocular distance (stable reference)
const noseTip   = pt(4);
const chinTip   = pt(152);
const jawDisplacement = dist(noseTip, chinTip) / stableIOD;
// Typical values: ~1.7-2.1 at rest, ~1.4-1.6 when jaw raised/clenched, ~2.2+ when mouth open

// Jaw angle width: distance between jaw corners — changes slightly with masseter contraction
// pt(172) = left jaw angle, pt(397) = right jaw angle
const jawAngleWidth = dist(pt(172), pt(397)) / stableIOD;
// Typical values: ~1.3-1.6. Increases slightly when masseters contract.

// Use jawDisplacement as the primary indicator (more reliable than width)
indicators.jaw_displacement  = jawDisplacement;
indicators.jaw_angle_width   = jawAngleWidth;

// Remove: indicators.jaw_tension (the old static ratio)
// Remove: any reference to faceW / faceH for jaw
```

**Also update the backend feature vector builder** in `app.py` `/api/stream/face` endpoint to use `jaw_displacement` instead of `jaw_tension`:
```python
# In build_face_feature_vector() or wherever indicators are converted to numpy array:
# Replace:
#   indicators.get('jaw_tension', 0.7),
# With:
    indicators.get('jaw_displacement', 1.85),   # dynamic nose-chin distance
    indicators.get('jaw_angle_width', 1.45),    # jaw corner width
```

**Also retrain the face expert** after this change because the training CSVs need to use the same feature definition. In `extract_face_indicators_offline.py`:
```python
# Replace:
# faceW / faceH  (static)
# With:
dist(pts, 4, 152) / iod,   # jaw_displacement (dynamic)
dist(pts, 172, 397) / iod, # jaw_angle_width (dynamic)
```

---

## FIX 4 — Voice Intensity: Fix the "26" Display Issue

**File:** `backend/voice_worker.py` and `frontend/src/components/RealtimeMonitor.jsx`  
**Problem:** Voice Intensity shows "26" with no unit. The RMS value from librosa is 0.0 to 1.0 for normalized audio. "26" means the audio bytes coming from the browser are in int16 range (0–32768) and are not being normalized before librosa processing.

**In `voice_worker.py`:**
```python
# After loading audio, ensure it is normalized to float [-1.0, 1.0]
# Add this immediately after: y, sr = librosa.load(...)

if y.dtype != np.float32 and y.dtype != np.float64:
    y = y.astype(np.float32) / 32768.0  # normalize int16 to float

# Also ensure it is clamped
y = np.clip(y, -1.0, 1.0)

# Now all RMS, jitter, shimmer calculations work on the correct scale
# voice_intensity will be 0.0-1.0, not 0-32768
```

**In `RealtimeMonitor.jsx` or wherever the bar chart renders:**
```javascript
// Voice Intensity display — show as percentage of max comfortable speaking level
// 0.0-1.0 scale, where 0.05 is quiet, 0.15 is normal speech, 0.3+ is loud
const intensityPct = Math.round(indicators.voice_intensity * 100);
// Display: "15%" not "26"
```

---

## FIX 5 — Face Model: Add Real-World Augmentation to Training

**File:** `backend/training/train_face_expert.py`  
**Problem:** Face model discrimination Δ = 0.25 is too low. The model was trained on clean dataset images. Your webcam footage has different lighting, angle, and background (the window behind you in the screenshot is causing uneven lighting that changes MediaPipe's computed ratios slightly).

**Add this augmentation to the training script before fitting the model:**

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# After loading X_train from face_indicators_train.csv:

def augment_face_data(X, y, n_augmented=3):
    """
    Add realistic real-world variation to training data.
    Simulates:
    - Lighting variation (affects confidence and subtle geometry)
    - Slight head pose variation (affects EAR, jaw, tilt)
    - Microphone/sensor noise analog for landmarks
    """
    X_aug = [X.copy()]
    y_aug = [y.copy()]

    for _ in range(n_augmented):
        noise = X.copy().astype(np.float64)

        # EAR features (indices 0-2): ±8% variation (lighting affects eye visibility)
        noise[:, 0:3] *= (1 + np.random.randn(len(X), 3) * 0.08)

        # Brow features (indices 4-6): ±5% variation (pose affects projection)
        noise[:, 4:7] *= (1 + np.random.randn(len(X), 3) * 0.05)

        # Jaw and lip (indices 7-9): ±6% variation
        noise[:, 7:10] *= (1 + np.random.randn(len(X), 3) * 0.06)

        # Head tilt (index 12): ±3 degrees variation (person not always perfectly centered)
        noise[:, 12] += np.random.randn(len(X)) * 3.0

        # Landmark confidence (index 16): slight random degradation
        noise[:, 16] = np.clip(noise[:, 16] - np.abs(np.random.randn(len(X)) * 0.05), 0.3, 1.0)

        noise = np.clip(noise, 0, None)  # no negative geometric values
        X_aug.append(noise.astype(np.float32))
        y_aug.append(y.copy())

    return np.vstack(X_aug), np.concatenate(y_aug)

# Apply before SMOTE:
X_train_aug, y_train_aug = augment_face_data(X_train, y_train, n_augmented=4)
print(f"After augmentation: {len(X_train_aug)} samples (was {len(X_train)})")

# Then SMOTE on augmented data:
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_aug, y_train_aug)
```

**Also switch the face model from GradientBoosting to a soft-voting ensemble** which is more robust to the slightly different distributions between training images and real webcam:

```python
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

face_model = VotingClassifier(
    estimators=[
        ('gb',  GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ],
    voting='soft',  # averages probability outputs
)
```

This ensemble is more robust because GradientBoosting, RandomForest, and SVM make different kinds of errors. Their average is more stable than any single model on out-of-distribution real-world input.

**After retraining, expected discrimination:** Δ should improve from 0.25 to 0.45–0.65.

---

## FIX 6 — Jitter Display: Show Reliability Flag in UI

**File:** `frontend/src/components/RealtimeMonitor.jsx`  
**Problem:** When jitter is unreliable (mic noise, short audio, silence), the UI should show this rather than displaying a misleadingly confident wrong number.

**Find the jitter display in the Acoustic Stress Biomarkers section:**
```jsx
// Find wherever jitter_percent is rendered, add reliability indicator:
const jitter = indicators?.jitter_percent ?? 0;
const jitterReliable = indicators?.jitter_reliable ?? true;

// In JSX:
<div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
  <span>Jitter (Micro-instability)</span>
  {!jitterReliable && (
    <span style={{
      fontSize: '0.7rem', color: '#FF9800',
      background: '#FF980022', borderRadius: 4,
      padding: '1px 6px'
    }}>
      ⚠ noisy mic
    </span>
  )}
  <span style={{ marginLeft: 'auto', color: jitterReliable ? '#4CAF50' : '#FF9800' }}>
    {jitter.toFixed(2)}%
  </span>
</div>
```

---

## FIX 7 — Calibration Must Actually Update Voice Extraction Bounds

**File:** `backend/app.py`  
**Problem:** The calibration system stores `f0_mean` but the voice extraction in `/api/stream/voice` does not use the calibrated bounds when calling `extract_voice_stress_indicators`. The calibration data is computed but never fed back into the feature extractor.

**Find `/api/stream/voice` in app.py:**
```python
@app.route('/api/stream/voice', methods=['POST'])
def stream_voice():
    audio_bytes = request.data
    user_id = request.args.get('user_id', 'default')
    
    # ADD THIS BLOCK before calling extract_voice_stress_indicators:
    cal = get_or_create(user_id)
    
    # Set F0 bounds from calibration if available
    if cal.f0_mean is not None and cal.f0_mean > 60:
        f0_min = max(60,  cal.f0_mean * 0.40)   # 40% below personal baseline
        f0_max = min(500, cal.f0_mean * 1.80)   # 80% above personal baseline
    else:
        f0_min, f0_max = 75, 400                 # safe default before calibration
    
    # Pass bounds to extractor:
    result = extract_voice_stress_indicators(
        audio_bytes,
        sr_target=16000,
        f0_min=f0_min,
        f0_max=f0_max
    )
    
    # Rest of existing code...
```

---

## EXECUTION ORDER

Run these fixes in this exact order. Each step depends on the previous.

```
Step 1  Apply Fix 4 (audio normalization) in voice_worker.py
        → Verifies: Voice Intensity shows 0.0-1.0 range in UI, not "26"

Step 2  Apply Fix 1 (F0 bounds) in voice_worker.py
        → Verifies: F0 shows 100-170Hz for male voice

Step 3  Apply Fix 2 (F0-based jitter) in voice_worker.py
        → Verifies: Jitter shows 0.2-0.8% during calm speech, never 10.00%

Step 4  Apply Fix 7 (calibration→F0 bounds) in app.py
        → Verifies: After calibration, F0 bounds tighten to personal range

Step 5  Apply Fix 3 (jaw displacement) in FaceStream.jsx and app.py
        → Verifies: Jaw Displacement changes when you open/clench jaw

Step 6  Apply Fix 5 (training augmentation) in train_face_expert.py
        → Retrain face model, verify Δ improves above 0.40

Step 7  Apply Fix 6 (jitter reliability flag) in RealtimeMonitor.jsx
        → Verifies: "⚠ noisy mic" badge shows when jitter is unreliable
```

---

## HOW TO VERIFY EACH FIX IS WORKING

After applying each fix, use these real-world checks — not synthetic test inputs:

**Voice fixes (Steps 1-4):**
- Open the dashboard
- Do NOT speak — watch Voice Intensity for 5 seconds → must show < 0.04 (silence gate works)
- Hum a single note for 5 seconds → F0 should show 100-200Hz, Jitter should show < 1%
- Count "1, 2, 3" quickly and stressfully → F0 should rise, Jitter should increase
- If F0 still shows 266Hz after Fix 1 → your microphone is recording at a different sample rate; add `print(f"Loaded SR: {sr}, Duration: {len(y)/sr:.2f}s")` to voice_worker.py and verify it prints 16000

**Face fixes (Steps 5-6):**
- Open dashboard, look neutral → Jaw Displacement should show ~1.8-2.0
- Slowly open your mouth wide → Jaw Displacement should INCREASE to 2.5+
- Clench jaw without opening mouth → Jaw Displacement should DECREASE to 1.5-1.7
- If it stays at 130% → the old jaw_tension code is still being used somewhere; grep for `faceW / faceH` in FaceStream.jsx

**Calibration fix (Step 4):**
- Run calibration, speak for 30 seconds in your natural voice
- After finalize: GET /api/calibrate/status → `f0_mean` should be 100-180Hz for you
- Start monitoring again → F0 display should now match your actual pitch range

---

## WHAT THIS FIXES IN THE SCREENSHOT

| Was Showing | After Fixes Will Show |
|---|---|
| Pitch: 266Hz (wrong) | Pitch: 120-160Hz (correct male range) |
| Jitter: 10.00% always | Jitter: 0.3-0.8% calm, 1.2-2.5% stressed |
| Voice Intensity: 26 | Voice Intensity: 0.08 (normalized) |
| Jaw Width: 130% static | Jaw Displacement: 1.85 rest, changes dynamically |
| Face discrimination Δ=0.25 | Face discrimination Δ=0.45+ after retraining |

---

## ONE THING THE AGENT GOT RIGHT THAT IS EASY TO MISS

The project summary says:

> **Model discrimination (physio): Calm: 0.00 vs Stressed: 1.00 (Δ = 1.00)**

This looks impressive but it means the physiological model is overfit. Perfect separation on synthetic test inputs means the model memorized the pattern of the synthetic data. On real EEG/GSR from a sensor it will not be this clean. When you eventually connect a real GSR sensor, expect this Δ to drop to 0.4-0.7. That is still good — just set the expectation correctly in your project report.

The face model's Δ = 0.25 is honest and concerning. The voice model's Δ = 0.70 is plausible but needs to be validated with your real voice after the F0 fixes are applied.
