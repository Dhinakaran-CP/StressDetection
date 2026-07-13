# Multimodal Face–Voice Stress Detection: End‑to‑End Architecture & Methodology

> Design goal: never redo extraction. This architecture defines **exactly** how to go from raw videos/audios to a real‑time face+voice stress engine, including subject‑aware windowed datasets, LOSO evaluation, and a deployable late‑fusion model. [web:29][web:38]

## 1. High‑Level System Architecture

### 1.1 Runtime Components (Real‑Time Engine)

- **Client (React/Web Frontend)**  
  - Captures webcam frames (RGB) at configurable FPS (e.g., 5–10 fps) and microphone audio in streaming chunks (e.g., 1 s windows).  
  - Uses a Web Worker to encode frames as base64 and send them to the backend without blocking the UI.  
  - Streams audio in small chunks over WebSocket or batched REST.  
  - Subscribes to **Server‑Sent Events (SSE)** or Socket.IO channel for real‑time stress predictions and SHAP‑based explanations per modality. [web:29]

- **Backend (Flask + Eventlet + Socket.IO)**  
  - **Face pipeline**:  
    - Accepts base64 frames → decodes to RGB.  
    - Runs MediaPipe/TensorFlow face landmarks model.  
    - Computes geometric Action Units (AUs) (EAR, MAR, brow distance, etc.).  
    - Maintains sliding window buffers (e.g., last 10–15 s) of AU features per session.
  - **Voice pipeline**:  
    - Accepts streaming audio chunks (1 s windows).  
    - Runs Librosa + custom feature extractor for  
      pitch statistics, jitter, shimmer, HNR, spectral flux, MFCC statistics, pause ratio. [web:38]  
    - Maintains sliding window buffers per session.
  - **Physio pipeline (optional)**:  
    - Accepts EEG/GSR/PPG features from separate devices or preprocessed streams. [web:28]  
  - **Expert models**:  
    - `face_expert`: gradient boosting model over facial AUs.  
    - `voice_expert`: gradient boosting model over acoustic biomarkers.  
    - `physio_expert`: calibrated soft‑voting ensemble over EEG/GSR features.  
  - **Fusion engine**:  
    - Takes calibrated class probabilities from each expert.  
    - Applies **temporal smoothing** via score buffers (e.g., 15 s exponential decay).  
    - Combines via **weighted late fusion** (e.g., weights W_face, W_voice, W_physio tuned on validation). [web:29]  
    - Outputs stress class (e.g., low/med/high) and per‑modality contributions.
  - **Explainability (SHAP)**:  
    - Pre‑compute SHAP values offline for each expert model.  
    - At runtime, map the most important features to human labels ("Left brow tension", "High vocal jitter").

### 1.2 Offline Components (Training & Evaluation)

- **Raw data layer**  
  - Original StressID or equivalent dataset: synchronized **video, audio, labels, subject IDs, task types**. [web:29][web:39]

- **Feature extraction layer**  
  - Face extractor: window‑level AU features with `subject_id`, `task_id`, `video_id`, `window_start`, `window_end`.  
  - Voice extractor: same identifiers + acoustic biomarkers per window. [web:38]  
  - Physio extractor: EEG/GSR/PPG features per window with same identifiers. [web:28]

- **Training datasets**  
  - Clean, denormalized CSV/Parquet files for **each modality** plus a **multimodal join table** aligning face, voice, and physio windows.

- **Evaluation layer**  
  - Implements LOSO and temporal cross‑validation.  
  - Trains expert models and fusion engine with proper subject‑level splits to avoid leakage. [web:28][web:33]


## 2. Data Extraction Architecture (Face & Voice)

### 2.1 Design Principles

1. **Windowed, not averaged**: every row corresponds to a short, fixed window (e.g., 1–2 s) rather than an entire video. This allows temporal modeling and accurate simulation of real‑time smoothing. [web:29][web:38]
2. **Subject & task metadata in every row**: `subject_id`, `task_id`, `video_id`, `window_index`. This is mandatory to support LOSO, task‑specific analysis, and temporal segmentation. [web:39]  
3. **Synchronization across modalities**: face and voice windows use the **same time grid** (e.g., window start times). This enables clean late fusion and aligned multimodal training. [web:29][web:38]
4. **Deterministic pipelines**: same code paths for offline extraction and runtime streaming, so feature definitions cannot drift over time.

### 2.2 Unified Time Windowing

- Choose a **window length** L and **hop size** H (e.g., L = 1.0 s, H = 0.5 s).  
- For each labeled trial (e.g., 60 s video:
  - Generate windows: \(t_0 = 0\), \(t_1 = H\), ..., up to \(t_n = T - L\).  
  - Each window has index `k` and timestamps `start = k*H`, `end = k*H + L`.
- Align all modalities to this grid: the k‑th window of face, voice, and physio share the same `[subject_id, task_id, video_id, window_index]`. [web:29][web:38]

### 2.3 Face Feature Extraction Pipeline

**Input**: raw video file with known FPS and subject/task labels.

**Steps**:
1. **Frame sampling**  
   - Read frames at dataset FPS (or downsample to 15–25 fps if original FPS is higher).  
   - Attach timestamps using frame index / FPS.
2. **Face detection & landmarking**  
   - Use MediaPipe Face Mesh or Tasks to get 3D landmarks per frame. [web:29][web:38]  
   - If face not detected, mark frame as missing and interpolate AU features where needed.
3. **Geometric feature computation**  
   For each frame, compute:  
   - Eye Aspect Ratios (left/right EAR).  
   - Mouth Aspect Ratio (MAR).  
   - Brow raise and frown distances.  
   - Lip corner pull, jaw drop, etc.  
   These map to 10–20 interpretable "Action Units" used in prior audiovisual stress work. [web:34]
4. **Window aggregation**  
   For each time window `[start, end]`:
   - Collect all frames within the window.  
   - For each AU feature, compute **statistics**: mean, std, 25th/75th percentile, min/max, fraction of frames above AU threshold (e.g., proportion of frames with eyes narrowed).  
   - Optionally, compute simple temporal derivatives (difference between first and last frame values).
5. **Row construction**  
   For each window, output a row:

   ```text
   subject_id, task_id, video_id, window_index, window_start, window_end,
   label,  # stress label for that window (see 2.6)
   face_EAR_left_mean, face_EAR_left_std, ...,
   face_brow_tension_mean, ...
   ```

6. **Output file**  
   - `face_windows_stressid.csv` (or Parquet) containing one row per window.

### 2.4 Voice Feature Extraction Pipeline

**Input**: raw audio (WAV) aligned with each video, with subject/task labels.

**Steps**:
1. **Preprocessing**  
   - Resample to 16 kHz mono.  
   - Normalize amplitude; optionally remove leading/trailing silence.
2. **Windowing**  
   - Use the **same window grid** as face (same `[start, end]` times).  
   - Extract the corresponding audio segment for each window.
3. **Frame‑level features within a window**  
   For each window’s audio:
   - Compute short‑time pitch (F0) via autocorrelation or YAAPT.  
   - Extract jitter and shimmer (cycle‑to‑cycle variation measures).  
   - Compute Harmonics‑to‑Noise Ratio (HNR).  
   - Calculate MFCCs (e.g., 13 coefficients) + delta MFCC statistics.  
   - Compute spectral features: spectral centroid, bandwidth, rolloff, spectral flux.  
   - Compute voice activity / pause ratio (fraction of frames with energy above threshold). [web:38]
4. **Window aggregation**  
   For each feature, compute statistics over the window: mean, std, percentiles, min/max.
5. **Row construction**  
   For each window:

   ```text
   subject_id, task_id, video_id, window_index, window_start, window_end,
   label,
   voice_pitch_mean, voice_pitch_std, voice_jitter_local, voice_shimmer_local,
   voice_HNR_mean, voice_MFCC1_mean, ..., voice_pause_ratio, ...
   ```

6. **Output file**  
   - `voice_windows_stressid.csv` containing one row per window.

### 2.5 Multimodal Join & Physio Alignment

- Extract physio features (EEG, GSR, HRV/PPG) on the **same window grid**. [web:28][web:33]  
- For EEG, compute band powers (delta, theta, alpha, beta) and ratios like beta/alpha for each region or channel group. [web:24]  
- For GSR, compute mean SCL, sum of phasic responses, and peak counts per window. [web:28]
- Join all modalities on the shared keys:

```text
subject_id, task_id, video_id, window_index, window_start, window_end
→ left‑join face, voice, physio feature tables
```

- Output a **multimodal table**: `multimodal_windows_stressid.csv` with columns for all modalities.

### 2.6 Window Labeling Strategy

- If dataset provides **frame‑level** or **time‑aligned** labels, use majority label over the window.
- If labels are **trial‑level** (e.g., entire video = high stress), propagate that label to every window.
- Optionally mark **transition windows** at the edges of tasks as "ignore" to reduce label noise.
- Store label as both categorical (e.g., `low`, `high`) and numerical (e.g., 0/1/2) for model flexibility.


## 3. Offline Training & Validation Methodology

### 3.1 Expert Models per Modality

For each modality, train an expert model on its **window‑level** features.

- **Face expert**  
  - Model: Gradient Boosted Trees (e.g., XGBoost/LightGBM) or Random Forest. [web:29][web:34]  
  - Input: facial AU statistics per window.  
  - Output: calibrated probabilities for each stress class.  
  - Feature selection: remove highly correlated features; optionally use ANOVA/Chi2 or mRMR. [web:28]

- **Voice expert**  
  - Model: similar tree‑based classifier or shallow DNN.  
  - Input: acoustic biomarkers per window. [web:38]  
  - Output: calibrated class probabilities.  
  - Handle speaker imbalance via class weighting or SMOTE.

- **Physio expert**  
  - Model: calibrated soft‑voting ensemble (GBM + RF). [web:28]  
  - Input: EEG band powers, beta/alpha ratio, GSR phasic features, HRV features if available. [web:24][web:28]

### 3.2 Strict Subject‑Level Evaluation (No Leakage)

To avoid the kind of data leakage previously encountered, enforce **subject‑level splits**:

- **LOSO (Leave‑One‑Subject‑Out)**  
  - For each subject S:  
    - Train all expert models on windows from all other subjects.  
    - Evaluate on all windows from subject S.  
  - Aggregate performance over subjects (mean accuracy, F1, AUC). [web:28][web:33]

- **Alternative K‑Fold with grouped CV**  
  - Use GroupKFold with `subject_id` as the grouping key so no subject appears in both train and validation sets.

- **Temporal simulation**  
  - Within each subject’s test windows, sort by `window_start` and pass them through the same **score_buffer** logic used in real‑time inference (e.g., exponential decay over 15 s).  
  - This simulates how the system behaves online.

### 3.3 Fusion Engine Training

- Define fusion weights or a small meta‑classifier:

  - **Option A: Fixed weighted average**  
    - Choose weights \(w_f, w_v, w_p\) (face, voice, physio) based on validation performance, with \(w_f + w_v + w_p = 1\). [web:29]  
    - Final probability:  
      \(P_{final}(c) = w_f P_f(c) + w_v P_v(c) + w_p P_p(c)\).

  - **Option B: Meta‑learner**  
    - Train a logistic regression or shallow GBM whose inputs are concatenated expert probabilities, outputting final class probabilities.

- Training procedure:
  - In each LOSO fold, train experts on training subjects.  
  - Compute expert probabilities for validation subjects.  
  - Fit fusion weights/meta‑learner only on validation probabilities; evaluate on held‑out subject(s). [web:29]

- Ensure that no window from the test subject leaks into expert or fusion training.

### 3.4 Explainable AI (SHAP) Setup

- For each expert model (face, voice, physio):
  - Sample a large number of training windows.  
  - Compute SHAP values for each feature with respect to the stress class.  
  - Store feature → description mappings, e.g.:  
    - `face_brow_tension_mean` → "Left/right brow tension".  
    - `voice_jitter_local` → "High vocal jitter (unstable pitch)".  
    - `eeg_beta_alpha_ratio_frontal` → "High beta/alpha ratio in frontal cortex".

- At runtime, map top‑k features per modality to human‑readable reasons and send them in the SSE payload.


## 4. Real‑Time Inference Architecture

### 4.1 Session Management

- Each browser session gets a unique `session_id`.  
- Backend maintains per‑session state:
  - Face feature buffer: last N windows of face features.  
  - Voice feature buffer: last N windows of voice features.  
  - Physio feature buffer (if available).  
  - Score buffers: exponential decay of expert probabilities.

### 4.2 Online Feature Extraction

- **Face**  
  - For each incoming frame: compute AU features.  
  - Aggregate into a window once L seconds elapsed (e.g., 1 s).  
  - Normalize using training set statistics (mean/std) stored from offline training. [web:29]

- **Voice**  
  - For each audio chunk of length L: extract acoustic biomarkers, normalize, and push into voice buffer. [web:38]

### 4.3 Online Prediction & Smoothing

For every new window (aligned across modalities):

1. Compute expert probabilities:  
   \(P_f(c), P_v(c), P_p(c)\).
2. Update score buffers with exponential decay:  
   \(S_{mod,new}(c) = \alpha P_{mod}(c) + (1-\alpha) S_{mod,prev}(c)\).  
3. Apply fusion:  
   - Weighted combination of smoothed scores.  
4. Threshold or argmax to produce the final stress level.  
5. Retrieve SHAP‑derived explanations and send via SSE to the frontend.

### 4.4 Frontend Visualization

- Real‑time stress gauge or bar.  
- Per‑modality radar chart of biomarker contributions. [web:29]  
- Textual explanations ("High vocal jitter" / "Sustained brow tension").


## 5. Implementation Checklist (So You Never Redo Extraction)

### 5.1 Changes to `colab_training.py`

1. **Add metadata columns**  
   - Parse `subject_id`, `task_id`, `video_id` from file paths or labels.  
   - For every window written, include:  
     `subject_id, task_id, video_id, window_index, window_start, window_end`.

2. **Replace video‑level averaging with window‑level extraction**  
   - Remove global averaging logic (`np.mean(frame_indicators, axis=0)`).  
   - Implement window grid with length L, hop H.  
   - For each window, compute aggregated features and write **one row per window**.

3. **Synchronize face & voice**  
   - Use the same windowing parameters and indices.  
   - Ensure any dropped/ignored windows are dropped in all modalities or marked clearly.

4. **Output separate modality files + multimodal join**  
   - `face_windows_stressid.csv`  
   - `voice_windows_stressid.csv`  
   - `physio_windows_stressid.csv` (from your existing physio pipeline)  
   - `multimodal_windows_stressid.csv` built by joining on key columns.

5. **Store normalization stats**  
   - Compute mean/std for each feature on the training set only.  
   - Save to `scaler_face.pkl`, `scaler_voice.pkl`, `scaler_physio.pkl` for consistent runtime normalization.

### 5.2 Validation Scripts (`strict_fused_evaluation.py`, `evaluate_fused_engine_bootstrapped.py`)

- Use **GroupKFold / LOSO** based on `subject_id`.  
- Simulate real‑time score buffers on validation folds.  
- Report:  
  - Per‑modality performance.  
  - Fused engine performance.  
  - Subject‑wise accuracy and confusion matrices.


## 6. Supporting Evidence from Literature & Patents (Design Justification)

- Real‑time multimodal stress detection with face+voice+physio using late fusion and streaming architecture has been demonstrated in prior work, validating this decoupled expert + fusion design. [cite:29]  
- Audio‑visual stress classification pipelines consistently use windowed feature extraction (MFCC, pitch, jitter, facial geometry) with late fusion, reporting significant gains over unimodal models. [cite:38][cite:34]  
- Multimodal sensor pipelines emphasize the importance of **systematic feature extraction, subject‑level splits, and feature selection** to avoid overfitting, which this architecture explicitly incorporates. [cite:28][cite:33]

