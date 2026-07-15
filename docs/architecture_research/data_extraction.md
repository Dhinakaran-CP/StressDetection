# Rich Multimodal Feature Extraction Blueprint for StressID-Compatible Stress Modeling

## 1. Goal

Build a rich, subject-aware, yet subject-independent feature extraction pipeline that supports:
- classical machine learning,
- early fusion,
- late fusion,
- gated fusion,
- mixture-of-experts,
- cross-attention,
- and adversarial subject-invariant models.

The extraction phase must preserve raw information, baseline-normalized information, temporal dynamics, and quality metadata so that any downstream model can use the same feature store reliably [web:455].

## 2. Why this pipeline

StressID already demonstrates that cleaned and modality-specific features are coherent and predictive. Its baseline extracts:
- 35 ECG features,
- 23 EDA features,
- 40 respiration features,
- 84 video features,
- 140 handcrafted audio features,
- and 513 Wav2Vec audio embeddings [web:455].

This blueprint extends that idea so the extracted dataset is rich enough for any downstream architecture and still safe for strict LOSO evaluation.

## 3. Core design principles

- Keep modality-specific richness.
- Preserve temporal information.
- Include both absolute and baseline-relative features.
- Attach quality flags to every window.
- Avoid global normalization before subject split.
- Support both handcrafted and learned embeddings.
- Keep feature groups separated but alignment-ready.

## 4. Inputs

### 4.1 Modalities
- Face video.
- Voice/audio.
- Physiology.

### 4.2 Required metadata
- Subject ID.
- Session ID.
- Task label.
- Timestamp.
- Window index.
- Modality quality score.
- Baseline confidence score.

## 5. Extraction architecture

### Stage A: Raw signal cleaning
For each modality:
1. Load the raw stream.
2. Remove corrupted or incomplete segments.
3. Detect outliers and extreme artifacts.
4. Flag silent, clipped, or unstable windows.
5. Store only valid windows for feature extraction.

This cleaning stage is required before any model-independent representation is produced [web:455].

### Stage B: Windowing
Split each recording into fixed windows:
- short window: 2 seconds,
- medium window: 5 seconds,
- optional long window: 10 seconds.

Use overlap if needed:
- 25% or 50%.

Each window gets a unique ID and a timestamp range.

### Stage C: Modality-specific feature extraction

#### Face features
Extract from each window:
- action units,
- gaze direction,
- eye openness,
- blink rate,
- brow motion,
- lip compression,
- jaw tension,
- head pose,
- facial velocity,
- facial acceleration,
- region-wise variance,
- short-term temporal slopes.

Recommended feature groups:
- eye region,
- brow region,
- mouth/jaw region,
- head motion region,
- global facial activity.

#### Voice features
Extract:
- MFCCs,
- delta MFCCs,
- delta-delta MFCCs,
- pitch,
- pitch range,
- pitch variance,
- jitter,
- shimmer,
- intensity,
- spectral centroid,
- spectral bandwidth,
- spectral contrast,
- flatness,
- rolloff,
- pause ratio,
- voiced/unvoiced ratio,
- speaking-rate proxy,
- energy dynamics,
- short-term trend features.

#### Physiology features
Extract:
- ECG:
  - HR,
  - RR interval statistics,
  - SDNN,
  - RMSSD,
  - pNN20,
  - pNN50,
  - frequency-domain HRV,
  - nonlinear HRV.
- EDA:
  - tonic level,
  - phasic level,
  - SCL slope,
  - SCL dynamic range,
  - SCR count,
  - SCR amplitude,
  - SCR duration.
- Respiration:
  - respiration rate,
  - respiration-rate variability,
  - rate trends,
  - cycle amplitude,
  - cycle stability.

## 6. Baseline calibration

### 6.1 Verified baseline collection
For each subject, use the calm-state baseline segment and verify that it is stable enough to serve as reference.

The baseline should be accepted only if:
- motion is low,
- audio is calm,
- physiology is stable,
- and the segment is not contaminated by stress.

### 6.2 Baseline statistics
For every feature in each modality compute:
- baseline mean,
- baseline standard deviation,
- baseline median,
- baseline interquartile range,
- baseline confidence.

### 6.3 Baseline-calibrated outputs
For each feature create:
- raw value,
- z-score,
- deviation from baseline,
- percent change from baseline,
- optional clipped normalized value.

This dual representation is important because it preserves:
- personal absolute context,
- and subject-independent change [web:490][web:455].

## 7. Feature richness layers

### 7.1 Primary features
These are the direct extracted signals from StressID-style pipelines:
- ECG, EDA, respiration,
- AUs and gaze,
- audio acoustic and embedding features [web:455].

### 7.2 Temporal dynamics
For each feature, add:
- mean,
- std,
- min,
- max,
- slope,
- derivative,
- rolling variance,
- peak count,
- change-point indicator.

### 7.3 Cross-window context
Add:
- previous-window delta,
- next-window delta when available,
- session-position index,
- task-phase index.

### 7.4 Quality metadata
Add:
- quality score,
- motion score,
- audio SNR,
- face confidence,
- physiological continuity flag.

## 8. Output format

Produce three feature stores:

### 8.1 Raw store
Contains:
- absolute features only.

### 8.2 Calibrated store
Contains:
- baseline-relative z-scores,
- normalized deviations,
- subject-calibrated features.

### 8.3 Fusion store
Contains:
- concatenated raw + calibrated features,
- quality metadata,
- timestamps,
- labels,
- subject/session identifiers.

This makes the dataset usable for any downstream approach.

## 9. Recommended feature schema

### 9.1 Face schema
- `face_au_*`
- `face_gaze_*`
- `face_blink_*`
- `face_pose_*`
- `face_motion_*`
- `face_region_eye_*`
- `face_region_brow_*`
- `face_region_mouth_*`

### 9.2 Voice schema
- `voice_mfcc_*`
- `voice_pitch_*`
- `voice_jitter_*`
- `voice_shimmer_*`
- `voice_energy_*`
- `voice_spectral_*`
- `voice_pause_*`
- `voice_embedding_*`

### 9.3 Physiology schema
- `ecg_hrv_*`
- `eda_tonic_*`
- `eda_phasic_*`
- `resp_rate_*`
- `resp_variability_*`

## 10. Processing pipeline for implementation

### Step 1
Load all subject recordings.

### Step 2
Verify integrity and create a subject/session registry.

### Step 3
Apply cleaning and artifact rejection.

### Step 4
Segment into fixed windows.

### Step 5
Extract modality-specific features.

### Step 6
Compute baseline statistics using only verified calm windows.

### Step 7
Create raw and calibrated versions.

### Step 8
Attach quality flags and metadata.

### Step 9
Export feature tables per modality.

### Step 10
Export a final fused table for multimodal training.

## 11. How this supports different model types

### Classical ML
Use the raw or calibrated store directly with:
- Random Forest,
- SVM,
- Logistic Regression,
- XGBoost.

### Deep learning
Use the raw sequence windows and/or the calibrated windows as input to:
- CNN,
- GRU,
- Transformer,
- cross-attention,
- MoE.

### Hybrid models
Use:
- handcrafted features for classical branches,
- learned embeddings for deep branches,
- both combined in fusion.

## 12. Why this is more robust

This blueprint is more robust than a minimal pipeline because it preserves:
- expressive modality-specific information,
- subject-independent normalization,
- temporal behavior,
- and quality control [web:455][web:463][web:490].

It also supports models that need region-level and time-level richness, especially for face and voice MoE-style expert routing.

## 13. GPU usage

Use GPU for:
- deep face feature extraction,
- audio embeddings such as Wav2Vec,
- heavy CNN/Transformer encoders.

Use CPU for:
- windowing,
- cleaning,
- baseline subtraction,
- z-scoring,
- and simple summary statistics.

GPU improves speed only for compute-heavy parts, not for plain statistical feature computation.

## 14. Final recommended extraction strategy

The best strategy is:

1. StressID-style cleaning and feature families.
2. Verified baseline calibration.
3. Dual representation: raw + baseline-relative.
4. Multi-scale temporal summaries.
5. Quality-aware metadata.
6. Modality-separated and fusion-ready export.

This creates a dataset representation that can support **any future model architecture** reliably [web:455][web:462][web:490].

## 15. Final statement

Use StressID as the baseline feature backbone, but enrich it with verified baseline calibration, dual raw/z-score representation, temporal dynamics, and quality-aware metadata. That will give you a rich, model-agnostic feature store that is suitable for classical ML, fusion models, MoE, and cross-attention training under strict LOSO evaluation [web:455][web:463][web:490].
