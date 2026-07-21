# 5-Stage Raw Feature Extraction Pipeline

## Overview

A systematic pipeline to extract rich, high-quality features from all 3 raw stress datasets (StressID, WESAD, EmpathicSchool) with **window-level confidence scoring**, **proper naming conventions**, and **multi-scale temporal windows**.

Each stage builds on the previous, culminating in a unified, confidence-scored feature bank ready for downstream modelling.

---

## Stage 1 — Single Dataset Full Extraction

**Purpose**: Extract every available modality from each dataset independently. Outputs raw windowed sequences per dataset.

**Naming Convention**: `{dataset}_{subject}_{task}_{modality}_{window_start}-{window_end}.npz`

### 1.1 StressID Extraction

| Modality | Raw Source | Output Features | Window Strategy |
|----------|-----------|-----------------|-----------------|
| **Face** | `data/stressid/Videos/{subj}/{subj}_{task}.mp4` | 34-D (eye AR, mouth, head pose, deltas) | Sliding window, 3 fps, 10s window, 5s stride |
| **Voice** | `data/stressid/Audio/{subj}/{subj}_{task}.wav` | 24-D (RMS, ZCR, f0, MFCCs, spectral) | Same window alignment |
| **Physio** | `data/stressid/Physiological/{subj}/{subj}_{task}.txt` | 14-D (HR, HRV, EDA clean/tonic/phasic, SCR, resp rate/amp) | Same window alignment |

**Output**: `data/extracted/stressid/sequences/{subj}/{task}_*.npz`

**Code Structure**:
```python
def extract_stressid_full(subject_id: str, task: str, window_sec: int = 10, stride_sec: int = 5) -> Dict:
    """
    Extracts face, voice, and physio features for one StressID task.
    Returns aligned windows with metadata.
    """
    video_path = f"data/stressid/Videos/{subject_id}/{subject_id}_{task}.mp4"
    audio_path = f"data/stressid/Audio/{subject_id}/{subject_id}_{task}.wav"
    physio_path = f"data/stressid/Physiological/{subject_id}/{subject_id}_{task}.txt"
    
    # Extract per-modality sequences
    face_seq = extract_face_sequence(video_path, target_fps=3)        # [T, 34]
    voice_seq = extract_voice_sequence(audio_path, target_fps=3)      # [T, 24]
    physio_seq = extract_physio_sequence(physio_path, target_fps=3)   # [T, 14]
    
    # Window into overlapping segments
    windows = {}
    for name, seq in [("face", face_seq), ("voice", voice_seq), ("physio", physio_seq)]:
        if seq is not None:
            windows[name] = sliding_window(seq, window_sec * 3, stride_sec * 3)
    
    return windows
```

### 1.2 WESAD Extraction

| Modality | Raw Source | Output Features | Notes |
|----------|-----------|-----------------|-------|
| **Physio** | `data/wesad/{S*}/{S*}.pkl` | 14-D (HR, HRV, EDA, SCR, resp, temp, ACC) | Chest + wrist sensors |
| **Face** | None | All-zeros (34-D) | No camera in WESAD |
| **Voice** | None | All-zeros (24-D) | No microphone in WESAD |

**Output**: `data/extracted/wesad/sequences/{subj}/all_*.npz`

### 1.3 EmpathicSchool Extraction

| Modality | Raw Source | Output Features | Notes |
|----------|-----------|-----------------|-------|
| **Face** | `data/empathicschool/{S*}/**/*.mp4` | 34-D | Recursive search for MP4s |
| **Physio** | `data/empathicschool/{S*}/**/*EDA.csv` etc. | 14-D (from E4 wristband: EDA, HR, TEMP, ACC) | Resample E4 4Hz → 3fps |
| **Voice** | None | All-zeros (24-D) | Only audio embedded in video (low quality) |

**Output**: `data/extracted/empathicschool/sequences/{subj}/{task}_*.npz`

### 1.4 Unified Stage 1 Entry Point

```python
def stage_1_extract_datasets(datasets: List[str] = ["stressid", "wesad", "empathicschool"],
                              window_sec: int = 10, stride_sec: int = 5):
    """
    Stage 1: Extract all features from all specified datasets.
    Saves per-dataset, per-subject, per-task windowed sequences.
    """
    for ds in datasets:
        if ds == "stressid":
            extract_stressid_all(window_sec, stride_sec)
        elif ds == "wesad":
            extract_wesad_all(window_sec, stride_sec)
        elif ds == "empathicschool":
            extract_empathicschool_all(window_sec, stride_sec)
```

---

## Stage 2 — Single Modality Extraction

**Purpose**: Extract each modality independently across all datasets. Enables per-modality analysis and ablation studies.

**Naming Convention**: `{modality}__{dataset}__{subject}__{window_id}.npz`

### 2.1 Face Modality (all datasets with face data)

```python
def stage_2_extract_modality(modality: str = "face", 
                              datasets: List[str] = ["stressid", "empathicschool"],
                              window_sec: int = 10, stride_sec: int = 5):
    """
    Extract a single modality across all datasets that have it.
    WESAD is excluded for face (no camera).
    """
    results = []
    for ds in datasets:
        ds_results = extract_modality_from_dataset(ds, modality, window_sec, stride_sec)
        results.extend(ds_results)
    return results
```

**Face features** (34-D):
```
[0]  left_ear              [1]  right_ear
[2]  avg_ear               [3]  blink_velocity
[4]  eye_openness_ratio    [5]  brow_descent_left
[6]  brow_descent_right    [7]  brow_asymmetry
[8]  lip_compression       [9]  jaw_tension
[10] mouth_corner_pull     [11] EXCLUDED (face_height_norm)
[12] forehead_tension      [13] head_tilt
[14] pitch                 [15] yaw
[16] roll                  [17] nose_wrinkle
[18-33] Temporal deltas of [0-17]
```

### 2.2 Voice Modality (StressID only)

**Voice features** (24-D):
```
[0]  rms                   [1]  zcr
[2]  EXCLUDED (f0_mean)    [3]  f0_std
[4]  f0_min                [5]  f0_max
[6]  f0_range              [7]  voice_prob
[8]  loudness              [9]  loudness_std
[10] hnr                   [11] jitter
[12] shimmer               [13-25] mfcc_0..mfcc_12
[26] spectral_centroid     [27] spectral_bandwidth
[28] spectral_rolloff      [29] spectral_flatness
[30] chroma_stft
```

### 2.3 Physio Modality (all 3 datasets)

**Physio features** (14-D):
```
[0]  hr                    [1]  hrv_rmssd
[2]  eda_clean             [3]  EXCLUDED (eda_tonic_scl)
[4]  eda_phasic            [5]  scr_count
[6]  resp_rate             [7]  resp_amplitude
[8]  temp_mean             [9]  temp_std
[10] acc_x                 [11] acc_y
[12] acc_z                 [13] acc_mag
```

---

## Stage 3 — Combined Dataset Extraction

**Purpose**: Merge all 3 datasets into a unified feature matrix with proper subject prefixing and modality alignment.

**Naming Convention**: `combined__{dataset}__{subject}__{window_id}.npz`

### 3.1 Subject Identity Disambiguation

```python
SUBJECT_PREFIX = {
    "stressid":        "SID",
    "wesad":           "WSD",
    "empathicschool":  "EMP",
}

def prefix_subject(dataset: str, subject: str) -> str:
    return f"{SUBJECT_PREFIX[dataset]}_{subject}"
```

This prevents the WESAD/EmpathicSchool subject ID collision (both use `s2`..`s17`).

### 3.2 Modality Alignment Matrix

Each window across all datasets always has **69 feature channels** (after privacy exclusions), with missing modalities zero-filled:

```python
# For each window in the combined dataset:
combined_window = np.zeros((window_len, 69), dtype=np.float32)

# Column mapping (69-D):
# [0:9]    face_eye            (9)
# [9:15]   face_mouth          (6)
# [15:33]  face_global_face    (18)
# [33:41]  voice_spectral_prosody  (8)
# [41:54]  voice_mfcc          (13)
# [54:56]  voice_quality       (2)
# [56:58]  physio_cardio       (2)
# [58:61]  physio_eda          (3)
# [61:69]  physio_somatic      (8)
```

### 3.3 Combined Stage 3 Entry Point

```python
def stage_3_combine_datasets(window_sec: int = 10, stride_sec: int = 5):
    """
    Stage 3: Load per-dataset extractions, merge into unified matrix.
    Output: single NPZ + metadata Parquet for combined training.
    """
    all_features = defaultdict(list)
    all_metadata = []
    window_id = 0
    
    for ds in ["stressid", "wesad", "empathicschool"]:
        ds_data = load_dataset_windows(ds, window_sec, stride_sec)
        for window in ds_data:
            # Align features to 69-D unified space
            aligned = align_to_unified(window["features"], ds)
            all_features["sequences"].append(aligned)
            
            all_metadata.append({
                "window_id": window_id,
                "subject_id": prefix_subject(ds, window["subject"]),
                "dataset": ds,
                "label": window["label"],
                "modalities_present": window["modalities_present"],
            })
            window_id += 1
    
    save_combined(all_features, all_metadata)
```

---

## Stage 4 — Multi-Window Scale Extraction

**Purpose**: Extract features at multiple temporal resolutions to capture different stress patterns.

| Window Size | Stride | Temporal Resolution | Best For |
|-------------|--------|-------------------|----------|
| **2 seconds** (6 frames) | 1s | High | Micro-expressions, heart rate spikes, voice tremor |
| **5 seconds** (15 frames) | 2s | Medium | Short-term EDA fluctuations, breathing patterns |
| **10 seconds** (30 frames) | 5s | Standard | Balance of temporal context and localization |
| **30 seconds** (90 frames) | 15s | Low | Long-term HRV trends, sustained stress episodes |

### 4.1 Multi-Scale Extraction

```python
WINDOW_CONFIGS = {
    2:  {"stride": 1,  "fps": 3,  "label": "micro"},
    5:  {"stride": 2,  "fps": 3,  "label": "short"},
    10: {"stride": 5,  "fps": 3,  "label": "standard"},
    30: {"stride": 15, "fps": 3,  "label": "long"},
}

def stage_4_multi_scale_extraction(dataset: str = "combined"):
    """
    Stage 4: Re-window the combined sequences at multiple scales.
    Each scale is saved as a separate sub-directory.
    """
    combined_seqs = load_combined_sequences(dataset)  # [N, T_max=90, 69]
    
    for window_sec, config in WINDOW_CONFIGS.items():
        target_frames = window_sec * config["fps"]
        stride_frames = config["stride"] * config["fps"]
        
        # Subsample or pad to target length
        resampled = resample_sequences(combined_seqs, target_frames)
        
        # Window
        windows = sliding_window_3d(resampled, target_frames, stride_frames)
        
        save_scale_output(dataset, window_sec, windows, config)
```

### 4.2 Output Structure

```
data/extracted/
  combined/
    scale_02s/    (2-second windows)
    scale_05s/    (5-second windows)
    scale_10s/    (10-second windows, default)
    scale_30s/    (30-second windows)
```

---

## Stage 5 — Confidence-Scored Window Extraction

**Purpose**: Every window receives a per-modality **confidence score** based on signal quality. Enables confidence-aware training and inference.

**Naming Convention**: `{window_id}__conf_{overall_confidence:.3f}.npz`

### 5.1 Confidence Sources

| Modality | Confidence Metric | Range | Computation |
|----------|------------------|-------|-------------|
| **Face** | Detection rate | [0, 1] | Fraction of frames where face was detected |
| **Face** | Landmark quality | [0, 1] | Mean detection confidence from MediaPipe |
| **Voice** | Voiced ratio | [0, 1] | Fraction of frames with f0 present (voiced speech) |
| **Voice** | Signal-to-noise | [0, 1] | Normalized RMS / background RMS |
| **Physio** | Signal quality | [0, 1] | NeuroKit2 signal quality index |
| **Physio** | Missing data rate | [0, 1] | 1 - (NaN count / total) |
| **Overall** | Modality availability | [0, 1] | Fraction of expected modalities that are non-zero |

### 5.2 Confidence Computation

```python
def compute_window_confidence(
    face_seq: np.ndarray,    # [T, 34]
    voice_seq: np.ndarray,   # [T, 24]
    physio_seq: np.ndarray,  # [T, 14]
    face_meta: Dict = None   # Per-frame detection metadata
) -> Dict[str, float]:
    """
    Compute per-modality and overall confidence for a window.
    Returns:
        {
            "face_confidence": 0.85,   # Face detected in 85% of frames
            "voice_confidence": 0.72,  # Voiced speech in 72% of frames
            "physio_confidence": 0.95, # 95% signal quality
            "overall_confidence": 0.84 # Weighted average
        }
    """
    conf = {}
    
    # Face confidence: detection rate
    if face_seq is not None and len(face_seq) > 0:
        # Non-zero frames indicate face detected
        face_energy = np.abs(face_seq).sum(axis=1)
        detection_rate = (face_energy > 1e-6).mean()
        conf["face"] = float(detection_rate)
    else:
        conf["face"] = 0.0
    
    # Voice confidence: voiced ratio
    if voice_seq is not None and len(voice_seq) > 0:
        f0_col = 3  # Column index for f0_std (proxy for pitch presence)
        voiced_ratio = (np.abs(voice_seq[:, f0_col]) > 0.01).mean()
        conf["voice"] = float(voiced_ratio)
    else:
        conf["voice"] = 0.0
    
    # Physio confidence: signal quality + missing data
    if physio_seq is not None and len(physio_seq) > 0:
        # Zero-energy check (sensor disconnected)
        physio_energy = np.abs(physio_seq).sum(axis=1)
        active_ratio = (physio_energy > 1e-6).mean()
        conf["physio"] = float(active_ratio)
    else:
        conf["physio"] = 0.0
    
    # Modality availability
    available = sum(1 for v in conf.values() if v > 0.5)
    total = len(conf)
    conf["modality_availability"] = available / total if total > 0 else 0.0
    
    # Overall confidence (weighted: physio > face > voice)
    weights = {"face": 0.3, "voice": 0.2, "physio": 0.5}
    conf["overall"] = sum(conf.get(k, 0) * v for k, v in weights.items()) / sum(weights.values())
    
    return conf
```

### 5.3 Quality-Gated Extraction

```python
def stage_5_confidence_scored_extraction(
    confidence_threshold: float = 0.3,
    window_sec: int = 10,
    stride_sec: int = 5
):
    """
    Stage 5: Extract windows with associated confidence scores.
    Low-confidence windows (< threshold) are flagged but preserved
    (not filtered out) to allow downstream confidence-aware training.
    """
    for ds in ["stressid", "wesad", "empathicschool", "combined"]:
        windows = load_all_windows(ds, window_sec, stride_sec)
        
        for window in windows:
            conf = compute_window_confidence(
                window.get("face"), window.get("voice"), window.get("physio")
            )
            
            window["confidence"] = conf
            
            # Flag low-quality windows
            window["is_low_quality"] = conf["overall"] < confidence_threshold
            
            save_confidence_window(ds, window)
```

### 5.4 Confidence-Aware Output Format

Each window is saved with an additional `confidence.json` sidecar:

```json
{
    "window_id": "SID_2ea4_Baseline_0000",
    "dataset": "stressid",
    "subject": "SID_2ea4",
    "task": "Baseline",
    "label": 0,
    "confidence": {
        "face": 0.97,
        "voice": 0.88,
        "physio": 0.99,
        "modality_availability": 1.0,
        "overall": 0.97
    },
    "is_low_quality": false,
    "feature_path": "data/extracted/stressid/windows/SID_2ea4_Baseline_0000.npz"
}
```

---

## Naming Conventions Summary

| Entity | Convention | Example |
|--------|-----------|---------|
| **Dataset prefix** | SID_, WSD_, EMP_ | `SID_2ea4`, `WSD_s10`, `EMP_s18` |
| **Subject ID** | `{prefix}{original_id}` | `SID_2ea4`, `WSD_s10` |
| **Window ID** | `{dataset}__{subject}__{task}__{window_index}` | `stressid__SID_2ea4__Counting1__0003` |
| **Feature file** | `{window_id}.npz` | `SID_2ea4_Baseline_0000.npz` |
| **Confidence file** | `{window_id}__conf.json` | `SID_2ea4_Baseline_0000__conf.json` |
| **Log file** | `extraction_{dataset}_{stage}.log` | `extraction_stressid_stage1.log` |

---

## Pipeline Orchestration

```python
def run_full_pipeline(window_sec: int = 10, stride_sec: int = 5):
    """
    Execute all 5 stages sequentially.
    """
    print("=" * 60)
    print("  FULL 5-STAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Stage 1: Per-dataset full extraction
    print("\n[Stage 1/5] Single Dataset Full Extraction...")
    stage_1_extract_datasets(window_sec=window_sec, stride_sec=stride_sec)
    
    # Stage 2: Per-modality extraction
    print("\n[Stage 2/5] Single Modality Extraction...")
    for mod in ["face", "voice", "physio"]:
        stage_2_extract_modality(mod, window_sec=window_sec, stride_sec=stride_sec)
    
    # Stage 3: Combined dataset
    print("\n[Stage 3/5] Combined Dataset Extraction...")
    stage_3_combine_datasets(window_sec=window_sec, stride_sec=stride_sec)
    
    # Stage 4: Multi-window scale
    print("\n[Stage 4/5] Multi-Window Scale Extraction...")
    stage_4_multi_scale_extraction()
    
    # Stage 5: Confidence scoring
    print("\n[Stage 5/5] Confidence-Scored Window Extraction...")
    stage_5_confidence_scored_extraction(
        confidence_threshold=0.3,
        window_sec=window_sec,
        stride_sec=stride_sec
    )
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
```

---

## Output Directory Structure

```
data/extracted/
├── stressid/
│   ├── sequences/           # Stage 1: Raw windowed sequences
│   │   └── {subject}/
│   │       └── {task}__{window_id}.npz
│   ├── modalities/          # Stage 2: Per-modality
│   │   ├── face/
│   │   ├── voice/
│   │   └── physio/
│   ├── scale_*s/            # Stage 4: Multi-window
│   └── confidence/          # Stage 5: Confidence scored
│       └── {window_id}__conf.json
├── wesad/                   # (same structure)
├── empathicschool/          # (same structure)
└── combined/                # Stage 3: Unified
    ├── sequences/
    ├── modalities/
    ├── scale_*s/
    └── confidence/
```

---

## Implementation Script

The full implementation is at:
`webapp/training/phase8/feature_extraction_service.py`

It implements all 5 stages with:
- `FeatureExtractor` class — robust per-modality extraction
- `WindowProcessor` class — multi-scale windowing
- `FeatureExtractionService` class — per-dataset orchestration
- `ExtractionOrchestrator` — full pipeline runner
