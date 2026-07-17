# AGENT EXECUTION SPECIFICATION
# Unified Multimodal Stress Detection Pipeline (StressID + EmpathicSchool)

> **Target runtime:** Antigravity autonomous agent
> **Execution model:** Sequential tasks with hard gates. Do not skip. Do not reorder.
> **Golden rule:** Every task has explicit INPUTS, ACTIONS, OUTPUTS, and ACCEPTANCE CRITERIA. A task is complete only when its acceptance criteria pass. If they fail, HALT and write a failure report — never proceed on defective state.

---

## HOW TO READ THIS DOCUMENT (AGENT DIRECTIVE)

You are an autonomous engineering agent. This document is your work order. Follow these rules for the entire run:

1. **Execute tasks in numeric order.** `TASK-01` before `TASK-02`, and so on. Each task lists the tasks it depends on. Never start a task whose dependencies have not passed acceptance.
2. **Every artifact you produce must be persisted to disk** at the exact path specified, so later tasks and future runs can reuse it. Never hold important state only in memory.
3. **After each task, self-verify against ACCEPTANCE CRITERIA.** Write the result to `pipeline/logs/task_status.jsonl` as one JSON line: `{"task":"TASK-XX","status":"PASS|FAIL","timestamp":...,"metrics":{...}}`.
4. **Commit to git after each passing task** using the commit message given in the task.
5. **On any FAIL or GATE violation:** stop immediately, write `pipeline/logs/FAILURE_REPORT_TASK-XX.md` describing what failed and why, and do not continue.
6. **Determinism is mandatory.** Import and call `set_determinism()` (defined in TASK-00) as the first line of every script you write.
7. **Never hardcode feature dimensions or paths.** Read them from `pipeline/config/config.yaml` and `pipeline/config/feature_contract.yaml`. This is what makes the pipeline extensible.
8. **Log verbosely.** Every script writes a human-readable log to `pipeline/logs/{task_id}.log` describing what it did, how many items it processed, and any items it skipped and why.

---

## PROJECT CONTEXT (READ BEFORE STARTING)

You are extending an existing stress-detection project, not starting fresh. The project already contains professional model architectures (SSVB-CASA-AIS, VBC-CASA-IS, Subject-Adversarial CNN-GRU Router), a version registry, and strict Leave-One-Subject-Out (LOSO) evaluation discipline. Your job is to:

- Re-extract richer features from two datasets (StressID + EmpathicSchool).
- Store everything in a clean, reusable, versioned structure.
- Re-train and fairly cross-validate ALL models (classical, deep, and the existing professional models) on identical data and folds.
- Select the best model by F1-macro, subject to generalization gates.
- Package the winner for production.

**Two hard constraints that must never be violated:**
- **LOSO only.** No subject may appear in both train and test of any fold. Ever.
- **Generalization-first.** The model must learn stress biomarkers common across people (vocal jitter, HRV, EDA, facial tension), not personal identity. Calibration is used ONLY to normalize input against a person's own baseline — never to define what stress is.

---

## DATASET FACTS (GROUND TRUTH — DO NOT ASSUME BEYOND THIS)

### StressID (Primary)
- 65 subjects, 11 tasks each, 39+ hours.
- Modalities: RGB facial video, audio (microphone), ECG, EDA, respiration.
- Labels: binary stress + 3-class + self-reports. Label granularity: task-level.
- Physiological sampling: high-rate (ECG/EDA/respiration).

### EmpathicSchool (Supplementary)
- 30 subjects, ~90 min each, ~40 hours.
- Modalities: RGB video 1080p @ 30fps; EDA, BVP, ACC, HR, TEMP (Empatica E4).
- Labels: self-assessment stress + NASA-TLX workload at 2-minute intervals.
- **No audio.** These subjects contribute face + physio only.

### Combined
- 95 unique subjects. This is the LOSO evaluation pool.

---

## CANONICAL DIRECTORY LAYOUT (CREATE EXACTLY THIS)

```
pipeline/
├── config/
│   ├── config.yaml                 # all tunable parameters (single source of truth)
│   └── feature_contract.yaml       # feature names + dimensions per modality
├── common/
│   ├── determinism.py              # global seed control
│   ├── io_utils.py                 # parquet/json read-write helpers
│   └── loso.py                     # the ONE LOSO split function used everywhere
├── audit/
│   └── audit_datasets.py
├── extraction/
│   ├── face_extractor.py
│   ├── voice_extractor.py
│   └── physio_extractor.py
├── merge/
│   └── build_combined_matrix.py
├── normalize/
│   └── subject_adaptive_norm.py
├── models/
│   ├── classical.py                # RF, LightGBM, XGBoost
│   ├── cnn_gru.py                  # standard + adversarial
│   ├── professional.py             # SSVB-CASA-AIS, VBC-CASA-IS  (reuse existing arch)
│   └── transformer_fusion.py       # new shared-token Transformer
├── train/
│   └── run_model_zoo.py            # trains every registered model under LOSO
├── evaluation/
│   └── build_leaderboard.py
├── audits/
│   └── generalization_gates.py
├── package/
│   └── package_production.py
├── data/                           # PERSISTENT extracted data (reusable)
│   ├── raw/                        # symlinks or copies of source datasets
│   ├── stressid/                   # extracted per-modality features
│   ├── empathicschool/
│   └── combined/                   # merged, normalized matrices
├── artifacts/                      # trained models, scalers, stats
├── reports/                        # per-model reports, leaderboard, audits
└── logs/                           # task_status.jsonl, per-task logs, failure reports
```

---

# TASK-00 — ENVIRONMENT AND DETERMINISM SETUP

**DEPENDS ON:** none

**INPUTS:** none

**ACTIONS:**
1. Create the full directory layout above.
2. Install dependencies:
   ```bash
   pip install lightgbm xgboost scikit-learn torch torchvision torchaudio \
       mediapipe librosa opencv-python-headless scipy neurokit2 biosppy \
       heartpy pandas numpy pyarrow tqdm shap matplotlib seaborn \
       imbalanced-learn pyyaml --break-system-packages
   ```
3. Write `pipeline/common/determinism.py`:
   ```python
   import os, random, numpy as np
   SEED = 42
   def set_determinism(seed: int = SEED):
       os.environ["PYTHONHASHSEED"] = str(seed)
       random.seed(seed); np.random.seed(seed)
       try:
           import torch
           torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
       except ImportError:
           pass
   ```
4. Write `pipeline/config/config.yaml` (template below — fill real paths):
   ```yaml
   seed: 42
   datasets:
     stressid:
       raw_path: "pipeline/data/raw/stressid"
       modalities: [face, voice, physio]
       physio_signals: [ecg, eda, respiration]
       video_fps: 30
       audio_sr: 16000
       label_granularity: task
     empathicschool:
       raw_path: "pipeline/data/raw/empathicschool"
       modalities: [face, physio]
       physio_signals: [bvp, eda, hr, temp, acc]
       video_fps: 30
       label_granularity: interval_2min
       nasa_tlx_stress_threshold: 50
   windowing:
     window_sec: 10
     stride_sec: 5
     face_sample_fps: 3
   loso:
     n_splits: 5
     group_col: subject_id
   training:
     batch_size: 32
     epochs: 80
     patience: 15
     focal_gamma: 2.0
     focal_alpha: 0.25
     adv_lambda: 0.02
   gates:
     max_missing_modality_frac: 0.20
     max_fold_std: 0.08
     max_domain_sep_acc: 0.75
     max_leakage_gap: 0.10
     min_face_stress_f1: 0.40
   ```

**OUTPUTS:** directory tree, `determinism.py`, `config.yaml`.

**ACCEPTANCE CRITERIA:**
- All directories exist.
- `python -c "from pipeline.common.determinism import set_determinism; set_determinism()"` runs without error.
- `config.yaml` loads as valid YAML.

**COMMIT:** `TASK-00: environment and determinism setup`

---

# TASK-01 — DATA AUDIT AND COMPLETENESS GATE

**DEPENDS ON:** TASK-00

**INPUTS:** raw datasets at paths in `config.yaml`.

**ACTIONS:**
Write and run `pipeline/audit/audit_datasets.py`. For each dataset, build an inventory and verify completeness. The agent must:
1. Walk every subject folder and record, per subject: which modality files exist, their durations, and whether a usable label exists.
2. For StressID: confirm 65 subjects, ~11 tasks each, and presence of video, audio, ECG, EDA, respiration per task.
3. For EmpathicSchool: confirm 30 subjects, presence of video + EDA/BVP/HR/TEMP/ACC, and NASA-TLX labels at 2-minute intervals. Binarize labels with `stress = 1 if NASA_TLX >= 50 else 0` (threshold from config).
4. Compute and record class balance (stressed vs non-stressed) per dataset.
5. Flag any segment shorter than 30 seconds as unusable.
6. Verify video and physiological timestamps can be aligned (overlapping time ranges).

**OUTPUTS:**
- `pipeline/logs/audit_report.json` — per-subject completeness, class balance, flagged segments.
- `pipeline/logs/audit_summary.md` — human-readable summary.

**ACCEPTANCE CRITERIA (GATE G1):**
- Every required modality is present for at least 80% of each dataset's subjects.
- Class balance is logged for both datasets.
- If any modality is missing for >20% of subjects → **HALT**, write `FAILURE_REPORT_TASK-01.md`.

**COMMIT:** `TASK-01: data audit complete`

---

# TASK-02 — FACE FEATURE EXTRACTION (34 FEATURES, BOTH DATASETS)

**DEPENDS ON:** TASK-01

**INPUTS:** video files for all subjects in both datasets; `config.yaml`; `audit_report.json`.

**CONTEXT — WHY THIS MATTERS:** The previous project's face model failed (stressed-class F1 = 0.089) because it used only 18 static geometric features with no motion signal. You will fix this by adding 16 temporal delta features. Extract **exactly 34 features per frame**.

**ACTIONS:**
1. Write `pipeline/extraction/face_extractor.py` using MediaPipe FaceLandmarker (Tasks API). Download `face_landmarker.task` if absent.
2. For each video: sample at 3 fps (every 10th frame at 30fps).
3. Per sampled frame, compute the **18 static features**:
   `left_ear, right_ear, avg_ear, blink_velocity, eye_openness_ratio, brow_descent_left, brow_descent_right, brow_asymmetry, lip_compression, jaw_tension, mouth_corner_pull, forehead_tension, nose_wrinkle, face_height_norm, head_tilt, pitch, yaw, roll`.
4. Compute the **16 temporal delta features** = frame-to-frame difference of: `left_ear, right_ear, blink_velocity, brow_descent_left, brow_descent_right, brow_asymmetry, lip_compression, jaw_tension, mouth_corner_pull, forehead_tension, head_tilt, pitch, yaw, roll, eye_openness_ratio, nose_wrinkle`.
5. Build 10-second windows (5-second stride). Per window:
   - **Flat representation:** mean, std, min, max, range of each of the 34 features → 170 aggregated values.
   - **Sequence representation:** the raw 30×34 frame matrix (for temporal models).
6. If a face is undetected in >50% of a window's frames: set `face_available = 0`, fill features with **NaN** (never zeros).
7. Attach `subject_id`, `dataset_source`, `task_name`, `window_id`, and the aligned `binary_stress` label to each window.

**OUTPUTS (PERSIST FOR REUSE):**
- `pipeline/data/stressid/face_windows.parquet`
- `pipeline/data/empathicschool/face_windows.parquet`
- `pipeline/data/stressid/face_sequences.npy` + index parquet (sequences aligned to window_id)
- `pipeline/data/empathicschool/face_sequences.npy` + index parquet
- `pipeline/logs/face_extraction.log` (frames processed, faces missed, windows produced per subject)

**ACCEPTANCE CRITERIA:**
- Every eligible subject has ≥1 face window.
- Column count of flat parquet = 170 feature columns + metadata columns.
- `face_available` flag present and populated.
- Log reports per-subject window counts.

**COMMIT:** `TASK-02: face extraction complete (34 features)`

---

# TASK-03 — VOICE FEATURE EXTRACTION (24 FEATURES, STRESSID ONLY)

**DEPENDS ON:** TASK-01

**INPUTS:** audio files for StressID subjects; `config.yaml`.

**CONTEXT:** EmpathicSchool has no audio — skip it here; those subjects get `voice_available = 0` later. Previous voice model scored worse than the mean (R² = −0.35) due to weak features and uncorrected class imbalance. Extract **24 features** including delta MFCCs.

**ACTIONS:**
1. Write `pipeline/extraction/voice_extractor.py` using librosa. Resample all audio to 16 kHz. Use 25 ms window, 10 ms hop.
2. Per 10-second window (5-second stride), extract 24 features:
   - **Prosodic (7):** f0_mean, f0_std, f0_range, f0_delta_mean, speaking_rate, pause_ratio, voiced_fraction.
   - **Voice quality (4):** jitter_percent, shimmer_db, hnr, spectral_flux.
   - **Spectral (5):** spectral_centroid_mean, spectral_centroid_std, high_freq_ratio, voice_intensity, intensity_std.
   - **MFCC + deltas (8):** mfcc_2_mean…mfcc_5_mean, delta_mfcc_2_mean…delta_mfcc_5_mean.
3. Retain frame-level sequences for temporal models.
4. Log per-subject stressed vs non-stressed window counts to `pipeline/logs/voice_class_balance.json`. **Do not oversample here** — imbalance is corrected during training via class weighting.

**OUTPUTS (PERSIST):**
- `pipeline/data/stressid/voice_windows.parquet`
- `pipeline/data/stressid/voice_sequences.npy` + index parquet
- `pipeline/logs/voice_class_balance.json`
- `pipeline/logs/voice_extraction.log`

**ACCEPTANCE CRITERIA:**
- 24 voice feature columns present.
- Class balance file written and shows the imbalance ratio.
- Every StressID subject with audio has ≥1 voice window.

**COMMIT:** `TASK-03: voice extraction complete (24 features)`

---

# TASK-04 — PHYSIOLOGICAL FEATURE EXTRACTION (14 FEATURES, BOTH DATASETS)

**DEPENDS ON:** TASK-01

**INPUTS:** physiological signal files for both datasets; `config.yaml`.

**CONTEXT:** StressID physio = ECG + EDA + respiration. EmpathicSchool physio = BVP (→ HRV) + EDA + HR + TEMP. Use `neurokit2` exclusively — no manual peak detection.

**ACTIONS:**
1. Write `pipeline/extraction/physio_extractor.py`. Apply the StressID-baseline filtering:
   - ECG: 0.5 Hz high-pass Butterworth order 5.
   - EDA: 5 Hz low-pass Butterworth order 4.
   - Respiration: 0.1–0.5 Hz bandpass.
   - BVP (EmpathicSchool): 0.5–4 Hz bandpass before HRV extraction.
2. Per 10-second window, extract 14 features:
   - **HR/HRV (5):** hr_mean, hr_std, hrv_rmssd, hrv_sdnn, hrv_lf_hf_ratio.
   - **EDA (3):** eda_scl_mean, eda_scr_count, eda_scr_amplitude_mean.
   - **Respiration (4, StressID only):** resp_rate_mean, resp_rate_std, resp_amplitude, resp_irregularity.
   - **Temperature (2, EmpathicSchool only):** temp_mean, temp_slope.
3. Align physiology to the SAME window boundaries as face/voice using timestamps.
4. For features unavailable in a dataset (e.g. respiration for EmpathicSchool), fill 0 and set the appropriate availability sub-flag; set `physio_available = 1` if the core HRV/EDA block is present.

**OUTPUTS (PERSIST):**
- `pipeline/data/stressid/physio_windows.parquet`
- `pipeline/data/empathicschool/physio_windows.parquet`
- `pipeline/logs/physio_extraction.log`

**ACCEPTANCE CRITERIA:**
- 14 physio feature columns present in both.
- Window boundaries match those used in face extraction (same window_id scheme).
- `physio_available` populated.

**COMMIT:** `TASK-04: physio extraction complete (14 features)`

---

# TASK-05 — MERGE INTO COMBINED FEATURE MATRIX

**DEPENDS ON:** TASK-02, TASK-03, TASK-04

**INPUTS:** all per-modality parquet files.

**ACTIONS:**
Write `pipeline/merge/build_combined_matrix.py`. Deterministic merge:
1. Prefix subject IDs: `SID_` (StressID), `ES_` (EmpathicSchool).
2. Join face + voice + physio on `window_id` per subject. For EmpathicSchool, fill all 24 voice columns with 0 and set `voice_available = 0`.
3. Unify labels to a single `binary_stress` column.
4. Ensure three availability flags: `face_available, voice_available, physio_available`.
5. Sort by `subject_id`, then `window_id`.
6. Write the exact realized feature count to the feature contract (do not assume 174 vs 208 — compute it).

**OUTPUTS (PERSIST):**
- `pipeline/data/combined/combined_features_10s.parquet`
- `pipeline/config/feature_contract.yaml` (updated with real feature names + counts per modality)
- `pipeline/logs/merge_summary.md` (row count, subject count = 95, feature count, per-source counts)

**ACCEPTANCE CRITERIA:**
- Subject count == 95.
- No `window_id` collisions across datasets.
- Feature contract lists face=34, voice=24, physio=14 base features and the realized aggregated flat count.

**COMMIT:** `TASK-05: combined 95-subject matrix built`

---

# TASK-06 — SUBJECT-ADAPTIVE NORMALIZATION (GENERALIZATION CONTROL)

**DEPENDS ON:** TASK-05

**INPUTS:** `combined_features_10s.parquet`.

**CONTEXT — THIS IS THE GENERALIZATION MECHANISM:** For every feature and every subject, replace the raw value with its z-score against that subject's own distribution: `z = (x − subject_mean) / (subject_std + 1e-8)`. After this, absolute identity signals (face shape, voice timbre, baseline EDA) collapse to zero; only *deviation from the person's own baseline* remains. The model can therefore only learn generic stress dynamics, not who the person is. This is calibration used strictly as normalization.

**ACTIONS:**
1. Write `pipeline/normalize/subject_adaptive_norm.py`.
2. Provide a function that, given a set of subjects, computes per-subject per-feature mean/std and applies z-scoring.
3. **Do NOT globally normalize the whole dataset here.** Normalization is fold-aware and happens inside the LOSO loop (TASK-08). This task produces the reusable function and validates it on a held-out sanity subject, plus persists global per-subject stats for inference reuse.
4. Persist per-subject statistics to `pipeline/artifacts/subject_stats.json`.

**OUTPUTS:**
- `pipeline/normalize/subject_adaptive_norm.py` (importable, deterministic).
- `pipeline/artifacts/subject_stats.json`.
- `pipeline/logs/normalization_check.md` (before/after variance showing identity features collapse).

**ACCEPTANCE CRITERIA:**
- After normalization on a sample subject, each feature has mean ≈ 0 and std ≈ 1 within that subject.
- Function accepts arbitrary subject subsets (required for fold-correct use).

**COMMIT:** `TASK-06: subject-adaptive normalization ready`

---

# TASK-07 — LOSO SPLIT MODULE (THE SINGLE SOURCE OF FOLDS)

**DEPENDS ON:** TASK-05

**INPUTS:** `combined_features_10s.parquet`; `config.yaml`.

**ACTIONS:**
Write `pipeline/common/loso.py` with ONE function used by every model:
```python
from sklearn.model_selection import GroupKFold
NON_FEATURE_COLS = ["subject_id","window_id","binary_stress","dataset_source","task_name"]
def loso_splits(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df["binary_stress"].values
    groups = df["subject_id"].values
    return list(gkf.split(X, y, groups=groups))
```
Every model in the zoo MUST import and use this. No model defines its own splits.

**OUTPUTS:** `pipeline/common/loso.py`; `pipeline/logs/loso_folds.json` (which subjects fall in which fold — for reproducibility).

**ACCEPTANCE CRITERIA:**
- No subject appears in both train and test of any fold (assert and log).
- Fold assignment is identical across repeated runs (determinism).

**COMMIT:** `TASK-07: LOSO split module`

---

# TASK-08 — MODEL ZOO TRAINING (FAIR, IDENTICAL-FOOTING COMPETITION)

**DEPENDS ON:** TASK-06, TASK-07

**INPUTS:** combined matrix; normalization function; LOSO folds; feature contract.

**CONTEXT:** Every model competes on identical features, identical folds, identical metrics. The existing professional models are re-instantiated for the new feature dimensions (their architecture is reused; old weights are retired) and compete equally. Winner is chosen later by F1-macro.

**MODELS TO REGISTER AND TRAIN:**

| Group | Model | File | Input |
|---|---|---|---|
| Classical | Random Forest (reference) | classical.py | flat |
| Classical | LightGBM (`class_weight=balanced`) | classical.py | flat |
| Classical | XGBoost | classical.py | flat |
| Deep | CNN-GRU expert per modality (standard) | cnn_gru.py | sequences |
| Deep | Subject-Adversarial CNN-GRU Router (λ=0.02) | cnn_gru.py | sequences |
| Professional | SSVB-CASA-AIS (reuse arch, new input dims) | professional.py | sequences |
| Professional | VBC-CASA-IS (reuse arch, new input dims) | professional.py | sequences |
| New | Shared Transformer Fusion | transformer_fusion.py | tokens |

**COMMON TRAINING CONTRACT (all deep models):**
- Loss: Focal Loss (γ=2.0, α=0.25). Classical models: `class_weight='balanced'`.
- Optimizer AdamW, weight_decay 1e-4; lr 1e-3 (CNN-GRU) / 5e-4 (attention & Transformer).
- CosineAnnealingLR; early stopping on val F1-macro, patience 15.
- Read all input dims from `feature_contract.yaml` — never hardcode.

**ACTIONS (per model, inside the LOSO loop):**
1. For each fold: compute normalization stats from TRAIN subjects only; for the held-out subject, compute stats from that subject's own windows (simulates deployment calibration). Apply z-scoring to train and test separately.
2. Train on the training fold; evaluate on the held-out fold.
3. Record per-fold metrics AND per-subject accuracy.
4. Run the full modality ablation: face-only, voice-only, physio-only, face+physio, face+voice, all-three, EmpathicSchool-only, combined-95. (Voice combinations = StressID subjects only.)

**OUTPUTS (PERSIST per model):**
```
pipeline/reports/{model_name}/
├── fold_results.csv
├── per_subject_results.csv
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png       # SHAP for tree models
├── metrics_summary.json         # mean ± std for every metric
└── summary.md
pipeline/artifacts/{model_name}/  # trained fold checkpoints + scalers
pipeline/reports/ablation_results.csv  # appended by every model
```

**ACCEPTANCE CRITERIA (per model):**
- All 5 folds completed; no subject leakage (assert against `loso_folds.json`).
- `metrics_summary.json` contains mean and std for: accuracy, f1_macro, f1_stress, recall_stress, precision_stress, roc_auc, balanced_accuracy, per_subject_acc_std.

**COMMIT (per model):** `TASK-08: {model} LOSO complete — acc [X] f1_macro [Y]`

---

# TASK-09 — MASTER LEADERBOARD

**DEPENDS ON:** TASK-08 (all models)

**INPUTS:** every `metrics_summary.json`.

**ACTIONS:**
Write `pipeline/evaluation/build_leaderboard.py`. Aggregate all models into one table sorted by **F1-macro using (mean − std)** as the ranking key, so stability is rewarded. Include accuracy, F1-macro, F1-stress, recall-stress, ROC-AUC, per-subject std, and runtime.

**OUTPUTS:**
- `pipeline/reports/MASTER_LEADERBOARD.csv`
- `pipeline/reports/MASTER_LEADERBOARD.md` (formatted table + one-paragraph verdict).

**ACCEPTANCE CRITERIA:**
- Every trained model appears exactly once.
- Ranking key documented and applied consistently.

**COMMIT:** `TASK-09: master leaderboard`

---

# TASK-10 — GENERALIZATION GATES (ELIGIBILITY FOR PRODUCTION)

**DEPENDS ON:** TASK-09

**INPUTS:** leaderboard; per-model results; combined matrix.

**ACTIONS:**
Write `pipeline/audits/generalization_gates.py`. Evaluate the top models against every gate. A model is production-eligible only if it passes ALL gates.

| Gate | Test | Threshold |
|---|---|---|
| **G2 Stability** | fold-to-fold accuracy std | ≤ 0.08 |
| **G3 Biomarker validity** | top features (SHAP/attention) include recognized stress biomarkers (jitter, delta-MFCC, f0_std, hrv_rmssd, lf_hf_ratio, eda_scr_count, brow/jaw/forehead tension, blink dynamics) | ≥1 in top ranks |
| **G4 Identity suppression** | random-split acc − LOSO acc | ≤ 0.10 |
| **G5 Cross-dataset transfer** | train StressID → test EmpathicSchool (face+physio) reported; domain classifier (SID vs ES) separation | domain acc ≤ 0.75 |

If the domain classifier separates datasets at >0.75 → switch to per-dataset training + inference-time ensembling (document the switch; do not naively pool).

Also run per-subject error analysis: are misclassified subjects concentrated by dataset, task type, or signal quality?

**OUTPUTS:**
- `pipeline/reports/generalization_gates.json` (pass/fail per gate per top model).
- `pipeline/reports/hard_subjects.json`.
- `pipeline/reports/generalization_audit.md`.

**ACCEPTANCE CRITERIA (GATE):**
- At least one model passes all of G2–G5. If none pass → HALT, write failure report recommending remediation.
- G6 sanity: combined 95-subject LOSO of the leader ≥ 74% (prior baseline). If below → regression, HALT.
- Face stressed-class F1 of the leader ≥ 0.40 (confirms D1 fixed). If below → HALT.

**COMMIT:** `TASK-10: generalization gates complete`

---

# TASK-11 — PRODUCTION PACKAGING

**DEPENDS ON:** TASK-10

**INPUTS:** winning model; feature contract; existing VersionRegistry.

**ACTIONS:**
Write `pipeline/package/package_production.py`.
1. Select the highest F1-macro model that passed all gates.
2. Retrain it on ALL 95 subjects (no held-out fold) for deployment.
3. Register via the existing VersionRegistry with: artifact hash, LOSO metrics, feature-contract version, and the passing-gate record.
4. Update `webapp/configs/feature_contract.yaml` to the new 34/24/14 dimensions.
5. Update `webapp/backend/calibration.py`: the 30-second calibration window must compute per-subject mean/std for all features and apply z-scoring before every inference (the deployment mirror of TASK-06).
6. Write a migration note describing what changed for anyone loading old artifacts.

**OUTPUTS:**
- Registered production model in `webapp/models/` + updated `registry.json`.
- Updated `feature_contract.yaml` and `calibration.py`.
- `pipeline/reports/PRODUCTION_RELEASE.md`.

**ACCEPTANCE CRITERIA:**
- Registry entry has hash + metrics + gate record.
- Feature contract matches extraction (34/24/14).
- Calibration applies subject-adaptive z-scoring.

**COMMIT:** `TASK-11: production packaging — winner deployed`

---

## GLOBAL GATES SUMMARY (AGENT MUST ENFORCE)

| Gate | Where | Condition to PASS | On FAIL |
|---|---|---|---|
| G1 Completeness | TASK-01 | ≤20% subjects missing any modality | HALT |
| G2 Stability | TASK-10 | fold acc std ≤ 0.08 | exclude model |
| G3 Biomarkers | TASK-10 | stress biomarker in top features | reject model |
| G4 Leakage | TASK-10 | random−LOSO gap ≤ 0.10 | reject model |
| G5 Domain shift | TASK-10 | domain sep ≤ 0.75 else ensemble | switch strategy |
| G6 No regression | TASK-10 | leader LOSO ≥ 74% | HALT |
| D1 Face fixed | TASK-10 | face stress F1 ≥ 0.40 | HALT |

---

## EXTENSIBILITY CONTRACT (FOR FUTURE UPDATES)

To add a new dataset: register it under `datasets:` in `config.yaml`, provide an extractor that writes to `pipeline/data/{name}/`, and re-run from TASK-05. Nothing upstream changes.

To add a new model: add it to `pipeline/models/` and register it in `run_model_zoo.py`. It automatically inherits the same LOSO folds, normalization, metrics, and gates. The registry keeps all prior models, so the new one is measured against the established baseline rather than silently replacing it.

To add a new modality: extend `feature_contract.yaml` and add an extractor; the merge, normalization, and token/flat builders read dimensions from the contract, so no downstream code is rewritten.

---

## EXPECTED PERFORMANCE (HONEST TARGETS)

| Model | LOSO Accuracy | F1-macro | Confidence |
|---|---|---|---|
| Random Forest (reference) | 75–77% | 0.68–0.71 | High |
| **LightGBM** | **78–82%** | **0.72–0.76** | High |
| Subject-Adversarial CNN-GRU (fixed face) | 73–76% | 0.68–0.72 | Medium |
| SSVB-CASA-AIS (re-validated) | 72–76% | 0.66–0.72 | Medium |
| VBC-CASA-IS (re-validated) | 70–74% | 0.64–0.70 | Medium |
| Shared Transformer Fusion | 75–79% | 0.70–0.74 | Medium |
| Winner retrained on 95 subjects | +1–3% over best | — | High |

Realistic ceiling under strict LOSO on this data is 80–85%. Above 85% is unlikely without more subjects; the StressID authors' own best baseline is ~0.75 weighted F1. These are defensible numbers, not inflated ones.

---

## REFERENCES

1. Chaptoukaev, H. et al. *StressID: a Multimodal Dataset for Stress Identification.* NeurIPS 2023 Datasets & Benchmarks. https://project.inria.fr/stressid/
2. Hosseini, M. et al. *A multimodal stress detection dataset with facial expressions and physiological signals.* Scientific Data (Nature), 2025.
3. Hosseini, M. et al. *Multimodal Stress Detection Using Facial Landmarks and Biometric Feedback.* arXiv:2311.03606, 2023.
4. Ke, G. et al. *LightGBM.* NeurIPS 2017.
5. Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
6. Ganin, Y. et al. *Domain-Adversarial Training of Neural Networks.* JMLR 2016.
7. Makowski, D. et al. *NeuroKit2.* Behavior Research Methods, 2021.
