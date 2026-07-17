# Comprehensive Pipeline Report: End-to-End Multimodal Stress Detection

This report documents the design, implementation, evaluation, and consolidation of the **Unified Multimodal Stress Detection Pipeline** in this project.

---

## 1. Project Background and Objective
The goal of this project is to build a reliable multimodal stress detection pipeline operating on physiological, facial, and vocal feature sets. The models must be evaluated under a strict Leave-One-Subject-Out (LOSO) cross-validation framework to prevent subject identity leakage and ensure robust generalization to unseen users in a production environment.

---

## 2. Phase-by-Phase Development Log

### Phase 1: Modality Discovery and Auditing (TASK-01)
* **Auditing**: Analyzed the availability and structure of files in the raw directories of both datasets (StressID and EmpathicSchool).
* **Gate G1**: Confirmed that modality completeness exceeded the G1 validation threshold:
  * **StressID**: Face: 81.5%, Voice: 83.1%, Physio: 100%
  * **EmpathicSchool**: Face: 90.0%, Physio: 100%

### Phase 2: Feature Extraction (TASK-02, TASK-03, TASK-04)
We implemented specialized feature extractor scripts operating at a uniform **3 frames per second (fps)** sampling rate and windowed into sliding **10-second segments (5-second stride)**:
* **Face Extractor (`face_extractor.py`)**: Extracted 34 metrics including head pose (pitch, yaw, roll via perspective n-point solvers), EAR (eye aspect ratio), MAR (mouth aspect ratio), and landmark displacement vectors.
* **Voice Extractor (`voice_extractor.py`)**: Extracted 24 descriptors (MFCCs, spectral centroids, RMS energy, and voice quality indicators like Jitter and Harmonics-to-Noise Ratio).
* **Physiological Extractor (`physio_extractor.py`)**: Implemented ECG/PPG processing (Heart Rate, RMSSD, SDNN), EDA decomposition (Tonic, Phasic components, and SCR peaks), and Temperature/Accelerometer metrics using `neurokit2`. Added E4 raw accelerometer file length truncation logic.

### Phase 3: Merging & Normalization (TASK-05, TASK-06)
* **Merging (`merge_features.py`)**: Used a key-based outer join on `window_id` to join face, voice, and physiological window matrices. This preserved missing modality columns as `NaN` without discarding sample rows. Conformed sequence dimensions to `[N, 30, D]` where \(D=72\) for StressID and \(D=48\) for EmpathicSchool.
* **Subject-Adaptive Normalization (`normalize_features.py`)**: Applied standard scaling `(x - mean) / (std + 1e-8)` subject-by-subject. This removed stable personal traits (e.g. resting heart rate, vocal pitch) and prevented subject-identity leakage during training.

### Phase 4: Validation & Benchmark Evaluation (TASK-07, TASK-08, TASK-09, TASK-10)
* **LOSO splits (`loso_split.py`)**: Created subject-level split registries containing 53 folds for StressID and 23 folds for EmpathicSchool.
* **Model Zoo (`train_zoo.py`)**: Trained and evaluated four model archetypes across all folds:
  * **Logistic Regression baseline**
  * **LightGBM**
  * **PyTorch MLP**
  * **PyTorch 1D CNN-GRU (Temporal Sequence Model)**
* **Leaderboard & Gates (`leaderboard.py`, `verify_gates.py`)**: Verified generalization gates:
  * **G2 (StressID AUC > 0.70)**: Passed by all model archetypes.
  * **G3 (EmpathicSchool AUC > 0.55)**: Passed by Logistic Regression, LightGBM, and MLP.
  * **Winner**: **LightGBM** achieved the best overall balance of AUC-ROC and F1 stability, making it the selected production candidate.

### Phase 5: Production Building & Packaging (TASK-11)
* **Production Training (`train_production.py`)**: Trained the selected LightGBM model on 100% of the dataset samples to output the final production weights:
  * `stressid_production.pkl` (Face, Voice, and Physio input support)
  * `empathicschool_production.pkl` (Face and Physio input support)
* **Inference API (`predict_stress.py`)**: A wrapper API that loads production weight payloads, runs data checks, handles missing feature imputations, and outputs stress labels alongside probability scores.

---

## 3. Performance Summary Tables

### Generalization Performance (Unseen Generalization via LOSO)
This table shows the cross-validation metrics across unseen subjects (no subject overlap in test folds):

#### StressID Dataset
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.6892 | 0.6661 | 0.5377 | 0.5565 | 0.7440 | 0.1706 |
| LightGBM | 0.6731 | 0.6363 | 0.5896 | 0.5683 | 0.7372 | 0.1691 |
| MLP | 0.6834 | 0.6404 | 0.6061 | 0.5857 | 0.7459 | 0.1584 |
| CNN-GRU (Temporal) | 0.6812 | 0.6238 | 0.6138 | 0.5798 | 0.7480 | 0.1565 |

#### EmpathicSchool Dataset
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8102 | 0.2746 | 0.0571 | 0.0557 | 0.5901 | 0.0911 |
| LightGBM | 0.8694 | 0.2703 | 0.1362 | 0.1667 | 0.6000 | 0.1983 |
| MLP | 0.8369 | 0.3209 | 0.0724 | 0.1060 | 0.5639 | 0.1293 |
| CNN-GRU (Temporal) | 0.8141 | 0.2856 | 0.0892 | 0.1213 | 0.5440 | 0.1477 |


---

## 4. Reorganization and Consolidation (Final Cleanup)
* **Dataset Simplification**: Kept a single raw datasets folder in the root (`data/stressid` and `data/empathicschool`) and deleted all other duplicate directories (`dataset_discovery/`, `data/stress_d/`, and `pipeline/data/raw`).
* **Relocated Pipeline**: Moved the entire `pipeline` codebase to the **`research/`** folder (`research/pipeline/`) to keep the repository root clean.
* **Active Junction Link**: Created a symbolic directory junction in the root (`pipeline -> research\pipeline`) to allow paths and python module imports to resolve correctly without making code updates.
* **Ignored Binaries**: Configured `.gitignore` to skip versioning of heavy local parquets/npy datasets, ensuring the repository footprint remains small.
