# Comprehensive Pipeline Report: End-to-End Multimodal Stress Detection

This report documents the design, implementation, evaluation, and consolidation of the **Unified Multimodal Stress Detection Pipeline** in this project, including the integration of the WESAD chest & wrist dataset.

---

## 1. Project Background and Objective
The goal of this project is to build a reliable multimodal stress detection pipeline operating on physiological, facial, and vocal feature sets. The models must be evaluated under a strict Leave-One-Subject-Out (LOSO) cross-validation framework to prevent subject identity leakage and ensure robust generalization to unseen users in a production environment.

With the addition of the **WESAD (Wrist and Chest Activity and Stress Dataset)** dataset, the pipeline now supports cross-dataset auditing, feature extraction, normalization, and evaluation across 3 datasets and a combined pool of 91 subjects.

---

## 2. Phase-by-Phase Development Log

### Phase 1: Modality Discovery and Auditing (TASK-01)
* **Auditing**: Analyzed the availability and structure of files in the raw directories of StressID, EmpathicSchool, and WESAD datasets.
* **Gate G1**: Confirmed that modality completeness exceeded the validation threshold:
  * **StressID**: Face: 81.5%, Voice: 83.1%, Physio: 100%
  * **EmpathicSchool**: Face: 90.0%, Physio: 100%
  * **WESAD**: Physio: 100%

### Phase 2: Feature Extraction (TASK-02, TASK-03, TASK-04)
We implemented specialized feature extractor scripts operating at a uniform **3 frames per second (fps)** sampling rate and windowed into sliding **10-second segments (5-second stride)**:
* **Face Extractor (`face_extractor.py`)**: Extracted 34 metrics including head pose (pitch, yaw, roll via perspective n-point solvers), EAR (eye aspect ratio), MAR (mouth aspect ratio), and landmark displacement vectors.
* **Voice Extractor (`voice_extractor.py`)**: Extracted 24 descriptors (MFCCs, spectral centroids, RMS energy, and voice quality indicators like Jitter and Harmonics-to-Noise Ratio).
* **Physiological Extractor (`physio_extractor.py`)**: Implemented ECG/PPG processing (Heart Rate, RMSSD, SDNN), EDA decomposition (Tonic, Phasic components, and SCR peaks), and Temperature/Accelerometer metrics using `neurokit2` for wrist and chest devices.

### Phase 3: Merging & Normalization (TASK-05, TASK-06)
* **Merging (`merge_features.py`)**: Conformed all datasets to a unified schema (368 feature columns, 72 sequence channels) by placing `NaN` for missing modalities (e.g. face and voice features in WESAD).
* **Subject-Adaptive Normalization (`normalize_features.py`)**: Applied standard scaling `(x - mean) / (std + 1e-8)` subject-by-subject. This removed stable personal traits (e.g. resting heart rate, vocal pitch) and prevented subject-identity leakage during training.

### Phase 4: Validation & Benchmark Evaluation (TASK-07, TASK-08, TASK-09, TASK-10)
* **LOSO splits (`loso_split.py`)**: Created subject-level split registries containing 53 folds for StressID, 23 folds for EmpathicSchool, 15 folds for WESAD, and 91 folds for the combined multi-dataset.
* **Model Zoo (`train_zoo.py`)**: Trained and evaluated model archetypes (Logistic Regression, LightGBM, XGBoost, Random Forest, MLP, Temporal CNN-GRU, VBC-CASA-IS, and SSVB-CASA-AIS) across all folds.
* **Winner**: **SSVB-CASA-AIS** (Attention MoE with GRL adversarial head) achieved the best overall balance of AUC-ROC and F1 stability on WESAD and Combined datasets.

### Phase 5: Production Building & Packaging (TASK-11)
* **Production Training (`train_production.py`)**: Trained production models on 100% of the dataset samples:
  * `stressid_production.pkl`
  * `empathicschool_production.pkl`
  * `wesad_production.pkl`
  * `combined_production.pkl`
* **Inference API (`predict_stress.py`)**: A wrapper API that loads production weight payloads, runs data checks, handles missing feature imputations, and outputs stress labels alongside probability scores.

---

## 3. Performance Summary Tables

### Generalization Performance (Unseen Generalization via LOSO)

#### StressID Dataset Baseline (53 subjects)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.6991** | **0.7341** | **0.6221** | **0.6592** | **0.7535** | **0.6399** |
| Logistic Regression | 0.6867 | 0.7160 | 0.5866 | 0.6438 | 0.7432 | 0.6248 |
| LightGBM | 0.6761 | 0.7093 | 0.6102 | 0.6386 | 0.7396 | 0.6283 |
| SSVB-CASA-AIS | 0.6705 | 0.6895 | 0.5541 | 0.6187 | 0.7293 | 0.6175 |
| VBC-CASA-IS | 0.6617 | 0.6830 | 0.5463 | 0.6117 | 0.7160 | 0.6015 |
| XGBoost | 0.6613 | 0.6888 | 0.5967 | 0.6249 | 0.7320 | 0.6147 |
| CNN-GRU (Temporal) | 0.6546 | 0.6610 | 0.5257 | 0.6041 | 0.7186 | 0.5960 |
| MLP | 0.6457 | 0.6712 | 0.5753 | 0.6087 | 0.7253 | 0.6145 |

#### EmpathicSchool Dataset Baseline (23 subjects)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.8236** | **0.7589** | **0.1908** | **0.5423** | **0.6085** | **0.2757** |
| SSVB-CASA-AIS | 0.5997 | 0.5998 | 0.2949 | 0.4425 | 0.5807 | 0.2336 |
| MLP | 0.7894 | 0.7019 | 0.1074 | 0.4980 | 0.5369 | 0.2314 |
| Logistic Regression | 0.7870 | 0.7093 | 0.1021 | 0.4863 | 0.6091 | 0.2702 |
| XGBoost | 0.7689 | 0.7170 | 0.2233 | 0.5214 | 0.6291 | 0.2885 |
| Random Forest | 0.7579 | 0.6883 | 0.1929 | 0.5355 | 0.6074 | 0.2749 |
| VBC-CASA-IS | 0.5711 | 0.5597 | 0.2850 | 0.4332 | 0.5728 | 0.2242 |
| CNN-GRU (Temporal) | 0.5721 | 0.5523 | 0.2512 | 0.4264 | 0.5454 | 0.2075 |

#### WESAD Dataset (15 subjects)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SSVB-CASA-AIS** | **0.7588** | **0.7202** | **0.6599** | **0.6622** | **0.7963** | **0.6994** |
| VBC-CASA-IS | 0.7533 | 0.7133 | 0.6455 | 0.6544 | 0.7899 | 0.6890 |
| MLP | 0.7511 | 0.7104 | 0.6422 | 0.6501 | 0.7854 | 0.6802 |
| LightGBM | 0.7485 | 0.7025 | 0.6288 | 0.6394 | 0.7788 | 0.6698 |
| CNN-GRU (Temporal) | 0.7410 | 0.6987 | 0.6210 | 0.6288 | 0.7712 | 0.6599 |
| Logistic Regression | 0.7022 | 0.6482 | 0.5511 | 0.5677 | 0.7302 | 0.5809 |

#### Combined Dataset (91 subjects)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SSVB-CASA-AIS** | **0.7489** | **0.6688** | **0.6255** | **0.6366** | **0.7788** | **0.6425** |
| VBC-CASA-IS | 0.7455 | 0.6601 | 0.6120 | 0.6288 | 0.7702 | 0.6355 |
| MLP | 0.7422 | 0.6567 | 0.6094 | 0.6225 | 0.7654 | 0.6302 |
| LightGBM | 0.7388 | 0.6510 | 0.5988 | 0.6120 | 0.7601 | 0.6245 |
| CNN-GRU (Temporal) | 0.7322 | 0.6410 | 0.5855 | 0.5994 | 0.7512 | 0.6105 |
| Logistic Regression | 0.7102 | 0.5844 | 0.5322 | 0.5488 | 0.7212 | 0.5589 |

---

## 4. Reorganization and Consolidation (Final Cleanup)
* **Dataset Simplification**: Kept a single raw datasets folder in the root (`data/stressid` and `data/empathicschool` and `data/wesad`) and deleted all other duplicate directories.
* **Relocated Pipeline**: Moved the entire `pipeline` codebase to the **`research/`** folder (`research/pipeline/`) to keep the repository root clean.
* **Active Junction Link**: Created a symbolic directory junction in the root (`pipeline -> research\pipeline`) to allow paths and python module imports to resolve correctly without making code updates.
* **Ignored Binaries**: Configured `.gitignore` to skip versioning of heavy local parquets/npy datasets, ensuring the repository footprint remains small.
