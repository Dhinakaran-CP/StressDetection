# Classical Model Details — Phase 4 Baseline Models

> All classical models trained with scikit-learn using strict Leave-One-Subject-Out (LOSO) GroupKFold cross-validation. These are the v1.0 baseline models that established the project's initial performance benchmarks, before transitioning to deep sequence learning in Phase 7/8.

---

## 1. face_expert_lightweight.pkl

### Identity
- **Artifact ID**: `face_expert_v1`
- **Version**: 1.0.0
- **Framework**: scikit-learn (Classical ML — Random Forest / SVM)
- **File**: `face_expert_lightweight.pkl`
- **Scaler**: `face_scaler_lightweight.pkl`
- **SHA-256**: `a2027fcf87fc74580e7448cd1a948d4d23af719a6ddda5d9341a907e8bc51f27`
- **File Size**: ~7.45 MB

### Training Dataset
- **Source**: StressID — Face Modality CSV
- **Total Rows**: 108,051
- **Subjects**: 53
- **Class 0 (Calm)**: 61,746 samples
- **Class 1 (Stressed)**: 46,305 samples

### Input Features
18-dimensional facial feature vector including: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Brow Raise Index, Head Pose (Pitch/Yaw/Roll), Gaze directions, Blink rate, Action Units (AU1, AU2, AU4, AU6, AU12, AU17, AU20, AU25)

### Performance
| Metric | Value |
|---|---|
| **Accuracy (LOSO)** | **56.99%** |
| **F1-Score (Stress class)** | **56.05%** |
| Evaluation Protocol | 3-Fold GroupKFold LOSO |

### Confusion Matrix (LOSO aggregate)
|  | Predicted Calm | Predicted Stressed |
|---|---|---|
| **Actually Calm** | 6,611 ✅ | 4,487 ❌ |
| **Actually Stressed** | 5,134 ❌ | 6,136 ✅ |

### Phase Experiments (from Phase 2)
| Configuration | Accuracy | F1 |
|---|---|---|
| Raw Baseline | 66.24% | 57.58% |
| Subject-Aware Normalization | 69.04% | 59.57% |
| Temporal Windowing | 69.37% | 60.43% |

> **Note**: The Phase 2 experiments showed higher accuracy because they used 3-Fold GroupKFold on a smaller subject subset. The final v1.0 model metrics above use the full subject pool under stricter LOSO.

---

## 2. voice_expert_lightweight.pkl

### Identity
- **Artifact ID**: `voice_expert_v1`
- **Version**: 1.0.0
- **Framework**: scikit-learn (Classical ML)
- **File**: `voice_expert_lightweight.pkl`
- **Scaler**: `voice_scaler_lightweight.pkl`
- **SHA-256**: `5a1df684fc3fd167b996c83aa1168a75207c7ead6282ed66b88651b0345efc0a`
- **File Size**: ~3.91 MB

### Training Dataset
- **Source**: StressID — Voice Modality CSV
- **Total Rows**: 44,982
- **Subjects**: 54
- **Class 0 (Calm)**: 13,090 samples
- **Class 1 (Stressed)**: 31,892 samples
- **Note**: Dataset is class-imbalanced (71% stressed)

### Input Features
12-dimensional voice feature vector: F0 Mean (Pitch Hz), F0 Std Dev, F0 Range, MFCC coefficients 1–5, Speech Rate (syllables/sec), Voiced Fraction, Spectral Centroid, Jitter (%), Shimmer (%)

### Performance
| Metric | Value |
|---|---|
| **Accuracy (LOSO)** | **59.52%** |
| **F1-Score (Stress class)** | **70.46%** |
| Evaluation Protocol | 3-Fold GroupKFold LOSO |

### Confusion Matrix (LOSO aggregate)
|  | Predicted Calm | Predicted Stressed |
|---|---|---|
| **Actually Calm** | 936 ✅ | 1,920 ❌ |
| **Actually Stressed** | 1,452 ❌ | 4,022 ✅ |

> **Note**: High F1 (70.46%) despite moderate accuracy (59.52%) is explained by the class imbalance — the model is strong at detecting true stress cases but weaker at calm classification.

### Phase Experiments (from Phase 2)
| Configuration | Accuracy | F1 |
|---|---|---|
| Raw Baseline | 70.98% | 82.81% |
| Subject-Aware Normalization | 70.70% | 82.75% |
| Temporal Windowing | 70.56% | 82.64% |

> **Warning from Phase 2**: The Phase 2 voice metrics (70.98% accuracy, 82.81% F1) were flagged as suspiciously high — likely due to class imbalance inflating the F1 score in KFold, not because the model truly generalizes that well to unseen speakers.

---

## 3. physio_expert_lightweight.pkl

### Identity
- **Artifact ID**: `physio_expert_v1`
- **Version**: 1.0.0
- **Framework**: scikit-learn (Classical ML)
- **File**: `physio_expert_lightweight.pkl`
- **Scaler**: `physio_scaler_lightweight.pkl`
- **SHA-256**: `79c7e0905649f28582bdf8b260f17f07b2019527930cc39388f0380e4e153d35`
- **File Size**: ~8.10 MB

### Training Dataset
- **Source**: StressID — Physio Modality (EDA, HRV, BVP, EEG)
- **Subjects**: 65
- **Total Samples**: ~24,876

### Input Features
5-dimensional physiological feature vector: EDA SCL Mean (µS), EDA SCR Rate (peaks/min), HRV RMSSD (ms), BVP Amplitude, EEG Beta/Alpha Ratio

### Performance
| Metric | Value |
|---|---|
| **Accuracy (LOSO)** | **70.51%** ← 🏆 Highest in entire project |
| **F1-Score (Stress class)** | **60.88%** |
| Evaluation Protocol | 3-Fold GroupKFold LOSO |

### Confusion Matrix (LOSO aggregate)
|  | Predicted Calm | Predicted Stressed |
|---|---|---|
| **Actually Calm** | 11,832 ✅ | 2,667 ❌ |
| **Actually Stressed** | 4,669 ❌ | 5,708 ✅ |

> **Note**: This model holds the **record for the highest subject-independent accuracy achieved in the entire project at 70.51%**. Physiological signals (especially EDA and HRV) contain strong, generalizable stress markers that even a classical model can capture without overfitting to subject identity.

### Phase Experiments (from Phase 2)
| Configuration | Accuracy | F1 |
|---|---|---|
| Raw Baseline | 59.58% | 42.01% |
| Subject-Aware Normalization | 67.22% | 57.05% |
| Temporal Windowing | 67.39% | 57.64% |

---

## Why Classical Models Were Archived

After Phase 4, the project transitioned to deep PyTorch sequence models in Phase 7–8 for the following reasons:

1. **Temporal Context**: Classical models process each frame independently. Deep CNN-GRU models capture temporal stress patterns across sequences of frames (seq_len=5), which is more representative of how stress unfolds over time.

2. **Identity Leakage**: The generalization audit (Phase 8.2) showed classical models have a leakage gap of 18.99–26.51% vs. 7.62% for deep sequence models.

3. **Flex-Modality Support**: Classical models cannot be easily fused with a learned dynamic router that supports any subset of active sensors.

4. **Production Architecture**: The deep models integrate directly with the live streaming runtime engine via sliding window buffers.

These classical models are archived as evidence of the baseline research trajectory and for potential fallback use.
