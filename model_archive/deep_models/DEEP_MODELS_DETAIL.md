# Deep Model Details — Phase 8 Production Models

> All models trained with PyTorch using strict Leave-One-Subject-Out (LOSO) 5-Fold GroupKFold validation on the full 65-subject StressID dataset. Training used **Time Masking** data augmentation.

---

## 1. deep_face_expert.pt

### Identity
- **Artifact ID**: `face_expert_v2`
- **Version**: 2.0.0
- **Framework**: PyTorch 1D-CNN + GRU
- **File**: `deep_face_expert.pt`
- **Scaler**: `deep_face_scaler.pkl`
- **SHA-256**: `7dc1652cb7c0f3b763eaa3de36428a09587be427dca97f3da52fd49a0d1495fe`

### Architecture
```
Input: [batch, seq_len=5, input_dim=18]  ← 18 facial features per frame
Conv1D(18 → 32, kernel=3, padding=1) + ReLU
Conv1D(32 → 64, kernel=3, padding=1) + ReLU
GRU(input=64, hidden=32, layers=1, batch_first=True) → last hidden state
Linear(32 → 16) + ReLU
Linear(16 → 2) → softmax → [P(calm), P(stress)]
```

### Input Features (18 dimensions)
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Brow Raise Index, Head Pose Pitch, Head Pose Yaw, Head Pose Roll, Gaze X, Gaze Y, Blink Rate, Pupil Dilation, AU1 (Inner Brow Raise), AU2 (Outer Brow Raise), AU4 (Brow Lowerer), AU6 (Cheek Raiser), AU12 (Lip Corner Puller), AU17 (Chin Raiser), AU20 (Lip Stretcher), AU25 (Lips Part)

### Training Config
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 20
- Batch size: 64
- Sequence length: 5
- Augmentation: Time Masking (mask up to 2 timesteps)
- Subjects: 65 (full dataset)

### Performance
| Metric | Value |
|---|---|
| LOSO Accuracy | **55.10%** |
| LOSO Std Dev | ± 4.58% |
| Evaluation Protocol | 5-Fold GroupKFold LOSO |
| Training Subjects | 65 |

### Version History
| Version | Accuracy | Notes |
|---|---|---|
| v1.0 (Classical) | 56.99% | Phase 4, sklearn RF, F1=56.05% |
| v2.0a (Deep, 15-subj) | 66.30% | Phase 7, 15-subject subset |
| **v2.0b (Deep, 65-subj)** | **55.10%** | **Phase 8, full dataset — CURRENT** |

---

## 2. deep_voice_expert.pt

### Identity
- **Artifact ID**: `voice_expert_v2`
- **Version**: 2.0.0
- **Framework**: PyTorch 1D-CNN + GRU
- **File**: `deep_voice_expert.pt`
- **Scaler**: `deep_voice_scaler.pkl`
- **SHA-256**: `db41acbc550915d476ba56b06c76c21d27cb4822b06aad49f5579ff5c1ec677e`

### Architecture
```
Input: [batch, seq_len=5, input_dim=12]  ← 12 voice features per frame
Conv1D(12 → 32, kernel=3, padding=1) + ReLU
Conv1D(32 → 64, kernel=3, padding=1) + ReLU
GRU(input=64, hidden=32, layers=1, batch_first=True) → last hidden state
Linear(32 → 16) + ReLU
Linear(16 → 2) → softmax → [P(calm), P(stress)]
```

### Input Features (12 dimensions)
F0 Mean (Pitch Hz), F0 Std Dev, F0 Range, MFCC 1–5 (5 coefficients), Speech Rate, Voiced Fraction, Spectral Centroid, Jitter, Shimmer

### Performance
| Metric | Value |
|---|---|
| LOSO Accuracy | **61.46%** ← Best single modality |
| LOSO Std Dev | ± 3.14% |
| Evaluation Protocol | 5-Fold GroupKFold LOSO |
| Training Subjects | 65 |

### Version History
| Version | Accuracy | F1 | Notes |
|---|---|---|---|
| v1.0 (Classical) | 59.52% | 70.46% | Phase 4, sklearn, LOSO |
| **v2.0 (Deep, 65-subj)** | **61.46%** | ~62% | **Phase 8 — CURRENT** |

> **Note**: Voice achieved the highest single-modality LOSO accuracy in the project. This is the most reliable unimodal stress indicator in this dataset.

---

## 3. deep_physio_expert.pt

### Identity
- **Artifact ID**: `physio_expert_v2`
- **Version**: 2.0.0
- **Framework**: PyTorch 1D-CNN + GRU
- **File**: `deep_physio_expert.pt`
- **Scaler**: `deep_physio_scaler.pkl`
- **SHA-256**: `b7e7568ef077352aa67415d4cacc45e26945864dc414d059edb6fce8aa566623`

### Architecture
```
Input: [batch, seq_len=5, input_dim=5]  ← 5 physiological features per frame
Conv1D(5 → 32, kernel=3, padding=1) + ReLU
Conv1D(32 → 64, kernel=3, padding=1) + ReLU
GRU(input=64, hidden=32, layers=1, batch_first=True) → last hidden state
Linear(32 → 16) + ReLU
Linear(16 → 2) → softmax → [P(calm), P(stress)]
```

### Input Features (5 dimensions)
EDA SCL Mean (µS), EDA SCR Rate (peaks/min), HRV RMSSD (ms), BVP Amplitude, EEG Beta/Alpha Ratio

### Performance
| Metric | Value |
|---|---|
| LOSO Accuracy | **58.95%** |
| LOSO Std Dev | ± 4.48% |
| Evaluation Protocol | 5-Fold GroupKFold LOSO |
| Training Subjects | 65 |

### Version History
| Version | Accuracy | F1 | Notes |
|---|---|---|---|
| v1.0 (Classical) | **70.51%** | 60.88% | Phase 4 — Highest ever in project |
| v2.0a (Deep, 15-subj) | 64.94% | ~65% | Phase 7, 15 subjects |
| **v2.0b (Deep, 65-subj)** | **58.95%** | ~59% | **Phase 8 — CURRENT** |

> **Note**: The classical Physio v1 (70.51%) is the highest accuracy achieved in the entire project across all modalities and phases. The deep model shows lower LOSO because strict subject-separation on a larger, more diverse pool is a harder task.

---

## 4. deep_fusion_router.pt

### Identity
- **Artifact ID**: `deep_fusion_router_v2`
- **Version**: 2.0.0
- **Framework**: PyTorch MLP Flex-Router
- **File**: `deep_fusion_router.pt`
- **Config**: `deep_fusion_config.json`
- **SHA-256**: `a562be250cd15d607d3c79f93f5f96bce3314c89b3839c047e186526852d48c2`

### Architecture
```
Input (9D): [P_face_calm, P_face_stress, P_voice_calm, P_voice_stress,
             P_physio_calm, P_physio_stress, mask_face, mask_voice, mask_physio]
Linear(9 → 32) + ReLU
Linear(32 → 16) + ReLU
Linear(16 → 3)  ← raw weights [w_face, w_voice, w_physio]

Post-processing:
  active_weights = raw_weights × [mask_face, mask_voice, mask_physio]
  norm_weights = active_weights / sum(active_weights)
  final_stress_prob = sum(norm_weight_m × P_stress_m) for active modalities
```

### Training Strategy: Modality Dropout
During training, 1 or 2 modalities were randomly masked out per batch (probability 0.5 each), forcing the router to learn optimal fusion weights for **any** combination of active sensors. This means at inference time it gracefully handles:
- Face only
- Voice only
- Physio only
- Face + Voice
- Face + Physio
- Voice + Physio
- Face + Voice + Physio (all)

### Runtime Fusion Config (`deep_fusion_config.json`)
```json
{
    "sequence_length": 5,
    "use_dynamic_router": true,
    "active_modalities": ["face", "voice", "physio"]
}
```

### Performance — All Sensor Combinations (LOSO 5-Fold, 65 subjects)
| Active Sensors | Mean Accuracy | Std Dev |
|---|---|---|
| Face Only | 55.10% | ± 4.58% |
| Voice Only | 61.46% | ± 3.14% |
| Physio Only | 58.95% | ± 4.48% |
| Face + Physio | 57.89% | ± 3.70% |
| Face + Voice | 55.57% | ± 3.86% |
| Voice + Physio | 58.27% | ± 2.53% |
| **All Three** | **58.26%** | **± 3.03%** |

### Version History
| Version | Architecture | Accuracy | Notes |
|---|---|---|---|
| v1.0 | MLP 2-way Router | 67.44% | Phase 7, Face+Physio, 15 subjects |
| **v2.0 (CURRENT)** | **MLP Flex-Router** | **58.26%** | **Phase 8, 3-way, 65 subjects** |
