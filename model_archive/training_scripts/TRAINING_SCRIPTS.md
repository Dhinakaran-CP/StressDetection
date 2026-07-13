# Training Scripts — Evidence & Documentation

> This folder contains all Python scripts used to train, evaluate, and package the StressDetectionUsingML models across Phases 4 through 8. These are preserved as evidence of the complete experimental pipeline.

---

## Script Index

| Script | Phase | Purpose | Models Produced |
|---|---|---|---|
| `phase4_experiments.py` | Phase 2–4 | Classical ML ablations (raw vs norm vs windowed) | Baseline metrics only |
| `train_phase4_release.py` | Phase 4 | Train & register classical baseline models | `face/voice/physio_expert_lightweight.pkl` |
| `train_face_expert_release.py` | Phase 4 | Train Face classical expert only | `face_expert_lightweight.pkl` |
| `train_voice_expert_release.py` | Phase 4 | Train Voice classical expert only | `voice_expert_lightweight.pkl` |
| `train_physio_expert_release.py` | Phase 4 | Train Physio classical expert only | `physio_expert_lightweight.pkl` |
| `release_expert_model.py` | Phase 4–5 | Register a trained model into `models/registry.json` | Registry entries |
| `phase6_multimodal_research.py` | Phase 6 | Multimodal fusion strategy comparisons | Fusion benchmark reports |
| `augmentation.py` | Phase 7 | Data augmentation utilities (Time Masking, Noise, Mixup) | Used by other scripts |
| `run_augmentation_experiments.py` | Phase 7 | Ablation: compare 5 augmentation strategies per modality | Reports only |
| `phase7_deep_learning_research.py` | Phase 7 | Deep CNN-GRU research on 15-subject subset | Early deep model benchmarks |
| `phase8_best_expert_fusion.py` | Phase 8 | Best expert selection + 2-way Face+Physio fusion | Intermediate deep models |
| `package_phase8_production.py` | Phase 8 | **FINAL** — Train all 3 CNN-GRU encoders + Flex Router on 65 subjects | **All production `.pt` models** |
| `generalization_research.py` | Phase 8.2 | Leakage audit: 5 strategies, random-split vs LOSO | Leakage audit report |

---

## Script Details

### phase4_experiments.py
**Phase**: 2–4  
**Purpose**: Ran the initial classical ML experiments comparing three normalization/windowing strategies for each modality.

**Experiments run**:
- Raw Feature Baseline (no normalization)
- Subject-Aware Normalization (subtract per-subject calm baseline)
- Temporal Windowing (rolling window of 5 frames)

**Key results produced**:
```
Face:   Raw=66.24% → SubjNorm=69.04% → Windowed=69.37%
Voice:  Raw=70.98% → SubjNorm=70.70% → Windowed=70.56%
Physio: Raw=59.58% → SubjNorm=67.22% → Windowed=67.39%
```

**Validation**: 3-Fold GroupKFold (subject-independent)

---

### train_phase4_release.py
**Phase**: 4  
**Purpose**: Trains and registers the production classical baseline experts for all three modalities.

**Models produced**:
- `models/face_expert_lightweight.pkl` → LOSO Accuracy: 56.99%, F1: 56.05%
- `models/voice_expert_lightweight.pkl` → LOSO Accuracy: 59.52%, F1: 70.46%
- `models/physio_expert_lightweight.pkl` → LOSO Accuracy: **70.51%** (project record), F1: 60.88%

**Registers into**: `models/registry.json` under `history_models`

---

### train_face_expert_release.py / train_voice_expert_release.py / train_physio_expert_release.py
**Phase**: 4  
**Purpose**: Per-modality training scripts. Each script trains its respective classical expert independently and saves the model + scaler.

---

### release_expert_model.py
**Phase**: 4–5  
**Purpose**: Utility to register any trained model artifact into the `models/registry.json` manifest with hash, timestamp, and metadata. Called by the train scripts after saving model files.

---

### phase6_multimodal_research.py
**Phase**: 6  
**Purpose**: Researched different multimodal fusion strategies — naive average, stacked meta-learner, calibrated fusion.

**Strategies compared**:
1. Naive Average (equal weights)
2. Learned Stack (meta RF on probability outputs)
3. Calibrated Average (temperature-scaled)

**Best result**: Calibrated Naive Average 3-Way ≈ 64.63% LOSO

---

### augmentation.py
**Phase**: 7  
**Purpose**: Data augmentation utility module containing implementations of:
- **Gaussian Noise** — adds small random noise to feature values
- **Time Masking** — randomly zeros out 1–2 timesteps in a sequence
- **Feature Masking** — randomly zeros out individual feature dimensions
- **Mixup** — interpolates between two samples with a random λ coefficient
- **Time Jittering** — randomly shifts sequence timestamps

---

### run_augmentation_experiments.py
**Phase**: 7  
**Purpose**: Ablation study comparing all augmentation strategies on the deep CNN-GRU model using 15-subject subset.

**Results (Face modality, 15 subjects)**:
```
No Augmentation:  63.21%
Gaussian Noise:   61.48%
Time Masking:     65.75%  ← WINNER (selected for Phase 8)
Feature Masking:  63.89%
Mixup:            60.94%
Time Jittering:   62.37%
```

**Decision**: Time Masking selected as the augmentation strategy for Phase 8 full-dataset training.

---

### phase7_deep_learning_research.py
**Phase**: 7  
**Purpose**: Researched and benchmarked the deep 1D-CNN + GRU sequence encoder architecture on a 15-subject subset before scaling to all 65 subjects.

**Architecture tested**: 1D-CNN + GRU with seq_len=5 and Time Masking

**Results (15-subject subset, LOSO)**:
```
Face Encoder:      65.75% (± 0.0652)
Voice Encoder:     58.89% (± 0.0401)
Physio Encoder:    65.44% (± 0.0729)
Deep Gated Fusion: 64.59% (± 0.0541)
```

**Decision**: Confirmed deep sequence learning architecture is viable. Proceeded to Phase 8 with full 65-subject dataset.

---

### phase8_best_expert_fusion.py
**Phase**: 8 (intermediate)  
**Purpose**: Best expert selection + 2-way dynamic fusion (Face + Physio only — Voice excluded at this stage due to technical issues).

**Results (65 subjects, LOSO)**:
```
Face-Only Encoder:   59.12% (± 0.0747)
Physio-Only Encoder: 54.85% (± 0.0852)
2-Way Dynamic Fusion: 58.75% (± 0.0847)
```

**Models produced** (intermediate, superseded by package_phase8_production.py):
- `deep_fusion_router.pt` v1 (2-way only) → 67.44% on 15-subj, 58.75% on 65-subj

---

### package_phase8_production.py  ⭐ FINAL PRODUCTION SCRIPT
**Phase**: 8  
**Purpose**: The master production training script. Trains all 3 CNN-GRU modality encoders plus the Flex-Modality Dynamic Router MLP on the full 65-subject dataset with Modality Dropout.

**Training Steps**:
1. Load and preprocess all three modality datasets (Face: 108,051 rows, Voice: 44,982 rows, Physio: ~24,876 rows)
2. Fit `StandardScaler` per modality on training folds only (no leakage)
3. Train `StressSeqEncoder` (CNN-GRU) for Face (input_dim=18), Voice (input_dim=12), Physio (input_dim=5) using Time Masking
4. Evaluate each encoder under 5-Fold GroupKFold LOSO
5. Train `FlexDynamicRouter` (MLP) with Modality Dropout (randomly mask 1–2 modalities per batch)
6. Evaluate router under all 7 modality combinations
7. Save models + scalers + config to `models/`
8. Update `models/registry.json` and per-file manifests

**Models produced (final production)**:
- `models/deep_face_expert.pt` → LOSO 55.10% ± 4.58%
- `models/deep_voice_expert.pt` → LOSO 61.46% ± 3.14% ← Best unimodal
- `models/deep_physio_expert.pt` → LOSO 58.95% ± 4.48%
- `models/deep_fusion_router.pt` → LOSO 58.26% ± 3.03% (all 3 sensors)
- `models/deep_face_scaler.pkl`, `models/deep_voice_scaler.pkl`, `models/deep_physio_scaler.pkl`
- `models/deep_fusion_config.json`

**Run command**:
```bash
python training/package_phase8_production.py
```

---

### generalization_research.py
**Phase**: 8.2  
**Purpose**: Formal identity-leakage audit comparing 5 training strategies under both random-split and strict LOSO validation.

**Strategies**:
1. Classical RF with raw features
2. Classical RF with subject-normalized features
3. Classical RF with stress-only feature subset (filtered identity-adjacent features)
4. Deep CNN-GRU sequence model (subject-adaptive normalization)
5. Deep CNN-GRU with adversarial identity suppression (gradient reversal on subject head)

**Key outputs**:
```
Strategy 1 — Classical Raw:        Random=87.96%  LOSO=61.45%  Gap=26.51%
Strategy 2 — Normalized RF:        Random=85.93%  LOSO=66.94%  Gap=18.99%
Strategy 3 — Stress-Only RF:       Random=85.61%  LOSO=66.81%  Gap=18.80%
Strategy 4 — Deep CNN-GRU:         Random=74.52%  LOSO=66.91%  Gap=7.62%  ← SELECTED
Strategy 5 — Adversarial Deep:     Random=73.08%  LOSO=65.64%  Gap=7.43%
```

**Run command**:
```bash
python training/generalization_research.py
```

---

## How to Reproduce Training from Scratch

```bash
# Step 1: Train classical Phase 4 baselines
python training/train_phase4_release.py

# Step 2: Run augmentation ablation (optional, for research)
python training/run_augmentation_experiments.py

# Step 3: Run Phase 7 deep learning research (15-subj, optional)
python training/phase7_deep_learning_research.py

# Step 4: Train final production models (65 subjects, all modalities)
python training/package_phase8_production.py

# Step 5: Run generalization audit (optional, for research)
python training/generalization_research.py
```

> **Note**: Steps 1 and 4 produce the actual model files saved to `models/`. Steps 2, 3, and 5 produce research reports only and do not modify production model files.
