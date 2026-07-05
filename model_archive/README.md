# 🧠 StressDetectionUsingML — Model Archive & Preservation Record

> **Purpose**: This folder is the permanent preservation archive for all trained model artifacts in the StressDetectionUsingML project. It serves as an evidence record of every model version, their training history, architecture details, and validated performance metrics. No model files in this folder should ever be deleted or overwritten.

---

## 📁 Archive Structure

```
model_archive/
├── README.md                          ← This file — master evidence record
├── deep_models/                       ← Production PyTorch deep neural network models (Phase 8)
│   ├── deep_face_expert.pt            ← Face CNN-GRU sequence encoder
│   ├── deep_face_scaler.pkl           ← StandardScaler for Face features
│   ├── deep_voice_expert.pt           ← Voice CNN-GRU sequence encoder
│   ├── deep_voice_scaler.pkl          ← StandardScaler for Voice features
│   ├── deep_physio_expert.pt          ← Physio CNN-GRU sequence encoder
│   ├── deep_physio_scaler.pkl         ← StandardScaler for Physio features
│   ├── deep_fusion_router.pt          ← Flex-Modality Dynamic Router MLP
│   ├── deep_fusion_config.json        ← Runtime fusion configuration
│   └── DEEP_MODELS_DETAIL.md          ← Detailed per-model documentation
├── classical_models/                  ← Phase 4 classical ML baseline models (sklearn)
│   ├── face_expert_lightweight.pkl    ← Face classical expert (Random Forest)
│   ├── face_scaler_lightweight.pkl    ← StandardScaler for classical Face features
│   ├── voice_expert_lightweight.pkl   ← Voice classical expert
│   ├── voice_scaler_lightweight.pkl   ← StandardScaler for classical Voice features
│   ├── physio_expert_lightweight.pkl  ← Physio classical expert
│   ├── physio_scaler_lightweight.pkl  ← StandardScaler for classical Physio features
│   └── CLASSICAL_MODELS_DETAIL.md    ← Detailed per-model documentation
├── training_scripts/                  ← All Python scripts used to train the models
│   ├── phase4_experiments.py          ← Phase 2-4 classical ML ablations
│   ├── train_phase4_release.py        ← Trains & registers all classical baseline models
│   ├── train_face_expert_release.py   ← Trains Face classical expert only
│   ├── train_voice_expert_release.py  ← Trains Voice classical expert only
│   ├── train_physio_expert_release.py ← Trains Physio classical expert only
│   ├── release_expert_model.py        ← Registers model into registry.json
│   ├── augmentation.py                ← Data augmentation utilities
│   ├── run_augmentation_experiments.py← Ablation: 5 augmentation strategies
│   ├── phase6_multimodal_research.py  ← Fusion strategy comparisons
│   ├── phase7_deep_learning_research.py← Deep CNN-GRU research (15-subject subset)
│   ├── phase8_best_expert_fusion.py   ← Intermediate 2-way fusion
│   ├── package_phase8_production.py   ← FINAL production training (all 65 subjects)
│   ├── generalization_research.py     ← Identity leakage audit (5 strategies)
│   └── TRAINING_SCRIPTS.md           ← Detailed per-script documentation
├── reports/                           ← All benchmark and research reports
│   ├── phase8_final_fusion_benchmark.md
│   ├── generalization_leakage_audit.md
│   ├── phase7_deep_learning_benchmark.md
│   ├── phase7_augmentation_comparison.md
│   ├── phase6_fusion_benchmark.md
│   └── multimodal_architecture_comparison.md
└── docs/                              ← Copied research documentation
    ├── RESEARCH_PHASE_LOG.md
    └── stress_detection_architecture_report.md
```

---

## 🏆 Final Production Models Summary

| Model File | Version | Architecture | Validation | Accuracy | F1-Score |
|---|---|---|---|---|---|
| `deep_face_expert.pt` | v2.0 | PyTorch 1D-CNN + GRU | LOSO 5-Fold (65 subjects) | **55.10%** | ~55% |
| `deep_voice_expert.pt` | v2.0 | PyTorch 1D-CNN + GRU | LOSO 5-Fold (65 subjects) | **61.46%** | ~62% |
| `deep_physio_expert.pt` | v2.0 | PyTorch 1D-CNN + GRU | LOSO 5-Fold (65 subjects) | **58.95%** | ~59% |
| `deep_fusion_router.pt` | v2.0 | PyTorch MLP Flex-Router | LOSO 5-Fold (65 subjects) | **58.26%** (all 3 active) | ~58% |

> **Validation Protocol**: Strict Leave-One-Subject-Out (LOSO) 5-Fold GroupKFold. No subject appears in both train and test folds. Prevents subject-identity leakage.

---

## 📊 Full Fusion Benchmark — All Sensor Combinations

| Modality Combination | Mean Accuracy (LOSO) | Std Dev |
|---|---|---|
| Face Only | 55.10% | ± 4.58% |
| Voice Only | **61.46%** | ± 3.14% |
| Physio Only | 58.95% | ± 4.48% |
| Face + Physio | 57.89% | ± 3.70% |
| Face + Voice | 55.57% | ± 3.86% |
| Voice + Physio | 58.27% | ± 2.53% |
| **Face + Voice + Physio (All)** | **58.26%** | ± 3.03% |

---

## 📈 Model Training History (All Versions)

### Face Expert

| Version | Phase | Framework | Accuracy | Notes |
|---|---|---|---|---|
| v1.0 (`face_expert_v1`) | Phase 4 | sklearn Classical RF | **56.99%** | F1=56.05%, LOSO |
| v2.0a | Phase 7 (15-subj) | PyTorch CNN-GRU | **66.30%** | 15-subject subset |
| **v2.0b (CURRENT)** | **Phase 8 (65-subj)** | **PyTorch CNN-GRU** | **55.10%** | **Full 65 subjects** |

### Voice Expert

| Version | Phase | Framework | Accuracy | F1 | Notes |
|---|---|---|---|---|---|
| v1.0 (`voice_expert_v1`) | Phase 4 | sklearn Classical RF | **59.52%** | 70.46% | LOSO |
| **v2.0 (CURRENT)** | **Phase 8 (65-subj)** | **PyTorch CNN-GRU** | **61.46%** | ~62% | **Best unimodal** |

### Physio Expert

| Version | Phase | Framework | Accuracy | F1 | Notes |
|---|---|---|---|---|---|
| v1.0 (`physio_expert_v1`) | Phase 4 | sklearn Classical RF | **70.51%** | 60.88% | Highest ever recorded |
| v2.0a | Phase 7 (15-subj) | PyTorch CNN-GRU | **64.94%** | ~65% | 15-subject subset |
| **v2.0b (CURRENT)** | **Phase 8 (65-subj)** | **PyTorch CNN-GRU** | **58.95%** | ~59% | **Full 65 subjects** |

### Fusion Router

| Version | Phase | Architecture | Accuracy | Notes |
|---|---|---|---|---|
| v1.0 | Phase 7 (15-subj) | PyTorch MLP 2-way | **67.44%** | Face+Physio only |
| **v2.0 (CURRENT)** | **Phase 8 (65-subj)** | **PyTorch MLP Flex-Router** | **58.26%** | **3-way + Modality Dropout** |

---

## 🔬 Generalization & Leakage Audit

| Strategy | Random Split | LOSO | Leakage Gap | Decision |
|---|---|---|---|---|
| Classical RF (raw) | 87.96% | 61.45% | 26.51% | ❌ Too much leakage |
| Subject-Normalized RF | 85.93% | 66.94% | 18.99% | ❌ High leakage |
| Stress-Only Features RF | 85.61% | 66.81% | 18.80% | ❌ Similar leakage |
| **Deep CNN-GRU** | **74.52%** | **66.91%** | **7.62%** | ✅ **Selected** |
| Adversarial Deep CNN-GRU | 73.08% | 65.64% | 7.43% | ⚠️ Lower abs. score |

---

## 🔐 SHA-256 Integrity Hashes

| File | Hash |
|---|---|
| `deep_face_expert.pt` | `7dc1652cb7c0f3b763eaa3de36428a09587be427dca97f3da52fd49a0d1495fe` |
| `deep_voice_expert.pt` | `db41acbc550915d476ba56b06c76c21d27cb4822b06aad49f5579ff5c1ec677e` |
| `deep_physio_expert.pt` | `b7e7568ef077352aa67415d4cacc45e26945864dc414d059edb6fce8aa566623` |
| `deep_fusion_router.pt` | `a562be250cd15d607d3c79f93f5f96bce3314c89b3839c047e186526852d48c2` |
| `face_expert_lightweight.pkl` | `a2027fcf87fc74580e7448cd1a948d4d23af719a6ddda5d9341a907e8bc51f27` |
| `voice_expert_lightweight.pkl` | `5a1df684fc3fd167b996c83aa1168a75207c7ead6282ed66b88651b0345efc0a` |
| `physio_expert_lightweight.pkl` | `79c7e0905649f28582bdf8b260f17f07b2019527930cc39388f0380e4e153d35` |

---

## 📅 Archive Metadata

- **Date Archived**: 2026-07-05
- **Project**: StressDetectionUsingML
- **Repository**: Kishor-9361/StressDetectionUsingML
- **Total Model Versions Trained**: 11 versions across 4 model types
- **Total Research Phases**: 8 phases (Phase 1–8.2)
- **Total Ablation Configurations Tested**: 20+
- **Final Test Suite Result**: 97 tests, 100% pass rate
- **Dataset**: StressID (LORIA Lab, France) — 65 subjects, 11 task conditions
