# ML Research and Validation Phase Log

This log records the progress, metrics, and decisions for the 8-phase StressDetectionUsingML research pipeline.

---

## Phase 1 Log

- Date: 2026-07-04
- Branch: `research-loso-pipeline`
- Commit: `phase1: baseline audit and repo structure`
- Objective: Scan and audit the repository to map files, identify redundant/conflicting components, and inventory current baselines.
- Changes:
  - Created [docs/RESEARCH_PHASE_LOG.md](file:///e:/Document/GitHub/StressDetectionUsingML/docs/RESEARCH_PHASE_LOG.md) to track experimental research progress.
  - Audited training scripts: [phase4_experiments.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase4_experiments.py), [phase6_multimodal_research.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase6_multimodal_research.py), [phase7_deep_learning_research.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase7_deep_learning_research.py), [phase8_best_expert_fusion.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase8_best_expert_fusion.py), [package_phase8_production.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/package_phase8_production.py).
  - Audited active model registry: [models/registry.json](file:///e:/Document/GitHub/StressDetectionUsingML/models/registry.json).
  - Audited feature configurations: [configs/feature_contract.yaml](file:///e:/Document/GitHub/StressDetectionUsingML/configs/feature_contract.yaml).
- Metrics:
  - **Classical Baseline Experts (from registry.json)**:
    - Face Expert: Accuracy = 56.99%, F1-Score = 56.05%
    - Voice Expert: Accuracy = 59.52%, F1-Score = 70.46%
    - Physio Expert: Accuracy = 70.51%, F1-Score = 60.88%
  - **Calibrated Classical Naive Average (3-Way)**: ~64.63%
  - **Deep CNN-GRU Baseline (from reports)**:
    - Face Encoder: 65.75%
    - Voice Encoder: 58.89%
    - Physio Encoder: 65.44%
    - Deep Gated Fusion: 64.59%
- Validation: Strict Leave-One-Subject-Out (LOSO) cross-validation utilizing `GroupKFold` on certified datasets.
- Risks:
  - **Identity Leakage**: Occurs if subject scaling/normalization is computed across the entire dataset rather than within training folds or per-subject calm periods.
  - **Code Clutter**: Multiple versions of training scripts exist, some of which are untracked draft files.
- Decision: Approved Phase 1 audit. Proceeding to Phase 2 to run a subject-safe classical pipeline with strict validation.

## Phase 2 Log

- Date: 2026-07-04
- Branch: `research-loso-pipeline`
- Commit: `phase2: subject-safe classical pipeline`
- Objective: Establish a subject-safe classical baseline and verify the impact of subject-aware normalization and temporal windowing.
- Changes:
  - Verified and executed [training/phase4_experiments.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase4_experiments.py).
- Metrics:
  - **Face Modality**:
    - Raw Baseline: Accuracy = 66.24%, F1 = 57.58%
    - Subject-Aware Norm: Accuracy = 69.04%, F1 = 59.57%
    - Temporal Windowing: Accuracy = 69.37%, F1 = 60.43%
  - **Voice Modality**:
    - Raw Baseline: Accuracy = 70.98%, F1 = 82.81%
    - Subject-Aware Norm: Accuracy = 70.70%, F1 = 82.75%
    - Temporal Windowing: Accuracy = 70.56%, F1 = 82.64%
  - **Physio Modality**:
    - Raw Baseline: Accuracy = 59.58%, F1 = 42.01%
    - Subject-Aware Norm: Accuracy = 67.22%, F1 = 57.05%
    - Temporal Windowing: Accuracy = 67.39%, F1 = 57.64%
- Validation: 3-fold GroupKFold cross-validation (subject-independent).
- Risks: Voice modality baseline metrics are surprisingly high (F1 > 0.82), which might indicate sample imbalance or a bias in the Voice dataset class labels, whereas training a heavier tuned model on it causes massive overfitting/performance drop (Accuracy drops to 51.68%).
- Decision: Approved Phase 2 results. Proceeding to Phase 3 (Temporal and Calibration Checks).

## Phase 3 Log

- Date: 2026-07-04
- Branch: `research-loso-pipeline`
- Commit: `phase3: calibration and temporal aggregation`
- Objective: Confirm the effect of temporal aggregation, probability calibration, and compare naive average against stacking.
- Changes:
  - Executed [training/phase6_multimodal_research.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase6_multimodal_research.py) to train calibrated base classifiers (Face MLP, Voice RF, Physio GBM) under GroupKFold with temporal rolling windows (window=3).
- Metrics:
  - **Calibrated Base Models**:
    - Face-Only (MLP): 0.5821 (± 0.0387)
    - Voice-Only (RF): 0.5872 (± 0.0545)
    - Physio-Only (GBM): 0.5541 (± 0.0602)
  - **Calibrated Fusion Results**:
    - Naive Average (3-Way): 0.6463 (± 0.0181)
    - Meta-Fusion Stacking (3-Way): 0.6302 (± 0.0497)
- Validation: 5-fold GroupKFold cross-validation (strict subject-independent).
- Risks: Meta-fusion stacking performs slightly worse and has higher fold variance than naive average, showing that learned meta-learners can overfit to specific subjects even under grouped validation.
- Decision: Confirmed the calibrated naive average 3-way configuration as the best classical baseline. Proceeding to Phase 4 (Deep Modality Encoders).

## Phase 4 Log

- Date: 2026-07-04
- Branch: `research-loso-pipeline`
- Commit: `phase4: deep modality encoders`
- Objective: Train compact 1D-CNN + GRU deep unimodal encoders and evaluate deep unimodal performance under strict subject-independent validation.
- Changes:
  - Executed [training/phase7_deep_learning_research.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase7_deep_learning_research.py) to train PyTorch-based sequence models (`seq_len=5`) using 1D-CNN + GRU architectures on individual modalities (Face, Voice, Physio) and a gated attention fusion network.
- Metrics:
  - **Deep Unimodal Encoders**:
    - Face-Only: 0.6630 (± 0.0325)
    - Voice-Only: 0.6069 (± 0.0433)
    - Physio-Only: 0.6494 (± 0.0276)
  - **Deep Gated Fusion Results**:
    - Gated Attention Fusion: 0.6744 (± 0.0290)
- Validation: 5-fold GroupKFold cross-validation (strict subject-independent, sliding window sequence length = 5).
- Risks: Training CNN-GRU sequence models on CPU is slow (~25 minutes), but yields significantly better unimodal representation learning, especially for Face (+8.1% accuracy gain) and Physio (+9.5% accuracy gain) over classical manual features.
- Decision: Confirmed the superiority of deep learning representation encoders for Face and Physio. Proceeding to Phase 5 (Best-Expert Selection).

## Phase 5 Log

- Date: 2026-07-05
- Branch: `research-loso-pipeline`
- Commit: `phase5: best-expert selection`
- Objective: Compare all candidate face, voice, and physio models to select exactly one best expert per modality based on strict subject-independent performance.
- Changes:
  - Executed [training/phase8_best_expert_fusion.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase8_best_expert_fusion.py) to evaluate unimodal sequence-based deep encoders under LOSO.
- Metrics (15-subject subset):
  - Face Expert (CNN-GRU): 0.5989 (± 0.0755)
  - Voice Expert (CNN-GRU): 0.5581 (± 0.1003)
  - Physio Expert (CNN-GRU): 0.5885 (± 0.0323)
- Validation: 5-fold GroupKFold (subject-independent).
- Risks: The Voice expert underperforms significantly compared to Face and Physio, exhibiting high variance (± 0.1003), which poses a high risk of degrading the fused pipeline performance.
- Decision: Select the CNN-GRU encoders for Face and Physio as the modality experts. Drop/exclude Voice from fusion consideration. Proceeding to Phase 6 (Fusion Engine).




