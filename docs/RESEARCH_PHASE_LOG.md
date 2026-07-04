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

