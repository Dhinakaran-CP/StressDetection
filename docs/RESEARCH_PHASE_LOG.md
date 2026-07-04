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
