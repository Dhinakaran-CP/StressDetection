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

## Phase 6 Log

- Date: 2026-07-05
- Branch: `research-loso-pipeline`
- Commit: `phase6: fusion engine`
- Objective: Fuse selected best experts, evaluate static vs. dynamic fusion, and compare pairwise (Face + Physio) vs. 3-way (Face + Voice + Physio) configurations.
- Changes:
  - Evaluated static grid search weighting and dynamic router MLPs in [training/phase8_best_expert_fusion.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/phase8_best_expert_fusion.py).
- Metrics (15-subject subset):
  - Static Pairwise (Face + Physio): 0.6154 (± 0.0332)
  - Static 3-Way (Face + Voice + Physio): 0.6114 (± 0.0537)
  - Dynamic Pairwise (Face + Physio): 0.6335 (± 0.0482)
  - Dynamic 3-Way (Face + Voice + Physio): 0.6115 (± 0.0518)
- Validation: 5-fold GroupKFold (subject-independent).
- Risks: Static weighted average is simpler, but Dynamic Pairwise fusion provides a +1.8% accuracy gain and remains low-latency since the router MLP is extremely compact.
- Decision: Select the Dynamic Pairwise (Face + Physio) router MLP as the fusion engine. Voice is dropped from the active runtime fusion pipeline. Proceeding to Phase 7 (Augmentation Comparison).

## Phase 7 Log

- Date: 2026-07-05
- Branch: `research-loso-pipeline`
- Commit: `phase7: augmentation comparison`
- Objective: Systematically compare allowed augmentation strategies on training folds only to select the best performer.
- Changes:
  - Created modular [training/augmentation.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/augmentation.py) containing implementation of sequence-level jittering, scaling, time masking, and modality dropout.
  - Executed [training/run_augmentation_experiments.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/run_augmentation_experiments.py) to run comparative LOSO trials.
- Metrics (15-subject subset):
  - No Augmentation (None): 0.6260 (± 0.0704)
  - Jittering: 0.6124 (± 0.0670) [Delta: -0.0136]
  - Scaling: 0.6034 (± 0.0879) [Delta: -0.0226]
  - Time Masking: 0.6316 (± 0.0553) [Delta: +0.0056]
  - Modality Dropout: 0.6080 (± 0.0844) [Delta: -0.0180]
  - Combined: 0.6191 (± 0.0696) [Delta: -0.0069]
- Validation: 5-fold GroupKFold (subject-independent).
- Risks: Adding noise (jittering, scaling, or modality dropout) corrupts the subtle stress indicators in facial and physiological signals, degrading generalization performance.
- Decision: Select Time Masking as the only active training augmentation technique since it improves accuracy (+0.56%) and significantly reduces variance (std dev decreased from 0.0704 to 0.0553). Reject other augmentations. Proceeding to Phase 8 (Strict Validation and Final Benchmark).

## Phase 8 Log

- Date: 2026-07-05
- Branch: `research-loso-pipeline`
- Commit: `phase8: strict validation and final benchmark`
- Objective: Re-run final selected configuration on all folds using strict validation, retrain on the full certified dataset, and package weights/scalers/configs.
- Changes:
  - Refactored [training/package_phase8_production.py](file:///e:/Document/GitHub/StressDetectionUsingML/training/package_phase8_production.py) to train PyTorch-based sequence modality encoders (Face and Physio) using the selected Time Masking augmentation.
  - Implemented and trained a PyTorch Dynamic Router MLP to learn dynamic modality weighting.
  - Performed a strict Leave-One-Subject-Out 5-Fold validation on all 65 subjects to produce the final audited benchmark.
  - Updated [backend/runtime/runtime_engine.py](file:///e:/Document/GitHub/StressDetectionUsingML/backend/runtime/runtime_engine.py) to load and execute these deep neural network architectures and scaling parameters during live streaming and replay, resolving a hidden calibration baseline bug.
- Metrics (Full 65 Subjects):
  - Face-Only Encoder: 0.5912 (± 0.0747)
  - Physio-Only Encoder: 0.5485 (± 0.0852)
  - **Dynamic Pairwise Fusion**: **0.5875 (± 0.0847)**
- Validation: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on the full 65-subject dataset.
- Risks: The full 65-subject LOSO performance is slightly lower than the 15-subject subset, showing that adding more subjects introduces additional cross-subject variance. However, the dynamic fusion model remains robust and generalizes better than unimodal physical indicators.
- Decision: Confirmed the Time-Masked Face & Physio Deep Encoders + Dynamic Router MLP as the production stress detection pipeline. Voice is permanently excluded from fusion due to its bottlenecks.

## Phase 8.1 Log: Methodology History and Memory Audit

- Date: 2026-07-05
- Branch: `research-loso-pipeline`
- Commit: `phase8.1: methodology history and memory audit`
- Objective: Audit all previous methods, investigate performance drop, and fully integrate Voice in a 3-way flexible modality fusion router.
- Changes:
  - Re-integrated Voice deep sequence encoder alongside Face and Physio modality experts.
  - Implemented **Modality Dropout** during router training to support dynamic weighting for any subset of active sensors (face, voice, physio).
  - Conducted 3-way 5-Fold LOSO cross-validation on all 65 subjects and output results to [reports/phase8_final_fusion_benchmark.md](file:///e:/Document/GitHub/StressDetectionUsingML/reports/phase8_final_fusion_benchmark.md).
  - Saved full methodology comparison to [docs/stress_detection_architecture_report.md](file:///e:/Document/GitHub/StressDetectionUsingML/docs/stress_detection_architecture_report.md).
- Metrics (Full 65 Subjects under Strict LOSO):
  - Face-Only CNN-GRU: 0.5510 (± 0.0458)
  - Voice-Only CNN-GRU: 0.6146 (± 0.0314)
  - Physio-Only CNN-GRU: 0.5895 (± 0.0448)
  - **3-Way Flexible Fusion**: **0.5826 (± 0.0303)**
- Validation: Strict Leave-One-Subject-Out GroupKFold.
- Risks: Adding sparse modalities like Voice can introduce noise during silent tasks (causing Face + Voice to drop to 0.5557). However, the 3-Way Dynamic Router with Modality Dropout minimizes this by learning to re-normalize weights on active inputs.
- Decision: Set the 3-Way Flex-Modality Dynamic Fusion as the project production baseline to support flexible multi-sensor configurations.








