# StressDetectionUsingML — Phase-by-Phase Multimodal Architecture Plan

## Purpose

This document defines the full end-to-end architecture and execution plan for the repository.

It is written for a new agent so the agent can understand:
- the data contract,
- the validation rules,
- the model evolution history,
- the current production baseline,
- the research directions,
- and the exact step-by-step procedure to improve the system safely.

The project is a synchronized multimodal stress detection system built from Face, Voice, and Physio data, and it must remain subject-independent, leakage-safe, and production-ready [web:247][web:243][web:283][web:301].

---

## 1. Project Objective

The goal is to detect stress from multimodal data with the highest possible subject-independent generalization.

The system must:
- learn stress-related patterns,
- suppress subject identity leakage,
- support synchronized modalities,
- work under strict GroupKFold / LOSO evaluation,
- and remain easy to deploy, test, and extend.

This repository is not a notebook experiment folder. It is a production-oriented research system with a clear baseline, fallback, and future research path [web:247][web:243][web:301].

---

## 2. Current Final State

The project currently contains:
- a primary adversarial CNN-GRU production path,
- a fallback standard CNN-GRU production path,
- synchronized multimodal inputs,
- strict subject-wise validation,
- tuned identity suppression,
- runtime artifact routing,
- and a verified test suite.

These decisions must be preserved unless a new architecture clearly outperforms them under subject-independent evaluation [web:283][web:301][web:305].

---

## 3. Data Contract

The data is synchronized across modalities and aligned by:
- `subject_id`,
- `task_id`,
- `video_id`,
- `window_index`.

The agent must preserve this synchronization contract throughout all future work.

### Modalities
- Face.
- Voice.
- Physio.

### Required assumption
Each multimodal sample refers to the same subject and time window across modalities.

If this contract breaks, all fusion results become unreliable.

---

## 4. Validation Policy

This project uses strict leakage-aware validation.

### Allowed
- GroupKFold.
- Leave-One-Subject-Out.
- Subject-wise calibration.
- Fold-wise preprocessing.

### Forbidden
- Random row-wise splitting for final evaluation.
- Global scaling before splitting.
- Feature selection using held-out subject information.
- Any evaluation where the same subject appears in train and test.

This policy is mandatory because subject identity leakage can inflate metrics and hide poor real-world generalization [web:247][web:243][web:301].

---

## 5. Feature Policy

The current feature contract contains 35 features.

Some features were identified as risky because they may encode identity or baseline rather than stress response:
- `face_height_norm`
- `landmark_confidence`
- `f0_mean`
- `f0_range`
- `eda_scl_mean`

The agent must treat these as leakage-risk candidates and not as automatically trusted stress features.

---

## 6. Architecture Evolution

The project evolved through several phases:

### Phase A: Classical baselines
- Face-only.
- Voice-only.
- Physio-only.
- Random split and KFold exploration.

### Phase B: Leakage hardening
- Subject-wise splitting.
- Calibration-aware training.
- Feature filtering.
- Leakage-gap analysis.

### Phase C: Deep sequence models
- CNN-GRU encoders.
- Subject-safe LOSO evaluation.
- Production candidate ranking.

### Phase D: Adversarial identity suppression
- Gradient reversal.
- Identity suppression tuning.
- Strategy 5 primary baseline.

### Phase E: Modular multimodal production
- Flex router.
- Primary / fallback artifact selection.
- Runtime loader logic.
- Stable deployment packaging.

### Phase F: Future high-capacity research direction
- Cross-attention.
- Gated fusion.
- Multi-expert specialization.
- Modality-internal expert routing.
- Hybrid MoE architectures.

---

## 7. Locked Production Baselines

### Primary
Strategy 5 — adversarial CNN-GRU production system.

Purpose:
- suppress identity cues,
- preserve subject-independent learning,
- improve generalization safety.

### Fallback
Strategy 4 — standard CNN-GRU production system.

Purpose:
- provide a stable fallback if adversarial artifacts fail,
- preserve deployability,
- maintain testable continuity.

The agent must not replace these without a documented, fold-safe, subject-independent win.

---

## 8. Current Research Direction

The project now supports more powerful multimodal research because compute and storage are not constraints.

The strongest families to explore are:

### 8.1 Late fusion
Simple averaging or weighted voting over modality experts.

Useful as a control baseline and runtime-safe fallback.

### 8.2 Early fusion
Concatenate synchronized modality embeddings early and learn joint representation.

Useful when windows are strictly aligned and interaction is strong.

### 8.3 Gated fusion
Let a router weight each modality dynamically.

Useful when one modality is noisy, weak, or missing.

### 8.4 Cross-attention fusion
Let Face attend to Voice/Physio, Voice attend to Face/Physio, etc.

Useful when synchronized modalities should explicitly exchange information.

### 8.5 Multi-expert / MoE fusion
Use multiple experts per modality or per feature group and let a router choose.

Useful when different feature groups specialize in different stress cues [web:283][web:301][web:305][web:259].

### 8.6 Hybrid architecture
Use:
- modality experts,
- cross-attention,
- gating,
- and calibration together.

This is the highest-capacity direction and the most promising research path for this repo [web:283][web:301][web:305].

---

## 9. Expert Design Philosophy

Do not create one full model per scalar feature.

Instead, create **coherent experts** for feature groups.

### Face example
- Eye expert.
- Mouth/lip expert.
- Global face expert.

### Voice example
- Prosody expert.
- Spectral expert.
- Voice quality expert.

### Physio example
- Cardio expert.
- EDA expert.
- Motion or breathing expert.

This matches modern MoE design principles, where a few specialized experts are more effective than many tiny feature-wise models [web:299][web:300][web:301][web:305].

---

## 10. Recommended High-Capacity Architecture Ladder

The agent must evaluate models in this order:

### Stage 1: Unimodal expert baselines
Train separate Face, Voice, and Physio experts.

Goal:
- establish modality quality,
- verify each modality’s independent signal.

### Stage 2: Simple fusion
Build late fusion and weighted averaging baselines.

Goal:
- produce a low-risk multimodal reference.

### Stage 3: Gated fusion
Add a router that learns modality weights.

Goal:
- adapt to task-wise and sample-wise modality reliability.

### Stage 4: Cross-attention fusion
Add inter-modality interaction.

Goal:
- model agreement, disagreement, and contextual support.

### Stage 5: Multi-expert specialization
Split each modality into sub-experts by feature group.

Goal:
- capture intra-modality structure.

### Stage 6: Hybrid MoE + attention
Combine:
- experts,
- attention,
- gating,
- and calibration.

Goal:
- maximize subject-independent performance and robustness [web:283][web:301][web:305][web:259].

---

## 11. Feature Extraction Policy

Do not rebuild feature extraction just because a new architecture exists.

Only redesign extraction if it produces a clear improvement in:
- representation quality,
- synchronization accuracy,
- or modality-specific signal usefulness.

### Keep extraction if:
- windows are aligned correctly,
- feature contract is stable,
- and performance is already consistent.

### Rebuild extraction if:
- features are too shallow,
- there is temporal mismatch,
- or deeper learned encoders can replace manual features.

The current best practice is to improve representation and fusion first, not to repeatedly re-extract the same raw signals without cause [web:247][web:250][web:287].

---

## 12. Identity Leakage Control

The agent must actively suppress subject identity.

Required methods:
- subject-wise splitting,
- per-fold preprocessing,
- baseline normalization where valid,
- feature filtering,
- adversarial identity suppression,
- and leakage-gap measurement.

If random-split performance is much higher than LOSO performance, the model is learning identity shortcuts and must be revised.

---

## 13. Calibration Policy

Every final model must be calibrated.

Reason:
- fusion outputs need stable probabilities,
- router outputs can become overconfident,
- and production inference must remain meaningful.

Calibration must be performed fold-safely and must not leak test subjects.

---

## 14. Runtime Architecture

The runtime loader must:
- read the primary strategy from config,
- load adversarial artifacts first,
- verify required model and scaler files,
- fall back to the standard deep model if necessary,
- and never mix incompatible artifact sets.

The runtime system must be deterministic and reproducible.

---

## 15. Repository Organization

The repository should be cleanly separated into:
- backend,
- frontend,
- configs,
- models,
- model_archive,
- reports,
- docs,
- tests,
- and deployment artifacts.

The agent must not leave production assets mixed with experimental outputs.

---

## 16. Clinical / Professional Presentation

The project presentation should look:
- clinical,
- professional,
- clean,
- and research-grade.

This includes:
- a calm healthcare palette,
- high-contrast readable text,
- clear chart styling,
- and organized documentation.

---

## 17. Phase-by-Phase Execution Procedure

### Phase 1: Audit
- Read this file first.
- Confirm the current artifact locations.
- Confirm feature contract and modality synchronization.
- Confirm the locked primary and fallback strategies.

### Phase 2: Data validation
- Verify synchronized window alignment.
- Check for missing windows or mismatch.
- Confirm that train/test grouping is subject-safe.

### Phase 3: Baseline preservation
- Keep Strategy 5 and Strategy 4 production paths.
- Do not break runtime fallback logic.

### Phase 4: Build modality experts
- Define expert groups per modality.
- Train unimodal experts first.
- Evaluate each expert independently.

### Phase 5: Add fusion strategies
- Implement late fusion.
- Implement gated fusion.
- Implement cross-attention fusion.
- Implement MoE routing if resources allow.

### Phase 6: Identity suppression
- Use adversarial loss where applicable.
- Verify that the leakage gap decreases.

### Phase 7: Calibration and validation
- Calibrate outputs fold-wise.
- Compare random split vs subject-wise split.
- Keep only the subject-safe results.

### Phase 8: Production packaging
- Register approved artifacts.
- Keep primary and fallback models separate.
- Confirm runtime loader behavior.

### Phase 9: Documentation
- Update architecture docs.
- Update feature docs.
- Update model registry.
- Update phase log.

### Phase 10: Test and commit
- Run full tests.
- Verify deployment.
- Commit only validated changes.

---

## 18. Decision Rules

The agent must follow these rules:

- Prefer subject-independent performance over random-split gains.
- Prefer modular expert systems over a single monolithic fusion block.
- Prefer gated or attention-based fusion for synchronized data.
- Prefer feature-group experts over one-model-per-feature.
- Prefer a hybrid architecture when compute is available.
- Prefer measurable leakage reduction over intuition.
- Never remove current production baselines without a better validated replacement.

---

## 19. What the Agent Must Preserve

The agent must preserve:
- synchronized data alignment,
- feature contract,
- subject-wise validation,
- adversarial primary strategy,
- fallback strategy,
- runtime load safety,
- and repository cleanliness.

---

## 20. Final System View

The final system is a synchronized multimodal stress detection platform that:
- learns stress cues rather than identity cues,
- supports modular experts,
- can scale to cross-attention and MoE fusion,
- and remains deployable with a primary/fallback runtime policy.

The project is ready for deeper architecture work, but only if the next changes are guided by strict validation and clean documentation.

---

## 21. Agent Instruction Block

Read this document before any change.

When implementing:
- preserve the current locked baselines,
- keep the data synchronization contract,
- expand to modular experts only with proper validation,
- compare fusion strategies methodically,
- and update this file whenever the architecture changes.

Do not make changes that weaken subject-independent generalization.