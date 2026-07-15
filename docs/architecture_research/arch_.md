# StressDetectionUsingML — Full Architecture Understanding and Project Memory Audit

## Purpose

This document is the source-of-truth orientation guide for any new agent working on this repository.

It explains the full project architecture, the data pipeline, the model design choices, the validation rules, the deployment layout, and the final production decisions so the agent can understand the system end-to-end before making changes.

The project has already gone through multiple stages of data preparation, subject-leakage control, unimodal modeling, multimodal fusion, adversarial identity suppression, runtime packaging, and production verification.

---

## 1. Project Goal

The goal of the project is to detect stress from multimodal data using:
- Face,
- Voice,
- Physio,
- and fused multimodal representations.

The system must generalize across unseen subjects and must not rely on subject identity leakage, random split optimism, or hidden baseline shortcuts.

---

## 2. Current Final Philosophy

The project is no longer treated as a simple model benchmark.

It is a production-oriented multimodal stress detection system with:
- strict subject-independent evaluation,
- calibrated prediction pipelines,
- subject-adaptive and leakage-safe normalization,
- modular unimodal experts,
- fusion and router logic,
- and a primary/fallback deployment strategy.

The current architecture is designed to maximize generalization, runtime stability, and interpretability.

---

## 3. Data Sources

The repository uses synchronized multimodal windows across:
- Face,
- Voice,
- Physio.

The data is aligned by:
- `subject_id`,
- `task_id`,
- `video_id`,
- and `window_index`.

This alignment ensures that corresponding Face, Voice, and Physio samples represent the same subject and temporal segment.

---

## 4. Dataset Extraction Logic

The extraction phase created synchronized windows so that each modality follows the same time reference.

Key assumptions:
- the modalities are temporally aligned,
- the same window index refers to the same event segment,
- and windows are grouped by subject for evaluation.

The agent must preserve this contract.

---

## 5. Leakage and Validation Policy

This project treats leakage control as mandatory.

### Allowed validation
- GroupKFold.
- Leave-One-Subject-Out (LOSO).
- Strict subject-wise evaluation only.

### Forbidden validation
- Random row-wise split for final reporting.
- Any split where the same subject appears in both train and test.
- Global preprocessing before fold separation.

The project previously observed inflated results under random splits and lower but more trustworthy results under strict subject-wise validation.

---

## 6. Feature Contract

The active feature contract contains 35 features.

Some features were identified as risky because they may encode identity or absolute baseline behavior rather than stress response.

Risky features filtered from emphasis include:
- `face_height_norm`
- `landmark_confidence`
- `f0_mean`
- `f0_range`
- `eda_scl_mean`

The agent must not reintroduce these as primary stress cues without explicit justification.

---

## 7. Core Modeling History

The project evaluated multiple methodology families:

### Classical baselines
- Face-only.
- Voice-only.
- Physio-only.
- Calibrated versions of each.
- Naive average fusion.
- Meta stacking fusion.

### Deep sequence models
- Face CNN-GRU.
- Voice CNN-GRU.
- Physio CNN-GRU.
- Flex-router multimodal fusion.

### Leakage-aware models
- Adversarial CNN-GRU encoders.
- Subject-identity suppression using gradient reversal.
- Tuned adversarial loss weighting.

### Modular design direction
- Separate modality experts.
- Dynamic routing.
- Lightweight fallback logic.
- Multi-expert decision architecture.

---

## 8. Final Production Strategies

Two production strategies are currently locked:

### Strategy 5 — Primary
- Adversarial CNN-GRU encoders.
- Gradient reversal identity suppression.
- Tuned adversarial coefficient.
- Primary production deployment.

### Strategy 4 — Secondary fallback
- Standard CNN-GRU encoders.
- Non-adversarial fallback path.
- Used when adversarial artifacts are missing or invalid.

The runtime loader must try Strategy 5 first and fall back to Strategy 4 if needed.

---

## 9. Why Strategy 5 is Primary

Strategy 5 is the primary system because it:
- suppresses subject identity,
- reduces leakage risk,
- and preserves strong subject-independent performance.

The production decision is based on the balance of:
- accuracy,
- leakage gap,
- stability,
- and deployability.

---

## 10. Fusion and Multi-Expert Direction

The project is also exploring richer multimodal interaction methods.

The likely research-grade directions include:
- early fusion,
- gated fusion,
- cross-attention,
- co-attention,
- and committee / mixture-of-experts decision layers.

The important idea is that synchronized modalities can be fused not just by concatenation, but by learning:
- modality agreement,
- modality reliability,
- and expert selection.

This is relevant because the data is synchronized and window-aligned.

---

## 11. Current Architectural Interpretation

The system should be understood as a layered architecture:

### Layer A: Data ingestion
- Load synchronized multimodal windows.
- Maintain subject and task integrity.
- Preserve the feature contract.

### Layer B: Preprocessing
- Apply fold-safe scaling.
- Apply subject-aware baseline normalization where applicable.
- Avoid leakage across subjects.

### Layer C: Modality encoders
- Encode Face, Voice, and Physio separately.
- Allow unimodal performance measurement.

### Layer D: Fusion / routing
- Combine modalities using router, gating, or expert decision logic.
- Handle missing or weak modalities gracefully.

### Layer E: Final classifier
- Predict stress class.
- Emit calibrated output.

---

## 12. Runtime Loader Behavior

The runtime engine must:
- read the primary strategy from config,
- attempt to load adversarial artifacts first,
- verify model and scaler presence,
- and fall back to standard artifacts if needed.

The loader must not silently mix incompatible artifact sets.

The routing behavior is part of the production contract.

---

## 13. Model Registry and Artifact Policy

All approved artifacts must be recorded in the model registry.

The registry should distinguish:
- primary production models,
- fallback production models,
- archived experiments,
- and retired leakage-prone baselines.

The repository should keep experimental outputs separate from production assets.

---

## 14. Test and Verification Status

The repository has already passed the complete automated test suite.

The current production setup was verified against:
- backend code,
- runtime loading,
- model artifact selection,
- and fallback behavior.

The agent must preserve testability and never merge unverified runtime changes into production paths.

---

## 15. Repository Organization

The repository should conceptually be organized around:
- backend,
- frontend,
- models,
- model_archive,
- docs,
- configs,
- tests,
- and reports.

Root-level clutter should not be treated as part of the active architecture.

The agent should keep the repo audit-friendly and minimize ambiguous file placement.

---

## 16. Clinical / Professional Presentation Direction

The project presentation is expected to look:
- professional,
- clinical,
- clean,
- and research-grade.

This includes:
- consistent color systems,
- readable charts,
- clean UI hierarchy,
- and clearly labeled modality and model sections.

The project should present as a serious multimodal stress detection platform, not a prototype notebook dump.

---

## 17. Current Research Direction

The project is now open to high-end experimentation because compute/storage are not constraints.

Recommended future research directions include:
- cross-attention on synchronized windows,
- gated fusion,
- mixture-of-experts,
- modality-specific experts,
- and hybrid multi-stage fusion systems.

The agent should compare these methods experimentally rather than replacing the production baseline without evidence.

---

## 18. Decision Rules

The agent must follow these rules:

- Preserve the primary Strategy 5 baseline.
- Preserve Strategy 4 as fallback.
- Do not reintroduce leakage-prone validation.
- Do not use random split results as final proof.
- Keep synchronized multimodal alignment intact.
- Keep production artifact loading deterministic.
- Treat new fusion ideas as research extensions unless they beat the locked baseline.

---

## 19. What the Agent Must Understand

Before making any changes, the agent must understand:
- how the data was extracted,
- how modality synchronization works,
- why subject-independent validation is required,
- why the adversarial model was introduced,
- why fallback support exists,
- and why the final architecture is modular rather than monolithic.

---

## 20. Current System Summary

The final system is:
- a synchronized multimodal stress detection pipeline,
- trained under subject-independent validation,
- hardened against leakage,
- packaged with primary and fallback production strategies,
- and ready for advanced multimodal research extensions.

Any future changes must improve the system without breaking these guarantees.

---

## 21. Agent Instruction Block

Use this document as the authoritative architecture map.

When modifying the repository:
- read this file first,
- preserve the current production assumptions,
- update this file whenever the architecture changes,
- and never make changes that conflict with the locked strategy or validation policy.

---

## 22. Final Note

This repository is no longer in exploration mode.

It is a production-grade multimodal stress detection system with a controlled architecture, verified runtime behavior, and a clear future research path.