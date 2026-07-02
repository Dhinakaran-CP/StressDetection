# Agent Implementation Guide: Phase-by-Phase Execution Plan for the Multimodal Stress Platform

This document is a practical execution guide for an AI agent or developer to implement the architecture in the existing codebase **phase by phase**, updating or replacing files only where necessary, and creating a clean Git history with one commit per completed phase.

The goal is to make implementation:
- non-regressive,
- easy to review,
- easy to rollback,
- and easy to retrieve later using commit history.

This guide assumes the project already has an existing frontend, backend, model code, training code, and evaluation scripts.

---

## 1. Working Rules for the Agent

Before touching any code, the agent must follow these rules:

1. **Never modify multiple architectural phases in one commit.**  
   One commit must correspond to one completed phase only.

2. **Always inspect impact before editing.**  
   For every file to be changed, classify it as:
   - `REUSE`: keep as-is.
   - `UPDATE`: modify carefully.
   - `REPLACE`: rewrite because current structure blocks architecture.
   - `NEW`: add a new file.

3. **Do not silently change schemas, feature order, API payloads, or model input shape.**  
   Such changes must happen only in the correct contract phase.

4. **Do not delete old logic immediately if it can help backward compatibility.**  
   Prefer deprecation wrappers and adapters first.

5. **Every phase must end with: test, verification, documentation update, git commit.**

6. **Each commit message must be searchable and phase-specific.**

---

## 2. Repository Impact Mapping Strategy

Before implementation, the agent must scan the repo and map files into these likely groups.

### 2.1 Likely Existing Files

Based on your current project description, the likely existing important files are:

#### Backend runtime
- `backend/app.py`
- `backend/realtime_core.py`
- `backend/model.py`
- `backend/score_buffer.py`
- `backend/voice_worker.py`

#### Training and evaluation
- `backend/training/colab_training.py`
- `backend/training/strict_fused_evaluation.py`
- `backend/training/evaluate_fused_engine_bootstrapped.py`

#### Frontend
- `frontend/src/...`
- `public/facePostWorker.js`

#### Models / artifacts
- `face_expert_lightweight.pkl`
- `voice_expert_lightweight.pkl`
- `physio_expert.pkl`

### 2.2 Impact Categories

For each phase, the agent must create an internal impact table like this:

| File | Impact | Why |
|---|---|---|
| `backend/app.py` | UPDATE | Must load released runtime engine, not raw model calls |
| `backend/model.py` | REPLACE or SPLIT | Currently too overloaded; must become shared extraction + inference utilities |
| `backend/realtime_core.py` | UPDATE | Must use released contracts and runtime buffers |
| `backend/score_buffer.py` | UPDATE | Must support certified temporal logic |
| `backend/voice_worker.py` | UPDATE | Must use feature runtime lock and released feature ordering |
| `backend/training/colab_training.py` | UPDATE | Only if extraction contract is being aligned |
| `strict_fused_evaluation.py` | REPLACE | Must become grouped, contract-aware evaluator |

The agent should maintain this table inside the implementation notes for traceability.

---

## 3. Recommended Phase Implementation Order

The implementation should be done in **nine phases** for better control.

1. Phase 0 – Repository audit and architecture freeze  
2. Phase 1 – Contracts and versioning foundation  
3. Phase 2 – Dataset certification layer  
4. Phase 3 – Feature runtime lock layer  
5. Phase 4 – Expert training release pipeline  
6. Phase 5 – Fusion and temporal intelligence pipeline  
7. Phase 6 – Explainability release pipeline  
8. Phase 7 – Runtime engine integration  
9. Phase 8 – Integration, monitoring, rollout, and cleanup

Each phase is described below with:
- objective,
- files to update/create,
- implementation steps,
- tests,
- and git commit message.

---

## 4. Phase 0 – Repository Audit and Architecture Freeze

### Objective
Understand the current project state and create the implementation baseline.

### Files
- `NEW`: `docs/IMPLEMENTATION_BASELINE.md`
- `NEW`: `docs/REPO_IMPACT_MAP.md`
- `NEW`: `docs/ARCHITECTURE_PHASE_PLAN.md`

### Steps
1. Inspect repository structure.
2. List all runtime, training, evaluation, frontend, and model files.
3. Mark each file as `REUSE`, `UPDATE`, `REPLACE`, or `NEW`.
4. Record current entrypoints, current model loading flow, current API flow, and current evaluation flow.
5. Freeze implementation order so no later agent changes it casually.

### Completion Tests
- Repo map completed.
- All core files categorized.
- Existing runtime flow documented.
- Existing model and artifact dependencies documented.

### Git Commit
```bash
git add docs/IMPLEMENTATION_BASELINE.md docs/REPO_IMPACT_MAP.md docs/ARCHITECTURE_PHASE_PLAN.md
git commit -m "phase-0: audit repository and freeze implementation roadmap"
```

---

## 5. Phase 1 – Contracts and Versioning Foundation

### Objective
Create project-wide contracts so later phases cannot drift.

### Files
- `NEW`: `contracts/schema_contract.yaml`
- `NEW`: `contracts/feature_contract.yaml`
- `NEW`: `contracts/performance_contract.yaml`
- `NEW`: `contracts/api_contract.yaml`
- `NEW`: `backend/core/version_registry.py`
- `NEW`: `backend/core/artifact_manifest.py`

### Update Guidance
- No runtime behavior should change yet.
- This phase only introduces formal contracts and version machinery.

### Steps
1. Define schema fields for all extracted modality tables.
2. Define feature ordering and meaning for each model input.
3. Define latency, RAM, CPU, and missing-modality constraints.
4. Define backend-to-frontend API payload contract.
5. Implement version manifest utilities that register model, scaler, dataset, and explainability bundle versions.

### Completion Tests
- Contracts parse correctly.
- Registry utilities can write and read manifests.
- No code path uses hardcoded version values anymore.

### Git Commit
```bash
git add contracts/ backend/core/version_registry.py backend/core/artifact_manifest.py
git commit -m "phase-1: add contracts and artifact versioning foundation"
```

---

## 6. Phase 2 – Dataset Certification Layer

### Objective
Replace loose loading with certified dataset release logic.

### Files
- `NEW`: `backend/core/dataset_certifier.py`
- `NEW`: `backend/core/dataset_release.py`
- `NEW`: `backend/tests/test_dataset_certifier.py`
- `UPDATE`: existing training data loading utilities
- `UPDATE`: any scripts directly loading raw CSVs without validation

### Steps
1. Implement schema validation against `schema_contract.yaml`.
2. Add alignment validation across modalities.
3. Add baseline, transition, missingness, and chronology checks.
4. Generate release reports and checksums for certified datasets.
5. Update all training/evaluation scripts to consume only certified releases.

### Completion Tests
- Certified dataset release can be generated from current extracted files.
- Invalid schema fails clearly.
- Misaligned subject/task/window keys are detected.
- Certification report is created and versioned.

### Git Commit
```bash
git add backend/core/dataset_certifier.py backend/core/dataset_release.py backend/tests/test_dataset_certifier.py
git commit -m "phase-2: add certified dataset release and validation gates"
```

---

## 7. Phase 3 – Feature Runtime Lock Layer

### Objective
Guarantee identical transformation logic in offline and online paths.

### Files
- `NEW`: `backend/core/feature_runtime_lock.py`
- `NEW`: `backend/tests/test_feature_runtime_lock.py`
- `UPDATE`: `backend/model.py`
- `UPDATE`: `backend/voice_worker.py`
- `UPDATE`: training feature preparation modules

### Replace/Update Recommendation
- If `backend/model.py` currently mixes extraction, inference, and helper utilities, split it into:
  - `backend/core/feature_runtime_lock.py`
  - `backend/core/extractors/face_extractor.py`
  - `backend/core/extractors/voice_extractor.py`
  - `backend/core/inference_helpers.py`

### Steps
1. Move feature ordering into one locked module.
2. Move scaling, missing-value handling, baseline normalization, and feature grouping into that module.
3. Make both training and runtime import from this same module.
4. Add strict assertions for feature dimension and ordering.

### Completion Tests
- Same input gives same vector in training and runtime.
- Missing modality handling is deterministic.
- Feature order hash matches feature contract.

### Git Commit
```bash
git add backend/core/feature_runtime_lock.py backend/tests/test_feature_runtime_lock.py backend/model.py backend/voice_worker.py
git commit -m "phase-3: lock feature transformation path for offline-online parity"
```

---

## 8. Phase 4 – Expert Training Release Pipeline

### Objective
Refactor expert training into a release-based system with grouped evaluation and promotion rules.

### Files
- `NEW`: `backend/training/train_face_expert_release.py`
- `NEW`: `backend/training/train_voice_expert_release.py`
- `NEW`: `backend/training/train_physio_expert_release.py`
- `NEW`: `backend/training/release_expert_model.py`
- `NEW`: `backend/tests/test_expert_release_pipeline.py`
- `UPDATE`: `strict_fused_evaluation.py`
- `UPDATE`: `evaluate_fused_engine_bootstrapped.py`

### Replace/Update Recommendation
- If current evaluation scripts are too coupled or leaky, replace them with:
  - `evaluate_experts_grouped.py`
  - `evaluate_experts_loso.py`

### Steps
1. Make training scripts consume certified datasets only.
2. Use grouped CV or LOSO only.
3. Save model, calibration object, scaler reference, feature hash, metrics report, and manifest.
4. Reject a release if latency or LOSO metrics fail contract thresholds.

### Completion Tests
- No subject leakage.
- Each expert creates a versioned release folder.
- Inference latency benchmark passes.
- Calibration and stability reports exist.

### Git Commit
```bash
git add backend/training/ backend/tests/test_expert_release_pipeline.py
git commit -m "phase-4: implement expert model release pipeline with grouped validation"
```

---

## 9. Phase 5 – Fusion and Temporal Intelligence Pipeline

### Objective
Build release-grade buffer-aware fusion.

### Files
- `NEW`: `backend/fusion/train_fusion_release.py`
- `NEW`: `backend/fusion/fusion_contract.py`
- `NEW`: `backend/fusion/temporal_buffer_engine.py`
- `NEW`: `backend/tests/test_fusion_pipeline.py`
- `UPDATE`: `backend/score_buffer.py`
- `UPDATE`: fusion-related evaluation scripts

### Steps
1. Consume only released expert outputs.
2. Build temporal buffer simulation identical to runtime.
3. Train fixed-weight or meta-fusion model.
4. Add missing-modality and confidence-aware logic.
5. Create versioned fusion release bundle.

### Completion Tests
- Fusion outperforms or matches best unimodal expert.
- Runtime and offline temporal logic are equivalent.
- Missing modality fallback behaves correctly.

### Git Commit
```bash
git add backend/fusion/ backend/score_buffer.py backend/tests/test_fusion_pipeline.py
git commit -m "phase-5: add buffer-aware fusion release pipeline"
```

---

## 10. Phase 6 – Explainability Release Pipeline

### Objective
Generate stable, versioned explanation assets.

### Files
- `NEW`: `backend/explainability/build_explainability_bundle.py`
- `NEW`: `backend/explainability/explainability_contract.py`
- `NEW`: `backend/tests/test_explainability_bundle.py`
- `UPDATE`: any SHAP-related notebook or helper scripts

### Steps
1. Compute explanation tables from released experts.
2. Map every feature to a human-readable label and feature group.
3. Save explanation bundles with version manifests.
4. Ensure bundles are lightweight enough for runtime.

### Completion Tests
- Every feature used by a model has explanation metadata.
- Bundle loads correctly.
- Top explanations are stable enough across samples.

### Git Commit
```bash
git add backend/explainability/ backend/tests/test_explainability_bundle.py
git commit -m "phase-6: build versioned explainability release bundle"
```

---

## 11. Phase 7 – Runtime Engine Integration

### Objective
Refactor runtime to consume only released artifacts and locked feature transforms.

### Files
- `UPDATE`: `backend/app.py`
- `UPDATE`: `backend/realtime_core.py`
- `UPDATE`: `backend/model.py`
- `UPDATE`: `backend/score_buffer.py`
- `UPDATE`: `backend/voice_worker.py`
- `NEW`: `backend/runtime/runtime_engine.py`
- `NEW`: `backend/runtime/session_state.py`
- `NEW`: `backend/tests/test_runtime_engine.py`

### Steps
1. Create runtime engine wrapper around released experts, fusion, and explanations.
2. Replace direct model loading in `app.py` and `realtime_core.py` with artifact registry loading.
3. Ensure all inference passes through `feature_runtime_lock.py`.
4. Add graceful degradation for missing face, voice, or physio inputs.
5. Add deterministic replay test path.

### Completion Tests
- Runtime reproduces offline outputs on replay set.
- CPU and RAM remain within target.
- Streaming flow remains stable.

### Git Commit
```bash
git add backend/app.py backend/realtime_core.py backend/model.py backend/score_buffer.py backend/voice_worker.py backend/runtime/ backend/tests/test_runtime_engine.py
git commit -m "phase-7: integrate released runtime engine with contract-safe inference"
```

---

## 12. Phase 8 – Integration, Monitoring, Rollback, and Cleanup

### Objective
Finalize deployment behavior, observability, and rollback safety.

### Files
- `NEW`: `backend/monitoring/runtime_metrics.py`
- `NEW`: `backend/monitoring/drift_monitor.py`
- `NEW`: `backend/monitoring/golden_replay.py`
- `NEW`: `backend/tests/test_monitoring_and_rollback.py`
- `UPDATE`: frontend API consumers
- `UPDATE`: SSE payload handling
- `NEW`: `docs/ROLLBACK_POLICY.md`
- `NEW`: `docs/RELEASE_PROMOTION_CHECKLIST.md`

### Steps
1. Add runtime telemetry for latency, missing-modality rate, confidence, and drift.
2. Add golden replay sessions for regression checks.
3. Implement rollback logic to previous release manifests.
4. Update frontend integration if payload shape changed under the API contract.
5. Remove dead code only after replacement paths are stable.

### Completion Tests
- Monitoring detects injected failures.
- Rollback restores previous released model stack successfully.
- Frontend still renders correct stress outputs and explanations.

### Git Commit
```bash
git add backend/monitoring/ backend/tests/test_monitoring_and_rollback.py docs/ROLLBACK_POLICY.md docs/RELEASE_PROMOTION_CHECKLIST.md
git commit -m "phase-8: add monitoring rollback and release promotion controls"
```

---

## 13. Final Cleanup Phase (Optional but Recommended)

### Objective
Remove dead legacy code after all replacements are stable.

### Files
- old deprecated loaders
- old direct model-loading code
- old evaluation scripts
- obsolete helper notebooks/scripts

### Rule
Do this **only after** at least one full end-to-end run succeeds on released artifacts.

### Git Commit
```bash
git add -A
git commit -m "cleanup: remove deprecated legacy paths after release validation"
```

---

## 14. Required Agent Checklist Before Ending Any Phase

Before the agent ends a phase, it must confirm all of the following:

- Code compiles/imports correctly.
- Tests for that phase pass.
- No upstream contract was changed illegally.
- Documentation for that phase is updated.
- Release artifacts/manifests are created if applicable.
- Git diff contains only files relevant to that phase.
- Commit message follows the exact phase naming convention.

---

## 15. Recommended Commit Naming Convention

Use this exact structure:

```text
phase-X: short action-oriented summary
```

Examples:
- `phase-2: add certified dataset release and validation gates`
- `phase-5: add buffer-aware fusion release pipeline`
- `phase-7: integrate released runtime engine with contract-safe inference`

For bug fixes after a phase is released:

```text
phase-X-fix: short summary
```

Examples:
- `phase-4-fix: correct loso grouping leakage in voice expert release`

---

## 16. Agent Prompting Notes for Future Implementation Runs

When giving this work to an agent, the instruction should include:

- Implement only one phase at a time.
- Start with repository audit if not already done.
- Update existing files only if impact map says `UPDATE` or `REPLACE`.
- Preserve backward compatibility unless the phase explicitly authorizes a replacement.
- Run tests before commit.
- Create the git commit after phase completion.
- Stop and summarize if a contract conflict is discovered.

A strong control prompt is:

> “Implement only the next unfinished phase from `agent_phase_implementation_guide.md`. First inspect repo impact, then update/create only the files required for that phase, run the phase tests, and make exactly one git commit using the documented message. Do not touch future phases.”

---

## 17. Final Advice

This implementation plan is intentionally strict because your project goal is not just a demo. It is a **patent-oriented, production-credible, reviewable system**. The cleanest way to reach that goal is:

- one architectural phase at a time,
- one verification boundary at a time,
- one retrievable Git commit at a time.

That approach will make the codebase understandable for humans, controllable for agents, and defensible during patent drafting, academic review, and product development.

