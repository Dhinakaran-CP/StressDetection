# Multimodal Stress Detection: Phase Log

This document records the exact changes, files, and milestones accomplished during each phase of the architecture hardening process.

## Phase 1: Contracts and Versioning Foundation
**Status**: Completed (Git Commit: `phase-1: add contracts and artifact versioning foundation`)

- Created `contracts/schema_contract.yaml` to enforce the 7 required columns for dataset validity.
- Created `contracts/feature_contract.yaml` to enforce exact feature dimension (18 face, 12 voice).
- Created `contracts/api_contract.yaml` for SSE payload structure.
- Created `contracts/performance_contract.yaml` for max latency requirements.
- Implemented `backend/core/artifact_manifest.py` and `backend/core/version_registry.py` to stamp datasets and models with SHA-256 hashes and track active versions.

## Phase 2: Dataset Certification Layer
**Status**: Completed (Git Commit: `phase-2: add certified dataset release and validation gates`)

- Created `backend/core/dataset_certifier.py` to mathematically prove CSV compliance against the Schema Contract (checking for missing data, duplicates, and monotonicity).
- Created `backend/core/dataset_release.py` to read raw files, certify them, and dump hashed/versioned copies into the `dataset_certified/` vault.
- Wrote `backend/tests/test_dataset_certifier.py` to prove the certifier successfully rejects malformed inputs.
- Successfully certified 108,051 Face rows and 44,982 Voice rows with zero schema violations.

## Phase 3: Feature Runtime Lock Layer
**Status**: Completed (Git Commit: `phase-3: lock feature transformation path for offline-online parity`)

- Separated extraction code out of `backend/model.py` into `backend/core/extractors/face_extractor.py` and `voice_extractor.py`.
- Created `backend/core/feature_runtime_lock.py` to ensure live webcam data and offline training data undergo identical scaling and missing-value fills according to `feature_contract.yaml`.
- Wrote unit tests for the runtime lock to prove dimensions are strictly enforced.
- Refactored `backend/model.py` into a lightweight inference engine that delegates to the extractors and strictly passes raw arrays through the runtime lock before prediction.

## Phase 4: Expert Training Release Pipeline
**Status**: Completed (Git Commit: `phase-4: implement expert model release pipeline with grouped validation`)

- Archived deprecated, leakage-prone training scripts.
- Created `backend/training/train_face_expert_release.py` and `train_voice_expert_release.py`.
- Enforced Leave-One-Subject-Out (LOSO) cross-validation utilizing `GroupKFold` on the certified datasets.
- Created a deployment pipeline (`release_expert_model.py`) that packages models, scalers, and manifests directly into `backend/expert_models/` stamped with SHA-256 hashes via the Version Registry.

## Phase 5: Engine Integration & Explainability
**Status**: Completed

- Updated `backend/model.py`: Physio routed through `FeatureRuntimeLock.process_physio_features()`. Replaced naive `np.mean()` fusion with Phase 4 optimal weights (Face: 0.30, Voice: 0.40, Physio: 0.30) with auto-renormalization for absent modalities.
- Updated `backend/app.py`: Loaded `physio_expert_lightweight.pkl` at startup. SHAP explanations now use the app-level physio expert. Added human-readable feature labels (FACE/VOICE/PHYSIO_FEATURE_LABELS). `fuse_predictions()` (SSE stream) updated to Phase 4 weights.
- Fixed `FeatureRuntimeLock.process_physio_features()`: Added `mean_fill` strategy handler (was silently not filling NaNs because contract says `mean_fill`, not `zero_fill`).
- Verified: 34/34 unit tests passed, 3-way fusion weights confirmed as `{facial: 0.3, voice: 0.4, physiological: 0.3}`.

## Phase 6: Explainability Release Pipeline
**Status**: Completed

- Created `backend/explainability/` package with `__init__.py`.
- Created `backend/explainability/explainability_contract.py`: Defines BUNDLE_VERSION, TOP_K constants, human-readable feature label lists for all 3 modalities (18 face, 12 voice, 5 physio), feature group tags for UI color-coding, and bundle JSON schema.
- Created `backend/explainability/build_explainability_bundle.py`: Offline script that loads each expert model + certified CSV, computes SHAP TreeExplainer values, maps feature indices to human labels, and saves `backend/expert_models/explainability_bundle.json` with version manifest.
- Built `explainability_bundle.json` with real SHAP values for all 3 modalities. Top drivers: Face=Lip Compression, Voice=F0 Mean (Pitch Hz), Physio=EDA SCL Mean.
- Created `backend/explainability/explainability_engine.py`: Runtime loader. Loads the bundle at startup, exposes `explain_modality()` and `build_full_payload()` — no live SHAP per request.
- Updated `backend/app.py`: `build_explainability_payload()` now delegates to `ExplainabilityEngine` (bundle path) first, falling back to live SHAP only if bundle unavailable. Added `/api/explainability/status` endpoint. Updated `/api/health` to include engine status.
- Created `backend/tests/test_explainability_bundle.py`: 22 tests covering contract schema, bundle file integrity, engine unit tests, and graceful fallback.
