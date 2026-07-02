# Agent Detailed Phase Playbook for the Multimodal Stress Detection Platform

This document explains, in a practical and realistic way, what an implementation agent should do in each phase of the project. It is written to avoid vague instructions, fake completeness, or unrealistic assumptions. The agent is expected to work against an existing repository that already contains a frontend, backend runtime, training scripts, and evaluation scripts.

The purpose of this playbook is to make every phase understandable, executable, and reviewable. Each phase describes:

- why the phase exists,
- what the agent must inspect first,
- what the agent should change,
- what the agent must not change,
- what outputs must exist before the phase is considered complete,
- what risks usually appear in that phase,
- and what evidence must be collected before committing code.

This phase-oriented execution style follows production MLOps guidance that emphasizes controlled iteration, data validation, automation, testing, and governed deployment rather than ad hoc experimentation.[cite:1][cite:3][cite:5]

---

## System assumptions

The playbook assumes the project is trying to become a production-grade, subject-aware, leakage-safe multimodal stress detection system with real-time face and voice inference, plus optional physiological fusion when available. In the literature, subject-independent validation and protocol-aware multimodal evaluation are necessary because subject-dependent setups usually overestimate performance compared with LOSO or other unseen-subject protocols.[cite:2][cite:8] It also assumes the project wants reproducibility, artifact traceability, and online-offline parity, which are all standard MLOps requirements for reliable deployment.[cite:1][cite:3][cite:5]

---

## Phase 0 — Repository audit and architecture freeze

### Why this phase exists

An agent should not start coding from assumptions. The current repository may already contain hidden coupling between extraction, training, inference, evaluation, and frontend payloads. If the agent skips the audit phase, later changes can break the runtime while appearing correct locally. Production ML guidance recommends understanding the current pipeline, dependencies, and operational boundaries before automation or deployment changes are introduced.[cite:1][cite:5]

### What the agent should do

1. Read the repository tree and identify all backend, training, evaluation, frontend, model artifact, and documentation files.
2. Open the key entrypoints first: `app.py`, `realtime_core.py`, `model.py`, `voice_worker.py`, `score_buffer.py`, `colab_training.py`, `strict_fused_evaluation.py`, and `evaluate_fused_engine_bootstrapped.py`.
3. Record the current control flow:
   - how a webcam frame is posted,
   - where face features are extracted,
   - where audio is processed,
   - how model files are loaded,
   - how predictions are buffered,
   - how SSE events are emitted,
   - how evaluation scripts currently split data.
4. Create an impact map labeling each important file as `REUSE`, `UPDATE`, `REPLACE`, or `NEW`.
5. Freeze a phase order so future agents do not implement later phases early.

### What the agent should not do

- It should not refactor code in this phase.
- It should not rename files for style reasons.
- It should not fix bugs unless they block reading the repository.

### Outputs required

- `docs/IMPLEMENTATION_BASELINE.md`
- `docs/REPO_IMPACT_MAP.md`
- `docs/ARCHITECTURE_PHASE_PLAN.md`

### Risks in this phase

The common mistake is underestimating hidden dependencies. For example, `model.py` may be used both in offline training utilities and live inference; changing it later without documenting that coupling can cause train-serve skew. Another risk is missing frontend assumptions about payload keys or explanation labels.

### Evidence before commit

The audit is complete only when a reviewer can read the baseline documents and understand the current system without reopening the whole repo.

---

## Phase 1 — Contracts and versioning foundation

### Why this phase exists

Most ML systems become unstable because important interfaces remain implicit. Data columns, feature order, API payload shape, and latency targets are often “known by code” instead of written down. MLOps practice recommends explicit contracts and versioned artifacts so later work can evolve safely.[cite:1][cite:3][cite:5]

### What the agent should do

1. Define a data schema contract for extracted rows. For this project, the contract should include at least:
   - `subject_id`
   - `session_id`
   - `task_name`
   - `window_id`
   - `window_start_ms`
   - `window_end_ms`
   - `label`
   - modality-specific features
   - optional quality/confidence fields
2. Define a feature contract for each expert:
   - exact feature names,
   - order,
   - allowed missingness handling,
   - scaling rules,
   - baseline normalization rules,
   - quality filters.
3. Define an API contract for runtime output:
   - predicted stress score,
   - class label,
   - confidence,
   - active modalities,
   - explanation payload,
   - timestamp.
4. Define a performance contract:
   - max latency per inference,
   - RAM ceiling,
   - CPU ceiling,
   - degraded mode behavior.
5. Add version manifest utilities so every release artifact carries schema version, feature hash, model hash, and evaluation summary.

### What the agent should not do

- It should not yet rewrite extraction or training logic.
- It should not introduce new feature ideas here unless the contract requires placeholders for future fields.

### Outputs required

- `contracts/schema_contract.yaml`
- `contracts/feature_contract.yaml`
- `contracts/api_contract.yaml`
- `contracts/performance_contract.yaml`
- version registry and artifact manifest code

### Risks in this phase

A common failure is writing contracts that are too vague to enforce. For example, saying “save timestamps” is weak; saying “save `window_start_ms` and `window_end_ms` relative to source media start” is enforceable. Another failure is forgetting that runtime and training need the same feature semantics.

### Evidence before commit

Contracts must be machine-readable and specific enough that later scripts can validate against them automatically.

---

## Phase 2 — Dataset certification layer

### Why this phase exists

Your current problem came from extraction outputs that lacked enough metadata to support LOSO and temporal evaluation. That is exactly why dataset certification must be a dedicated phase. Realistic ML systems cannot rely on “whatever CSV exists”; they need acceptance checks before training starts.[cite:1][cite:4][cite:5] Subject-independent stress literature also shows that rigorous evaluation depends on preserving subject identity and segmentation structure.[cite:2][cite:6][cite:8]

### What the agent should do

1. Build a dataset certifier that validates:
   - required columns exist,
   - subject IDs are present and parseable,
   - task names exist,
   - window chronology is monotonic within subject-session-task,
   - labels are valid,
   - no duplicate keys exist for `(subject_id, session_id, task_name, window_id, modality)`.
2. Validate segmentation consistency:
   - expected window length,
   - expected stride,
   - no impossible gaps unless marked missing,
   - no overlapping windows unless overlap is intentionally defined.
3. Validate multimodal alignment:
   - face and voice windows align to the same time base,
   - optional physio windows can be mapped or flagged when unavailable.
4. Produce a certification report:
   - row counts per subject,
   - class balance per task,
   - missingness per modality,
   - dropped rows and reasons,
   - checksum and release ID.
5. Update all training and evaluation scripts so they only accept certified datasets.

### What the agent should not do

- It should not train models in this phase.
- It should not manually fix data by editing CSVs silently; all cleaning rules must be explicit in code.

### Outputs required

- dataset certifier code
- release manifest
- validation report
- tests for invalid schema, duplication, and alignment failures

### Risks in this phase

One realistic risk is discovering that current extracted files are fundamentally unrecoverable for LOSO because `subject_id` is absent. If that happens, the certifier should fail loudly and direct the workflow back to re-extraction from raw files. That is not a phase failure; it is the correct outcome. Another risk is keeping too many “optional” fields optional, which weakens downstream rigor.

### Evidence before commit

A sample certification run must produce a report showing either a valid certified dataset or a precise failure reason.

---

## Phase 3 — Feature runtime lock layer

### Why this phase exists

In ML systems, one of the most damaging issues is train-serve skew: the model is trained on one feature pipeline and served with another. Online-offline parity is a central production ML requirement because even a small mismatch in scaling, feature order, or missing-value treatment can invalidate deployment.[cite:1][cite:3][cite:7]

### What the agent should do

1. Identify where face and voice features are currently created in offline code and where they are created in runtime code.
2. Move the common transformation rules into a shared module, including:
   - feature ordering,
   - scaling,
   - clipping,
   - null handling,
   - baseline normalization,
   - quality gating.
3. Split overloaded modules if necessary. If `model.py` mixes extraction, feature engineering, and inference, separate them into focused modules.
4. Add assertions that compare produced feature names and dimensions against the contract.
5. Add deterministic parity tests using fixed input snapshots.

### What the agent should not do

- It should not optimize performance prematurely.
- It should not change the meaning of features during this phase.

### Outputs required

- `feature_runtime_lock.py`
- parity tests
- updated imports so both training and runtime call the same transformation path

### Risks in this phase

The biggest risk is that the live code has convenience shortcuts, such as skipping unstable features or using fallback defaults, while the training code uses a richer path. Those shortcuts must be surfaced and reconciled rather than hidden.

### Evidence before commit

A fixed input sample should yield equivalent transformed vectors in both offline and live code paths.

---

## Phase 4 — Expert training release pipeline

### Why this phase exists

Once data and feature contracts are stable, expert training can be made release-based. This means training is no longer a loose notebook-style activity but a governed process that produces artifacts with metrics, manifests, and promotion decisions. For stress detection, subject-aware evaluation is essential because subject-dependent splits can inflate performance compared with subject-independent protocols.[cite:2][cite:6][cite:8]

### What the agent should do

1. Create independent release scripts for face, voice, and physio experts.
2. Make each script consume only certified datasets.
3. Implement grouped evaluation:
   - LOSO where subject IDs exist,
   - grouped cross-validation if full LOSO is too expensive for internal experiments,
   - never random row-level splits across synchronized windows.
4. Save the full release bundle:
   - trained model,
   - calibration object,
   - feature hash,
   - training config,
   - metrics summary,
   - confusion matrix,
   - manifest.
5. Add promotion gates:
   - minimum LOSO metric,
   - maximum calibration error,
   - maximum latency,
   - no feature-contract mismatch.

### What the agent should not do

- It should not hand-pick only the best split.
- It should not compare models trained on uncertified data.
- It should not publish an expert without calibration and manifest files.

### Outputs required

- release-based expert training scripts
- grouped evaluation reports
- saved release folders for each expert

### Risks in this phase

One common issue is that LOSO results may be much lower than previous random-split results. That is not a reason to weaken evaluation; it is usually a sign that the earlier numbers were optimistic. Another realistic issue is that face and voice may need different thresholding or calibration behavior.

### Evidence before commit

The release folder must be reproducible from command line and contain enough metadata for another agent to load it without reading training code.

---

## Phase 5 — Fusion and temporal intelligence pipeline

### Why this phase exists

The real product is not just separate experts; it is a real-time fused engine with temporal smoothing. Multimodal stress research shows that modalities often have different temporal behavior and reliability, so simple averaging is rarely enough. The fusion phase must therefore model confidence, missingness, and temporal dynamics in a controlled way.[cite:2][cite:6][cite:8]

### What the agent should do

1. Load released outputs from the expert models rather than retraining inside the fusion script.
2. Build an offline temporal simulation that mirrors runtime buffering:
   - face update frequency,
   - voice window frequency,
   - score decay behavior,
   - smoothing horizon.
3. Evaluate fusion strategies realistically:
   - fixed weighted fusion,
   - confidence-aware weighted fusion,
   - simple meta-learner if justified by data size.
4. Implement missing-modality logic explicitly:
   - face-only fallback,
   - voice-only fallback,
   - confidence-based suppression,
   - stale-window expiration.
5. Save a fusion release bundle with configuration and metrics.

### What the agent should not do

- It should not assume all modalities are always available.
- It should not tune weights on the final test set.
- It should not use temporal information offline that the runtime will not have.

### Outputs required

- fusion training/evaluation code
- temporal buffer engine
- released fusion artifact and manifest

### Risks in this phase

A realistic risk is that a modality contributes noise in some tasks or subjects. The architecture should allow the fusion layer to down-weight or ignore weak inputs rather than force contribution. Another risk is train-runtime mismatch in the buffering horizon.

### Evidence before commit

Offline replay should show that the fusion engine behaves plausibly under modality loss, latency jitter, and confidence changes.

---

## Phase 6 — Explainability release pipeline

### Why this phase exists

Explainability in this project is part of the product promise, not an optional notebook feature. That means explanation assets must be versioned and tied to released models. Literature on interpretable multimodal stress systems also emphasizes that explainability should support auditability and trust, not just visualization.[cite:8][cite:9]

### What the agent should do

1. Build explanation bundles from released expert models.
2. Map raw feature names into human-readable biomarker labels.
3. Group features by physiological meaning or facial/vocal semantics.
4. Save enough metadata so runtime can display top contributors quickly.
5. Verify that explanations remain valid when models are retrained or feature order changes.

### What the agent should not do

- It should not compute explanations dynamically in a way that makes runtime unstable without testing.
- It should not display raw feature IDs to end users.

### Outputs required

- explanation bundle builder
- feature label maps
- runtime-loadable explanation artifacts

### Risks in this phase

The common problem is explanation drift: the model changes, but the displayed labels do not. Another issue is using feature names that are understandable to developers but not to users.

### Evidence before commit

The runtime should be able to load a release bundle and return top explanations with correct human-readable labels.

---

## Phase 7 — Runtime engine integration

### Why this phase exists

This phase connects the governed ML pipeline back into the live application. It is where architecture becomes product behavior. Production ML systems need a clean separation between trained artifacts and serving code, with tested loading, inference, and degradation paths.[cite:1][cite:3][cite:5]

### What the agent should do

1. Create a runtime engine wrapper that loads released experts, released fusion configuration, and released explanation bundles through manifests.
2. Update `app.py` to initialize the runtime engine rather than directly loading pickles ad hoc.
3. Update `realtime_core.py` so sessions, buffering, and SSE emissions call the runtime engine consistently.
4. Update `voice_worker.py` and face handling code to use the locked feature pipeline.
5. Add degraded mode handling:
   - no face detected,
   - low audio quality,
   - temporary modality dropout,
   - stale session state.
6. Add replay-mode tests using frozen certified inputs.

### What the agent should not do

- It should not bypass manifests for quick loading shortcuts.
- It should not silently substitute missing models.

### Outputs required

- runtime engine module
- integrated backend paths
- replay tests
- stable SSE payload behavior

### Risks in this phase

The realistic danger here is hidden latency or concurrency problems. A design that looks correct in isolated tests may still behave poorly under live streaming. Another issue is frontend breakage if output payloads changed but consumer code was not updated.

### Evidence before commit

A controlled local run must show that the frontend receives stable predictions and explanation payloads from the new runtime stack.

---

## Phase 8 — Monitoring, rollback, and release control

### Why this phase exists

A production-grade system does not end at inference. It needs observation and safe rollback. MLOps frameworks consistently treat monitoring, deployment control, and traceability as core parts of the lifecycle.[cite:1][cite:3][cite:5]

### What the agent should do

1. Add runtime metrics collection for:
   - inference latency,
   - face detection failure rate,
   - voice quality rejection rate,
   - missing-modality rate,
   - confidence distribution,
   - drift indicators.
2. Add golden replay tests that re-run known sessions against the current release.
3. Implement rollback logic to a previous release manifest.
4. Add release promotion checklists for moving from internal testing to normal use.
5. Update frontend handling if runtime payload metadata is enriched.

### What the agent should not do

- It should not treat monitoring as a logging afterthought.
- It should not delete old release manifests before rollback is tested.

### Outputs required

- runtime monitoring module
- drift checks
- golden replay suite
- rollback policy document
- promotion checklist

### Risks in this phase

A frequent mistake is collecting too many logs but not the right operational indicators. For this project, missing-modality rate and latency stability matter more than generic debug prints. Another risk is building rollback in theory but never proving it on an actual previous release.

### Evidence before commit

The team should be able to intentionally load a previous release and confirm the system returns to the earlier state cleanly.

---

## Phase 9 — Controlled cleanup and deprecation

### Why this phase exists

Legacy cleanup should happen only after the new stack works end to end. Removing deprecated code too early creates panic-driven restores and breaks traceability. Good ML operations practice favors controlled deprecation after replacement behavior is proven.[cite:1][cite:5]

### What the agent should do

1. Identify now-unused loaders, scripts, helpers, and deprecated runtime paths.
2. Remove only those that are clearly replaced by released equivalents.
3. Keep migration notes in docs for future contributors.
4. Ensure tests still pass after cleanup.

### What the agent should not do

- It should not delete historical artifacts needed for reproducibility.
- It should not remove code just because it looks old.

### Outputs required

- cleaned repository
- deprecation notes
- final validation run

### Risks in this phase

The biggest risk is deleting a path still used by a hidden notebook, maintenance script, or frontend fallback. This is why cleanup must happen last.

### Evidence before commit

A fresh clone and setup should still reproduce the main training and runtime flows without relying on deleted paths.

---

## Cross-phase rules for the agent

The agent should follow these rules in every phase:

- Implement only one phase at a time.[cite:5]
- Do not edit future-phase files unless the current phase genuinely requires a tiny compatibility stub.[cite:1][cite:5]
- Run tests before every commit.[cite:1]
- Keep phase commits retrievable and narrowly scoped.[cite:1][cite:3]
- Stop and report if a contract conflict is found instead of improvising around it.[cite:1][cite:5]
- Prefer explicit adapters over silent breaking changes.[cite:3]

---

## What makes this playbook realistic

This playbook is realistic because it assumes failure can happen at any phase and defines how the agent should react. For example, if subject IDs are absent in extracted CSVs, the correct action is to stop certification and require re-extraction from raw data, not to invent subject grouping heuristics. If LOSO results are lower than old random-split numbers, the correct action is to keep the stricter protocol, not to relax evaluation. If runtime payloads change, the agent must update frontend consumers deliberately rather than assume compatibility. These are the practical patterns seen in real ML system hardening rather than idealized diagrams.[cite:1][cite:2][cite:5][cite:8]

---

## Recommended usage

This document should be used together with the implementation guide. The implementation guide tells the agent **what to implement and commit phase by phase**, while this playbook tells the agent **what it is actually expected to do inside each phase**, what mistakes to avoid, and what evidence must exist before declaring the phase complete.[cite:1][cite:3][cite:5]

