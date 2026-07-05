# StressDetectionUsingML - Automated Audit & Implementation Plan

## Document Control

- Project: StressDetectionUsingML
- Purpose: Define the implementation contract for the full 8-phase research pipeline, strict subject-independent validation, augmentation comparison, and best-expert fusion development.
- Status: Draft for automated audit
- Scope: Repository-wide implementation, experiment logging, benchmark updates, and branch-based phase commits
- Validation Standard: Strict subject-independent evaluation only
- Primary Novelty Check: Generalization to unseen subjects under leakage-free validation

## Non-Negotiable Rules

1. Do not use random train/test splits for final reporting.
2. Do not use leaked normalization, scaling, calibration, augmentation, or feature extraction.
3. Do not evaluate on held-out subjects during tuning.
4. Do not mix experimental outputs with final benchmark outputs.
5. Do not overwrite historical phase logs.
6. Do not commit a phase as complete until its required artifacts and logs are present.
7. Do not claim novelty from training accuracy or non-subject-independent validation.
8. Do not alter labels, subject IDs, or modality alignment during augmentation.

## Repository Workflow

### Branching Policy

- Create one dedicated implementation branch for the full research line.
- Use a single branch for the 8 phases unless a phase requires a corrective hotfix branch.
- Keep the main branch untouched until the full pipeline is validated.

### Commit Policy

- Commit after each phase is completed.
- Each commit must include:
  - code changes,
  - experiment outputs,
  - updated markdown logs,
  - and a concise phase message.
- Each commit message must mention the phase name and the main change.

### Required Commit Pattern

- `phase1: baseline audit and repo structure`
- `phase2: subject-safe classical pipeline`
- `phase3: calibration and temporal aggregation`
- `phase4: deep modality encoders`
- `phase5: best-expert selection`
- `phase6: fusion engine`
- `phase7: augmentation comparison`
- `phase8: strict validation and final benchmark`

## Project Objective

Build a leakage-free multimodal stress detection system that proves novelty through strict subject-independent performance. The project must compare classical and deep models, select the best expert per modality, test static and dynamic fusion, and evaluate augmentation systematically.

The final claim must be based on unseen-subject generalization, not on easy validation splits.

## 8-Phase Implementation Plan

### Phase 1: Repository Audit and Baseline Mapping

#### Goals
- Map the full repository structure.
- Identify the training, inference, preprocessing, calibration, fusion, and evaluation paths.
- Detect redundant, unused, missing, or conflicting files.
- Document all current baselines.

#### Required Actions
- Scan the entire repository tree.
- Identify which files are used in training.
- Identify which files are used in inference.
- Identify any stale scripts or duplicate experiment folders.
- Identify all model artifacts and scalers.
- Document the current baseline configurations.

#### Required Output
- Phase 1 log entry.
- Repository structure summary.
- Baseline inventory.
- Risk list.

#### Completion Criteria
- Repository layout is understood.
- All model/data paths are documented.
- No missing core file dependencies remain undocumented.

### Phase 2: Subject-Safe Classical Pipeline

#### Goals
- Maintain the calibrated classical benchmark.
- Enforce subject-safe subject-independent preprocessing.
- Preserve the known best classical baseline.

#### Required Actions
- Validate subject-wise grouping.
- Validate temporal aggregation behavior.
- Validate subject-adaptive scaling.
- Validate calibration behavior.
- Confirm face, voice, and physio classical baselines.

#### Required Output
- Fold-wise metrics.
- Mean and variance metrics.
- Calibration summary.
- Baseline comparison table.

#### Completion Criteria
- Classical pipeline reproduces expected subject-independent behavior.
- No leakage detected.

### Phase 3: Temporal and Calibration Checks

#### Goals
- Confirm the effect of temporal aggregation.
- Confirm the effect of probability calibration.
- Compare naive average against stacking.

#### Required Actions
- Run controlled ablations for:
  - no temporal aggregation,
  - temporal aggregation enabled,
  - calibrated base models,
  - uncalibrated base models,
  - naive average,
  - meta stacking.
- Compare fold variance and calibration quality.

#### Required Output
- Ablation report.
- Phase 3 log entry.
- Updated benchmark table.

#### Completion Criteria
- Best calibrated classical configuration is confirmed.
- Fusion choice is justified by validation results.

### Phase 4: Deep Modality Encoders

#### Goals
- Train compact deep encoders for each modality.
- Evaluate deep unimodal performance under strict validation.
- Compare against classical unimodal baselines.

#### Required Actions
- Implement face encoder.
- Implement voice encoder.
- Implement physio encoder.
- Use low-capacity models first.
- Keep regularization strong.
- Keep subject-independent splits intact.

#### Required Output
- Unimodal deep benchmark table.
- Fold-wise deep model logs.
- Comparison against classical unimodal models.

#### Completion Criteria
- Best deep model per modality is identified.
- Deep results are reproducible and leakage-free.

### Phase 5: Best-Expert Selection

#### Goals
- Select the single best expert for each modality.
- Use only the best face model, best voice model, and best physio model.

#### Required Actions
- Compare all candidate face models.
- Compare all candidate voice models.
- Compare all candidate physio models.
- Select one best expert per modality by strict subject-independent performance.

#### Required Selection Criteria
- Mean LOSO performance.
- Stability across folds.
- Calibration quality.
- Runtime cost.
- Robustness under weak folds.

#### Required Output
- Best-expert selection table.
- Justification for each modality choice.
- Phase 5 log entry.

#### Completion Criteria
- Exactly one best expert per modality is chosen.
- Weak or redundant model variants are excluded from fusion experiments.

### Phase 6: Fusion Engine

#### Goals
- Fuse the selected best experts.
- Test static and dynamic fusion.
- Keep the system low-latency.

#### Required Actions
- Test static weighted fusion.
- Test naive average.
- Test learned gating or router fusion.
- Test pairwise Face + Physio fusion.
- Test full 3-way fusion if justified.

#### Required Fusion Rules
- Start with Face + Physio as the primary candidate.
- Treat Voice as optional if it degrades performance.
- Use dynamic gating only if it improves validation stability and accuracy.
- Keep the gate lightweight.

#### Required Output
- Fusion benchmark table.
- Runtime comparison.
- Variance comparison.
- Phase 6 log entry.

#### Completion Criteria
- Final fusion design is selected based on subject-independent validation.
- Low-latency constraint is preserved.

### Phase 7: Augmentation Comparison

#### Goals
- Compare augmentation methods systematically.
- Choose the most reliable method per modality or fused pipeline.
- Avoid random augmentation choices.

#### Required Actions
- Compare augmentation families separately.
- Evaluate only on training folds.
- Keep test folds untouched.
- Measure gain, variance, and calibration impact.

#### Augmentation Families to Compare
- Physiological time-series augmentation.
- Light face augmentation.
- Light voice augmentation.
- Multimodal modality dropout.
- Optional GAN-based synthetic physio augmentation.

#### Required Output
- Augmentation comparison table.
- Per-method performance delta.
- Per-method variance delta.
- Phase 7 log entry.

#### Completion Criteria
- Best augmentation method is selected or augmentation is rejected if not beneficial.
- No augmentation is kept without strict validation evidence.

### Phase 8: Strict Validation and Final Benchmark

#### Goals
- Use strict subject-independent validation as the project novelty check.
- Produce the final audited benchmark.
- Finalize the research claim.

#### Required Actions
- Re-run the final selected system using strict subject-independent validation.
- Confirm no leakage in preprocessing, calibration, or augmentation.
- Confirm final metrics.
- Freeze final benchmark outputs.

#### Required Output
- Final benchmark table.
- Final phase log entry.
- Final novelty statement.
- Final decision on production baseline.

#### Completion Criteria
- Final benchmark is reproducible.
- Final novelty claim is supported by strict validation.
- No open leakage or ambiguity remains.

## Augmentation Policy

### Allowed Augmentations

#### Physiological Signals
- Jittering
- Scaling
- Time warping
- Magnitude warping
- Window masking
- Time masking
- Random cropping

#### Face Data
- Small rotation
- Small crop/shift
- Brightness jitter
- Contrast jitter
- Mild blur
- Frame dropout
- Mild temporal subsampling

#### Voice Data
- Additive noise
- Small pitch shift
- Small speed perturbation
- Masking
- Short-segment dropout

#### Multimodal
- Modality dropout
- Synchronized window perturbation
- Aligned temporal cropping
- Quality-aware masking

### Restricted Augmentations

- No augmentation on test subjects.
- No augmentation that breaks label meaning.
- No independent unsynchronized modality corruption.
- No aggressive synthetic generation unless it passes strict ablation.
- No random augmentation without analysis.

### Augmentation Selection Rule

Choose the simplest augmentation method that improves:
- mean LOSO accuracy,
- macro F1,
- balanced accuracy,
- calibration,
- and fold stability.

Reject augmentation if it improves training metrics only.

## Validation Policy

### Mandatory Validation Standard

- Use strict subject-independent validation for all final claims.
- Use grouped splits by subject.
- Use nested tuning where applicable.
- Maintain leakage checks for:
  - normalization,
  - scaling,
  - calibration,
  - augmentation,
  - temporal aggregation,
  - and feature engineering.

### Forbidden Validation Patterns

- Random split for final benchmark.
- Subject leakage across folds.
- Tuning on held-out subjects.
- Reporting training accuracy as benchmark evidence.

### Required Metrics

- Accuracy
- Macro F1
- Balanced accuracy
- Calibration error
- Per-subject variance
- Runtime cost
- Confusion matrix

## Logging Policy

### Phase Log Requirements

The markdown file must keep a running log with one entry per phase.

Each entry must contain:
- Phase number
- Date
- Commit hash if available
- Main changes
- Metrics
- Risks
- Decision status

### Log Update Rule

- Append new entries only.
- Do not remove earlier phase logs.
- Do not rewrite history.
- Keep the log machine-readable and human-readable.

## Risk and Audit Checks

### High-Risk Issues to Check
- Data leakage
- Inconsistent preprocessing between training and inference
- Model/scaler mismatch
- Hidden use of random splits
- Unused model files
- Hardcoded paths
- Dead code
- Incomplete experiment logging
- Insecure deployment settings

### Required Audit Deliverables
- File inventory
- Dependency map
- Model artifact map
- Leakage checklist
- Final status report

## Best-Expert Fusion Decision Rule

1. Identify the best model per modality.
2. Freeze those best experts.
3. Test static fusion first.
4. Test dynamic fusion only if it improves validation.
5. Keep Voice only if it helps.
6. Prefer the most stable low-latency system over a marginally higher but unstable system.

## Final Novelty Statement Rule

The final novelty statement must be based on:
- subject-independent generalization,
- strict validation,
- stable fusion,
- and reproducible augmentation results.

It must not be based on:
- training accuracy,
- random split results,
- or non-audited experiments.

## Required Phase Log Template

```md
## Phase N Log

- Date:
- Branch:
- Commit:
- Objective:
- Changes:
- Metrics:
- Validation:
- Risks:
- Decision:
```

## Required Final Acceptance Criteria

The project is complete only when:
- all 8 phases are implemented,
- all phase logs are updated,
- all benchmarks are reproducible,
- all validation rules are satisfied,
- and the final system is selected by strict subject-independent evidence.

## Agent Instruction Block

Use this instruction for implementation:

> Implement the full 8-phase StressDetectionUsingML plan with strict subject-independent validation as the core novelty check. Create a dedicated branch, commit after every phase, and update the markdown phase log after every commit. Audit the repository for leakage, stale code, and missing links between preprocessing, training, calibration, fusion, and inference. Select the best expert per modality, test static and dynamic fusion, and compare augmentation methods systematically rather than randomly. Keep only the configuration that improves unseen-subject generalization, remains stable across folds, and satisfies the low-latency requirement.

## Final Note

The repository must be treated as an audited research system, not just a demo. Every final result must be traceable to a phase, a commit, and a strict subject-independent validation run.