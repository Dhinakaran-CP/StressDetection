# Stress Detection Using ML — Chat Log

## Project Objective
Execute the full research pipeline (Phase 1 diagnostics → Phase 2 dataset audit → Phase 3 retraining → Phase 4-8 analysis) for publication-quality multimodal stress detection using CNNBaselineGRL and SSVB architectures.

## Data Pipeline
- **Enriched data** (`data/enriched_training_data/`) is the sole data source:
  - **StressID**: 53 subjects, 16,974 windows
  - **WESAD**: 15 subjects, 5,517 windows
  - **EmpathicSchool**: 66,622 windows (physio-only — face/voice groups are 100% zero)
- Research plan mandates: no new architectures beyond approved CNNBaselineGRL and SSVB; no dataset modifications without proven data issue.
- GPU confirmed: `venv\Scripts\python.exe` on CUDA.

---

## Phase 1 — Subject Diagnostics

### Phase 1.1 — m8g5 (StressID)
- **Initial suspicion**: Label inversion (AUC=0.0878 in prior run).
- **Final conclusion**: **NO label inversion.** LR achieves AUC=0.928. Old 0.0878 was a training artifact.
- Report saved to `phase1_diagnostics/m8g5/diagnostic_report.json`.

### Phase 1.2 — 71i5 (StressID)
- **Initial suspicion**: Data corruption (near-chance AUC).
- **Final conclusion**: **High physiological reactor** — 74% stress ratio vs 42% dataset average. Near-chance AUC expected from class imbalance. `physio_cardio` shows weak signal (AUC=0.614). No corruption.
- Report saved to `phase1_diagnostics/71i5/diagnostic_report.json`.

### Phase 1.3 — wesad_s2 (WESAD)
- **Initial suspicion**: Data corruption.
- **Final conclusion**: **Threshold calibration issue.** AUC=0.907 with LR. F1 collapses at default 0.5 threshold (0.000) but recovers to 0.832 at optimal threshold 0.006. No corruption.
- Report saved to `phase1_diagnostics/wesad_s2/diagnostic_report.json`.

### Exclusion Strategy — SUPERSEDED
- Originally: three subjects flagged for exclusion.
- **Updated**: ALL three subjects are clean — **no exclusions needed.**
  - m8g5: LR AUC=0.928 (not label inversion)
  - 71i5: High reactor, 74% stress ratio (not corruption)
  - wesad_s2: Threshold calibration only, AUC=0.907

---

## Phase 2 — Dataset Audit
- All datasets clean: zero NaN/Inf, no duplicates, labels consistent within subject-task blocks, no subject leakage.
- EmpathicSchool confirmed physio-only (face/voice groups 100% zero).
- Cross-dataset feature dimensions match.
- Report saved to `phase2_dataset_audit/dataset_audit_report.json`.

---

## Phase 3 — CNNBaselineGRL Benchmark

### Original Benchmark Run
- **Script**: `scripts/run_all_models_benchmark.py`
- **Results saved to**: `benchmark_results/cnn_baseline_grl/benchmark.json`
- **Metricts**:
  - StressID: AUC=0.7113, F1=0.6910
  - WESAD: AUC=0.8723, F1=0.7283
  - Combined: AUC=0.6704, F1=0.2938
  - m8g5 fold: AUC=0.9458 (confirms clean)
- **Limitation**: Saves only aggregated metrics — no per-window predictions, no model weights. Bootstrap CIs and Phase 4 transfer eval require enhanced runner.

### Enhanced Benchmark Script
- **Script**: `scripts/phase3_train_and_analyze.py`
- Adds: bootstrap CI computation, per-window prediction saving, ROC/PR curves, confusion matrices, calibration curves, subject-level metrics.
- **Bugs fixed**:
  1. `AttributeError: 'ModelEntry' object has no attribute 'build_fn'` — fixed by importing `build_model` from benchmark.
  2. `ImportError: cannot import name 'calibration_curve' from 'sklearn.metrics'` — moved import to `sklearn.calibration`.
  3. `UnusedImport: scipy.stats.bootstrap` — removed.
  4. `TypeError: forward_model() missing 1 required positional argument: 'device'` — was missing `model_entry` arg; also line break caused `replaceAll` to miss two call sites.
- **Status**: Running (awaiting completion).

---

## Work State Summary

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1.1 (m8g5) | ✅ Complete | Clean — LR AUC=0.928 |
| Phase 1.2 (71i5) | ✅ Complete | High reactor — 74% stress |
| Phase 1.3 (wesad_s2) | ✅ Complete | Threshold calibration — AUC=0.907 |
| Phase 2 (Dataset Audit) | ✅ Complete | All datasets clean |
| Phase 3 (CNNBaselineGRL) | 🟡 Running | Enhanced script executing |
| Phase 4 (Transfer Eval) | ⏳ Pending | Blocked on Phase 3 results |
| Phase 5 (SSVB Training) | ⏳ Pending | |
| Phase 6 (Statistical Comp) | ⏳ Pending | |
| Phase 7 (Ablation) | ⏳ Pending | |
| Phase 8 (Figures/Tables) | ⏳ Pending | |

## Relevant Files
- `scripts/phase3_train_and_analyze.py` — enhanced benchmark with predictions/bootstrap
- `scripts/run_all_models_benchmark.py` — original benchmark (aggregated metrics only)
- `phase3_production/train.py` — production training loop (SSVBDataset, LOSO, CNNBaselineGRL)
- `scripts/phase1_step1_m8g5_diagnostic.py` — m8g5 diagnostic analysis
- `scripts/phase1_step3_wesad_s2_diagnostic.py` — wesad_s2 threshold/PR analysis
- `scripts/phase2_dataset_audit.py` — full dataset quality audit
- `phase1_diagnostics/` — per-subject diagnostic reports
- `phase2_dataset_audit/dataset_audit_report.json` — dataset quality audit report
- `benchmark_results/cnn_baseline_grl/benchmark.json` — aggregated Phase 3 results
- `data/enriched_training_data/` — enriched parquet + npz per dataset