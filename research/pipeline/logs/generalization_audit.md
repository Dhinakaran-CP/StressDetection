# Generalization Gates Audit Report

**Leader Model Evaluated:** `logistic_regression`

| Gate | Audit Test Name | Realized Metric | Status |
| :--- | :--- | :--- | :--- |
| G2 | Stability Fold Acc Std | 0.1247 (threshold <= 0.08) | ❌ FAIL |
| G3 | Biomarkers in Top Ranks | 9 verified (threshold >= 1) | ✅ PASS |
| G4 | Identity Suppression Leakage Gap | 0.2570 (threshold <= 0.10) | ❌ FAIL |
| G5 | Domain Classifier Accuracy | 0.9998 (threshold <= 0.75) | ❌ FAIL |
| G6 | Combined 95-Subject LOSO Acc | 0.6907 (threshold >= 0.74) | ❌ FAIL |
| D1 | Face Stressed F1-Score | 0.5875 (threshold >= 0.40) | ✅ PASS |