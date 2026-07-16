# Random Forest Master & Specialist Ensemble Comparison Report

This report compiles performance comparisons for the Tuned Single Random Forest and the Combined Specialist Ensemble across all window scales (2s, 5s, 10s).

| Scale | Tuned Single Forest Accuracy | Tuned Single Forest F1 | Combined Ensemble Accuracy | Combined Ensemble F1 | Selection Decision | Runtime |
| --- | --- | --- | --- | --- | --- | --- |
| 2sec | 0.7426 | 0.6353 | 0.7385 | 0.6316 | RETAINED Tuned Single Forest (ensemble gain too small or negative) | 126.6147 |
| 5sec | 0.7431 | 0.6339 | 0.7386 | 0.6309 | RETAINED Tuned Single Forest (ensemble gain too small or negative) | 47.2158 |
| 10sec | 0.7399 | 0.6244 | 0.7430 | 0.6325 | PROMOTED Combined Ensemble (improved > 0.5% F1-score) | 23.0753 |


*All plots and detailed reports have been categorized into the outputs/random_forest_master/ directory.*