# Benchmarking Scorecard: Random Forest Master (2sec)

This scorecard compares the tuned baseline champion Random Forest with the specialist modality classifiers and the combined soft-voting ensemble.

### Performance Metrics Comparison
| Model / Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Tuned Single Forest** | 0.7426 | 0.7079 | 0.5762 | 0.6353 | 0.7236 | 13.32 |
| Face Specialist | 0.6028 | 0.4860 | 0.3622 | 0.4150 | 0.5731 | 9.79 |
| Voice Specialist | 0.7426 | 0.7082 | 0.5757 | 0.6351 | 0.6955 | 7.42 |
| Physio Specialist | 0.7365 | 0.7005 | 0.5638 | 0.6247 | 0.6996 | 2.82 |
| **Combined Ensemble** | 0.7385 | 0.6988 | 0.5762 | 0.6316 | 0.6931 | 20.03 |

### Final Selection Decision
* **Decision**: **RETAINED Tuned Single Forest (ensemble gain too small or negative)** (Ensemble F1 Gain = -0.0037)
