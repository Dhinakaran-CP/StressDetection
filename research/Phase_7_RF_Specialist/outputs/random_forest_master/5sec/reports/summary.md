# Benchmarking Scorecard: Random Forest Master (5sec)

This scorecard compares the tuned baseline champion Random Forest with the specialist modality classifiers and the combined soft-voting ensemble.

### Performance Metrics Comparison
| Model / Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Tuned Single Forest** | 0.7431 | 0.7085 | 0.5736 | 0.6339 | 0.7145 | 5.08 |
| Face Specialist | 0.6066 | 0.4900 | 0.3611 | 0.4158 | 0.5895 | 3.81 |
| Voice Specialist | 0.7422 | 0.7073 | 0.5715 | 0.6322 | 0.6957 | 2.78 |
| Physio Specialist | 0.7306 | 0.6830 | 0.5696 | 0.6212 | 0.7086 | 1.06 |
| **Combined Ensemble** | 0.7386 | 0.6970 | 0.5762 | 0.6309 | 0.7032 | 7.64 |

### Final Selection Decision
* **Decision**: **RETAINED Tuned Single Forest (ensemble gain too small or negative)** (Ensemble F1 Gain = -0.0030)
