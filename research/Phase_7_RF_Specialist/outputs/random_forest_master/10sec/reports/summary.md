# Benchmarking Scorecard: Random Forest Master (10sec)

This scorecard compares the tuned baseline champion Random Forest with the specialist modality classifiers and the combined soft-voting ensemble.

### Performance Metrics Comparison
| Model / Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Tuned Single Forest** | 0.7399 | 0.7040 | 0.5610 | 0.6244 | 0.7209 | 2.75 |
| Face Specialist | 0.6040 | 0.4821 | 0.3714 | 0.4196 | 0.5848 | 2.11 |
| Voice Specialist | 0.7415 | 0.7059 | 0.5645 | 0.6273 | 0.6941 | 1.70 |
| Physio Specialist | 0.7300 | 0.6764 | 0.5740 | 0.6210 | 0.7103 | 0.89 |
| **Combined Ensemble** | 0.7430 | 0.7048 | 0.5736 | 0.6325 | 0.6967 | 4.71 |

### Final Selection Decision
* **Decision**: **PROMOTED Combined Ensemble (improved > 0.5% F1-score)** (Ensemble F1 Gain = 0.0081)
