# Early Fusion Models Benchmarking Report

This report documents the detailed Leave-One-Subject-Out (LOSO) training and evaluation metrics collected from the consolidated early fusion pipeline run on the GPU.

---

## 📊 Summary of Mean Accuracy Results

| Model Configuration | Mean Accuracy | Evaluation Protocol |
| :--- | :---: | :--- |
| Early Fusion Classifier | **0.6790** | 5-Fold LOSO GroupKFold |
| Gated Fusion Classifier | **0.6736** | 5-Fold LOSO GroupKFold |
| Cross Attention Fusion Classifier | **0.6727** | 5-Fold LOSO GroupKFold |
| FlexiModal MoE Classifier | **0.6704** | 5-Fold LOSO GroupKFold |

---

## 📈 Detailed Metrics for All Configurations

### Early Fusion Classifier
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6790 | 0.0105 |
| Precision | 0.6309 | 0.0762 |
| Recall | 0.5297 | 0.0529 |
| F1-Score | 0.5747 | 0.0572 |
| ROC-AUC | 0.7083 | 0.0373 |
| MSE | 0.2314 | 0.0063 |
| MAE | 0.3607 | 0.0176 |
| R2-Score | 0.0365 | 0.0615 |
| RMSE | 0.4810 | 0.0065 |

### Gated Fusion Classifier
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6736 | 0.0145 |
| Precision | 0.6284 | 0.0761 |
| Recall | 0.5415 | 0.0591 |
| F1-Score | 0.5760 | 0.0421 |
| ROC-AUC | 0.7164 | 0.0274 |
| MSE | 0.2355 | 0.0039 |
| MAE | 0.3580 | 0.0219 |
| R2-Score | 0.0197 | 0.0498 |
| RMSE | 0.4853 | 0.0040 |

### Cross Attention Fusion Classifier
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6727 | 0.0192 |
| Precision | 0.6268 | 0.0815 |
| Recall | 0.5314 | 0.0633 |
| F1-Score | 0.5704 | 0.0536 |
| ROC-AUC | 0.7148 | 0.0353 |
| MSE | 0.2317 | 0.0116 |
| MAE | 0.3605 | 0.0212 |
| R2-Score | 0.0357 | 0.0668 |
| RMSE | 0.4812 | 0.0120 |

### FlexiModal MoE Classifier
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6704 | 0.0206 |
| Precision | 0.6211 | 0.0769 |
| Recall | 0.5257 | 0.0274 |
| F1-Score | 0.5677 | 0.0434 |
| ROC-AUC | 0.6996 | 0.0380 |
| MSE | 0.2506 | 0.0150 |
| MAE | 0.3603 | 0.0223 |
| R2-Score | -0.0424 | 0.0682 |
| RMSE | 0.5004 | 0.0149 |
