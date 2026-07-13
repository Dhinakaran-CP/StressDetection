# High-Capacity Research Architectures Benchmarking Report

This report documents the detailed Leave-One-Subject-Out (LOSO) training and evaluation metrics collected from the high-capacity research pipeline execution run on the GPU.

---

## 📊 Summary of Mean Accuracy Results

| Modality Configuration / Model | Mean Accuracy | Evaluation Protocol |
| :--- | :---: | :--- |
| Unimodal Face Expert | **0.6664** | 5-Fold LOSO GroupKFold |
| Unimodal Voice Expert | **0.7153** | 5-Fold LOSO GroupKFold |
| Unimodal Physio Expert | **0.6466** | 5-Fold LOSO GroupKFold |
| Early Fusion Model | **0.6725** | 5-Fold LOSO GroupKFold |
| Gated Fusion Model | **0.6765** | 5-Fold LOSO GroupKFold |
| Cross Attention Fusion Model | **0.6728** | 5-Fold LOSO GroupKFold |
| Hybrid MoE Attention Model | **0.6687** | 5-Fold LOSO GroupKFold |
| Adversarial Hybrid MoE Attention Model | **0.6704** | 5-Fold LOSO GroupKFold |

---

## 📈 Detailed Metrics for All Modalities

### Unimodal Face Expert
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6664 | 0.0091 |
| Precision | 0.6272 | 0.0624 |
| Recall | 0.4748 | 0.1229 |
| F1-Score | 0.5310 | 0.0835 |
| ROC-AUC | 0.6817 | 0.0351 |
| MSE | 0.2325 | 0.0083 |
| MAE | 0.3966 | 0.0139 |
| R2-Score | 0.0331 | 0.0361 |
| RMSE | 0.4821 | 0.0086 |

### Unimodal Voice Expert
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.7153 | 0.0187 |
| Precision | 0.7248 | 0.0619 |
| Recall | 0.4807 | 0.0901 |
| F1-Score | 0.5762 | 0.0864 |
| ROC-AUC | 0.7023 | 0.0420 |
| MSE | 0.2026 | 0.0092 |
| MAE | 0.3877 | 0.0066 |
| R2-Score | 0.1568 | 0.0552 |
| RMSE | 0.4500 | 0.0102 |

### Unimodal Physio Expert
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6466 | 0.0211 |
| Precision | 0.5876 | 0.0414 |
| Recall | 0.5024 | 0.0291 |
| F1-Score | 0.5398 | 0.0140 |
| ROC-AUC | 0.6832 | 0.0219 |
| MSE | 0.2213 | 0.0114 |
| MAE | 0.4233 | 0.0125 |
| R2-Score | 0.0810 | 0.0209 |
| RMSE | 0.4702 | 0.0123 |

### Early Fusion Model
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6725 | 0.0118 |
| Precision | 0.6280 | 0.0760 |
| Recall | 0.5290 | 0.0658 |
| F1-Score | 0.5687 | 0.0485 |
| ROC-AUC | 0.7052 | 0.0300 |
| MSE | 0.2373 | 0.0065 |
| MAE | 0.3656 | 0.0182 |
| R2-Score | 0.0121 | 0.0578 |
| RMSE | 0.4871 | 0.0067 |

### Gated Fusion Model
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6765 | 0.0088 |
| Precision | 0.6192 | 0.0606 |
| Recall | 0.5540 | 0.0379 |
| F1-Score | 0.5844 | 0.0468 |
| ROC-AUC | 0.7043 | 0.0242 |
| MSE | 0.2363 | 0.0104 |
| MAE | 0.3626 | 0.0131 |
| R2-Score | 0.0168 | 0.0552 |
| RMSE | 0.4860 | 0.0106 |

### Cross Attention Fusion Model
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6728 | 0.0147 |
| Precision | 0.6071 | 0.0737 |
| Recall | 0.5851 | 0.0406 |
| F1-Score | 0.5948 | 0.0532 |
| ROC-AUC | 0.7099 | 0.0303 |
| MSE | 0.2363 | 0.0097 |
| MAE | 0.3646 | 0.0163 |
| R2-Score | 0.0159 | 0.0725 |
| RMSE | 0.4860 | 0.0099 |

### Hybrid MoE Attention Model
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6687 | 0.0173 |
| Precision | 0.6205 | 0.0718 |
| Recall | 0.5180 | 0.0581 |
| F1-Score | 0.5614 | 0.0519 |
| ROC-AUC | 0.7057 | 0.0328 |
| MSE | 0.2451 | 0.0064 |
| MAE | 0.3598 | 0.0186 |
| R2-Score | -0.0203 | 0.0562 |
| RMSE | 0.4951 | 0.0065 |

### Adversarial Hybrid MoE Attention Model
| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | 0.6704 | 0.0227 |
| Precision | 0.6104 | 0.0915 |
| Recall | 0.5321 | 0.0748 |
| F1-Score | 0.5680 | 0.0804 |
| ROC-AUC | 0.7001 | 0.0469 |
| MSE | 0.2426 | 0.0136 |
| MAE | 0.3594 | 0.0211 |
| R2-Score | -0.0110 | 0.0919 |
| RMSE | 0.4923 | 0.0139 |
