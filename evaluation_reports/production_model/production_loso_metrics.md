# Production Models Multi-Strategy LOSO Benchmarking Report

This report summarizes the execution logs and detailed validation metrics of the production packaging run (`package_phase8_production.py`) executed on the GPU.

---

## 📊 Evaluation Summary

- **Dataset**: Certified Multimodal Dataset (43,110 synchronized frames aligned by subject, task, and window)
- **Validation Protocol**: 5-Fold Leave-One-Subject-Out (LOSO) GroupKFold on 65 Subjects
- **Device**: NVIDIA CUDA (GPU Accelerated)
- **Epochs**: 10 (Modality Encoders) | 50 (Dynamic Routers)

---

## 📈 Strategy 4 (Standard CNN-GRU) LOSO Results

### Face Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6338 | 0.0170 |
| **Precision** | 0.5680 | 0.0639 |
| **Recall** | 0.4895 | 0.0947 |
| **F1-Score** | 0.5197 | 0.0629 |
| **ROC-AUC** | 0.6548 | 0.0301 |
| **MSE** | 0.2452 | 0.0113 |
| **MAE** | 0.4136 | 0.0177 |
| **R2-Score** | -0.0200 | 0.0549 |
| **RMSE** | 0.4951 | 0.0113 |

### Voice Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6772 | 0.0338 |
| **Precision** | 0.6259 | 0.0526 |
| **Recall** | 0.5717 | 0.0982 |
| **F1-Score** | 0.5904 | 0.0501 |
| **ROC-AUC** | 0.7136 | 0.0588 |
| **MSE** | 0.2277 | 0.0204 |
| **MAE** | 0.3784 | 0.0294 |
| **R2-Score** | 0.0535 | 0.0827 |
| **RMSE** | 0.4767 | 0.0211 |

### Physio Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6430 | 0.0261 |
| **Precision** | 0.5745 | 0.0565 |
| **Recall** | 0.5444 | 0.0228 |
| **F1-Score** | 0.5574 | 0.0289 |
| **ROC-AUC** | 0.6674 | 0.0252 |
| **MSE** | 0.2274 | 0.0123 |
| **MAE** | 0.4318 | 0.0082 |
| **R2-Score** | 0.0552 | 0.0362 |
| **RMSE** | 0.4767 | 0.0130 |

### Face + Physio
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6521 | 0.0108 |
| **Precision** | 0.5909 | 0.0614 |
| **Recall** | 0.5147 | 0.0738 |
| **F1-Score** | 0.5468 | 0.0541 |
| **ROC-AUC** | 0.6851 | 0.0189 |
| **MSE** | 0.2232 | 0.0055 |
| **MAE** | 0.4186 | 0.0144 |
| **R2-Score** | 0.0715 | 0.0399 |
| **RMSE** | 0.4724 | 0.0057 |

### Face + Voice
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6870 | 0.0228 |
| **Precision** | 0.6376 | 0.0690 |
| **Recall** | 0.5480 | 0.1025 |
| **F1-Score** | 0.5854 | 0.0775 |
| **ROC-AUC** | 0.7128 | 0.0389 |
| **MSE** | 0.2135 | 0.0116 |
| **MAE** | 0.3950 | 0.0236 |
| **R2-Score** | 0.1117 | 0.0598 |
| **RMSE** | 0.4619 | 0.0126 |

### Voice + Physio
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6986 | 0.0265 |
| **Precision** | 0.6549 | 0.0497 |
| **Recall** | 0.5717 | 0.0925 |
| **F1-Score** | 0.6063 | 0.0615 |
| **ROC-AUC** | 0.7201 | 0.0198 |
| **MSE** | 0.2110 | 0.0083 |
| **MAE** | 0.3929 | 0.0250 |
| **R2-Score** | 0.1225 | 0.0363 |
| **RMSE** | 0.4593 | 0.0090 |

### All 3 Modalities (Fusion Router)
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6944 | 0.0163 |
| **Precision** | 0.6513 | 0.0630 |
| **Recall** | 0.5448 | 0.0977 |
| **F1-Score** | 0.5899 | 0.0752 |
| **ROC-AUC** | 0.7254 | 0.0227 |
| **MSE** | 0.2079 | 0.0073 |
| **MAE** | 0.4004 | 0.0215 |
| **R2-Score** | 0.1348 | 0.0462 |
| **RMSE** | 0.4559 | 0.0080 |

---

## 📈 Strategy 5 (Adversarial CNN-GRU) LOSO Results (PRIMARY)

### Face Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6603 | 0.0136 |
| **Precision** | 0.6111 | 0.0711 |
| **Recall** | 0.4610 | 0.0937 |
| **F1-Score** | 0.5229 | 0.0807 |
| **ROC-AUC** | 0.6672 | 0.0349 |
| **MSE** | 0.2312 | 0.0082 |
| **MAE** | 0.4149 | 0.0162 |
| **R2-Score** | 0.0383 | 0.0498 |
| **RMSE** | 0.4807 | 0.0085 |

### Voice Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6816 | 0.0481 |
| **Precision** | 0.6485 | 0.0712 |
| **Recall** | 0.5496 | 0.1176 |
| **F1-Score** | 0.5833 | 0.0603 |
| **ROC-AUC** | 0.6939 | 0.0457 |
| **MSE** | 0.2192 | 0.0180 |
| **MAE** | 0.3998 | 0.0404 |
| **R2-Score** | 0.0889 | 0.0719 |
| **RMSE** | 0.4678 | 0.0187 |

### Physio Only
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6603 | 0.0151 |
| **Precision** | 0.6138 | 0.0605 |
| **Recall** | 0.5012 | 0.0289 |
| **F1-Score** | 0.5488 | 0.0161 |
| **ROC-AUC** | 0.6918 | 0.0313 |
| **MSE** | 0.2223 | 0.0072 |
| **MAE** | 0.4363 | 0.0039 |
| **R2-Score** | 0.0756 | 0.0375 |
| **RMSE** | 0.4714 | 0.0076 |

### Face + Physio
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6703 | 0.0132 |
| **Precision** | 0.6355 | 0.0716 |
| **Recall** | 0.4898 | 0.0522 |
| **F1-Score** | 0.5492 | 0.0410 |
| **ROC-AUC** | 0.6998 | 0.0191 |
| **MSE** | 0.2171 | 0.0071 |
| **MAE** | 0.4210 | 0.0112 |
| **R2-Score** | 0.0969 | 0.0440 |
| **RMSE** | 0.4659 | 0.0077 |

### Face + Voice
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6981 | 0.0229 |
| **Precision** | 0.6640 | 0.0644 |
| **Recall** | 0.5312 | 0.1003 |
| **F1-Score** | 0.5867 | 0.0780 |
| **ROC-AUC** | 0.6984 | 0.0415 |
| **MSE** | 0.2102 | 0.0122 |
| **MAE** | 0.4054 | 0.0282 |
| **R2-Score** | 0.1253 | 0.0616 |
| **RMSE** | 0.4583 | 0.0134 |

### Voice + Physio
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.6937 | 0.0245 |
| **Precision** | 0.6598 | 0.0537 |
| **Recall** | 0.5436 | 0.0965 |
| **F1-Score** | 0.5899 | 0.0597 |
| **ROC-AUC** | 0.7240 | 0.0309 |
| **MSE** | 0.2082 | 0.0085 |
| **MAE** | 0.4063 | 0.0326 |
| **R2-Score** | 0.1340 | 0.0451 |
| **RMSE** | 0.4562 | 0.0093 |

### All 3 Modalities (Adversarial Router)
| Metric | Mean | Std |
| :--- | :---: | :---: |
| **Accuracy** | 0.7051 | 0.0216 |
| **Precision** | 0.6775 | 0.0638 |
| **Recall** | 0.5312 | 0.0910 |
| **F1-Score** | 0.5931 | 0.0748 |
| **ROC-AUC** | 0.7184 | 0.0279 |
| **MSE** | 0.2065 | 0.0104 |
| **MAE** | 0.4089 | 0.0252 |
| **R2-Score** | 0.1410 | 0.0550 |
| **RMSE** | 0.4542 | 0.0116 |

---

## 🛠️ Artifact Registration Details
All artifacts have been successfully built, optimized for GPU inference, and saved to disk. Standard and adversarial metadata configurations have been registered in the Version Registry (`models/registry.json`).
