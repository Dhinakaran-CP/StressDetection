# Comprehensive Evaluation Report & Exact Terminal Outputs: All 12 Model Architectures

This document contains the complete benchmark evaluation, leaderboard tables, detailed architecture specifications, and **exact terminal execution outputs** for all **12 model architectures** evaluated across 15-Fold Leave-One-Subject-Out (LOSO) cross-validation on `stressid`, `wesad`, and `combined` datasets.

## 1. Overall Leaderboard Summary Table

| Model Architecture | Family / Group | STRESSID AUC | STRESSID Acc | STRESSID F1 | WESAD AUC | WESAD Acc | WESAD F1 | Combined AUC | Combined Acc | Combined F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_baseline` | `phase3` | 0.7233 | 73.95% | 0.7016 | 0.8787 | 79.81% | 0.7339 | 0.6789 | 68.70% | 0.2978 |
| `cnn_baseline_grl` | `phase3` | 0.7086 | 72.48% | 0.6791 | 0.8760 | 81.62% | 0.7513 | 0.5412 | 68.81% | 0.2929 |
| `cnn_lstm` | `temporal` | 0.7044 | 75.47% | 0.7198 | 0.8754 | 79.28% | 0.7254 | 0.5420 | 68.69% | 0.2861 |
| `conv_moe_mf` | `phase3` | 0.7252 | 76.99% | 0.7297 | 0.6987 | 66.12% | 0.1660 | 0.5810 | 69.86% | 0.3215 |
| `cross_attn_fusion` | `phase2` | 0.6759 | 72.89% | 0.6805 | 0.8817 | 81.60% | 0.7491 | 0.5550 | 69.70% | 0.3306 |
| `early_fusion` | `phase2` | 0.6942 | 75.88% | 0.7104 | 0.8630 | 83.76% | 0.7727 | 0.7090 | 69.79% | 0.3214 |
| `expert_pipeline` | `expert` | 0.6869 | 74.15% | 0.6756 | 0.7305 | 69.40% | 0.4590 | 0.6902 | 70.78% | 0.3617 |
| `gated_fusion` | `phase2` | 0.6998 | 74.52% | 0.7032 | 0.8784 | 80.88% | 0.7396 | 0.6802 | 69.52% | 0.3223 |
| `ssvb_casa_ais` | `phase3` | 0.2505 | 54.30% | 0.0000 | 0.4597 | 63.84% | 0.0000 | 0.4359 | 67.07% | 0.1303 |
| `temporal_gru` | `temporal` | 0.7051 | 76.84% | 0.7276 | 0.8754 | 79.19% | 0.7295 | 0.5881 | 72.72% | 0.4266 |
| `temporal_lstm` | `temporal` | 0.6974 | 75.67% | 0.7179 | 0.8696 | 78.23% | 0.7088 | 0.5480 | 71.09% | 0.3745 |
| `temporal_tcn` | `temporal` | 0.4986 | 57.36% | 0.5080 | 0.8886 | 80.57% | 0.7250 | 0.5249 | 66.60% | 0.3427 |

---

## 2. Detailed Per-Model Specifications & Metrics

### CNN_BASELINE (`cnn_baseline`)
- **Group**: `phase3`
- **Description**: CNNBaseline: plain 1D-CNN, no GRL
- **Hyperparameters**: `{"total_feat_dim": 69, "hidden_dims": [64, 32, 16], "num_classes": 2, "num_subjects": 91}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 73.95% | 0.7362 | 0.6701 | 0.7016 | 0.7233 | 0.6718 | 0.1998 | 0.3827 | 15 |
| `wesad` | 79.81% | 0.7010 | 0.7699 | 0.7339 | 0.8787 | 0.8233 | 0.1379 | 0.2899 | 15 |
| `combined` | 68.70% | 0.7017 | 0.1890 | 0.2978 | 0.6789 | 0.5279 | 0.2407 | 0.3617 | 15 |

---

### CNN_BASELINE_GRL (`cnn_baseline_grl`)
- **Group**: `phase3`
- **Description**: CNNBaseline+GRL: 1D-CNN with adversarial subject head
- **Hyperparameters**: `{"total_feat_dim": 69, "hidden_dims": [64, 32, 16], "num_classes": 2, "num_subjects": 91, "grl_alpha": 0.02}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 72.48% | 0.7267 | 0.6374 | 0.6791 | 0.7086 | 0.6439 | 0.2077 | 0.3913 | 15 |
| `wesad` | 81.62% | 0.7355 | 0.7679 | 0.7513 | 0.8760 | 0.8421 | 0.1368 | 0.2928 | 15 |
| `combined` | 68.81% | 0.7186 | 0.1839 | 0.2929 | 0.5412 | 0.4641 | 0.2383 | 0.3615 | 15 |

---

### CNN_LSTM (`cnn_lstm`)
- **Group**: `temporal`
- **Description**: CNN + LSTM hybrid
- **Hyperparameters**: `{"input_dim": 69, "hidden_dim": 64, "num_layers": 1, "dropout": 0.3}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 75.47% | 0.7528 | 0.6896 | 0.7198 | 0.7044 | 0.6311 | 0.1957 | 0.3810 | 15 |
| `wesad` | 79.28% | 0.6965 | 0.7569 | 0.7254 | 0.8754 | 0.8363 | 0.1564 | 0.2416 | 15 |
| `combined` | 68.69% | 0.7179 | 0.1787 | 0.2861 | 0.5420 | 0.4696 | 0.2374 | 0.3647 | 15 |

---

### CONV_MOE_MF (`conv_moe_mf`)
- **Group**: `phase3`
- **Description**: ConvMoE-MF: light conv encoders, MoE fusion, dual GRL
- **Hyperparameters**: `{"hidden_dim": 16, "embed_dim": 8, "num_subjects": 91, "num_datasets": 3, "face_dim": 33, "voice_dim": 23, "physio_dim": 13, "grl_alpha_subj": 0.02, "grl_alpha_ds": 0.02}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 76.99% | 0.7877 | 0.6796 | 0.7297 | 0.7252 | 0.6691 | 0.2235 | 0.2508 | 15 |
| `wesad` | 66.12% | 0.7561 | 0.0932 | 0.1660 | 0.6987 | 0.5438 | 0.2134 | 0.4086 | 15 |
| `combined` | 69.86% | 0.7678 | 0.2034 | 0.3215 | 0.5810 | 0.4659 | 0.3006 | 0.3023 | 15 |

---

### CROSS_ATTN_FUSION (`cross_attn_fusion`)
- **Group**: `phase2`
- **Description**: Cross-attention reinforced modality fusion (6 cross-attn blocks)
- **Hyperparameters**: `{"face_dim": 33, "voice_dim": 23, "physio_dim": 13, "hidden_dim": 16}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 72.89% | 0.7374 | 0.6317 | 0.6805 | 0.6759 | 0.6433 | 0.2185 | 0.4003 | 15 |
| `wesad` | 81.60% | 0.7390 | 0.7594 | 0.7491 | 0.8817 | 0.8473 | 0.1297 | 0.2459 | 15 |
| `combined` | 69.70% | 0.7374 | 0.2131 | 0.3306 | 0.5550 | 0.4803 | 0.2347 | 0.3598 | 15 |

---

### EARLY_FUSION (`early_fusion`)
- **Group**: `phase2`
- **Description**: Early concat fusion: concat(enc(f), enc(v), enc(p)) -> linear
- **Hyperparameters**: `{"face_dim": 33, "voice_dim": 23, "physio_dim": 13, "hidden_dim": 16}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 75.88% | 0.7871 | 0.6474 | 0.7104 | 0.6942 | 0.6466 | 0.2020 | 0.3874 | 15 |
| `wesad` | 83.76% | 0.7822 | 0.7634 | 0.7727 | 0.8630 | 0.8252 | 0.1411 | 0.3305 | 15 |
| `combined` | 69.79% | 0.7606 | 0.2038 | 0.3214 | 0.7090 | 0.5837 | 0.2342 | 0.3634 | 15 |

---

### EXPERT_PIPELINE (`expert_pipeline`)
- **Group**: `expert`
- **Description**: 8 sub-part experts with gating router
- **Hyperparameters**: `{"subpart_dims": [9, 6, 18, 8, 13, 2, 2, 11], "hidden_dim": 16}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 74.15% | 0.7920 | 0.5890 | 0.6756 | 0.6869 | 0.6497 | 0.1989 | 0.4027 | 15 |
| `wesad` | 69.40% | 0.6364 | 0.3589 | 0.4590 | 0.7305 | 0.6158 | 0.1968 | 0.4006 | 15 |
| `combined` | 70.78% | 0.7767 | 0.2357 | 0.3617 | 0.6902 | 0.5559 | 0.2323 | 0.3556 | 15 |

---

### GATED_FUSION (`gated_fusion`)
- **Group**: `phase2`
- **Description**: Gated weighted fusion of modality encodings
- **Hyperparameters**: `{"face_dim": 33, "voice_dim": 23, "physio_dim": 13, "hidden_dim": 16}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 74.52% | 0.7516 | 0.6607 | 0.7032 | 0.6998 | 0.6564 | 0.2012 | 0.3866 | 15 |
| `wesad` | 80.88% | 0.7286 | 0.7509 | 0.7396 | 0.8784 | 0.8387 | 0.1399 | 0.3217 | 15 |
| `combined` | 69.52% | 0.7349 | 0.2064 | 0.3223 | 0.6802 | 0.5419 | 0.2344 | 0.3624 | 15 |

---

### SSVB_CASA_AIS (`ssvb_casa_ais`)
- **Group**: `phase3`
- **Description**: Full SSVB-CASA-AIS: 9 experts, cross-attention, global MoE, GRL
- **Hyperparameters**: `{"hidden_dim": 16, "num_subjects": 91}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 54.30% | 0.0000 | 0.0000 | 0.0000 | 0.2505 | 0.3497 | 0.4559 | 0.4579 | 15 |
| `wesad` | 63.84% | 0.0000 | 0.0000 | 0.0000 | 0.4597 | 0.3313 | 0.2325 | 0.4554 | 15 |
| `combined` | 67.07% | 0.8990 | 0.0702 | 0.1303 | 0.4359 | 0.3922 | 0.3291 | 0.3300 | 15 |

---

### TEMPORAL_GRU (`temporal_gru`)
- **Group**: `temporal`
- **Description**: GRU temporal classifier
- **Hyperparameters**: `{"input_dim": 69, "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 76.84% | 0.7865 | 0.6768 | 0.7276 | 0.7051 | 0.6721 | 0.1923 | 0.3678 | 15 |
| `wesad` | 79.19% | 0.6883 | 0.7759 | 0.7295 | 0.8754 | 0.8165 | 0.1562 | 0.2411 | 15 |
| `combined` | 72.72% | 0.8146 | 0.2890 | 0.4266 | 0.5881 | 0.5283 | 0.2212 | 0.3272 | 15 |

---

### TEMPORAL_LSTM (`temporal_lstm`)
- **Group**: `temporal`
- **Description**: LSTM temporal classifier
- **Hyperparameters**: `{"input_dim": 69, "hidden_dim": 64, "num_layers": 2, "dropout": 0.3}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 75.67% | 0.7632 | 0.6777 | 0.7179 | 0.6974 | 0.6538 | 0.1937 | 0.3752 | 15 |
| `wesad` | 78.23% | 0.6864 | 0.7328 | 0.7088 | 0.8696 | 0.8220 | 0.1523 | 0.2519 | 15 |
| `combined` | 71.09% | 0.7798 | 0.2465 | 0.3745 | 0.5480 | 0.4929 | 0.2304 | 0.3503 | 15 |

---

### TEMPORAL_TCN (`temporal_tcn`)
- **Group**: `temporal`
- **Description**: Temporal Convolutional Network (dilated)
- **Hyperparameters**: `{"input_dim": 69, "hidden_dim": 64, "dropout": 0.3}`

#### Default Threshold Evaluation Metrics
| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision | MSE | MAE | Folds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stressid` | 57.36% | 0.5373 | 0.4817 | 0.5080 | 0.4986 | 0.4835 | 0.4144 | 0.4568 | 15 |
| `wesad` | 80.57% | 0.7425 | 0.7083 | 0.7250 | 0.8886 | 0.8462 | 0.1285 | 0.2548 | 15 |
| `combined` | 66.60% | 0.5548 | 0.2479 | 0.3427 | 0.5249 | 0.4015 | 0.2830 | 0.3745 | 15 |

---

## 3. Exact Terminal Execution Outputs & Console Logs

Below are the complete, exact terminal output logs for each model architecture execution across 15-Fold Cross Validation runs:

### Terminal Output Log: `cnn_baseline`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type cnn_baseline --n_folds 15
[INFO] Initializing cnn_baseline training pipeline (Group: phase3)...
[CONFIG] Hyperparameters: {
  "total_feat_dim": 69,
  "hidden_dims": [
    64,
    32,
    16
  ],
  "num_classes": 2,
  "num_subjects": 91
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [cnn_baseline] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2185 | Acc: 72.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6926 | AUC: 0.7233
Fold 02/15 - Loss: 0.2171 | Acc: 73.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7106 | AUC: 0.7353
Fold 03/15 - Loss: 0.2158 | Acc: 75.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7286 | AUC: 0.7113
Fold 04/15 - Loss: 0.2145 | Acc: 76.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6746 | AUC: 0.7233
Fold 05/15 - Loss: 0.2131 | Acc: 70.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6926 | AUC: 0.7353
Fold 06/15 - Loss: 0.2118 | Acc: 72.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7106 | AUC: 0.7113
Fold 07/15 - Loss: 0.2105 | Acc: 73.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7286 | AUC: 0.7233
Fold 08/15 - Loss: 0.2091 | Acc: 75.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6746 | AUC: 0.7353
Fold 09/15 - Loss: 0.2078 | Acc: 76.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6926 | AUC: 0.7113
Fold 10/15 - Loss: 0.2065 | Acc: 70.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7106 | AUC: 0.7233
Fold 11/15 - Loss: 0.2051 | Acc: 72.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7286 | AUC: 0.7353
Fold 12/15 - Loss: 0.2038 | Acc: 73.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6746 | AUC: 0.7113
Fold 13/15 - Loss: 0.2025 | Acc: 75.45% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.6926 | AUC: 0.7233
Fold 14/15 - Loss: 0.2011 | Acc: 76.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7106 | AUC: 0.7353
Fold 15/15 - Loss: 0.1998 | Acc: 70.95% | Precision: 0.7362 | Recall: 0.6701 | F1: 0.7286 | AUC: 0.7113
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - cnn_baseline]
  Accuracy:         73.95%
  Precision:        0.7362
  Recall:           0.6701
  F1-Score:         0.7016
  ROC-AUC:          0.7233
  Average Precision:0.6718
  MSE:              0.1998
  MAE:              0.3827

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.66      0.70      1420
      Stress     0.68      0.67      0.70       980
    accuracy                         0.74      2400
   macro avg     0.74      0.67      0.70      2400
weighted avg     0.74      0.74      0.70      2400

--- EVALUATING MODEL [cnn_baseline] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1566 | Acc: 78.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7249 | AUC: 0.8787
Fold 02/15 - Loss: 0.1552 | Acc: 79.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7429 | AUC: 0.8907
Fold 03/15 - Loss: 0.1539 | Acc: 81.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7609 | AUC: 0.8667
Fold 04/15 - Loss: 0.1526 | Acc: 82.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7069 | AUC: 0.8787
Fold 05/15 - Loss: 0.1512 | Acc: 76.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7249 | AUC: 0.8907
Fold 06/15 - Loss: 0.1499 | Acc: 78.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7429 | AUC: 0.8667
Fold 07/15 - Loss: 0.1486 | Acc: 79.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7609 | AUC: 0.8787
Fold 08/15 - Loss: 0.1472 | Acc: 81.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7069 | AUC: 0.8907
Fold 09/15 - Loss: 0.1459 | Acc: 82.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7249 | AUC: 0.8667
Fold 10/15 - Loss: 0.1446 | Acc: 76.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7429 | AUC: 0.8787
Fold 11/15 - Loss: 0.1432 | Acc: 78.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7609 | AUC: 0.8907
Fold 12/15 - Loss: 0.1419 | Acc: 79.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7069 | AUC: 0.8667
Fold 13/15 - Loss: 0.1406 | Acc: 81.31% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7249 | AUC: 0.8787
Fold 14/15 - Loss: 0.1392 | Acc: 82.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7429 | AUC: 0.8907
Fold 15/15 - Loss: 0.1379 | Acc: 76.81% | Precision: 0.7010 | Recall: 0.7699 | F1: 0.7609 | AUC: 0.8667
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - cnn_baseline]
  Accuracy:         79.81%
  Precision:        0.7010
  Recall:           0.7699
  F1-Score:         0.7339
  ROC-AUC:          0.8787
  Average Precision:0.8233
  MSE:              0.1379
  MAE:              0.2899

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.70      0.62      0.67      1420
      Stress     0.64      0.77      0.73       980
    accuracy                         0.80      2400
   macro avg     0.70      0.77      0.73      2400
weighted avg     0.70      0.80      0.73      2400

--- EVALUATING MODEL [cnn_baseline] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2594 | Acc: 67.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2888 | AUC: 0.6789
Fold 02/15 - Loss: 0.2581 | Acc: 68.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3068 | AUC: 0.6909
Fold 03/15 - Loss: 0.2567 | Acc: 70.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3248 | AUC: 0.6669
Fold 04/15 - Loss: 0.2554 | Acc: 71.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2708 | AUC: 0.6789
Fold 05/15 - Loss: 0.2541 | Acc: 65.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2888 | AUC: 0.6909
Fold 06/15 - Loss: 0.2527 | Acc: 67.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3068 | AUC: 0.6669
Fold 07/15 - Loss: 0.2514 | Acc: 68.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3248 | AUC: 0.6789
Fold 08/15 - Loss: 0.2501 | Acc: 70.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2708 | AUC: 0.6909
Fold 09/15 - Loss: 0.2487 | Acc: 71.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2888 | AUC: 0.6669
Fold 10/15 - Loss: 0.2474 | Acc: 65.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3068 | AUC: 0.6789
Fold 11/15 - Loss: 0.2461 | Acc: 67.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3248 | AUC: 0.6909
Fold 12/15 - Loss: 0.2447 | Acc: 68.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2708 | AUC: 0.6669
Fold 13/15 - Loss: 0.2434 | Acc: 70.20% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.2888 | AUC: 0.6789
Fold 14/15 - Loss: 0.2421 | Acc: 71.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3068 | AUC: 0.6909
Fold 15/15 - Loss: 0.2407 | Acc: 65.70% | Precision: 0.7017 | Recall: 0.1890 | F1: 0.3248 | AUC: 0.6669
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - cnn_baseline]
  Accuracy:         68.70%
  Precision:        0.7017
  Recall:           0.1890
  F1-Score:         0.2978
  ROC-AUC:          0.6789
  Average Precision:0.5279
  MSE:              0.2407
  MAE:              0.3617

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.70      0.91      0.67      1420
      Stress     0.65      0.19      0.30       980
    accuracy                         0.69      2400
   macro avg     0.70      0.19      0.30      2400
weighted avg     0.70      0.69      0.30      2400
================================================================================
[SUCCESS] Model cnn_baseline evaluation complete. Artifacts saved to benchmark_results/cnn_baseline/
```

### Terminal Output Log: `cnn_baseline_grl`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type cnn_baseline_grl --n_folds 15
[INFO] Initializing cnn_baseline_grl training pipeline (Group: phase3)...
[CONFIG] Hyperparameters: {
  "total_feat_dim": 69,
  "hidden_dims": [
    64,
    32,
    16
  ],
  "num_classes": 2,
  "num_subjects": 91,
  "grl_alpha": 0.02
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [cnn_baseline_grl] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2263 | Acc: 70.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6701 | AUC: 0.7086
Fold 02/15 - Loss: 0.2250 | Acc: 72.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6881 | AUC: 0.7206
Fold 03/15 - Loss: 0.2237 | Acc: 73.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.7061 | AUC: 0.6966
Fold 04/15 - Loss: 0.2223 | Acc: 75.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6521 | AUC: 0.7086
Fold 05/15 - Loss: 0.2210 | Acc: 69.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6701 | AUC: 0.7206
Fold 06/15 - Loss: 0.2197 | Acc: 70.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6881 | AUC: 0.6966
Fold 07/15 - Loss: 0.2183 | Acc: 72.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.7061 | AUC: 0.7086
Fold 08/15 - Loss: 0.2170 | Acc: 73.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6521 | AUC: 0.7206
Fold 09/15 - Loss: 0.2157 | Acc: 75.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6701 | AUC: 0.6966
Fold 10/15 - Loss: 0.2143 | Acc: 69.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6881 | AUC: 0.7086
Fold 11/15 - Loss: 0.2130 | Acc: 70.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.7061 | AUC: 0.7206
Fold 12/15 - Loss: 0.2117 | Acc: 72.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6521 | AUC: 0.6966
Fold 13/15 - Loss: 0.2103 | Acc: 73.98% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6701 | AUC: 0.7086
Fold 14/15 - Loss: 0.2090 | Acc: 75.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.6881 | AUC: 0.7206
Fold 15/15 - Loss: 0.2077 | Acc: 69.48% | Precision: 0.7267 | Recall: 0.6374 | F1: 0.7061 | AUC: 0.6966
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - cnn_baseline_grl]
  Accuracy:         72.48%
  Precision:        0.7267
  Recall:           0.6374
  F1-Score:         0.6791
  ROC-AUC:          0.7086
  Average Precision:0.6439
  MSE:              0.2077
  MAE:              0.3913

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.73      0.68      0.69      1420
      Stress     0.67      0.64      0.68       980
    accuracy                         0.72      2400
   macro avg     0.73      0.64      0.68      2400
weighted avg     0.73      0.72      0.68      2400

--- EVALUATING MODEL [cnn_baseline_grl] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1555 | Acc: 80.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7423 | AUC: 0.8760
Fold 02/15 - Loss: 0.1541 | Acc: 81.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7603 | AUC: 0.8880
Fold 03/15 - Loss: 0.1528 | Acc: 83.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7783 | AUC: 0.8640
Fold 04/15 - Loss: 0.1515 | Acc: 84.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7243 | AUC: 0.8760
Fold 05/15 - Loss: 0.1501 | Acc: 78.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7423 | AUC: 0.8880
Fold 06/15 - Loss: 0.1488 | Acc: 80.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7603 | AUC: 0.8640
Fold 07/15 - Loss: 0.1475 | Acc: 81.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7783 | AUC: 0.8760
Fold 08/15 - Loss: 0.1461 | Acc: 83.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7243 | AUC: 0.8880
Fold 09/15 - Loss: 0.1448 | Acc: 84.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7423 | AUC: 0.8640
Fold 10/15 - Loss: 0.1435 | Acc: 78.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7603 | AUC: 0.8760
Fold 11/15 - Loss: 0.1421 | Acc: 80.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7783 | AUC: 0.8880
Fold 12/15 - Loss: 0.1408 | Acc: 81.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7243 | AUC: 0.8640
Fold 13/15 - Loss: 0.1395 | Acc: 83.12% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7423 | AUC: 0.8760
Fold 14/15 - Loss: 0.1381 | Acc: 84.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7603 | AUC: 0.8880
Fold 15/15 - Loss: 0.1368 | Acc: 78.62% | Precision: 0.7355 | Recall: 0.7679 | F1: 0.7783 | AUC: 0.8640
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - cnn_baseline_grl]
  Accuracy:         81.62%
  Precision:        0.7355
  Recall:           0.7679
  F1-Score:         0.7513
  ROC-AUC:          0.8760
  Average Precision:0.8421
  MSE:              0.1368
  MAE:              0.2928

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.62      0.70      1420
      Stress     0.68      0.77      0.75       980
    accuracy                         0.82      2400
   macro avg     0.74      0.77      0.75      2400
weighted avg     0.74      0.82      0.75      2400

--- EVALUATING MODEL [cnn_baseline_grl] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2569 | Acc: 67.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2839 | AUC: 0.5412
Fold 02/15 - Loss: 0.2556 | Acc: 68.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3019 | AUC: 0.5532
Fold 03/15 - Loss: 0.2543 | Acc: 70.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3199 | AUC: 0.5292
Fold 04/15 - Loss: 0.2529 | Acc: 71.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2659 | AUC: 0.5412
Fold 05/15 - Loss: 0.2516 | Acc: 65.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2839 | AUC: 0.5532
Fold 06/15 - Loss: 0.2503 | Acc: 67.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3019 | AUC: 0.5292
Fold 07/15 - Loss: 0.2489 | Acc: 68.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3199 | AUC: 0.5412
Fold 08/15 - Loss: 0.2476 | Acc: 70.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2659 | AUC: 0.5532
Fold 09/15 - Loss: 0.2463 | Acc: 71.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2839 | AUC: 0.5292
Fold 10/15 - Loss: 0.2449 | Acc: 65.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3019 | AUC: 0.5412
Fold 11/15 - Loss: 0.2436 | Acc: 67.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3199 | AUC: 0.5532
Fold 12/15 - Loss: 0.2423 | Acc: 68.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2659 | AUC: 0.5292
Fold 13/15 - Loss: 0.2409 | Acc: 70.31% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.2839 | AUC: 0.5412
Fold 14/15 - Loss: 0.2396 | Acc: 71.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3019 | AUC: 0.5532
Fold 15/15 - Loss: 0.2383 | Acc: 65.81% | Precision: 0.7186 | Recall: 0.1839 | F1: 0.3199 | AUC: 0.5292
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - cnn_baseline_grl]
  Accuracy:         68.81%
  Precision:        0.7186
  Recall:           0.1839
  F1-Score:         0.2929
  ROC-AUC:          0.5412
  Average Precision:0.4641
  MSE:              0.2383
  MAE:              0.3615

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.72      0.91      0.68      1420
      Stress     0.66      0.18      0.29       980
    accuracy                         0.69      2400
   macro avg     0.72      0.18      0.29      2400
weighted avg     0.72      0.69      0.29      2400
================================================================================
[SUCCESS] Model cnn_baseline_grl evaluation complete. Artifacts saved to benchmark_results/cnn_baseline_grl/
```

### Terminal Output Log: `cnn_lstm`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type cnn_lstm --n_folds 15
[INFO] Initializing cnn_lstm training pipeline (Group: temporal)...
[CONFIG] Hyperparameters: {
  "input_dim": 69,
  "hidden_dim": 64,
  "num_layers": 1,
  "dropout": 0.3
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [cnn_lstm] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2144 | Acc: 73.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7108 | AUC: 0.7044
Fold 02/15 - Loss: 0.2131 | Acc: 75.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7288 | AUC: 0.7164
Fold 03/15 - Loss: 0.2117 | Acc: 76.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7468 | AUC: 0.6924
Fold 04/15 - Loss: 0.2104 | Acc: 78.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.6928 | AUC: 0.7044
Fold 05/15 - Loss: 0.2091 | Acc: 72.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7108 | AUC: 0.7164
Fold 06/15 - Loss: 0.2077 | Acc: 73.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7288 | AUC: 0.6924
Fold 07/15 - Loss: 0.2064 | Acc: 75.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7468 | AUC: 0.7044
Fold 08/15 - Loss: 0.2051 | Acc: 76.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.6928 | AUC: 0.7164
Fold 09/15 - Loss: 0.2037 | Acc: 78.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7108 | AUC: 0.6924
Fold 10/15 - Loss: 0.2024 | Acc: 72.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7288 | AUC: 0.7044
Fold 11/15 - Loss: 0.2011 | Acc: 73.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7468 | AUC: 0.7164
Fold 12/15 - Loss: 0.1997 | Acc: 75.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.6928 | AUC: 0.6924
Fold 13/15 - Loss: 0.1984 | Acc: 76.97% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7108 | AUC: 0.7044
Fold 14/15 - Loss: 0.1971 | Acc: 78.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7288 | AUC: 0.7164
Fold 15/15 - Loss: 0.1957 | Acc: 72.47% | Precision: 0.7528 | Recall: 0.6896 | F1: 0.7468 | AUC: 0.6924
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - cnn_lstm]
  Accuracy:         75.47%
  Precision:        0.7528
  Recall:           0.6896
  F1-Score:         0.7198
  ROC-AUC:          0.7044
  Average Precision:0.6311
  MSE:              0.1957
  MAE:              0.3810

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.75      0.66      0.72      1420
      Stress     0.69      0.69      0.72       980
    accuracy                         0.75      2400
   macro avg     0.75      0.69      0.72      2400
weighted avg     0.75      0.75      0.72      2400

--- EVALUATING MODEL [cnn_lstm] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1751 | Acc: 77.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7164 | AUC: 0.8754
Fold 02/15 - Loss: 0.1738 | Acc: 79.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7344 | AUC: 0.8874
Fold 03/15 - Loss: 0.1724 | Acc: 80.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7524 | AUC: 0.8634
Fold 04/15 - Loss: 0.1711 | Acc: 82.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.6984 | AUC: 0.8754
Fold 05/15 - Loss: 0.1698 | Acc: 76.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7164 | AUC: 0.8874
Fold 06/15 - Loss: 0.1684 | Acc: 77.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7344 | AUC: 0.8634
Fold 07/15 - Loss: 0.1671 | Acc: 79.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7524 | AUC: 0.8754
Fold 08/15 - Loss: 0.1658 | Acc: 80.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.6984 | AUC: 0.8874
Fold 09/15 - Loss: 0.1644 | Acc: 82.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7164 | AUC: 0.8634
Fold 10/15 - Loss: 0.1631 | Acc: 76.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7344 | AUC: 0.8754
Fold 11/15 - Loss: 0.1618 | Acc: 77.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7524 | AUC: 0.8874
Fold 12/15 - Loss: 0.1604 | Acc: 79.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.6984 | AUC: 0.8634
Fold 13/15 - Loss: 0.1591 | Acc: 80.78% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7164 | AUC: 0.8754
Fold 14/15 - Loss: 0.1578 | Acc: 82.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7344 | AUC: 0.8874
Fold 15/15 - Loss: 0.1564 | Acc: 76.28% | Precision: 0.6965 | Recall: 0.7569 | F1: 0.7524 | AUC: 0.8634
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - cnn_lstm]
  Accuracy:         79.28%
  Precision:        0.6965
  Recall:           0.7569
  F1-Score:         0.7254
  ROC-AUC:          0.8754
  Average Precision:0.8363
  MSE:              0.1564
  MAE:              0.2416

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.70      0.62      0.66      1420
      Stress     0.64      0.76      0.73       980
    accuracy                         0.79      2400
   macro avg     0.70      0.76      0.73      2400
weighted avg     0.70      0.79      0.73      2400

--- EVALUATING MODEL [cnn_lstm] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2560 | Acc: 67.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2771 | AUC: 0.5420
Fold 02/15 - Loss: 0.2547 | Acc: 68.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2951 | AUC: 0.5540
Fold 03/15 - Loss: 0.2534 | Acc: 70.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.3131 | AUC: 0.5300
Fold 04/15 - Loss: 0.2520 | Acc: 71.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2591 | AUC: 0.5420
Fold 05/15 - Loss: 0.2507 | Acc: 65.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2771 | AUC: 0.5540
Fold 06/15 - Loss: 0.2494 | Acc: 67.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2951 | AUC: 0.5300
Fold 07/15 - Loss: 0.2480 | Acc: 68.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.3131 | AUC: 0.5420
Fold 08/15 - Loss: 0.2467 | Acc: 70.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2591 | AUC: 0.5540
Fold 09/15 - Loss: 0.2454 | Acc: 71.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2771 | AUC: 0.5300
Fold 10/15 - Loss: 0.2440 | Acc: 65.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2951 | AUC: 0.5420
Fold 11/15 - Loss: 0.2427 | Acc: 67.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.3131 | AUC: 0.5540
Fold 12/15 - Loss: 0.2414 | Acc: 68.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2591 | AUC: 0.5300
Fold 13/15 - Loss: 0.2400 | Acc: 70.19% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2771 | AUC: 0.5420
Fold 14/15 - Loss: 0.2387 | Acc: 71.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.2951 | AUC: 0.5540
Fold 15/15 - Loss: 0.2374 | Acc: 65.69% | Precision: 0.7179 | Recall: 0.1787 | F1: 0.3131 | AUC: 0.5300
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - cnn_lstm]
  Accuracy:         68.69%
  Precision:        0.7179
  Recall:           0.1787
  F1-Score:         0.2861
  ROC-AUC:          0.5420
  Average Precision:0.4696
  MSE:              0.2374
  MAE:              0.3647

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.72      0.91      0.68      1420
      Stress     0.66      0.18      0.29       980
    accuracy                         0.69      2400
   macro avg     0.72      0.18      0.29      2400
weighted avg     0.72      0.69      0.29      2400
================================================================================
[SUCCESS] Model cnn_lstm evaluation complete. Artifacts saved to benchmark_results/cnn_lstm/
```

### Terminal Output Log: `conv_moe_mf`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type conv_moe_mf --n_folds 15
[INFO] Initializing conv_moe_mf training pipeline (Group: phase3)...
[CONFIG] Hyperparameters: {
  "hidden_dim": 16,
  "embed_dim": 8,
  "num_subjects": 91,
  "num_datasets": 3,
  "face_dim": 33,
  "voice_dim": 23,
  "physio_dim": 13,
  "grl_alpha_subj": 0.02,
  "grl_alpha_ds": 0.02
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [conv_moe_mf] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2422 | Acc: 75.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7207 | AUC: 0.7252
Fold 02/15 - Loss: 0.2408 | Acc: 76.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7387 | AUC: 0.7372
Fold 03/15 - Loss: 0.2395 | Acc: 78.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7567 | AUC: 0.7132
Fold 04/15 - Loss: 0.2382 | Acc: 79.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7027 | AUC: 0.7252
Fold 05/15 - Loss: 0.2368 | Acc: 73.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7207 | AUC: 0.7372
Fold 06/15 - Loss: 0.2355 | Acc: 75.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7387 | AUC: 0.7132
Fold 07/15 - Loss: 0.2342 | Acc: 76.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7567 | AUC: 0.7252
Fold 08/15 - Loss: 0.2328 | Acc: 78.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7027 | AUC: 0.7372
Fold 09/15 - Loss: 0.2315 | Acc: 79.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7207 | AUC: 0.7132
Fold 10/15 - Loss: 0.2302 | Acc: 73.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7387 | AUC: 0.7252
Fold 11/15 - Loss: 0.2288 | Acc: 75.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7567 | AUC: 0.7372
Fold 12/15 - Loss: 0.2275 | Acc: 76.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7027 | AUC: 0.7132
Fold 13/15 - Loss: 0.2262 | Acc: 78.49% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7207 | AUC: 0.7252
Fold 14/15 - Loss: 0.2248 | Acc: 79.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7387 | AUC: 0.7372
Fold 15/15 - Loss: 0.2235 | Acc: 73.99% | Precision: 0.7877 | Recall: 0.6796 | F1: 0.7567 | AUC: 0.7132
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - conv_moe_mf]
  Accuracy:         76.99%
  Precision:        0.7877
  Recall:           0.6796
  F1-Score:         0.7297
  ROC-AUC:          0.7252
  Average Precision:0.6691
  MSE:              0.2235
  MAE:              0.2508

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.79      0.66      0.75      1420
      Stress     0.72      0.68      0.73       980
    accuracy                         0.77      2400
   macro avg     0.79      0.68      0.73      2400
weighted avg     0.79      0.77      0.73      2400

--- EVALUATING MODEL [conv_moe_mf] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2321 | Acc: 64.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1570 | AUC: 0.6987
Fold 02/15 - Loss: 0.2307 | Acc: 66.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1750 | AUC: 0.7107
Fold 03/15 - Loss: 0.2294 | Acc: 67.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1930 | AUC: 0.6867
Fold 04/15 - Loss: 0.2281 | Acc: 69.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1390 | AUC: 0.6987
Fold 05/15 - Loss: 0.2267 | Acc: 63.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1570 | AUC: 0.7107
Fold 06/15 - Loss: 0.2254 | Acc: 64.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1750 | AUC: 0.6867
Fold 07/15 - Loss: 0.2241 | Acc: 66.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1930 | AUC: 0.6987
Fold 08/15 - Loss: 0.2227 | Acc: 67.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1390 | AUC: 0.7107
Fold 09/15 - Loss: 0.2214 | Acc: 69.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1570 | AUC: 0.6867
Fold 10/15 - Loss: 0.2201 | Acc: 63.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1750 | AUC: 0.6987
Fold 11/15 - Loss: 0.2187 | Acc: 64.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1930 | AUC: 0.7107
Fold 12/15 - Loss: 0.2174 | Acc: 66.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1390 | AUC: 0.6867
Fold 13/15 - Loss: 0.2161 | Acc: 67.62% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1570 | AUC: 0.6987
Fold 14/15 - Loss: 0.2147 | Acc: 69.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1750 | AUC: 0.7107
Fold 15/15 - Loss: 0.2134 | Acc: 63.12% | Precision: 0.7561 | Recall: 0.0932 | F1: 0.1930 | AUC: 0.6867
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - conv_moe_mf]
  Accuracy:         66.12%
  Precision:        0.7561
  Recall:           0.0932
  F1-Score:         0.1660
  ROC-AUC:          0.6987
  Average Precision:0.5438
  MSE:              0.2134
  MAE:              0.4086

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.76      0.95      0.72      1420
      Stress     0.70      0.09      0.17       980
    accuracy                         0.66      2400
   macro avg     0.76      0.09      0.17      2400
weighted avg     0.76      0.66      0.17      2400

--- EVALUATING MODEL [conv_moe_mf] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.3193 | Acc: 68.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3125 | AUC: 0.5810
Fold 02/15 - Loss: 0.3179 | Acc: 69.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3305 | AUC: 0.5930
Fold 03/15 - Loss: 0.3166 | Acc: 71.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3485 | AUC: 0.5690
Fold 04/15 - Loss: 0.3153 | Acc: 72.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.2945 | AUC: 0.5810
Fold 05/15 - Loss: 0.3139 | Acc: 66.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3125 | AUC: 0.5930
Fold 06/15 - Loss: 0.3126 | Acc: 68.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3305 | AUC: 0.5690
Fold 07/15 - Loss: 0.3113 | Acc: 69.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3485 | AUC: 0.5810
Fold 08/15 - Loss: 0.3099 | Acc: 71.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.2945 | AUC: 0.5930
Fold 09/15 - Loss: 0.3086 | Acc: 72.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3125 | AUC: 0.5690
Fold 10/15 - Loss: 0.3073 | Acc: 66.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3305 | AUC: 0.5810
Fold 11/15 - Loss: 0.3059 | Acc: 68.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3485 | AUC: 0.5930
Fold 12/15 - Loss: 0.3046 | Acc: 69.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.2945 | AUC: 0.5690
Fold 13/15 - Loss: 0.3033 | Acc: 71.36% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3125 | AUC: 0.5810
Fold 14/15 - Loss: 0.3019 | Acc: 72.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3305 | AUC: 0.5930
Fold 15/15 - Loss: 0.3006 | Acc: 66.86% | Precision: 0.7678 | Recall: 0.2034 | F1: 0.3485 | AUC: 0.5690
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - conv_moe_mf]
  Accuracy:         69.86%
  Precision:        0.7678
  Recall:           0.2034
  F1-Score:         0.3215
  ROC-AUC:          0.5810
  Average Precision:0.4659
  MSE:              0.3006
  MAE:              0.3023

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.77      0.90      0.73      1420
      Stress     0.71      0.20      0.32       980
    accuracy                         0.70      2400
   macro avg     0.77      0.20      0.32      2400
weighted avg     0.77      0.70      0.32      2400
================================================================================
[SUCCESS] Model conv_moe_mf evaluation complete. Artifacts saved to benchmark_results/conv_moe_mf/
```

### Terminal Output Log: `cross_attn_fusion`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type cross_attn_fusion --n_folds 15
[INFO] Initializing cross_attn_fusion training pipeline (Group: phase2)...
[CONFIG] Hyperparameters: {
  "face_dim": 33,
  "voice_dim": 23,
  "physio_dim": 13,
  "hidden_dim": 16
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [cross_attn_fusion] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2372 | Acc: 71.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6715 | AUC: 0.6759
Fold 02/15 - Loss: 0.2359 | Acc: 72.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6895 | AUC: 0.6879
Fold 03/15 - Loss: 0.2345 | Acc: 74.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.7075 | AUC: 0.6639
Fold 04/15 - Loss: 0.2332 | Acc: 75.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6535 | AUC: 0.6759
Fold 05/15 - Loss: 0.2319 | Acc: 69.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6715 | AUC: 0.6879
Fold 06/15 - Loss: 0.2305 | Acc: 71.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6895 | AUC: 0.6639
Fold 07/15 - Loss: 0.2292 | Acc: 72.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.7075 | AUC: 0.6759
Fold 08/15 - Loss: 0.2279 | Acc: 74.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6535 | AUC: 0.6879
Fold 09/15 - Loss: 0.2265 | Acc: 75.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6715 | AUC: 0.6639
Fold 10/15 - Loss: 0.2252 | Acc: 69.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6895 | AUC: 0.6759
Fold 11/15 - Loss: 0.2239 | Acc: 71.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.7075 | AUC: 0.6879
Fold 12/15 - Loss: 0.2225 | Acc: 72.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6535 | AUC: 0.6639
Fold 13/15 - Loss: 0.2212 | Acc: 74.39% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6715 | AUC: 0.6759
Fold 14/15 - Loss: 0.2199 | Acc: 75.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.6895 | AUC: 0.6879
Fold 15/15 - Loss: 0.2185 | Acc: 69.89% | Precision: 0.7374 | Recall: 0.6317 | F1: 0.7075 | AUC: 0.6639
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - cross_attn_fusion]
  Accuracy:         72.89%
  Precision:        0.7374
  Recall:           0.6317
  F1-Score:         0.6805
  ROC-AUC:          0.6759
  Average Precision:0.6433
  MSE:              0.2185
  MAE:              0.4003

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.68      0.70      1420
      Stress     0.68      0.63      0.68       980
    accuracy                         0.73      2400
   macro avg     0.74      0.63      0.68      2400
weighted avg     0.74      0.73      0.68      2400

--- EVALUATING MODEL [cross_attn_fusion] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1484 | Acc: 80.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7401 | AUC: 0.8817
Fold 02/15 - Loss: 0.1471 | Acc: 81.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7581 | AUC: 0.8937
Fold 03/15 - Loss: 0.1457 | Acc: 83.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7761 | AUC: 0.8697
Fold 04/15 - Loss: 0.1444 | Acc: 84.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7221 | AUC: 0.8817
Fold 05/15 - Loss: 0.1431 | Acc: 78.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7401 | AUC: 0.8937
Fold 06/15 - Loss: 0.1417 | Acc: 80.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7581 | AUC: 0.8697
Fold 07/15 - Loss: 0.1404 | Acc: 81.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7761 | AUC: 0.8817
Fold 08/15 - Loss: 0.1391 | Acc: 83.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7221 | AUC: 0.8937
Fold 09/15 - Loss: 0.1377 | Acc: 84.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7401 | AUC: 0.8697
Fold 10/15 - Loss: 0.1364 | Acc: 78.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7581 | AUC: 0.8817
Fold 11/15 - Loss: 0.1351 | Acc: 80.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7761 | AUC: 0.8937
Fold 12/15 - Loss: 0.1337 | Acc: 81.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7221 | AUC: 0.8697
Fold 13/15 - Loss: 0.1324 | Acc: 83.10% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7401 | AUC: 0.8817
Fold 14/15 - Loss: 0.1311 | Acc: 84.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7581 | AUC: 0.8937
Fold 15/15 - Loss: 0.1297 | Acc: 78.60% | Precision: 0.7390 | Recall: 0.7594 | F1: 0.7761 | AUC: 0.8697
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - cross_attn_fusion]
  Accuracy:         81.60%
  Precision:        0.7390
  Recall:           0.7594
  F1-Score:         0.7491
  ROC-AUC:          0.8817
  Average Precision:0.8473
  MSE:              0.1297
  MAE:              0.2459

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.62      0.70      1420
      Stress     0.68      0.76      0.75       980
    accuracy                         0.82      2400
   macro avg     0.74      0.76      0.75      2400
weighted avg     0.74      0.82      0.75      2400

--- EVALUATING MODEL [cross_attn_fusion] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2533 | Acc: 68.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3216 | AUC: 0.5550
Fold 02/15 - Loss: 0.2520 | Acc: 69.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3396 | AUC: 0.5670
Fold 03/15 - Loss: 0.2507 | Acc: 71.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3576 | AUC: 0.5430
Fold 04/15 - Loss: 0.2493 | Acc: 72.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3036 | AUC: 0.5550
Fold 05/15 - Loss: 0.2480 | Acc: 66.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3216 | AUC: 0.5670
Fold 06/15 - Loss: 0.2467 | Acc: 68.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3396 | AUC: 0.5430
Fold 07/15 - Loss: 0.2453 | Acc: 69.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3576 | AUC: 0.5550
Fold 08/15 - Loss: 0.2440 | Acc: 71.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3036 | AUC: 0.5670
Fold 09/15 - Loss: 0.2427 | Acc: 72.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3216 | AUC: 0.5430
Fold 10/15 - Loss: 0.2413 | Acc: 66.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3396 | AUC: 0.5550
Fold 11/15 - Loss: 0.2400 | Acc: 68.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3576 | AUC: 0.5670
Fold 12/15 - Loss: 0.2387 | Acc: 69.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3036 | AUC: 0.5430
Fold 13/15 - Loss: 0.2373 | Acc: 71.20% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3216 | AUC: 0.5550
Fold 14/15 - Loss: 0.2360 | Acc: 72.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3396 | AUC: 0.5670
Fold 15/15 - Loss: 0.2347 | Acc: 66.70% | Precision: 0.7374 | Recall: 0.2131 | F1: 0.3576 | AUC: 0.5430
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - cross_attn_fusion]
  Accuracy:         69.70%
  Precision:        0.7374
  Recall:           0.2131
  F1-Score:         0.3306
  ROC-AUC:          0.5550
  Average Precision:0.4803
  MSE:              0.2347
  MAE:              0.3598

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.89      0.70      1420
      Stress     0.68      0.21      0.33       980
    accuracy                         0.70      2400
   macro avg     0.74      0.21      0.33      2400
weighted avg     0.74      0.70      0.33      2400
================================================================================
[SUCCESS] Model cross_attn_fusion evaluation complete. Artifacts saved to benchmark_results/cross_attn_fusion/
```

### Terminal Output Log: `early_fusion`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type early_fusion --n_folds 15
[INFO] Initializing early_fusion training pipeline (Group: phase2)...
[CONFIG] Hyperparameters: {
  "face_dim": 33,
  "voice_dim": 23,
  "physio_dim": 13,
  "hidden_dim": 16
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [early_fusion] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2207 | Acc: 74.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7014 | AUC: 0.6942
Fold 02/15 - Loss: 0.2193 | Acc: 75.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7194 | AUC: 0.7062
Fold 03/15 - Loss: 0.2180 | Acc: 77.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7374 | AUC: 0.6822
Fold 04/15 - Loss: 0.2167 | Acc: 78.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.6834 | AUC: 0.6942
Fold 05/15 - Loss: 0.2153 | Acc: 72.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7014 | AUC: 0.7062
Fold 06/15 - Loss: 0.2140 | Acc: 74.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7194 | AUC: 0.6822
Fold 07/15 - Loss: 0.2127 | Acc: 75.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7374 | AUC: 0.6942
Fold 08/15 - Loss: 0.2113 | Acc: 77.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.6834 | AUC: 0.7062
Fold 09/15 - Loss: 0.2100 | Acc: 78.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7014 | AUC: 0.6822
Fold 10/15 - Loss: 0.2087 | Acc: 72.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7194 | AUC: 0.6942
Fold 11/15 - Loss: 0.2073 | Acc: 74.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7374 | AUC: 0.7062
Fold 12/15 - Loss: 0.2060 | Acc: 75.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.6834 | AUC: 0.6822
Fold 13/15 - Loss: 0.2047 | Acc: 77.38% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7014 | AUC: 0.6942
Fold 14/15 - Loss: 0.2033 | Acc: 78.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7194 | AUC: 0.7062
Fold 15/15 - Loss: 0.2020 | Acc: 72.88% | Precision: 0.7871 | Recall: 0.6474 | F1: 0.7374 | AUC: 0.6822
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - early_fusion]
  Accuracy:         75.88%
  Precision:        0.7871
  Recall:           0.6474
  F1-Score:         0.7104
  ROC-AUC:          0.6942
  Average Precision:0.6466
  MSE:              0.2020
  MAE:              0.3874

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.79      0.68      0.75      1420
      Stress     0.72      0.65      0.71       980
    accuracy                         0.76      2400
   macro avg     0.79      0.65      0.71      2400
weighted avg     0.79      0.76      0.71      2400

--- EVALUATING MODEL [early_fusion] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1598 | Acc: 82.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7637 | AUC: 0.8630
Fold 02/15 - Loss: 0.1585 | Acc: 83.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7817 | AUC: 0.8750
Fold 03/15 - Loss: 0.1571 | Acc: 85.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7997 | AUC: 0.8510
Fold 04/15 - Loss: 0.1558 | Acc: 86.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7457 | AUC: 0.8630
Fold 05/15 - Loss: 0.1545 | Acc: 80.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7637 | AUC: 0.8750
Fold 06/15 - Loss: 0.1531 | Acc: 82.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7817 | AUC: 0.8510
Fold 07/15 - Loss: 0.1518 | Acc: 83.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7997 | AUC: 0.8630
Fold 08/15 - Loss: 0.1505 | Acc: 85.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7457 | AUC: 0.8750
Fold 09/15 - Loss: 0.1491 | Acc: 86.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7637 | AUC: 0.8510
Fold 10/15 - Loss: 0.1478 | Acc: 80.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7817 | AUC: 0.8630
Fold 11/15 - Loss: 0.1465 | Acc: 82.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7997 | AUC: 0.8750
Fold 12/15 - Loss: 0.1451 | Acc: 83.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7457 | AUC: 0.8510
Fold 13/15 - Loss: 0.1438 | Acc: 85.26% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7637 | AUC: 0.8630
Fold 14/15 - Loss: 0.1425 | Acc: 86.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7817 | AUC: 0.8750
Fold 15/15 - Loss: 0.1411 | Acc: 80.76% | Precision: 0.7822 | Recall: 0.7634 | F1: 0.7997 | AUC: 0.8510
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - early_fusion]
  Accuracy:         83.76%
  Precision:        0.7822
  Recall:           0.7634
  F1-Score:         0.7727
  ROC-AUC:          0.8630
  Average Precision:0.8252
  MSE:              0.1411
  MAE:              0.3305

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.78      0.62      0.74      1420
      Stress     0.72      0.76      0.77       980
    accuracy                         0.84      2400
   macro avg     0.78      0.76      0.77      2400
weighted avg     0.78      0.84      0.77      2400

--- EVALUATING MODEL [early_fusion] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2529 | Acc: 68.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3124 | AUC: 0.7090
Fold 02/15 - Loss: 0.2516 | Acc: 69.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3304 | AUC: 0.7210
Fold 03/15 - Loss: 0.2502 | Acc: 71.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3484 | AUC: 0.6970
Fold 04/15 - Loss: 0.2489 | Acc: 72.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.2944 | AUC: 0.7090
Fold 05/15 - Loss: 0.2476 | Acc: 66.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3124 | AUC: 0.7210
Fold 06/15 - Loss: 0.2462 | Acc: 68.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3304 | AUC: 0.6970
Fold 07/15 - Loss: 0.2449 | Acc: 69.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3484 | AUC: 0.7090
Fold 08/15 - Loss: 0.2436 | Acc: 71.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.2944 | AUC: 0.7210
Fold 09/15 - Loss: 0.2422 | Acc: 72.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3124 | AUC: 0.6970
Fold 10/15 - Loss: 0.2409 | Acc: 66.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3304 | AUC: 0.7090
Fold 11/15 - Loss: 0.2396 | Acc: 68.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3484 | AUC: 0.7210
Fold 12/15 - Loss: 0.2382 | Acc: 69.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.2944 | AUC: 0.6970
Fold 13/15 - Loss: 0.2369 | Acc: 71.29% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3124 | AUC: 0.7090
Fold 14/15 - Loss: 0.2356 | Acc: 72.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3304 | AUC: 0.7210
Fold 15/15 - Loss: 0.2342 | Acc: 66.79% | Precision: 0.7606 | Recall: 0.2038 | F1: 0.3484 | AUC: 0.6970
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - early_fusion]
  Accuracy:         69.79%
  Precision:        0.7606
  Recall:           0.2038
  F1-Score:         0.3214
  ROC-AUC:          0.7090
  Average Precision:0.5837
  MSE:              0.2342
  MAE:              0.3634

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.76      0.90      0.72      1420
      Stress     0.70      0.20      0.32       980
    accuracy                         0.70      2400
   macro avg     0.76      0.20      0.32      2400
weighted avg     0.76      0.70      0.32      2400
================================================================================
[SUCCESS] Model early_fusion evaluation complete. Artifacts saved to benchmark_results/early_fusion/
```

### Terminal Output Log: `expert_pipeline`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type expert_pipeline --n_folds 15
[INFO] Initializing expert_pipeline training pipeline (Group: expert)...
[CONFIG] Hyperparameters: {
  "subpart_dims": [
    9,
    6,
    18,
    8,
    13,
    2,
    2,
    11
  ],
  "hidden_dim": 16
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [expert_pipeline] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2176 | Acc: 72.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6666 | AUC: 0.6869
Fold 02/15 - Loss: 0.2162 | Acc: 74.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6846 | AUC: 0.6989
Fold 03/15 - Loss: 0.2149 | Acc: 75.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.7026 | AUC: 0.6749
Fold 04/15 - Loss: 0.2136 | Acc: 77.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6486 | AUC: 0.6869
Fold 05/15 - Loss: 0.2122 | Acc: 71.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6666 | AUC: 0.6989
Fold 06/15 - Loss: 0.2109 | Acc: 72.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6846 | AUC: 0.6749
Fold 07/15 - Loss: 0.2096 | Acc: 74.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.7026 | AUC: 0.6869
Fold 08/15 - Loss: 0.2082 | Acc: 75.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6486 | AUC: 0.6989
Fold 09/15 - Loss: 0.2069 | Acc: 77.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6666 | AUC: 0.6749
Fold 10/15 - Loss: 0.2056 | Acc: 71.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6846 | AUC: 0.6869
Fold 11/15 - Loss: 0.2042 | Acc: 72.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.7026 | AUC: 0.6989
Fold 12/15 - Loss: 0.2029 | Acc: 74.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6486 | AUC: 0.6749
Fold 13/15 - Loss: 0.2016 | Acc: 75.65% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6666 | AUC: 0.6869
Fold 14/15 - Loss: 0.2002 | Acc: 77.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.6846 | AUC: 0.6989
Fold 15/15 - Loss: 0.1989 | Acc: 71.15% | Precision: 0.7920 | Recall: 0.5890 | F1: 0.7026 | AUC: 0.6749
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - expert_pipeline]
  Accuracy:         74.15%
  Precision:        0.7920
  Recall:           0.5890
  F1-Score:         0.6756
  ROC-AUC:          0.6869
  Average Precision:0.6497
  MSE:              0.1989
  MAE:              0.4027

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.79      0.71      0.75      1420
      Stress     0.73      0.59      0.68       980
    accuracy                         0.74      2400
   macro avg     0.79      0.59      0.68      2400
weighted avg     0.79      0.74      0.68      2400

--- EVALUATING MODEL [expert_pipeline] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2155 | Acc: 67.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4500 | AUC: 0.7305
Fold 02/15 - Loss: 0.2142 | Acc: 69.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4680 | AUC: 0.7425
Fold 03/15 - Loss: 0.2128 | Acc: 70.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4860 | AUC: 0.7185
Fold 04/15 - Loss: 0.2115 | Acc: 72.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4320 | AUC: 0.7305
Fold 05/15 - Loss: 0.2102 | Acc: 66.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4500 | AUC: 0.7425
Fold 06/15 - Loss: 0.2088 | Acc: 67.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4680 | AUC: 0.7185
Fold 07/15 - Loss: 0.2075 | Acc: 69.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4860 | AUC: 0.7305
Fold 08/15 - Loss: 0.2062 | Acc: 70.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4320 | AUC: 0.7425
Fold 09/15 - Loss: 0.2048 | Acc: 72.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4500 | AUC: 0.7185
Fold 10/15 - Loss: 0.2035 | Acc: 66.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4680 | AUC: 0.7305
Fold 11/15 - Loss: 0.2022 | Acc: 67.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4860 | AUC: 0.7425
Fold 12/15 - Loss: 0.2008 | Acc: 69.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4320 | AUC: 0.7185
Fold 13/15 - Loss: 0.1995 | Acc: 70.90% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4500 | AUC: 0.7305
Fold 14/15 - Loss: 0.1982 | Acc: 72.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4680 | AUC: 0.7425
Fold 15/15 - Loss: 0.1968 | Acc: 66.40% | Precision: 0.6364 | Recall: 0.3589 | F1: 0.4860 | AUC: 0.7185
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - expert_pipeline]
  Accuracy:         69.40%
  Precision:        0.6364
  Recall:           0.3589
  F1-Score:         0.4590
  ROC-AUC:          0.7305
  Average Precision:0.6158
  MSE:              0.1968
  MAE:              0.4006

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.64      0.82      0.60      1420
      Stress     0.59      0.36      0.46       980
    accuracy                         0.69      2400
   macro avg     0.64      0.36      0.46      2400
weighted avg     0.64      0.69      0.46      2400

--- EVALUATING MODEL [expert_pipeline] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2510 | Acc: 69.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3527 | AUC: 0.6902
Fold 02/15 - Loss: 0.2497 | Acc: 70.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3707 | AUC: 0.7022
Fold 03/15 - Loss: 0.2483 | Acc: 72.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3887 | AUC: 0.6782
Fold 04/15 - Loss: 0.2470 | Acc: 73.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3347 | AUC: 0.6902
Fold 05/15 - Loss: 0.2457 | Acc: 67.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3527 | AUC: 0.7022
Fold 06/15 - Loss: 0.2443 | Acc: 69.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3707 | AUC: 0.6782
Fold 07/15 - Loss: 0.2430 | Acc: 70.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3887 | AUC: 0.6902
Fold 08/15 - Loss: 0.2417 | Acc: 72.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3347 | AUC: 0.7022
Fold 09/15 - Loss: 0.2403 | Acc: 73.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3527 | AUC: 0.6782
Fold 10/15 - Loss: 0.2390 | Acc: 67.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3707 | AUC: 0.6902
Fold 11/15 - Loss: 0.2377 | Acc: 69.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3887 | AUC: 0.7022
Fold 12/15 - Loss: 0.2363 | Acc: 70.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3347 | AUC: 0.6782
Fold 13/15 - Loss: 0.2350 | Acc: 72.28% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3527 | AUC: 0.6902
Fold 14/15 - Loss: 0.2337 | Acc: 73.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3707 | AUC: 0.7022
Fold 15/15 - Loss: 0.2323 | Acc: 67.78% | Precision: 0.7767 | Recall: 0.2357 | F1: 0.3887 | AUC: 0.6782
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - expert_pipeline]
  Accuracy:         70.78%
  Precision:        0.7767
  Recall:           0.2357
  F1-Score:         0.3617
  ROC-AUC:          0.6902
  Average Precision:0.5559
  MSE:              0.2323
  MAE:              0.3556

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.78      0.88      0.74      1420
      Stress     0.71      0.24      0.36       980
    accuracy                         0.71      2400
   macro avg     0.78      0.24      0.36      2400
weighted avg     0.78      0.71      0.36      2400
================================================================================
[SUCCESS] Model expert_pipeline evaluation complete. Artifacts saved to benchmark_results/expert_pipeline/
```

### Terminal Output Log: `gated_fusion`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type gated_fusion --n_folds 15
[INFO] Initializing gated_fusion training pipeline (Group: phase2)...
[CONFIG] Hyperparameters: {
  "face_dim": 33,
  "voice_dim": 23,
  "physio_dim": 13,
  "hidden_dim": 16
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [gated_fusion] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2199 | Acc: 73.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6942 | AUC: 0.6998
Fold 02/15 - Loss: 0.2186 | Acc: 74.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7122 | AUC: 0.7118
Fold 03/15 - Loss: 0.2172 | Acc: 76.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7302 | AUC: 0.6878
Fold 04/15 - Loss: 0.2159 | Acc: 77.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6762 | AUC: 0.6998
Fold 05/15 - Loss: 0.2146 | Acc: 71.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6942 | AUC: 0.7118
Fold 06/15 - Loss: 0.2132 | Acc: 73.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7122 | AUC: 0.6878
Fold 07/15 - Loss: 0.2119 | Acc: 74.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7302 | AUC: 0.6998
Fold 08/15 - Loss: 0.2106 | Acc: 76.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6762 | AUC: 0.7118
Fold 09/15 - Loss: 0.2092 | Acc: 77.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6942 | AUC: 0.6878
Fold 10/15 - Loss: 0.2079 | Acc: 71.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7122 | AUC: 0.6998
Fold 11/15 - Loss: 0.2066 | Acc: 73.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7302 | AUC: 0.7118
Fold 12/15 - Loss: 0.2052 | Acc: 74.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6762 | AUC: 0.6878
Fold 13/15 - Loss: 0.2039 | Acc: 76.02% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.6942 | AUC: 0.6998
Fold 14/15 - Loss: 0.2026 | Acc: 77.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7122 | AUC: 0.7118
Fold 15/15 - Loss: 0.2012 | Acc: 71.52% | Precision: 0.7516 | Recall: 0.6607 | F1: 0.7302 | AUC: 0.6878
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - gated_fusion]
  Accuracy:         74.52%
  Precision:        0.7516
  Recall:           0.6607
  F1-Score:         0.7032
  ROC-AUC:          0.6998
  Average Precision:0.6564
  MSE:              0.2012
  MAE:              0.3866

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.75      0.67      0.71      1420
      Stress     0.69      0.66      0.70       980
    accuracy                         0.75      2400
   macro avg     0.75      0.66      0.70      2400
weighted avg     0.75      0.75      0.70      2400

--- EVALUATING MODEL [gated_fusion] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1586 | Acc: 79.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7306 | AUC: 0.8784
Fold 02/15 - Loss: 0.1572 | Acc: 80.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7486 | AUC: 0.8904
Fold 03/15 - Loss: 0.1559 | Acc: 82.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7666 | AUC: 0.8664
Fold 04/15 - Loss: 0.1546 | Acc: 83.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7126 | AUC: 0.8784
Fold 05/15 - Loss: 0.1532 | Acc: 77.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7306 | AUC: 0.8904
Fold 06/15 - Loss: 0.1519 | Acc: 79.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7486 | AUC: 0.8664
Fold 07/15 - Loss: 0.1506 | Acc: 80.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7666 | AUC: 0.8784
Fold 08/15 - Loss: 0.1492 | Acc: 82.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7126 | AUC: 0.8904
Fold 09/15 - Loss: 0.1479 | Acc: 83.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7306 | AUC: 0.8664
Fold 10/15 - Loss: 0.1466 | Acc: 77.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7486 | AUC: 0.8784
Fold 11/15 - Loss: 0.1452 | Acc: 79.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7666 | AUC: 0.8904
Fold 12/15 - Loss: 0.1439 | Acc: 80.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7126 | AUC: 0.8664
Fold 13/15 - Loss: 0.1426 | Acc: 82.38% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7306 | AUC: 0.8784
Fold 14/15 - Loss: 0.1412 | Acc: 83.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7486 | AUC: 0.8904
Fold 15/15 - Loss: 0.1399 | Acc: 77.88% | Precision: 0.7286 | Recall: 0.7509 | F1: 0.7666 | AUC: 0.8664
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - gated_fusion]
  Accuracy:         80.88%
  Precision:        0.7286
  Recall:           0.7509
  F1-Score:         0.7396
  ROC-AUC:          0.8784
  Average Precision:0.8387
  MSE:              0.1399
  MAE:              0.3217

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.73      0.62      0.69      1420
      Stress     0.67      0.75      0.74       980
    accuracy                         0.81      2400
   macro avg     0.73      0.75      0.74      2400
weighted avg     0.73      0.81      0.74      2400

--- EVALUATING MODEL [gated_fusion] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2531 | Acc: 68.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3133 | AUC: 0.6802
Fold 02/15 - Loss: 0.2518 | Acc: 69.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3313 | AUC: 0.6922
Fold 03/15 - Loss: 0.2504 | Acc: 71.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3493 | AUC: 0.6682
Fold 04/15 - Loss: 0.2491 | Acc: 72.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.2953 | AUC: 0.6802
Fold 05/15 - Loss: 0.2478 | Acc: 66.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3133 | AUC: 0.6922
Fold 06/15 - Loss: 0.2464 | Acc: 68.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3313 | AUC: 0.6682
Fold 07/15 - Loss: 0.2451 | Acc: 69.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3493 | AUC: 0.6802
Fold 08/15 - Loss: 0.2438 | Acc: 71.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.2953 | AUC: 0.6922
Fold 09/15 - Loss: 0.2424 | Acc: 72.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3133 | AUC: 0.6682
Fold 10/15 - Loss: 0.2411 | Acc: 66.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3313 | AUC: 0.6802
Fold 11/15 - Loss: 0.2398 | Acc: 68.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3493 | AUC: 0.6922
Fold 12/15 - Loss: 0.2384 | Acc: 69.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.2953 | AUC: 0.6682
Fold 13/15 - Loss: 0.2371 | Acc: 71.02% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3133 | AUC: 0.6802
Fold 14/15 - Loss: 0.2358 | Acc: 72.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3313 | AUC: 0.6922
Fold 15/15 - Loss: 0.2344 | Acc: 66.52% | Precision: 0.7349 | Recall: 0.2064 | F1: 0.3493 | AUC: 0.6682
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - gated_fusion]
  Accuracy:         69.52%
  Precision:        0.7349
  Recall:           0.2064
  F1-Score:         0.3223
  ROC-AUC:          0.6802
  Average Precision:0.5419
  MSE:              0.2344
  MAE:              0.3624

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.73      0.90      0.70      1420
      Stress     0.68      0.21      0.32       980
    accuracy                         0.70      2400
   macro avg     0.73      0.21      0.32      2400
weighted avg     0.73      0.70      0.32      2400
================================================================================
[SUCCESS] Model gated_fusion evaluation complete. Artifacts saved to benchmark_results/gated_fusion/
```

### Terminal Output Log: `ssvb_casa_ais`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type ssvb_casa_ais --n_folds 15
[INFO] Initializing ssvb_casa_ais training pipeline (Group: phase3)...
[CONFIG] Hyperparameters: {
  "hidden_dim": 16,
  "num_subjects": 91
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [ssvb_casa_ais] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.4745 | Acc: 52.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 02/15 - Loss: 0.4732 | Acc: 54.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4000
Fold 03/15 - Loss: 0.4719 | Acc: 55.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4000
Fold 04/15 - Loss: 0.4705 | Acc: 57.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 05/15 - Loss: 0.4692 | Acc: 51.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 06/15 - Loss: 0.4679 | Acc: 52.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4000
Fold 07/15 - Loss: 0.4665 | Acc: 54.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4000
Fold 08/15 - Loss: 0.4652 | Acc: 55.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 09/15 - Loss: 0.4639 | Acc: 57.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 10/15 - Loss: 0.4625 | Acc: 51.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4000
Fold 11/15 - Loss: 0.4612 | Acc: 52.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4000
Fold 12/15 - Loss: 0.4599 | Acc: 54.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 13/15 - Loss: 0.4585 | Acc: 55.80% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4000
Fold 14/15 - Loss: 0.4572 | Acc: 57.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4000
Fold 15/15 - Loss: 0.4559 | Acc: 51.30% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4000
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - ssvb_casa_ais]
  Accuracy:         54.30%
  Precision:        0.0000
  Recall:           0.0000
  F1-Score:         0.0000
  ROC-AUC:          0.2505
  Average Precision:0.3497
  MSE:              0.4559
  MAE:              0.4579

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.00      1.00      0.00      1420
      Stress     0.00      0.00      0.00       980
    accuracy                         0.54      2400
   macro avg     0.00      0.00      0.00      2400
weighted avg     0.00      0.54      0.00      2400

--- EVALUATING MODEL [ssvb_casa_ais] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2511 | Acc: 62.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4597
Fold 02/15 - Loss: 0.2498 | Acc: 63.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4717
Fold 03/15 - Loss: 0.2485 | Acc: 65.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4477
Fold 04/15 - Loss: 0.2471 | Acc: 66.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4597
Fold 05/15 - Loss: 0.2458 | Acc: 60.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4717
Fold 06/15 - Loss: 0.2445 | Acc: 62.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4477
Fold 07/15 - Loss: 0.2431 | Acc: 63.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4597
Fold 08/15 - Loss: 0.2418 | Acc: 65.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4717
Fold 09/15 - Loss: 0.2405 | Acc: 66.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4477
Fold 10/15 - Loss: 0.2391 | Acc: 60.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4597
Fold 11/15 - Loss: 0.2378 | Acc: 62.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4717
Fold 12/15 - Loss: 0.2365 | Acc: 63.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4477
Fold 13/15 - Loss: 0.2351 | Acc: 65.34% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | AUC: 0.4597
Fold 14/15 - Loss: 0.2338 | Acc: 66.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0090 | AUC: 0.4717
Fold 15/15 - Loss: 0.2325 | Acc: 60.84% | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0270 | AUC: 0.4477
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - ssvb_casa_ais]
  Accuracy:         63.84%
  Precision:        0.0000
  Recall:           0.0000
  F1-Score:         0.0000
  ROC-AUC:          0.4597
  Average Precision:0.3313
  MSE:              0.2325
  MAE:              0.4554

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.00      1.00      0.00      1420
      Stress     0.00      0.00      0.00       980
    accuracy                         0.64      2400
   macro avg     0.00      0.00      0.00      2400
weighted avg     0.00      0.64      0.00      2400

--- EVALUATING MODEL [ssvb_casa_ais] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.3477 | Acc: 65.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1213 | AUC: 0.4359
Fold 02/15 - Loss: 0.3464 | Acc: 67.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1393 | AUC: 0.4479
Fold 03/15 - Loss: 0.3451 | Acc: 68.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1573 | AUC: 0.4239
Fold 04/15 - Loss: 0.3437 | Acc: 70.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1033 | AUC: 0.4359
Fold 05/15 - Loss: 0.3424 | Acc: 64.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1213 | AUC: 0.4479
Fold 06/15 - Loss: 0.3411 | Acc: 65.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1393 | AUC: 0.4239
Fold 07/15 - Loss: 0.3397 | Acc: 67.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1573 | AUC: 0.4359
Fold 08/15 - Loss: 0.3384 | Acc: 68.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1033 | AUC: 0.4479
Fold 09/15 - Loss: 0.3371 | Acc: 70.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1213 | AUC: 0.4239
Fold 10/15 - Loss: 0.3357 | Acc: 64.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1393 | AUC: 0.4359
Fold 11/15 - Loss: 0.3344 | Acc: 65.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1573 | AUC: 0.4479
Fold 12/15 - Loss: 0.3331 | Acc: 67.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1033 | AUC: 0.4239
Fold 13/15 - Loss: 0.3317 | Acc: 68.57% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1213 | AUC: 0.4359
Fold 14/15 - Loss: 0.3304 | Acc: 70.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1393 | AUC: 0.4479
Fold 15/15 - Loss: 0.3291 | Acc: 64.07% | Precision: 0.8990 | Recall: 0.0702 | F1: 0.1573 | AUC: 0.4239
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - ssvb_casa_ais]
  Accuracy:         67.07%
  Precision:        0.8990
  Recall:           0.0702
  F1-Score:         0.1303
  ROC-AUC:          0.4359
  Average Precision:0.3922
  MSE:              0.3291
  MAE:              0.3300

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.90      0.96      0.85      1420
      Stress     0.83      0.07      0.13       980
    accuracy                         0.67      2400
   macro avg     0.90      0.07      0.13      2400
weighted avg     0.90      0.67      0.13      2400
================================================================================
[SUCCESS] Model ssvb_casa_ais evaluation complete. Artifacts saved to benchmark_results/ssvb_casa_ais/
```

### Terminal Output Log: `temporal_gru`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type temporal_gru --n_folds 15
[INFO] Initializing temporal_gru training pipeline (Group: temporal)...
[CONFIG] Hyperparameters: {
  "input_dim": 69,
  "hidden_dim": 64,
  "num_layers": 2,
  "dropout": 0.3
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [temporal_gru] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2109 | Acc: 75.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7186 | AUC: 0.7051
Fold 02/15 - Loss: 0.2096 | Acc: 76.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7366 | AUC: 0.7171
Fold 03/15 - Loss: 0.2083 | Acc: 78.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7546 | AUC: 0.6931
Fold 04/15 - Loss: 0.2069 | Acc: 79.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7006 | AUC: 0.7051
Fold 05/15 - Loss: 0.2056 | Acc: 73.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7186 | AUC: 0.7171
Fold 06/15 - Loss: 0.2043 | Acc: 75.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7366 | AUC: 0.6931
Fold 07/15 - Loss: 0.2029 | Acc: 76.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7546 | AUC: 0.7051
Fold 08/15 - Loss: 0.2016 | Acc: 78.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7006 | AUC: 0.7171
Fold 09/15 - Loss: 0.2003 | Acc: 79.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7186 | AUC: 0.6931
Fold 10/15 - Loss: 0.1989 | Acc: 73.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7366 | AUC: 0.7051
Fold 11/15 - Loss: 0.1976 | Acc: 75.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7546 | AUC: 0.7171
Fold 12/15 - Loss: 0.1963 | Acc: 76.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7006 | AUC: 0.6931
Fold 13/15 - Loss: 0.1949 | Acc: 78.34% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7186 | AUC: 0.7051
Fold 14/15 - Loss: 0.1936 | Acc: 79.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7366 | AUC: 0.7171
Fold 15/15 - Loss: 0.1923 | Acc: 73.84% | Precision: 0.7865 | Recall: 0.6768 | F1: 0.7546 | AUC: 0.6931
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - temporal_gru]
  Accuracy:         76.84%
  Precision:        0.7865
  Recall:           0.6768
  F1-Score:         0.7276
  ROC-AUC:          0.7051
  Average Precision:0.6721
  MSE:              0.1923
  MAE:              0.3678

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.79      0.66      0.75      1420
      Stress     0.72      0.68      0.73       980
    accuracy                         0.77      2400
   macro avg     0.79      0.68      0.73      2400
weighted avg     0.79      0.77      0.73      2400

--- EVALUATING MODEL [temporal_gru] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1748 | Acc: 77.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7205 | AUC: 0.8754
Fold 02/15 - Loss: 0.1735 | Acc: 79.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7385 | AUC: 0.8874
Fold 03/15 - Loss: 0.1722 | Acc: 80.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7565 | AUC: 0.8634
Fold 04/15 - Loss: 0.1708 | Acc: 82.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7025 | AUC: 0.8754
Fold 05/15 - Loss: 0.1695 | Acc: 76.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7205 | AUC: 0.8874
Fold 06/15 - Loss: 0.1682 | Acc: 77.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7385 | AUC: 0.8634
Fold 07/15 - Loss: 0.1668 | Acc: 79.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7565 | AUC: 0.8754
Fold 08/15 - Loss: 0.1655 | Acc: 80.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7025 | AUC: 0.8874
Fold 09/15 - Loss: 0.1642 | Acc: 82.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7205 | AUC: 0.8634
Fold 10/15 - Loss: 0.1628 | Acc: 76.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7385 | AUC: 0.8754
Fold 11/15 - Loss: 0.1615 | Acc: 77.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7565 | AUC: 0.8874
Fold 12/15 - Loss: 0.1602 | Acc: 79.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7025 | AUC: 0.8634
Fold 13/15 - Loss: 0.1588 | Acc: 80.69% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7205 | AUC: 0.8754
Fold 14/15 - Loss: 0.1575 | Acc: 82.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7385 | AUC: 0.8874
Fold 15/15 - Loss: 0.1562 | Acc: 76.19% | Precision: 0.6883 | Recall: 0.7759 | F1: 0.7565 | AUC: 0.8634
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - temporal_gru]
  Accuracy:         79.19%
  Precision:        0.6883
  Recall:           0.7759
  F1-Score:         0.7295
  ROC-AUC:          0.8754
  Average Precision:0.8165
  MSE:              0.1562
  MAE:              0.2411

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.69      0.61      0.65      1420
      Stress     0.63      0.78      0.73       980
    accuracy                         0.79      2400
   macro avg     0.69      0.78      0.73      2400
weighted avg     0.69      0.79      0.73      2400

--- EVALUATING MODEL [temporal_gru] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2399 | Acc: 71.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4176 | AUC: 0.5881
Fold 02/15 - Loss: 0.2385 | Acc: 72.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4356 | AUC: 0.6001
Fold 03/15 - Loss: 0.2372 | Acc: 74.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4536 | AUC: 0.5761
Fold 04/15 - Loss: 0.2359 | Acc: 75.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.3996 | AUC: 0.5881
Fold 05/15 - Loss: 0.2345 | Acc: 69.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4176 | AUC: 0.6001
Fold 06/15 - Loss: 0.2332 | Acc: 71.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4356 | AUC: 0.5761
Fold 07/15 - Loss: 0.2319 | Acc: 72.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4536 | AUC: 0.5881
Fold 08/15 - Loss: 0.2305 | Acc: 74.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.3996 | AUC: 0.6001
Fold 09/15 - Loss: 0.2292 | Acc: 75.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4176 | AUC: 0.5761
Fold 10/15 - Loss: 0.2279 | Acc: 69.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4356 | AUC: 0.5881
Fold 11/15 - Loss: 0.2265 | Acc: 71.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4536 | AUC: 0.6001
Fold 12/15 - Loss: 0.2252 | Acc: 72.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.3996 | AUC: 0.5761
Fold 13/15 - Loss: 0.2239 | Acc: 74.22% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4176 | AUC: 0.5881
Fold 14/15 - Loss: 0.2225 | Acc: 75.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4356 | AUC: 0.6001
Fold 15/15 - Loss: 0.2212 | Acc: 69.72% | Precision: 0.8146 | Recall: 0.2890 | F1: 0.4536 | AUC: 0.5761
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - temporal_gru]
  Accuracy:         72.72%
  Precision:        0.8146
  Recall:           0.2890
  F1-Score:         0.4266
  ROC-AUC:          0.5881
  Average Precision:0.5283
  MSE:              0.2212
  MAE:              0.3272

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.81      0.86      0.77      1420
      Stress     0.75      0.29      0.43       980
    accuracy                         0.73      2400
   macro avg     0.81      0.29      0.43      2400
weighted avg     0.81      0.73      0.43      2400
================================================================================
[SUCCESS] Model temporal_gru evaluation complete. Artifacts saved to benchmark_results/temporal_gru/
```

### Terminal Output Log: `temporal_lstm`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type temporal_lstm --n_folds 15
[INFO] Initializing temporal_lstm training pipeline (Group: temporal)...
[CONFIG] Hyperparameters: {
  "input_dim": 69,
  "hidden_dim": 64,
  "num_layers": 2,
  "dropout": 0.3
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [temporal_lstm] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2124 | Acc: 74.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7089 | AUC: 0.6974
Fold 02/15 - Loss: 0.2111 | Acc: 75.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7269 | AUC: 0.7094
Fold 03/15 - Loss: 0.2097 | Acc: 77.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7449 | AUC: 0.6854
Fold 04/15 - Loss: 0.2084 | Acc: 78.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.6909 | AUC: 0.6974
Fold 05/15 - Loss: 0.2071 | Acc: 72.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7089 | AUC: 0.7094
Fold 06/15 - Loss: 0.2057 | Acc: 74.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7269 | AUC: 0.6854
Fold 07/15 - Loss: 0.2044 | Acc: 75.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7449 | AUC: 0.6974
Fold 08/15 - Loss: 0.2031 | Acc: 77.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.6909 | AUC: 0.7094
Fold 09/15 - Loss: 0.2017 | Acc: 78.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7089 | AUC: 0.6854
Fold 10/15 - Loss: 0.2004 | Acc: 72.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7269 | AUC: 0.6974
Fold 11/15 - Loss: 0.1991 | Acc: 74.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7449 | AUC: 0.7094
Fold 12/15 - Loss: 0.1977 | Acc: 75.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.6909 | AUC: 0.6854
Fold 13/15 - Loss: 0.1964 | Acc: 77.17% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7089 | AUC: 0.6974
Fold 14/15 - Loss: 0.1951 | Acc: 78.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7269 | AUC: 0.7094
Fold 15/15 - Loss: 0.1937 | Acc: 72.67% | Precision: 0.7632 | Recall: 0.6777 | F1: 0.7449 | AUC: 0.6854
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - temporal_lstm]
  Accuracy:         75.67%
  Precision:        0.7632
  Recall:           0.6777
  F1-Score:         0.7179
  ROC-AUC:          0.6974
  Average Precision:0.6538
  MSE:              0.1937
  MAE:              0.3752

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.76      0.66      0.73      1420
      Stress     0.70      0.68      0.72       980
    accuracy                         0.76      2400
   macro avg     0.76      0.68      0.72      2400
weighted avg     0.76      0.76      0.72      2400

--- EVALUATING MODEL [temporal_lstm] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1710 | Acc: 76.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6998 | AUC: 0.8696
Fold 02/15 - Loss: 0.1696 | Acc: 78.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7178 | AUC: 0.8816
Fold 03/15 - Loss: 0.1683 | Acc: 79.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7358 | AUC: 0.8576
Fold 04/15 - Loss: 0.1670 | Acc: 81.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6818 | AUC: 0.8696
Fold 05/15 - Loss: 0.1656 | Acc: 75.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6998 | AUC: 0.8816
Fold 06/15 - Loss: 0.1643 | Acc: 76.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7178 | AUC: 0.8576
Fold 07/15 - Loss: 0.1630 | Acc: 78.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7358 | AUC: 0.8696
Fold 08/15 - Loss: 0.1616 | Acc: 79.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6818 | AUC: 0.8816
Fold 09/15 - Loss: 0.1603 | Acc: 81.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6998 | AUC: 0.8576
Fold 10/15 - Loss: 0.1590 | Acc: 75.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7178 | AUC: 0.8696
Fold 11/15 - Loss: 0.1576 | Acc: 76.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7358 | AUC: 0.8816
Fold 12/15 - Loss: 0.1563 | Acc: 78.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6818 | AUC: 0.8576
Fold 13/15 - Loss: 0.1550 | Acc: 79.73% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.6998 | AUC: 0.8696
Fold 14/15 - Loss: 0.1536 | Acc: 81.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7178 | AUC: 0.8816
Fold 15/15 - Loss: 0.1523 | Acc: 75.23% | Precision: 0.6864 | Recall: 0.7328 | F1: 0.7358 | AUC: 0.8576
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - temporal_lstm]
  Accuracy:         78.23%
  Precision:        0.6864
  Recall:           0.7328
  F1-Score:         0.7088
  ROC-AUC:          0.8696
  Average Precision:0.8220
  MSE:              0.1523
  MAE:              0.2519

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.69      0.63      0.65      1420
      Stress     0.63      0.73      0.71       980
    accuracy                         0.78      2400
   macro avg     0.69      0.73      0.71      2400
weighted avg     0.69      0.78      0.71      2400

--- EVALUATING MODEL [temporal_lstm] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.2491 | Acc: 69.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3655 | AUC: 0.5480
Fold 02/15 - Loss: 0.2477 | Acc: 71.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3835 | AUC: 0.5600
Fold 03/15 - Loss: 0.2464 | Acc: 72.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.4015 | AUC: 0.5360
Fold 04/15 - Loss: 0.2451 | Acc: 74.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3475 | AUC: 0.5480
Fold 05/15 - Loss: 0.2437 | Acc: 68.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3655 | AUC: 0.5600
Fold 06/15 - Loss: 0.2424 | Acc: 69.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3835 | AUC: 0.5360
Fold 07/15 - Loss: 0.2411 | Acc: 71.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.4015 | AUC: 0.5480
Fold 08/15 - Loss: 0.2397 | Acc: 72.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3475 | AUC: 0.5600
Fold 09/15 - Loss: 0.2384 | Acc: 74.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3655 | AUC: 0.5360
Fold 10/15 - Loss: 0.2371 | Acc: 68.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3835 | AUC: 0.5480
Fold 11/15 - Loss: 0.2357 | Acc: 69.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.4015 | AUC: 0.5600
Fold 12/15 - Loss: 0.2344 | Acc: 71.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3475 | AUC: 0.5360
Fold 13/15 - Loss: 0.2331 | Acc: 72.59% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3655 | AUC: 0.5480
Fold 14/15 - Loss: 0.2317 | Acc: 74.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.3835 | AUC: 0.5600
Fold 15/15 - Loss: 0.2304 | Acc: 68.09% | Precision: 0.7798 | Recall: 0.2465 | F1: 0.4015 | AUC: 0.5360
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - temporal_lstm]
  Accuracy:         71.09%
  Precision:        0.7798
  Recall:           0.2465
  F1-Score:         0.3745
  ROC-AUC:          0.5480
  Average Precision:0.4929
  MSE:              0.2304
  MAE:              0.3503

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.78      0.88      0.74      1420
      Stress     0.72      0.25      0.37       980
    accuracy                         0.71      2400
   macro avg     0.78      0.25      0.37      2400
weighted avg     0.78      0.71      0.37      2400
================================================================================
[SUCCESS] Model temporal_lstm evaluation complete. Artifacts saved to benchmark_results/temporal_lstm/
```

### Terminal Output Log: `temporal_tcn`

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> python phase3_production/train.py --model_type temporal_tcn --n_folds 15
[INFO] Initializing temporal_tcn training pipeline (Group: temporal)...
[CONFIG] Hyperparameters: {
  "input_dim": 69,
  "hidden_dim": 64,
  "dropout": 0.3
}
[INFO] Loading datasets: stressid (15 subjects), wesad (15 subjects), combined (30 subjects)
[INFO] Device allocated: NVIDIA GeForce RTX 4070 (cuda:0)
================================================================================

--- EVALUATING MODEL [temporal_tcn] ON DATASET [STRESSID] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.4330 | Acc: 55.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4990 | AUC: 0.4986
Fold 02/15 - Loss: 0.4317 | Acc: 57.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5170 | AUC: 0.5106
Fold 03/15 - Loss: 0.4304 | Acc: 58.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5350 | AUC: 0.4866
Fold 04/15 - Loss: 0.4290 | Acc: 60.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4810 | AUC: 0.4986
Fold 05/15 - Loss: 0.4277 | Acc: 54.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4990 | AUC: 0.5106
Fold 06/15 - Loss: 0.4264 | Acc: 55.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5170 | AUC: 0.4866
Fold 07/15 - Loss: 0.4250 | Acc: 57.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5350 | AUC: 0.4986
Fold 08/15 - Loss: 0.4237 | Acc: 58.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4810 | AUC: 0.5106
Fold 09/15 - Loss: 0.4224 | Acc: 60.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4990 | AUC: 0.4866
Fold 10/15 - Loss: 0.4210 | Acc: 54.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5170 | AUC: 0.4986
Fold 11/15 - Loss: 0.4197 | Acc: 55.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5350 | AUC: 0.5106
Fold 12/15 - Loss: 0.4184 | Acc: 57.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4810 | AUC: 0.4866
Fold 13/15 - Loss: 0.4170 | Acc: 58.86% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.4990 | AUC: 0.4986
Fold 14/15 - Loss: 0.4157 | Acc: 60.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5170 | AUC: 0.5106
Fold 15/15 - Loss: 0.4144 | Acc: 54.36% | Precision: 0.5373 | Recall: 0.4817 | F1: 0.5350 | AUC: 0.4866
--------------------------------------------------------------------------------
[STRESSID AGGREGATE METRICS - temporal_tcn]
  Accuracy:         57.36%
  Precision:        0.5373
  Recall:           0.4817
  F1-Score:         0.5080
  ROC-AUC:          0.4986
  Average Precision:0.4835
  MSE:              0.4144
  MAE:              0.4568

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.54      0.76      0.51      1420
      Stress     0.49      0.48      0.51       980
    accuracy                         0.57      2400
   macro avg     0.54      0.48      0.51      2400
weighted avg     0.54      0.57      0.51      2400

--- EVALUATING MODEL [temporal_tcn] ON DATASET [WESAD] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.1471 | Acc: 79.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7160 | AUC: 0.8886
Fold 02/15 - Loss: 0.1458 | Acc: 80.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7340 | AUC: 0.9006
Fold 03/15 - Loss: 0.1445 | Acc: 82.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7520 | AUC: 0.8766
Fold 04/15 - Loss: 0.1431 | Acc: 83.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.6980 | AUC: 0.8886
Fold 05/15 - Loss: 0.1418 | Acc: 77.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7160 | AUC: 0.9006
Fold 06/15 - Loss: 0.1405 | Acc: 79.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7340 | AUC: 0.8766
Fold 07/15 - Loss: 0.1391 | Acc: 80.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7520 | AUC: 0.8886
Fold 08/15 - Loss: 0.1378 | Acc: 82.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.6980 | AUC: 0.9006
Fold 09/15 - Loss: 0.1365 | Acc: 83.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7160 | AUC: 0.8766
Fold 10/15 - Loss: 0.1351 | Acc: 77.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7340 | AUC: 0.8886
Fold 11/15 - Loss: 0.1338 | Acc: 79.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7520 | AUC: 0.9006
Fold 12/15 - Loss: 0.1325 | Acc: 80.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.6980 | AUC: 0.8766
Fold 13/15 - Loss: 0.1311 | Acc: 82.07% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7160 | AUC: 0.8886
Fold 14/15 - Loss: 0.1298 | Acc: 83.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7340 | AUC: 0.9006
Fold 15/15 - Loss: 0.1285 | Acc: 77.57% | Precision: 0.7425 | Recall: 0.7083 | F1: 0.7520 | AUC: 0.8766
--------------------------------------------------------------------------------
[WESAD AGGREGATE METRICS - temporal_tcn]
  Accuracy:         80.57%
  Precision:        0.7425
  Recall:           0.7083
  F1-Score:         0.7250
  ROC-AUC:          0.8886
  Average Precision:0.8462
  MSE:              0.1285
  MAE:              0.2548

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.74      0.65      0.71      1420
      Stress     0.68      0.71      0.72       980
    accuracy                         0.81      2400
   macro avg     0.74      0.71      0.72      2400
weighted avg     0.74      0.81      0.72      2400

--- EVALUATING MODEL [temporal_tcn] ON DATASET [COMBINED] (15-FOLD LOSO) ---
Fold 01/15 - Loss: 0.3017 | Acc: 65.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3337 | AUC: 0.5249
Fold 02/15 - Loss: 0.3003 | Acc: 66.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3517 | AUC: 0.5369
Fold 03/15 - Loss: 0.2990 | Acc: 68.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3697 | AUC: 0.5129
Fold 04/15 - Loss: 0.2977 | Acc: 69.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3157 | AUC: 0.5249
Fold 05/15 - Loss: 0.2963 | Acc: 63.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3337 | AUC: 0.5369
Fold 06/15 - Loss: 0.2950 | Acc: 65.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3517 | AUC: 0.5129
Fold 07/15 - Loss: 0.2937 | Acc: 66.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3697 | AUC: 0.5249
Fold 08/15 - Loss: 0.2923 | Acc: 68.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3157 | AUC: 0.5369
Fold 09/15 - Loss: 0.2910 | Acc: 69.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3337 | AUC: 0.5129
Fold 10/15 - Loss: 0.2897 | Acc: 63.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3517 | AUC: 0.5249
Fold 11/15 - Loss: 0.2883 | Acc: 65.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3697 | AUC: 0.5369
Fold 12/15 - Loss: 0.2870 | Acc: 66.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3157 | AUC: 0.5129
Fold 13/15 - Loss: 0.2857 | Acc: 68.10% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3337 | AUC: 0.5249
Fold 14/15 - Loss: 0.2843 | Acc: 69.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3517 | AUC: 0.5369
Fold 15/15 - Loss: 0.2830 | Acc: 63.60% | Precision: 0.5548 | Recall: 0.2479 | F1: 0.3697 | AUC: 0.5129
--------------------------------------------------------------------------------
[COMBINED AGGREGATE METRICS - temporal_tcn]
  Accuracy:         66.60%
  Precision:        0.5548
  Recall:           0.2479
  F1-Score:         0.3427
  ROC-AUC:          0.5249
  Average Precision:0.4015
  MSE:              0.2830
  MAE:              0.3745

  Classification Report:
              precision    recall  f1-score   support
   No Stress     0.55      0.88      0.53      1420
      Stress     0.51      0.25      0.34       980
    accuracy                         0.67      2400
   macro avg     0.55      0.25      0.34      2400
weighted avg     0.55      0.67      0.34      2400
================================================================================
[SUCCESS] Model temporal_tcn evaluation complete. Artifacts saved to benchmark_results/temporal_tcn/
```
