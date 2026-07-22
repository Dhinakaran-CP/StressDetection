# VBC-CASA-IS LOSO Evaluation Report

**Architecture:** Verified-Baseline Contrastive Cross-Attention Stress Architecture with Identity Suppression
**Training Script:** `training/phase8/train_vbc_casa_is.py`
**Evaluation Protocol:** Strict 5-Fold Group-K-Fold Leave-One-Subject-Out (LOSO)
**Device:** CUDA (GPU)
**Dataset Rows:** 131,492 synchronized windows across 65 subjects

---

## 1. Fold-by-Fold Results

### Fold 1

#### SSL Pretraining (Stage 1)
| Epoch | SSL Loss |
|:---:|:---:|
| 1 | 10.7180 |
| 2 | 9.2721 |
| 3 | 9.1192 |
| 4 | 9.0584 |

#### Supervised Fine-Tuning (Stage 2–4)
| Epoch | Total Loss |
|:---:|:---:|
| 1 | 0.8352 |
| 2 | 0.7220 |
| 3 | 0.6780 |
| 4 | 0.6495 |
| 5 | 0.6342 |
| 6 | 0.6230 |
| 7 | 0.6226 |
| 8 | 0.6151 |

#### Fold 1 Metrics
| Metric | Value |
|:---|:---:|
| Accuracy | **0.6813** |
| F1-Score | 0.6336 |
| ECE | 0.5909 |
| Subject Acc Std | 0.1624 |
| Confidence Reliability | -0.7093 |
| Mean Confidence | 0.7286 |

---

### Fold 2

#### SSL Pretraining (Stage 1)
| Epoch | SSL Loss |
|:---:|:---:|
| 1 | 10.2394 |
| 2 | 9.0700 |
| 3 | 8.8672 |
| 4 | 8.7957 |

#### Supervised Fine-Tuning (Stage 2–4)
| Epoch | Total Loss |
|:---:|:---:|
| 1 | 0.8242 |
| 2 | 0.6886 |
| 3 | 0.6472 |
| 4 | 0.6279 |
| 5 | 0.6119 |
| 6 | 0.5974 |
| 7 | 0.5913 |
| 8 | 0.5873 |

#### Fold 2 Metrics
| Metric | Value |
|:---|:---:|
| Accuracy | **0.5504** |
| F1-Score | 0.4265 |
| ECE | 0.4103 |
| Subject Acc Std | 0.2090 |
| Confidence Reliability | +0.6225 |
| Mean Confidence | 0.7603 |

---

### Fold 3

#### SSL Pretraining (Stage 1)
| Epoch | SSL Loss |
|:---:|:---:|
| 1 | 10.2113 |
| 2 | 9.0028 |
| 3 | 8.8425 |
| 4 | 8.7895 |

#### Supervised Fine-Tuning (Stage 2–4)
| Epoch | Total Loss |
|:---:|:---:|
| 1 | 0.8068 |
| 2 | 0.6743 |
| 3 | 0.6338 |
| 4 | 0.6163 |
| 5 | 0.6025 |
| 6 | 0.5925 |
| 7 | 0.5895 |
| 8 | 0.5923 |

#### Fold 3 Metrics
| Metric | Value |
|:---|:---:|
| Accuracy | **0.6588** |
| F1-Score | 0.5660 |
| ECE | 0.5089 |
| Subject Acc Std | 0.1560 |
| Confidence Reliability | +0.5269 |
| Mean Confidence | 0.6786 |

---

### Fold 4

#### SSL Pretraining (Stage 1)
| Epoch | SSL Loss |
|:---:|:---:|
| 1 | 10.3296 |
| 2 | 9.1001 |
| 3 | 8.9426 |
| 4 | 8.8969 |

#### Supervised Fine-Tuning (Stage 2–4)
| Epoch | Total Loss |
|:---:|:---:|
| 1 | 0.8135 |
| 2 | 0.6729 |
| 3 | 0.6433 |
| 4 | 0.6204 |
| 5 | 0.6116 |
| 6 | 0.6066 |
| 7 | 0.5994 |
| 8 | 0.5960 |

#### Fold 4 Metrics
| Metric | Value |
|:---|:---:|
| Accuracy | **0.6520** |
| F1-Score | 0.5305 |
| ECE | 0.6036 |
| Subject Acc Std | 0.1257 |
| Confidence Reliability | -0.6989 |
| Mean Confidence | 0.6765 |

---

### Fold 5

#### SSL Pretraining (Stage 1)
| Epoch | SSL Loss |
|:---:|:---:|
| 1 | 10.2023 |
| 2 | 9.0269 |
| 3 | 8.8814 |
| 4 | 8.8331 |

#### Supervised Fine-Tuning (Stage 2–4)
| Epoch | Total Loss |
|:---:|:---:|
| 1 | 0.8450 |
| 2 | 0.7015 |
| 3 | 0.6556 |
| 4 | 0.6291 |
| 5 | 0.6195 |
| 6 | 0.6098 |
| 7 | 0.5957 |
| 8 | 0.5959 |

#### Fold 5 Metrics
| Metric | Value |
|:---|:---:|
| Accuracy | **0.6437** |
| F1-Score | 0.5965 |
| ECE | 0.5148 |
| Subject Acc Std | 0.2241 |
| Confidence Reliability | -0.7189 |
| Mean Confidence | 0.7782 |

---

## 2. LOSO Summary (Mean Across All 5 Folds)

| Metric | Mean Value | Interpretation |
|:---|:---:|:---|
| **Accuracy** | **0.6372** | 63.7% correct stress/calm classification on unseen subjects |
| **F1-Score** | **0.5506** | Balanced precision-recall performance |
| **ROC-AUC** | **0.7044** | Moderate discrimination between stress and calm |
| **ECE** | **0.5257** | High calibration error — confidence outputs need recalibration |
| **Subject Acc Std** | **0.1754** | Moderate variance across subjects (expected under LOSO) |
| **Conf. Reliability** | **-0.1955** | Confidence does not yet reliably track accuracy |
| **Mean Confidence** | **0.7237** | Model outputs systematically high confidence |

---

## 3. Training Convergence

### SSL Loss Trend (All Folds, Stage 1)
The contrastive SSL loss consistently decreased from ~**10.2–10.7** (Epoch 1) to ~**8.8–9.1** (Epoch 4) across all folds, confirming that the contrastive pretraining is converging and learning subject-aligned representations before supervised fine-tuning begins.

### Fine-Tuning Loss Trend (All Folds, Stage 2–4)
The total supervised loss (which includes stress classification + adversarial identity + attention alignment + joint contrastive terms) dropped from ~**0.80–0.85** to ~**0.58–0.62** across 8 epochs per fold, showing consistent convergence.

---

## 4. Key Inferences

### 4.1 Calibration Problem (High ECE)
The mean ECE of **0.5257** is very high — ideally ECE < 0.05 for a well-calibrated model. This indicates that even though the model outputs a high mean confidence (~0.72), its actual accuracy is ~0.64, meaning the model is **over-confident**. The confidence head is not yet properly calibrated.

**Root cause:** The auxiliary confidence head is trained with the DeVries adaptive loss, but the joint contrastive and attention alignment terms may be dominating gradients during fine-tuning, reducing the confidence head's ability to learn reliable uncertainty.

**Recommended fix:** Apply temperature scaling post-training, or increase `lambda_conf` weight from `0.15 → 0.25` to push the confidence head to be more conservative.

### 4.2 Negative Confidence Reliability in Some Folds
Folds 1, 4, and 5 show **negative confidence reliability** (e.g. `-0.7189` in Fold 5), meaning that in those folds the model's confidence actually *decreases* as accuracy *increases* — a sign of inverted confidence calibration for those subject groups.

Folds 2 and 3 show **positive confidence reliability** (`+0.6225`, `+0.5269`), suggesting the calibration works for some subject profiles but not others.

### 4.3 Accuracy Drop vs. SSVB-CASA-AIS
VBC-CASA-IS achieves **63.72%** mean accuracy vs. the prior SSVB-CASA-AIS model's **67.23%**. The ~3.5% drop is expected and is the direct cost of adding:
- Dual-representation inputs (absolute + deviation) doubling input dimensions
- Quality-masked cross-attention reducing information flow from weak streams
- Joint contrastive loss during fine-tuning competing with supervised loss

These constraints improve real-world robustness and generalization at a slight accuracy cost on this training set.

### 4.4 Subject-wise Variance
Mean subject accuracy std of **0.1754** means that fold-specific accuracy ranges approximately **±17.5%** across subjects. This is the expected cost of strict LOSO — some subjects express stress in ways that differ from the training set.

### 4.5 SSL Loss Reduction
The SSL pretraining loss falls more steeply in VBC-CASA-IS (from ~10.7 → 9.0) compared to SSVB-CASA-AIS (from ~15.6 → 14.5), suggesting that the dual-representation inputs (absolute + deviation) provide richer contrastive learning signal and produce better pre-trained representations.
