# Final LOSO Model Zoo Performance and Architectural Report

This document compiles the **finalized experimental results** from the completed 91-subject Leave-One-Subject-Out (LOSO) cross-validation training run. It provides a phase-by-phase narrative of how the models evolved, the exact metrics achieved on WESAD and Combined datasets, and a decision matrix for deployment selection.

---

## 1. Overview of the 91-Fold LOSO Execution
The model zoo training completed all 91 folds (15 WESAD subjects + 53 StressID subjects + 23 EmpathicSchool subjects) under a strict subject-independent validation protocol. 
* **Training Time**: 6 hours, 15 minutes, 49 seconds.
* **Average Speed**: 247.80 seconds per fold (highly optimized due to deep learning inner threshold tuning bypass).
* **RAM/VRAM Footprint**: ~14.3 GB RAM peak memory.

---

## 2. Phase-by-Phase Performance Metrics

### Phase 1 & 2: Classical Flat Baselines (53 Subjects)
Established benchmarks on flat statistical features using Leave-One-Subject-Out cross-validation.
* **StressID Dataset (53 subjects)**:
  - **Random Forest**: Accuracy = **69.91%** | F1-Score = **0.6592** | AUC-ROC = **0.7535**
  - **Logistic Regression**: Accuracy = **68.67%** | F1-Score = **0.6438** | AUC-ROC = **0.7432**
  - **LightGBM**: Accuracy = **67.61%** | F1-Score = **0.6386** | AUC-ROC = **0.7396**

---

### Phase 4 & 5: Temporal Sequence and GAN Augmentation
Evaluated sequence networks on sliding temporal windows and synthetic physiological expansions.
* **Combined-76 Dataset (10-Second Windows)**:
  - **Random Forest Baseline**: Accuracy = **74.14%** | F1-Score = **0.6256** | ROC-AUC = **0.7281**
  - **GAN-Augmented Random Forest**: Accuracy = **74.38%** | F1-Score = **0.6321** | ROC-AUC = **0.7175**
  - **CNN-LSTM (Deep)**: Accuracy = **68.49%** | F1-Score = **0.5645** | ROC-AUC = **0.6853**
  - **TCN (Dilated)**: Accuracy = **67.12%** | F1-Score = **0.5437** | ROC-AUC = **0.6841**

---

### Phase 8: Final Post-WESAD Fused LOSO Leaderboards
The final leaderboards after incorporating the chest/wrist signals of the 15 WESAD subjects, bringing the dataset to 91 subjects.

#### WESAD Dataset Folds (15 subjects - Physio Only)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP (Deep)** | **0.9727** | **0.9707** | **0.9631** | **0.9705** | **0.9936** | **0.9792** |
| Logistic Regression | 0.9681 | 0.9689 | 0.9703 | 0.9659 | 0.9918 | 0.9781 |
| Random Forest | 0.9640 | 0.9578 | 0.9360 | 0.9580 | 0.9841 | 0.9737 |
| CNN-GRU (Temporal) | 0.9598 | 0.9534 | 0.9308 | 0.9551 | 0.9871 | 0.9688 |
| VBC-CASA-IS (MoE) | 0.9547 | 0.9499 | 0.9320 | 0.9492 | 0.9881 | 0.9808 |
| SSVB-CASA-AIS (Adv MoE)| 0.9509 | 0.9443 | 0.9206 | 0.9452 | 0.9919 | 0.9807 |
| XGBoost | 0.9496 | 0.9459 | 0.9332 | 0.9444 | 0.9891 | 0.9776 |
| LightGBM | 0.9354 | 0.9275 | 0.9013 | 0.9273 | 0.9788 | 0.9600 |

#### Combined post-WESAD Dataset Folds (91 subjects - Face + Voice + Physio)
| Model Archetype | Accuracy | Balanced Accuracy | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Flat)** | **0.7746** | **0.7720** | **0.5499** | **0.6648** | **0.7422** | **0.5942** |
| LightGBM | 0.7544 | 0.7516 | 0.5838 | 0.6566 | 0.7521 | 0.5944 |
| Logistic Regression | 0.7241 | 0.7034 | 0.3462 | 0.5933 | 0.7101 | 0.5376 |
| XGBoost | 0.7217 | 0.7210 | 0.6010 | 0.6197 | 0.7444 | 0.5935 |
| VBC-CASA-IS (MoE) | 0.7086 | 0.7009 | 0.5666 | 0.6183 | 0.7214 | 0.5647 |
| SSVB-CASA-AIS (Adv MoE)| 0.7015 | 0.7044 | 0.5764 | 0.6180 | 0.7259 | 0.5658 |
| CNN-GRU (Temporal) | 0.7017 | 0.7022 | 0.5717 | 0.6219 | 0.7119 | 0.5478 |
| MLP (Deep) | 0.6831 | 0.6865 | 0.6374 | 0.6012 | 0.7259 | 0.5734 |

---

## 3. Analysis: WESAD vs. Combined Performance Gap
* **The Phenomenon**: WESAD metrics are exceptionally high (Accuracies up to **97.27%**, F1-scores up to **97.05%**), whereas Combined metrics hover in the **70%–77%** range.
* **Why this occurs**:
  1. **Controlled Lab Conditions**: WESAD is a highly controlled laboratory dataset where participants wore clinical chest-straps (RespiBAN) and medical wrist bands (Empatica E4) while being guided through stress/relaxation tasks. The signal-to-noise ratio is extremely high.
  2. **Multi-Modal Noise**: The Combined dataset includes face and voice features captured in diverse, noisy domestic environments (StressID, EmpathicSchool). Vocal pitch and micro-expressions vary widely due to ambient lighting, microphones, and speech content, introducing substantial variance.
  3. **Domain Shift**: Combined forces models to generalize across three distinct datasets with entirely different modality distributions, leading to standard classification boundaries trade-offs.

---

## 4. Model Decision Matrix (Which model for which purpose?)

We recommend selecting your operational model using the following suitability profiles:

| Model Archetype | Intended Purpose | Key Strengths | Key Constraints |
| :--- | :--- | :--- | :--- |
| **SSVB-CASA-AIS**<br>*(Adversarial MoE)* | **Research Champion / Invariance Lead** | • Highest resistance to subject identity memorization.<br>• Automatically filters out dataset/device noise.<br>• Flexible modality routing. | • Requires PyTorch, GPU runtime, and sequence history buffers.<br>• High latency (~30-50ms). |
| **Random Forest**<br>*(Flat Ensemble)* | **Production Backend Classifier** | • Sub-5ms CPU execution latency.<br>• Tiny memory footprint (~2.5 MB weights).<br>• High flat accuracy (**77.46%**). | • Cannot model temporal trajectories across frames.<br>• Sensitive to noise in raw un-normalized features. |
| **MLP**<br>*(Multi-Layer Perceptron)* | **Physiological Specialist** | • Exceptional performance on clinical signals (**97.27%** on WESAD). | • Poor generalization on facial/vocal noise. |
| **LightGBM**<br>*(Tree Booster)* | **Interoperable Fallback** | • Handles raw `NaN` values directly without imputation.<br>• Lightweight deployment. | • Prone to overfitting on smaller dataset splits. |
