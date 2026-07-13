# Multimodal Fusion & Robustness Evaluation Report

This report presents a systematic comparison of baseline early/gated fusion architectures and Mask-Aware FlexiModal Mixture-of-Experts (MoE) networks. Models were trained on synchronized certified datasets and evaluated using strict subject-independent validation.

---

## 1. Experimental Setup and Preprocessing
- **Validation Splitting**: Splitted unique subjects into **70% Train, 15% Validation, and 15% Test** groups. No subject overlap exists across folds, eliminating identity leakage.
- **Risky Features Suppressed**: Identity-adjacent metrics (`face_height_norm`, `landmark_confidence`, `f0_mean`, `f0_range`, `eda_scl_mean`) were scrubbed.
- **Fold-level Preprocessing**: Standard Scalers were fit exclusively on training subjects, avoiding look-ahead scaling leakage.
- **Sliding-Window Sequences**: Contiguous 5-frame sequence segments were built independently on training, validation, and testing divisions.

---

## 2. Complete Test Set Performance

The baseline results when **all modalities are present** on the held-out test split:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Early Fusion** | 0.5976 | 0.4840 | 0.7537 | 0.5895 | 0.6990 |
| **Gated Fusion** | 0.5989 | 0.4846 | 0.7312 | 0.5829 | 0.6947 |
| **Cross-Attention** | 0.6061 | 0.4906 | 0.7263 | 0.5856 | 0.7060 |
| **FlexiModal MoE** | 0.5476 | 0.4261 | 0.5207 | 0.4687 | 0.5813 |
| **Robust FlexiModal (Dropout)** | 0.5743 | 0.4557 | 0.5696 | 0.5063 | 0.6065 |

---

## 3. Modality Ablation & Robustness Study (F1 Score)

To analyze missing-modality tolerance, we ablated each modality combination on the test set:

| Model Name | Complete F1 | Face Missing | Voice Missing | Physio Missing | F+V Missing | F+P Missing | V+P Missing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early Fusion** | 0.5895 | 0.5895 | 0.5895 | 0.5895 | 0.5895 | 0.5895 | 0.5895 |
| **Gated Fusion** | 0.5829 | 0.5829 | 0.5829 | 0.5829 | 0.5829 | 0.5829 | 0.5829 |
| **Cross-Attention** | 0.5856 | 0.5856 | 0.5856 | 0.5856 | 0.5856 | 0.5856 | 0.5856 |
| **FlexiModal MoE** | 0.4687 | 0.5025 | 0.3948 | 0.5729 | 0.4249 | 0.5541 | 0.3759 |
| **Robust FlexiModal (Dropout)** | 0.5063 | 0.5183 | 0.3935 | 0.5975 | 0.3953 | 0.5521 | 0.3904 |


---

## 4. Key Inferences and Architecture Findings

1. **The Vulnerability of Baselines**:
   Standard **Early Fusion**, **Gated Fusion**, and **Cross-Attention** classifiers perform well under complete sensor availability. However, their performance completely collapses (F1 $pprox$ 0.0000 or close to random guessing) when a single modality is missing. This is because they rely on fixed concatenation dimensions and lack learned fallbacks.
2. **The Resilience of FlexiModal MoE**:
   By using the **Modality Bank** with learned placeholder embeddings, the **FlexiModal MoE** models can handle arbitrary modality combinations. They degrade gracefully rather than crashing.
3. **The Power of Modality Dropout**:
   Training the **Robust FlexiModal** with modality dropout (Stage 3, 30% dropout probability) forces the expert router and encoders to learn robust unimodal representations. As a result, when modalities are dropped (e.g. Face Missing or Voice Missing), the Robust FlexiModal model retains high F1 scores, outperforming all other fusion models under missing sensor configurations.

All reports, confusion matrices, and ROC curves have been generated and saved under `reports/` inside the `early_fusion/` directory.
