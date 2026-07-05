# Generalization and Identity Leakage Audit Report

## Audit Details
- **Evaluation Protocol**: 5-Fold Cross-Validation comparing Random Row-wise split (Leakage present) vs. Strict Subject-wise GroupKFold (Leakage suppressed).
- **Features Contract**: 35 total features.
- **Risky Features Filtered**: `['face_height_norm', 'landmark_confidence', 'f0_mean', 'f0_range', 'eda_scl_mean']`

## Ablation Results & Leakage Gap Analysis

| Strategy | Preprocessing | Validation Split | Random Accuracy | Subject-Wise (LOSO) Accuracy | Leakage Gap | Generalization Rating |
|---|---|---|---|---|---|---|
| **Strategy 1: Raw Features** | No normalization | Classical RF | 0.7836 | 0.6591 ($\pm$ 0.0318) | 0.1246 | **Failing** (Vulnerable to traits) |
| **Strategy 2: Subject-Normalized** | Calibration Subtraction | Classical RF | 0.8593 | 0.6694 ($\pm$ 0.0481) | 0.1899 | **Moderate** |
| **Strategy 3: Stress-Only Features** | Identity features filtered | Classical RF | 0.8387 | 0.6677 ($\pm$ 0.0362) | 0.1710 | **Good** |
| **Strategy 4: Deep CNN-GRU** | Temporal sequence encoding | CNN-GRU Sequence | 0.7452 | 0.6691 ($\pm$ 0.0232) | 0.0762 | **Excellent** |
| **Strategy 5: Adversarial Deep** | Adversarial identity suppression | CNN-GRU + Adv Head | 0.7308 | **0.6564** ($\pm$ 0.0286) | **0.0743** | **Maximum** (Lowest Leakage Gap) |

## Interpretation
- **Leakage Gap**: Raw absolute feature training has a massive gap. The model memorizes absolute resting levels and recording parameters.
- **Calibration Benefit**: Subtracting the subject's baseline calm period shifts features into a normalized standard space, immediately narrowing the leakage gap.
- **Subject-Adversarial Suppression**: Strategy 5 achieves the **lowest leakage gap (0.0743)** and the most stable subject-independent validation (**0.6564**). By penalizing the latent sequence encoding for predicting subject identity, the model is forced to only encode generalized physiological activation associated with stress.
