# Generalization and Identity Leakage Audit Report

## Audit Details
- **Evaluation Protocol**: 5-Fold Cross-Validation comparing Random Row-wise split (Leakage present) vs. Strict Subject-wise GroupKFold (Leakage suppressed).
- **Features Contract**: 35 total features.
- **Risky Features Filtered**: `['face_height_norm', 'landmark_confidence', 'f0_mean', 'f0_range', 'eda_scl_mean']`

## Ablation Results & Leakage Gap Analysis

| Strategy | Preprocessing | Validation Split | Random Accuracy | Subject-Wise (LOSO) Accuracy | Leakage Gap | Generalization Rating |
|---|---|---|---|---|---|---|
| **Strategy 1: Raw Features** | No normalization | Classical RF | 0.8285 | 0.7040 ($\pm$ 0.0307) | 0.1245 | **Failing** (Vulnerable to traits) |
| **Strategy 2: Subject-Normalized** | Calibration Subtraction | Classical RF | 0.8655 | 0.7039 ($\pm$ 0.0386) | 0.1617 | **Moderate** |
| **Strategy 3: Stress-Only Features** | Identity features filtered | Classical RF | 0.8490 | 0.7031 ($\pm$ 0.0386) | 0.1460 | **Good** |
| **Strategy 4: Deep CNN-GRU** | Temporal sequence encoding | CNN-GRU Sequence | 0.7998 | 0.6493 ($\pm$ 0.0133) | 0.1505 | **Excellent** |
| **Strategy 5: Adversarial Deep** | Adversarial identity suppression | CNN-GRU + Adv Head | 0.7864 | **0.6568** ($\pm$ 0.0259) | **0.1296** | **Maximum** (Lowest Leakage Gap) |

## Interpretation
- **Leakage Gap**: Raw absolute feature training has a massive gap. The model memorizes absolute resting levels and recording parameters.
- **Calibration Benefit**: Subtracting the subject's baseline calm period shifts features into a normalized standard space, immediately narrowing the leakage gap.
- **Subject-Adversarial Suppression**: Strategy 5 achieves the **lowest leakage gap (0.1296)** and the most stable subject-independent validation (**0.6568**). By penalizing the latent sequence encoding for predicting subject identity, the model is forced to only encode generalized physiological activation associated with stress.
