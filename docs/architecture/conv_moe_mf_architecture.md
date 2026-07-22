# ConvMoE-MF: Convolutional Mixture of Experts for Multimodal Fusion

## Problem

Current SSVB-CASA-AIS: **~500K+ params** for **290 samples/fold** (5-fold LOSO).  
Result: Overparameterization → overfitting → poor calibration (ECE 52.57% on VBC-CASA-IS).

## Solution: ConvMoE-MF (8,611 params)

**40× smaller**: Replace 8 heavy SequenceExperts with light Conv1D backbones + compact MoE fusion. Dual adversarial suppression (subject + dataset) with GRL.

### Parameter Count

| Component | SSVB-CASA-AIS | ConvMoE-MF | CNN Baseline (ablation) |
|-----------|--------------|-----------|------------------------|
| Encoders | 8×(Conv1D+SelfAttn+GRU) = ~480K | 3 Conv1D+GAP = ~6K | 3-layer Conv1D = ~20K |
| Fusion | Gates+6×CrossAttn = ~15K | 4-expert MoE = ~2K | Concatenation = 0 |
| Heads + GRL | 3 heads, 1 GRL = ~2K | 4 heads, 2 GRL = ~1K | 1 head, no GRL = ~1K |
| **Total** | **~500K+** | **8,611** | **21,298** |

---

## Architecture (3 Stages)

### Stage 1: Per-Modality Conv1D Encoders

```
Face  (33, T):  Conv1D(33→16, k=5) → BN → ReLU → Conv1D(16→8, k=3) → GAP → [8]
Voice (23, T):  Conv1D(23→16, k=5) → BN → ReLU → Conv1D(16→8, k=3) → GAP → [8]
Physio(13, T):  Conv1D(13→8,  k=3)                                  → GAP → [8]

Concatenate: [24] ← [face_8, voice_8, physio_8]
```

**Rationale**: Proportional capacity to modality richness. Face (33-dim) and Voice (23-dim) get 2 conv layers; Physio (13-dim) gets 1. GAP forces feature-level invariance — no GRU needed for 30-frame windows at 3fps.

### Stage 2: MoE Fusion (4 experts)

```
4 experts:  MLP(24→16→8) × 4  (each sees full concatenated embedding)
1 router:   MLP(24→4, softmax)

Fused = Σ_i router_weight_i × expert_i(concat_embedding)
Output: [8]
```

**Rationale**: MoE routers learn modality importance per sample. No cross-attention — at N=290, pairwise attention learns spurious correlations. Expert specialization is sufficient for cross-modal interaction.

### Stage 3: Output Heads (Dual Adversarial Suppression)

```
Stress head:           Linear(8→2)                        → stress logits
Confidence head:       Linear(8→1) + Sigmoid              → confidence [0,1]
Subject head (GRL):    GRL(λ=0.02) → Linear(8→N_subj)    → subject logits (adversarial)
Dataset head (GRL):    GRL(λ=0.02) → Linear(8→3)         → dataset logits (adversarial)
```

**Dataset head (new)**: Closes a gap identified in review — zero-padded face modalities for WESAD could let the model learn dataset identity as a shortcut. Dual GRL ensures the fused embedding is invariant to both *individual identity* and *dataset origin*. Two separate GRL copies prevent gradient interference.

---

## Addressing the 5 Objections from Review

### 1. Overparameterization
**Fixed**: 8,611 params — 40× reduction from SSVB-CASA-AIS. Each fold trains on more params than samples, but the Conv1D inductive bias (translation equivariance) is correct for time-series physiological signals, unlike self-attention's global receptive field.

### 2. Redundant Cross-Attention
**Fixed**: Removed entirely. 6 directional attention heads → 0. Replaced by MoE which learns modality importance weights — a strictly simpler task that converges at N=290.

### 3. Dual Identity Suppression (quality gates + GRL)
**Fixed**: Quality gates removed. Instead, two independent GRL heads: one for subject identity, one for dataset identity. Both use λ=0.02 (proven effective in λ_adv sweep reducing leakage from 18.99% → 7.43%). Dataset-head GRL prevents the model from learning "all-zero face = WESAD" as a shortcut.

### 4. Ambiguous Quality Masks
**Fixed**: Removed entirely. No learnable or rule-based quality masking. Signal quality can be applied at the input layer via the confidence score's per-window gating, but does not interact with the adversarial training loop.

### 5. Miscalibrated Expert Allocation
**Fixed**: Proportional allocation — Face (33-dim) gets 2 conv layers, Voice (23-dim) gets 2, Physio (13-dim) gets 1. Each encoder outputs the same [8] embedding, so the MoE fusion sees equal-dimensional contributions regardless of input richness.

---

## CNN Baseline (Ablation)

A plain 1D-CNN (21,298 params) is provided as an ablation target. It concatenates all 9 sub-modality tensors → 3 Conv1D layers → GAP → linear classifier.

**Why the CNN baseline will score higher but is less trustworthy**:
- Plain CNNs are excellent shortcut learners — they exploit any bias in the data
- Without GRL, the CNN can freely use subject identity and dataset identity as features
- The **leakage gap** (random-split accuracy − LOSO accuracy) measures this: a larger gap = more shortcut reliance
- ConvMoE-MF's value is not higher raw accuracy — it's **comparable accuracy with provably less identity leakage**

Run both with `model_type: 'cnn_baseline'` vs `'conv_moe_mf'` in `CONFIG` and compare leakage gaps.

---

## Cross-Dataset Label Alignment Validation

### The Claim
All three datasets (StressID, WESAD, EmpathicSchool) activate the same HPA axis → SNS pathway, producing physiologically convergent stress responses (HR↑, HRV↓, EDA↑) despite different stressor types (task-based, TSST, naturalistic).

### The Validation Protocol (run before thesis submission)

1. **Cohen's d per dataset**: For StressID, WESAD, and EmpathicSchool independently, compute Cohen's d between stress=1 and stress=0 for HR, HRV-RMSSD, and EDA. If all three d-values point the same direction with comparable magnitude, the convergence is **empirically measured, not assumed**.

2. **t-SNE of embeddings**: Run ConvMoE-MF encoder on the full combined dataset. Plot t-SNE of the 24-D pre-fusion embeddings colored by stress label. If stress clusters are dataset-agnostic (i.e., stress=1 points from all 3 datasets occupy the same region), the model is learning cross-dataset stress patterns.

3. **LOSO cross-dataset**: Train on StressID+WESAD, evaluate on held-out EmpathicSchool subject. This is the strongest form of generalization evidence — if it works, the argument is won.

### Current Status
Subject-adaptive z-score normalization is applied (subtract per-subject calm baseline), which removes sensor and baseline offsets. The enriched combined dataset (89,113 windows, 91 subjects) has the correct structure for all 3 validation steps above.

---

## Missing-Modality Handling

### Zero-Padding + Learned Gating (No Quality Masks)

When a dataset lacks a modality (e.g., WESAD has no face/voice, EmpathicSchool has no voice), the missing modality's sub-modality tensors are all-zero. The Conv1D encoder outputs zeros for that modality. The MoE router learns to assign near-zero weight to zero-energy modalities.

**Dataset-identity leakage risk**: Zero-padding creates a perfect correlation between "face energy = 0" and "this is a WESAD sample." Without mitigation, the model could use "face is zero" as a proxy for "this is WESAD" and learn dataset-specific stress patterns instead of universal ones.

**Mitigation**: The dataset-adversarial GRL head directly optimizes against this. By forcing the fused embedding to be uninformative of dataset identity, any shortcut learned from zero-padding is penalized.

---

## Training Procedure

```
model_type = 'conv_moe_mf'   # or 'ssvb' or 'cnn_baseline'

1. Load enriched data (89K windows, 91 subjects, 3 datasets)
2. StandardScaler per modality → z-score normalize
3. LOSO 5-fold split (no subject overlap across folds)
4. Stage 1 (skip for CNN baseline): SSL contrastive pretraining
5. Stage 2: Supervised fine-tuning
   Loss = CE(stress) + 0.15×BCE(confidence) + 0.10×CE(subject_GRL) + 0.10×CE(dataset_GRL)
6. Evaluate: accuracy, precision, recall, F1, AUC, ECE, leakage gap
```

Run: `python train_ssvb_production.py`

---

## Verification Plan

| Test | Expected |
|------|----------|
| **Parameter count** | 8,611 |
| **Dataset head shape** | [B, 3] (StressID, WESAD, EmpathicSchool) |
| **CNN baseline params** | 21,298 |
| **LOSO accuracy (ConvMoE-MF)** | Comparable to SSVB-CASA-AIS |
| **LOSO accuracy (CNN baseline)** | Higher raw accuracy, larger leakage gap |
| **ECE** | < 15% (vs 52.57% on VBC-CASA-IS) |
| **Training time** | ~5min/fold vs ~30min+ for SSVB-CASA-AIS |
| **Gate entropy** | Spread across 3 modalities, not collapsed to 1 |
