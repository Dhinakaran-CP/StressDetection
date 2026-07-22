# Session History — 2026-07-21 (Updated)

## Summary

Addressed 5 detailed objections about cross-dataset validity, missing-modality leakage, data corruption, and CNN-vs-MoE comparison. Implemented:
- **Dual GRL** (subject + dataset adversarial heads) in ConvMoE-MF
- **CNN baseline** (21K params) in training pipeline for leakage-gap ablation
- **3 new validation docs** with concrete protocols (Cohen's d, t-SNE, LOSO cross-dataset)
- Updated architecture doc and training pipeline with all fixes

## Objections Addressed (from Detailed Review)

### 1. Cross-Dataset Label Alignment (Convincing the Examiner)
**Fix**: Replaced illustrative example numbers with a 3-step validation protocol:
- Step 1: Compute real Cohen's d per dataset before thesis submission
- Step 2: t-SNE of pre-fusion embeddings to verify dataset-agnostic stress clustering
- Step 3: LOSO cross-dataset (train on StressID+WESAD, eval on EmpathicSchool)
- Documented at `docs/cross_dataset_label_alignment_validation.md`

### 2. Missing-Modality Dataset-Identity Leakage
**Fix**: Added **dataset_head with independent GRL** (λ=0.02) alongside existing subj_head. Two separate GRL copies prevent gradient interference. The dataset head predicts {StressID, WESAD, EmpathicSchool} from the fused embedding and is adversarially suppressed. This closes the "face energy = 0 means WESAD" shortcut.
- Model: `webapp/backend/runtime/conv_moe_mf.py` (8,611 params, +27 from subj-only)
- Loss term: `lambda_dataset * CE(dataset_logits, dataset_id)` with λ=0.10
- Documented at `docs/cross_dataset_modality_handling.txt`

### 3. Data Corruption Validation Workflow
**Fix**: Created `docs/5_stage_feature_extraction_pipeline.md` which:
- Documents the known corruption (body of the objection's table is now in the doc)
- Establishes `validate_raw_sources.py` as the first-diagnostic rule
- Recommends confidence-scored gating (preserve low-quality windows, don't discard)

### 4. CNN vs ConvMoE-MF Comparison
**Fix**: Added `CNNBaseline` class (21,298 params) to `train_ssvb_production.py` with:
- Same 9 sub-modality input interface (drop-in compatible)
- 3 conv1d layers + GAP + linear classifier
- No GRL, no confidence head, no attention
- Activated via `model_type: 'cnn_baseline'` in CONFIG
- Ablation metric: leakage gap = random-split accuracy − LOSO accuracy

### 5. Architecture Doc Updated
**Fix**: `docs/conv_moe_mf_architecture.md` now includes:
- Dual GRL section (subject + dataset)
- Parameter comparison table (SSVB vs ConvMoE-MF vs CNN baseline)
- Full treatment of all 5 objections with how each was addressed

## Files Changed/Created

| File | Action | Description |
|------|--------|-------------|
| `webapp/backend/runtime/conv_moe_mf.py` | **Updated** | Added dataset_head with independent GRL; +27 params (8,584 → 8,611) |
| `webapp/training/phase8/train_ssvb_production.py` | **Updated** | ConvMoE_MF import, lambda_dataset loss, CNNBaseline class, SSVBDataset returns dataset_id, _unpack_batch handles 3 metadata fields |
| `docs/conv_moe_mf_architecture.md` | **Updated** | Dual GRL, CNN baseline, objection treatment, Cohen's d protocol |
| `docs/cross_dataset_label_alignment_validation.md` | **Created** | 3-step validation protocol (Cohen's d, t-SNE, LOSO cross-dataset) |
| `docs/cross_dataset_modality_handling.txt` | **Created** | Zero-padding, dataset-GRL mitigation, modality combinations table |
| `docs/5_stage_feature_extraction_pipeline.md` | **Created** | Extraction stages, corruption table, validate_raw_sources workflow |
| `docs/session_history_2026-07-21.md` | **Updated** | This file |

## Verified Outputs

```
ConvMoE-MF (with dataset_head): 8,611 params
  stress_logits: [4, 2]
  dataset_logits: [4, 3]    # NEW: StressID, WESAD, EmpathicSchool
  confidence:    [4]
  subj_logits:   [4, 65]

CNN Baseline: 21,298 params
  stress_logits: [4, 2]
  confidence:    [4]   (dummy = 1.0, no confidence head)

Training pipeline dry-run: PASS (stressid + combined, both at seq_len=5 and seq_len=30)
```

## Remaining Work

- Run production training: `python train_ssvb_production.py`
- Run Cohen's d computation: `python compute_cohens_d.py` (not yet written)
- Compare leakage gaps: ConvMoE-MF vs CNN baseline under identical LOSO protocol
