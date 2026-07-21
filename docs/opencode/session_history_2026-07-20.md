# Session History — SSVB-CASA-AIS Production Pipeline

## Date: 2026-07-20

---

## 1. Objective

Train and deploy the full **SSVB-CASA-AIS (Hybrid MoE with Cross-Attention and Adversarial Identity Suppression)** as the primary production model, using all 3 raw datasets (StressID, WESAD, EmpathicSchool) with comprehensive stress-relevant feature extraction.

---

## 2. Key Decisions & Discoveries

### 2.1 Architecture Gap Identified
- The deployed `adv_` `ModalityEncoder` (Conv1D+GRU) was a **simplified version**.
- The full architecture diagram (`final_arc(moe2).png`) shows a **6-stage pipeline**:
  1. Raw data → 8/10 sub-modalities
  2. SequenceExperts (Conv1D + SelfAttn + GRU)
  3. IntraModalityGates (gated fusion)
  4. CrossAttentionBlocks (6 directional pairs)
  5. GlobalMoERouter (learned modality routing)
  6. Output heads (Stress + Confidence + Subject_id with GRL)

### 2.2 Feature Coverage Gap Identified
- Original pipeline: 30 features (16 face + 10 voice + 4 physio)
- Research pipeline: **72 channels** (34 face + 24 voice + 14 physio)
- Critical missing features found: 3D head pose, temporal deltas, 13 MFCCs, spectral descriptors, EDA phasic/SCR, temp mean/std, accelerometer, `f0_std`

### 2.3 `f0_std` Fix
- `SUB_MODALITY_INDICES` in `ssvb_casa_ais.py` was missing `f0_std` (index 1) from voice quality group
- Fixed from `[3,4,5,6,7]` to `[1,3,4,5,6,7]`

### 2.4 Model Refactored to 10 Sub-Experts
- **Face** (3 experts): eye(9), mouth(6), global_face(18)
- **Voice** (3 experts): spectral_prosody(8), mfcc(13), quality(2)
- **Physio** (3 experts): cardio(2), eda(3), somatic(8)
- **Total**: 69 features (72 raw - 3 privacy exclusions)
- `gate_physio` changed from `num_sub=2` to `num_sub=3`
- Forward signature changed from 8 params to 10 params

### 2.5 SequenceExpert Enhancement
- Updated to match research `SequenceEncoder`
- Added `nn.MultiheadAttention` (4 heads) before GRU
- Supports `return_sequence=True` for training

### 2.6 Pass-Through Initialization
All new sub-experts, gates, cross-attention, and heads initialized to preserve existing `adv_` encoder predictions:
- New sub-experts: small random weights
- Intra-modality gates: equal weights (1/3)
- Cross-attention `out_proj`: zero
- Global MoE: equal weights
- Stress head: zero
- Confidence head: bias=0

---

## 3. Data Quality Audit (Critical Discovery)

### 3.1 Pipeline Data Audit Results
Ran comprehensive audit (`audit_data_quality.py`) on pipeline `combined_sequences.npy`:

| Dataset | NaN Rate | Extreme Values | Notes |
|---------|----------|----------------|-------|
| StressID | 33.7% | None | Voice: 60% NaN, Physio: 70% NaN |
| WESAD | 80.6% | None | Face/Voice: 100% NaN (expected, missing modalities) |
| EmpathicSchool | 83.5% | **1.6 billion** | Extreme sensor artifacts in physio channels |

### 3.2 Raw Source Validation
Validated all issues against **actual raw data files**:

| Issue | Raw Source | Pipeline Introduced? |
|-------|-----------|---------------------|
| StressID physio NaN | **0% NaN** (720K rows sampled) | YES — neurokit2 processing failures |
| EmpathicSchool 1.6B outliers | **0 extreme** (E4 CSVs clean) | YES — bug in physio_extractor |
| EmpathicSchool face NaN | MP4s exist | YES — face_extractor failure |
| Subject ID overlap | WESAD/ES share `s2..s17` | Real issue, needs fix |

**Conclusion: Raw data is clean. Pipeline extraction code introduced artifacts.**

### 3.3 Blocking Fixes Applied

**Fix 1: Subject ID Prefix**
- Modified `build_enriched_training_data.py` to prefix IDs: `stressid_2ea4`, `wesad_s2`, `empathicschool_s2`
- Combined: 53 + 15 + 23 = **91 unique subjects** (previously 76 with collisions)
- Verified: `CLEAN — No cross-dataset issues`

**Fix 2: Outlier Clipping**
- Per-channel winsorization at 99.9th percentile
- Absolute threshold clip (> 1e6 → NaN → 0)
- Applied to WESAD and EmpathicSchool (StressID was clean)

**Fix 3: NaN → 0 Imputation**
- Added `np.nan_to_num(..., nan=0.0)` in `SSVBDataset.__getitem__`
- Missing modalities (WESAD face/voice, ES voice) naturally become zeros

---

## 4. Extraction Pipeline Development

### 4.1 `build_enriched_training_data.py`
- Maps pipeline [N, 30, 72] sequences → 10 sub-modality groups (69 features)
- Added subject prefixing, outlier clipping, NaN imputation
- Output: `data/enriched_training_data/{stressid,wesad,empathicschool,combined}/`

### 4.2 `clean_data_pipeline.py`
- Direct raw data extraction (bypasses pipeline intermediate files)
- Multi-window support (2s, 5s, 10s, 30s)
- Robust error handling for face/voice/physio extraction
- Fixed: MediaPipe API error, voice hop_length error, OpenCV cascade path

### 4.3 `feature_extraction_service.py`
- **Modular, class-based design** for other agents
- `FeatureExtractor`: static methods for robust signal extraction
- `WindowProcessor`: multi-scale chunking
- `FeatureExtractionService`: public API with per-dataset methods
- Usage:
  ```python
  from webapp.training.phase8.feature_extraction_service import FeatureExtractionService
  svc = FeatureExtractionService()
  feats, meta = svc.extract_stressid(window_sec=10)
  ```

---

## 5. Training Pipeline

### 5.1 `train_ssvb_production.py`
- **SSL contrastive pretraining** (InfoNCE, same-subject positive pairs)
- **Supervised fine-tuning** (confidence-aware cross-entropy + GRL adversarial subject)
- **LOSO** (Leave-One-Subject-Out) GroupKFold, 5-fold CV
- Data augmentation: Gaussian noise (σ=0.02), modality dropout (15%)
- Metrics: ACC, F1, ROC-AUC, ECE calibration, per-subject breakdown
- Model checkpointing + deployment weight export

### 5.2 Training Functions Updated
- `_unpack_batch()`: reorders 9 sorted-group tensors → model parameter order
- `train_ssl_epoch()`: 10-parameter forward signature
- `train_supervised_epoch()`: confidence-aware loss + GRL
- `evaluate()`: returns probs, labels, confidence, subject IDs

### 5.3 Dry-Run Verification
```bash
.\venv\Scripts\python train_ssvb_production.py --dry-run
```
- Dataset sample: 11 items (9 feats + label + subj_id) ✓
- All feature shapes match model input dims ✓
- Model init: 39,897 params ✓

---

## 6. Files Created/Modified

### Architecture
- `webapp/backend/runtime/ssvb_casa_ais.py` — 10-expert refactored model
- `webapp/backend/runtime/runtime_engine.py` — SSVB integration in `predict_fused()`

### Training
- `webapp/training/phase8/train_ssvb_production.py` — Production training pipeline
- `webapp/training/phase8/build_enriched_training_data.py` — Enriched data builder

### Extraction
- `webapp/training/phase8/clean_data_pipeline.py` — Direct raw extraction
- `webapp/training/phase8/feature_extraction_service.py` — Modular agent-friendly pipeline

### Audit
- `webapp/training/phase8/audit_data_quality.py` — Pipeline data quality checks
- `webapp/training/phase8/investigate_data.py` — Deep-dive NaN/outlier analysis
- `webapp/training/phase8/validate_raw_sources.py` — Raw source validation

### Documentation
- `docs/ssvb_casa_ais_architecture.txt` — Full 6-stage architecture
- `docs/cross_dataset_modality_handling.txt` — Missing modality strategy

### Data
- `data/enriched_training_data/{stressid,wesad,empathicschool,combined}/` — Clean enriched data

---

## 7. Final Data Summary

| Dataset | Windows | Subjects | Stress % | Modalities Available |
|---------|---------|----------|----------|---------------------|
| StressID | 16,974 | 53 | 42.1% | Face + Voice + Physio |
| WESAD | 5,517 | 15 | 36.2% | Physio only |
| EmpathicSchool | 66,622 | 23 | 15.5% | Face + Physio |
| **Combined** | **89,113** | **91** | **21.8%** | All (with padding) |

---

## 8. Next Steps

1. Run production training:
   ```bash
   .\venv\Scripts\python webapp/training/phase8/train_ssvb_production.py
   ```
2. Update `runtime_engine.py` to load `ssvb_casa_ais_production.pt`
3. Verify inference matches pass-through prediction
4. Benchmark latency vs DynamicRouter path
