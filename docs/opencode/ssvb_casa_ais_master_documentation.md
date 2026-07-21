# SSVB-CASA-AIS Master Documentation

This document serves as the master documentation for the SSVB-CASA-AIS (Shared-Subject Variational Bayes - Cross-Attention Sequence Aggregation with Adversarial Identity Suppression) project, combining all architecture, pipeline, validation, and session history details.

---

## Part 1: SSVB-CASA-AIS Architecture (10 Sub-Modality Version)

### 1.1 Overview
- **Full name**: Shared-Subject Variational Bayes - Cross-Attention Sequence Aggregation with Adversarial Identity Suppression
- **Type**: Hybrid Mixture-of-Experts with Cross-Attention
- **Input**: 10 sub-modality groups, 69 total features (from 72 raw channels, minus 3 privacy exclusions)
- **Output**: Binary stress classification + confidence score

### 1.2 Architecture Stages (6-Stage Pipeline)

**Stage 1: RAW DATA LOADING (External)**
- 3 datasets: StressID (all 3 mod), WESAD (physio only), EmpathicSchool (face+physio)
- Missing modalities filled with zeros at inference

**Stage 2: SUB-MODALITY EXPERTS (10 SequenceExperts)**
- Each expert: Conv1D(3,4) -> BN -> ReLU -> MultiheadSelfAttn(4 heads) -> GRU(hidden_dim)
- **Face group (3 experts)**:
  * `exp_eye`: input_dim=9 (eye aspect ratios, deltas)
  * `exp_mouth`: input_dim=6 (lip compression, jaw, deltas)
  * `exp_global_face`: input_dim=18 (brow, head pose, forehead, nose, deltas)
- **Voice group (3 experts)**:
  * `exp_spectral_prosody`: input_dim=8 (rms, zcr, f0_std, spectral features)
  * `exp_mfcc`: input_dim=13 (13 MFCC coefficients)
  * `exp_quality`: input_dim=2 (hnr, jitter)
- **Physio group (3 experts)**:
  * `exp_cardio`: input_dim=2 (hr, hrv_rmssd)
  * `exp_eda`: input_dim=3 (eda_clean, eda_phasic, scr_count)
  * `exp_somatic`: input_dim=8 (resp_rate, resp_amp, temp, acc x/y/z/mag)
- Each expert outputs a `[batch, hidden_dim]` embedding
- Supports `return_sequence=True` for loss computation

**Stage 3: INTRA-MODALITY GATES (3 GatedFusion modules)**
- `gate_face`: fuses {e_eye, e_mouth, e_gface} -> face latent
- `gate_voice`: fuses {e_sp, e_mfcc, e_qual} -> voice latent
- `gate_physio`: fuses {e_cardio, e_eda, e_soma} -> physio latent
- Each gate: learned weighted sum with softmax routing

**Stage 4: CROSS-ATTENTION BLOCKS (6 CrossAttentionBlocks)**
- 6 directional pairs: fv, fp, vf, vp, pf, pv
- Each block: target attends source with quality masking support
- Outputs fused per-modality: f_re, v_re, p_re
- Projection: proj_f/proj_v/proj_p map concat(cross_pair) -> hidden_dim

**Stage 5: GLOBAL MOE ROUTER**
- GlobalMoERouter: 3-modality gated fusion
- Learned routing weights per sample
- Final fused representation: f_rep

**Stage 6: OUTPUT HEADS (3 heads)**
- `stress_head`: Linear(hidden_dim, 2) -> logits
- `confidence_head`: Linear(hidden_dim, 1) -> sigmoid score
- `subj_head` (with GradientReversal): Linear(hidden_dim, num_subjects)
  * GRL alpha=0.02 for adversarial subject de-identification
- Confidence-aware loss (DeVries et al.):
  `probs_adj = conf * softmax(logits) + (1-conf) * onehot(label)`

### 1.3 Pass-Through Initialization
- All sub-experts: small random weights (Kaiming uniform)
- Intra-modality gates: equal weights (1/3 each)
- Cross-attention `out_proj`: zero initialized (no attention residual at init)
- Global MoE: equal weights (1/3 per modality)
- Stress head: zero initialized
- Confidence head: bias=0
- Subj head: weights=1e-3, bias=0
Purpose: Existing ModalityEncoder predictions preserved at deployment

### 1.4 Sub-Modality Group Mapping (72 raw channels -> 69 features)
Excluded (privacy): face[11]=face_height_norm, voice[36]=f0_mean, physio[61]=eda_tonic_scl

- **Face (34 raw, 33 used)**:
  * eye: [0,1,2,3,4, 18,19,20,32] -> 9 feats
  * mouth: [8,9,10, 24,25,26] -> 6 feats
  * global_face: [5,6,7,12,13,14,15,16,17, 21,22,23,27,28,29,30,31,33] -> 18 feats
- **Voice (24 raw, 23 used)**:
  * spectral_prosody: [34,35,37, 51,52,53,54,55] -> 8 feats
  * mfcc: [38..50] -> 13 feats
  * quality: [56,57] -> 2 feats
- **Physio (14 raw, 13 used)**:
  * cardio: [58,59] -> 2 feats
  * eda: [60,62,63] -> 3 feats
  * somatic: [64,65,66,67,68,69,70,71] -> 8 feats
Total: 33 + 23 + 13 = 69 features

---

## Part 2: Cross-Dataset Modality Handling

### 2.1 The Problem
- **StressID**: Has Face (video) + Voice (audio) + Physio (ECG/EDA/RR)
- **WESAD**: Has Physio ONLY (chest + wrist) - NO face/voice
- **EmpathicSchool**: Has Face (video) + Physio (E4 wristband) - NO voice

The model expects all 10 sub-modality inputs. Missing modalities must be handled gracefully.

### 2.2 The Solution: Zero-Padding + Gated Attention

**Stage 2 (Sub-Experts)**:
- For missing modalities, input is a zero tensor `[T, feat_dim]`
- The Conv1D->GRU expert processes zeros → near-zero output
- This is a "null" embedding

**Stage 3 (Intra-Modality Gates)**:
- For WESAD (no face): face_eye, face_mouth, face_global_face all produce near-zero embeddings
- `gate_face` sees 3 near-zero inputs → learned to weight them ~0
- `gate_physio` sees real physio inputs → weight ~1.0
- The gate learns which sub-experts carry signal

**Stage 4 (Cross-Attention)**:
- A zero-modality latent produces near-zero attention keys/values
- Cross-attention with a zero source: target updates minimally
- Quality masks can explicitly zero out missing modalities (optional, not required since zeros flow through naturally)

**Stage 5 (Global MoE)**:
- f_rep, v_rep, p_rep: zero for missing modalities
- GlobalMoERouter weights: learned to ignore zero-modalities
- Final representation is from available modalities only

### 2.3 Dataset-Specific Feature Coverage

| Modality Group | StressID | WESAD | EmpathicSchool |
|---|---|---|---|
| Face eye | 9 feats | zeros | 9 feats |
| Face mouth | 6 feats | zeros | 6 feats |
| Face global | 18 feats | zeros | 18 feats |
| Voice spectral | 8 feats | zeros | zeros |
| Voice MFCC | 13 feats | zeros | zeros |
| Voice quality | 2 feats | zeros | zeros |
| Physio cardio | 2 feats | 2 feats | 2 feats |
| Physio eda | 3 feats | 3 feats | 3 feats |
| Physio somatic | 8 feats | 8 feats | 8 feats |

**Key insight**: Physio features across datasets use different sensors:
- StressID: 500Hz ECG/EDA chest sensors (professional grade)
- WESAD: 700Hz chest + 32Hz wrist (research grade)
- EmpathicSchool: 4Hz E4 wristband (consumer grade)

The model learns to extract stress-relevant patterns from whatever sensor data is available.

### 2.4 Why This Works
- The gating mechanisms (IntraModalityGate, GlobalMoERouter) are learned weighted averages. They naturally learn to weight informative inputs higher and zero/null inputs lower.
- Cross-attention with a zero source produces a zero delta, making it a no-op.
- During LOSO training, a fold leaving out StressID subjects still sees WESAD+EmpathicSchool physio patterns, and vice versa.

### 2.5 Verification
- Enriched data: 91 total unique subjects (53+15+23)
  * StressID: 53 subjects, all 10 groups non-zero
  * WESAD: 15 subjects, only physio groups non-zero (face/voice=0)
  * EmpathicSchool: 23 subjects, face+physio non-zero (voice=0)
- The combined dataset concatenates all, and the model learns per-sample gating during training.

---

## Part 3: 5-Stage Raw Feature Extraction Pipeline

### 3.1 Overview
A systematic pipeline to extract rich, high-quality features from all 3 raw stress datasets (StressID, WESAD, EmpathicSchool) with **window-level confidence scoring**, **proper naming conventions**, and **multi-scale temporal windows**. Each stage builds on the previous, culminating in a unified, confidence-scored feature bank ready for downstream modelling.

### 3.2 Stage 1 — Single Dataset Full Extraction
**Purpose**: Extract every available modality from each dataset independently. Outputs raw windowed sequences per dataset.
**Naming Convention**: `{dataset}_{subject}_{task}_{modality}_{window_start}-{window_end}.npz`

**1.1 StressID Extraction**
| Modality | Raw Source | Output Features | Window Strategy |
|---|---|---|---|
| Face | `Videos/{subj}/{subj}_{task}.mp4` | 34-D (eye AR, mouth, head pose, deltas) | 3 fps, 10s window, 5s stride |
| Voice | `Audio/{subj}/{subj}_{task}.wav` | 24-D (RMS, ZCR, f0, MFCCs, spectral) | Same window alignment |
| Physio | `Physiological/{subj}/{subj}_{task}.txt` | 14-D (HR, HRV, EDA, SCR, resp) | Same window alignment |

**1.2 WESAD Extraction**
| Modality | Raw Source | Output Features | Notes |
|---|---|---|---|
| Physio | `data/wesad/{S*}/{S*}.pkl` | 14-D | Chest + wrist sensors |
| Face | None | All-zeros (34-D) | No camera in WESAD |
| Voice | None | All-zeros (24-D) | No microphone in WESAD |

**1.3 EmpathicSchool Extraction**
| Modality | Raw Source | Output Features | Notes |
|---|---|---|---|
| Face | `data/empathicschool/{S*}/**/*.mp4` | 34-D | Recursive search for MP4s |
| Physio | `**/*EDA.csv` etc. | 14-D | Resample E4 4Hz → 3fps |
| Voice | None | All-zeros (24-D) | Audio is low quality |

### 3.3 Stage 2 — Single Modality Extraction
**Purpose**: Extract each modality independently across all datasets. Enables per-modality analysis and ablation studies.
**Naming Convention**: `{modality}__{dataset}__{subject}__{window_id}.npz`

### 3.4 Stage 3 — Combined Dataset Extraction
**Purpose**: Merge all 3 datasets into a unified feature matrix with proper subject prefixing and modality alignment.
**Naming Convention**: `combined__{dataset}__{subject}__{window_id}.npz`

**Subject Identity Disambiguation**: Prefix subjects with `SID_`, `WSD_`, `EMP_` to prevent collisions (e.g. `WSD_s2` and `EMP_s2`).
**Modality Alignment Matrix**: Each window always has **69 feature channels**, zero-filled if missing.

### 3.5 Stage 4 — Multi-Window Scale Extraction
**Purpose**: Extract features at multiple temporal resolutions to capture different stress patterns.
- **2s (micro)**: High resolution. Micro-expressions, HR spikes.
- **5s (short)**: Medium. Short-term EDA fluctuations.
- **10s (standard)**: Balance of context and localization.
- **30s (long)**: Low resolution. Long-term HRV trends.

### 3.6 Stage 5 — Confidence-Scored Window Extraction
**Purpose**: Every window receives a per-modality **confidence score** based on signal quality.
**Naming Convention**: `{window_id}__conf_{overall_confidence:.3f}.npz`

- Face: Detection rate, landmark quality
- Voice: Voiced ratio, SNR
- Physio: NeuroKit2 signal quality, missing data rate
- Overall: Modality availability weighted (physio=0.5, face=0.3, voice=0.2)

Low-confidence windows are flagged but preserved for confidence-aware training. Sidecar JSON provides detailed metrics.

---

## Part 4: Cross-Dataset Stress Label Alignment Validation

### 4.1 The Problem
Each dataset defines "stress" differently:
- **StressID**: Task-based labels (Math, Speaking, Stroop, etc.)
- **WESAD**: Protocol-based (TSST: Public speaking + mental arithmetic)
- **EmpathicSchool**: Self-report + academic tasks (exams, presentations)

**Key Question**: Does the same physiological pattern emerge across different definitions? Can a single model learn both?

### 4.2 Theoretical Basis: Convergent Physiology of Stress
Regardless of the stressor type, the human stress response activates the HPA axis and the Sympathetic Nervous System (SNS), resulting in a consistent physiological cascade:
- Heart rate ↑
- HRV ↓
- Skin conductance (EDA/SCR) ↑
- Respiration rate ↑

The physiological signature is the same regardless of the stressor.

### 4.3 Why A Shared Model Works
**The Model Learns Physiology, Not Tasks**: The model doesn't see "task type", only features.
**Physiological Overlap**: Stress states across StressID, WESAD, and EmpathicSchool all show elevated HR, decreased HRV, and elevated EDA compared to their baselines.
**Subject-Adaptive Normalization**: Per-subject z-score normalization (`(raw - mean) / std`) removes individual baselines and sensor offsets, making the relative directional changes comparable across datasets.

### 4.4 Risks & Mitigations
- **Label Noise**: Tasks might not always induce stress (StressID), or naturalistic labels might be weak (EmpathicSchool 15.5%). Mitigated by confidence head weighting.
- **Dataset Bias**: Sensor differences (500Hz vs 4Hz). Mitigated by normalization and Leave-One-Subject-Out (LOSO) validation.

**LOSO Validation as Ground Truth**: Testing on a held-out subject from a different dataset ensures the model has learned a general stress pattern, not dataset-specific artifacts.

### 4.5 Expected Performance
- **StressID**: F1 0.70-0.75 (clean labels, all modalities)
- **WESAD**: F1 0.88-0.95 (strong TSST induction, clean protocol)
- **EmpathicSchool**: F1 0.50-0.65 (weak naturalistic labels, missing modalities)
- **Combined**: F1 0.65-0.72 (cross-dataset generalization)

---

## Part 5: Session History (2026-07-20)

### 5.1 Objective
Train and deploy the full SSVB-CASA-AIS model as the primary production model using all 3 raw datasets.

### 5.2 Key Decisions & Discoveries
- **Architecture Gap**: The deployed `adv_ModalityEncoder` was simplified. Full architecture is a 6-stage pipeline.
- **Feature Coverage Gap**: Original pipeline used 30 features; research pipeline uses **72 channels**. Missing features included 3D head pose, MFCCs, temp, acc, etc.
- **Model Refactored**: Expanded to 10 sub-experts and 69 used features.
- **Pass-Through Initialization**: All new sub-experts and attention mechanisms initialized to preserve existing predictions before finetuning.

### 5.3 Data Quality Audit (Critical Discovery)
- Found massive NaN rates (33% StressID, 80% WESAD/ES) and extreme values (1.6 billion in ES).
- Raw data was verified to be clean; pipeline extraction code introduced artifacts (neurokit2 failures, bugs).
- **Fixes Applied**: Prefix subject IDs (prevented collision, now 91 total unique subjects), outlier clipping (99.9th percentile winsorization), and NaN to 0 imputation (graceful handling of missing modalities).

### 5.4 Pipeline & Training Development
- **Extraction**: Created `feature_extraction_service.py` for a modular, class-based extraction process and `clean_data_pipeline.py` for direct raw extraction.
- **Training**: Set up `train_ssvb_production.py` for SSL contrastive pretraining, followed by supervised fine-tuning with confidence-aware cross-entropy and GRL adversarial subject training. Added LOSO validation and data augmentation.

### 5.5 Final Data Summary
- **StressID**: 16,974 windows, 53 subjects, 42.1% stress (Face + Voice + Physio)
- **WESAD**: 5,517 windows, 15 subjects, 36.2% stress (Physio only)
- **EmpathicSchool**: 66,622 windows, 23 subjects, 15.5% stress (Face + Physio)
- **Combined**: **89,113 windows**, **91 subjects**, **21.8% stress** (All, with padding)
