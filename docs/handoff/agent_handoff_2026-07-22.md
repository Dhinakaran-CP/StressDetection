# Agent Handoff — 4-Model Production Pipeline & Comparative Study

## 1. Mission

Run the **full production pipeline** comparing 4 models on the combined enriched dataset (89,113 windows, 91 subjects) with 5-fold LOSO cross-validation. Each model produces a complete diagnostic suite. A final comparison report selects the production champion.

## 2. The 4 Models

| Step | Model | Params | GRL | MoE | Description |
|:----:|-------|:------:|:---:|:---:|-------------|
| 1 | **SSVB-CASA-AIS** | ~41K | ❌ | ✅ (10 experts) | Original 6-stage architecture — overfitting ceiling |
| 2 | **CNNBaseline** | ~21K | ❌ | ❌ | Plain 1D-CNN, concat all features — physiology lower bound |
| 3 | **CNNBaseline+GRL** | ~23K | ✅ subj | ❌ | CNN + subject-adversarial GRL — isolates GRL effect |
| 4 | **ConvMoE-MF** | **8.8K** | ✅ **dual** | ✅ **(4 experts)** | Production target — MoE + dual GRL + confidence |

## 3. Per-Model Outputs

After each model completes, results are saved under:
`research/Phase_3_Production/production_model/ssvb_casa_ais_production/{model_tag}/`

```
{model_tag}/
  combined/
    aggregate_metrics.json       # ACC, precision, recall, F1, AUC-ROC, avg_precision
    metrics.json                 # Full report (per-dataset, per-subject, config)
    fold_metrics.csv             # Per-fold breakdown
    predictions.csv              # Per-sample true/prob/confidence/subject_id
    roc_auc.png                  # ROC curve
    pr_roc.png                   # Precision-Recall curve
    confusion_matrix.png         # Confusion matrix
    classification_report.txt    # Text classification report
  stressid/                      # Same structure for individual datasets
  wesad/
  empathicschool/
```

**Checkpoints**: `checkpoints/{dataset}_{model_tag}_best.pt`
**Deployment weights**: `webapp/backend/runtime/models/ssvb_casa_ais_production_{model_tag}.pt`

## 4. How to Run

### Single command — full pipeline:
```powershell
.\venv\Scripts\python webapp/training/phase8/run_pipeline.py
```

### Step-by-step (if you want to monitor):
```powershell
# Step 1: SSVB-CASA-AIS
.\venv\Scripts\python webapp/training/phase8/train_ssvb_production.py --dataset combined --model_type ssvb

# Step 2: CNNBaseline
.\venv\Scripts\python webapp/training/phase8/train_ssvb_production.py --dataset combined --model_type cnn_baseline

# Step 3: CNNBaseline+GRL
.\venv\Scripts\python webapp/training/phase8/train_ssvb_production.py --dataset combined --model_type cnn_baseline_grl

# Step 4: ConvMoE-MF (production target)
.\venv\Scripts\python webapp/training/phase8/train_ssvb_production.py --dataset combined --model_type conv_moe_mf
```

### Resume from a specific step:
```powershell
.\venv\Scripts\python webapp/training/phase8/run_pipeline.py --start-step 3
```

### Dry-run to validate:
```powershell
.\venv\Scripts\python webapp/training/phase8/run_pipeline.py --dry-run
```

## 5. Final Comparison Report

After all 4 models finish, the pipeline automatically builds:

`research/Phase_3_Production/production_model/ssvb_casa_ais_production/comparison_report.csv`

| model | accuracy | precision | recall | f1 | auc_roc | avg_precision |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:-------------:|
| ssvb | | | | | | |
| cnn_baseline | | | | | | |
| cnn_baseline_grl | | | | | | |
| conv_moe_mf | | | | | | |

**Selection criteria** (in order):
1. **AUC-ROC** on combined LOSO — primary metric
2. **F1 score** — class balance matters with 22% stress prevalence
3. **Avg Precision (AP)** — precision-focused for imbalanced data
4. **Per-subject accuracy std** — lower = more consistent across subjects
5. **Parameter efficiency** — ConvMoE-MF should match/best larger models at 8.8K params

A successful ConvMoE-MF shows:
- **AUC-ROC > 0.70** (meaningful stress detection)
- **AUC-ROC ≥ SSVB-CASA-AIS** (small model matches big model)
- **AUC-ROC ≥ CNNBaseline+GRL** (MoE + dual GRL adds value over GRL alone)
- **AUC-ROC > CNNBaseline** (GRL provides measurable gain)

## 6. Known Issues (verified)

| Issue | Status | Action |
|-------|--------|--------|
| StressID HR: 39% all-zero (rPPG fail) | Accepted | Zero-padding is correct encoding; dataset-GRL prevents shortcut |
| WESAD/EmpathicSchool: 100% face/voice zero | Accepted | Dataset-GRL prevents dataset-identity shortcut |
| EmpathicSchool EDA reversed (d=-0.18) | Accepted | Downweighted to 0.3× loss; keep for generalization |
| Combined Cohen's d HR reversed (d=-0.07) | Non-issue | Aggregate artifact; per-sample training unaffected |
| Per-dataset loss weighting | Implemented | EmpathicSchool at 0.3× via `dataset_weight_*` in CONFIG |
| SSL branching | Fixed | `train_ssl_epoch` branches by model type (SSVB: `exp_*`, ConvMoE: `enc_*`) |
| All 4 models smoke-tested | ✅ PASS | 1-fold, 2-epoch each, no errors |
| GRL alpha sweep | Scaffolded | `sweep_grl_alpha.py` created; run separately if needed |

## 7. Code Map

```
webapp/
  backend/runtime/
    conv_moe_mf.py                        # ConvMoE-MF (configurable GRL alphas)
    ssvb_casa_ais.py                      # Legacy SSVB-CASA-AIS
    models/
      ssvb_casa_ais_production_{tag}.pt   # Per-model deployment weights
  training/phase8/
    train_ssvb_production.py              # Main training (all 4 models, per-model dirs)
    run_pipeline.py                       # Sequential 4-model runner + comparison report
    sweep_grl_alpha.py                    # GRL alpha sweep
    compute_cohens_d.py                   # Cohen's d
    validate_raw_sources.py               # Source validation
data/
  enriched_training_data/combined/        # 89K windows (metadata.parquet + sequences.npz)
research/Phase_3_Production/production_model/ssvb_casa_ais_production/
  {model_tag}/combined/                   # Per-model results
  comparison_report.csv                   # Final cross-model comparison
docs/
  agent_handoff_2026-07-22.md             # ← This file
```

## 8. Quick Start

```powershell
# From project root:
cd webapp/training/phase8
.\venv\Scripts\python run_pipeline.py
```

Expected runtime: ~45-60 min per model × 4 = **~3-4 hours total** on CUDA GPU. Each model does 5 LOSO folds × (4 SSL + 8 supervised epochs) on 89K windows.
