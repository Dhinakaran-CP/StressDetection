# Implementation Plan: 5-Model Ablation Study & Directory Structure Alignment (arch_2.md)

This plan details the implementation steps to align the `early_fusion/` project layout with the directory specification in [arch_2.md](file:///e:/Document/GitHub/StressDetectionUsingML/early_fusion/arch_2.md) and implement the 5-model comparative ablation study (Early, Gated, Cross-Attention, Standard MoE, and Robust MoE with Modality Dropout) across training, validation, and selection reports.

---

## User Review Required

> [!IMPORTANT]
> **Core Expansion: 5-Model Ablation Study**
> We will expand the notebooks and training pipeline to train, validate, and compare exactly five models:
> 1. **Early Fusion Classifier:** Concatenation baseline.
> 2. **Gated Fusion Classifier:** Dynamic scalar gates.
> 3. **Cross-Attention Classifier:** Inter-modality attention projections.
> 4. **Standard MoE Classifier:** FlexiModal MoE model trained *without* training modality dropout.
> 5. **Robust FlexiModal MoE Classifier:** FlexiModal MoE model trained *with* independent 30% modality dropout during training.

> [!WARNING]
> **Directory Rearrangement**
> We will create and organize the subfolders under `early_fusion/data/` exactly as specified:
> * `data/interim/raw_audit/`, `data/interim/synced_pairs/`, `data/interim/extracted_windows/`, `data/interim/quality_checks/`
> * `data/processed/train/`, `data/processed/val/`, `data/processed/test/`, `data/processed/metadata/`
> The processing scripts and notebooks will be updated to output intermediate steps and splits into these folders.

---

## Proposed Changes

### 1. Directory Initialization
We will create a helper script or run commands to establish the empty subfolders under `early_fusion/` to comply with Section 4 of [arch_2.md](file:///e:/Document/GitHub/StressDetectionUsingML/early_fusion/arch_2.md).

---

### 2. Component Updates

#### [MODIFY] [create_notebooks.py](file:///e:/Document/GitHub/StressDetectionUsingML/early_fusion/scripts/create_notebooks.py)
We will modify the notebook cell generation script:
* **Notebook 2:** Update to save alignment metadata outputs under `data/interim/raw_audit/` and `data/interim/synced_pairs/`, and save split datasets (`df_train`, `df_val`, `df_test`) under `data/processed/train/`, `data/processed/val/`, and `data/processed/test/` respectively.
* **Notebook 3 (Model Training):** 
  * Update to train all 5 models (Early, Gated, Cross-Attention, Standard MoE, Robust MoE).
  * Standard MoE will use `is_train=False` during dataset mapping, and Robust MoE will use `is_train=True` to activate Modality Dropout.
  * Save checkpoints for all 5 models under `outputs/checkpoints/` (e.g. `best_early.pt`, `best_gated.pt`, `best_attention.pt`, `best_moe_standard.pt`, `best_moe_robust.pt`).
* **Notebook 4 (Validation):**
  * Load all 5 checkpoints.
  * Evaluate each model under the simulated sensor-dropout scenarios (All present, Missing Face, Missing Voice, Missing Physio).
  * Compute parameter sizes (`sum(p.numel() for p in model.parameters())`) and average inference latencies (inference cost in milliseconds) for each model.
  * Save the complete comparative ablation metrics to `reports/evaluation/robustness_metrics.csv`.
* **Notebook 5 (Final Report):**
  * Read the robustness metrics and plot/display the final Stage 8 comparative study table.

---

## Verification Plan

### Automated Tests
* Re-run the notebook generator to verify the `.ipynb` files are successfully written:
  `python early_fusion/scripts/create_notebooks.py`
* Run model architecture unit tests:
  `pytest early_fusion/tests/test_early_fusion_moe.py`

### Manual Verification
* The user will run the notebooks in Google Colab (from Notebook 2 to Notebook 5) to verify:
  1. Files are correctly organized in the new `data/interim/` and `data/processed/` subdirectories.
  2. Training completes for all 5 models.
  3. The final comparative report displays the ablation metrics, model sizes, and latency costs.
