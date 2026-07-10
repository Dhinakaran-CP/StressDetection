# End-to-End Multimodal Stress Detection Project Instructions

## Project Goal
Build a robust multimodal stress detection system using Face/Video, Voice/Audio, and Physio signals. The raw data is partially synchronized, so the project must first verify synchronization, then create a clean synchronized subset, and finally train and validate multiple multimodal fusion methods.

The final system should support:
- synchronized multimodal training,
- missing-modality awareness,
- subject-independent validation,
- and fair comparison of multiple fusion strategies.

---

## 1. Project Principles

1. Do not assume all raw files are synchronized.
2. Use raw data verification before extraction.
3. Use only fully synchronized samples for the main synchronized fusion training.
4. Keep partial or missing-modality samples separate for robustness analysis.
5. Split data by subject, not by row.
6. Record every transformation, filter, and decision in logs and CSV reports.

---

## 2. Required Folder Structure

Create the following project structure:

```text
project_root/
├── data/
│   ├── raw/
│   │   ├── video/
│   │   ├── audio/
│   │   ├── physio/
│   │   └── manifests/
│   ├── interim/
│   │   ├── raw_audit/
│   │   ├── synced_pairs/
│   │   ├── extracted_windows/
│   │   └── quality_checks/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── metadata/
│   └── splits/
│       ├── train_ids.csv
│       ├── val_ids.csv
│       └── test_ids.csv
├── notebooks/
│   ├── 01_raw_sync_audit.ipynb
│   ├── 02_extract_and_align.ipynb
│   ├── 03_train_models.ipynb
│   ├── 04_validate_models.ipynb
│   └── 05_final_report.ipynb
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── utils/
├── experiments/
│   ├── baseline_early_fusion/
│   ├── gated_fusion/
│   ├── cross_attention/
│   ├── moe_fusion/
│   └── missing_modality_robust/
├── reports/
│   ├── synchronization/
│   ├── extraction/
│   ├── training/
│   ├── evaluation/
│   └── figures/
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── final_model/
├── logs/
├── configs/
├── scripts/
├── README.md
└── requirements.txt
```

---

## 3. Stage A: Raw Data Audit

### Objective
Verify whether the raw video, audio, and physio files are present, mapped correctly, and aligned enough for extraction.

### Required tasks
1. Recursively scan the raw directory in Google Drive.
2. Group files by `subject_id` and `task_id`.
3. Identify file modality by path and extension.
4. Measure recording duration for each modality.
5. Detect missing modalities.
6. Detect duration mismatches.
7. Save a raw audit CSV.
8. Save a summary report.

### Output
- Raw synchronization status.
- Total subject-task pairs.
- Counts of complete and incomplete recordings.
- Duration mismatch count.
- File-level audit CSV.

### Acceptance rule
A recording is considered raw-aligned only if video, audio, and physio exist for the same subject-task pair and durations are close enough.

---

## 4. Stage B: Clean Alignment and Extraction

### Objective
Build the dataset that will be used for model training.

### Required tasks
1. Normalize `task_id` to lowercase across all extracted datasets.
2. Build alignment keys using:
   - `subject_id`
   - `task_id`
   - `window_index`
3. Keep only rows present in all three modalities.
4. Verify that:
   - `window_start` matches,
   - `window_end` matches,
   - labels match.
5. Save the synchronized subset separately.
6. Keep a log of excluded rows and why they were excluded.

### Output
- Fully synchronized subset.
- Partial-overlap subset.
- Exclusion report.

### Acceptance rule
Only rows with exact modality alignment and correct labels should enter the synchronized training set.

---

## 5. Stage C: Dataset Splitting

### Objective
Avoid leakage by splitting on subject identity.

### Required tasks
1. Split subjects into train, validation, and test groups.
2. Ensure no subject appears in more than one split.
3. Preserve class balance as much as possible.
4. Save split lists to CSV.
5. Apply the same subject split to all modalities.

### Recommended split
- 70% train
- 15% validation
- 15% test

If the number of subjects is limited, use a stratified subject-aware split or cross-validation.

---

## 6. Stage D: Feature and Window Preparation

### Objective
Prepare model-ready tensors or feature tables.

### Required tasks
1. Extract or load features for each modality.
2. Ensure fixed window length for all synchronized rows.
3. Standardize input formats.
4. Normalize features within training-only statistics.
5. Store processed train/val/test sets separately.

### Output
- `data/processed/train`
- `data/processed/val`
- `data/processed/test`
- metadata files with sample IDs and labels

---

## 7. Stage E: Model Methodologies

Train and compare the following methods.

### Method 1: Early Fusion Baseline
- Concatenate modality embeddings.
- Feed into a classifier head.
- Use as the simplest baseline.

### Method 2: Gated Fusion
- Build modality-specific encoders.
- Learn gating weights per sample.
- Weight each modality dynamically before classification.

### Method 3: Cross-Attention Fusion
- Use one modality to attend to another.
- Model richer interaction between synchronized signals.
- Strong for samples where modalities carry complementary information.

### Method 4: Mixture-of-Experts Fusion
- Use modality-specific experts.
- Route samples dynamically to experts.
- Add gating/router network and load-balancing or entropy regularization.
- Make this model missing-modality aware.

### Method 5: Hybrid Robust Fusion
Recommended final model:
- modality-specific encoders,
- cross-attention interaction,
- gating or router,
- sparse MoE fusion,
- final classifier head.

This is the most novel and robust method for your dataset.

---

## 8. Stage F: Training Procedure

### Required tasks
1. Train each model on the synchronized training set.
2. Validate on the validation set after each epoch.
3. Use early stopping based on validation F1 or loss.
4. Save checkpoints for the best validation model.
5. Log training loss, validation loss, accuracy, precision, recall, and F1.
6. Use the same random seed across all experiments for reproducibility.

### Recommended loss
- Cross-entropy for classification.
- Add auxiliary gating regularization if using MoE.
- Add modality-consistency loss if your implementation supports it.

### Output
- Best model checkpoint.
- Training logs.
- Epoch metrics CSV.

---

## 9. Stage G: Final Validation and Testing

### Required tasks
1. Load the best checkpoint for each methodology.
2. Evaluate on the held-out test set.
3. Compute:
   - accuracy,
   - precision,
   - recall,
   - F1 score,
   - confusion matrix,
   - per-class metrics.
4. Save predictions and probabilities.
5. Compare all methodologies fairly using the same test split.

### Acceptance rule
The selected model must be the one that performs best on the validation set and remains stable on the test set.

---

## 10. Stage H: Robustness Analysis

### Objective
Evaluate how robust the model is when modalities are missing.

### Required tasks
1. Test the final model on samples with missing Face.
2. Test on samples with missing Voice.
3. Test on samples with missing Physio.
4. Compare performance drop versus complete samples.
5. Report which modality is most important.

### Output
- Missing-modality robustness table.
- Performance degradation analysis.
- Modality ablation report.

---

## 11. Stage I: Ablation Study

Compare the following:
- Early fusion only.
- Gated fusion only.
- Cross-attention only.
- MoE only.
- Hybrid cross-attention + gating + MoE.

### Required reporting
For each method, report:
- test accuracy,
- F1 score,
- parameter count,
- runtime or inference cost,
- missing-modality robustness.

---

## 12. Stage J: Final Reporting

### Final deliverables
1. Raw synchronization audit report.
2. Clean synchronization subset report.
3. Train/val/test split report.
4. Model comparison report.
5. Robustness report.
6. Final model summary.
7. Final conclusion on whether the project is valid for synchronized multimodal fusion.

### Final conclusion rule
State clearly:
- how many raw records were fully synchronized,
- how many were excluded,
- which model was best,
- and whether the dataset supports a synchronized fusion architecture.

---

## 13. Python/Colab Rules

When running in Google Colab:
1. Mount Google Drive.
2. Read raw files from Drive.
3. Save all intermediate and final outputs to the project folder.
4. Never overwrite raw files.
5. Keep logs and CSV reports for every stage.
6. Use pandas for tabular operations and PyTorch/TensorFlow for modeling as appropriate.

---

## 14. Important Decision Rules

- Use the fully synchronized subset for the main synchronized fusion model.
- Keep incomplete samples out of synchronized training unless the model explicitly handles missing modalities.
- Do not mix raw audit rows with extracted window rows.
- Do not split by row.
- Do not train on mislabeled or mismatched windows.

---

## 15. Recommended Final Model Choice

If you need one final architecture recommendation, use:

- modality-specific encoders,
- temporal projection,
- cross-attention fusion,
- gating/router,
- sparse mixture-of-experts,
- classification head.

This is the best balance of novelty, robustness, and practical performance for partially synchronized multimodal stress data.