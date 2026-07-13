# Robust Multimodal Stress Detection Agent Instructions
# Architecture: Mask-Aware FlexiModal MoE with Modality Dropout

## 1. Project Objective
Build a robust multimodal stress detection model using Face/Video, Voice/Audio, and Physio signals. The dataset is partially synchronized and contains missing modalities, so the model must:
- avoid shortcut learning from sensor presence/absence,
- handle missing modalities at training and inference,
- remain valid under subject-independent evaluation,
- and support multiple modality combinations without failure.

---

## 2. Core Architectural Principle
Use a **mask-aware FlexiModal MoE architecture** with:
- modality-specific encoders,
- learned modality mask embeddings,
- modality dropout during training,
- masked cross-attention,
- sparse expert routing,
- and final prediction head.

This architecture must not rely on fixed all-modal input availability.

---

## 3. Why This Architecture
The dataset has:
- synchronized subsets,
- partially missing modalities,
- and label skew caused by modality availability.

To prevent the model from learning shortcuts, the training pipeline must explicitly randomize modality availability during training. The model should learn stress patterns from features, not from whether a sensor is present.

---

## 4. Required Folder Structure
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
│   ├── 02_window_alignment.ipynb
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
│   ├── fleximoe/
│   └── robust_dropout/
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

## 5. Stage 1: Raw Data Audit
### Goal
Verify raw video, audio, and physio files before extraction.

### Tasks
1. Recursively scan the raw folder in Google Drive.
2. Group files by subject and task.
3. Detect modality presence.
4. Measure durations.
5. Detect missing modalities.
6. Save audit logs and CSV files.

### Output
- total raw subject-task pairs,
- counts of complete/incomplete recordings,
- duration mismatch statistics,
- file-level audit report.

---

## 6. Stage 2: Synchronization Verification
### Goal
Build the clean synchronized subset.

### Tasks
1. Normalize `task_id` to lowercase.
2. Build sync keys using:
   - `subject_id`
   - `task_id`
   - `window_index`
3. Keep only rows present in all modalities.
4. Check `window_start`, `window_end`, and label agreement.
5. Save:
   - synchronized subset,
   - missing-modality subset,
   - mismatch report.

### Important
Do not train the main synchronized model on mismatched rows.

---

## 7. Stage 3: Data Leakage Protection
### Goal
Prevent shortcut learning from modality availability.

### Tasks
1. Detect whether a modality’s presence correlates with the label.
2. Apply **modality dropout** during training.
3. Randomly mask entire modalities with probability 0.3 or a tuned value.
4. Ensure at least one modality remains active per sample.
5. Do not allow the model to infer the label from missingness patterns.

### Rule
Modality dropout is mandatory for training the robust architecture.

---

## 8. Stage 4: Feature Preparation
### Goal
Prepare model-ready features with explicit missingness handling.

### Tasks
1. Encode each modality separately.
2. Normalize features using training statistics only.
3. Convert each modality to the same latent dimension.
4. Create binary modality masks.
5. Create learned missing embeddings for absent modalities.
6. Save processed train/val/test tensors or tables.

### Important
Missing modalities must be represented explicitly, not silently removed.

---

## 9. Stage 5: Model Architecture
### Final architecture to implement
Use the following structure:

#### A. Modality Encoders
- Video encoder
- Audio encoder
- Physio encoder

#### B. Mask Embeddings
- Add learned mask tokens for missing modalities.
- Use masks to inform the network that a modality is absent.

#### C. Masked Cross-Attention
- Apply cross-attention only between available modalities.
- Do not attend to missing inputs.

#### D. Sparse FlexiModal MoE
- Use modality-specific expert routing.
- Support arbitrary modality combinations.
- Use top-k expert selection.
- Add load-balancing or entropy regularization if implemented.

#### E. Fusion Head
- Merge fused latent vectors.
- Predict stress class.

---

## 10. Stage 6: Training Strategy
### Tasks
1. Train on the synchronized subset.
2. Apply modality dropout during training.
3. Include missing-modality combinations in training batches where possible.
4. Use early stopping on validation F1.
5. Save best checkpoints.
6. Log all metrics per epoch.

### Required losses
- main classification loss,
- optional modality consistency loss,
- optional gating regularization loss.

---

## 11. Stage 7: Validation and Testing
### Tasks
Evaluate on:
- complete samples,
- Face-missing samples,
- Audio-missing samples,
- Physio-missing samples,
- multiple missing combinations.

### Metrics
Report:
- accuracy,
- precision,
- recall,
- F1 score,
- confusion matrix,
- per-class metrics.

### Expected result
The model should still produce predictions even if one modality is unavailable.

---

## 12. Stage 8: Baselines and Ablations
Train and compare:
- Early fusion baseline
- Gated fusion baseline
- Cross-attention fusion
- MoE fusion
- Final robust FlexiModal model with modality dropout

### Required comparison
For each model report:
- validation F1,
- test F1,
- robustness to missing modalities,
- number of parameters,
- inference cost.

---

## 13. Stage 9: Final Model Selection
Select the final model based on:
- best validation F1,
- best test F1,
- lowest performance drop under missing modalities,
- cleanest routing interpretation,
- stability across random seeds.

### Final recommendation
Prefer the model that remains usable when any single modality is missing.

---

## 14. Stage 10: Final Deliverables
Produce:
- raw synchronization audit report,
- synchronized subset report,
- modality dropout training report,
- ablation study report,
- final test report,
- final selected model checkpoint,
- prediction CSV,
- final markdown summary.

---

## 15. Hard Rules
- Never assume all modalities will always be present.
- Never let missingness itself become a label shortcut.
- Never train the full model only on complete data if you want robustness.
- Never drop incomplete samples without reporting them.
- Always preserve raw, interim, processed, and output data separately.
- Always split by subject to avoid leakage.

---

## 16. Final Model Expectation
The final model should be able to:
- train on synchronized complete data,
- learn from partially available modality patterns,
- avoid shortcut learning,
- and still produce valid predictions when one or more modalities are missing.