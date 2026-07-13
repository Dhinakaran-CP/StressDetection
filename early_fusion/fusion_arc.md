# Robust Multimodal Stress Detection Agent Instructions
## Goal
Build a multimodal stress detection system that never breaks when one modality is missing. The project must support video, audio, and physio inputs with partial synchronization and missing-modality tolerance.

## Core Design Choice
Use a FlexiModal MoE architecture:
- modality-specific encoders,
- temporal projection,
- missing-modality bank,
- per-modality routers,
- sparse MoE fusion,
- Laplace gating if available,
- final classification head.

## Folder Structure
Create:
- `data/raw/`
- `data/interim/raw_audit/`
- `data/interim/synced_pairs/`
- `data/interim/extracted_windows/`
- `data/interim/quality_checks/`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`
- `data/processed/metadata/`
- `reports/synchronization/`
- `reports/extraction/`
- `reports/training/`
- `reports/evaluation/`
- `reports/figures/`
- `outputs/checkpoints/`
- `outputs/predictions/`
- `outputs/final_model/`
- `logs/`
- `configs/`
- `scripts/`

## Phase 1: Raw Audit
1. Recursively scan raw files in Drive.
2. Group by `subject_id` and `task_id`.
3. Detect modality presence.
4. Measure duration per file.
5. Save raw synchronization audit.
6. Record missing modalities and mismatched durations.

## Phase 2: Alignment and Extraction
1. Normalize `task_id` to lowercase.
2. Build keys using `subject_id + task_id + window_index`.
3. Keep only fully synchronized rows for the main synchronized subset.
4. Verify `window_start`, `window_end`, and label agreement.
5. Save a clean synchronized manifest.

## Phase 3: Split Strategy
1. Split by subject, not by row.
2. Create train/validation/test subject lists.
3. Ensure no subject leakage across splits.
4. Apply the same split to all modalities.

## Phase 4: Feature Preparation
1. Prepare per-modality features or tensors.
2. Convert each modality to a shared latent size.
3. For missing modalities, use learned missing embeddings instead of zero-padding.
4. Save processed data separately for train/val/test.

## Phase 5: Model Training
Train and compare:
- Early fusion baseline.
- Gated fusion baseline.
- Cross-attention fusion.
- MoE fusion.
- Final FlexiModal MoE hybrid.

The final model should include:
- per-modality routers,
- expert specialization,
- generalized training on complete samples,
- specialized handling of incomplete combinations,
- missing-modality bank,
- entropy or load-balancing regularization.

## Phase 6: Validation
Validate after every epoch using:
- accuracy,
- precision,
- recall,
- F1 score,
- confusion matrix,
- per-class metrics.

Use early stopping based on validation F1.

## Phase 7: Missing-Modality Robustness
Test the final model under:
- video missing,
- audio missing,
- physio missing,
- multiple missing combinations.

Report performance drop and robustness stability.

## Phase 8: Final Selection
Choose the best model based on:
- validation F1,
- test F1,
- robustness to missing modalities,
- interpretability of expert routing,
- stability across seeds.

## Important Rules
- Never assume all modalities will be available during inference.
- Never train the main fusion model on mismatched or unsynchronized rows.
- Never rely on fixed concatenation alone.
- Use learned missing embeddings instead of hard zero-padding whenever possible.
- Save every report and model checkpoint.