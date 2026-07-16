# End-to-End Temporal Model Pipeline

## 1. Purpose

This instruction file defines a complete training, validation, and evaluation pipeline for temporal classification models using the already extracted 2-second, 5-second, and 10-second datasets. The agent must run every model end to end, save all performance plots and metrics, and store the outputs in a dedicated folder for each model and each window size [web:646][web:654].

## 2. Input datasets

The agent must use the extracted datasets only:
- 2-second windows.
- 5-second windows.
- 10-second windows.

Each dataset must already contain:
- subject ID,
- session ID,
- window ID,
- label,
- timestamp,
- quality flags,
- missing-modality masks if available,
- engineered feature columns,
- and baseline-calibrated features if present.

## 3. Execution rules

### 3.1 Subject-independent evaluation
The agent must split data by subject, not by row. No subject may appear in more than one of train, validation, or test sets [web:654][web:663].

### 3.2 Training only on training data
All scalers, normalizers, feature selectors, and imputers must be fit on training data only, then applied to validation and test sets.

### 3.3 Separate runs
The agent must run all models separately for:
- 2-second data,
- 5-second data,
- 10-second data.

Do not mix the window sizes unless explicitly testing a multi-scale fusion model.

### 3.4 Reproducibility
The agent must fix random seeds, save configurations, save split indices, and log all hyperparameters.

## 4. Folder structure

The agent must create the following structure:

```text
pipeline_runs/
├── configs/
├── logs/
├── models/
├── metrics/
├── plots/
├── reports/
└── outputs/
    ├── 2sec/
    ├── 5sec/
    └── 10sec/
```

Inside each window folder, create one folder per model:

```text
outputs/5sec/TCN/
outputs/5sec/LSTM/
outputs/5sec/Transformer/
```

Each model folder must contain all artifacts for that specific run.

## 5. Model families

The agent must run the following model families:

### 5.1 Classical baselines
- Logistic Regression.
- SVM.
- Random Forest.
- XGBoost.
- KNN.

### 5.2 Temporal deep learning models
- GRU.
- LSTM.
- CNN-LSTM.
- TCN.
- Transformer encoder.

### 5.3 Optional comparison model
- Best production candidate selected from prior experiments.

## 6. Pipeline steps

The agent must follow this exact order for every model and every window size.

### Step 1: Load data
Load the chosen window-size dataset from disk.

### Step 2: Verify schema
Check that required columns exist:
- subject ID,
- session ID,
- window ID,
- label,
- feature columns.

If columns are missing, stop and log the issue.

### Step 3: Define inputs and labels
Separate:
- feature matrix,
- labels,
- metadata columns.

Do not include identifiers like subject ID in the training features.

### Step 4: Create subject splits
Build subject-grouped train, validation, and test sets.

Recommended split:
- train: 70%.
- validation: 15%.
- test: 15%.

If subject count is limited, use grouped cross-validation or LOSO.

### Step 5: Fit preprocessing on train only
Fit on training data only:
- scaler,
- imputer,
- encoder if needed,
- sequence padding rules if needed.

Then transform validation and test data using the same fitted objects.

### Step 6: Build model
Construct the selected model architecture.

### Step 7: Train model
Train using:
- early stopping,
- best checkpoint saving,
- dropout where appropriate,
- batch normalization where appropriate,
- learning-rate scheduling if useful.

### Step 8: Validate model
Evaluate on the validation set each epoch or fold.

### Step 9: Test model
Run final evaluation on the untouched test set.

### Step 10: Save artifacts
Save:
- trained model,
- predictions,
- probabilities,
- metrics,
- plots,
- config,
- split indices,
- summary report.

## 7. Architecture rules by model

### 7.1 Logistic Regression
Use flattened window features.
- Input: standardized tabular features.
- Output: class probabilities.
- No sequence layers.

### 7.2 SVM
Use standardized flattened features.
- Enable probability estimates if available.
- Use class weighting if imbalance exists.

### 7.3 Random Forest
Use flattened features.
- No scaling required, but preprocessing consistency is still recommended.

### 7.4 XGBoost
Use flattened features.
- Tune depth, learning rate, and estimators.
- Use early stopping if possible.

### 7.5 GRU
Use ordered sequence tensors.
- Input shape: `(timesteps, features)`.
- Apply dropout.
- Apply batch normalization only in compatible non-recurrent parts.

### 7.6 LSTM
Use ordered sequence tensors.
- One or more LSTM layers.
- Dropout between layers.
- Dense classification head.

### 7.7 CNN-LSTM
Use local pattern extraction plus sequence memory.
- Conv1D -> BatchNorm -> activation -> pooling.
- LSTM on extracted temporal maps.
- Dense head with dropout.

### 7.8 TCN
Use causal temporal convolutions.
- Stacked Conv1D blocks.
- Dilations and residual blocks.
- BatchNorm in convolution blocks.
- Global pooling.
- Dense classifier.

### 7.9 Transformer encoder
Use self-attention for temporal dependencies.
- Input projection.
- Positional encoding.
- Transformer encoder blocks.
- Layer norm or appropriate temporal normalization.
- Dense classifier head.

## 8. Batch normalization guidance

The agent must use batch normalization only where it improves optimization:
- after Conv1D layers,
- in feed-forward blocks,
- in hybrid CNN-based models.

Do not blindly add batch normalization inside recurrent core logic unless the architecture supports it cleanly.

## 9. Training configuration

Use these defaults unless overridden:
- optimizer: Adam.
- loss: categorical cross-entropy or binary cross-entropy as required.
- early stopping patience: 10.
- checkpoint: monitor validation F1 or validation loss.
- dropout: 0.2 to 0.5.
- batch size: tune by model and memory.
- epochs: sufficiently large with early stopping.
- class weights: enable when imbalance exists.

## 10. Metrics to compute

For every model, compute:
- accuracy,
- precision,
- recall,
- F1-score,
- balanced accuracy,
- ROC-AUC,
- confusion matrix,
- per-class precision/recall/F1,
- mean and std across folds if cross-validation is used.

If binary classification or probability scores are available, also compute:
- precision-recall curve,
- ROC curve,
- threshold sweep,
- calibration curve.

## 11. Required plots

For every model and every window size, save these images:

- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `learning_curve.png`
- `fold_metrics.png`
- `class_distribution.png`
- `calibration_curve.png` if applicable

Each plot must be saved in the corresponding model folder.

## 12. Output folder template

For each run, store artifacts like this:

```text
outputs/10sec/TCN/
├── model.pkl or model.h5 or model.pt
├── predictions.csv
├── metrics.csv
├── fold_results.csv
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── learning_curve.png
├── fold_metrics.png
├── class_distribution.png
├── calibration_curve.png
├── config.json
├── split_indices.json
└── summary.md
```

## 13. Comparison strategy

The agent must compare models only after all runs finish.

The comparison report must rank:
- best model per window size,
- best overall model,
- best classical baseline,
- best deep temporal model,
- and the most stable model across folds.

The report must also include:
- runtime,
- inference speed,
- memory footprint,
- and stability across subject folds.

## 14. Selection logic

The final selected model should not be the one with the highest training score. It must be the one with the best held-out subject performance and stable fold results.

## 15. Failure handling

If any model fails:
- log the reason,
- save partial outputs if available,
- continue with the remaining models,
- mark the failed run in the final report.

## 16. Final report

At the end, the agent must create a consolidated report containing:
- dataset used,
- window size,
- model name,
- metrics,
- selected best model,
- plot locations,
- and a short recommendation for deployment.

## 17. Final instruction

The agent must execute the pipeline exactly as defined above for the 2-second, 5-second, and 10-second datasets, use subject-independent validation, apply batch normalization only where appropriate, and save every metric and image inside the matching model folder.