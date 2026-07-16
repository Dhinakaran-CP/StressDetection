# Model Training Pipeline Agent Instructions

## 1. Purpose

This file instructs the agent to run the complete model training and evaluation pipeline using the already extracted 2-second, 5-second, and 10-second datasets. The pipeline must cover classical models, unimodal deep models, multimodal fusion models, and advanced production-grade models, while producing validation plots and metrics for each run [web:400][web:606][web:609].

## 2. Input data

The agent must load the following prepared feature datasets:
- `2sec` extracted dataset.
- `5sec` extracted dataset.
- `10sec` extracted dataset.

Each dataset must already contain:
- subject ID,
- session ID,
- window ID,
- label,
- timestamps,
- feature columns,
- baseline-calibrated columns if available,
- quality flags,
- missing-modality masks if available.

## 3. General execution rules

### 3.1 One shared workflow
The agent must use one common training workflow for all models:
1. load dataset,
2. select modality view,
3. split by subject,
4. train model,
5. validate,
6. test,
7. save metrics and figures,
8. write a summary report.

### 3.2 No leakage
The agent must prevent leakage by ensuring:
- subject-independent splits,
- no data from a test subject in training,
- no overlap between train/validation/test windows,
- no split after augmentation if it changes identity.

### 3.3 Keep window sizes separate
The agent must run the entire pipeline separately for:
- 2-second features,
- 5-second features,
- 10-second features.

Do not mix them in the same training run unless explicitly testing a multi-scale fusion model.

## 4. Project folder structure

The agent must create the following structure:

```text
loso_evaluation_results/
├── README.md
├── configs/
├── data_links/
├── logs/
├── metrics/
├── plots/
├── reports/
├── models/
│   ├── classical/
│   ├── unimodal_deep/
│   ├── fusion/
│   └── production/
├── outputs/
│   ├── 2sec/
│   ├── 5sec/
│   └── 10sec/
└── docs/
```

Inside each window-size folder, create subfolders for each model:
```text
outputs/5sec/<model_name>/
```

## 5. Models to run

### 5.1 Classical models
Run:
- Logistic Regression,
- SVM,
- Random Forest,
- XGBoost,
- KNN.

### 5.2 Unimodal deep models
Run:
- CNN for face,
- Audio CNN or spectrogram model,
- GRU/LSTM for physiology,
- Transformer if sequence format is available.

### 5.3 Fusion models
Run:
- early fusion,
- late fusion,
- gated fusion,
- cross-attention fusion,
- mixture-of-experts,
- confidence-aware fusion.

### 5.4 Production-grade models
Run the most robust selected model family from prior experiments as the final candidate.

## 6. Validation plan

The agent must run the following validations for each model and each window size:

### 6.1 Subject-independent validation
Use LOSO if possible. If not possible, use grouped subject folds.

### 6.2 Internal validation
Use a validation split inside the training subjects only.

### 6.3 Optional robustness validation
If missing-modality or quality-mask data exists, test:
- full-modality subset,
- partial-modality subset,
- quality-filtered subset.

## 7. Metrics to compute

For every model, compute:
- accuracy,
- precision,
- recall,
- F1-score,
- ROC-AUC if applicable,
- balanced accuracy,
- confusion matrix,
- per-fold mean and std.

If probability outputs are available, also compute:
- calibration curve,
- confidence distribution,
- threshold sweep metrics.

## 8. Plot outputs

For every model and window size, generate and save:
- confusion matrix image,
- ROC curve,
- precision-recall curve,
- training curve if the model supports epochs,
- fold metric plot,
- class distribution plot,
- optional calibration plot.

All images must be stored inside the folder for that specific model.

## 9. Output folder rule

For each run, save outputs like this:

```text
outputs/5sec/RandomForest/
├── metrics.csv
├── confusion_matrix.png
├── roc_curve.png
├── pr_curve.png
├── fold_results.csv
├── predictions.csv
├── summary.md
```

Use the same pattern for all models and all window sizes.

## 10. Execution order

The agent must process the pipeline in this order:

1. 2-second dataset.
2. 5-second dataset.
3. 10-second dataset.

For each window size:
1. run classical models,
2. run unimodal deep models,
3. run multimodal fusion models,
4. run production-grade candidate models,
5. save all results.

## 11. Required logging

The agent must log:
- dataset version used,
- window size,
- model name,
- feature set name,
- split strategy,
- seed,
- metric summary,
- runtime duration,
- any failure or skipped step.

## 12. Reproducibility rules

The agent must:
- fix random seeds,
- save configuration files,
- store feature column lists,
- store training indices,
- store validation indices,
- store test indices,
- and record all hyperparameters used.

## 13. Best practice for comparing window sizes

The agent must not compare 2s, 5s, and 10s models by raw numbers only. It must also report:
- stability across folds,
- training time,
- inference time,
- and memory usage.

This is important because smaller windows may capture more detail but can also increase noise, while longer windows may improve stability but reduce temporal sensitivity [web:400][web:608].

## 14. Final summary report

At the end of the pipeline, the agent must generate a summary report that includes:
- best model per window size,
- best overall model,
- best modality or fusion strategy,
- recommendation for final deployment,
- and observed trade-offs.

## 15. Final instruction

The agent must use the already extracted feature files and run the complete evaluation pipeline for all model families at 2-second, 5-second, and 10-second resolutions, saving every model’s metrics and plots into its own named output folder. The pipeline must be strict about subject independence and reproducibility, because the next phases depend on these results being reliable [web:400][web:606][web:609].