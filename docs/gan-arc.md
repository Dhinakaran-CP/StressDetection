# GAN Experiment Pipeline

## 1. Purpose

This file defines the GAN experiment branch for the project. The agent must test whether synthetic augmentation improves performance for the extracted 2-second, 5-second, and 10-second datasets, while keeping validation and test sets fully real and untouched [web:684][web:682][web:687].

## 2. Core rule

The agent must never generate synthetic data before splitting.  
The correct sequence is:
1. split the real dataset by subject,
2. train GAN only on the training split,
3. generate synthetic samples only for training,
4. retrain the downstream models,
5. compare against the real-only baseline [web:684][web:682][web:688].

## 3. Input datasets

The agent must use only the already extracted datasets:
- 2-second dataset.
- 5-second dataset.
- 10-second dataset.

Each dataset must already contain:
- subject ID,
- session ID,
- window ID,
- label,
- feature columns,
- quality flags,
- missingness masks if available.

## 4. GAN objective

The GAN branch must be used to improve:
- class imbalance,
- minority class coverage,
- robustness to limited data,
- and possibly generalization in difficult window sizes [web:684][web:685][web:689].

The agent must treat GAN as an experiment, not as a guaranteed improvement.

## 5. Recommended GAN types

For this project, the agent must test these GAN styles in order of simplicity:

### 5.1 Conditional tabular GAN
Use this first for extracted feature vectors.  
This is the safest and most practical GAN type for window-based tabular features [web:684][web:688].

### 5.2 Class-conditional GAN
Condition generation on the label so that minority classes can be oversampled.

### 5.3 Window-size-specific GAN
Train one GAN for each of:
- 2-second windows,
- 5-second windows,
- 10-second windows.

Do not mix window sizes in one GAN unless explicitly testing a multi-scale generator.

## 6. Splitting protocol

The agent must split the real dataset by subject before any GAN training.

Recommended split:
- train,
- validation,
- test.

Rules:
- GAN training uses training data only.
- Validation and test sets remain real only.
- No synthetic sample may be inserted into validation or test [web:682][web:684][web:687].

## 7. GAN training protocol

The agent must:
1. fit preprocessing on the training split only,
2. transform training data,
3. train the GAN on transformed training features,
4. monitor generator and discriminator losses,
5. detect mode collapse or instability,
6. stop if generated samples become unrealistic,
7. save checkpoints and logs [web:685][web:689][web:690].

If GAN training is unstable, the agent may try:
- lower learning rate,
- stronger regularization,
- conditional training,
- class-balanced sampling,
- or a simpler GAN variant.

## 8. Synthetic sample generation

After training, the agent must generate synthetic samples only for the training fold.

Synthetic generation rules:
- generate only for the minority or underrepresented classes if imbalance exists,
- preserve the same feature schema,
- keep the same label encoding,
- keep the same window-size context,
- do not create synthetic subject IDs that could cause leakage,
- mark all synthetic rows with `is_synthetic = true`.

## 9. Quality checks

Before using synthetic samples for model training, the agent must perform quality checks:
- compare feature distributions,
- compare label balance,
- compare summary statistics,
- check nearest-neighbor similarity,
- check duplicate or near-duplicate rows,
- check whether synthetic samples are too close to real rows,
- reject low-quality synthetic data [web:684][web:688][web:689].

## 10. Downstream training pipeline

After synthetic data passes quality checks, the agent must rerun the full model training pipeline using:
- real training data + synthetic training data,
- real validation data,
- real test data.

The same models must be retrained:
- Logistic Regression,
- SVM,
- Random Forest,
- XGBoost,
- GRU,
- LSTM,
- CNN-LSTM,
- TCN,
- Transformer encoder.

The agent must keep the pipeline identical so the comparison stays fair.

## 11. Comparison protocol

The agent must compare:
- real-only baseline,
- GAN-augmented training,
- and different GAN variants if tested.

The comparison must be done separately for:
- 2-second windows,
- 5-second windows,
- 10-second windows.

The selected GAN strategy must be the one that improves:
- held-out subject performance,
- minority class recall,
- and overall stability across folds [web:682][web:684][web:687].

## 12. Metrics to generate

For every GAN experiment and every downstream model, compute:
- accuracy,
- precision,
- recall,
- F1-score,
- balanced accuracy,
- ROC-AUC,
- confusion matrix,
- per-class metrics,
- fold mean and std.

For GAN quality itself, also record:
- training loss curves,
- discriminator loss,
- generator loss,
- sample statistics,
- distribution overlap scores if available.

## 13. Required plots

The agent must save these images for each GAN experiment:
- `gan_loss_curve.png`
- `generated_vs_real_distribution.png`
- `feature_similarity.png`
- `sample_quality_check.png`

For each downstream model retrained with GAN data, also save:
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `learning_curve.png`

## 14. Output folder structure

The agent must create separate folders for real-only and GAN experiments:

```text
outputs/
├── real_only/
│   ├── 2sec/
│   ├── 5sec/
│   └── 10sec/
└── gan_augmented/
    ├── 2sec/
    ├── 5sec/
    └── 10sec/
```

Inside each window folder, create one folder per GAN type and model:

```text
outputs/gan_augmented/5sec/CTGAN/TCN/
```

Each folder must contain:
- metrics,
- plots,
- saved model,
- synthetic-data summary,
- and config files.

## 15. Logging and reproducibility

The agent must log:
- GAN type,
- window size,
- training split size,
- synthetic sample count,
- quality decisions,
- downstream model name,
- seed,
- and final performance.

The agent must also save:
- split indices,
- preprocessing config,
- GAN config,
- downstream model config,
- and version numbers.

## 16. Acceptance criteria

The GAN branch is accepted only if it improves performance on real held-out data consistently across folds or subject splits.  
If performance improves on training but not on validation or test, the GAN method must be rejected [web:684][web:682][web:687].

## 17. Final instruction

The agent must follow this pipeline exactly:
1. split real data,
2. train GAN on training data only,
3. generate synthetic training rows,
4. quality-check synthetic rows,
5. retrain all downstream models,
6. evaluate on real validation and test sets,
7. compare against the real-only baseline,
8. keep only the GAN approach that improves generalization.