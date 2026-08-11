# Production Pipeline Final Execution Report

**File**: `Final.md`  
**Updated**: July 23, 2026  
**Pipeline**: SSVB-CASA-AIS Multimodal Stress Detection Production Pipeline (`phase3_production/train.py`)

---

## 1. Terminal Execution & Command Details

### Terminal Process Information
* **Terminal Name**: `powershell`
* **Process ID**: `23016`
* **Execution Command**:
  ```powershell
  venv\Scripts\python.exe phase3_production\train.py --exclude-dataset empathicschool
  ```
* **Environment / Interpreter**: `c:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\venv\Scripts\python.exe`
* **CWD**: `c:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML`
* **Device**: `cuda` (NVIDIA GeForce RTX 4070)
* **Training Scope**: 15-Fold Cross-Validation excluding `empathicschool` from training (held out for zero-shot transfer test evaluation).

---

## 2. Pipeline Configuration (`CONFIG`)

```json
{
  "seed": 42,
  "seq_len": 5,
  "batch_size": 256,
  "ssl_epochs": 4,
  "ft_epochs": 8,
  "lr_ssl": 0.001,
  "lr_ft": 0.0005,
  "weight_decay": 0.0001,
  "hidden_dim": 16,
  "modality_dropout": 0.15,
  "noise_std": 0.02,
  "lambda_conf": 0.15,
  "lambda_subj": 0.10,
  "lambda_dataset": 0.10,
  "lambda_attn": 0.05,
  "lambda_ssl": 0.05,
  "grl_alpha_subj": 0.02,
  "grl_alpha_ds": 0.05,
  "dataset_weight_empathicschool": 0.3,
  "dataset_weight_stressid": 1.0,
  "dataset_weight_wesad": 1.0,
  "n_folds": 15,
  "model_type": "cnn_baseline_grl",
  "device": "cuda"
}
```

---

## 3. Exact Terminal Output (Process ID 23016)

```text
Device: cuda
  Excluding dataset from training: empathicschool (held-out transfer test only)
============================================================
  SSVB-CASA-AIS Production Training Pipeline
============================================================
  Config: {
  "seed": 42,
  "seq_len": 5,
  "batch_size": 256,
  "ssl_epochs": 4,
  "ft_epochs": 8,
  "lr_ssl": 0.001,
  "lr_ft": 0.0005,
  "weight_decay": 0.0001,
  "hidden_dim": 16,
  "modality_dropout": 0.15,
  "noise_std": 0.02,
  "lambda_conf": 0.15,
  "lambda_subj": 0.1,
  "lambda_dataset": 0.1,
  "lambda_attn": 0.05,
  "lambda_ssl": 0.05,
  "grl_alpha_subj": 0.02,
  "grl_alpha_ds": 0.05,
  "dataset_weight_empathicschool": 0.3,
  "dataset_weight_stressid": 1.0,
  "dataset_weight_wesad": 1.0,
  "n_folds": 15,
  "model_type": "cnn_baseline_grl",
  "device": "cuda"
}
  Datasets: ['combined']
  Reports: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results
  Checkpoints: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results\checkpoints
  Deploy: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results\deploy

------------------------------------------------------------
  Validating enriched datasets...
  combined: 89113 windows, 91 subjects
  All datasets validated.

============================================================
  Training: COMBINED (89113 windows, 91 subjects)
============================================================
  Subjects: 68 total, 50 multi-class, 18 single-class (train-only)

============================================================
  combined — Fold 1/15 (test: stressid_2ea4)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0367  val AUC: 0.9587
    Epoch 4/8  loss: 0.9637  val AUC: 0.9631
    Epoch 8/8  loss: 0.9304  val AUC: 0.9572
  → Fold 1: ACC=0.6964  F1=0.5278  AUC=0.8645  Conf=1.0000

============================================================
  combined — Fold 2/15 (test: stressid_h8r2)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0367  val AUC: 0.9587
    Epoch 4/8  loss: 0.9637  val AUC: 0.9631
    Epoch 8/8  loss: 0.9304  val AUC: 0.9572
  → Fold 2: ACC=0.9391  F1=0.9231  AUC=0.9631  Conf=1.0000

============================================================
  combined — Fold 3/15 (test: stressid_h8s1)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0741  val AUC: 1.0000
    Epoch 4/8  loss: 0.9735  val AUC: 1.0000
    Epoch 8/8  loss: 0.9477  val AUC: 1.0000
  → Fold 3: ACC=1.0000  F1=1.0000  AUC=1.0000  Conf=1.0000

============================================================
  combined — Fold 4/15 (test: stressid_x1q3)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0307  val AUC: 0.8303
    Epoch 4/8  loss: 0.9627  val AUC: 0.7091
    Epoch 8/8  loss: 0.9299  val AUC: 0.7624
  → Fold 4: ACC=0.7173  F1=0.7368  AUC=0.8371  Conf=1.0000

============================================================
  combined — Fold 5/15 (test: stressid_qw5t)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0286  val AUC: 0.8241
    Epoch 4/8  loss: 0.9539  val AUC: 0.7688
    Epoch 8/8  loss: 0.9232  val AUC: 0.7832
  → Fold 5: ACC=0.6250  F1=0.6182  AUC=0.8241  Conf=1.0000

============================================================
  combined — Fold 6/15 (test: stressid_v8mh)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0592  val AUC: 0.9714
    Epoch 4/8  loss: 0.9696  val AUC: 0.9812
    Epoch 8/8  loss: 0.9406  val AUC: 0.9689
  → Fold 6: ACC=0.9405  F1=0.9194  AUC=0.9812  Conf=1.0000

============================================================
  combined — Fold 7/15 (test: stressid_4woj)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0338  val AUC: 0.8720
    Epoch 4/8  loss: 0.9590  val AUC: 0.9281
    Epoch 8/8  loss: 0.9300  val AUC: 0.9496
  → Fold 7: ACC=0.8358  F1=0.7273  AUC=0.9496  Conf=1.0000

============================================================
  combined — Fold 8/15 (test: stressid_9t6n)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0889  val AUC: 0.8044
    Epoch 4/8  loss: 0.9687  val AUC: 0.7923
    Epoch 8/8  loss: 0.9378  val AUC: 0.7734
  → Fold 8: ACC=0.7173  F1=0.7368  AUC=0.8044  Conf=1.0000

============================================================
  combined — Fold 9/15 (test: stressid_71i5)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0605  val AUC: 0.5089
    Epoch 4/8  loss: 0.9583  val AUC: 0.4786
    Epoch 8/8  loss: 0.9313  val AUC: 0.4679
  → Fold 9: ACC=0.5298  F1=0.6274  AUC=0.5089  Conf=1.0000

============================================================
  combined — Fold 10/15 (test: stressid_45lx)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0300  val AUC: 0.9254
    Epoch 4/8  loss: 0.9625  val AUC: 0.9193
    Epoch 8/8  loss: 0.9343  val AUC: 0.9063
  → Fold 10: ACC=0.8869  F1=0.8333  AUC=0.9279  Conf=1.0000

============================================================
  combined — Fold 11/15 (test: stressid_6g6y)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0335  val AUC: 0.7593
    Epoch 4/8  loss: 0.9647  val AUC: 0.6804
    Epoch 8/8  loss: 0.9336  val AUC: 0.5949
  → Fold 11: ACC=0.6310  F1=0.6821  AUC=0.7809  Conf=1.0000

============================================================
  combined — Fold 12/15 (test: wesad_s5)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0272  val AUC: 1.0000
    Epoch 4/8  loss: 0.9597  val AUC: 1.0000
    Epoch 8/8  loss: 0.9307  val AUC: 1.0000
  → Fold 12: ACC=0.6504  F1=0.0000  AUC=1.0000  Conf=1.0000

============================================================
  combined — Fold 13/15 (test: wesad_s2)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0401  val AUC: 0.8920
    Epoch 4/8  loss: 0.9623  val AUC: 0.9452
    Epoch 8/8  loss: 0.9308  val AUC: 0.9377
  → Fold 13: ACC=0.8324  F1=0.6974  AUC=0.9452  Conf=1.0000

============================================================
  combined — Fold 14/15 (test: wesad_s10)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0236  val AUC: 0.5090
    Epoch 4/8  loss: 0.9567  val AUC: 0.5058
    Epoch 8/8  loss: 0.9282  val AUC: 0.5376
  → Fold 14: ACC=0.3727  F1=0.5430  AUC=0.5376  Conf=1.0000

============================================================
  combined — Fold 15/15 (test: wesad_s13)
============================================================
  Stage 2: Supervised fine-tuning (8 epochs)
    Epoch 1/8  loss: 1.0316  val AUC: 0.9541
    Epoch 4/8  loss: 0.9553  val AUC: 0.9384
    Epoch 8/8  loss: 0.9211  val AUC: 0.9635
  → Fold 15: ACC=0.6016  F1=0.6406  AUC=0.9635  Conf=1.0000

  combined — AGGREGATE (15 folds): ACC=0.7160  F1=0.6808  AUC=0.7509

  ============================================================
  Held-out transfer test: empathicschool (66622 windows)
  ============================================================
  Held-out empathicschool: AUC=0.5123  ACC=0.6734  F1=0.1820
    Optimal threshold=0.330: P=0.1656  R=0.9849  F1=0.2835

  Optimal threshold (max F1): 0.340
    At optimal: prec=0.6361  recall=0.8622  f1=0.7321
    thresh=0.20  prec=0.4864  recall=0.9471  f1=0.6427
    thresh=0.32  prec=0.5751  recall=0.8823  f1=0.6963
    thresh=0.33  prec=0.6203  recall=0.8823  f1=0.7284
    thresh=0.34  prec=0.6361  recall=0.8622  f1=0.7321
    thresh=0.50  prec=0.6884  recall=0.6734  f1=0.6808

  [DEPLOY] Production weights saved to C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results\deploy\ssvb_casa_ais_production_cnn_baseline_grl.pt
  [DEPLOY] Per-dataset thresholds: {'stressid': 0.34, 'wesad': 0.33}

============================================================
  FINAL RESULTS — cnn_baseline_grl
============================================================
  stressid              ACC=0.7343  F1=0.7029  AUC=0.7330  Conf=1.0000
  wesad                 ACC=0.7694  F1=0.7050  AUC=0.8582  Conf=1.0000
  combined              ACC=0.7160  F1=0.6808  AUC=0.7509  Conf=1.0000
  combined              ACC=0.7413  F1=0.6961  AUC=0.7870

  Per-subject accuracy: mean=0.6698 std=0.2047
  Per-dataset: {'combined': 0.7161912579519804, 'stressid': 0.6779440468445023, 'wesad': 0.6443719412724307}
  Per-source-dataset (threshold=0.50):
    stressid              AUC=0.7648  F1=0.7175  ACC=0.7460  n=8013
    wesad                 AUC=0.8159  F1=0.6685  ACC=0.7358  n=6988
  Per-source-dataset (optimal threshold per dataset):
    stressid              AUC=0.7648  optF1=0.7277  thresh=0.41  prec=0.7247  rec=0.7307
    wesad                 AUC=0.8159  optF1=0.6685  thresh=0.50  prec=0.6116  rec=0.7370

  Reports: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results\cnn_baseline_grl\combined
  Weights: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\phase3_production\results\deploy/ssvb_casa_ais_production_cnn_baseline_grl.pt
============================================================
```

---

## 4. Overall Combined Summary Metrics

| Metric | Value (Default Threshold 0.50) | Value (Optimal Threshold 0.34) |
| :--- | :---: | :---: |
| **Accuracy** | 71.60% | **71.62%** |
| **Precision** | 68.84% | 63.61% |
| **Recall** | 67.34% | **86.22%** |
| **F1 Score** | 0.6808 | **0.7321** |
| **ROC-AUC** | **0.7509** (0.7870 weighted) | **0.7509** |
| **Average Precision (AP)** | 0.6552 (0.7239 weighted) | 0.6552 |
| **MSE / MAE** | 0.2031 / 0.3982 | 0.2031 / 0.3982 |
| **Mean Confidence** | 1.0000 | 1.0000 |

---

## 5. Dataset 1: `stressid` Performance Output & Fold Breakdown

### `stressid` Summary Metrics
* **Total Windows Evaluated**: 4,611
* **Accuracy**: `73.43%`
* **Precision**: `0.7188`
* **Recall**: `0.6877`
* **F1-Score**: `0.7029` (In-domain optimal threshold F1: `0.7277` @ `0.41`)
* **ROC-AUC**: `0.7330` (`0.7648` per-source)
* **Average Precision (AP)**: `0.6882`
* **MSE / MAE**: `0.1990` / `0.3931`

### `stressid` Classification Report
```text
Classification Report - cnn_baseline_grl (stressid)
==================================================

              precision    recall  f1-score   support

        Calm     0.7464    0.7736    0.7598      2504
      Stress     0.7188    0.6877    0.7029      2107

    accuracy                         0.7343      4611
   macro avg     0.7326    0.7306    0.7313      4611
weighted avg     0.7338    0.7343    0.7338      4611

AUC-ROC: 0.7330
Average Precision (AP): 0.6882
```

### `stressid` 15-Fold Cross-Validation Breakdown Table

| Fold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision | MSE | MAE | Mean Conf |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.6548 | 0.3295 | 1.0000 | 0.4957 | 0.8569 | 0.3892 | 0.2497 | 0.4522 | 1.0000 |
| 2 | 0.9391 | 0.8571 | 1.0000 | 0.9231 | 0.9615 | 0.9172 | 0.1285 | 0.3439 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0825 | 0.2807 | 1.0000 |
| 4 | 0.7173 | 1.0000 | 0.5833 | 0.7368 | 0.8006 | 0.9101 | 0.2089 | 0.3772 | 1.0000 |
| 5 | 0.7173 | 1.0000 | 0.5833 | 0.7368 | 0.7903 | 0.9102 | 0.2092 | 0.4178 | 1.0000 |
| 6 | 0.9435 | 0.8571 | 1.0000 | 0.9231 | 0.9773 | 0.9477 | 0.0856 | 0.2618 | 1.0000 |
| 7 | 0.8358 | 0.5714 | 1.0000 | 0.7273 | 0.9246 | 0.6983 | 0.1462 | 0.3183 | 1.0000 |
| 8 | 0.7173 | 1.0000 | 0.5833 | 0.7368 | 0.8653 | 0.9450 | 0.2162 | 0.4241 | 1.0000 |
| 9 | 0.4524 | 0.7305 | 0.4137 | 0.5282 | 0.5376 | 0.7377 | 0.3173 | 0.5198 | 1.0000 |
| 10 | 0.8869 | 0.7143 | 1.0000 | 0.8333 | 0.9307 | 0.7402 | 0.1322 | 0.3377 | 1.0000 |
| 11 | 0.6339 | 0.8261 | 0.5833 | 0.6838 | 0.7541 | 0.8952 | 0.2034 | 0.4241 | 1.0000 |
| 12 | 0.4776 | 0.8571 | 0.4125 | 0.5570 | 0.6309 | 0.8794 | 0.4150 | 0.5659 | 1.0000 |
| 13 | 0.7738 | 0.7143 | 0.7143 | 0.7143 | 0.7455 | 0.6474 | 0.1802 | 0.3578 | 1.0000 |
| 14 | 0.6935 | 0.4798 | 1.0000 | 0.6485 | 0.9434 | 0.8359 | 0.2073 | 0.3926 | 1.0000 |
| 15 | 0.6310 | 0.3149 | 1.0000 | 0.4790 | 0.9376 | 0.6863 | 0.2161 | 0.4137 | 1.0000 |

---

## 6. Dataset 2: `wesad` Performance Output & Fold Breakdown

### `wesad` Summary Metrics
* **Total Windows Evaluated**: 5,517
* **Accuracy**: `76.94%`
* **Precision**: `0.6560`
* **Recall**: `0.7619`
* **F1-Score**: `0.7050` (In-domain optimal threshold F1: `0.6685` @ `0.50`)
* **ROC-AUC**: `0.8582` (`0.8159` per-source)
* **Average Precision (AP)**: `0.8187`
* **MSE / MAE**: `0.1558` / `0.3404`

### `wesad` Classification Report
```text
Classification Report - cnn_baseline_grl (wesad)
==================================================

              precision    recall  f1-score   support

        Calm     0.8516    0.7737    0.8108      3522
      Stress     0.6560    0.7619    0.7050      1995

    accuracy                         0.7694      5517
   macro avg     0.7538    0.7678    0.7579      5517
weighted avg     0.7809    0.7694    0.7725      5517

AUC-ROC: 0.8582
Average Precision (AP): 0.8187
```

### `wesad` 15-Fold Cross-Validation Breakdown Table

| Fold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision | MSE | MAE | Mean Conf |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.7737 | 1.0000 | 0.3622 | 0.5318 | 0.9913 | 0.9824 | 0.1322 | 0.2397 | 1.0000 |
| 2 | 0.8169 | 0.8539 | 0.5846 | 0.6941 | 0.9276 | 0.8350 | 0.1301 | 0.3074 | 1.0000 |
| 3 | 0.8556 | 0.9327 | 0.6690 | 0.7791 | 0.9351 | 0.9154 | 0.1448 | 0.3592 | 1.0000 |
| 4 | 0.9484 | 0.9752 | 0.8806 | 0.9255 | 0.9957 | 0.9922 | 0.1708 | 0.4104 | 1.0000 |
| 5 | 0.7035 | 0.5510 | 1.0000 | 0.7105 | 1.0000 | 1.0000 | 0.1771 | 0.3778 | 1.0000 |
| 6 | 0.8933 | 1.0000 | 0.7031 | 0.8257 | 0.9968 | 0.9941 | 0.0704 | 0.1727 | 1.0000 |
| 7 | 0.7100 | 0.5542 | 1.0000 | 0.7131 | 0.9946 | 0.9914 | 0.1798 | 0.3671 | 1.0000 |
| 8 | 0.3978 | 0.3778 | 1.0000 | 0.5484 | 1.0000 | 1.0000 | 0.1996 | 0.4041 | 1.0000 |
| 9 | 0.7397 | 0.8864 | 0.3023 | 0.4509 | 0.9279 | 0.8622 | 0.1548 | 0.3312 | 1.0000 |
| 10 | 0.8874 | 0.9286 | 0.7536 | 0.8320 | 0.9540 | 0.9373 | 0.1464 | 0.3706 | 1.0000 |
| 11 | 0.6818 | 0.6410 | 0.2033 | 0.3086 | 0.6887 | 0.5251 | 0.2136 | 0.4464 | 1.0000 |
| 12 | 0.9539 | 0.9746 | 0.8915 | 0.9312 | 0.9968 | 0.9935 | 0.1040 | 0.3117 | 1.0000 |
| 13 | 0.3507 | 0.3507 | 1.0000 | 0.5193 | 0.9866 | 0.9795 | 0.3111 | 0.4874 | 1.0000 |
| 14 | 0.8571 | 0.7181 | 1.0000 | 0.8359 | 1.0000 | 1.0000 | 0.1632 | 0.3767 | 1.0000 |
| 15 | 0.9606 | 0.9167 | 0.9862 | 0.9502 | 0.9893 | 0.9808 | 0.0425 | 0.1453 | 1.0000 |

---

## 7. Held-Out Transfer Test (`empathicschool` — 66,622 windows)

* **Zero-Shot Transfer AUC**: `0.5123`
* **Accuracy / F1**: `ACC=0.6734`, `F1=0.1820`
* **Optimal Transfer Threshold (0.330)**: `Precision=0.1656`, `Recall=0.9849`, `F1=0.2835`

---

## 8. Full Combined Set Classification Report (15,001 Windows)

```text
Classification Report - cnn_baseline_grl (All Datasets)
==================================================

              precision    recall  f1-score   support

        Calm     0.8244    0.5658    0.6710      8707
      Stress     0.5811    0.8333    0.6847      6294

    accuracy                         0.6780     15001
   macro avg     0.7028    0.6995    0.6779     15001
weighted avg     0.7223    0.6780    0.6768     15001

AUC-ROC: 0.7870
Average Precision (AP): 0.7239
```

---

## 9. Combined Cross-Validation 15-Fold Breakdown Table

| Fold | Test Subject/Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision | MSE | MAE | Mean Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `stressid_2ea4` | 0.6964 | 0.3585 | 1.0000 | 0.5278 | 0.8645 | 0.4179 | 0.1998 | 0.3862 | 1.0000 |
| 2 | `stressid_h8r2` | 0.9391 | 0.8571 | 1.0000 | 0.9231 | 0.9631 | 0.9130 | 0.1117 | 0.3137 | 1.0000 |
| 3 | `stressid_h8s1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0835 | 0.2771 | 1.0000 |
| 4 | `stressid_x1q3` | 0.7173 | 1.0000 | 0.5833 | 0.7368 | 0.8371 | 0.9289 | 0.2190 | 0.3976 | 1.0000 |
| 5 | `stressid_qw5t` | 0.6250 | 1.0000 | 0.4474 | 0.6182 | 0.8241 | 0.9292 | 0.2549 | 0.4448 | 1.0000 |
| 6 | `stressid_v8mh` | 0.9405 | 0.8507 | 1.0000 | 0.9194 | 0.9812 | 0.9627 | 0.0808 | 0.2537 | 1.0000 |
| 7 | `stressid_4woj` | 0.8358 | 0.5714 | 1.0000 | 0.7273 | 0.9496 | 0.8061 | 0.1307 | 0.3013 | 1.0000 |
| 8 | `stressid_9t6n` | 0.7173 | 1.0000 | 0.5833 | 0.7368 | 0.8044 | 0.9157 | 0.1911 | 0.4151 | 1.0000 |
| 9 | `stressid_71i5` | 0.5298 | 0.7600 | 0.5341 | 0.6274 | 0.5089 | 0.7829 | 0.2582 | 0.4613 | 1.0000 |
| 10 | `stressid_45lx` | 0.8869 | 0.7143 | 1.0000 | 0.8333 | 0.9279 | 0.7909 | 0.1192 | 0.3153 | 1.0000 |
| 11 | `stressid_6g6y` | 0.6310 | 0.8210 | 0.5833 | 0.6821 | 0.7809 | 0.9045 | 0.2347 | 0.4388 | 1.0000 |
| 12 | `wesad_s5` | 0.6504 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2221 | 0.4442 | 1.0000 |
| 13 | `wesad_s2` | 0.8324 | 0.9444 | 0.5528 | 0.6974 | 0.9452 | 0.8906 | 0.1817 | 0.4193 | 1.0000 |
| 14 | `wesad_s10` | 0.3727 | 0.3757 | 0.9793 | 0.5430 | 0.5376 | 0.5626 | 0.4466 | 0.5840 | 1.0000 |
| 15 | `wesad_s13` | 0.6016 | 0.4746 | 0.9850 | 0.6406 | 0.9635 | 0.9577 | 0.1960 | 0.3970 | 1.0000 |

---

## 10. Saved Checkpoints & Deployment Assets

* **Evaluation Reports & Figures**:
  * Combined: `phase3_production/results/cnn_baseline_grl/combined/`
  * StressID: `phase3_production/results/cnn_baseline_grl/stressid/`
  * WESAD: `phase3_production/results/cnn_baseline_grl/wesad/`
* **Production Model Weights**: `phase3_production/results/deploy/ssvb_casa_ais_production_cnn_baseline_grl.pt`
* **Per-Dataset Deployment Thresholds**:
  * `stressid`: `0.34`
  * `wesad`: `0.33`
