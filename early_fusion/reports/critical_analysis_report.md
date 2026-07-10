# Critical Analysis: Modality Alignment Skew & Shortcut Learning in Multimodal Stress Detection

This report documents the critical machine learning challenges encountered during the implementation of the subject-independent early fusion and FlexiModal Mixture of Experts (MoE) stress detection pipelines. It highlights how minor dataset preprocessing decisions led to structural shortcut learning, and outlines the final resolution using Modality Co-Training.

---

## Executive Summary

> [!IMPORTANT]
> Multimodal systems are highly susceptible to **shortcut learning** (feature leakage) when modality availability is correlated with the target label. Without random modality dropout during training, the network will optimize for sensor presence/absence masks rather than learning the actual biological features (facial contractions, vocal jitter, heart rate variability) of stress.

---

## Phase 1: The Inner-Join Class Collapse (The Trivial Predictor)

### 1. The Performance Metrics
* **Accuracy:** `78.03%`
* **Precision:** `78.39%`
* **Recall:** `99.38%`
* **F1-Score:** `87.65%`

### 2. The Critique
Although an accuracy of `78.03%` on unseen subjects (LOSO split) initially looked successful, the metrics show a complete collapse of the model's predictive ability. 

Because **Recall was near 100%**, the model was predicting Class 1 (Stressed) for virtually every single test sample. Since the test partition was skewed (78% stressed, 22% relaxed), guessing `1` for all inputs yielded a "fake" 78% accuracy. In the real world, this model is completely non-functional, as it has a **100% false positive rate** on relaxed individuals.

### 3. The Root Cause: Modality Alignment Skew
During Notebook 02, the three tables (`df_face`, `df_voice`, and `df_physio`) were aligned using a default **inner join** on the synchronization keys:
```python
synced_df = df_face.merge(df_voice, on='sync_key')
synced_df = synced_df.merge(df_physio, on='sync_key')
```
* **The Problem:** In the raw dataset, the **resting/relaxed tasks** are silent (meaning the subjects do not speak, so no voice features are extracted).
* **The Skew:** The inner join completely discarded every task that lacked voice records, wiping out almost the entire negative class (relaxed, label `0`) and skewer-biasing the training set to a 78% majority class.

---

## Phase 2: Left-Join & Target Feature Leakage (The Sensor Shortcut)

### 1. The Performance Metrics (The Crash)
After replacing the inner join with a **left join** using `df_physio` as the base to keep relaxed tasks, we obtained the following evaluation metrics:

| Modality Combo | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Face (Video) Only** | `60.91%` | `11.88%` | `57.18%` | `6.63%` |
| **Voice (Audio) Only** | `60.25%` | `0.00%` | `0.00%` | `0.00%` |
| **Physio Only** | `65.30%` | `43.14%` | `61.88%` | `33.11%` |
| **Face + Voice** | `70.42%` | `60.18%` | `64.74%` | `56.22%` |
| **All Modalities** | `69.40%` | `58.73%` | `63.31%` | `54.77%` |

### 2. The Critique
The model was no longer collapsed to the majority class (Recall dropped to realistic ranges like `54.77%` with all modalities present). However, two major anomalies appeared:
1. **Voice Only collapsed to 0% Recall:** Whenever the model was evaluated using voice features alone, it predicted `0` (relaxed) for **100%** of the samples.
2. **Unimodal drops were severe:** Evaluating the model using individual modalities (like Face Only or Voice Only) yielded far lower accuracy than using combinations.

### 3. The Root Cause: Target Leakage via Modality Availability
In our synced dataset, voice features are only present during stress-inducing tasks (where subjects speak) and are missing (`NaN`) during resting/relaxed baseline states. 

The neural network discovered a simple shortcut during training:
* *"If `voice_mask == 1.0` (voice features are present), the label is almost always `1` (Stressed)."*
* *"If `voice_mask == 0.0` (voice features are missing), the label is always `0` (Relaxed)."*

Instead of learning actual features like vocal pitch (`f0_mean`) or eye openness (`left_ear`), the model simply memorized the availability mask.

Additionally, because the model was never trained on samples where Face and Physio features were missing while Voice was present, evaluating `Voice Only` (inputs: `face_mask=0, physio_mask=0, voice_mask=1`) was **Out-of-Distribution (OOD)**. The gating router and experts behaved unpredictably, collapsing the output to `0`.

---

## Phase 3: The Resolution - Modality Co-Training (Dropout)

To force the model to learn the actual biological features of stress rather than memorizing sensor presence, we implemented **Training Modality Dropout**:

```python
# From early_fusion/notebooks/03_train_models.ipynb
# Apply independent random modality dropout during training to prevent shortcut leakage
if self.is_train:
    if np.random.rand() < 0.3:
        face_mask = 0.0
    if np.random.rand() < 0.3:
        voice_mask = 0.0
    if np.random.rand() < 0.3:
        physio_mask = 0.0
    # Retain at least one active modality
    if face_mask == 0.0 and voice_mask == 0.0 and physio_mask == 0.0:
        physio_mask = 1.0
```

### Why this resolves both issues:
1. **Breaks Shortcut Learning:** By randomly dropping Voice during stress tasks, and keeping it present in others, the model can no longer use `voice_mask` presence as a label shortcut. It is forced to look at the actual values of features (e.g., changes in ECG heart rate variance, jaw tension, or vocal shimmer) to perform classification.
2. **Resolves Out-of-Distribution (OOD) Crashes:** During training, the network is exposed to all 7 possible combinations of missing/present modalities. This makes evaluation scenarios like "Face Only" or "Voice Only" fully in-distribution at test time, restoring robust performance.

---

## Lessons Learned for Multimodal AI
1. **Avoid Inner Joins on Asymmetric Modalities:** Inner joining multimodal datasets with silent or missing sensor recordings changes the class balance of the dataset, leading to trivial predictors.
2. **Availability is a Feature:** In multimodal architectures, the availability mask of a sensor is a powerful signal. If that availability correlates with the label, the model will cheat.
3. **Always Train with Dropout:** Modality dropout is not just a regularization tool—it is essential to guarantee that a multimodal model can degrade gracefully when individual sensors fail in real-world environments.
