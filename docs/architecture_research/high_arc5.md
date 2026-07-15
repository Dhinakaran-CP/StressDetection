# Verified-Baseline Contrastive Cross-Attention Stress Architecture with Identity Suppression

## 1. Purpose

This document defines a novelty-oriented multimodal stress detection architecture for strict Leave-One-Subject-Out (LOSO) evaluation. The design combines verified baseline normalization, self-supervised contrastive pretraining, temporal cross-attention fusion, and adversarial identity suppression to improve generalization to unseen users without leaking subject identity [web:415][web:111][web:388].

## 2. Design Goals

- Preserve strict subject independence under LOSO.
- Reduce biometric trait leakage from face, voice, and physiology.
- Learn transferable representations before supervised fine-tuning.
- Align asynchronous modalities using temporal attention.
- Provide confidence, uncertainty, and explanation at inference time.
- Support real-time deployment with lightweight client-side preprocessing.

## 3. Libraries and Methods

### 3.1 Face Pipeline
- **MediaPipe Face Mesh** for browser-side facial landmark extraction.
- **HTML5 Web Worker** for background network posting.
- **Camera throttling** and low-frequency sampling for efficiency.

### 3.2 Voice Pipeline
- **Web Audio API** for browser capture and streaming.
- **Librosa** for pitch estimation, MFCC extraction, and spectral analysis.
- **SciPy** for signal decoding and preprocessing.

### 3.3 Modeling Pipeline
- **PyTorch** for encoders, attention, and adversarial training.
- **Contrastive learning** for self-supervised pretraining.
- **Multi-head self-attention** inside each modality stream.
- **Cross-attention** for multimodal alignment.
- **Gradient Reversal Layer (GRL)** for identity suppression.
- **Auxiliary confidence head** for uncertainty estimation.

## 4. Input Modalities

### 4.1 Face
Use facial landmarks and derived temporal features such as:
- Eye aspect ratio.
- Brow movement.
- Lip compression.
- Jaw motion.
- Head pose variation.
- Facial temporal variance.

### 4.2 Voice
Use acoustic features such as:
- MFCC.
- Pitch contour.
- Jitter.
- Shimmer.
- RMS energy.
- Pause ratio.
- Speaking-rate proxy.
- Spectral flux.

### 4.3 Physiology
Use physiological features such as:
- Heart rate.
- HRV RMSSD.
- HRV SDNN.
- Respiration rate.
- EDA if available.

## 5. System Architecture

### 5.1 Acquisition Layer
Collect face, voice, and physiology from the client or wearable layer. Face and voice should be captured in real time, while physiology can be streamed or batch-processed depending on the device.

### 5.2 Quality Gate
Before any inference, pass each modality through a signal-quality gate:
- Reject low-visibility face windows.
- Reject clipped or silent voice windows.
- Reject incomplete or noisy physiological windows.
- Attach a quality score to every modality.

### 5.3 Verified Baseline Module
Collect a short neutral calibration window for each user. Validate whether it is actually calm before accepting it as baseline. If the baseline looks contaminated, ask for recalibration.

Baseline statistics to store:
- Mean.
- Standard deviation.
- Minimum.
- Maximum.
- Baseline confidence label.

### 5.4 Baseline Normalization
Convert each incoming sample into:
- absolute feature values,
- and baseline-relative deviations.

This ensures the model sees both raw dynamics and user-specific change.

### 5.5 Self-Supervised Pretraining
Pretrain the modality encoders before supervised stress learning.

#### Pretraining input
- Use the certified datasets directly.
- Ignore stress labels during pretraining.
- Preserve LOSO subject separation in later supervised stages.

#### SSL objective
Use **contrastive loss** as the primary objective:
- Positive pair: two windows from the same user/session.
- Negative pair: windows from different users or sessions.

Optional auxiliary objective:
- Masked reconstruction on a small portion of the sequence.

### 5.6 Temporal Encoders
Use a separate encoder for each modality.

#### Face encoder
- 1D CNN or CNN-GRU.
- Add self-attention inside the stream.

#### Voice encoder
- 1D CNN or CNN-GRU.
- Add self-attention inside the stream.

#### Physiology encoder
- Temporal CNN or TCN.
- Add self-attention inside the stream.

### 5.7 Identity Suppression
Attach a subject-classifier head through a **Gradient Reversal Layer**.

Training objectives:
- Main task: predict stress.
- Adversarial task: predict subject ID.
- GRL makes the encoder forget identity-specific features.

### 5.8 Multimodal Fusion
Fuse modality embeddings using **multi-head cross-attention**.

Fusion logic:
- Face attends to voice and physiology.
- Voice attends to face and physiology.
- Physiology attends to face and voice.
- Use modality quality masks to reduce weak-stream influence.

### 5.9 Confidence and Uncertainty
Add an auxiliary confidence head to predict prediction confidence. Use uncertainty to:
- flag weak predictions,
- trigger recalibration,
- or lower model reliance when modalities disagree.

## 6. Training Workflow

### Step 1: Pretraining
Train each modality encoder with contrastive loss on unlabeled or label-ignored windows.

### Step 2: Supervised Fine-Tuning
Fine-tune the full architecture on labeled stress data using strict LOSO splits.

### Step 3: Adversarial Training
Enable GRL subject suppression during supervised training.

### Step 4: Fusion Training
Train the cross-attention fusion module with modality dropout and missing-stream simulation.

### Step 5: Threshold Calibration
Tune stress thresholds only on validation folds, not on the held-out LOSO subject.

## 7. Inference Workflow

1. Collect face, voice, and physiology.
2. Run quality gate.
3. Load or verify user baseline.
4. Normalize features using baseline-relative values.
5. Encode each modality.
6. Apply identity suppression representation.
7. Fuse modalities through cross-attention.
8. Predict stress, confidence, and uncertainty.
9. If confidence is low, request recalibration or additional data.

## 8. Loss Functions

Use the following combined objective:

\[
\mathcal{L}_{total} = \mathcal{L}_{stress} + \lambda_1 \mathcal{L}_{contrastive} + \lambda_2 \mathcal{L}_{attention} - \lambda_3 \mathcal{L}_{subject} + \lambda_4 \mathcal{L}_{confidence}
\]

Where:
- \(\mathcal{L}_{stress}\) is the supervised stress classification loss.
- \(\mathcal{L}_{contrastive}\) is the self-supervised pretraining objective.
- \(\mathcal{L}_{attention}\) encourages temporal alignment.
- \(\mathcal{L}_{subject}\) is the adversarial subject ID loss.
- \(\mathcal{L}_{confidence}\) regularizes uncertainty prediction.

## 9. Evaluation Protocol

Use strict LOSO only.

Report:
- Accuracy.
- Macro F1.
- ROC-AUC.
- Calibration error.
- Subject-wise variance.
- Confidence reliability.

Do not use random row-wise splits for final claims because they leak subject identity.

## 10. Expected Benefits

This design is expected to improve reliability because:
- contrastive pretraining improves transferable representations,
- baseline verification reduces personalization contamination,
- cross-attention handles timing mismatch,
- GRL reduces subject memorization,
- confidence routing improves practical deployment.

## 11. Novelty Statement

The novelty of this architecture is the combination of:
- verified baseline calibration,
- self-supervised contrastive pretraining,
- temporal cross-attention fusion,
- and adversarial identity suppression

under strict LOSO evaluation.

## 12. Final Architecture Name

**Verified-Baseline Contrastive Cross-Attention Stress Architecture with Identity Suppression**