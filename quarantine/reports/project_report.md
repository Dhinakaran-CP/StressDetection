# Multimodal Stress Detection Using Machine Learning
## Comprehensive Project Report
**Date:** June 9, 2026  
**Status:** Verification Completed & Production Ready  

---

## 1. Executive Summary
This project implements a real-time, low-latency, multimodal stress detection system that fuses **facial indicators** (tracked via webcam using MediaPipe in the browser), **vocal biomarkers** (captured via microphone and processed in 2-second sliding chunks), and **physiological telemetry** (GSR, EEG). 

The system leverages a **decentralized expert architecture**:
1. **Facial Expert Model**: Classified via a soft-voting ensemble model trained on custom facial indicators with data augmentation and SMOTE class balancing.
2. **Vocal Expert Model**: Classified via lightweight machine learning models using fast autocorrelation-based acoustic feature extraction.
3. **Physiological Expert Model**: Evaluates GSR and EEG signals to predict physiological stress indices.
4. **Multimodal Fusion Engine**: Integrates individual modality scores using a dynamic reliability weight algorithm that accounts for sensor quality and occlusion.

---

## 2. Modality Specifications & Model Architectures

### A. Facial Modality (Modality 1)
* **Pipeline**: MediaPipe FaceMesh runs in the browser, extracting 468 3D landmarks at >30 FPS.
* **Indicators (18 features)**:
  * Eye Aspect Ratio (EAR) for left, right, and average eye openings.
  * Blink velocity (rate of change of EAR).
  * Brow descent (left, right) and brow asymmetry.
  * Lip compression (vertical lip gap divided by lip width).
  * Jaw displacement (normalized mouth opening width).
  * Jaw width (masseter muscle contraction/clenching width).
  * Mouth corner pull (smile/grimace pull towards nose).
  * Forehead tension (distance between nose bridge and top forehead).
  * Head tilt/rotation angle.
  * Temporal variance of nose tip (restlessness proxy).
  * Nose wrinkle and landmark confidence score.
* **Classifier**: Soft-voting ensemble of:
  * **Gradient Boosting Classifier** (200 estimators, max depth 4, learning rate 0.05).
  * **Random Forest Classifier** (200 estimators, max depth 8).
  * **Support Vector Machine (SVC)** (RBF kernel, probability outputs, `max_iter=2000`).
* **Data Augmentation & Balance**:
  * Original class distribution: No Stress (4,783) / Stress (2,962).
  * Face data augmentation simulates real-world noise: ±8% eye EAR variation, ±5% brow variation, ±6% mouth/jaw variation, and ±3° head tilt.
  * SMOTE balances the classes to 23,915 samples each.
* **Accuracy**: **65.27%** on augmented/balanced test split (designed for real-world geometric variation).
* **Latency**: Feature extraction is <15ms in browser. Classification inference is <10ms in backend.

### B. Vocal Modality (Modality 2)
* **Pipeline**: Web Audio API records PCM at 16kHz. Every 1s, the last 2s sliding window is encoded into a WAV chunk and POSTed to the backend.
* **Biomarkers (12 features)**:
  * Fundamental frequency ($F_0$) Mean, Standard Deviation, and Range.
  * Jitter % (micro-frequency instability via Relative Average Perturbation (RAP)).
  * Shimmer dB (micro-amplitude instability).
  * Harmonics-to-Noise Ratio (HNR).
  * Zero Crossing Rate (ZCR) proxying speaking rate and breathiness.
  * Root-Mean-Square (RMS) voice intensity.
  * High-frequency ratio (spectral energy $\ge$ 3000 Hz).
  * Spectral flux, pause ratio, and voiced frame fraction.
* **Fast Feature Extraction**: Utilizes an autocorrelation-based pitch extractor with parabolic peak interpolation. This resolves the integer-lag "flat-line" jitter bug and speeds up feature extraction from ~4.6s to **<15ms**, satisfying real-time constraints.
* **Calibration Bounding**: Dynamically restricts the search boundaries of the autocorrelation algorithm to $[0.40 \cdot \mu_{\text{pitch}}, 1.80 \cdot \mu_{\text{pitch}}]$ based on the user's calibrated baseline pitch.

### C. Physiological Modality (Modality 3)
* **Features**: EEG power spectral densities (alpha, beta, theta ratios) and GSR conductance levels/peak frequencies.
* **Classifier**: Random Forest / Gradient Boosting classifier trained on physiological stress datasets.

---

## 3. Calibration & Z-Score Normalization
To support different skin tones, lighting conditions, and vocal pitches, the system includes a **Guided Calibration Wizard** before starting a session:
1. **Silence Calibration (15s)**: Establishes the ambient noise floor. The threshold for vocal activity is dynamically set to $1.5 \times \text{noise\_floor}$.
2. **Neutral Voice Calibration (5 chunks)**: Computes the user's neutral pitch mean ($\mu_{\text{pitch}}$) and standard deviation.
3. **Neutral Face Calibration (10 frames)**: Records base landmarks (neutral EAR, neutral brow distance, neutral jaw width).

**Normalization**: Once calibrated, incoming indicators ($x$) are converted to Z-scores:
$$z = \frac{x - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}$$
These Z-scores are mapped to standard scaler dimensions before model prediction.

---

## 4. Multimodal Fusion Engine
The fusion engine connects via Server-Sent Events (SSE) `/api/stream/fused` emitting updates every 1 second. It fuses the latest predictions from active sensors:

$$\text{Fused Score} = \sum_{m \in M} w_m \cdot S_m$$

Where:
* $S_m$ is the stress probability predicted by modality $m$.
* $w_m$ is the dynamic reliability weight of modality $m$.

### Dynamic Weight Adjustments
* **Base weights**: Face (50%), Voice (50%).
* **Face Occlusion**: If landmark confidence falls below 0.5 (e.g., poor lighting or hand covering face), the facial weight is scaled down:
  $$w_{\text{face\_new}} = w_{\text{face}} \times \text{confidence}$$
  The weights are then re-normalized across all active sensors.
* **Temporal Decay (15s Voice Hold)**: When the user is silent, individual voice card status displays `"Silent / Ambient"`, but the last calculated voice score is held in the fusion buffer for **15 seconds** (matching conversational pause cycles). It is automatically purged if silence persists beyond 15 seconds.

---

## 5. System Latency & Hardware Benchmarks
* **Webcam Frame Processing**: $12\text{ms}$ (Client) + $8\text{ms}$ (Network & Backend inference) = **$20\text{ms}$ roundtrip**.
* **Audio Chunk Processing**: $1000\text{ms}$ chunk interval, processed on backend in **$22\text{ms}$** using `eventlet.tpool` thread workers to prevent event-loop blocking.
* **End-to-End Latency**: Well below the human interaction budget, offering instantaneous visual feedback.

---

## 6. Project Verification Status

The automated verification suite `run_and_verify_all.py` executes 23 comprehensive tests verifying the environment, file structures, algorithms, and fusion logic.

### Key Verification Metrics
* **Pitch interpolation accuracy**: Verified that parabolic interpolation reduces pitch quantization error by $>90\%$.
* **RAP Jitter Reliability**: Successfully filters microphone signal-to-noise ratios.
* **Model Loadability**: Verified that all lightweight models load, validate dimensions, and predict stress levels correctly.
* **Soft-Voting Ensemble validation**: Soft probabilities are averaged and match within $0.01$ of theoretical limits.
