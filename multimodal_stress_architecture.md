# Formal Architecture Document: Personalized Multimodal Stress Detection System

## Overview

This document defines a novelty-oriented architecture for a personalized multimodal stress detection system that combines browser-side facial and vocal feature extraction, optional physiological sensing, adaptive baseline verification, confidence-aware multimodal fusion, and explainable stress inference. The design is intended for real-time deployment while addressing key practical issues in stress modeling: individual variability, noisy signals, contaminated calibration, asynchronous modalities, and low user trust in opaque predictions [cite:372][cite:375][cite:388].

The recommended architecture extends lightweight client-side extraction with a backend pipeline that performs temporal encoding, baseline-relative adaptation, uncertainty-aware fusion, and explanation generation. This makes the system more defensible as a novelty-grade product than a simple fixed-window GRU over handcrafted features, because it incorporates personalization, calibration validation, modality quality control, and explainable decision logic [cite:111][cite:363][cite:365][cite:371].

## Design Objectives

The proposed system is built around five objectives:

- Real-time inference from browser and wearable streams with low latency [cite:372][cite:380].
- Personalized stress assessment using user-specific baseline calibration rather than only population-level thresholds [cite:357][cite:363][cite:371].
- Robustness to poor lighting, head movement, microphone noise, and asynchronous sensor arrival through signal-quality gating and dynamic fusion [cite:372][cite:375][cite:111].
- Explainable outputs that identify likely contributing modalities and biomarkers so the result is actionable and reviewable by the user [cite:365][cite:388].
- Product-level novelty based on verified-baseline personalization and confidence-aware multimodal inference rather than only standard feature extraction [cite:363][cite:365][cite:368].

## System Architecture

The architecture is organized into six layers.

### 1. Edge Acquisition Layer

The edge layer captures face and voice data directly in the browser and can optionally ingest physiological streams from wearables or external devices. MediaPipe Face Mesh supports real-time facial landmark extraction from a single RGB camera, making it suitable for lightweight browser deployment [cite:372]. On the audio side, 16 kHz sampling with browser audio processing is sufficient for speech-oriented features such as pitch, energy, jitter, shimmer, and spectral coefficients [cite:376][cite:380].

This layer should also compute signal-quality metadata alongside raw features. For facial input, quality checks should include face presence, landmark stability, head pose, illumination stability, and motion level; for voice, they should include speech activity, signal-to-noise ratio, clipping, and silence ratio [cite:372][cite:375].

### 2. Local Feature Extraction Layer

The local feature layer computes low-cost indicators for rapid feedback and efficient streaming. Facial indicators can include eye aspect ratio, brow position, lip compression, jaw displacement, and optional temporal motion summaries, but these should be treated as proxies rather than direct stress evidence because facial geometry alone is not specific to stress [cite:375].

Voice features should include RMS energy, pitch trajectory, jitter, shimmer, voicing consistency, and optionally compact spectral summaries. Server-side processing can then refine this with robust extraction such as YIN-based fundamental frequency tracking and MFCC computation using librosa [cite:376][cite:380].

### 3. Baseline Verification Layer

Baseline verification is the first major novelty component. Personalized stress systems benefit from short user-specific calibration windows, but calibration can fail if the user is already stressed, distracted, speaking intensely, or exposed to environmental noise during baseline collection [cite:359][cite:360][cite:363].

The system should therefore validate every proposed calibration segment before accepting it as the user baseline. A candidate baseline window is passed through a low-stress verification model, checked for signal quality, and compared against expected neutral ranges. If the baseline appears suspicious, the system should present an explanation and ask the user whether the captured window truly reflects their normal state [cite:365][cite:371].

Accepted baselines should store, at minimum:

- Mean and variance of each modality feature.
- Baseline quality score.
- Timestamp and environmental metadata.
- Confidence label such as `verified`, `low_confidence`, or `recalibration_needed`.

### 4. Temporal Encoding Layer

Each modality should be processed by its own temporal encoder before fusion. This is preferable to flattening all features together because each stream has different time structure, noise characteristics, and failure modes [cite:330][cite:388].

Recommended encoders:

| Modality | Recommended encoder | Rationale |
|---|---|---|
| Face | Temporal CNN or GRU over landmark-derived features; optional rPPG branch | Captures micro-patterns and short-term facial change [cite:375][cite:384] |
| Voice | CNN-GRU or CNN-Transformer over MFCC, pitch, jitter, shimmer, RMS | Captures spectral and temporal speech stress cues [cite:379][cite:385] |
| Physio | Dual-branch CNN over time and frequency features | Supports robust stress modeling from HR/HRV, EDA, TEMP, IMU [cite:330] |

Each encoder should produce:

- A latent feature embedding.
- A modality-specific stress score.
- A modality confidence or quality estimate.

### 5. Cross-Modal Fusion Layer

The fusion layer is the second major novelty component. Instead of simple concatenation or equal weighting, the system should use cross-modal attention or dynamically aligned fusion so that modalities can influence one another while preserving modality-specific confidence [cite:111][cite:388].

This is especially important because stress cues are asynchronous in practice. A vocal stress marker may rise before a facial change appears, while physiological arousal may persist after the speech event ends. Dynamic alignment and attention-based fusion are specifically suited for these timing mismatches and have been proposed to improve multimodal time-series learning over fixed alignment assumptions [cite:111].

The fusion layer should combine:

- Raw modality embeddings.
- Baseline-relative features such as z-score deviation from user baseline.
- Signal-quality scores.
- Modality confidence estimates.
- Short-term temporal context from adjacent windows.

### 6. Decision and Explanation Layer

The final layer should output more than a class label. For practical deployment, the model should return:

- Stress probability.
- Stress level or ordinal category.
- Uncertainty score.
- Modality contribution scores.
- Human-readable explanation.
- Recalibration recommendation when confidence is poor [cite:365][cite:388].

Explainable stress systems are increasingly important because users and evaluators need to understand whether a decision came from voice tension, facial strain, physiological elevation, or multimodal agreement [cite:365]. Uncertainty estimation, such as confidence heads or Monte Carlo dropout, can further reduce unsafe overconfident predictions under weak signal quality or missing data [cite:388].

## Recommended Data Flow

The end-to-end inference flow should follow the sequence below:

1. Capture face, voice, and optional physiological data.
2. Compute signal-quality indicators per modality.
3. Extract lightweight local features at the edge.
4. Stream features or compressed windows to the backend.
5. Verify or retrieve the user baseline.
6. Transform incoming features into both raw and baseline-relative forms.
7. Encode each modality with its temporal model.
8. Fuse modalities using confidence-aware cross-attention.
9. Estimate stress probability, stress category, and uncertainty.
10. Generate explanation and user feedback.
11. Trigger recalibration if baseline confidence or signal quality is inadequate [cite:363][cite:365][cite:371][cite:388].

## Baseline Verification and Consent Algorithm

```text
Input: candidate calibration window W for face, voice, and physiology
Output: ACCEPT_BASELINE, ACCEPT_WITH_LOW_CONFIDENCE, or RECALIBRATE

1. Measure signal quality Q for each modality.
2. If Q is below threshold for one or more critical modalities, return RECALIBRATE.
3. Pass W through the baseline verification model.
4. Compute:
   - predicted low-stress probability,
   - per-modality anomaly score,
   - baseline consistency score,
   - explanation features.
5. If predicted low-stress probability is high and anomaly score is low:
   return ACCEPT_BASELINE.
6. If predicted stress probability is moderate or high:
   show explanation to user and request confirmation.
7. If user says the window is not representative:
   return RECALIBRATE.
8. If user says the window is representative despite the warning:
   return ACCEPT_WITH_LOW_CONFIDENCE.
9. Store baseline profile and confidence state for later inference.
```

This algorithm prevents contaminated calibration windows from silently becoming the personalization anchor, which is a known risk in user-specific stress adaptation [cite:359][cite:360][cite:363].

## Personalization Strategy

The system should not rely only on global training patterns. It should combine population learning with user-specific normalization and adaptation. Personalized stress detection research consistently motivates short baseline adaptation because physiological and behavioral responses vary substantially across individuals [cite:357][cite:360][cite:371].

A practical personalization strategy is:

- Train a global multimodal model on strict subject-wise splits.
- Maintain user baseline statistics per modality.
- Convert current features to baseline-relative deltas and standardized deviations.
- Shift decision thresholds using user-specific calibration confidence.
- Refresh the baseline only during likely neutral periods or after explicit user consent [cite:363][cite:368][cite:371].

## Training Methodology

A staged training process is recommended.

### Stage 1: Unimodal pretraining

Train face, voice, and physiological encoders independently using user-held-out splits. This helps each encoder learn stable modality-specific patterns before multimodal interaction is introduced [cite:330][cite:388].

### Stage 2: Multimodal fusion training

Train the fusion layer on synchronized windows using cross-attention and modality-quality inputs. This stage should include modality dropout so the model learns not to fail when one stream is weak or missing [cite:111][cite:330].

### Stage 3: Personalization training

Augment the model with baseline-relative features and a calibration-confidence input. This allows the final prediction layer to account for user-specific normal ranges rather than absolute feature magnitudes alone [cite:357][cite:368].

### Stage 4: Baseline verification model

Train a dedicated low-stress or neutral-baseline verifier using verified neutral segments, low-quality windows, and deliberately contaminated baseline examples. This model is separate from the main stress classifier and protects the personalization pipeline [cite:363][cite:365].

### Stage 5: Final evaluation

Use leave-one-subject-out or strict userID-based held-out testing. This is necessary to validate generalization to unseen users rather than inflated performance caused by user overlap [cite:388].

## Data Augmentation Recommendations

Moderate augmentation is useful, but it should remain physiologically and behaviorally plausible. Recommended methods include overlapping windows, slight time shifts, low-amplitude noise injection, magnitude scaling, and modality dropout [cite:330]. More aggressive synthetic generation can help with class imbalance, but it should be validated carefully under subject-wise evaluation because stress signals are subtle and can be distorted by unrealistic synthesis [cite:388].

## Explainability Framework

The explanation module should translate model internals into user-facing rationale. A recommended explanation format is:

- Overall stress likelihood.
- Confidence of the prediction.
- Dominant contributing modality.
- Key biomarkers or deviations from baseline.
- Whether the result is stable across multiple windows.

Example:

> Elevated stress likelihood detected with moderate confidence. Voice pitch instability and increased energy were above the personal baseline, while facial tension indicators were mildly elevated. Physiological evidence was unavailable, so confidence is reduced.

This kind of explanation is more aligned with medically oriented explainable stress systems than a raw scalar score alone [cite:365].

## Reliability Constraints

The architecture is strong, but several limitations must be explicitly recognized:

- Facial geometry is not stress-specific and is affected by expression style, fatigue, camera angle, and speech [cite:375].
- Voice stress markers are affected by language, illness, room acoustics, and microphone quality [cite:376][cite:380].
- Remote signals are vulnerable to lighting instability, movement, and missing modalities [cite:372][cite:375].
- Personalization fails if calibration is contaminated and not verified [cite:359][cite:363].

For this reason, the system should be positioned as stress monitoring or early warning rather than definitive diagnosis unless much stronger validation is completed [cite:361][cite:371].

## Novelty Statement

The main novelty of the proposed architecture lies in the integration of four elements into one deployable system:

1. Browser-native multimodal extraction for low-friction deployment [cite:372][cite:380].
2. Verified personal baseline calibration with user consent and recalibration logic [cite:363][cite:365][cite:371].
3. Confidence-aware temporal cross-modal fusion for asynchronous and noisy signals [cite:111][cite:388].
4. Explainable stress output with uncertainty estimation and modality-level rationale [cite:365][cite:388].

This combination is more product-relevant and more original than a conventional GRU-based classifier over fixed windows because it addresses operational deployment constraints rather than only offline classification accuracy [cite:111][cite:363][cite:388].

## Recommended Final Architecture Name

**Adaptive Verified-Baseline Multimodal Stress Inference Architecture (AVB-MSIA)**

This name reflects the two strongest novel elements of the design: baseline verification and adaptive multimodal inference [cite:363][cite:388].

## Implementation Priorities

The recommended implementation order is:

1. Add signal-quality scoring to the current face and voice pipelines.
2. Implement a baseline verification and consent loop.
3. Add storage of baseline statistics and confidence labels.
4. Train unimodal temporal encoders.
5. Add confidence-aware cross-modal fusion.
6. Add uncertainty estimation and explanation generation.
7. Validate strictly on unseen users with subject-wise evaluation [cite:372][cite:365][cite:388].

## Conclusion

The proposed architecture is suitable for a novelty-oriented stress monitoring product because it builds on lightweight real-time feature extraction but adds the missing components required for reliable deployment: validated personalization, multimodal confidence handling, temporal fusion, and explainability [cite:372][cite:111][cite:365][cite:388]. This makes it substantially stronger than a fixed-window GRU pipeline and provides a defensible basis for research publication, system design documentation, and product differentiation [cite:330][cite:363][cite:371].
