# Phase 8: Best-Expert Multimodal Fusion Plan

## Goal

Build a low-latency multimodal stress detection engine by first selecting the best-performing model for each modality, then fusing only those best experts with either static or dynamic weighting.

The purpose of this phase is to avoid mixing weaker variants into the final system and to keep the architecture efficient, stable, and subject-generalized.

Recent work on multimodal fusion shows that modality quality is often uneven, and adaptive weighting or routing can be better than treating all modalities equally.[web:111][web:162][web:164]

---

## Core Idea

Instead of asking:
- “Which fusion method works best for all models?”

Ask:
- “Which single model is best for Face?”
- “Which single model is best for Voice?”
- “Which single model is best for Physio?”

Then:
- keep only those best experts,
- fuse their outputs,
- and choose static or dynamic weighting based on validation performance.

This is a cleaner design because expert selection and fusion design are separate problems.

---

## Expert Selection Rule

For each modality, the agent must evaluate all available candidate models and keep only the best one according to strict LOSO performance.

### Face expert
Select the single best face model from:
- classical baseline,
- calibrated classical model,
- deep learning model,
- any other certified face variant.

### Voice expert
Select the single best voice model from:
- classical baseline,
- calibrated classical model,
- deep learning model,
- any other certified voice variant.

### Physio expert
Select the single best physio model from:
- classical baseline,
- calibrated classical model,
- deep learning model,
- any other certified physio variant.

The selected expert must be the one that performs best not only on mean accuracy, but also on stability, calibration, and fold-to-fold consistency.

---

## Selection Criteria

The agent must not choose a model only because it has the highest average accuracy.

It must evaluate:
- mean LOSO accuracy,
- macro F1,
- balanced accuracy,
- calibration quality,
- per-subject variance,
- runtime cost,
- and robustness to weak folds.

A model is only “best” if it is both strong and stable.

---

## Fusion Strategy

After selecting the best expert for each modality, the agent must test:

### 1. Static weighted fusion
Use fixed weights such as:
- equal weights,
- validation-derived weights,
- or manually optimized weights.

This is the preferred low-latency baseline.

### 2. Dynamic weighted fusion
Use a lightweight gate or router that assigns modality weights per sample or per subject.

This should only be used if it improves validation results without increasing instability or runtime too much.

### 3. Hybrid selection fusion
Optionally allow the engine to choose between:
- Face + Physio only,
- Face + Voice + Physio,
- or Face-only / Physio-only fallback,
based on confidence or gate output.

This can be useful if Voice remains noisy.

---

## Recommended Default Path

Based on current benchmark behavior, the default production path should be:

- select the best Face expert,
- select the best Physio expert,
- treat Voice as optional unless it improves the fused result,
- use static weighted Face + Physio fusion as the main baseline,
- test dynamic gating only if it gives a clear gain.

Voice should not be forced into the final system if it consistently reduces performance.

---

## Why This Helps

This design avoids the common mistake of fusing all available models just because they exist.

Benefits:
- lower latency,
- cleaner architecture,
- better interpretability,
- less noise from weak experts,
- easier debugging,
- and more stable LOSO behavior.

It also allows the final system to be built from the best available parts rather than the average of all parts.

---

## Experimental Steps

### Step 1: Benchmark all candidate models
For each modality, evaluate every candidate model under the same LOSO protocol.

### Step 2: Select the winner per modality
Choose the single best model for each modality.

### Step 3: Freeze the selected experts
Do not keep re-tuning the experts during fusion experiments.

### Step 4: Test static fusion
Evaluate:
- equal weights,
- tuned weights,
- pairwise fusion,
- 3-way fusion.

### Step 5: Test dynamic fusion
Evaluate a lightweight gate or router that can change weights based on input quality.

### Step 6: Compare against baselines
Compare the best-expert fusion system against:
- unimodal best experts,
- previous calibrated classical fusion,
- previous deep fusion,
- and naive averaging.

---

## Runtime Requirements

The final architecture must remain suitable for low-latency inference.

The fusion layer should:
- be small,
- add minimal compute overhead,
- and not require a large meta-classifier.

If dynamic fusion is too slow or unstable, static fusion should be kept.

---

## Evaluation Rules

The agent must use:
- strict LOSO,
- subject-disjoint splits,
- no leakage in normalization or calibration,
- no test-subject tuning.

Report:
- accuracy,
- macro F1,
- balanced accuracy,
- calibration error,
- fold variance,
- and inference cost.

---

## Success Criteria

The phase is successful only if:
- the best expert per modality is clearly identified,
- the final fusion improves over the current baseline,
- the system stays stable across subjects,
- and runtime remains practical.

If the Voice expert remains harmful, it may be dropped from the main path and kept only as an auxiliary experiment.

---

## Failure Criteria

Reject the design if:
- weak experts are forced into fusion,
- dynamic gating overfits,
- static weighting gives no improvement,
- or runtime becomes too expensive.

Do not keep a more complex architecture unless it is measurably better.

---

## Agent Instruction Block

Use this instruction for implementation:

> Evaluate all available models for Face, Voice, and Physio under strict LOSO and select the single best expert for each modality based on mean performance, stability, and calibration. Freeze those best experts, then build a low-latency fusion engine on top of them using static weighted fusion first and dynamic gating only if it improves validation results. Do not fuse weaker variants just because they exist. Compare the best-expert fusion system against all prior baselines and keep only the configuration that is both highest-performing and most stable.

---

## Final Note

This phase is about assembling the final system from the strongest parts, not averaging everything blindly. The agent should treat expert selection as part of the architecture, not as an afterthought.