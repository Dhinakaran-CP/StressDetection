# Phase 6: Multimodal Research Plan for StressID-Generalized Fusion

## Goal

Improve the multimodal stress detection architecture beyond the current handcrafted fusion pipeline by testing subject-invariant, temporally aware, and confidence-aware fusion strategies on the certified Face, Voice, and Physio modalities.

The aim is not to chase an artificial number, but to build a stronger generalization model that is more likely to transfer across unseen subjects under strict LOSO validation.

StressID is a synchronized multimodal dataset with face video, audio, ECG, EDA, and respiration from 65 participants across 11 tasks, and its baseline results show that fusion is not automatically superior to strong unimodal systems.[page:0][page:1]

---

## Problem Statement

The current late-fusion pipeline improves over single-modality baselines, but the gain is still limited. This usually means one or more of the following are true:

- The modality representations are still too shallow.
- The modalities are not perfectly aligned in time.
- Subject identity is still leaking into the learned features.
- The fusion rule is too simple for the signal complexity.
- Some modalities are contributing noise instead of complementary information.

StressID’s baseline section shows that handcrafted feature fusion and decision fusion behave inconsistently, and the multimodal setting can underperform unimodal baselines when the fusion strategy is weak.[page:1]

---

## Research Hypothesis

The model will improve if we move from simple feature fusion to a more structured multimodal architecture that includes:

- modality-specific encoders,
- temporal sequence modeling,
- subject-adaptive normalization,
- confidence-aware fusion,
- and calibration before final decision-making.

StressID itself suggests that subject-specific, task-specific, and modality-specific analysis are all relevant, which supports a deeper architecture than a flat tabular classifier.[page:0]

---

## What to investigate

### 1. Modality-specific encoders

Build separate encoders for face, voice, and physio rather than treating all features as one flat table.

Possible options:
- MLP encoder for tabular features.
- 1D temporal encoder for window sequences.
- Lightweight CNN or TCN for time-ordered feature blocks.

The purpose is to let each modality learn its own signal geometry before fusion.

### 2. Temporal aggregation

Instead of collapsing windows too early, preserve a short sequence of windows per task and let the model aggregate them.

Recommended tests:
- rolling mean sequence,
- LSTM aggregation,
- temporal convolution,
- attention over windows.

StressID contains task structure and synchronized recordings, so temporal dynamics are meaningful and should not be discarded too early.[page:0]

### 3. Learned normalization

Test whether fixed calm-baseline subtraction can be replaced or supplemented by a learned subject-normalization layer.

Test options:
- current baseline subtraction,
- z-score by subject,
- adaptive instance normalization,
- learned affine normalization.

The aim is to reduce subject identity bias without destroying stress-related variance.

### 4. Confidence-aware fusion

Do not assume every modality should contribute equally every time.

Test:
- naive averaging,
- weighted averaging,
- dynamic gating,
- calibration-weighted fusion,
- stacking on calibrated outputs.

StressID’s published baselines show that average decision fusion can outperform feature fusion in some settings, but the result is still far from perfect, so fusion policy matters a lot.[page:1]

### 5. Missing-modality robustness

Train and evaluate the fusion model so it does not collapse when one modality is weaker or missing.

This is important because StressID itself contains missing modalities for some participants, and a practical system must remain stable under partial input.[page:0][page:1]

---

## Experimental Priorities

### Priority A: Strong unimodal representations
First confirm that each modality has the best possible subject-generalized encoder.

Focus on:
- face stability,
- voice calibration,
- physio temporal structure.

### Priority B: Pairwise fusion
Test:
- face + voice,
- face + physio,
- voice + physio.

This will reveal which modality pairs are genuinely complementary.

### Priority C: Full 3-way fusion
Test:
- naive average,
- weighted average,
- gated fusion,
- calibrated stacking.

Only keep a 3-way method if it truly beats the best pairwise model and stays stable across subjects.

---

## Suggested Architecture Direction

### Stage 1: Encoders
- Face encoder learns facial stress features.
- Voice encoder learns speech stress features.
- Physio encoder learns autonomic stress patterns.

### Stage 2: Temporal layer
Each modality receives a short temporal context instead of isolated rows.

### Stage 3: Calibration
Each modality output is calibrated before fusion.

### Stage 4: Fusion gate
A learned gate decides how much each modality contributes for a given window or task.

### Stage 5: Final classifier
A compact classifier produces the final stress prediction.

This structure is more aligned with StressID’s multimodal nature than a single flat fusion model.[page:0][page:1]

---

## Evaluation Rules

The agent must evaluate every experiment under:
- strict LOSO / GroupKFold by subject,
- no subject leakage,
- no calibration leakage,
- no preprocessing leakage,
- no hyperparameter tuning on the test subject.

Metrics to report:
- accuracy,
- macro F1,
- balanced accuracy,
- per-subject variance,
- calibration error,
- confusion matrix.

The StressID paper itself uses subject/task-aware analysis and reports both unimodal and multimodal comparisons, so the new experiments must be equally disciplined.[page:1]

---

## Success Criteria

A candidate is worth keeping only if it:
- improves LOSO performance,
- remains stable across folds,
- does not overfit,
- generalizes better than the current fusion baseline,
- and is practical enough for the runtime engine.

Do not keep a method just because it improves one metric on one split.

---

## What not to do

- Do not rely only on more features.
- Do not assume more modalities guarantee higher accuracy.
- Do not use random splits.
- Do not tune on the test subject.
- Do not hide instability behind a single average score.
- Do not replace a stable simple fusion method with a more complex one unless it actually wins under LOSO.

StressID’s own benchmark shows that multimodal learning can be harder than expected, and strong evaluation discipline matters more than complexity alone.[page:1]

---

## Recommended Research Sequence

1. Reproduce current unimodal and fused LOSO baselines.
2. Add modality-specific encoders.
3. Add temporal aggregation.
4. Add calibrated outputs.
5. Add confidence-aware fusion.
6. Compare against naive averaging and stacking.
7. Keep only the most stable candidate.

---

## Agent Instruction Block

Use this instruction for the implementation agent:

> Investigate the current multimodal fusion architecture and improve it using StressID-informed strategies. Keep strict LOSO validation and no leakage. Build modality-specific encoders for face, voice, and physio, test temporal aggregation, subject-adaptive normalization, calibration, confidence-aware fusion, and missing-modality robustness. Compare pairwise and 3-way fusion strategies against the current baseline, and keep only methods that improve subject-generalized performance without overfitting. Do not assume more modalities automatically improve accuracy; let the LOSO results decide.

---

## Final Note

This phase is about building a more research-grade fusion engine, not just a bigger classifier. The goal is to discover which combination of representation learning, temporal context, and fusion policy actually improves unseen-subject stress detection under realistic evaluation.[page:0][page:1]