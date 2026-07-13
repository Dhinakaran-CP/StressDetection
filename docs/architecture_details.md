# StressDetectionUsingML — Multimodal Architecture Implementation Plan

## Purpose

This document gives the exact implementation procedure for building each architecture in the repository.

It is written so a new agent can implement the full multimodal system phase by phase, from simple baselines to high-capacity expert-based fusion.

The repository uses synchronized Face, Voice, and Physio windows, subject-independent evaluation, and a locked production baseline. Any new implementation must preserve these constraints [web:111][web:255][web:309][web:312].

---

## 1. Global Implementation Rules

Before implementing any architecture, the agent must follow these rules:

- Preserve synchronized windows across modalities.
- Keep `subject_id` strictly out of model input.
- Use GroupKFold or LOSO for all final validation.
- Fit scalers and normalization only inside each training fold.
- Do not leak held-out subject information into preprocessing or feature selection.
- Keep primary and fallback production models intact.
- Calibrate outputs before final reporting.
- Track both accuracy and leakage-gap metrics.

These rules are mandatory for every architecture [web:111][web:247][web:301].

---

## 2. Common Data Pipeline

All architectures must start from the same data pipeline.

### Step 1: Load synchronized data
- Load Face, Voice, and Physio files.
- Verify alignment by `subject_id`, `task_id`, `video_id`, and `window_index`.
- Drop any inconsistent or unmatched windows.

### Step 2: Build fold splits
- Use subject-wise GroupKFold or LOSO.
- Ensure each subject appears in only one fold.
- Save fold index maps for reproducibility.

### Step 3: Apply fold-safe preprocessing
- Fit scaler only on training fold.
- Transform train and test separately.
- Apply baseline normalization only inside the fold.

### Step 4: Build labels
- Confirm binary or class labels.
- Check class balance inside each fold.
- Preserve all task metadata for analysis, not as model input.

### Step 5: Export ready tensors
- Create modality tensors per window.
- Keep a modality mask for missing or silent windows.
- Save prepared fold data for repeatable training.

---

## 3. Architecture A — Unimodal Baseline Models

This is the first implementation layer.

### Goal
Create separate Face, Voice, and Physio models to measure each modality independently.

### For each modality
- Build an input encoder.
- Train a classifier head.
- Evaluate under strict subject-wise validation.

### Face model
1. Choose a compact sequence encoder or MLP/CNN-GRU.
2. Feed facial feature windows only.
3. Train on subject-safe folds.
4. Record accuracy, F1, and leakage gap.

### Voice model
1. Build a temporal encoder for acoustic windows.
2. Use only voice-specific features.
3. Mask silent or empty windows if needed.
4. Train and evaluate fold-wise.

### Physio model
1. Build a temporal encoder for physiological windows.
2. Use HR, EDA, and related features.
3. Train and evaluate under the same protocol.

### Output
These unimodal baselines become reference experts for later fusion [web:111][web:255][web:298].

---

## 4. Architecture B — Late Fusion

This is the simplest multimodal fusion method.

### Goal
Combine modality predictions after each unimodal model has produced a score.

### Implementation steps
1. Train the Face, Voice, and Physio experts separately.
2. Obtain calibrated probabilities from each expert.
3. Average or weighted-average the probabilities.
4. Tune weights on training folds only.
5. Evaluate on the held-out fold.

### Recommended variants
- Simple average.
- Weighted average.
- Confidence-weighted average.

### When to use
- As a low-risk multimodal baseline.
- As a fallback inference method.
- As a stability comparison against richer fusion.

Late fusion is reliable and easy to debug, but it does not learn deep interactions [web:247][web:243].

---

## 5. Architecture C — Early Fusion

This is the first joint representation method.

### Goal
Combine synchronized modality embeddings early and let the model learn one shared representation.

### Implementation steps
1. Train lightweight encoders for each modality.
2. Extract an embedding per modality for each synchronized window.
3. Concatenate embeddings.
4. Pass them into an MLP, GRU, or transformer block.
5. Train a final classifier.

### Required properties
- Inputs must be synchronized.
- Missing modality handling must be explicit.
- Feature scales must be fold-normalized.

### Advantages
- Strong joint representation.
- Simple implementation.
- Good for aligned windows.

### Risks
- Can overfit if one modality dominates.
- Can mix noisy signals without gating.

Early fusion should be treated as a mid-level baseline before attention and MoE [web:243][web:247][web:309].

---

## 6. Architecture D — Gated Fusion

This is the first reliability-aware fusion strategy.

### Goal
Learn sample-wise modality weights.

### Implementation steps
1. Build unimodal encoders or embeddings.
2. Feed the embeddings into a gate network.
3. Let the gate output one weight per modality.
4. Multiply each embedding by its learned weight.
5. Fuse the weighted embeddings.
6. Pass them to a classifier.

### Suggested gating inputs
- modality embedding,
- modality quality indicators,
- missing-mask flags,
- task context if allowed.

### Why this works
- Downweights weak or silent modalities.
- Helps with task-dependent modality usefulness.
- Reduces interference from poor inputs.

### Strong use case
Voice may be weak during silent tasks, so gating can lower its influence automatically [web:111][web:314].

---

## 7. Architecture E — Cross-Attention Fusion

This is the main interaction-based architecture.

### Goal
Let one modality attend to another modality’s representation.

### Implementation steps
1. Build a modality encoder for each stream.
2. Produce a sequence or window embedding for each modality.
3. Apply cross-attention blocks:
   - Face attends to Voice and Physio.
   - Voice attends to Face and Physio.
   - Physio attends to Face and Voice.
4. Optionally add self-attention inside each modality.
5. Fuse the attended representations.
6. Classify the final state.

### Key design rule
Cross-attention should operate on synchronized windows only.

### Recommended extra loss
- Add a consistency loss between modality embeddings.
- Add alignment regularization if the representations drift apart.

Cross-attention is one of the strongest choices for synchronized multimodal learning [web:111][web:247][web:314].

---

## 8. Architecture F — Multi-Expert Modality Internals

This is the sub-modality expert design.

### Goal
Create multiple experts inside each modality, each focused on a coherent feature group.

### Do not do
- Do not create one full model per scalar feature.
- Do not create dozens of tiny isolated models.

### Do instead
Create 3–5 experts per modality.

### Face experts
1. Eye expert.
2. Mouth/lip expert.
3. Global facial tension expert.

### Voice experts
1. Prosody expert.
2. Spectral expert.
3. Voice quality expert.

### Physio experts
1. Cardio expert.
2. EDA expert.
3. Motion or breathing expert.

### Implementation steps
1. Split the feature set into coherent groups.
2. Train a small encoder for each group.
3. Produce one embedding per expert.
4. Use a modality-internal gate to combine experts.
5. Produce one modality embedding.
6. Pass modality embeddings into the fusion layer.

### Why this is useful
- Experts specialize.
- Interference is reduced.
- The model can capture fine-grained cues.

This is the right way to implement the user’s “one expert per facial part” idea [web:299][web:300][web:305].

---

## 9. Architecture G — Mixture of Experts Fusion

This is the high-capacity version.

### Goal
Let a router select among multiple experts dynamically.

### Implementation steps
1. Build the expert pools for each modality or sub-modality.
2. Add a gating router that sees the current embedding.
3. Route each sample to the best expert or a weighted subset.
4. Combine expert outputs.
5. Feed the fused representation to the final classifier.

### Router choices
- Soft gating.
- Top-k routing.
- Reliability-aware routing.
- Modality-specific routing pools.

### Recommended design
Use:
- modality-specific expert pools,
- plus a shared cross-modal expert.

This supports both specialization and shared learning [web:283][web:301][web:309][web:312].

---

## 10. Architecture H — Hybrid MoE + Cross-Attention

This is the recommended research-grade target.

### Goal
Combine:
- sub-modality experts,
- cross-attention,
- and gating.

### Implementation flow
1. Build sub-modality experts.
2. Fuse inside each modality using gating.
3. Create modality embeddings.
4. Apply cross-attention across modalities.
5. Apply a global MoE router.
6. Produce final logits.
7. Calibrate outputs.

### Benefits
- Fine-grained specialization.
- Strong inter-modality interaction.
- Better robustness to noisy input.
- Better scalability for future experiments.

This is the most promising architecture for your project if you want maximum research value [web:255][web:283][web:301].

---

## 11. Identity Suppression Layer

This layer is optional but strongly recommended.

### Goal
Prevent the model from learning subject identity.

### Implementation steps
1. Add a gradient reversal branch.
2. Add a subject classifier head.
3. Train the main model to improve stress prediction.
4. Train the auxiliary branch to predict subject identity.
5. Reverse the gradient so subject identity becomes hard to encode.
6. Tune the adversarial weight carefully.

### Important
Do not let the adversarial branch dominate the main objective.

This must be tuned so performance stays stable while leakage is reduced.

---

## 12. Calibration Layer

Every final architecture should be calibrated.

### Implementation steps
1. Take validation logits.
2. Fit calibration on training-fold outputs only.
3. Apply temperature scaling or similar calibration.
4. Recheck metrics after calibration.

### Why
- Better probability quality.
- Better router confidence.
- Better production reliability.

---

## 13. Evaluation Procedure

Every architecture must be evaluated with the same protocol.

### Required metrics
- Accuracy.
- Macro F1.
- Balanced accuracy.
- Fold standard deviation.
- Leakage gap.
- Calibration quality.

### Required comparisons
- Random split vs subject-wise split.
- Raw vs normalized features.
- Unimodal vs fused models.
- Without vs with adversarial suppression.
- Without vs with gating or attention.

### Selection rule
Choose the architecture that improves subject-independent performance and reduces leakage gap, not the one that only wins on random split.

---

## 14. Implementation Phases

### Phase 1: Data audit
- Confirm synchronization.
- Confirm features.
- Confirm fold grouping.

### Phase 2: Unimodal experts
- Build Face, Voice, and Physio baselines.
- Calibrate them.
- Log metrics.

### Phase 3: Simple multimodal fusion
- Late fusion.
- Early fusion.
- Pick a stable comparison baseline.

### Phase 4: Gated fusion
- Add reliability-aware weighting.

### Phase 5: Cross-attention fusion
- Add bidirectional cross-modal attention.

### Phase 6: Sub-modality experts
- Split each modality into feature groups.
- Train internal experts.

### Phase 7: MoE router
- Add dynamic expert selection.

### Phase 8: Hybrid final architecture
- Combine experts, gating, and attention.
- Add calibration.

### Phase 9: Identity suppression
- Add adversarial branch if needed.

### Phase 10: Packaging and tests
- Save artifacts.
- Register models.
- Verify runtime load paths.
- Run full test suite.

---

## 15. Production Policy

The locked production system must remain intact:
- Strategy 5 primary.
- Strategy 4 fallback.

New architectures are research candidates only until they beat the locked production system under strict validation.

---

## 16. Repository Safety Rules

The agent must:
- update paths carefully,
- preserve deployment compatibility,
- keep archive and production assets separate,
- and never break existing runtime resolution logic.

---

## 17. Final Guidance

The implementation should move from simple to complex in a controlled way:
- baseline,
- fusion,
- gating,
- attention,
- experts,
- MoE,
- hybrid high-capacity system.

This is the safest and most research-credible way to build the next version of the project.

---

## 18. Agent Instruction Block

Follow these exact rules:
- Implement each architecture in the order listed.
- Validate each architecture under subject-wise splits.
- Keep preprocessing fold-safe.
- Do not let subject identity leak into training or evaluation.
- Use modality-specific experts only for coherent feature groups.
- Prefer gated or cross-attention fusion for synchronized data.
- Add MoE only after simpler fusion methods are benchmarked.
- Keep the production baselines unchanged unless the new model clearly wins.

---

## 19. End State

The final system should be a modular multimodal stress detection platform with:
- reliable unimodal experts,
- multiple fusion options,
- cross-modal interaction,
- sub-modality specialization,
- and a production-safe baseline/fallback structure.

This document is the implementation roadmap for the agent.