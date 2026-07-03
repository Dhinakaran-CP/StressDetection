# Experiment Plan for Improving LOSO Stress Detection Performance

This document gives a realistic, non-fake improvement plan for the multimodal stress detection project. It is written for an implementation agent that must test which methods are actually helpful for unseen-subject performance before deciding whether re-extraction is needed.

The plan is based on a simple principle: first exploit the current certified data as much as possible, then only re-extract if the extracted schema is too weak to support subject-aware learning.

The literature on multimodal stress detection shows that performance can improve through sliding-window augmentation, better feature handling, dimensionality reduction, and fusion design, but these gains must be validated under true subject-independent evaluation such as LOSO.[web:98][web:100][web:92]

---

## 1. Goal

The agent must determine the most promising and honest path to improve LOSO performance from the current 63 percent face and 56 percent voice baseline.

The goal is not to force 70 percent by changing evaluation. The goal is to find the strongest valid model stack that generalizes to unseen subjects.

---

## 2. First decision: re-extraction or not

The agent should only re-extract data if the current extracted files are missing any of the following:
- `subject_id`,
- `task_name`,
- window boundaries,
- usable feature schema,
- or any reliable way to reconstruct temporal windows.

If subject IDs and windows are already present, the agent should first try improvements inside the existing dataset.

This is realistic because production ML guidance emphasizes strengthening a certified dataset before rebuilding the data pipeline from scratch.[web:79][web:84]

---

## 3. What to test, in order

### Stage 1: Reproduce the baseline

The agent should rerun the current LOSO setup and confirm the known baseline for each modality.

Deliverables:
- fold-level face metrics,
- fold-level voice metrics,
- confusion matrices,
- per-subject score table,
- reproducible seed list.

This stage exists so every later improvement is compared against the same honest reference.

### Stage 2: Subject-aware normalization

The agent should test normalization strategies that reduce subject identity bias.

For face:
- normalize landmark geometry to a stable reference point,
- use per-subject baseline subtraction if calm windows exist,
- prefer deltas or ratios over raw absolute values where valid.

For voice:
- normalize pitch, intensity, and spectral descriptors against a session or subject baseline if possible,
- remove silence-dominated windows,
- standardize acoustic scales carefully.

This stage is important because subject variation is a major reason LOSO performance drops compared with random splits.[web:92][web:97]

### Stage 3: Windowing refinement

The agent should test several window sizes and overlaps instead of using one full-video average.

Recommended trials:
- face: 0.5 s, 1.0 s, 2.0 s,
- voice: 1.5 s, 2.5 s, 3.0 s,
- overlap: 25 percent, 50 percent.

The agent should compare whether shorter windows improve stress sensitivity without adding too much noise.

Sliding-window segmentation is a standard and honest way to enrich temporal examples because it does not invent new labels.[web:98][web:100]

### Stage 4: Feature selection

The agent should reduce weak or redundant features before trying a heavier model.

For each modality, test:
- mutual information,
- permutation importance,
- recursive feature elimination,
- regularized selection,
- correlation pruning.

A smaller, cleaner feature set often improves robustness in stress detection because not all extracted biomarkers are equally useful.[web:88][web:93]

### Stage 5: Per-modality tuning

The agent should tune each expert independently using grouped validation.

Tune for gradient boosting or any current tree-based expert:
- number of estimators,
- learning rate,
- max depth,
- min samples split,
- min samples leaf,
- subsample,
- feature subsample,
- class weighting,
- calibration.

All tuning must remain subject-aware. No parameter may be selected using the held-out LOSO subject.

### Stage 6: Calibrated fusion

The agent should test whether fusion improves the final result.

Fusion candidates:
- weighted average,
- confidence-weighted average,
- calibrated average,
- logistic stacking,
- simple meta-learner on out-of-fold predictions.

A fusion method is only valid if the meta-step is trained on validation predictions, not on the final LOSO test subject.

Intermediate or fusion-based stress models are commonly improved by using dimensionality reduction or structured fusion rather than naive concatenation.[web:100][web:43]

### Stage 7: Conservative augmentation

The agent may test augmentation only if it remains label-preserving and realistic.

Allowed:
- jitter small feature noise,
- slightly shifted sliding windows,
- bootstrap samples inside training folds only,
- robust noise filtering,
- speaker/frame quality filtering.

Not allowed:
- invented stress labels,
- GAN-generated samples unless separately justified,
- synthetic subject creation,
- augmentation that leaks test identity.

Recent work on stress detection supports sliding-window augmentation and jittering as legitimate ways to improve robustness.[web:98][web:46]

### Stage 8: Optional stacking

If simple fusion is not enough, the agent can test stacking.

Rules for stacking:
- create out-of-fold predictions from the base experts,
- train the stacker only on those out-of-fold predictions,
- keep the stacker small and interpretable,
- compare against a simple weighted ensemble.

Stacking should only be accepted if it improves mean LOSO performance and does not make the system unstable or too heavy for runtime.

---

## 4. What to measure

The agent should not rely on accuracy alone.

For each experiment, record:
- accuracy,
- F1 score,
- precision,
- recall,
- balanced accuracy,
- per-subject scores,
- calibration quality,
- latency,
- memory cost,
- failure rate when a modality is missing.

This matters because an experiment that improves average accuracy but fails badly on a few subjects is not good enough for deployment.

---

## 5. Decision rules

Keep a method only if:
- it improves LOSO mean performance,
- it improves robustness across subjects,
- it does not break runtime performance,
- and it stays scientifically valid.

Reject a method if:
- it only improves random split results,
- it depends on test fold information,
- it adds large complexity for a tiny gain,
- it reduces calibration quality,
- or it creates fragile behavior.

If no method crosses 70 percent honestly, document the strongest stable configuration and explain why the current dataset limits further gains.

---

## 6. Agent workflow

### Step 1: Data audit

The agent checks whether current extracted data is enough for subject-aware experiments.

Outputs:
- schema check,
- subject coverage,
- missingness report,
- temporal structure report.

### Step 2: Baseline reproduction

The agent reproduces the existing LOSO baseline exactly.

Outputs:
- frozen baseline metrics,
- fold logs,
- reproducible run command.

### Step 3: Single-method experiments

The agent tests one family at a time:
- normalization,
- windowing,
- feature selection,
- tuning,
- augmentation.

Outputs:
- one result table per method family,
- best configuration per family.

### Step 4: Fusion experiments

The agent tests weighted averaging, calibrated fusion, and stacking.

Outputs:
- fusion comparison table,
- selected fusion strategy,
- validation curves.

### Step 5: Final comparison

The agent compares all winning candidates and selects the best release candidate.

Outputs:
- final ranked list,
- final recommendation,
- commit-ready experiment summary.

---

## 7. Suggested experiment matrix

| Experiment | Face | Voice | Expected value | Risk |
|---|---:|---:|---|---|
| Baseline LOSO | yes | yes | reference point | low |
| Subject normalization | yes | yes | high | low |
| Window tuning | yes | yes | high | low |
| Feature selection | yes | yes | medium-high | low |
| Hyperparameter tuning | yes | yes | medium-high | low |
| Weighted fusion | yes | yes | medium | low |
| Calibrated fusion | yes | yes | medium-high | low |
| Stacking | yes | yes | medium-high | medium |
| Conservative augmentation | yes | yes | medium | low |

The agent should keep the best result in each row and then compare only the winners.

---

## 8. Recommended execution order

The safest order is:
1. reproduce the baseline,
2. add subject-aware normalization,
3. optimize windows,
4. select features,
5. tune hyperparameters,
6. test fusion,
7. test stacking,
8. test conservative augmentation.

This order is preferred because it improves the signal before adding model complexity.

---

## 9. Re-extraction rule

Do not re-extract just because the accuracy is below 70 percent.

Re-extract only if the current files cannot support subject-aware learning or temporal windows.

If the current extracted data still contains enough structure, the better path is to improve the pipeline first and only re-extract later if required.

---

## 10. Agent instruction

Use this instruction for implementation:

> First audit the certified face and voice datasets and confirm whether subject-aware improvement is possible without re-extraction. Reproduce the current LOSO baseline, then test subject-aware normalization, temporal windowing, feature selection, per-modality hyperparameter tuning, calibrated fusion, stacking, and conservative augmentation. Compare every method only under LOSO or grouped validation. Do not use random splits to claim improvement. Do not use synthetic samples unless they are clearly label-preserving and justified. Produce a results table, choose the most stable model stack, and report honestly whether re-extraction is still necessary.

---

## 11. Final note

The best improvement path is often not a larger model. It is usually cleaner features, better windows, better normalization, and better fusion. If that still does not reach 70 percent, the honest answer is that the dataset and task difficulty are limiting the result, not that the model is broken.