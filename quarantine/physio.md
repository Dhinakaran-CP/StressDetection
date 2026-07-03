# Physio Extraction Plan for StressID

This document tells the agent how to extract and certify the physiological modality from the StressID data already added to the repository. The goal is to build a physiology pipeline that matches the same quality, leakage control, and release discipline already used for face and voice.

StressID contains synchronized physiological signals composed of ECG, EDA, and respiration, recorded with wearable sensors and split into subject-task recordings with self-assessment labels.[page:0] The dataset paper also states that each task is identified by `subjectname_task`, which makes subject-aware extraction and LOSO evaluation appropriate.[page:0]

---

## 1. Objective

The agent must create a certified physiological dataset release that:

- preserves `subject_id`,
- preserves `task_name`,
- preserves task chronology,
- extracts short fixed windows instead of whole-session averages,
- and supports strict LOSO training and evaluation.

The output must be compatible with the current face and voice release pipeline so the final fusion engine can treat all three modalities consistently.

---

## 2. What the agent must not do

The agent must not:

- average an entire task into one row,
- remove subject identity,
- mix cleaning and feature extraction without clear stages,
- evaluate with random splits,
- or create features that cannot be reproduced at runtime.

The physiological branch should be a first-class expert, not a one-off preprocessing script.

---

## 3. Data inputs

The agent should first inspect the uploaded StressID physiological files and identify:

- raw ECG recordings,
- raw EDA recordings,
- raw respiration recordings,
- session or task metadata,
- subject identifiers,
- label files or annotation tables,
- and any timestamps or task boundary markers.

StressID records ECG, EDA, and respiration at 500 Hz with synchronized acquisition, and the tasks are available as subject-task recordings.[page:0] That means the agent should aim to reconstruct window-level samples from each task rather than compressing the entire task into a single feature vector.

---

## 4. Target schema

Every extracted physiological row must contain at least:

- `subject_id`
- `session_id` if available
- `task_name`
- `window_id`
- `window_start_ms`
- `window_end_ms`
- `modality = physio`
- `label`
- `ecg_*` features
- `eda_*` features
- `resp_*` features
- `signal_quality`
- `source_file`
- `release_id`

If the source naming already contains `subjectname_task`, the agent should normalize it into a structured schema rather than keeping it as an opaque string.

---

## 5. Extraction workflow

### Step 1: Inventory the raw files

The agent should map every physiological recording to its subject and task.

Required actions:
- list all files,
- infer modality per file,
- infer subject and task from filename or metadata,
- verify all available tasks,
- identify missing or corrupted recordings.

If the mapping is ambiguous, the agent must stop and report the ambiguity before extraction begins.

### Step 2: Segment into windows

The agent should split each physiological recording into fixed-length windows.

Recommended starting point:
- 10 to 30 second windows for physiological signals,
- with overlap if the source signal is long enough.

If the dataset task duration is short, the agent can adjust the window length, but it must stay consistent across the full release.

### Step 3: Clean each signal

Before feature extraction, each physiological stream must be cleaned separately.

ECG:
- bandpass filtering,
- R-peak detection,
- artifact rejection,
- RR interval cleanup.

EDA:
- low-pass or smoothing filter,
- tonic/phasic decomposition if available,
- spike removal,
- missing segment handling.

Respiration:
- baseline smoothing,
- peak/trough cleanup,
- artifact suppression,
- cycle validation.

The cleaning rules must be written in code, not described only in a notebook.

### Step 4: Extract features

The agent should extract features in three separate groups.

#### ECG / HRV features

Recommended feature families:
- RR statistics,
- SDNN,
- RMSSD,
- pNN20 / pNN50,
- mean HR,
- HRV frequency-domain features,
- nonlinear HRV features.

StressID explicitly describes ECG-derived HRV features as part of the physiological baseline, so these should be the core of the ECG branch.[page:0]

#### EDA features

Recommended feature families:
- tonic level statistics,
- phasic response count,
- SCR peak count,
- SCR amplitude,
- SCR duration,
- rise time,
- recovery time,
- slope or derivative features.

StressID reports EDA features based on skin conductance level and response components, including peak count and amplitude.[page:0]

#### Respiration features

Recommended feature families:
- breathing rate,
- respiration interval variability,
- amplitude statistics,
- cycle length,
- frequency-domain respiration measures,
- variability ratios.

StressID includes respiration as a separate physiological input, so the agent should preserve it as its own feature family rather than folding it into ECG or EDA.[page:0]

### Step 5: Build the physio row

For each window, the agent should combine the ECG, EDA, and respiration features into one structured row. The row must keep subject and task metadata intact.

### Step 6: Certify the dataset

After extraction, the agent must validate:
- schema completeness,
- subject coverage,
- task coverage,
- duplicate rows,
- monotonic time ordering,
- label validity,
- window alignment,
- feature count consistency,
- missingness rate.

If any check fails, the release is not certified.

---

## 6. Training compatibility

The extracted physio dataset must support the same project-level rules used by face and voice:

- certification before training,
- subject-disjoint LOSO evaluation,
- release manifests,
- reproducible feature order,
- runtime/offline parity,
- and versioned model artifacts.

The StressID paper notes that physiological baselines and multimodal baselines are both available, which makes the physio expert suitable for direct integration into the current late-fusion architecture.[page:0]

---

## 7. Recommended model path

The agent should start with a lightweight classical model, not a heavy deep model.

Suggested first models:
- Random Forest,
- SVM,
- Gradient Boosting,
- calibrated ensemble if needed.

This follows the same release discipline as face and voice and is consistent with the StressID baseline style, which uses classical machine learning for physiological features.[page:0]

---

## 8. Evaluation protocol

The agent must evaluate the physio expert using:

- strict LOSO,
- subject-level grouping,
- no test-subject leakage,
- and fold-level metric logging.

Recommended metrics:
- accuracy,
- F1 score,
- balanced accuracy,
- calibration score,
- per-subject confusion summary.

If LOSO is materially lower than random split, that is expected and should not be treated as failure.

---

## 9. Integration into the existing architecture

The physio branch should be added as a full expert module:

1. raw signal loader,
2. cleaning layer,
3. windowing layer,
4. feature extractor,
5. certifier,
6. physio expert trainer,
7. release packager,
8. fusion input for the final engine.

The new branch should reuse the same release manifest and registry style already used by face and voice so the final fusion engine can load all three experts in a common way.

---

## 10. Agent instruction block

Use this instruction for implementation:

> Inspect the uploaded StressID physiological data and build a certified ECG, EDA, and respiration extraction pipeline that preserves subject identity, task identity, and short temporal windows. Clean each signal, extract window-level physiological features, certify the resulting dataset, and train a lightweight LOSO physio expert with versioned release artifacts. Do not average entire tasks into one row, do not drop subject metadata, and do not evaluate with random splits.

---

## 11. Final note

The physiological branch should be treated as a serious expert, not a side experiment. StressID explicitly supports synchronized physiological, video, and audio data, and it uses task-level subject identifiers that make a careful extraction pipeline possible.[page:0] If the agent builds this branch correctly, it will fit directly into the same architecture as face and voice and strengthen the final late-fusion system.