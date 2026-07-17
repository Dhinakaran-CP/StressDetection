# Dataset Discovery Agent Instructions

## 1. Purpose

This file defines the discovery phase for the raw dataset stored as per-user ZIP archives under `data/stress_d/`.  
The agent must inspect the dataset structure, infer the internal schema, identify modality files and labels, and decide the correct extraction strategy before performing any feature extraction or windowing [web:732][web:738][web:743].

## 2. Core principle

Do not assume a fixed internal ZIP layout.  
Do not start windowing before discovery.  
Do not extract features before confirming the data structure and label mapping.

The agent must first understand the dataset, then select the appropriate downstream pipeline.

## 3. Inputs

The agent must scan:

```text
data/stress_d/
```

Each ZIP file must be treated as one user or one subject source unless the internal metadata says otherwise.

## 4. Discovery objectives

The agent must identify:
- archive integrity,
- subject or user ID,
- session folders,
- modality files,
- annotation files,
- timestamp availability,
- label format,
- file naming conventions,
- sampling frequency,
- and whether data is raw, partially processed, or already windowed.

## 5. Folder structure

The agent must create:

```text
dataset_discovery/
├── zip_inventory/
├── extracted_preview/
├── schema_reports/
├── modality_maps/
├── label_maps/
├── timestamp_maps/
├── logs/
└── decision_reports/
```

## 6. Discovery workflow

### Step 1: ZIP inventory
List all ZIP files in `data/stress_d/`.  
Record:
- filename,
- size,
- detected subject ID,
- modified date if available.

### Step 2: Safe preview extraction
Extract only one ZIP at a time into a staging folder.  
Do not process the full dataset yet.

### Step 3: Structure scan
Inspect the extracted contents and detect:
- top-level folders,
- nested session folders,
- raw signal files,
- video or image files,
- label files,
- metadata files,
- logs or sidecar files.

### Step 4: Schema inference
Infer whether the ZIP contains:
- raw continuous streams,
- pre-segmented samples,
- already-windowed records,
- or mixed structure.

### Step 5: Modality detection
Map each file to a modality:
- face/video,
- audio,
- physiology,
- labels,
- metadata.

### Step 6: Timestamp check
Determine whether files contain:
- absolute timestamps,
- relative timestamps,
- frame indexes,
- event markers,
- or no timing information.

### Step 7: Label inspection
Identify:
- label file names,
- label encoding,
- class names,
- per-session labels,
- per-window labels,
- or per-subject labels.

### Step 8: Alignment feasibility
Decide whether:
- modalities are already aligned,
- alignment must be done by timestamp,
- alignment must be done by event markers,
- or the data must be handled as non-synchronized feature tables.

### Step 9: Decide extraction mode
The agent must choose one of the following modes:

- **Mode A: Session-first extraction**
  Use when the ZIP contains session folders or recording blocks.

- **Mode B: Timestamp alignment first**
  Use when raw synchronized streams are present.

- **Mode C: Window-first validation**
  Use when samples are already segmented or windowed.

- **Mode D: Feature-table normalization**
  Use when the dataset is already converted into tabular feature rows.

## 7. Decision rules

The agent must follow these rules:

- If the dataset contains raw streams with timestamps, align first and window later.
- If the dataset contains session folders, extract session-wise first.
- If the dataset contains pre-windowed rows, validate schema first and then normalize.
- If labels are stored separately, map them only after subject and session identification.
- If a modality is missing for a subject, record the missingness and continue if the rest of the sample is usable.

## 8. Validation checks

For each ZIP, the agent must verify:
- archive integrity,
- readable files,
- valid timestamps,
- consistent subject ID,
- consistent session structure,
- label presence,
- modality completeness.

If a ZIP fails validation, the agent must log it and continue to the next one.

## 9. Output reports

For every inspected ZIP, the agent must write a report containing:
- detected structure,
- detected modalities,
- detected label format,
- timestamp format,
- alignment mode recommendation,
- extraction mode recommendation,
- any warnings or missing data.

## 10. Consolidated dataset decision

After inspecting a representative sample of ZIP files, the agent must decide:
- whether one unified extraction script is sufficient,
- or whether different ZIP groups require different parsing rules.

If the dataset varies across subjects, the agent must create a fallback parser strategy.

## 11. Final decision logic

The agent must not hardcode assumptions from one ZIP to the entire dataset unless the inspection proves that the structure is consistent across users.

The output of discovery must define:
- how to extract,
- how to align,
- how to window,
- and how to label the data in the next pipeline.

## 12. Final instruction

Inspect each ZIP carefully, infer its schema, map modalities and labels, determine the correct extraction mode, and only then hand off to the feature-extraction pipeline.