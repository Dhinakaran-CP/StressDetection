# Full Codebase Audit and Architecture Reconstruction Plan

## 1. Purpose

This document instructs the agent to read the entire repository carefully and reconstruct a complete, clean, and accurate architecture narrative for the project.  
The agent must inspect every folder, every `.md` file, every `.py` file, and every important config or script file, then summarize what each part does, what data it uses, what features it produces, what metrics it reports, and how the project evolved across phases [web:765][web:771][web:773].

## 2. Core rule

The agent must not skip files.  
The agent must not rely on directory names alone.  
The agent must read the actual file contents and trace relationships across the codebase before writing the final architecture summary [web:765][web:772][web:773].

If a file is binary, generated, or unreadable, the agent must log it explicitly and mark it as skipped with a reason.

## 3. Audit scope

The audit must include:
- all `.md` files,
- all `.py` files,
- all configuration files,
- all notebooks if present,
- all scripts,
- all phase folders,
- all pipeline folders,
- all model folders,
- all training and evaluation code,
- all documentation files,
- and all registry or metadata files.

## 4. Required output

The agent must produce a final markdown report that includes:
- repository-wide architecture overview,
- folder-by-folder function summary,
- phase-by-phase project history,
- data sources and dataset lineage,
- feature extraction methods,
- model families,
- evaluation protocols,
- performance metrics,
- and final model progression logic.

## 5. Reading protocol

The agent must process the repository in this order:

### 5.1 Top-level inventory
List all top-level files and directories.

### 5.2 Recursive folder scan
Read every subfolder recursively and identify:
- purpose,
- code ownership,
- phase alignment,
- and file dependencies.

### 5.3 Documentation scan
Read every `.md` file and compare it against actual code behavior.

### 5.4 Python scan
Read every `.py` file and extract:
- imports,
- key functions,
- model definitions,
- preprocessing logic,
- training loops,
- evaluation code,
- export code,
- and hardcoded paths.

### 5.5 Config scan
Read all config files and map them to the code they control.

### 5.6 Phase linkage
For each phase folder, identify:
- what dataset was used,
- what features were extracted,
- what models were trained,
- what metrics were reported,
- what improved,
- and what the next phase changed.

## 6. Architecture reconstruction goals

The agent must reconstruct the project as a clean architecture story:
- raw data ingestion,
- discovery and validation,
- alignment and windowing,
- feature extraction,
- classical baselines,
- temporal deep models,
- GAN augmentation,
- expert routing,
- random forest specialists,
- production model selection,
- and backend deployment [web:765][web:771].

## 7. Phase documentation requirements

For each phase, the agent must report:

### 7.1 Dataset used
Which dataset(s) were used in the phase.

### 7.2 Features used
What features, signals, or windows were used.

### 7.3 Models used
What model families or architectures were trained.

### 7.4 Evaluation protocol
How the model was evaluated:
- LOSO,
- random split,
- cross-validation,
- validation-only,
- test-only,
- or cross-dataset.

### 7.5 Performance metrics
Report the available metrics for that phase:
- accuracy,
- balanced accuracy,
- precision,
- recall,
- F1-score,
- ROC-AUC,
- PR-AUC,
- runtime,
- and any stability measures.

### 7.6 Transition to next phase
Explain why the project moved to the next phase.

## 8. Folder-level reporting

For every folder, the agent must write:
- folder path,
- folder purpose,
- file inventory,
- key dependencies,
- related phase,
- and whether it is research, production, documentation, or archive.

## 9. Metrics reporting rule

The agent must not invent metrics.  
Only metrics explicitly found in the codebase or documentation may be reported.  
If a metric is missing for a folder or phase, the agent must say so clearly.

## 10. Traceability requirement

Every major statement in the final report must be traceable to one or more source files.  
If the report says a phase used a dataset or achieved a metric, the agent must identify the file where that information was found.

## 11. Architecture output format

The final report must contain these sections:

### 11.1 Repository overview
A concise but complete map of the entire project.

### 11.2 Data flow architecture
How data moves from raw input to training and deployment.

### 11.3 Phase-by-phase evolution
What changed from phase to phase.

### 11.4 Model family summary
Which model families were explored and why.

### 11.5 Performance summary
What performed best and under what conditions.

### 11.6 Production selection summary
Which model was ultimately selected for application/backend use.

### 11.7 Open gaps
What still needs improvement, if anything.

## 12. File handling rules

The agent must:
- read every markdown file,
- read every Python file,
- inspect every config file,
- and flag anything that is stale, duplicated, or unused.

If a file appears irrelevant but cannot be confirmed as unused, the agent must mark it as “uncertain” rather than deleting or ignoring it.

## 13. Cleanup recommendations

After the audit, the agent must recommend:
- files to archive,
- files to refactor,
- docs to update,
- duplicated logic to merge,
- and missing architecture docs to create.

The agent must not change code in this phase unless explicitly asked.

## 14. Final validation rule

Before writing the final report, the agent must ensure that:
- every major directory has been inspected,
- every `.md` file has been read,
- every `.py` file has been read,
- and the architecture summary reflects the actual repository, not assumptions.

## 15. Final instruction

Read the entire repository thoroughly, reconstruct the project architecture from the actual files, document the evolution of the system phase by phase, and produce a clean, accurate, and traceable master architecture report.