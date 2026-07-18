# Final Model Selection and Backend Promotion Plan

## 1. Purpose

This file defines the final model choice workflow for the repository. The goal is to choose one research champion and one deployable backend model using subject-independent validation and cross-dataset generalization.

## 2. Updated model preference

Based on the latest combined evidence:
- **Primary research champion:** SSVB-CASA-AIS.
- **Primary backend fallback:** Random Forest.

## 3. Selection criteria

Select the final backend model using:
- LOSO validation accuracy,
- balanced accuracy,
- stressed-class F1,
- AUC-ROC,
- PR-AUC,
- fold stability,
- cross-dataset robustness,
- runtime,
- and implementation complexity.

## 4. Preferred hierarchy

1. SSVB-CASA-AIS.
2. Random Forest.
3. GAN-augmented Random Forest.
4. Expert ensemble / MoE variants.
5. Temporal deep baselines.

## 5. Validation protocol

The agent must evaluate candidates on:
- WESAD physio-only benchmark,
- combined multi-dataset benchmark,
- and the project’s LOSO folds.

All candidates must be tested under the same split policy.

## 6. Backend selection rule

If SSVB-CASA-AIS is consistently superior and deployment cost is acceptable, promote it to the backend model.

If runtime, portability, or operational complexity is too high, promote Random Forest as the backend model and keep SSVB-CASA-AIS as the research champion.

## 7. Repository changes

The agent must update:
- `webapp/models/`
- `webapp/backend/`
- model registry files,
- preprocessing configs,
- and inference wrappers.

Suggested structure:

```text
webapp/models/
├── research_champion/
├── backend_selected/
├── fallback_models/
└── model_registry.json
```

## 8. Export requirements

For the final selected model, export:
- model weights,
- preprocessing pipeline,
- feature schema,
- label encoder,
- class mapping,
- and model card with validation metrics.

## 9. Final instruction

Choose SSVB-CASA-AIS as the main candidate and compare it directly with Random Forest under the same validation protocol. Promote the model that best balances generalization, stability, and deployment practicality.