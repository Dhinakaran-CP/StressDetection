# Implementation Plan: Codebase Audit & Production Integration

This plan outlines the two-part process to (1) perform a complete repository-wide codebase audit and (2) implement the production backend integration and frontend wiring for real-time and upload stress detection.

---

## User Review Required

> [!IMPORTANT]
> **Scope of Deliverables**:
> 1. **Codebase Audit**: We will conduct a thorough scan of all Python files, configurations, and markdown documentation to reconstruct the project's evolution and file linkages.
> 2. **API & Frontend Wiring**: We will modify `webapp/backend/app.py` to expose endpoints for health, model versions, real-time predictions, file uploads, SHAP explanations, modality status, and fallback mechanisms. We will update the React frontend dashboard to display predictions, confidence cards, modality contributions, and fallback status badges.

## Open Questions

> [!IMPORTANT]
> **1. Real-time SHAP Approximations**: Real-time SHAP values for tree models can be computationally expensive to compute per frame. We propose using a fast, pre-cached model-explanation driver matrix (aggregated modality importances) for real-time inference, and running local SHAP explanations on-demand or during file/batch uploads. Please confirm if this is acceptable.
>
> **2. React Charting Library**: For displaying modality contributions and feature drivers on the React frontend, we propose using standard styled CSS bars or SVG-based charts to avoid introducing heavy external npm charting dependencies (like `recharts` or `chart.js`), unless you have a preference. Please let us know.
>
> **3. Multimodal Fallbacks**: In case of a fallback, if `SSVB-CASA-AIS` is unavailable, we will route to `VBC-CASA-IS` (for sequence/realtime) or the production `Random Forest` model. We will verify the registry config to ensure fallbacks are resolved dynamically.

---

## Proposed Changes

### 1. Codebase Inventory and Scan (Phase A)
* Recursively scan all files in `research/`, `webapp/`, and `pipeline/` to build a complete inventory.
* Read documentation files to trace how the project progressed from single-dataset baselines to the 91-fold LOSO model zoo.
* Generate a comprehensive audit report at **`research/codebase_architecture_report.md`**.

---

### 2. Backend API Wiring (Phase B)

#### [MODIFY] [app.py](file:///c:/Users/StressProject.DESKTOP-U6P7JQT/Desktop/StressDetectionUsingML/webapp/backend/app.py)
* Add `/health` and `/model/version` endpoints.
* Implement `/predict/realtime` and `/predict/upload` to handle camera, mic, and uploaded data payloads.
* Integrate the `VersionRegistry` and `RuntimeEngine` to handle model selection dynamically:
  - Route real-time data to `SSVB-CASA-AIS`.
  - Route uploaded files/flat features to `Random Forest`.
  - Handle VBC-CASA-IS and LightGBM fallbacks deterministic based on availability or manual configs.
* Add `/explain/shap` to compute local SHAP contributions for Random Forest and modality-level contributions for the deep experts.
* Implement modality status monitoring and penalty calculations when modalities are missing.

---

### 3. Frontend Dashboard Wiring (Phase B)

#### [MODIFY] [Dashboard.js](file:///c:/Users/StressProject.DESKTOP-U6P7JQT/Desktop/StressDetectionUsingML/webapp/frontend/src/pages/Dashboard.js)
* Update the live prediction card to render the predicted stress label, probability, and confidence score.
* Add a **Resilience/Fallback Badge** indicating which model is currently running and if a fallback is active.
* Add a **Modality Contribution Panel** showing the contribution percentage of Face, Voice, and Physio streams.
* Add a **SHAP Feature Drivers list** showing the top positive and negative features influencing the stress decision.
* Wire up the **File Upload panel** to let users upload feature sets or media, calling `/predict/upload` and displaying results.

---

## Verification Plan

### Automated & Integration Checks
1. Run backend tests to verify that all endpoints (`/predict/realtime`, `/predict/upload`, `/explain/shap`, etc.) return 200 OK with correct JSON contracts.
2. Launch the FastAPI server and React frontend locally.
3. Simulate missing modalities (e.g. camera off, mic muted) and verify that the backend does not crash and the frontend correctly renders fallback badges.
4. Export walkthrough findings with screenshots.
