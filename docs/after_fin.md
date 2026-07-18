# Production Backend Integration and Frontend Wiring Plan

## 1. Goal

This document instructs the agent to promote the approved production models into the working backend, connect them to the frontend, and expose all required user-facing features for a complete stress-detection product.

The product must support:
- realtime inference,
- uploaded file inference,
- confidence scores,
- SHAP explanations,
- per-modality contribution,
- resilience when a modality is missing,
- and a clear fallback path when the primary model is unavailable.

## 2. Selected production models

The agent must treat the following as the production model set:

1. **SSVB-CASA-AIS** as the primary realtime multimodal engine.
2. **Random Forest** as the fast flat-feature backend classifier for uploaded or batch feature inference.
3. **VBC-CASA-IS** as the fallback multimodal production model if the primary model is unavailable.
4. **LightGBM** as an additional fallback/reference classifier if needed.

The agent must not replace these selections unless a documented gate or regression test proves otherwise.

## 3. Integration objectives

The agent must make the product work end-to-end across:
- frontend capture,
- backend preprocessing,
- model inference,
- explanation generation,
- result rendering,
- model registry resolution,
- and resilience handling.

The final system must behave consistently for:
- live camera input,
- live microphone input,
- live physiological input,
- uploaded image/video/audio files,
- and partial modality availability.

## 4. Backend responsibilities

The backend must become the single inference authority.

It must:
- load the correct production model from the registry,
- inspect the request type,
- route to the primary or fallback model,
- apply the proper preprocessing pipeline,
- return predictions with confidence,
- return per-modality contribution when available,
- return SHAP explanations for tree-based models,
- and return a resilience status explaining whether any modality was missing or substituted.

## 5. API design

The backend must expose at minimum these endpoints:

- `/health`
- `/model/version`
- `/predict/realtime`
- `/predict/upload`
- `/explain`
- `/explain/shap`
- `/modality/status`
- `/fallback/status`

Each endpoint must return structured JSON that the frontend can render without ad hoc parsing.

## 6. Confidence output

For every prediction, the backend must return:
- predicted class,
- probability or score,
- confidence percentage,
- threshold used,
- and an uncertainty note if the score is close to the decision boundary.

The frontend must display confidence clearly and never hide low-confidence predictions.

## 7. SHAP and attribution

The agent must implement explanation support for models that can be explained with SHAP or equivalent feature attribution.

For tree-based models:
- compute global feature importance,
- compute local SHAP values for the current prediction,
- and expose the top positive and negative drivers.

For multimodal models:
- aggregate contributions into modality-level summaries,
- so the UI can show face, voice, and physio contribution separately.

If exact SHAP is too expensive for realtime inference, the agent must generate:
- cached background explanations,
- approximate local explanations,
- or deferred explanation payloads.

## 8. Per-modality contribution

The backend must report:
- face contribution,
- voice contribution,
- physio contribution,
- and missing-modality penalties or substitutions.

If a modality is unavailable, the backend must:
- mark it as missing,
- switch to fallback logic,
- and explain the impact on confidence and prediction quality.

## 9. Resilience rules

The product must survive missing inputs gracefully.

The agent must implement:
- camera off handling,
- microphone unavailable handling,
- physiologic stream unavailable handling,
- partial upload handling,
- and model fallback handling.

The system must never crash because one modality is absent.

If the primary model cannot run, the agent must:
- fall back to VBC-CASA-IS or Random Forest as configured,
- log the fallback reason,
- and surface the fallback in the UI.

## 10. Frontend responsibilities

The frontend must provide a clean user journey for:
- realtime capture,
- file upload,
- result viewing,
- explanation viewing,
- modality availability status,
- and model selection display.

The frontend must show:
- stress prediction,
- confidence score,
- top contributing features,
- modality contributions,
- fallback status,
- and current model version.

## 11. UI panels

The agent must add or update the frontend with these panels:
- live inference panel,
- upload inference panel,
- confidence card,
- explanation card,
- modality contribution chart,
- resilience/fallback badge,
- model metadata panel,
- and history or audit panel if already supported.

## 12. File and folder updates

The agent must update the existing backend and frontend structure without breaking the current registry.

Suggested areas to touch:
- `webapp/backend/`
- `webapp/backend/core/`
- `webapp/backend/runtime/`
- `webapp/backend/explainability/`
- `webapp/backend/monitoring/`
- `webapp/frontend/`
- `webapp/configs/`
- `webapp/models/`

If the repository already contains a registry or model manifest, the agent must keep it as the source of truth.

## 13. Model routing logic

The routing logic must be deterministic.

Recommended routing order:
1. Use SSVB-CASA-AIS for realtime multimodal inference when the required modalities are available.
2. Use Random Forest for uploaded or precomputed flat-feature inference.
3. Use VBC-CASA-IS if the primary multimodal model is unavailable.
4. Use LightGBM only if the above models are unavailable or explicitly configured as fallback.

## 14. Required logging

The agent must log:
- which model was used,
- which modalities were present,
- which modalities were missing,
- explanation generation time,
- inference time,
- and whether a fallback occurred.

Logs must be usable for debugging and model governance.

## 15. Validation checklist

Before declaring the product complete, the agent must verify:
- backend can load the selected production models,
- frontend can call the backend successfully,
- realtime and upload flows both work,
- confidence is rendered,
- SHAP/explanations are rendered,
- modality contribution is rendered,
- fallback behavior is visible,
- and no modality failure breaks the user experience.

## 16. Acceptance criteria

The implementation is only acceptable if:
- the primary and fallback models are reachable from the backend,
- the frontend can display predictions and explanations,
- resilience handling works for missing modalities,
- and the full product can run end-to-end without manual intervention.

## 17. Final instruction

Promote the selected production models into the working backend, connect them to the frontend, expose confidence and attribution features, implement resilience and fallback logic, and deliver a complete, production-ready stress-detection product.