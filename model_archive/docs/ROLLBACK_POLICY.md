# Stress Detection System - Rollback Policy

## 1. Objective
Define the triggers, procedures, and safety checks for hot-swapping a deployed machine learning model or explainability bundle back to a previously known-good state.

## 2. Triggers for Rollback
A rollback MUST be initiated if any of the following conditions are met within the telemetry dashboard:
- **Accuracy Drop**: The user-reported correctness score drops by >15% over a 48-hour period.
- **Latency Spikes**: The p95 inference latency exceeds 250ms for more than 5 minutes.
- **Feature Drift**: The `DriftMonitor` detects that the mean feature vector of incoming traffic deviates by >2 standard deviations from the baseline for >1 hour.
- **Missing Modality Errors**: Unhandled modality gaps (e.g., face processing fails >5% of the time due to strict landmarking).

## 3. Rollback Procedure
1. Identify the stable version from the registry history.
2. Issue an HTTP POST to the admin rollback endpoint:
   ```bash
   curl -X POST http://localhost:5000/api/admin/rollback \
        -H "Content-Type: application/json" \
        -d '{"model_key": "face", "version": "1.0.0"}'
   ```
3. Verify that `/api/runtime/status` reflects the target version.
4. Run the Golden Replay suite:
   ```bash
   curl -X POST http://localhost:5000/api/admin/golden_replay ...
   ```
   Ensure a 100% exact match against expected outputs.

## 4. Post-Mortem
All rollbacks must be followed by a post-mortem review documenting the root cause and a reproducible offline regression test before a new rollout is permitted.
