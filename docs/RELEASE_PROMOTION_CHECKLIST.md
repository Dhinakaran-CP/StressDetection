# Release Promotion Checklist

To promote a newly trained ML artifact (Model, Dataset, or Explainability Bundle) to the active runtime, complete this checklist:

## 1. Offline Verification
- [ ] Model architecture aligns with `FeatureRuntimeLock` dimensionality.
- [ ] Model performance on LOSO (Leave-One-Subject-Out) validation exceeds baseline threshold (70%).
- [ ] No data leakage observed (Subject IDs correctly isolated).
- [ ] SHAP values generate without errors on hold-out data.

## 2. Artifact Registration
- [ ] Hash digest generated for the artifact (SHA-256).
- [ ] Artifact packaged into `ArtifactManifest`.
- [ ] Registered via `VersionRegistry.register_model()` or equivalent script.
- [ ] Verify `registry.json` history list is correctly appended.

## 3. Golden Replay Verification
- [ ] Restart local backend.
- [ ] Submit Golden Replay rows to `/api/admin/golden_replay`.
- [ ] Ensure that `status: success` is returned and all probability outputs match expected validation splits exactly.

## 4. Gradual Rollout (Future Capability)
- [ ] Route 5% of staging traffic to the new model hash.
- [ ] Monitor `/api/admin/metrics` for latency regressions.
- [ ] If stable, cut over 100% of traffic.
