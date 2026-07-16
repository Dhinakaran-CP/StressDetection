"""
build_explainability_bundle.py
Phase 6 offline script — run once after expert models are trained.

Computes SHAP TreeExplainer values for each expert model using a
representative sample from certified datasets, then saves a versioned
explainability_bundle.json to models/.

Usage:
    python -m backend.explainability.build_explainability_bundle
    # or from project root:
    python backend/explainability/build_explainability_bundle.py
"""

import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.explainability.explainability_contract import (
    BUNDLE_VERSION, TOP_K_PER_MODALITY,
    MODALITY_LABELS, MODALITY_GROUPS,
)

EXPERT_MODELS_DIR = os.path.join(ROOT, "models")
CERTIFIED_DIR     = os.path.join(ROOT, "certified_data")
OUTPUT_PATH       = os.path.join(EXPERT_MODELS_DIR, "explainability_bundle.json")

# Models to explain: (modality_key, model_file, scaler_file, certified_csv, feature_cols_start)
MODALITY_CONFIG = {
    "face": {
        "model_file":  "face_expert_lightweight.pkl",
        "scaler_file": "face_scaler_lightweight.pkl",
        "csv":         "face_certified.csv",
        "n_features":  18,
    },
    "voice": {
        "model_file":  "voice_expert_lightweight.pkl",
        "scaler_file": "voice_scaler_lightweight.pkl",
        "csv":         "voice_certified.csv",
        "n_features":  12,
    },
    "physio": {
        "model_file":  "physio_expert_lightweight.pkl",
        "scaler_file": "physio_scaler_lightweight.pkl",
        "csv":         "physio_certified.csv",
        "n_features":  5,
    },
}

META_COLS = {"subject_id", "task_id", "video_id", "window_index",
             "window_start", "window_end", "label"}

SAMPLE_SIZE = 500  # rows to use for SHAP background


def _safe_load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _unwrap_estimator(model):
    """
    Return the innermost tree estimator that SHAP TreeExplainer can handle.
    Handles: raw GBC/RFC, CalibratedClassifierCV, VotingClassifier.
    """
    TREE_TYPES = ("GradientBoosting", "RandomForest", "ExtraTrees",
                  "DecisionTree", "AdaBoost", "XGB", "LGBM", "CatBoost")

    def _is_tree(obj):
        return any(k in type(obj).__name__ for k in TREE_TYPES)

    # Already a supported tree model
    if _is_tree(model):
        return model

    # CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0].estimator
        if _is_tree(inner):
            return inner

    # VotingClassifier / Pipeline with estimators_
    if hasattr(model, "estimators_"):
        for sub in model.estimators_:
            if _is_tree(sub) and "GradientBoosting" in type(sub).__name__:
                return sub  # prefer GBC
        for sub in model.estimators_:
            if _is_tree(sub):
                return sub

    return model  # best effort


def _extract_shap_class1(shap_values):
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        return np.array(shap_values[1], dtype=float)
    arr = np.array(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _build_modality_bundle(modality, cfg, shap_available):
    model_path  = os.path.join(EXPERT_MODELS_DIR, cfg["model_file"])
    scaler_path = os.path.join(EXPERT_MODELS_DIR, cfg["scaler_file"])
    csv_path    = os.path.join(CERTIFIED_DIR, cfg["csv"])
    labels      = MODALITY_LABELS[modality]
    groups      = MODALITY_GROUPS[modality]
    n_feat      = cfg["n_features"]

    if not os.path.exists(model_path):
        print(f"  [SKIP] {modality}: model not found at {model_path}")
        return None
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {modality}: certified CSV not found at {csv_path}")
        return None

    print(f"  Loading {modality} model + scaler …")
    model  = _safe_load_pkl(model_path)
    scaler = _safe_load_pkl(scaler_path) if os.path.exists(scaler_path) else None

    print(f"  Loading certified CSV ({cfg['csv']}) …")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS][:n_feat]

    if len(feature_cols) < n_feat:
        print(f"  [WARN] Only {len(feature_cols)} feature cols found, expected {n_feat}")

    sample = df[feature_cols].dropna().sample(
        n=min(SAMPLE_SIZE, len(df)), random_state=42
    ).values.astype(np.float64)

    if scaler is not None:
        sample_scaled = scaler.transform(sample)
    else:
        sample_scaled = sample

    shap_means = [0.0] * n_feat
    top_features = []

    if shap_available:
        import shap
        try:
            est = _unwrap_estimator(model)
            explainer   = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(sample_scaled)
            sv_class1   = _extract_shap_class1(shap_values)  # (n_samples, n_features)
            mean_abs    = np.mean(np.abs(sv_class1), axis=0)  # (n_features,)
            shap_means  = mean_abs.tolist()

            top_idx = np.argsort(mean_abs)[::-1][:TOP_K_PER_MODALITY]
            for idx in top_idx:
                idx_int = int(idx)
                top_features.append({
                    "feature_index": idx_int,
                    "feature_label": labels[idx_int] if idx_int < len(labels) else f"{modality}_{idx_int}",
                    "feature_group": groups[idx_int] if idx_int < len(groups) else "unknown",
                    "mean_abs_shap": float(mean_abs[idx_int]),
                })
            print(f"  SHAP OK -- top feature: {top_features[0]['feature_label']}")
        except Exception as exc:
            print(f"  [WARN] TreeExplainer failed for {modality}: {exc}")
            print(f"  Trying PermutationExplainer fallback...")
            try:
                background = shap.maskers.Independent(sample_scaled[:50])
                explainer   = shap.PermutationExplainer(model.predict_proba, background)
                shap_values = explainer(sample_scaled[:100])
                sv = shap_values.values  # (n_samples, n_features, n_classes)
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
                mean_abs = np.mean(np.abs(sv), axis=0)
                shap_means = mean_abs.tolist()
                top_idx = np.argsort(mean_abs)[::-1][:TOP_K_PER_MODALITY]
                for idx in top_idx:
                    idx_int = int(idx)
                    top_features.append({
                        "feature_index": idx_int,
                        "feature_label": labels[idx_int] if idx_int < len(labels) else f"{modality}_{idx_int}",
                        "feature_group": groups[idx_int] if idx_int < len(groups) else "unknown",
                        "mean_abs_shap": float(mean_abs[idx_int]),
                    })
                print(f"  PermutationExplainer OK -- top feature: {top_features[0]['feature_label']}")
            except Exception as exc2:
                print(f"  [WARN] PermutationExplainer also failed: {exc2} -- using zero importance")
    else:
        print(f"  [WARN] SHAP not installed — bundle will contain zero importances for {modality}")

    # Fallback: if SHAP unavailable or failed, populate top_features from label list only
    if not top_features:
        for idx in range(min(TOP_K_PER_MODALITY, n_feat)):
            top_features.append({
                "feature_index": idx,
                "feature_label": labels[idx] if idx < len(labels) else f"{modality}_{idx}",
                "feature_group": groups[idx] if idx < len(groups) else "unknown",
                "mean_abs_shap": 0.0,
            })

    return {
        "model_file":     cfg["model_file"],
        "n_features":     n_feat,
        "sample_size":    int(sample.shape[0]),
        "feature_labels": labels[:n_feat],
        "feature_groups": groups[:n_feat],
        "shap_means":     shap_means,
        "top_features":   top_features,
        "shap_available": shap_available,
    }


def build_bundle():
    try:
        import shap
        shap_available = True
        print("SHAP is available — computing real SHAP values.")
    except ImportError:
        shap_available = False
        print("SHAP not installed — bundle will use zero importances. Install with: pip install shap")

    bundle = {
        "version":    BUNDLE_VERSION,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "shap_available": shap_available,
        "models": {},
    }

    for modality, cfg in MODALITY_CONFIG.items():
        print(f"\n[{modality.upper()}]")
        result = _build_modality_bundle(modality, cfg, shap_available)
        if result:
            bundle["models"][modality] = result

    if not bundle["models"]:
        print("\nNo modality bundles could be built. Check expert_models/ and certified_data/.")
        return False

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"\nBundle saved -> {OUTPUT_PATH}")
    print(f"Modalities: {list(bundle['models'].keys())}")

    # SHA-256 manifest + VersionRegistry registration
    try:
        from backend.core.artifact_manifest import ArtifactManifest
        from backend.core.version_registry  import VersionRegistry

        manifest = ArtifactManifest(
            artifact_id   = "explainability_bundle_v1",
            artifact_type = "bundle",
            version       = BUNDLE_VERSION,
            metadata      = {
                "modalities":     list(bundle["models"].keys()),
                "shap_available": shap_available,
                "top_k_per_mod":  TOP_K_PER_MODALITY,
                "sample_sizes":   {m: bundle["models"][m]["sample_size"] for m in bundle["models"]},
                "top_drivers":    {m: bundle["models"][m]["top_features"][0]["feature_label"]
                                   for m in bundle["models"] if bundle["models"][m]["top_features"]},
            },
        )
        manifest.compute_hash(OUTPUT_PATH)
        manifest_path = manifest.save(OUTPUT_PATH)
        print(f"Manifest saved  -> {manifest_path}")
        print(f"SHA-256         -> {manifest.hash[:16]}...")

        registry = VersionRegistry()
        registry.register_bundle("explainability_bundle", manifest)
        print(f"Registered in   -> {registry.registry_path}")

    except Exception as exc:
        print(f"[WARN] Manifest/registry step failed: {exc}")

    return True


if __name__ == "__main__":
    success = build_bundle()
    sys.exit(0 if success else 1)

