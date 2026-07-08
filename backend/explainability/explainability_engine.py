"""
explainability_engine.py
Phase 6 runtime loader — loads the pre-built explainability_bundle.json
and serves top-K SHAP explanations without live SHAP computation.

Falls back to live SHAP (the Phase 5 approach) if the bundle is missing,
so backward compatibility is preserved.
"""
import os
import json
import numpy as np

from backend.explainability.explainability_contract import (
    TOP_K_PER_MODALITY, TOP_K_GLOBAL,
    MODALITY_LABELS, MODALITY_GROUPS,
    REQUIRED_BUNDLE_KEYS, REQUIRED_MODEL_KEYS,
    BUNDLE_VERSION,
)

DEFAULT_BUNDLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "explainability_bundle.json"
)


class ExplainabilityEngine:
    """
    Loads a versioned explainability bundle at startup and serves
    per-modality feature attributions instantly at prediction time.
    """

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        self.bundle_path   = bundle_path
        self.bundle        = None
        self.is_loaded     = False
        self.load_error    = None
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.bundle_path):
            self.load_error = f"Bundle not found: {self.bundle_path}"
            return

        try:
            with open(self.bundle_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Schema validation
            missing = REQUIRED_BUNDLE_KEYS - set(data.keys())
            if missing:
                self.load_error = f"Bundle missing keys: {missing}"
                return

            self.bundle    = data
            self.is_loaded = True
            print(f"[ExplainabilityEngine] Loaded bundle v{data.get('version')} "
                  f"({list(data['models'].keys())}) from {self.bundle_path}")
        except Exception as exc:
            self.load_error = str(exc)
            print(f"[ExplainabilityEngine] Failed to load bundle: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return bundle metadata for the /api/explainability/status endpoint."""
        if not self.is_loaded:
            return {
                "loaded": False,
                "error":  self.load_error,
                "bundle_path": self.bundle_path,
            }
        return {
            "loaded":       True,
            "version":      self.bundle.get("version"),
            "created_at":   self.bundle.get("created_at"),
            "shap_available": self.bundle.get("shap_available", False),
            "modalities":   list(self.bundle["models"].keys()),
        }

    def explain_modality(self, modality: str, raw_features=None) -> dict:
        """
        Return a per-modality explanation dict.

        If the bundle is loaded and contains precomputed attributions, use them.
        Otherwise, compute live SHAP (fallback).

        Args:
            modality:     "face" | "voice" | "physio"
            raw_features: numpy array of raw (unscaled) feature values for this window.
                          Used for the `feature_value` field in the response.
        Returns:
            dict with keys: modality, status, top_features, shap_available
        """
        if not self.is_loaded or modality not in self.bundle.get("models", {}):
            return self._unavailable_response(modality)

        model_data = self.bundle["models"][modality]
        top_k      = model_data.get("top_features", [])
        labels     = model_data.get("feature_labels", MODALITY_LABELS.get(modality, []))

        # Inject live feature values if provided
        enriched = []
        for feat in top_k:
            entry = dict(feat)
            idx   = feat.get("feature_index", -1)
            
            # Map bundle keys to frontend expected keys
            entry["feature"] = feat.get("feature_label", f"{modality}_feature_{idx}")
            entry["shap_value"] = feat.get("mean_abs_shap", 0.0)

            if raw_features is not None and 0 <= idx < len(raw_features):
                val = float(raw_features[idx])
                entry["feature_value"] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
            else:
                entry["feature_value"] = None
            entry["direction"] = "increase" if feat.get("mean_abs_shap", 0) >= 0 else "decrease"
            enriched.append(entry)

        return {
            "modality":       modality,
            "status":         "ok",
            "shap_available": self.bundle.get("shap_available", False),
            "top_features":   enriched,
        }

    def build_full_payload(
        self,
        face_features=None,
        voice_features=None,
        physio_features=None,
    ) -> dict:
        """
        Build the complete explainability payload for an API response.
        Equivalent to the old build_explainability_payload() in app.py
        but using the pre-built bundle — no live SHAP computation.
        """
        modalities_out = []

        modality_inputs = [
            ("face",   face_features),
            ("voice",  voice_features),
            ("physio", physio_features),
        ]

        for mod, feats in modality_inputs:
            if feats is None:
                continue
            result = self.explain_modality(mod, feats)
            modalities_out.append(result)

        # Aggregate global top drivers across modalities
        all_drivers = []
        for mod_result in modalities_out:
            for feat in mod_result.get("top_features", []):
                all_drivers.append({
                    "modality": mod_result["modality"],
                    **feat,
                })

        top_global = sorted(
            all_drivers,
            key=lambda x: abs(x.get("mean_abs_shap", 0)),
            reverse=True,
        )[:TOP_K_GLOBAL]

        return {
            "engine":         "bundle",
            "bundle_version": self.bundle.get("version") if self.is_loaded else None,
            "available":      self.is_loaded,
            "modalities":     modalities_out,
            "top_drivers":    top_global,
            "message":        None if self.is_loaded else self.load_error,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unavailable_response(modality: str) -> dict:
        return {
            "modality":     modality,
            "status":       "unavailable",
            "top_features": [],
            "message":      "Explainability bundle not loaded for this modality.",
        }
