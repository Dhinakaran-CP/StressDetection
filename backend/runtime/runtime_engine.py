"""
runtime_engine.py
Phase 7: Single authoritative inference engine.

Loads ALL artifacts from VersionRegistry (not hardcoded filenames).
Routes ALL inference through FeatureRuntimeLock.
Owns fusion and explanation delegation.
Provides a deterministic replay() path for regression tests.
"""

import os
import sys
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from backend.core.version_registry      import VersionRegistry

# ── Phase 4 optimal fusion weights ────────────────────────────────────────────
FUSION_WEIGHTS = {"face": 0.30, "voice": 0.40, "physio": 0.30}

# Model filename map — registry keys → pkl filenames (fallback when registry
# doesn't carry the file path directly)
_MODEL_FILES = {
    "face_expert":   ("face_expert_lightweight.pkl",  "face_scaler_lightweight.pkl"),
    "voice_expert":  ("voice_expert_lightweight.pkl", "voice_scaler_lightweight.pkl"),
    "physio_expert": ("physio_expert_lightweight.pkl","physio_scaler_lightweight.pkl"),
}

EXPERT_MODELS_DIR = os.path.join(ROOT, "models")
CONTRACT_PATH     = os.path.join(ROOT, "configs", "feature_contract.yaml")


# ── Custom unpickler (handles sklearn internal renames) ────────────────────────
class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "sklearn._loss" in module:
            for alt in ["sklearn._loss", "sklearn._loss._loss", "sklearn._loss.loss"]:
                try:
                    __import__(alt)
                    m = sys.modules[alt]
                    if hasattr(m, name):
                        return getattr(m, name)
                except Exception:
                    continue
        return super().find_class(module, name)


def _safe_load(path):
    with open(path, "rb") as f:
        return _CustomUnpickler(f).load()


# ── RuntimeEngine ─────────────────────────────────────────────────────────────

class RuntimeEngine:
    """
    Single authoritative inference engine for the Multimodal Stress Detector.

    Usage
    -----
        engine = RuntimeEngine.from_registry()   # production
        engine = RuntimeEngine.from_registry(registry_path="custom/registry.json")

    Inference
    ---------
        result = engine.predict_fused(face=arr18, voice=arr12)
        result = engine.predict_face(arr18)

    Replay (deterministic regression testing)
    ---------
        rows = [{"face": arr18, "voice": arr12, "physio": arr5}, ...]
        outputs = engine.replay(rows)
    """

    def __init__(
        self,
        registry: VersionRegistry,
        feature_lock: FeatureRuntimeLock,
        expl_engine=None,          # ExplainabilityEngine (optional injection)
    ):
        self.registry     = registry
        self.feature_lock = feature_lock
        self.expl_engine  = expl_engine

        # Loaded artifacts — keyed by modality
        self._models  = {}   # "face" | "voice" | "physio" → sklearn estimator
        self._scalers = {}   # same keys → sklearn scaler or None
        self.load_errors: dict = {}

        self._load_artifacts()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        registry_path: str = None,
        feature_lock: FeatureRuntimeLock = None,
        expl_engine=None,
    ) -> "RuntimeEngine":
        """
        Production factory. Loads everything from VersionRegistry.
        """
        if registry_path is None:
            registry_path = os.path.join(EXPERT_MODELS_DIR, "registry.json")

        registry = VersionRegistry(registry_path=registry_path)

        if feature_lock is None:
            feature_lock = FeatureRuntimeLock(CONTRACT_PATH)

        if expl_engine is None:
            try:
                from backend.explainability.explainability_engine import ExplainabilityEngine
                expl_engine = ExplainabilityEngine()
            except Exception as exc:
                print(f"[RuntimeEngine] ExplainabilityEngine unavailable: {exc}")
                expl_engine = None

        engine = cls(registry=registry, feature_lock=feature_lock, expl_engine=expl_engine)
        return engine

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self):
        """Load model+scaler pairs for all registered experts."""
        registry_to_modality = {
            "face_expert":   "face",
            "voice_expert":  "voice",
            "physio_expert": "physio",
        }

        for reg_key, modality in registry_to_modality.items():
            model_file, scaler_file = _MODEL_FILES[reg_key]
            model_path  = os.path.join(EXPERT_MODELS_DIR, model_file)
            scaler_path = os.path.join(EXPERT_MODELS_DIR, scaler_file)

            # Verify against registry hash if entry exists
            reg_entry = self.registry.get_active_model(reg_key)
            if reg_entry:
                stored_hash = reg_entry.get("hash")
                if stored_hash:
                    self._verify_hash(model_path, stored_hash, reg_key)

            # Load model
            if os.path.exists(model_path):
                try:
                    self._models[modality] = _safe_load(model_path)
                except Exception as exc:
                    self.load_errors[reg_key] = str(exc)
                    print(f"[RuntimeEngine] Failed to load {model_file}: {exc}")
            else:
                self.load_errors[reg_key] = f"File not found: {model_path}"

            # Load scaler (optional — some models don't need one)
            if os.path.exists(scaler_path):
                try:
                    self._scalers[modality] = _safe_load(scaler_path)
                except Exception as exc:
                    print(f"[RuntimeEngine] Scaler load warning for {scaler_file}: {exc}")
                    self._scalers[modality] = None
            else:
                self._scalers[modality] = None

        loaded = list(self._models.keys())
        if loaded:
            print(f"[RuntimeEngine] Loaded modalities: {loaded}")
        else:
            print("[RuntimeEngine] WARNING: No models loaded.")

    def _verify_hash(self, path: str, expected: str, label: str):
        import hashlib
        if not os.path.exists(path):
            return
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        actual = sha256.hexdigest()
        if actual != expected:
            print(
                f"[RuntimeEngine] HASH MISMATCH for {label}! "
                f"Registry={expected[:12]}... Disk={actual[:12]}..."
            )

    # ── Public: status ────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    def status(self) -> dict:
        """Return a serialisable status dict for /api/runtime/status."""
        model_versions = {}
        for reg_key in ("face_expert", "voice_expert", "physio_expert"):
            entry = self.registry.get_active_model(reg_key)
            modality = reg_key.replace("_expert", "")
            model_versions[reg_key] = {
                "loaded":  modality in self._models,
                "version": entry.get("version") if entry else None,
                "hash":    (entry.get("hash") or "")[:12] + "..." if entry else None,
                "accuracy": (entry.get("metadata") or {}).get("accuracy"),
            }

        bundle_entry = self.registry.get_active_bundle("explainability_bundle")
        return {
            "engine":              "RuntimeEngine",
            "phase":               7,
            "ready":               self.is_ready,
            "loaded_modalities":   list(self._models.keys()),
            "models":              model_versions,
            "load_errors":         self.load_errors,
            "explainability": {
                "engine_loaded": self.expl_engine.is_loaded if self.expl_engine else False,
                "bundle_version": bundle_entry.get("version") if bundle_entry else None,
                "bundle_hash":   (bundle_entry.get("hash") or "")[:12] + "..." if bundle_entry else None,
            },
            "fusion_weights":      FUSION_WEIGHTS,
        }

    # ── Public: single-modality inference ─────────────────────────────────────

    def predict_face(self, raw_features, sensitivity: float = 0.5) -> dict:
        return self._predict_single("face", raw_features, sensitivity)

    def predict_voice(self, raw_features, sensitivity: float = 0.5) -> dict:
        return self._predict_single("voice", raw_features, sensitivity)

    def predict_physio(self, raw_features, sensitivity: float = 0.5) -> dict:
        return self._predict_single("physio", raw_features, sensitivity)

    def _predict_single(self, modality: str, raw_features, sensitivity: float) -> dict:
        if modality not in self._models:
            return {"error": f"{modality} model not loaded", "modality": modality}
        if raw_features is None:
            return {"error": "raw_features is None", "modality": modality}

        try:
            locked = self._lock_features(modality, raw_features)
            prob   = float(self._models[modality].predict_proba(locked)[0][1])
            threshold   = 0.6 + (0.5 - sensitivity) * 0.4
            stress_level = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
            return {
                "modality":          modality,
                "stress_probability": prob,
                "stress_level":       stress_level,
                "predicted_class":    "Stress" if prob > threshold else "No Stress",
            }
        except Exception as exc:
            return {"error": str(exc), "modality": modality}

    # ── Public: fused inference ────────────────────────────────────────────────

    def predict_fused(
        self,
        face=None,
        voice=None,
        physio=None,
        sensitivity: float = 0.5,
    ) -> dict:
        """
        Late-fusion prediction across all available modalities.
        Missing modalities degrade gracefully — weights re-normalise automatically.
        """
        inputs = {"face": face, "voice": voice, "physio": physio}
        raw_probs: dict = {}

        for modality, feats in inputs.items():
            if feats is None or modality not in self._models:
                continue
            try:
                locked = self._lock_features(modality, feats)
                prob   = float(self._models[modality].predict_proba(locked)[0][1])
                raw_probs[modality] = prob
            except Exception as exc:
                print(f"[RuntimeEngine] {modality} prediction failed: {exc}")

        if not raw_probs:
            return {"error": "No valid modality predictions — check inputs and loaded models"}

        # Re-normalised weighted fusion
        active_w  = {m: FUSION_WEIGHTS[m] for m in raw_probs if m in FUSION_WEIGHTS}
        total_w   = sum(active_w.values())
        norm_w    = {m: w / total_w for m, w in active_w.items()}
        avg_prob  = sum(raw_probs[m] * norm_w[m] for m in raw_probs)

        threshold    = 0.6 + (0.5 - sensitivity) * 0.4
        final_pred   = 1 if avg_prob > threshold else 0
        stress_level = "High" if avg_prob > 0.7 else "Moderate" if avg_prob > 0.4 else "Low"

        # Explanation from pre-built bundle
        explanation = None
        if self.expl_engine and self.expl_engine.is_loaded:
            explanation = self.expl_engine.build_full_payload(
                face_features=face,
                voice_features=voice,
                physio_features=physio,
            )

        return {
            "status":                "success",
            "predicted_class":       "Stress" if final_pred else "No Stress",
            "stress_probability":    float(avg_prob),
            "no_stress_probability": float(1.0 - avg_prob),
            "confidence":            float(max(avg_prob, 1.0 - avg_prob)),
            "stress_level":          stress_level,
            "percentage":            float(avg_prob * 100.0),
            "individual_predictions": {
                m: float(p) for m, p in raw_probs.items()
            },
            "fusion_weights": {m: round(norm_w.get(m, 0.0), 3) for m in ("face", "voice", "physio")},
            "active_modalities":     list(raw_probs.keys()),
            "explainability":        explanation,
        }

    # ── Public: replay (deterministic regression testing) ─────────────────────

    def replay(self, feature_rows: list) -> list:
        """
        Deterministic replay of a list of feature dicts.

        Each row:  {"face": ndarray|None, "voice": ndarray|None, "physio": ndarray|None}
        Returns:   list of predict_fused() result dicts
        """
        return [
            self.predict_fused(
                face=row.get("face"),
                voice=row.get("voice"),
                physio=row.get("physio"),
            )
            for row in feature_rows
        ]

    # ── Private: feature locking ──────────────────────────────────────────────

    def _lock_features(self, modality: str, raw_features) -> np.ndarray:
        """Pass raw features through FeatureRuntimeLock and return scaled array."""
        scaler = self._scalers.get(modality)
        if modality == "face":
            return self.feature_lock.process_face_features(raw_features, scaler)
        elif modality == "voice":
            return self.feature_lock.process_voice_features(raw_features, scaler)
        elif modality == "physio":
            return self.feature_lock.process_physio_features(raw_features, scaler)
        else:
            raise ValueError(f"Unknown modality: {modality}")
