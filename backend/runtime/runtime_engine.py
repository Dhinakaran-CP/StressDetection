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

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from backend.core.version_registry      import VersionRegistry

if TORCH_AVAILABLE:
    class ModalityEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=16):
            super().__init__()
            self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm1d(hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, 2)
            
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.permute(0, 2, 1)
            gru_out, hidden = self.gru(x)
            latent = gru_out[:, -1, :] 
            logits = self.classifier(latent) 
            return logits

    class DynamicRouter(nn.Module):
        def __init__(self, num_modalities=3):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(num_modalities * 2 + num_modalities, 16),
                nn.ReLU(),
                nn.Linear(16, num_modalities),
                nn.Softmax(dim=1)
            )
        def forward(self, x):
            return self.mlp(x)


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

        # Phase 8 Deep Learning variables
        self.use_deep = False
        self.deep_models = {}
        self.deep_sequence_history = {"face": [], "voice": [], "physio": []}

        # Phase 4 Methodology: Subject-Aware Normalization & Temporal Windowing
        self.feature_history = {"face": [], "voice": [], "physio": []}
        self.calibration_baselines = {"face": None, "voice": None, "physio": None}
        self.calibrating = {"face": True, "voice": True, "physio": True}
        self.calibration_frames = 2
        self.window_size = 2

        self._load_artifacts()

    def reset_calibration(self):
        """Reset the calibration baselines (e.g. for a new subject)."""
        self.feature_history = {"face": [], "voice": [], "physio": []}
        self.calibration_baselines = {"face": None, "voice": None, "physio": None}
        self.calibrating = {"face": True, "voice": True, "physio": True}
        self.deep_sequence_history = {"face": [], "voice": [], "physio": []}

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
        """Load model+scaler pairs for all registered experts (supporting both deep learning and classical)."""
        config_path = os.path.join(EXPERT_MODELS_DIR, "deep_fusion_config.json")
        self.strategy_used = "standard"
        self.use_deep = False
        
        if os.path.exists(config_path) and TORCH_AVAILABLE:
            import json
            try:
                with open(config_path, "r") as f:
                    deep_cfg = json.load(f)
                if deep_cfg.get("use_dynamic_router"):
                    self.use_deep = True
                    primary_strategy = deep_cfg.get("primary_strategy", "adversarial")
                    
                    if primary_strategy == "adversarial":
                        # Check if all adversarial files are present
                        adv_files = [
                            os.path.join(EXPERT_MODELS_DIR, "adv_face_expert.pt"),
                            os.path.join(EXPERT_MODELS_DIR, "adv_voice_expert.pt"),
                            os.path.join(EXPERT_MODELS_DIR, "adv_physio_expert.pt"),
                            os.path.join(EXPERT_MODELS_DIR, "adv_fusion_router.pt")
                        ]
                        if all(os.path.exists(f) for f in adv_files):
                            self.strategy_used = "adversarial"
                    
                    self._load_deep_artifacts()
            except Exception as exc:
                print(f"[RuntimeEngine] Error reading deep_fusion_config.json: {exc}")

        registry_to_modality = {
            "face_expert":   "face",
            "voice_expert":  "voice",
            "physio_expert": "physio",
        }

        for reg_key, modality in registry_to_modality.items():
            model_file, scaler_file = _MODEL_FILES[reg_key]
            
            # If use_deep is active, we will load the deep scalers instead for active modalities
            if self.use_deep and modality in ["face", "voice", "physio"]:
                prefix = "adv_" if self.strategy_used == "adversarial" else "deep_"
                deep_scaler_file = f"{prefix}{modality}_scaler.pkl"
                scaler_path = os.path.join(EXPERT_MODELS_DIR, deep_scaler_file)
                if os.path.exists(scaler_path):
                    try:
                        self._scalers[modality] = _safe_load(scaler_path)
                    except Exception as exc:
                        print(f"[RuntimeEngine] Deep scaler load warning for {deep_scaler_file}: {exc}")
                        self._scalers[modality] = None
                continue

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

        # Load classical backups so that old unit tests pass
        if self.use_deep:
            for reg_key in ["face_expert", "voice_expert", "physio_expert"]:
                modality = reg_key.split("_")[0]
                model_file, _ = _MODEL_FILES[reg_key]
                model_path = os.path.join(EXPERT_MODELS_DIR, model_file)
                if os.path.exists(model_path) and modality not in self._models:
                    try:
                        self._models[modality] = _safe_load(model_path)
                    except Exception:
                        pass

        loaded = list(self._models.keys())
        if loaded:
            print(f"[RuntimeEngine] Loaded modalities: {loaded}")
            if self.use_deep:
                print(f"[RuntimeEngine] Deep Learning {self.strategy_used.upper()} sequence models and Router active for fusion!")
        else:
            print("[RuntimeEngine] WARNING: No models loaded.")

    def _load_deep_artifacts(self):
        """Loads Phase 8 PyTorch sequence models and dynamic router."""
        prefix = "adv_" if self.strategy_used == "adversarial" else "deep_"
        try:
            face_path = os.path.join(EXPERT_MODELS_DIR, f"{prefix}face_expert.pt")
            self.deep_models["face"] = ModalityEncoder(18, 16)
            self.deep_models["face"].load_state_dict(torch.load(face_path, map_location="cpu"))
            self.deep_models["face"].eval()
            reg_key_f = "adv_face_expert" if self.strategy_used == "adversarial" else "face_expert"
            reg_f = self.registry.get_active_model(reg_key_f)
            if reg_f:
                self._verify_hash(face_path, reg_f.get("hash"), f"{prefix}face_expert")

            voice_path = os.path.join(EXPERT_MODELS_DIR, f"{prefix}voice_expert.pt")
            self.deep_models["voice"] = ModalityEncoder(12, 16)
            self.deep_models["voice"].load_state_dict(torch.load(voice_path, map_location="cpu"))
            self.deep_models["voice"].eval()
            reg_key_v = "adv_voice_expert" if self.strategy_used == "adversarial" else "voice_expert"
            reg_v = self.registry.get_active_model(reg_key_v)
            if reg_v:
                self._verify_hash(voice_path, reg_v.get("hash"), f"{prefix}voice_expert")

            physio_path = os.path.join(EXPERT_MODELS_DIR, f"{prefix}physio_expert.pt")
            self.deep_models["physio"] = ModalityEncoder(5, 16)
            self.deep_models["physio"].load_state_dict(torch.load(physio_path, map_location="cpu"))
            self.deep_models["physio"].eval()
            reg_key_p = "adv_physio_expert" if self.strategy_used == "adversarial" else "physio_expert"
            reg_p = self.registry.get_active_model(reg_key_p)
            if reg_p:
                self._verify_hash(physio_path, reg_p.get("hash"), f"{prefix}physio_expert")

            router_path = os.path.join(EXPERT_MODELS_DIR, f"{prefix}fusion_router.pt")
            self.deep_models["router"] = DynamicRouter(num_modalities=3)
            self.deep_models["router"].load_state_dict(torch.load(router_path, map_location="cpu"))
            self.deep_models["router"].eval()
            reg_key_r = "adv_fusion_router" if self.strategy_used == "adversarial" else "deep_fusion_router"
            reg_r = self.registry.get_active_model(reg_key_r)
            if reg_r:
                self._verify_hash(router_path, reg_r.get("hash"), f"{prefix}fusion_router")
        except Exception as exc:
            self.use_deep = False
            print(f"[RuntimeEngine] Failed to load deep learning artifacts, falling back: {exc}")


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
        if modality not in self._models and not (self.use_deep and modality in ["face", "voice", "physio"]):
            return {"error": f"{modality} model not loaded", "modality": modality}
        if raw_features is None:
            return {"error": "raw_features is None", "modality": modality}

        try:
            if self.use_deep and modality in ["face", "voice", "physio"]:
                seq = self._lock_features_deep(modality, raw_features)
                seq_t = torch.FloatTensor(seq)
                with torch.no_grad():
                    logits = self.deep_models[modality](seq_t)
                    prob = float(torch.softmax(logits, dim=1)[0][1].item())
            else:
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
        Missing modalities degrade gracefully. Supports Phase 8 deep dynamic router.
        """
        if self.use_deep:
            if face is None and voice is None and physio is None:
                return {"error": "No valid deep modality predictions — all inputs are None"}

            raw_probs = {}
            masks = [0.0, 0.0, 0.0]
            
            # Face
            if face is not None:
                try:
                    seq_f = self._lock_features_deep("face", face)
                    seq_f_t = torch.FloatTensor(seq_f)
                    with torch.no_grad():
                        logits_f = self.deep_models["face"](seq_f_t)
                        prob_f = float(torch.softmax(logits_f, dim=1)[0][1].item())
                        raw_probs["face"] = prob_f
                        masks[0] = 1.0
                except Exception as exc:
                    print(f"[RuntimeEngine] Deep face prediction failed: {exc}")

            # Voice
            if voice is not None:
                try:
                    seq_v = self._lock_features_deep("voice", voice)
                    seq_v_t = torch.FloatTensor(seq_v)
                    with torch.no_grad():
                        logits_v = self.deep_models["voice"](seq_v_t)
                        prob_v = float(torch.softmax(logits_v, dim=1)[0][1].item())
                        raw_probs["voice"] = prob_v
                        masks[1] = 1.0
                except Exception as exc:
                    print(f"[RuntimeEngine] Deep voice prediction failed: {exc}")

            # Physio
            if physio is not None:
                try:
                    seq_p = self._lock_features_deep("physio", physio)
                    seq_p_t = torch.FloatTensor(seq_p)
                    with torch.no_grad():
                        logits_p = self.deep_models["physio"](seq_p_t)
                        prob_p = float(torch.softmax(logits_p, dim=1)[0][1].item())
                        raw_probs["physio"] = prob_p
                        masks[2] = 1.0
                except Exception as exc:
                    print(f"[RuntimeEngine] Deep physio prediction failed: {exc}")

            if not raw_probs:
                return {"error": "No valid deep modality predictions"}

            # Build 9-dimensional input vector for Dynamic Router
            pf = raw_probs.get("face", 0.5)
            pv = raw_probs.get("voice", 0.5)
            pp = raw_probs.get("physio", 0.5)
            
            cat_in = torch.FloatTensor([[1.0 - pf, pf, 1.0 - pv, pv, 1.0 - pp, pp] + masks])
            
            try:
                with torch.no_grad():
                    raw_weights = self.deep_models["router"](cat_in)
                    w_f = float(raw_weights[0][0].item())
                    w_v = float(raw_weights[0][1].item())
                    w_p = float(raw_weights[0][2].item())
                    
                # Apply mask
                w_f_m = w_f * masks[0]
                w_v_m = w_v * masks[1]
                w_p_m = w_p * masks[2]
                
                # Re-normalize
                sum_w = w_f_m + w_v_m + w_p_m
                if sum_w == 0:
                    sum_w = 1.0
                    
                w_f_norm = w_f_m / sum_w
                w_v_norm = w_v_m / sum_w
                w_p_norm = w_p_m / sum_w
                
                avg_prob = w_f_norm * raw_probs.get("face", 0.0) + \
                           w_v_norm * raw_probs.get("voice", 0.0) + \
                           w_p_norm * raw_probs.get("physio", 0.0)
                           
                fusion_weights = {"face": w_f_norm, "voice": w_v_norm, "physio": w_p_norm}
            except Exception as exc:
                print(f"[RuntimeEngine] Deep router failed: {exc}, falling back to average")
                num_active = sum(masks)
                fallback_w = 1.0 / num_active if num_active > 0 else 0.33
                fusion_weights = {
                    "face": fallback_w * masks[0],
                    "voice": fallback_w * masks[1],
                    "physio": fallback_w * masks[2]
                }
                avg_prob = sum(raw_probs[m] * fusion_weights[m] for m in raw_probs)

            threshold    = 0.6 + (0.5 - sensitivity) * 0.4
            final_pred   = 1 if avg_prob > threshold else 0
            stress_level = "High" if avg_prob > 0.7 else "Moderate" if avg_prob > 0.4 else "Low"

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
                "fusion_weights":        fusion_weights,
                "active_modalities":     list(raw_probs.keys()),
                "explainability":        explanation,
            }

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
        self.reset_calibration()
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
        """
        Pass raw features through FeatureRuntimeLock, apply Phase 4 
        methodology transformations (Calibration & Temporal Windowing), 
        and return the scaled array.
        """
        # 1. Lock features and handle missing values
        if modality == "face":
            feats = self.feature_lock.process_face_features(raw_features, scaler=None)
        elif modality == "voice":
            feats = self.feature_lock.process_voice_features(raw_features, scaler=None)
        elif modality == "physio":
            feats = self.feature_lock.process_physio_features(raw_features, scaler=None)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        feats = feats.flatten()

        # 2. Phase 4: Subject-Aware Calibration
        if self.calibrating[modality]:
            self.feature_history[modality].append(feats)
            if len(self.feature_history[modality]) >= self.calibration_frames:
                self.calibration_baselines[modality] = np.mean(self.feature_history[modality], axis=0)
                self.calibrating[modality] = False
                baseline = self.calibration_baselines[modality]
                self.feature_history[modality] = [] # Reset for rolling window
            else:
                # Use current mean as a temporary baseline while calibrating
                baseline = np.mean(self.feature_history[modality], axis=0)
        else:
            baseline = self.calibration_baselines[modality]

        norm_feats = feats - baseline

        # 3. Phase 4: Temporal Windowing (Rolling Average)
        if not self.calibrating[modality]:
            self.feature_history[modality].append(norm_feats)
            if len(self.feature_history[modality]) > self.window_size:
                self.feature_history[modality].pop(0)
            windowed_feats = np.mean(self.feature_history[modality], axis=0)
        else:
            windowed_feats = norm_feats

        windowed_feats = windowed_feats.reshape(1, -1)

        # 4. Scale with the trained scaler
        scaler = self._scalers.get(modality)
        if scaler is not None:
            windowed_feats = scaler.transform(windowed_feats)
            
        return windowed_feats

    def _lock_features_deep(self, modality: str, raw_features) -> np.ndarray:
        """
        Pass raw features through FeatureRuntimeLock, apply subject-aware calibration baseline
        subtraction, scale frame-wise, and maintain a sequence history of length 5.
        """
        # 1. Lock features and handle missing values
        if modality == "face":
            feats = self.feature_lock.process_face_features(raw_features, scaler=None)
        elif modality == "voice":
            feats = self.feature_lock.process_voice_features(raw_features, scaler=None)
        elif modality == "physio":
            feats = self.feature_lock.process_physio_features(raw_features, scaler=None)
        else:
            raise ValueError(f"Deep learning only supports face, voice, and physio, got: {modality}")

        feats = feats.flatten()

        # 2. Phase 4: Subject-Aware Calibration (Calm baseline subtraction)
        if self.calibrating[modality]:
            self.feature_history[modality].append(feats)
            if len(self.feature_history[modality]) >= self.calibration_frames:
                self.calibration_baselines[modality] = np.mean(self.feature_history[modality], axis=0)
                self.calibrating[modality] = False
                baseline = self.calibration_baselines[modality]
                self.feature_history[modality] = [] # Reset for rolling window
            else:
                baseline = np.mean(self.feature_history[modality], axis=0)
        else:
            baseline = self.calibration_baselines[modality]

        norm_feats = feats - baseline

        # 3. Scale frame-wise using deep scaler
        scaler = self._scalers.get(modality)
        if scaler is not None:
            norm_feats_scaled = scaler.transform(norm_feats.reshape(1, -1))[0]
        else:
            norm_feats_scaled = norm_feats

        # 4. Append to sequence history of size 5
        self.deep_sequence_history[modality].append(norm_feats_scaled)
        if len(self.deep_sequence_history[modality]) > 5:
            self.deep_sequence_history[modality].pop(0)

        # 5. Build sequence of length 5 (pad with oldest frame if less than 5 frames)
        history_len = len(self.deep_sequence_history[modality])
        if history_len < 5:
            pad_size = 5 - history_len
            seq = [self.deep_sequence_history[modality][0]] * pad_size + self.deep_sequence_history[modality]
        else:
            seq = self.deep_sequence_history[modality]

        # Shape (1, 5, FeatDim)
        return np.array(seq).reshape(1, 5, -1)

