"""
explainability_contract.py
Defines the schema, feature labels, and bundle structure for the
Phase 6 Explainability Release Pipeline.

All feature label lists MUST stay in sync with:
  - contracts/feature_contract.yaml  (ordering)
  - backend/core/feature_runtime_lock.py (dim assertions)
"""

BUNDLE_VERSION = "1.0.0"
TOP_K_PER_MODALITY = 6
TOP_K_GLOBAL = 8

# ── Human-readable labels for each feature index ────────────────────────────
# Face: 18 features (must match feature_contract.yaml ordering exactly)
FACE_FEATURE_LABELS = [
    "Left Eye Aspect Ratio",       # 0
    "Right Eye Aspect Ratio",      # 1
    "Avg Eye Aspect Ratio",        # 2
    "Blink Velocity",              # 3
    "Left Brow Descent",           # 4
    "Right Brow Descent",          # 5
    "Brow Asymmetry",              # 6
    "Lip Compression",             # 7
    "Jaw Displacement",            # 8
    "Mouth Corner Pull",           # 9
    "Forehead Tension",            # 10
    "Face Height Norm",            # 11
    "Head Tilt",                   # 12
    "Temporal X Variance",         # 13
    "Temporal Y Variance",         # 14
    "Eye Openness Ratio",          # 15
    "Landmark Confidence",         # 16
    "Nose Wrinkle",                # 17
]

# Voice: 12 features
VOICE_FEATURE_LABELS = [
    "F0 Mean (Pitch Hz)",          # 0
    "F0 Std (Pitch Variation)",    # 1
    "F0 Range",                    # 2
    "Jitter Percent",              # 3
    "Shimmer dB",                  # 4
    "HNR Mean",                    # 5
    "Speaking Rate Proxy",         # 6
    "Voice Intensity",             # 7
    "High Freq Ratio",             # 8
    "Spectral Flux",               # 9
    "Pause Ratio",                 # 10
    "Voiced Fraction",             # 11
]

# Physio: 5 features
PHYSIO_FEATURE_LABELS = [
    "Heart Rate Mean (BPM)",       # 0
    "HRV RMSSD (ms)",              # 1
    "HRV SDNN (ms)",               # 2
    "EDA SCL Mean (µS)",           # 3
    "Respiration Rate (br/min)",   # 4
]

MODALITY_LABELS = {
    "face":   FACE_FEATURE_LABELS,
    "voice":  VOICE_FEATURE_LABELS,
    "physio": PHYSIO_FEATURE_LABELS,
}

# ── Feature group tags for UI colour-coding ──────────────────────────────────
FACE_FEATURE_GROUPS = [
    "eye", "eye", "eye", "eye",
    "brow", "brow", "brow",
    "mouth", "mouth", "mouth",
    "face", "face", "head",
    "temporal", "temporal",
    "eye", "quality", "nose",
]
VOICE_FEATURE_GROUPS = [
    "pitch", "pitch", "pitch",
    "perturbation", "perturbation",
    "quality",
    "rate", "intensity",
    "spectral", "spectral",
    "prosody", "voice_activity",
]
PHYSIO_FEATURE_GROUPS = [
    "heart_rate", "hrv", "hrv",
    "eda", "respiration",
]

MODALITY_GROUPS = {
    "face":   FACE_FEATURE_GROUPS,
    "voice":  VOICE_FEATURE_GROUPS,
    "physio": PHYSIO_FEATURE_GROUPS,
}

# ── Bundle JSON schema ────────────────────────────────────────────────────────
# {
#   "version": "1.0.0",
#   "created_at": "ISO timestamp",
#   "models": {
#     "face":   {"model_file": "...", "feature_labels": [...], "shap_means": [...], "top_features": [...]},
#     "voice":  {...},
#     "physio": {...},
#   }
# }

REQUIRED_BUNDLE_KEYS = {"version", "created_at", "models"}
REQUIRED_MODEL_KEYS  = {"model_file", "feature_labels", "shap_means", "top_features"}
