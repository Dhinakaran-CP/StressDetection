"""
Build enriched training data from pipeline sequences for all 3 datasets.

The pipeline extractors produce [N, 30, 72] numpy sequences per dataset.
72 channels = 34 face + 24 voice + 14 physio after privacy exclusions.

This script:
  1. Reads pipeline sequences for StressID, WESAD, EmpathicSchool
  2. Maps ALL 72 channels to expanded sub-modality groups (69 features after 3 exclusions)
  3. Creates enriched Parquet + numpy training files compatible with the SSVB pipeline
  4. Handles missing modalities (WESAD=physio-only, EmpathicSchool=face+physio)

Usage:
    python build_enriched_training_data.py
"""
import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'webapp'))

PIPELINE_DATA  = os.path.join(PROJECT_ROOT, 'research', 'pipeline', 'data')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 72-channel to expanded sub-modality mapping
# ---------------------------------------------------------------------------
# Pipeline sequence shape: [N, 30, 72]
#   Face  (34): indices  0-33
#   Voice (24): indices 34-57
#   Physio(14): indices 58-71
#
# We exclude 3 privacy-risky features:
#   face[11] = face_height_norm, voice[36]=f0_mean, physio[61]=eda_tonic/SCL

CHANNEL_GROUPS = {
    "face": {
        "eye":         [0, 1, 2, 3, 4,   18, 19, 20, 32],
        "mouth":       [8, 9, 10,         24, 25, 26],
        "global_face": [5, 6, 7, 12, 13, 14, 15, 16, 17,
                        21, 22, 23, 27, 28, 29, 30, 31, 33],
    },
    "voice": {
        "spectral_prosody": [34, 35, 37, 51, 52, 53, 54, 55],
        "mfcc":             list(range(38, 51)),
        "quality":          [56, 57],
    },
    "physio": {
        "cardio":   [58, 59],
        "eda":      [60, 62, 63],
        "somatic":  [64, 65, 66, 67, 68, 69, 70, 71],
    },
}

CHANNEL_NAMES = {
    "face": {
        0: "left_ear", 1: "right_ear", 2: "avg_ear", 3: "blink_velocity",
        4: "eye_openness_ratio", 5: "brow_descent_left", 6: "brow_descent_right",
        7: "brow_asymmetry", 8: "lip_compression", 9: "jaw_tension",
        10: "mouth_corner_pull", 11: "face_height_norm_EXCLUDED",
        12: "forehead_tension", 13: "head_tilt", 14: "pitch", 15: "yaw",
        16: "roll", 17: "nose_wrinkle",
        18: "d_left_ear", 19: "d_right_ear", 20: "d_blink_velocity",
        21: "d_brow_descent_left", 22: "d_brow_descent_right",
        23: "d_brow_asymmetry", 24: "d_lip_compression", 25: "d_jaw_tension",
        26: "d_mouth_corner_pull", 27: "d_forehead_tension", 28: "d_head_tilt",
        29: "d_pitch", 30: "d_yaw", 31: "d_roll",
        32: "d_eye_openness_ratio", 33: "d_nose_wrinkle",
    },
    "voice": {
        34: "rms", 35: "zcr", 36: "f0_mean_EXCLUDED", 37: "f0_std",
        **{38+i: f"mfcc_{i+1}" for i in range(13)},
        51: "spectral_centroid", 52: "spectral_bandwidth",
        53: "spectral_rolloff", 54: "spectral_flatness", 55: "chroma_stft",
        56: "hnr", 57: "jitter",
    },
    "physio": {
        58: "ecg_hr", 59: "ecg_hrv_rmssd",
        60: "eda_clean", 61: "eda_tonic_scl_EXCLUDED", 62: "eda_phasic",
        63: "eda_scr_count", 64: "resp_rate", 65: "resp_amplitude",
        66: "temp_mean", 67: "temp_std",
        68: "acc_x", 69: "acc_y", 70: "acc_z", 71: "acc_mag",
    },
}

# Features to exclude (privacy / identity-leaking)
EXCLUDED_CHANNELS = {11, 36, 61}

# Outlier clipping: channels with values > 3*IQR from median are suspect
OUTLIER_THRESHOLD = 1e6  # absolute threshold for extreme artifacts
WINSOR_PERCENTILE = 99.9  # per-channel winsorization percentile

# ---------------------------------------------------------------------------
# Sub-modality dimension summary for model config
# ---------------------------------------------------------------------------
# face:   eye=9, mouth=6, global_face=18   → 33
# voice:  spectral_prosody=8, mfcc=13, quality=2  → 23
# physio: cardio=2, eda=3, somatic=8       → 13
# Total: 69 features (was 30)

def get_group_dims():
    dims = {}
    for mod, groups in CHANNEL_GROUPS.items():
        for gname, channels in groups.items():
            key = f"{mod}_{gname}"
            dims[key] = len(channels)
    return dims


def clip_outliers(seqs, dataset_name, threshold=OUTLIER_THRESHOLD, percentile=WINSOR_PERCENTILE):
    """Winsorize extreme outliers per channel."""
    N, T, C = seqs.shape
    clipped = 0
    total = N * T
    for ch in range(C):
        vals = seqs[:, :, ch]
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        # Absolute threshold clip (extreme artifacts like 1.6B)
        mask_extreme = np.abs(vals) > threshold
        n_extreme = mask_extreme.sum()
        if n_extreme > 0:
            clipped += n_extreme
            vals[mask_extreme] = np.nan  # mark for imputation
        # Per-channel percentile winsorization
        # (skip if already clean)
        finite_clean = vals[np.isfinite(vals)]
        if len(finite_clean) < 10:
            continue
        upper = np.percentile(finite_clean, percentile)
        lower = np.percentile(finite_clean, 100 - percentile)
        mask_upper = (vals > upper) & np.isfinite(vals)
        mask_lower = (vals < lower) & np.isfinite(vals)
        n_winsor = mask_upper.sum() + mask_lower.sum()
        if n_winsor > 0:
            clipped += n_winsor
            vals[mask_upper] = upper
            vals[mask_lower] = lower
    if clipped > 0:
        print(f"  {dataset_name}: clipped {clipped} outliers ({100*clipped/total:.2f}% of values)")
    return seqs


def impute_nan(seqs):
    """Replace NaN with 0 in-place (simplest imputation for missing modalities)."""
    n_nan = np.isnan(seqs).sum()
    if n_nan > 0:
        np.nan_to_num(seqs, nan=0.0, copy=False)
        print(f"  Imputed {n_nan} NaN values -> 0")
    return seqs


def load_pipeline_sequences(dataset_name):
    """Load [N, 30, 72] sequences + metadata for a dataset."""
    data_dir = os.path.join(PIPELINE_DATA, dataset_name)
    seq_path = os.path.join(data_dir, "combined_sequences.npy")
    if not os.path.exists(seq_path):
        seq_path = os.path.join(data_dir, "normalized_sequences.npy")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"No sequences found: {data_dir}")

    seqs = np.load(seq_path).astype(np.float32)  # [N, 30, 72]
    print(f"  {dataset_name}: loaded sequences {seqs.shape}")

    if dataset_name != 'stressid':
        # Clip and impute for non-StressID datasets (StressID is clean)
        seqs = clip_outliers(seqs, dataset_name)
    seqs = impute_nan(seqs)

    meta = None
    meta_path = os.path.join(data_dir, "combined_windows.parquet")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "normalized_windows.parquet")
    if os.path.exists(meta_path):
        meta = pd.read_parquet(meta_path)
        print(f"  {dataset_name}: loaded metadata {len(meta)} rows")

    return seqs, meta


def build_enriched_dataset(seqs, meta, dataset_name):
    """Convert [N, 30, 72] sequences to expanded sub-modality format.

    Returns:
        features: dict of {group_name: np.array [N, 30, feat_dim]}
        labels: np.array [N]
        subjects: list [N]
    """
    N, T, C = seqs.shape
    assert C == 72, f"Expected 72 channels, got {C}"

    # Extract sub-modality features
    features = {}
    for mod, groups in CHANNEL_GROUPS.items():
        for gname, channel_idxs in groups.items():
            valid = [c for c in channel_idxs if c not in EXCLUDED_CHANNELS]
            key = f"{mod}_{gname}"
            features[key] = seqs[:, :, valid]  # [N, 30, feat_dim]
            print(f"    {key:25s}  dim={len(valid):2d}  channels={valid}")

    # Labels
    labels = np.zeros(N, dtype=np.int64)
    subjects = np.full(N, "unknown", dtype=object)
    tasks = np.full(N, "task_0", dtype=object)
    windows = np.arange(N)

    if meta is not None:
        # Map metadata column names
        rename = {"binary_stress": "label", "task_name": "task_id",
                  "window_id": "window_index"}
        meta = meta.rename(columns=rename)
        for col in ["subject_id", "task_id", "window_index", "label"]:
            if col not in meta.columns:
                meta[col] = 0 if col in ("label", "window_index") else "unknown"

        meta = meta.iloc[:N]  # align in case of mismatch
        for col in ["subject_id", "task_id"]:
            meta[col] = meta[col].astype(str).str.lower().str.strip()

        labels = meta["label"].values.astype(np.int64)
        subjects = meta["subject_id"].values
        tasks = meta["task_id"].values
        windows = meta["window_index"].values

    return features, labels, subjects, tasks, windows


def save_enriched_dataset(dataset_name, features, labels, subjects,
                          tasks, windows):
    """Save enriched training data as Parquet + numpy."""
    ds_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    # Save features as numpy arrays (one per sub-modality group)
    np.savez(os.path.join(ds_dir, "sequences.npz"), **features)

    # Prefix subject IDs with dataset name to avoid cross-dataset overlap
    prefixed_subjects = np.array([
        f"{dataset_name}_{s}" if not str(s).startswith(dataset_name) else str(s)
        for s in subjects
    ], dtype=object)
    unique_before = len(np.unique(subjects))
    unique_after = len(np.unique(prefixed_subjects))
    if unique_after > unique_before:
        print(f"  Subject IDs: {unique_before} -> {unique_after} (prefix applied)")

    # Save metadata
    meta = pd.DataFrame({
        "subject_id": prefixed_subjects,
        "task_id": tasks,
        "window_index": windows,
        "label": labels,
        "dataset": dataset_name,
    })
    meta.to_parquet(os.path.join(ds_dir, "metadata.parquet"), index=False)

    # Save group dimensions for model config
    dims = {k: v.shape[-1] for k, v in features.items()}
    with open(os.path.join(ds_dir, "group_dims.json"), "w") as f:
        json.dump(dims, f, indent=2)

    N = len(labels)
    n_stress = int(labels.sum())
    print(f"  Saved: {N} windows, {n_stress} stress ({n_stress/max(N,1)*100:.1f}%), "
          f"{len(np.unique(subjects))} subjects")
    print(f"  Groups: {json.dumps(dims)}")
    return dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["stressid", "wesad", "empathicschool"],
                        help="Datasets to process")
    args = parser.parse_args()

    print("=" * 60)
    print("  Enriched Training Data Builder")
    print("=" * 60)
    print(f"  Pipeline data: {PIPELINE_DATA}")
    print(f"  Output:        {OUTPUT_DIR}")

    unified_features = {}
    unified_meta = []

    for ds_name in args.datasets:
        print(f"\n{'-'*60}")
        print(f"  Processing: {ds_name}")
        try:
            seqs, meta = load_pipeline_sequences(ds_name)
            features, labels, subjects, tasks, windows = build_enriched_dataset(
                seqs, meta, ds_name)
            dims = save_enriched_dataset(ds_name, features, labels,
                                         subjects, tasks, windows)

            # Collect for unified combined dataset
            for key, arr in features.items():
                if key not in unified_features:
                    unified_features[key] = []
                unified_features[key].append(arr)

            # Prefix subject IDs for combined dataset
            N = len(labels)
            prefixed = [f"{ds_name}_{s}" for s in subjects[:N]]
            unified_meta.append(pd.DataFrame({
                "subject_id": prefixed,
                "task_id": tasks[:N],
                "window_index": windows[:N],
                "label": labels[:N],
                "dataset": ds_name,
            }))

        except Exception as e:
            print(f"  SKIP: {e}")
            import traceback
            traceback.print_exc()

    # Build combined dataset
    if len(unified_meta) > 1:
        print(f"\n{'-'*60}")
        print("  Building Combined dataset...")
        combined_dir = os.path.join(OUTPUT_DIR, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        combined_features = {}
        for key in unified_features:
            arrays = [a for a in unified_features[key] if a is not None]
            if arrays:
                combined_features[key] = np.concatenate(arrays, axis=0)

        np.savez(os.path.join(combined_dir, "sequences.npz"),
                 **combined_features)

        combined_meta = pd.concat(unified_meta, ignore_index=True)
        combined_meta.to_parquet(os.path.join(combined_dir, "metadata.parquet"),
                                 index=False)

        dims = {k: v.shape[-1] for k, v in combined_features.items()}
        with open(os.path.join(combined_dir, "group_dims.json"), "w") as f:
            json.dump(dims, f, indent=2)

        N = len(combined_meta)
        n_stress = int(combined_meta["label"].sum())
        print(f"  Combined: {N} windows, {n_stress} stress "
              f"({n_stress/max(N,1)*100:.1f}%), "
              f"{combined_meta['subject_id'].nunique()} subjects")
        print(f"  Groups: {json.dumps(dims)}")

    # Print model configuration
    print(f"\n{'='*60}")
    print("  Model Configuration for Updated SSVBCASA_AIS")
    print("=" * 60)
    print("  10 sub-experts with input dimensions:")
    for key, dim in sorted(get_group_dims().items()):
        print(f"    {key:25s}  input_dim={dim}")
    print(f"\n  Total features: {sum(get_group_dims().values())}")
    print(f"  Excluded (privacy): {sorted(EXCLUDED_CHANNELS)}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  To train: python train_ssvb_production.py --enriched")


if __name__ == "__main__":
    main()
