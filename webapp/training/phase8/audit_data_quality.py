"""
Comprehensive data quality audit for pipeline + enriched training data.
Run before committing to GPU training.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'research', 'pipeline', 'data')
ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')

print("=" * 70)
print("  DATA QUALITY AUDIT")
print("=" * 70)


def print_sep(label):
    print("-" * 70)
    print(f"  {label}")
    print("-" * 70)


def audit_pipeline(dataset_name):
    print_sep(f"PIPELINE: {dataset_name}")
    
    base = os.path.join(PIPELINE_DIR, dataset_name)
    combined_path = os.path.join(base, 'combined_sequences.npy')
    norm_path = os.path.join(base, 'normalized_sequences.npy')
    
    if not os.path.exists(combined_path):
        print(f"  MISSING: combined_sequences.npy")
        return None, None
    
    # Check file size
    fsize = os.path.getsize(combined_path)
    print(f"  File size: {fsize / 1e6:.1f} MB")
    
    # Check for non-NaN data by loading with mmap
    data = np.load(combined_path, mmap_mode='r')
    N, T, C = data.shape
    print(f"  Shape: [{N}, {T}, {C}]  ({N} windows, {T} frames, {C} channels)")
    print(f"  Array memory: {data.nbytes / 1e6:.1f} MB")
    
    issues = []
    
    # Check for all-zero arrays (indicates pipeline failure)
    nz_mask = np.abs(data).sum(axis=(1,2)) > 0
    zero_count = (~nz_mask).sum()
    if zero_count > 0:
        issues.append(f"Zero windows (all channels=0): {zero_count} ({100*zero_count/N:.1f}%)")
    
    # NaN/Inf (sample if large)
    sample = data[:min(10000, N)]
    nan_count = np.isnan(sample).sum()
    inf_count = np.isinf(sample).sum()
    total_el = sample.size
    if nan_count > 0:
        issues.append(f"NaN count (sample): {nan_count} ({100*nan_count/total_el:.4f}%)")
    if inf_count > 0:
        issues.append(f"Inf count (sample): {inf_count} ({100*inf_count/total_el:.4f}%)")
    
    # Per-channel stats
    chan_min = np.nanmin(data, axis=(0,1))
    chan_max = np.nanmax(data, axis=(0,1))
    chan_mean = np.nanmean(data, axis=(0,1))
    chan_std = np.nanstd(data, axis=(0,1))
    
    # Dead channels (std=0 or all same value)
    dead = np.where(chan_std < 1e-10)[0]
    if len(dead) > 0:
        dead_details = []
        for d_idx in dead[:8]:
            v = chan_mean[d_idx]
            dead_details.append(f"{d_idx}(={v:.2f})")
        issues.append(f"Dead channels (std~0): {len(dead)} — {', '.join(dead_details)}")
        if len(dead) > 8:
            issues.append(f"  ... and {len(dead)-8} more")
    
    # Extreme outliers
    for ch in range(C):
        vals = data[:, :, ch].flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        outliers = ((vals < lower) | (vals > upper)).sum()
        outlier_pct = 100 * outliers / len(vals)
        if outlier_pct > 5:
            issues.append(f"Ch{ch}: {outlier_pct:.1f}% outliers (>3*IQR)")
    
    print(f"  Channel stats across {C} ch:  mean={chan_mean.mean():.4f}  std={chan_std.mean():.4f}")
    print(f"    Range: [{np.nanmin(chan_min):.4f}, {np.nanmax(chan_max):.4f}]")
    print(f"    Dead: {len(dead)}  Sample NaN: {nan_count}  Inf: {inf_count}")
    
    if issues:
        print(f"  Issues ({len(issues)}):")
        for iss in issues[:15]:
            print(f"    - {iss}")
    else:
        print(f"  CLEAN: No issues found")
    
    # Normalized check
    if os.path.exists(norm_path):
        norm = np.load(norm_path, mmap_mode='r')
        n_mean = norm.mean()
        n_std = norm.std()
        print(f"  Normalized: mean={n_mean:.4f} std={n_std:.4f}")
        norm_nan = np.isnan(norm[:min(1000, len(norm))]).sum()
        if norm_nan > 0:
            issues.append(f"NaN in normalized: {norm_nan}")
    else:
        print(f"  Normalized: NOT FOUND")
    
    return data, None


def audit_enriched(dataset_name):
    print_sep(f"ENRICHED: {dataset_name}")
    
    d = os.path.join(ENRICHED_DIR, dataset_name)
    if not os.path.isdir(d):
        print(f"  MISSING: {d}")
        return None, None
    
    issues = []
    
    # group_dims.json
    gd_path = os.path.join(d, 'group_dims.json')
    with open(gd_path) as f:
        group_dims = json.load(f)
    total_feats = sum(group_dims.values())
    print(f"  Groups: {len(group_dims)}  Total features: {total_feats}")
    for k, v in group_dims.items():
        print(f"    {k}: {v}")
    
    # Expected group dims
    expected = {
        'face_eye': 9, 'face_mouth': 6, 'face_global_face': 18,
        'voice_spectral_prosody': 8, 'voice_mfcc': 13, 'voice_quality': 2,
        'physio_cardio': 2, 'physio_eda': 3, 'physio_somatic': 8
    }
    for k, exp in expected.items():
        if k not in group_dims:
            issues.append(f"Missing group: {k}")
        elif group_dims[k] != exp:
            issues.append(f"Group {k}: expected dim {exp}, got {group_dims[k]}")
    
    # metadata.parquet
    meta_path = os.path.join(d, 'metadata.parquet')
    meta = pd.read_parquet(meta_path) if os.path.exists(meta_path) else None
    if meta is None:
        issues.append("Missing metadata.parquet")
    else:
        print(f"  Metadata: {len(meta)} rows, {meta['subject_id'].nunique()} subjects")
        
        required_cols = ['subject_id', 'window_index', 'label']
        missing_cols = [c for c in required_cols if c not in meta.columns]
        if missing_cols:
            issues.append(f"Missing metadata columns: {missing_cols}")
        
        # Label balance
        if 'label' in meta.columns:
            label_dist = meta['label'].value_counts()
            pos = label_dist.get(1, 0)
            neg = label_dist.get(0, 0)
            total = pos + neg
            pos_pct = 100 * pos / total if total > 0 else 0
            print(f"  Labels: stress={pos} ({pos_pct:.1f}%), not_stress={neg} ({100-pos_pct:.1f}%)")
            if pos_pct < 5 or pos_pct > 95:
                issues.append(f"Highly imbalanced: {pos_pct:.1f}% stress")
        
        # Duplicate detection
        if 'subject_id' in meta.columns and 'window_index' in meta.columns:
            dupes = meta.duplicated(subset=['subject_id', 'window_index'], keep=False).sum()
            if dupes > 0:
                issues.append(f"Duplicate (subject, window): {dupes} rows")
        
        # Subject distribution
        if 'subject_id' in meta.columns:
            subj_counts = meta['subject_id'].value_counts()
            print(f"  Subject windows: min={subj_counts.min()} max={subj_counts.max()} "
                  f"mean={subj_counts.mean():.0f} median={subj_counts.median():.0f}")
            low_subj = (subj_counts < 10).sum()
            if low_subj > 0:
                issues.append(f"Subjects with <10 windows: {low_subj}")
    
    # sequences.npz (sample-based check)
    npz_path = os.path.join(d, 'sequences.npz')
    if not os.path.exists(npz_path):
        issues.append("Missing sequences.npz")
    else:
        npy_size = os.path.getsize(npz_path)
        loaded = np.load(npz_path, mmap_mode='r')
        print(f"  NPZ keys: {list(loaded.keys())}  (file: {npy_size/1e6:.1f} MB)")
        
        for k in expected:
            if k not in loaded:
                issues.append(f"Missing NPZ key: {k}")
                continue
            arr = loaded[k]
            exp_d = expected[k]
            N, T, D = arr.shape
            if D != exp_d:
                issues.append(f"Group '{k}': expected dim {exp_d}, got {D}")
            if T != 30:
                issues.append(f"Group '{k}': expected T=30, got T={T}")
            # NaN/Inf sample
            s = arr[:min(1000, N)]
            snan = int(np.isnan(s).sum())
            sinf = int(np.isinf(s).sum())
            if snan > 0:
                issues.append(f"Group '{k}': {snan} NaN in sample")
            if sinf > 0:
                issues.append(f"Group '{k}': {sinf} Inf in sample")
            # All-zero check
            nz = (np.abs(s).sum(axis=(1,2)) > 0).sum()
            if nz == 0 and N > 0:
                issues.append(f"Group '{k}': ALL ZERO in sample")
    
    if issues:
        print(f"  Issues ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  CLEAN: No issues found")
    
    return meta, loaded if 'loaded' in dir() else None


def check_consistency():
    print_sep("CROSS-DATASET CONSISTENCY")
    
    issues = []
    datasets = ['stressid', 'wesad', 'empathicschool']
    
    metas = {}
    for ds in datasets:
        p = os.path.join(ENRICHED_DIR, ds, 'metadata.parquet')
        if os.path.exists(p):
            metas[ds] = pd.read_parquet(p)
            m = metas[ds]
            print(f"  {ds}: {len(m)} windows, {m['subject_id'].nunique()} subjects, "
                  f"stress={m['label'].sum() if 'label' in m.columns else '?'}")
    
    # Subject ID overlap check
    for ds1 in datasets:
        for ds2 in datasets:
            if ds1 >= ds2 or ds1 not in metas or ds2 not in metas:
                continue
            overlap = set(metas[ds1]['subject_id'].unique()) & set(metas[ds2]['subject_id'].unique())
            if overlap:
                issues.append(f"Subject ID overlap {ds1} vs {ds2}: {overlap}")
    
    # Combined check
    comb_path = os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')
    if os.path.exists(comb_path):
        comb_meta = pd.read_parquet(comb_path)
        print(f"  combined: {len(comb_meta)} windows, {comb_meta['subject_id'].nunique()} subjects, "
              f"stress={comb_meta['label'].sum() if 'label' in comb_meta.columns else '?'}")
        if all(ds in metas for ds in datasets):
            total = sum(len(metas[ds]) for ds in datasets)
            diff = abs(len(comb_meta) - total)
            if diff > 0:
                issues.append(f"Combined window diff: combined={len(comb_meta)} vs sum={total} (diff={diff})")
            else:
                print(f"  Window count matches sum of parts: {total}")
    
    if issues:
        print(f"  Issues ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  CLEAN: No cross-dataset issues")


def check_pipeline_labels():
    """Check pipeline labels if available (from subject_list files)."""
    print_sep("PIPELINE LABEL VALIDATION")
    
    # Check StressID
    sid_path = os.path.join(PIPELINE_DIR, 'stressid', 'combined_sequences.npy')
    if os.path.exists(sid_path):
        data = np.load(sid_path, mmap_mode='r')
        print(f"  StressID pipeline: {data.shape[0]} windows")
        
        # Check if labels exist alongside sequences (in directory)
        label_file = os.path.join(PIPELINE_DIR, 'stressid', 'labels.npy')
        if os.path.exists(label_file):
            labels = np.load(label_file)
            print(f"  Labels found: {labels.shape}, stress={labels.sum()}/{len(labels)}")
        else:
            print(f"  No standalone labels.npy (labels in metadata only)")
    
    for ds in ['wesad', 'empathicschool']:
        path = os.path.join(PIPELINE_DIR, ds, 'combined_sequences.npy')
        if os.path.exists(path):
            data = np.load(path, mmap_mode='r')
            print(f"  {ds}: {data.shape[0]} windows")
            
            # Check for any subject mapping files
            subj_file = os.path.join(PIPELINE_DIR, ds, 'subject_mapping.json')
            if os.path.exists(subj_file):
                with open(subj_file) as f:
                    sm = json.load(f)
                print(f"  Subject mapping: {len(sm)} subjects")


if __name__ == '__main__':
    print("\n[1] PIPELINE RAW DATA AUDIT")
    for ds in ['stressid', 'wesad', 'empathicschool']:
        audit_pipeline(ds)
        print()
    
    print("\n[2] ENRICHED TRAINING DATA AUDIT")
    for ds in ['stressid', 'wesad', 'empathicschool', 'combined']:
        audit_enriched(ds)
        print()
    
    print("\n[3] CROSS-DATASET CONSISTENCY")
    check_consistency()
    
    print("\n[4] PIPELINE LABEL CHECK")
    check_pipeline_labels()
    
    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)
