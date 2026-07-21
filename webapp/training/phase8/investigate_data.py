"""
Deep-dive investigation into critical audit findings.
"""
import os, json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'research', 'pipeline', 'data')
ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')

# 72-channel mapping from extractors
CH_NAMES = [
    # Face (0-35): eye_ar_0..11, mouth_ar_0..5, gface_0..17
    *(f'eye_ar_{i}' for i in range(12)),
    *(f'mouth_ar_{i}' for i in range(6)),
    *(f'gface_{i}' for i in range(18)),
    # Voice (36-59)
    'pitch','f0_mean','f0_std','f0_min','f0_max','f0_range',
    'voice_prob','loudness','loudness_std','hNR','jitter','shimmer',
    *(f'mfcc_{i}' for i in range(13)),
    'spectral_centroid','spectral_bandwidth','spectral_contrast',
    'spectral_flatness','spectral_rolloff','zcr',
    # Physio (60-71)
    'eda_phasic','eda_tonic_scl','scr','hr','hrv',
    'resp_amplitude','temp_mean','temp_std',
    'acc_x','acc_y','acc_z','acc_mag',
]
# Channel count is 72 per pipeline

FACE_CH = list(range(36))     # 0-35
VOICE_CH = list(range(36, 60)) # 36-59
PHYSIO_CH = list(range(60, 72)) # 60-71

def ch_type(idx):
    if idx in FACE_CH: return 'face'
    if idx in VOICE_CH: return 'voice'
    return 'physio'

def investigate_pipeline(ds_name):
    print(f"\n{'='*70}")
    print(f"  INVESTIGATION: {ds_name}")
    print(f"{'='*70}")
    
    path = os.path.join(PIPELINE_DIR, ds_name, 'combined_sequences.npy')
    norm_path = os.path.join(PIPELINE_DIR, ds_name, 'normalized_sequences.npy')
    
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    
    data = np.load(path, mmap_mode='r')
    N, T, C = data.shape
    
    # ---- 1. Per-modality NaN analysis ----
    print(f"\n  --- Modality Coverage (NaN analysis) ---")
    for label, chs in [('Face', FACE_CH), ('Voice', VOICE_CH), ('Physio', PHYSIO_CH)]:
        sl = data[:, :, chs]
        n_total = sl.size
        n_nan = np.isnan(sl).sum()
        n_zero = (np.abs(sl).sum(axis=(1,2)) == 0).sum()
        n_finite = n_total - n_nan
        print(f"  {label:6s} ch{chs[0]:2d}-{chs[-1]:2d}: {n_nan/n_total*100:6.2f}% NaN  "
              f"{n_zero/N*100:5.1f}% zero-wins  "
              f"finite={n_finite:>8d}/{n_total:<8d} ({100*n_finite/n_total:4.1f}%)")
    
    # ---- 2. Non-NaN value range check ----
    print(f"\n  --- Non-NaN Value Ranges ---")
    for label, chs in [('Face', FACE_CH), ('Voice', VOICE_CH), ('Physio', PHYSIO_CH)]:
        sl = data[:, :, chs]
        finite = sl[np.isfinite(sl)]
        if len(finite) == 0:
            print(f"  {label:6s}: ALL NaN")
        else:
            print(f"  {label:6s}: mean={finite.mean():.4f} std={finite.std():.4f}  "
                  f"[{finite.min():.4f}, {finite.max():.4f}]  n={len(finite)}")
    
    # ---- 3. Check specific extreme channels ----
    print(f"\n  --- Extreme Value Investigation ---")
    for ch in range(C):
        vals = data[:, :, ch]
        fvals = vals[np.isfinite(vals)]
        if len(fvals) == 0:
            continue
        if np.abs(fvals).max() > 1e6:
            name = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
            print(f"  Ch{ch:2d} ({name:20s} {ch_type(ch):6s}): "
                  f"max={fvals.max():.2e} min={fvals.min():.2e}")
    
    # ---- 4. Check normalized file ----
    if os.path.exists(norm_path):
        ndata = np.load(norm_path, mmap_mode='r')
        nn = ndata.shape[0]
        # Per-modality NaN in normalized
        print(f"\n  --- Normalized Coverage ---")
        for label, chs in [('Face', FACE_CH), ('Voice', VOICE_CH), ('Physio', PHYSIO_CH)]:
            sl = ndata[:, :, chs]
            n_nan = np.isnan(sl).sum()
            n_finite = sl.size - n_nan
            print(f"  {label}: {n_nan/sl.size*100:.1f}% NaN  "
                  f"finite={n_finite} / {sl.size}")
    
    return data


def investigate_enriched(ds_name):
    print(f"\n  --- Enriched Data Investigation ---")
    d = os.path.join(ENRICHED_DIR, ds_name)
    if not os.path.isdir(d):
        print(f"  MISSING: {d}")
        return
    
    meta = pd.read_parquet(os.path.join(d, 'metadata.parquet'))
    loaded = np.load(os.path.join(d, 'sequences.npz'), mmap_mode='r')
    
    # ---- Per-group NaN rates ----
    print(f"  Group NaN rates (sample first 100 samples x 30 frames):")
    for k in sorted(loaded.keys()):
        arr = loaded[k]
        s = arr[:min(1000, len(arr))]
        total = s.size
        nan = np.isnan(s).sum()
        inf = np.isinf(s).sum()
        z = (np.abs(s).sum(axis=(1,2)) == 0).sum()
        print(f"    {k:30s}: nan={nan/total*100:6.2f}%  inf={inf}  zero-win={z}/{len(s)}")
    
    # ---- Subject distribution ----
    print(f"\n  Subject distribution:")
    counts = meta['subject_id'].value_counts()
    print(f"    Total: {len(meta)} windows, {meta['subject_id'].nunique()} subjects")
    print(f"    Top 5: {counts.head(5).to_dict()}")
    print(f"    Bottom 5: {counts.tail(5).to_dict()}")


def check_subject_overlap():
    print(f"\n{'='*70}")
    print(f"  SUBJECT ID OVERLAP INVESTIGATION")
    print(f"{'='*70}")
    
    datasets = ['stressid', 'wesad', 'empathicschool']
    metas = {}
    for ds in datasets:
        p = os.path.join(ENRICHED_DIR, ds, 'metadata.parquet')
        if os.path.exists(p):
            metas[ds] = pd.read_parquet(p)
    
    # Overlap between WESAD and EmpathicSchool
    wes = set(metas['wesad']['subject_id'].unique())
    emp = set(metas['empathicschool']['subject_id'].unique())
    common = wes & emp
    print(f"\n  WESAD subjects: {sorted(wes)}")
    print(f"  EmpathicSchool subjects: {sorted(emp)}")
    print(f"  Overlapping IDs ({len(common)}): {sorted(common)}")
    
    # Same with StressID
    sid = set(metas['stressid']['subject_id'].unique())
    print(f"\n  StressID subjects: {sorted(sid)}")
    for other_name, other_set in [('WESAD', wes), ('EmpathicSchool', emp)]:
        overlap = sid & other_set
        if overlap:
            print(f"  Overlap StressID vs {other_name}: {sorted(overlap)}")
        else:
            print(f"  Overlap StressID vs {other_name}: None (good)")
    
    # Check label distributions for overlapping subjects
    print(f"\n  Comparing overlapping subject labels across datasets:")
    for subj in sorted(common):
        w_labels = metas['wesad'][metas['wesad']['subject_id']==subj]['label']
        e_labels = metas['empathicschool'][metas['empathicschool']['subject_id']==subj]['label']
        print(f"    {subj}: WESAD({len(w_labels)}w, {w_labels.sum()}s)  vs  "
              f"EmpathicSchool({len(e_labels)}w, {e_labels.sum()}s)")


if __name__ == '__main__':
    for ds in ['stressid', 'wesad', 'empathicschool']:
        investigate_pipeline(ds)
        investigate_enriched(ds)
    
    check_subject_overlap()
