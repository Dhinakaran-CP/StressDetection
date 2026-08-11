"""
Subject-Level Diagnostic Analysis for Stress Detection Pipeline.

Analyzes flagged subjects:
  - stressid_71i5  (AUC ~0.51)
  - stressid_m8g5  (AUC ~0.09 — suspicious)
  - wesad_s2       (AUC ~0.69, F1 ~0.31)
"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase3_production'))

from phase3_production.train import SSVBDataset

ENRICHED_DIR = str(PROJECT_ROOT / 'data' / 'enriched_training_data')
CERTIFIED_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'certified_data')

print("=" * 70)
print("  SUBJECT-LEVEL DIAGNOSTIC ANALYSIS")
print("=" * 70)

# ── Load enriched data for each dataset ──────────────────────────────
datasets_info = {}
for ds_name in ['stressid', 'wesad', 'combined']:
    meta_path = os.path.join(ENRICHED_DIR, ds_name, 'metadata.parquet')
    npz_path = os.path.join(ENRICHED_DIR, ds_name, 'sequences.npz')
    if not os.path.exists(meta_path):
        print(f"  {ds_name}: metadata not found, skip")
        continue
    meta = pd.read_parquet(meta_path)
    feats = np.load(npz_path)
    datasets_info[ds_name] = {'meta': meta, 'features': feats}
    print(f"\n  {ds_name}: {len(meta)} windows, {meta['subject_id'].nunique()} subjects")

# ── Load certified CSVs for label verification ──────────────────────
print(f"\n{'─' * 70}")
print("  LOADING CERTIFIED CSVs (for label verification)")
print(f"{'─' * 70}")

certified = {}
for mod in ['face', 'voice', 'physio']:
    path = os.path.join(CERTIFIED_DIR, f'{mod}_certified.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in ['subject_id', 'task_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        certified[mod] = df
        print(f"  {mod}_certified.csv: {len(df)} rows")
    else:
        print(f"  {mod}_certified.csv: NOT FOUND")

# Merge certified CSVs into one aligned dataframe
if certified:
    common_keys = ['subject_id', 'task_id', 'window_index', 'label']
    df_cert = None
    for mod, df in certified.items():
        if df_cert is None:
            df_cert = df.copy()
        else:
            df_cert = df_cert.merge(df, on=[k for k in common_keys if k in df.columns],
                                    how='outer', suffixes=('', f'_{mod}'))
    df_cert.sort_values(['subject_id', 'task_id', 'window_index'], inplace=True)
    df_cert.reset_index(drop=True, inplace=True)
    df_cert.fillna(0, inplace=True)
    print(f"\n  Merged certified: {len(df_cert)} rows, {df_cert['subject_id'].nunique()} subjects")
else:
    df_cert = None

# ── Compute per-subject statistics ──────────────────────────────────
def compute_subject_stats(ds_name, subject_id, meta, feats):
    """Compute diagnostic stats for a single subject."""
    subj_mask = meta['subject_id'] == subject_id
    n_windows = subj_mask.sum()
    if n_windows == 0:
        return None

    subj_meta = meta[subj_mask]
    labels = subj_meta['label'].values
    n_stress = (labels == 1).sum()
    n_calm = (labels == 0).sum()

    # Feature stats
    feats_dict = {}
    for key in sorted(feats.keys()):
        arr = feats[key][subj_mask.values]
        feats_dict[key] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'zero_frac': float((arr == 0).mean()),
        }
    
    # Overall feature norm
    all_feats = np.concatenate([feats[k][subj_mask.values] for k in sorted(feats.keys())], axis=-1)
    overall_mean = float(all_feats.mean())
    overall_std = float(all_feats.std())
    overall_zero_frac = float((all_feats == 0).mean())

    # Per-task label breakdown (if available)
    task_labels = {}
    if 'task_id' in subj_meta.columns:
        for task_id, grp in subj_meta.groupby('task_id'):
            task_labels[str(task_id)] = {
                'n': len(grp),
                'stress_frac': float(grp['label'].mean()),
            }

    # Window indices
    window_indices = sorted(subj_meta['window_index'].unique()) if 'window_index' in subj_meta.columns else []

    return {
        'dataset': ds_name,
        'subject': str(subject_id),
        'n_windows': int(n_windows),
        'n_calm': int(n_calm),
        'n_stress': int(n_stress),
        'stress_ratio': float(n_stress / max(n_windows, 1)),
        'feature_mean': overall_mean,
        'feature_std': overall_std,
        'feature_zero_frac': overall_zero_frac,
        'per_group': feats_dict,
        'per_task': task_labels,
        'window_indices': window_indices[:20],  # first 20
        'window_indices_count': len(window_indices),
    }


# ── Analyze flagged subjects ────────────────────────────────────────
flagged = ['stressid_71i5', 'stressid_m8g5', 'wesad_s2']
all_reports = {}

for ds_name, info in datasets_info.items():
    meta = info['meta']
    feats = info['features']
    for subj_id in flagged:
        subj_short = subj_id.split('_')[-1] if '_' in subj_id else subj_id
        if subj_id in meta['subject_id'].values or subj_short in meta['subject_id'].values:
            actual_id = subj_id if subj_id in meta['subject_id'].values else subj_short
            report = compute_subject_stats(ds_name, actual_id, meta, feats)
            if report:
                key = f"{ds_name}::{actual_id}"
                all_reports[key] = report

# ── Certified CSV label audit ────────────────────────────────────────
print(f"\n{'─' * 70}")
print("  CERTIFIED CSV LABEL AUDIT (flagged subjects)")
print(f"{'─' * 70}")

if df_cert is not None:
    for subj_id in flagged:
        subj_short = subj_id.split('_')[-1]
        subj_df = df_cert[df_cert['subject_id'] == subj_short] if subj_short in df_cert['subject_id'].values else \
                 df_cert[df_cert['subject_id'] == subj_id]
        if len(subj_df) == 0:
            print(f"\n  {subj_id}: NOT FOUND in certified CSV")
            continue
        print(f"\n  {subj_id}: {len(subj_df)} rows in certified CSV")
        print(f"    Label distribution: {subj_df['label'].value_counts().to_dict()}")
        if 'task_id' in subj_df.columns:
            print(f"    Per-task label means:")
            for task_id, grp in subj_df.groupby('task_id'):
                print(f"      task={task_id}: n={len(grp)}, stress_frac={grp['label'].mean():.3f}")
        if 'window_index' in subj_df.columns:
            wi = sorted(subj_df['window_index'].unique())
            print(f"    Window indices: {wi[:15]}{'...' if len(wi) > 15 else ''}")
            # Check for window index gaps
            expected = list(range(min(wi), max(wi) + 1))
            missing = sorted(set(expected) - set(wi))
            if missing:
                print(f"    WARNING: Missing window indices: {missing}")

# ── Compute dataset-wide statistics for comparison ─────────────────
print(f"\n{'─' * 70}")
print("  DATASET-WIDE STATISTICS (for comparison)")
print(f"{'─' * 70}")

dataset_stats = {}
for ds_name, info in datasets_info.items():
    meta = info['meta']
    feats = info['features']
    all_feats = np.concatenate([feats[k] for k in sorted(feats.keys())], axis=-1)
    stats = {
        'n_subjects': int(meta['subject_id'].nunique()),
        'n_windows': len(meta),
        'stress_ratio': float(meta['label'].mean()),
        'feature_mean': float(all_feats.mean()),
        'feature_std': float(all_feats.std()),
        'feature_zero_frac': float((all_feats == 0).mean()),
    }
    dataset_stats[ds_name] = stats
    print(f"  {ds_name}:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

# ── Anomaly detection ───────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("  ANOMALY DETECTION (comparing flagged subjects to dataset norms)")
print(f"{'─' * 70}")

anomalies = []
for key, report in all_reports.items():
    ds_name = report['dataset']
    ds_stats = dataset_stats.get(ds_name, {})
    
    subject = report['subject']
    
    # 1. Feature norm deviation
    global_feat_mean = ds_stats.get('feature_mean', 0)
    global_feat_std = ds_stats.get('feature_std', 1)
    z_score = (report['feature_mean'] - global_feat_mean) / max(global_feat_std, 1e-8)
    
    # 2. Zero fraction
    zero_dev = report['feature_zero_frac'] - ds_stats.get('feature_zero_frac', 0)
    
    # 3. Stress ratio deviation
    stress_dev = abs(report['stress_ratio'] - ds_stats.get('stress_ratio', 0.5))
    
    # 4. Window count anomaly
    subj_counts = datasets_info[ds_name]['meta'].groupby('subject_id').size()
    mean_windows = subj_counts.mean()
    std_windows = subj_counts.std()
    window_z = (report['n_windows'] - mean_windows) / max(std_windows, 1)
    
    flag_reasons = []
    if abs(z_score) > 2.0:
        flag_reasons.append(f"feature_mean z-score={z_score:.2f} (>2.0)")
    if zero_dev > 0.1:
        flag_reasons.append(f"zero_fraction={report['feature_zero_frac']:.3f} vs dataset={ds_stats.get('feature_zero_frac', 0):.3f}")
    if stress_dev > 0.2:
        flag_reasons.append(f"stress_ratio={report['stress_ratio']:.3f} deviates from dataset {ds_stats.get('stress_ratio', 0.5):.3f}")
    if abs(window_z) > 2.0:
        flag_reasons.append(f"n_windows z-score={window_z:.2f}")
    
    anomalies.append({
        'subject': subject,
        'dataset': ds_name,
        'reasons': flag_reasons,
        'feature_mean': report['feature_mean'],
        'feature_std': report['feature_std'],
        'feature_zero_frac': report['feature_zero_frac'],
        'stress_ratio': report['stress_ratio'],
        'n_windows': report['n_windows'],
    })
    
    print(f"\n  {subject} (from {ds_name}):")
    print(f"    Feature mean: {report['feature_mean']:.6f} (dataset: {global_feat_mean:.6f}, z={z_score:.2f})")
    print(f"    Feature std:  {report['feature_std']:.6f}")
    print(f"    Zero frac:    {report['feature_zero_frac']:.3f} (dataset: {ds_stats.get('feature_zero_frac', 0):.3f})")
    print(f"    Stress ratio: {report['stress_ratio']:.3f} (dataset: {ds_stats.get('stress_ratio', 0.5):.3f})")
    print(f"    Windows:      {report['n_windows']} (dataset avg: {mean_windows:.0f})")
    print(f"    Per-group breakdown:")
    for grp_name, grp_stats in report.get('per_group', {}).items():
        print(f"      {grp_name}: mean={grp_stats['mean']:.4f}, std={grp_stats['std']:.4f}, zero={grp_stats['zero_frac']:.3f}")
    if report.get('per_task'):
        print(f"    Per-task labels:")
        for task_id, tstat in report['per_task'].items():
            print(f"      {task_id}: n={tstat['n']}, stress_frac={tstat['stress_frac']:.3f}")
    if flag_reasons:
        print(f"    ⚠ FLAGS: {'; '.join(flag_reasons)}")
    else:
        print(f"    ✓ No anomalies detected")

# ── Compute combined subject stats (for per-subject ranking) ──────
print(f"\n{'─' * 70}")
print("  ALL SUBJECTS STATISTICS (StressID + WESAD combined)")
print(f"{'─' * 70}")

combined_meta = datasets_info.get('combined', {}).get('meta')
if combined_meta is not None:
    subj_stats = []
    for subj_id, grp in combined_meta.groupby('subject_id'):
        subj_stats.append({
            'subject': subj_id,
            'n_windows': len(grp),
            'stress_frac': float(grp['label'].mean()),
            'n_datasets': grp['dataset'].nunique() if 'dataset' in grp.columns else 1,
            'datasets': list(grp['dataset'].unique()) if 'dataset' in grp.columns else [],
        })
    df_subj = pd.DataFrame(subj_stats)
    df_subj = df_subj.sort_values('n_windows', ascending=False)
    print(f"\n  All {len(df_subj)} subjects:")
    print(f"  {'Subject':25s} {'Windows':8s} {'Stress%':8s} {'Datasets':25s}")
    print(f"  {'-'*66}")
    for _, row in df_subj.iterrows():
        marker = ' <<<' if any(f in str(row['subject']) for f in ['71i5', 'm8g5', 's2', 'wesad_s2']) else ''
        ds_list = ','.join(row['datasets']) if row['datasets'] else ''
        print(f"  {str(row['subject']):25s} {row['n_windows']:8d} {row['stress_frac']:7.1%} {ds_list:25s}{marker}")
    
    # Save to CSV
    df_subj.to_csv(os.path.join(PROJECT_ROOT, 'benchmark_results', 'all_subject_stats.csv'), index=False)
    print(f"\n  Saved to benchmark_results/all_subject_stats.csv")

# ── Print anomaly summary ──────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  ANOMALY SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Total flagged subjects analyzed: {len(all_reports)}")
for a in anomalies:
    if a['reasons']:
        print(f"\n  ⚠ {a['subject']} ({a['dataset']}): {'; '.join(a['reasons'])}")
    else:
        print(f"\n  ✓ {a['subject']} ({a['dataset']}): No anomalies in feature statistics")

# ── Label inversion check for m8g5 ─────────────────────────────────
print(f"\n{'─' * 70}")
print("  LABEL INVERSION CHECK")
print(f"{'─' * 70}")

print(f"  If stressid_m8g5 achieved AUC ≈ 0.0878 (below chance):")
print(f"    1 - 0.0878 = {1 - 0.0878:.4f}")
print(f"  If corrected AUC ≈ 0.91 → label inversion is confirmed.")
print(f"  Checking certified CSV labels for this subject...")

m8g5_keys = [k for k in all_reports.keys() if 'm8g5' in k.lower()]
for key in m8g5_keys:
    r = all_reports[key]
    print(f"  Found in dataset: {r['dataset']}")
    print(f"  Label counts: calm={r['n_calm']}, stress={r['n_stress']}")
    if r.get('per_task'):
        print(f"  Per-task labels:")
        for task_id, tstat in r['per_task'].items():
            print(f"    task_id={task_id}: n={tstat['n']}, stress_frac={tstat['stress_frac']:.3f}")

print(f"\n{'=' * 70}")
print(f"  DIAGNOSTIC COMPLETE")
print(f"{'=' * 70}")
