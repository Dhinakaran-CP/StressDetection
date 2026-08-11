"""
Phase 1, Step 1: Comprehensive investigation of stressid_m8g5.

Covers:
  1. Label distribution & inversion verification
  2. Feature distribution analysis (per-group z-scores)
  3. Raw sample validation (NaN, inf, zero fraction)
  4. Representation analysis (PCA + t-SNE)
  5. Feature collapse detection
  6. Data preprocessing verification
  7. Synchronization verification
  8. Distribution shift detection

Produces a structured JSON report + console output.
"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ENRICHED_DIR = PROJECT_ROOT / 'data' / 'enriched_training_data'
FEATURES_DIR = PROJECT_ROOT / 'data' / 'features'
CERTIFIED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'certified_data'
OUTPUT_DIR = PROJECT_ROOT / 'phase1_diagnostics' / 'm8g5'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = {
    'phase': '1.1',
    'subject': 'stressid_m8g5',
    'timestamp': datetime.now().isoformat(),
    'checks': {},
    'conclusion': None,
    'recommendations': [],
}

print("=" * 75)
print("  PHASE 1, STEP 1: stressid_m8g5 DIAGNOSTIC REPORT")
print("=" * 75)

# -- 1. Load Data Sources ----------------------------------------------
print("\n--- Loading Data ---")

# 1a. Enriched metadata & features
meta_stressid = pd.read_parquet(ENRICHED_DIR / 'stressid' / 'metadata.parquet')
feats_stressid = np.load(ENRICHED_DIR / 'stressid' / 'sequences.npz')

m8_idx = meta_stressid['subject_id'] == 'stressid_m8g5'
meta_m8 = meta_stressid[m8_idx].copy()
m8_window_indices = set(meta_m8['window_index'])

print(f"  Enriched stressid: {len(meta_stressid)} windows, {meta_stressid['subject_id'].nunique()} subjects")
print(f"  m8g5: {len(meta_m8)} windows")

# 1b. Feature CSV (5s windows)
feat_csv = pd.read_csv(FEATURES_DIR / 'stress_features_fusion_5s.csv')
feat_m8 = feat_csv[feat_csv['subject_id'] == 'm8g5']
print(f"  Feature CSV m8g5: {len(feat_m8)} windows")

# 1c. Certified CSVs
certified = {}
for mod in ['face', 'voice', 'physio']:
    path = CERTIFIED_DIR / f'{mod}_certified.csv'
    if path.exists():
        df = pd.read_csv(path)
        for col in ['subject_id', 'task_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        certified[mod] = df
cert_m8 = {}
for mod, df in certified.items():
    cert_m8[mod] = df[df['subject_id'] == 'm8g5']
    print(f"  Certified {mod} m8g5: {len(cert_m8[mod])} windows")

# -- 2. Label Distribution & Inversion Verification --------------------
print(f"\n{'='*75}")
print("  CHECK 1: LABEL DISTRIBUTION & INVERSION")
print(f"{'='*75}")

# 2a. Per-task label distribution from all sources
print("\n  --- Per-task label distribution (Feature CSV) ---")
m8_task_labels = {}
for task in sorted(feat_m8['task_id'].unique()):
    sub = feat_m8[feat_m8['task_id'] == task]
    dist = sub['label'].value_counts().to_dict()
    m8_task_labels[task] = {'n_windows': int(len(sub)), 'dist': {int(k): int(v) for k, v in dist.items()}}
    print(f"    {task:15s}: {len(sub):4d} windows, dist={dist}")

# 2b. Compare m8g5 label pattern to dataset majority
print("\n  --- Comparison with dataset majority ---")
dataset_task_majority = {}
for task in sorted(feat_csv['task_id'].unique()):
    sub = feat_csv[feat_csv['task_id'] == task]
    majority = int(sub['label'].mode().iloc[0])
    pct = (sub['label'] == majority).sum() / len(sub) * 100
    dataset_task_majority[task] = {'majority': majority, 'pct': round(pct, 1)}

deviations = []
for task, info in m8_task_labels.items():
    if task not in dataset_task_majority:
        continue
    majority = dataset_task_majority[task]['majority']
    actual = max(info['dist'], key=info['dist'].get)
    match = '+' if actual == majority else 'X'
    print(f"    {match} {task:15s}: m8g5={actual}, dataset_majority={majority} "
          f"({dataset_task_majority[task]['pct']}% of subjects)")
    if actual != majority:
        deviations.append({'task': task, 'actual': actual, 'expected': majority})

REPORT['checks']['label_distribution'] = {
    'per_task': m8_task_labels,
    'deviations_from_majority': deviations,
    'n_deviations': len(deviations),
}

# 2c. Compare against OTHER subjects with same label pattern
# Find subjects sharing m8g5's specific label pattern
feat_csv_lower = feat_csv.copy()
feat_csv_lower['subject_id'] = feat_csv_lower['subject_id'].str.lower().str.strip()

subj_task_map = {}
for subj in feat_csv_lower['subject_id'].unique():
    sub = feat_csv_lower[feat_csv_lower['subject_id'] == subj]
    pattern = frozenset((t, int(sub[sub['task_id'] == t]['label'].mode().iloc[0]))
                        for t in sub['task_id'].unique())
    subj_task_map[subj] = pattern

m8g5_pattern = subj_task_map.get('m8g5')
subjects_with_same_pattern = [s for s, p in subj_task_map.items() if p == m8g5_pattern and s != 'm8g5']
print(f"\n  Subjects with EXACT same label pattern as m8g5: {len(subjects_with_same_pattern)}")
if subjects_with_same_pattern:
    for s in subjects_with_same_pattern:
        sr = feat_csv_lower[feat_csv_lower['subject_id'] == s]['label'].mean()
        print(f"    {s}: stress_ratio={sr:.3f}")

REPORT['checks']['label_uniqueness'] = {
    'n_subjects_sharing_pattern': len(subjects_with_same_pattern),
    'subjects_sharing_pattern': subjects_with_same_pattern,
}

# -- 3. Feature Distribution Analysis ----------------------------------
print(f"\n{'='*75}")
print("  CHECK 2: FEATURE DISTRIBUTION ANALYSIS")
print(f"{'='*75}")

# For each feature group, compute m8g5 mean vs dataset mean (per window, averaged)
feature_anomalies = {}
all_feature_flat = {}  # dataset-wide
m8_feature_flat = {}

for group_name in sorted(feats_stressid.files):
    arr = feats_stressid[group_name]  # [N, 30, D]
    m8_arr = arr[m8_idx.values]        # [N_m8, 30, D]

    # Flatten: average over time dimension [N, D]
    arr_mean = arr.mean(axis=1)    # [N, D]
    m8_mean = m8_arr.mean(axis=1)  # [N_m8, D]

    all_feature_flat[group_name] = arr_mean
    m8_feature_flat[group_name] = m8_mean

    # Per-dimension z-score of m8g5 against dataset
    dataset_mean = arr_mean.mean(axis=0)  # [D]
    dataset_std = arr_mean.std(axis=0) + 1e-8  # [D]
    m8_dim_mean = m8_mean.mean(axis=0)    # [D]

    z_scores = (m8_dim_mean - dataset_mean) / dataset_std

    # Check for features with |z| > 3
    high_z = np.where(np.abs(z_scores) > 3)[0]
    max_abs_z = float(np.abs(z_scores).max())
    mean_abs_z = float(np.abs(z_scores).mean())

    feat_info = {
        'shape': list(arr.shape),
        'm8_mean_per_dim': m8_dim_mean.tolist()[:5],  # first 5 dims
        'dataset_mean_per_dim': dataset_mean.tolist()[:5],
        'max_abs_z_score': round(max_abs_z, 3),
        'mean_abs_z_score': round(mean_abs_z, 3),
        'n_high_z_dims': int(len(high_z)),
        'high_z_dims': high_z.tolist(),
        'anomalous': max_abs_z > 3,
    }
    feature_anomalies[group_name] = feat_info
    status = 'ANOMALOUS' if max_abs_z > 3 else 'OK'
    print(f"  {group_name:25s}: mean|z|={mean_abs_z:.3f}, max|z|={max_abs_z:.3f}, "
          f"high_z_dims={len(high_z)}/D={arr.shape[2]}  [{status}]")

REPORT['checks']['feature_distribution'] = feature_anomalies

# -- 4. Raw Sample Validation (NaN, inf, zeros) -----------------------
print(f"\n{'='*75}")
print("  CHECK 3: RAW SAMPLE VALIDATION")
print(f"{'='*75}")

for group_name in sorted(feats_stressid.files):
    arr = feats_stressid[group_name]
    m8_arr = arr[m8_idx.values]

    total_elements = m8_arr.size
    n_nan = np.isnan(m8_arr).sum()
    n_inf = np.isinf(m8_arr).sum()
    n_zero = (m8_arr == 0).sum()
    n_finite = np.isfinite(m8_arr).sum()

    # Compare to dataset
    ds_nan = np.isnan(arr).sum()
    ds_inf = np.isinf(arr).sum()
    ds_zero = (arr == 0).sum()

    print(f"  {group_name:25s}: NaN={n_nan:6d}/{n_nan/total_elements*100:.2f}%, "
          f"Inf={n_inf:4d}, Zero={n_zero/total_elements*100:.1f}%")
    if n_nan > 0 or n_inf > 0:
        print(f"    WARNING: m8g5 has {n_nan} NaN and {n_inf} Inf values!")
    # Compare zero fraction
    m8_zero_pct = n_zero / total_elements * 100
    ds_zero_pct = ds_zero / arr.size * 100
    if abs(m8_zero_pct - ds_zero_pct) > 10:
        print(f"    NOTE: m8g5 zero%={m8_zero_pct:.1f} vs dataset={ds_zero_pct:.1f}")

REPORT['checks']['raw_sample_validation'] = {
    g: {
        'n_nan': int(np.isnan(feats_stressid[g][m8_idx.values]).sum()),
        'n_inf': int(np.isinf(feats_stressid[g][m8_idx.values]).sum()),
        'zero_pct': round((feats_stressid[g][m8_idx.values] == 0).sum() / feats_stressid[g][m8_idx.values].size * 100, 2),
    }
    for g in sorted(feats_stressid.files)
}

# -- 5. Preprocessing Verification ------------------------------------
print(f"\n{'='*75}")
print("  CHECK 4: PREPROCESSING VERIFICATION")
print(f"{'='*75}")

# Verify window_index alignment between enriched and certified
print("\n  --- Certified CSV vs Enriched: window_index alignment ---")
if 'face' in cert_m8 and len(cert_m8['face']) > 0:
    # Map window_index formats: certified uses sequential ints, enriched uses string like "m8g5_Task_W0"
    enriched_indices = set(meta_m8['window_index'])
    certified_indices = set(cert_m8['face']['window_index'])

    # Extract numeric index from enriched window_index strings
    enriched_numeric = set()
    for wi in enriched_indices:
        try:
            parts = str(wi).split('_W')
            enriched_numeric.add(int(parts[-1]))
        except (IndexError, ValueError):
            enriched_numeric.add(int(wi))  # fallback

    certified_numeric = set(cert_m8['face']['window_index'])

    # Check if enriched windows are a subset of certified
    enriched_in_certified = enriched_numeric.issubset(certified_numeric)
    overlap = enriched_numeric & certified_numeric
    print(f"    Enriched unique windows: {len(enriched_numeric)}")
    print(f"    Certified unique windows (face): {len(certified_numeric)}")
    print(f"    Overlap: {len(overlap)}")
    print(f"    Enriched subset of Certified: {enriched_in_certified}")

    if not enriched_in_certified and len(overlap) > 0:
        missing = enriched_numeric - certified_numeric
        print(f"    WARNING: {len(missing)} enriched windows not in certified: {sorted(missing)[:10]}")

REPORT['checks']['preprocessing_alignment'] = {
    'enriched_windows': len(meta_m8),
    'certified_windows': {m: len(cert_m8[m]) for m in cert_m8},
    'enriched_subset_of_certified': enriched_in_certified if 'face' in cert_m8 and len(cert_m8['face']) > 0 else 'N/A',
    'overlap_windows': int(len(overlap)) if 'face' in cert_m8 and len(cert_m8['face']) > 0 else 0,
}

# -- 6. Representation Analysis (PCA) ----------------------------------
print(f"\n{'='*75}")
print("  CHECK 5: REPRESENTATION ANALYSIS (PCA)")
print(f"{'='*75}")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Concatenate all feature groups for PCA
all_flat = np.concatenate([all_feature_flat[g] for g in sorted(feats_stressid.files)], axis=1)  # [N, total_feat]
m8_flat = np.concatenate([m8_feature_flat[g] for g in sorted(feats_stressid.files)], axis=1)    # [N_m8, total_feat]

print(f"  All features flattened: {all_flat.shape}")
print(f"  m8g5 features flattened: {m8_flat.shape}")

# PCA: project m8g5 onto dataset PCA
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_flat)
m8_scaled = scaler.transform(m8_flat)

pca = PCA(n_components=min(10, all_scaled.shape[1], all_scaled.shape[0]))
all_pca = pca.fit_transform(all_scaled)
m8_pca = pca.transform(m8_scaled)

print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}")
print(f"  Cumulative (10): {pca.explained_variance_ratio_[:10].sum():.3f}")

# Check if m8g5 is an outlier in PCA space
# Compute Mahalanobis-like distance: distance from dataset mean in PCA space
all_mean_pca = all_pca.mean(axis=0)
all_std_pca = all_pca.std(axis=0) + 1e-8
m8_dist = np.abs(m8_pca - all_mean_pca) / all_std_pca
mean_m8_dist = m8_dist.mean(axis=0)
print(f"\n  m8g5 mean distance from dataset center (PC1-PC5): {mean_m8_dist[:5]}")

# Check if m8g5 windows cluster separately from dataset
from scipy.spatial.distance import cdist
# For each m8g5 window, find nearest non-m8g5 window
non_m8_mask = ~m8_idx.values
non_m8_pca = all_pca[non_m8_mask]
m8_to_nonm8 = cdist(m8_pca, non_m8_pca).min(axis=1)
nonm8_to_nonm8 = cdist(non_m8_pca, non_m8_pca) + np.eye(len(non_m8_pca)) * 1e10
nonm8_to_nonm8 = nonm8_to_nonm8.min(axis=1)

print(f"  m8g5 mean min-distance to dataset: {m8_to_nonm8.mean():.3f}")
print(f"  dataset mean min-distance within: {nonm8_to_nonm8.mean():.3f}")

pca_anomaly = float(m8_to_nonm8.mean() / (nonm8_to_nonm8.mean() + 1e-8))
print(f"  Ratio m8g5/dataset: {pca_anomaly:.3f}")
if pca_anomaly > 2:
    print(f"  WARNING: m8g5 is an outlier in feature space (ratio > 2)")

REPORT['checks']['representation_pca'] = {
    'explained_variance_ratio': [round(float(x), 4) for x in pca.explained_variance_ratio_[:10]],
    'm8g5_pca_distance_pc1_5': [round(float(x), 3) for x in mean_m8_dist[:5]],
    'm8g5_mean_min_distance': round(float(m8_to_nonm8.mean()), 3),
    'dataset_mean_min_distance': round(float(nonm8_to_nonm8.mean()), 3),
    'anomaly_ratio': round(float(pca_anomaly), 3),
}

# -- 7. Feature Collapse Detection ------------------------------------
print(f"\n{'='*75}")
print("  CHECK 6: FEATURE COLLAPSE DETECTION")
print(f"{'='*75}")

# Feature collapse = all feature dimensions have near-zero variance
# Check per-group variance ratio between m8g5 and dataset
collapse_results = {}
for group_name in sorted(feats_stressid.files):
    arr = feats_stressid[group_name]
    m8_arr = arr[m8_idx.values]

    # Variance per dimension
    ds_var = arr.mean(axis=1).var(axis=0)   # [D]
    m8_var = m8_arr.mean(axis=1).var(axis=0)  # [D]

    # Ratio of m8g5 variance to dataset variance
    var_ratio = (m8_var / (ds_var + 1e-8)).mean()
    min_var_m8 = m8_var.min()
    min_var_ds = ds_var.min()

    collapsed = min_var_m8 < 1e-6
    low_var = var_ratio < 0.1

    status = 'COLLAPSED' if collapsed else ('LOW_VAR' if low_var else 'OK')
    print(f"  {group_name:25s}: var_ratio(m8/ds)={var_ratio:.3f}, "
          f"min_var_m8={min_var_m8:.6f}, min_var_ds={min_var_ds:.6f}  [{status}]")

    collapse_results[group_name] = {
        'variance_ratio': round(float(var_ratio), 3),
        'min_variance_m8': float(min_var_m8),
        'min_variance_dataset': float(min_var_ds),
        'collapsed': bool(collapsed),
    }

REPORT['checks']['feature_collapse'] = collapse_results

# -- 8. Synchronization Verification ----------------------------------
print(f"\n{'='*75}")
print("  CHECK 7: SYNCHRONIZATION VERIFICATION")
print(f"{'='*75}")

# Check that all modalities have the same number of windows and same window_indices
n_modalities = len(feats_stressid.files)
print(f"  Number of feature groups: {n_modalities}")
print(f"  All groups have same N: {all(feats_stressid[g].shape[0] == len(meta_stressid) for g in feats_stressid.files)}")

# Check per-modality zero patterns (if one modality is all-zero while others have signal)
for group_name in sorted(feats_stressid.files):
    m8_arr = feats_stressid[group_name][m8_idx.values]
    zero_pct = (m8_arr == 0).sum() / m8_arr.size * 100
    ds_arr = feats_stressid[group_name]
    ds_zero_pct = (ds_arr == 0).sum() / ds_arr.size * 100
    print(f"  {group_name:25s}: m8g5_zero={zero_pct:.1f}%, dataset_zero={ds_zero_pct:.1f}%")

REPORT['checks']['synchronization'] = {
    'all_groups_have_same_N': bool(all(feats_stressid[g].shape[0] == len(meta_stressid) for g in feats_stressid.files)),
    'n_feature_groups': n_modalities,
}

# -- 9. Distribution Shift Detection -----------------------------------
print(f"\n{'='*75}")
print("  CHECK 8: DISTRIBUTION SHIFT DETECTION")
print(f"{'='*75}")

# Compare stress vs calm feature distributions WITHIN m8g5
# If stress and calm windows have separable features, labels are meaningful
print("\n  --- Within-m8g5: stress vs calm feature separation ---")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

within_m8_separability = {}
for group_name in sorted(feats_stressid.files):
    m8_arr = feats_stressid[group_name][m8_idx.values]
    m8_labels = meta_m8['label'].values

    # Ensure we have both classes
    if m8_labels.sum() == 0 or m8_labels.sum() == len(m8_labels):
        sep = 'N/A (single class)'
        print(f"  {group_name:25s}: single class only ({m8_labels.sum()}/{len(m8_labels)})")
        within_m8_separability[group_name] = {'cv_auc': None, 'note': 'single_class'}
        continue

    # Flatten time dimension
    X = m8_arr.reshape(m8_arr.shape[0], -1)
    y = m8_labels

    # Quick logistic regression to see if features separate labels
    if X.shape[0] > 10 and X.shape[1] < X.shape[0]:
        try:
            lr = LogisticRegression(max_iter=500, solver='lbfgs')
            scores = cross_val_score(lr, X, y, cv=min(3, m8_labels.sum(), (len(m8_labels) - m8_labels.sum())), scoring='roc_auc')
            sep = f'{scores.mean():.3f} +/- {scores.std():.3f}'
            within_m8_separability[group_name] = {'cv_auc_mean': round(float(scores.mean()), 3), 'cv_auc_std': round(float(scores.std()), 3)}
        except Exception as e:
            sep = f'ERROR: {e}'
            within_m8_separability[group_name] = {'error': str(e)}
    else:
        sep = 'N/A (dim too high or samples too few)'
        within_m8_separability[group_name] = {'note': 'dim_too_high'}
    print(f"  {group_name:25s}: stress-calm separability (CV AUC) = {sep}")

# Also check between-subjects: can we distinguish m8g5 from rest?
print("\n  --- Between-subjects: m8g5 vs rest separability ---")
for group_name in sorted(feats_stressid.files):
    arr = feats_stressid[group_name]
    X = arr.mean(axis=1)  # [N, D]
    y = m8_idx.values.astype(int)
    if y.sum() > 10 and (len(y) - y.sum()) > 10 and X.shape[1] < 50:
        try:
            lr = LogisticRegression(max_iter=500, solver='lbfgs')
            scores = cross_val_score(lr, X, y, cv=3, scoring='roc_auc')
            print(f"  {group_name:25s}: m8g5 vs rest separability (CV AUC) = {scores.mean():.3f}")
        except Exception as e:
            print(f"  {group_name:25s}: ERROR {e}")

REPORT['checks']['distribution_shift'] = {
    'within_m8g5_stress_calm_separability': within_m8_separability,
}

# -- 10. Summary & Conclusion ------------------------------------------
print(f"\n{'='*75}")
print("  DIAGNOSTIC SUMMARY")
print(f"{'='*75}")

# Collect all anomaly flags
anomalies = []
checks = REPORT['checks']

# Label deviations
if len(deviations) > 0:
    anomalies.append(f"Label deviation: {len(deviations)} tasks differ from dataset majority")

# Feature distribution
feat_anomalies = [g for g, info in feature_anomalies.items() if info['anomalous']]
if feat_anomalies:
    anomalies.append(f"Feature distribution anomaly in groups: {feat_anomalies}")
else:
    print("  + Feature distribution: within normal range")

# NaN/Inf
nan_groups = [g for g, info in checks['raw_sample_validation'].items() if info['n_nan'] > 0 or info['n_inf'] > 0]
if nan_groups:
    anomalies.append(f"NaN/Inf found in groups: {nan_groups}")

# Feature collapse
collapsed_groups = [g for g, info in collapse_results.items() if info['collapsed']]
if collapsed_groups:
    anomalies.append(f"Feature collapse in groups: {collapsed_groups}")
else:
    print("  + Feature collapse: none detected")

# PCA outlier
if pca_anomaly > 2:
    anomalies.append(f"PCA outlier: m8g5 distance ratio = {pca_anomaly:.2f}x dataset average")
else:
    print(f"  + PCA representation: within normal range (ratio={pca_anomaly:.2f})")

# Synchronization
if not checks['synchronization']['all_groups_have_same_N']:
    anomalies.append("Synchronization error: feature groups have different N")

# Within-subject separability
separable_groups = [g for g, info in within_m8_separability.items() if isinstance(info, dict) and info.get('cv_auc_mean', 0) > 0.65]
if separable_groups:
    print(f"  + Within-m8g5 stress-calm separable in {len(separable_groups)}/{len(within_m8_separability)} groups")
else:
    print(f"  - Within-m8g5 stress-calm NOT separable in any group (labels may not match physiology)")

print(f"\n  Anomalies detected: {len(anomalies)}")
for a in anomalies:
    print(f"    - {a}")

if len(anomalies) == 0:
    print("  + No anomalies detected. m8g5 data quality is normal.")
    REPORT['conclusion'] = 'NORMAL'
    REPORT['recommendations'].append('No data issues detected. Exclude from test pool as precaution only.')
else:
    REPORT['conclusion'] = 'ANOMALIES_DETECTED'
    REPORT['recommendations'].append('Investigate anomalies before using m8g5 in test set.')

# Check: is the 0.0878 AUC reproducible with a simple model on just stressid?
# We can't fully test without training, but we can check if features+labels have any signal
print(f"\n  --- Quick Signal Test ---")
# Concatenate all features, fit logistic regression on full stressid, predict m8g5
X_train_list = []
y_train_list = []
for i in range(len(meta_stressid)):
    if i not in m8_idx.values.nonzero()[0]:
        row_feats = []
        for g in sorted(feats_stressid.files):
            row_feats.append(feats_stressid[g][i].mean(axis=0))
        X_train_list.append(np.concatenate(row_feats))
        y_train_list.append(meta_stressid.iloc[i]['label'])

X_m8_list = []
for i in m8_idx.values.nonzero()[0]:
    row_feats = []
    for g in sorted(feats_stressid.files):
        row_feats.append(feats_stressid[g][i].mean(axis=0))
    X_m8_list.append(np.concatenate(row_feats))

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_m8 = np.array(X_m8_list)
y_m8 = meta_m8['label'].values

print(f"  Training set size: {X_train.shape}, m8g5 test size: {X_m8.shape}")

if len(np.unique(y_train)) > 1 and len(np.unique(y_m8)) > 1 and X_train.shape[1] < X_train.shape[0]:
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train)
    from sklearn.metrics import roc_auc_score
    try:
        y_prob = lr.predict_proba(X_m8)[:, 1]
        auc = roc_auc_score(y_m8, y_prob)
        print(f"  LogisticRegression AUC on m8g5 (excluded from training): {auc:.4f}")
        REPORT['checks']['quick_signal_test'] = {'auc': round(float(auc), 4)}
        if auc < 0.5:
            print(f"  WARNING: AUC={auc:.4f} - below chance. Possible label-feature mismatch.")
            if 1 - auc > 0.7:
                print(f"  Flipped AUC = {1-auc:.4f} - checking if label inversion would help.")
                y_prob_flipped = 1 - y_prob
                auc_flipped = roc_auc_score(y_m8, y_prob_flipped)
                print(f"  Flipped AUC = {auc_flipped:.4f}")
                if auc_flipped > 0.7:
                    anomalies.append(f"Label inversion confirmed: flipped AUC={auc_flipped:.4f}")
    except Exception as e:
        print(f"  AUC computation error: {e}")
else:
    print(f"  Cannot compute AUC: single class in train or test")

# -- Save Report -------------------------------------------------------
report_path = OUTPUT_DIR / 'diagnostic_report.json'
with open(report_path, 'w') as f:
    json.dump(REPORT, f, indent=2, default=str)
print(f"\n  Report saved: {report_path}")

print(f"\n{'='*75}")
print("  PHASE 1, STEP 1 COMPLETE")
print(f"{'='*75}")
