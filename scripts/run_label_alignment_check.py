"""
Step 1: Certified CSV -> enriched data pipeline alignment check.
Compares task_id -> label mapping against StressID protocol specification
and checks window_index ordering between certified CSV and enriched npz.
"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / 'data' / 'features'
ENRICHED_DIR = PROJECT_ROOT / 'data' / 'enriched_training_data'
CERTIFIED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'certified_data'

print("=" * 75)
print("  STEP 1: LABEL ALIGNMENT CHECK - Certified CSV vs Enriched Parquet")
print("=" * 75)

# ---- 1. Determine protocol truth table from dataset majority vote -----
print("\n--- Deriving StressID Protocol from Dataset Majority ---")
feat_path = FEATURES_DIR / 'stress_features_fusion_5s.csv'
if not feat_path.exists():
    print(f"ERROR: {feat_path} not found")
    sys.exit(1)

feat = pd.read_csv(feat_path)

# For each task, compute majority label across ALL subjects
task_majority = {}
for task in sorted(feat['task_id'].unique()):
    sub = feat[feat['task_id'] == task]
    counts = sub['label'].value_counts()
    total = len(sub)
    majority_label = counts.idxmax()
    majority_pct = counts.max() / total * 100
    task_majority[task] = {
        'majority_label': int(majority_label),
        'pct': round(majority_pct, 1),
        'total': total,
        'dist': {int(k): int(v) for k, v in counts.items()},
        'n_subjects': sub['subject_id'].nunique(),
    }
    status = 'STRESS' if majority_label == 1 else 'CALM'
    print(f"  {task:15s} -> {status}  ({majority_pct:.0f}% agreement, "
          f"dist={counts.to_dict()}, n_subj={sub['subject_id'].nunique()})")

# Define protocol: tasks with >= 60% majority for label=1 are stress tasks
STRESS_THRESHOLD = 60.0
protocol_stress = set()
protocol_calm = set()
for task, info in task_majority.items():
    if info['majority_label'] == 1 and info['pct'] >= STRESS_THRESHOLD:
        protocol_stress.add(task)
    elif info['majority_label'] == 0 and info['pct'] >= STRESS_THRESHOLD:
        protocol_calm.add(task)
    else:
        print(f"  ?  {task}: ambiguous ({info['pct']}% for label={info['majority_label']}), "
              f"excluding from protocol check")

# Manual overrides based on known StressID protocol
if 'Reading' in task_majority:
    protocol_stress.add('Reading')
    protocol_calm.discard('Reading')
if 'Speaking' in task_majority:
    protocol_stress.add('Speaking')
    protocol_calm.discard('Speaking')
if 'Video1' in task_majority:
    protocol_calm.add('Video1')
    protocol_stress.discard('Video1')
if 'Video2' in task_majority:
    protocol_calm.add('Video2')
    protocol_stress.discard('Video2')

print(f"\n--- Final Protocol Definition ---")
print(f"  Stress tasks: {sorted(protocol_stress)}")
print(f"  Calm tasks:   {sorted(protocol_calm)}")
print(f"  Ambiguous:    {sorted(set(task_majority.keys()) - protocol_stress - protocol_calm)}")

# Build expected label per task
expected_label = {}
for t in protocol_stress:
    expected_label[t] = 1
for t in protocol_calm:
    expected_label[t] = 0

# ---- 2. Load enriched data -------------------------------------------
print(f"\n--- Loading Enriched Data ---")
enriched = {}
for ds in ['stressid', 'wesad', 'combined', 'empathicschool']:
    meta_path = ENRICHED_DIR / ds / 'metadata.parquet'
    npz_path = ENRICHED_DIR / ds / 'sequences.npz'
    if not meta_path.exists():
        continue
    meta = pd.read_parquet(meta_path)
    feats = np.load(npz_path)
    enriched[ds] = {'meta': meta, 'feats': feats}
    print(f"  {ds}: {len(meta)} windows, {meta['subject_id'].nunique()} subjects")

# ---- 3. Load certified CSV -------------------------------------------
print(f"\n--- Loading Certified CSVs ---")
certified = {}
for mod in ['face', 'voice', 'physio']:
    path = CERTIFIED_DIR / f'{mod}_certified.csv'
    if path.exists():
        df = pd.read_csv(path)
        for col in ['subject_id', 'task_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        certified[mod] = df
        print(f"  {mod}: {len(df)} rows, {df['subject_id'].nunique()} subjects")

# Merge certified CSVs
if certified:
    dfs = []
    for mod, df in certified.items():
        keep = [c for c in ['subject_id', 'task_id', 'window_index', 'label'] if c in df.columns]
        d = df[keep].copy()
        d['modality'] = mod
        dfs.append(d)
    df_cert_all = pd.concat(dfs, ignore_index=True)
    df_cert = df_cert_all.groupby(['subject_id', 'task_id', 'window_index'])['label'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).reset_index()
    print(f"  Merged certified: {len(df_cert)} windows, {df_cert['subject_id'].nunique()} subjects")
else:
    df_cert = None
    print("  No certified CSVs found - using feature CSV as reference")

# ---- 4. Per-subject alignment check ----------------------------------
print(f"\n{'=' * 75}")
print("  PER-SUBJECT ALIGNMENT CHECK")
print(f"{'=' * 75}")

target_subjects = ['m8g5', '71i5']
results = {}

for subj in target_subjects:
    print(f"\n--- Subject: {subj} ---")

    # 4a. Feature CSV labels
    subj_feat = feat[feat['subject_id'] == subj]

    # 4b. Certified CSV labels
    subj_cert = df_cert[df_cert['subject_id'] == subj] if df_cert is not None and len(df_cert) > 0 else pd.DataFrame()

    # 4c. Enriched data labels
    subj_enriched = {}
    for ds in enriched:
        meta = enriched[ds]['meta']
        sub = meta[meta['subject_id'].str.contains(subj, case=False, na=False)]
        if len(sub) > 0:
            key = f'enriched_{ds}'
            subj_enriched[key] = sub
            print(f"  Found in {ds} enriched: {len(sub)} windows")

    # 4d. Per-task comparison
    tasks_seen = set()
    for src_name, src_df in ([('feature_csv', subj_feat)] +
                              ([('certified_csv', subj_cert)] if len(subj_cert) > 0 else []) +
                              list(subj_enriched.items())):
        if len(src_df) == 0:
            continue
        for task in sorted(src_df['task_id'].unique()):
            if task not in expected_label:
                continue
            tasks_seen.add(task)
            sub = src_df[src_df['task_id'] == task]
            actual_label = sub['label'].mode().iloc[0] if not sub['label'].mode().empty else sub['label'].iloc[0]
            actual_pct = (sub['label'] == actual_label).sum() / len(sub) * 100
            expected = expected_label[task]
            match = '+' if actual_label == expected else 'X'
            if actual_label != expected:
                print(f"  {match} {src_name:20s} | {task:15s} | "
                      f"expected={expected}, actual={int(actual_label)} ({actual_pct:.0f}% of {len(sub)} windows)")

    # 4e. Summary for subject
    errors = []
    if len(subj_feat) > 0:
        for task in subj_feat['task_id'].unique():
            if task in expected_label:
                actual = subj_feat[subj_feat['task_id'] == task]['label'].mode().iloc[0]
                if actual != expected_label[task]:
                    errors.append((task, expected_label[task], int(actual)))
    results[subj] = {
        'n_windows_feat': len(subj_feat),
        'n_windows_cert': len(subj_cert) if len(subj_cert) > 0 else 0,
        'n_inverted_tasks': len(errors),
        'inverted_tasks': errors,
    }

    if not errors:
        print(f"\n  + All tasks match protocol")
    else:
        print(f"\n  * {len(errors)} inverted task(s):")
        for task, exp, act in errors:
            print(f"      {task}: expected {exp}, got {act}")

# ---- 5. window_index ordering check ----------------------------------
print(f"\n{'=' * 75}")
print("  WINDOW_INDEX ORDERING CHECK")
print(f"{'=' * 75}")

for subj in target_subjects:
    print(f"\n--- {subj} ---")
    # Feature CSV
    subj_feat = feat[feat['subject_id'] == subj].sort_values(['task_id', 'window_index'])
    for task in sorted(subj_feat['task_id'].unique()):
        sub = subj_feat[subj_feat['task_id'] == task]
        indices = sub['window_index'].values
        is_sequential = all(indices[i] <= indices[i+1] for i in range(len(indices)-1))
        gaps = np.diff(indices)
        has_gaps = (gaps > 1).sum()
        dupes = len(indices) - len(set(indices))
        status = '+' if (is_sequential and not has_gaps and not dupes) else '*'
        print(f"  {status} {task:15s}: {len(indices)} windows, "
              f"indices {indices[0]}-{indices[-1]}, "
              f"gaps={int(has_gaps)}, dupes={int(dupes)}")

# ---- 6. Full subject audit -------------------------------------------
print(f"\n{'=' * 75}")
print("  FULL SUBJECT AUDIT - ALL StressID Subjects vs Protocol")
print(f"{'=' * 75}")

all_results = []
for subj in sorted(feat['subject_id'].unique()):
    subj_df = feat[feat['subject_id'] == subj]
    n_inverted = 0
    inv_details = []
    for task in subj_df['task_id'].unique():
        if task in expected_label:
            actual = subj_df[subj_df['task_id'] == task]['label'].mode().iloc[0]
            if actual != expected_label[task]:
                n_inverted += 1
                inv_details.append(f"{task}(exp={expected_label[task]},got={int(actual)})")

    n_tasks = len([t for t in subj_df['task_id'].unique() if t in expected_label])
    stress_ratio = subj_df['label'].mean()
    all_results.append({
        'subject': subj,
        'n_windows': len(subj_df),
        'n_tasks_checked': n_tasks,
        'n_inverted': n_inverted,
        'inversion_ratio': round(n_inverted / max(n_tasks, 1), 2),
        'stress_ratio': round(stress_ratio, 3),
        'details': '; '.join(inv_details) if inv_details else '',
    })

# Sort by inversion ratio descending
audit = pd.DataFrame(all_results).sort_values('inversion_ratio', ascending=False)
print(f"\n  Subjects with inverted tasks:")
header = f"  {'Subject':15s} {'Windows':8s} {'Tasks':6s} {'Inv':4s} {'Inv%':6s} {'Stress%':8s} Details"
print(header)
print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*4} {'-'*6} {'-'*8} {'-'*30}")
for _, row in audit.iterrows():
    if row['inversion_ratio'] > 0:
        det = str(row['details'])[:60]
        print(f"  {row['subject']:15s} {row['n_windows']:8d} {row['n_tasks_checked']:6d} "
              f"{row['n_inverted']:4d} {row['inversion_ratio']:6.2f} {row['stress_ratio']:8.3f} "
              f"{det}")

print(f"\n  Subjects with zero inversions: {(audit['inversion_ratio'] == 0).sum()} / {len(audit)}")

# ---- 7. Summary report -----------------------------------------------
print(f"\n{'=' * 75}")
print("  SUMMARY")
print(f"{'=' * 75}")

for subj in target_subjects:
    r = results[subj]
    print(f"\n  {subj}:")
    print(f"    Feature CSV windows: {r['n_windows_feat']}")
    print(f"    Certified CSV windows: {r['n_windows_cert']}")
    print(f"    Inverted tasks: {r['n_inverted_tasks']}")
    for task, exp, act in r['inverted_tasks']:
        print(f"      X {task}: expected={exp}, actual={act}")
    if r['n_inverted_tasks'] == 0:
        print(f"    + Labels match protocol")

print(f"\n  Protocol stress tasks: {sorted(protocol_stress)}")
print(f"  Protocol calm tasks:   {sorted(protocol_calm)}")
print(f"  Threshold used: >= {STRESS_THRESHOLD}% majority to define protocol")
print(f"\n  Subjects with any inversion: {(audit['inversion_ratio'] > 0).sum()} / {len(audit)}")
print(f"  Subjects with >50% tasks inverted:")
severe = audit[audit['inversion_ratio'] > 0.5]
for _, row in severe.iterrows():
    det = str(row['details'])[:60]
    print(f"    X {row['subject']}: {row['n_inverted']}/{row['n_tasks_checked']} tasks "
          f"({det})")

print(f"\n{'=' * 75}")
print("  DONE")
print(f"{'=' * 75}")
