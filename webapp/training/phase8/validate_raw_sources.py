"""
Validate our pipeline fixes against the actual raw source data.
Checks if NaN/extreme values exist in raw files or were introduced by pipeline.
"""
import os, json, pickle, zipfile, glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

print("=" * 70)
print("  RAW SOURCE DATA VALIDATION")
print("=" * 70)


# ── STRESSID: Check raw physiological CSVs ──────────────────────────

def check_stressid_raw():
    print("\n" + "-" * 70)
    print("  [1] STRESSID — Raw Physiological CSVs")
    print("-" * 70)
    
    physio_dir = os.path.join(DATA_DIR, 'stressid', 'Physiological')
    if not os.path.isdir(physio_dir):
        print(f"  NOT FOUND: {physio_dir}")
        return
    
    all_entries = sorted([d for d in os.listdir(physio_dir) 
                          if os.path.isdir(os.path.join(physio_dir, d))])
    subjects = all_entries[:5]  # sample 5 subjects
    print(f"  Sampling {len(subjects)} subjects from {len(all_entries)} total...")
    
    total_nan_ecg = total_nan_eda = total_nan_rr = 0
    total_rows = 0
    extreme_ecg = extreme_eda = extreme_rr = 0
    
    for subj in subjects:
        subj_dir = os.path.join(physio_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for fname in sorted(os.listdir(subj_dir))[:3]:  # 3 tasks per subject
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(subj_dir, fname)
            try:
                df = pd.read_csv(fpath)
            except:
                continue
            total_rows += len(df)
            if 'ECG' in df.columns:
                total_nan_ecg += df['ECG'].isna().sum()
                extreme_ecg += (np.abs(df['ECG']) > 1e6).sum()
            if 'EDA' in df.columns:
                total_nan_eda += df['EDA'].isna().sum()
                extreme_eda += (np.abs(df['EDA']) > 1e6).sum()
            if 'RR' in df.columns:
                total_nan_rr += df['RR'].isna().sum()
                extreme_rr += (np.abs(df['RR']) > 1e6).sum()
    
    print(f"  Files sampled: {len(subjects)} subjects × 3 tasks each")
    print(f"  Total rows: {total_rows}")
    print(f"  ECG: NaN={total_nan_ecg}  extreme>{'YES' if extreme_ecg>0 else 'NO'} ({extreme_ecg})")
    print(f"  EDA: NaN={total_nan_eda}  extreme>{'YES' if extreme_eda>0 else 'NO'} ({extreme_eda})")
    print(f"  RR:  NaN={total_nan_rr}  extreme>{'YES' if extreme_rr>0 else 'NO'} ({extreme_rr})")
    
    # Show a sample of actual values
    subj_files = sorted([f for f in os.listdir(os.path.join(physio_dir, subjects[0])) 
                         if f.endswith('.txt')])
    fpath = os.path.join(physio_dir, subjects[0], subj_files[0]) if subj_files else None
    if fpath is None or not os.path.exists(fpath):
        print("  No .txt files found in first subject dir")
        return
    df = pd.read_csv(fpath)
    print(f"\n  Sample raw values ({os.path.basename(fpath)}):")
    print(f"    ECG: min={df['ECG'].min()} max={df['ECG'].max()} mean={df['ECG'].mean():.0f}")
    print(f"    EDA: min={df['EDA'].min()} max={df['EDA'].max()} mean={df['EDA'].mean():.0f}")
    print(f"    RR:  min={df['RR'].min()}  max={df['RR'].max()}  mean={df['RR'].mean():.0f}")


# ── WESAD: Check raw pickle files ───────────────────────────────────

def check_wesad_raw():
    print("\n" + "-" * 70)
    print("  [2] WESAD — Raw Pickle Files")
    print("-" * 70)
    
    wesad_dir = os.path.join(DATA_DIR, 'wesad')
    subjects = sorted([s for s in os.listdir(wesad_dir) if s.startswith('S') and os.path.isdir(os.path.join(wesad_dir, s))])
    print(f"  Found {len(subjects)} subjects")
    
    # Sample 2 subjects
    for subj in subjects[:2]:
        pkl_path = os.path.join(wesad_dir, subj, f'{subj}.pkl')
        if not os.path.exists(pkl_path):
            print(f"  {subj}: no pickle found")
            continue
        fsize = os.path.getsize(pkl_path)
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"\n  {subj}: pickle loaded ({fsize/1e6:.0f} MB)")
            print(f"  Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    nan_pct = np.isnan(v).mean() * 100 if v.size > 0 else 0
                    extreme = (np.abs(v[np.isfinite(v)]) > 1e6).sum() if v.size > 0 else 0
                    print(f"    {k}: shape={v.shape} dtype={v.dtype} "
                          f"NaN={nan_pct:.1f}% extreme={extreme}")
            # Check label
            if 'label' in data:
                lbl = data['label']
                unique, counts = np.unique(lbl, return_counts=True)
                print(f"  Labels: {dict(zip(unique, counts))}")
        except Exception as e:
            print(f"  {subj}: ERROR loading pickle: {e}")


# ── EMPATHICSCHOOL: Check raw physio CSVs ──────────────────────────

def check_empathicschool_raw():
    print("\n" + "-" * 70)
    print("  [3] EMPATHICSCHOOL — Raw E4 CSV Files")
    print("-" * 70)
    
    es_dir = os.path.join(DATA_DIR, 'empathicschool')
    subjects = sorted([s for s in os.listdir(es_dir) 
                       if s.startswith('S') and os.path.isdir(os.path.join(es_dir, s))])
    print(f"  Found {len(subjects)} subjects")
    
    # Check EDA/HR/ACC values for extreme outliers
    extreme_vals = []
    nan_stats = []
    
    for subj in subjects[:5]:  # sample 5
        subj_dir = os.path.join(es_dir, subj)
        # Find CSV files recursively
        csv_files = []
        for root, dirs, files in os.walk(subj_dir):
            for f in files:
                if f.endswith('.csv') and not 'Landmark' in f and not 'Xception' in f:
                    csv_files.append(os.path.join(root, f))
        
        for csv_path in csv_files[:4]:  # max 4 per subject
            try:
                df = pd.read_csv(csv_path, nrows=5000)  # sample first 5000 rows
                for col in df.select_dtypes(include=[np.number]).columns:
                    vals = df[col].values
                    finite = vals[np.isfinite(vals)]
                    if len(finite) == 0:
                        continue
                    nan_pct = np.isnan(vals).mean() * 100
                    extreme = (np.abs(finite) > 1e6).sum()
                    if nan_pct > 0 or extreme > 0:
                        extreme_vals.append((subj, os.path.basename(csv_path), col, 
                                            nan_pct, extreme, 
                                            finite.min(), finite.max(), finite.mean()))
            except Exception as e:
                pass
    
    if extreme_vals:
        print(f"\n  Found {len(extreme_vals)} channels with NaN or extreme values:")
        for subj, fname, col, nan_pct, extreme, vmin, vmax, vmean in extreme_vals[:15]:
            print(f"    {subj}/{fname}/{col}: NaN={nan_pct:.1f}%  extreme={extreme}  "
                  f"range=[{vmin:.2f}, {vmax:.2f}] mean={vmean:.2f}")
    else:
        print(f"  No NaN or extreme values in sample")
    
    # Check face video files exist
    print(f"\n  Face video check:")
    video_count = 0
    for subj in subjects[:5]:
        subj_dir = os.path.join(es_dir, subj)
        for root, dirs, files in os.walk(subj_dir):
            for f in files:
                if f.endswith('.mp4'):
                    video_count += 1
    print(f"    Found {video_count} MP4 files in sample (5 subjects)")
    
    # Check voice files exist
    audio_count = 0
    for subj in subjects[:5]:
        subj_dir = os.path.join(es_dir, subj)
        for root, dirs, files in os.walk(subj_dir):
            for f in files:
                if f.endswith(('.wav', '.m4a', '.mp3')):
                    audio_count += 1
    print(f"    Found {audio_count} audio files in sample (5 subjects)")


# ── LABEL VALIDATION ─────────────────────────────────────────────────

def check_labels():
    print("\n" + "-" * 70)
    print("  [4] LABEL VALIDATION")
    print("-" * 70)
    
    # StressID labels
    labels_path = os.path.join(DATA_DIR, 'stressid', 'labels.csv')
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
        print(f"  StressID labels: {len(labels)} rows")
        print(f"    Columns: {list(labels.columns)}")
        if 'binary_stress' in labels.columns:
            print(f"    Stress distribution: {labels['binary_stress'].value_counts().to_dict()}")
        elif 'label' in labels.columns:
            print(f"    Stress distribution: {labels['label'].value_counts().to_dict()}")
    
    # WESAD labels
    wesad_dir = os.path.join(DATA_DIR, 'wesad')
    quest_files = glob.glob(os.path.join(wesad_dir, 'S*', '*quest.csv'))
    print(f"  WESAD quest files: {len(quest_files)}")
    if quest_files:
        q = pd.read_csv(quest_files[0])
        print(f"    Sample quest columns: {list(q.columns)}")
    
    # EmpathicSchool labels
    es_dir = os.path.join(DATA_DIR, 'empathicschool')
    tags_files = glob.glob(os.path.join(es_dir, 'S*', '*tags.csv'))
    print(f"  EmpathicSchool tags files: {len(tags_files)}")
    if tags_files:
        t = pd.read_csv(tags_files[0])
        print(f"    Sample tags columns: {list(t.columns)}")


if __name__ == '__main__':
    check_stressid_raw()
    check_wesad_raw()
    check_empathicschool_raw()
    check_labels()
    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)
