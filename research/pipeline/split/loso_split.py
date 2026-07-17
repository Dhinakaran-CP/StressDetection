import os
import json
import pandas as pd
from pathlib import Path
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json

# Set determinism
set_determinism()

def generate_loso_splits(dataset_name, parquet_path):
    print(f"Generating LOSO splits for {dataset_name}...")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Normalized parquet missing at {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    
    subjects = sorted(df["subject_id"].unique())
    folds = []
    
    for sub in subjects:
        # Test set is all windows of subject sub
        test_df = df[df["subject_id"] == sub]
        train_df = df[df["subject_id"] != sub]
        
        test_windows = test_df["window_id"].tolist()
        train_windows = train_df["window_id"].tolist()
        
        # Verify no leaks
        train_subs = set(train_df["subject_id"].unique())
        if sub in train_subs:
            raise RuntimeError(f"Subject leakage detected in fold for {sub}!")
            
        # Verify completeness
        total_w_count = len(test_windows) + len(train_windows)
        if total_w_count != len(df):
            raise RuntimeError(f"Window count mismatch in fold for {sub}! Expected {len(df)}, got {total_w_count}")
            
        folds.append({
            "test_subject": sub,
            "train_windows": train_windows,
            "test_windows": test_windows,
            "train_size": len(train_windows),
            "test_size": len(test_windows)
        })
        
    print(f"{dataset_name} LOSO: generated {len(folds)} folds.")
    return folds

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    sid_pq = base_dir / "pipeline" / "data" / "stressid" / "normalized_windows.parquet"
    es_pq = base_dir / "pipeline" / "data" / "empathicschool" / "normalized_windows.parquet"
    
    log_file = base_dir / "pipeline" / "logs" / "loso_split.log"
    if log_file.exists():
        log_file.unlink()
        
    # Generate splits
    sid_folds = generate_loso_splits("StressID", sid_pq)
    es_folds = generate_loso_splits("EmpathicSchool", es_pq)
    
    # Save splits registry
    registry = {
        "datasets": {
            "stressid": {
                "folds": sid_folds,
                "total_folds": len(sid_folds)
            },
            "empathicschool": {
                "folds": es_folds,
                "total_folds": len(es_folds)
            }
        }
    }
    
    registry_path = base_dir / "pipeline" / "logs" / "loso_splits.json"
    write_json(registry, registry_path)
    
    # Calculate stats for logging
    sid_test_sizes = [f["test_size"] for f in sid_folds]
    es_test_sizes = [f["test_size"] for f in es_folds]
    
    sid_avg_test = np.mean(sid_test_sizes)
    es_avg_test = np.mean(es_test_sizes)
    
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"StressID LOSO folds count: {len(sid_folds)}\n")
        f_log.write(f"StressID average test fold size (windows): {sid_avg_test:.2f}\n")
        f_log.write(f"EmpathicSchool LOSO folds count: {len(es_folds)}\n")
        f_log.write(f"EmpathicSchool average test fold size (windows): {es_avg_test:.2f}\n")
        
    # Self-verification check
    issues = []
    if len(sid_folds) == 0:
        issues.append("StressID folds count is 0")
    if len(es_folds) == 0:
        issues.append("EmpathicSchool folds count is 0")
        
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("LOSO split registry verification PASSED.")

# Import numpy locally inside main or at top
import numpy as np

if __name__ == "__main__":
    main()
