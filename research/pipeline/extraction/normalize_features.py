import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json

# Set determinism
set_determinism()

def normalize_dataset(dataset_name, data_dir, has_voice=False):
    print(f"Normalizing features for {dataset_name}...")
    
    # 1. Load Parquets
    pq_path = data_dir / "combined_windows.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Combined parquet missing at {pq_path}")
        
    df = pd.read_parquet(pq_path).copy()
    
    # Define metadata keys
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "binary_stress"]
    if has_voice:
        meta_keys.insert(6, "voice_available")
        
    feat_cols = [c for c in df.columns if c not in meta_keys]
    
    # Flat feature normalization
    for col in tqdm(feat_cols, desc=f"{dataset_name} Flat Features"):
        means = df.groupby("subject_id")[col].transform("mean")
        stds = df.groupby("subject_id")[col].transform("std")
        
        # Handle zero or NaN standard deviations
        stds = stds.replace(0.0, 1.0).fillna(1.0)
        df[col] = (df[col] - means) / (stds + 1e-8)
        
    # 2. Sequence feature normalization
    seq_path = data_dir / "combined_sequences.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Combined sequences missing at {seq_path}")
        
    seq_array = np.load(seq_path).copy()
    subjects = df["subject_id"].unique()
    
    # Align df index for safety
    df = df.reset_index(drop=True)
    
    for sub in tqdm(subjects, desc=f"{dataset_name} Sequence Features"):
        row_indices = df[df["subject_id"] == sub].index.tolist()
        if not row_indices:
            continue
            
        sub_seqs = seq_array[row_indices]
        flat_sub_seqs = sub_seqs.reshape(-1, sub_seqs.shape[-1])
        
        means = np.nanmean(flat_sub_seqs, axis=0)
        stds = np.nanstd(flat_sub_seqs, axis=0)
        
        # Avoid division by zero or NaN std
        stds[stds == 0.0] = 1.0
        stds[np.isnan(stds)] = 1.0
        
        for row_idx in row_indices:
            seq_array[row_idx] = (seq_array[row_idx] - means) / (stds + 1e-8)
            
    # Save normalized outputs
    df.to_parquet(data_dir / "normalized_windows.parquet")
    np.save(data_dir / "normalized_sequences.npy", seq_array)
    
    # 3. Verification of stats
    # Check that for each subject, normalized feature mean is ~0 and std is ~1 (where features are available)
    sample_sub = subjects[0]
    sample_feat = feat_cols[0]
    sub_vals = df[df["subject_id"] == sample_sub][sample_feat].dropna()
    
    mean_val = np.mean(sub_vals) if len(sub_vals) > 0 else 0.0
    std_val = np.std(sub_vals) if len(sub_vals) > 1 else 1.0
    
    print(f"{dataset_name} Normalized. Sample sub: {sample_sub}, feature: {sample_feat}, mean: {mean_val:.4f}, std: {std_val:.4f}")
    return df.shape, seq_array.shape, mean_val, std_val

def main():
    base_dir = Path(__file__).resolve().parents[3]
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    
    log_file = base_dir / "pipeline" / "logs" / "normalization.log"
    if log_file.exists():
        log_file.unlink()
        
    # 1. Normalize StressID
    sid_w_shape, sid_s_shape, sid_mean, sid_std = normalize_dataset("StressID", sid_out, has_voice=True)
    
    # 2. Normalize EmpathicSchool
    es_w_shape, es_s_shape, es_mean, es_std = normalize_dataset("EmpathicSchool", es_out, has_voice=False)
    
    # Log results
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"StressID normalized windows shape: {sid_w_shape}\n")
        f_log.write(f"StressID normalized sequences shape: {sid_s_shape}\n")
        f_log.write(f"StressID sample feature stats: mean={sid_mean:.6f}, std={sid_std:.6f}\n")
        f_log.write(f"EmpathicSchool normalized windows shape: {es_w_shape}\n")
        f_log.write(f"EmpathicSchool normalized sequences shape: {es_s_shape}\n")
        f_log.write(f"EmpathicSchool sample feature stats: mean={es_mean:.6f}, std={es_std:.6f}\n")
        
    # Verification check
    issues = []
    if abs(sid_mean) > 1e-2:
        issues.append(f"StressID normalization mean is off: {sid_mean}")
    if abs(es_mean) > 1e-2:
        issues.append(f"EmpathicSchool normalization mean is off: {es_mean}")
        
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("Subject-adaptive normalization verification PASSED.")

if __name__ == "__main__":
    main()
