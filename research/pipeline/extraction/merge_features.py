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

def merge_dataset_features(dataset_name, data_dir, ref_dir=None, has_face=True, has_voice=False):
    print(f"Merging features for {dataset_name}...")
    
    physio_path = data_dir / "physio_windows.parquet"
    if not physio_path.exists():
        raise FileNotFoundError(f"Required physio parquet missing in {data_dir}")
    df_physio = pd.read_parquet(physio_path)
    
    # 1. Resolve Face
    if has_face:
        face_path = data_dir / "face_windows.parquet"
        df_face = pd.read_parquet(face_path)
    else:
        # Load from reference (StressID) to get the face column structure
        ref_face_path = ref_dir / "face_windows.parquet"
        df_ref_face = pd.read_parquet(ref_face_path)
        # Create empty DataFrame with same columns and window_id matching df_physio
        df_face = pd.DataFrame(np.nan, index=np.arange(len(df_physio)), columns=df_ref_face.columns)
        df_face["window_id"] = df_physio["window_id"].values
        df_face["subject_id"] = df_physio["subject_id"].values
        df_face["dataset_source"] = df_physio["dataset_source"].values
        df_face["task_name"] = df_physio["task_name"].values
        df_face["binary_stress"] = df_physio["binary_stress"].values
        df_face["face_available"] = 0
        
    df_merged = pd.merge(df_face, df_physio, on="window_id", how="outer", suffixes=("_face", "_physio"))
    
    meta_cols = ["subject_id", "dataset_source", "task_name", "binary_stress"]
    for col in meta_cols:
        col_face = f"{col}_face"
        col_physio = f"{col}_physio"
        df_merged[col] = df_merged[col_face].fillna(df_merged[col_physio])
        df_merged.drop([col_face, col_physio], axis=1, inplace=True)
        
    df_merged["face_available"] = df_merged["face_available"].fillna(0).astype(int)
    df_merged["physio_available"] = df_merged["physio_available"].fillna(0).astype(int)
    
    # 2. Resolve Voice
    if has_voice:
        voice_path = data_dir / "voice_windows.parquet"
        df_voice = pd.read_parquet(voice_path)
        df_merged = pd.merge(df_merged, df_voice, on="window_id", how="outer", suffixes=("", "_voice"))
        for col in meta_cols:
            col_voice = f"{col}_voice"
            if col_voice in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(df_merged[col_voice])
                df_merged.drop(col_voice, axis=1, inplace=True)
        df_merged["voice_available"] = df_merged["voice_available"].fillna(0).astype(int)
    else:
        # Load from reference (StressID) to get the voice column structure
        ref_voice_path = ref_dir / "voice_windows.parquet"
        df_ref_voice = pd.read_parquet(ref_voice_path)
        df_voice_empty = pd.DataFrame(np.nan, index=np.arange(len(df_merged)), columns=[c for c in df_ref_voice.columns if c not in ["subject_id", "dataset_source", "task_name", "binary_stress", "voice_available"]])
        df_voice_empty["window_id"] = df_merged["window_id"].values
        df_merged = pd.merge(df_merged, df_voice_empty, on="window_id", how="left")
        df_merged["voice_available"] = 0
        
    # 3. Merge Sequences
    if has_face:
        face_seq = np.load(data_dir / "face_sequences.npy")
        face_idx = pd.read_parquet(data_dir / "face_sequences_index.parquet").set_index("window_id")["sequence_index"].to_dict()
    else:
        face_idx = {}
        
    physio_seq = np.load(data_dir / "physio_sequences.npy")
    physio_idx = pd.read_parquet(data_dir / "physio_sequences_index.parquet").set_index("window_id")["sequence_index"].to_dict()
    
    if has_voice:
        voice_seq = np.load(data_dir / "voice_sequences.npy")
        voice_idx = pd.read_parquet(data_dir / "voice_sequences_index.parquet").set_index("window_id")["sequence_index"].to_dict()
    else:
        voice_idx = {}
        
    merged_windows = df_merged["window_id"].tolist()
    combined_seqs = []
    window_meta = []
    
    empty_face = np.full((30, 34), np.nan, dtype=np.float32)
    empty_physio = np.full((30, 14), np.nan, dtype=np.float32)
    empty_voice = np.full((30, 24), np.nan, dtype=np.float32)
    
    for i, w_id in enumerate(tqdm(merged_windows, desc=f"{dataset_name} Sequences")):
        if has_face and w_id in face_idx:
            f_s = face_seq[face_idx[w_id]]
        else:
            f_s = empty_face
            
        if w_id in physio_idx:
            p_s = physio_seq[physio_idx[w_id]]
        else:
            p_s = empty_physio
            
        if has_voice and w_id in voice_idx:
            v_s = voice_seq[voice_idx[w_id]]
        else:
            v_s = empty_voice
            
        combined_segment = np.concatenate([f_s, v_s, p_s], axis=-1)
        combined_seqs.append(combined_segment)
        window_meta.append({
            "window_id": w_id,
            "sequence_index": i
        })
        
    # Re-order columns
    all_cols = list(df_merged.columns)
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "voice_available", "binary_stress"]
    
    feat_cols = [c for c in all_cols if c not in meta_keys]
    ordered_cols = meta_keys + sorted(feat_cols)
    df_merged = df_merged[ordered_cols]
    
    # Save outputs
    df_merged.to_parquet(data_dir / "combined_windows.parquet")
    np.save(data_dir / "combined_sequences.npy", np.array(combined_seqs, dtype=np.float32))
    pd.DataFrame(window_meta).to_parquet(data_dir / "combined_sequences_index.parquet")
    
    print(f"{dataset_name} Merged: Windows shape {df_merged.shape}, Sequences shape {np.shape(combined_seqs)}")
    return df_merged.shape, np.shape(combined_seqs)

def main():
    base_dir = Path(__file__).resolve().parents[3]
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    wesad_out = base_dir / "pipeline" / "data" / "wesad"
    
    log_file = base_dir / "pipeline" / "logs" / "merge_extraction.log"
    if log_file.exists():
        log_file.unlink()
        
    # 1. Merge StressID
    sid_w_shape, sid_s_shape = merge_dataset_features("StressID", sid_out, ref_dir=sid_out, has_face=True, has_voice=True)
    
    # 2. Merge EmpathicSchool
    es_w_shape, es_s_shape = merge_dataset_features("EmpathicSchool", es_out, ref_dir=sid_out, has_face=True, has_voice=False)
    
    # 3. Merge WESAD
    wesad_w_shape, wesad_s_shape = merge_dataset_features("WESAD", wesad_out, ref_dir=sid_out, has_face=False, has_voice=False)
    
    # Log results
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"StressID combined windows shape: {sid_w_shape}\n")
        f_log.write(f"StressID combined sequences shape: {sid_s_shape}\n")
        f_log.write(f"EmpathicSchool combined windows shape: {es_w_shape}\n")
        f_log.write(f"EmpathicSchool combined sequences shape: {es_s_shape}\n")
        f_log.write(f"WESAD combined windows shape: {wesad_w_shape}\n")
        f_log.write(f"WESAD combined sequences shape: {wesad_s_shape}\n")
        
    # Self-verification check
    # All datasets must have 368 columns, sequence shape (30, 72)
    issues = []
    for shape, seq_shape, name in [(sid_w_shape, sid_s_shape, "StressID"), 
                                   (es_w_shape, es_s_shape, "EmpathicSchool"), 
                                   (wesad_w_shape, wesad_s_shape, "WESAD")]:
        if shape[1] != 368:
            issues.append(f"{name} combined columns mismatch: expected 368, got {shape[1]}")
        if seq_shape[2] != 72:
            issues.append(f"{name} sequence features mismatch: expected 72, got {seq_shape[2]}")
            
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("Feature merge verification PASSED.")

if __name__ == "__main__":
    main()
