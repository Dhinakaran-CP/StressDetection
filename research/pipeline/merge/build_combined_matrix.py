import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json

set_determinism()

def main():
    base_dir = Path(__file__).resolve().parents[3]
    sid_dir = base_dir / "pipeline" / "data" / "stressid"
    es_dir = base_dir / "pipeline" / "data" / "empathicschool"
    wesad_dir = base_dir / "pipeline" / "data" / "wesad"
    combined_dir = base_dir / "pipeline" / "data" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading original unimodal parquets to resolve columns...")
    meta_cols = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "voice_available", "binary_stress"]
    
    df_face_ref = pd.read_parquet(sid_dir / "face_windows.parquet")
    face_cols = sorted([c for c in df_face_ref.columns if c not in meta_cols])
    
    df_voice_ref = pd.read_parquet(sid_dir / "voice_windows.parquet")
    voice_cols = sorted([c for c in df_voice_ref.columns if c not in meta_cols])
    
    df_physio_sid_ref = pd.read_parquet(sid_dir / "physio_windows.parquet")
    physio_cols_sid = sorted([c for c in df_physio_sid_ref.columns if c not in meta_cols])
    
    df_physio_es_ref = pd.read_parquet(es_dir / "physio_windows.parquet")
    physio_cols_es = sorted([c for c in df_physio_es_ref.columns if c not in meta_cols])
    
    # Combined distinct physio signals
    all_physio_cols = sorted(list(set(physio_cols_sid + physio_cols_es)))
    
    target_features = face_cols + voice_cols + all_physio_cols
    ordered_cols = [c for c in meta_cols if c != "voice_available"] + ["voice_available"] + target_features
    
    print(f"Verified feature counts: Face={len(face_cols)}, Voice={len(voice_cols)}, Physio={len(all_physio_cols)}")
    print(f"Total unified features count: {len(target_features)}")
    
    print("Loading normalized parquets...")
    df_sid = pd.read_parquet(sid_dir / "normalized_windows.parquet").copy()
    seq_sid = np.load(sid_dir / "normalized_sequences.npy")
    
    df_es = pd.read_parquet(es_dir / "normalized_windows.parquet").copy()
    seq_es = np.load(es_dir / "normalized_sequences.npy")
    
    df_wesad = pd.read_parquet(wesad_dir / "normalized_windows.parquet").copy()
    seq_wesad = np.load(wesad_dir / "normalized_sequences.npy")
    
    # Prefix subject IDs
    df_sid["subject_id"] = "SID_" + df_sid["subject_id"].astype(str)
    df_es["subject_id"] = "ES_" + df_es["subject_id"].astype(str)
    df_wesad["subject_id"] = "WESAD_" + df_wesad["subject_id"].astype(str)
    
    # Set dataset sources explicitly
    df_sid["dataset_source"] = "stressid"
    df_es["dataset_source"] = "empathicschool"
    df_wesad["dataset_source"] = "wesad"
    
    # Re-verify and conform columns to the target features (fill NaNs with 0.0)
    for df in [df_sid, df_es, df_wesad]:
        for col in target_features:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
                
    df_sid = df_sid[ordered_cols].copy()
    df_es = df_es[ordered_cols].copy()
    df_wesad = df_wesad[ordered_cols].copy()
    
    df_combined = pd.concat([df_sid, df_es, df_wesad], ignore_index=True)
    
    # Combine sequences and replace NaNs with 0.0
    print("Aligning and merging sequence arrays...")
    combined_seqs = np.concatenate([seq_sid, seq_es, seq_wesad], axis=0)
    combined_seqs = np.nan_to_num(combined_seqs, nan=0.0)
    
    window_meta = []
    for idx, row in df_combined.iterrows():
        window_meta.append({
            "window_id": row["window_id"],
            "sequence_index": idx
        })
        
    print("Saving combined outputs...")
    df_combined.to_parquet(combined_dir / "normalized_windows.parquet")
    np.save(combined_dir / "normalized_sequences.npy", combined_seqs)
    pd.DataFrame(window_meta).to_parquet(combined_dir / "combined_sequences_index.parquet")
    
    contract = {
        "features": {
            "face": len(face_cols),
            "voice": len(voice_cols),
            "physio": len(all_physio_cols),
            "total_flat": len(target_features)
        },
        "names": {
            "face_cols": face_cols,
            "voice_cols": voice_cols,
            "physio_cols": all_physio_cols
        }
    }
    write_json(contract, base_dir / "pipeline" / "config" / "feature_contract.json")
    print(f"Combined {len(df_combined['subject_id'].unique())}-subject matrix generated successfully. Total windows: {len(df_combined)}")

if __name__ == "__main__":
    main()
