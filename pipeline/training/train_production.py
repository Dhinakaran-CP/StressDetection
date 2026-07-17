import os
import pickle
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from pipeline.common.determinism import set_determinism

# Set determinism
set_determinism()

def train_production_model(dataset_name, data_dir, model_save_path, has_voice=False):
    print(f"Training production model for {dataset_name}...")
    pq_path = data_dir / "normalized_windows.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Normalized parquet missing at {pq_path}")
        
    df = pd.read_parquet(pq_path)
    
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "binary_stress"]
    if has_voice:
        meta_keys.insert(6, "voice_available")
        
    feat_cols = [c for c in df.columns if c not in meta_keys]
    
    X = df[feat_cols].values
    y = df["binary_stress"].values
    
    model = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X, y)
    
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "model": model,
        "feature_cols": feat_cols,
        "dataset_name": dataset_name,
        "has_voice": has_voice
    }
    
    with open(model_save_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"Saved {dataset_name} production model to {model_save_path}")
    return model_save_path

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    
    prod_dir = base_dir / "pipeline" / "models" / "production"
    
    # 1. Train StressID Production Model
    train_production_model(
        "StressID", 
        sid_out, 
        prod_dir / "stressid_production.pkl", 
        has_voice=True
    )
    
    # 2. Train EmpathicSchool Production Model
    train_production_model(
        "EmpathicSchool", 
        es_out, 
        prod_dir / "empathicschool_production.pkl", 
        has_voice=False
    )
    
    print("Production training completed successfully.")

if __name__ == "__main__":
    main()
