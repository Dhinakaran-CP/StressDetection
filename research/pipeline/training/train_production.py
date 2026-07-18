import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from pipeline.common.determinism import set_determinism

# Set determinism
set_determinism()

def train_production_model(dataset_name, data_dir, model_save_path, has_voice=False, model_type="lightgbm"):
    print(f"Training production model ({model_type}) for {dataset_name}...")
    pq_path = data_dir / "normalized_windows.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Normalized parquet missing at {pq_path}")
        
    df = pd.read_parquet(pq_path)
    
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "voice_available", "binary_stress"]
    if has_voice:
        meta_keys.insert(6, "voice_available")
    else:
        if "voice_available" in df.columns:
            meta_keys.insert(6, "voice_available")
        
    feat_cols = [c for c in df.columns if c not in meta_keys]
    
    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0.0) # Sklearn RF requires zero-imputed NaNs
    y = df["binary_stress"].values
    
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
    else:
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        spw = n_neg / (n_pos + 1e-8)
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=spw)
        
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
        
    print(f"Saved {dataset_name} production model ({model_type}) to {model_save_path}")
    return model_save_path

def main():
    base_dir = Path(__file__).resolve().parents[3]
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    wesad_out = base_dir / "pipeline" / "data" / "wesad"
    combined_out = base_dir / "pipeline" / "data" / "combined"
    
    prod_dir = base_dir / "pipeline" / "models" / "production"
    backend_dir = base_dir / "webapp" / "models" / "backend_selected"
    
    # 1. Train StressID Production Model (LightGBM)
    train_production_model(
        "StressID", 
        sid_out, 
        prod_dir / "stressid_production.pkl", 
        has_voice=True,
        model_type="lightgbm"
    )
    
    # 2. Train EmpathicSchool Production Model (LightGBM)
    train_production_model(
        "EmpathicSchool", 
        es_out, 
        prod_dir / "empathicschool_production.pkl", 
        has_voice=False,
        model_type="lightgbm"
    )
    
    # 3. Train WESAD Production Model (LightGBM)
    train_production_model(
        "WESAD",
        wesad_out,
        prod_dir / "wesad_production.pkl",
        has_voice=False,
        model_type="lightgbm"
    )
    
    # 4. Train Combined 91-subject Production Model (LightGBM)
    train_production_model(
        "Combined",
        combined_out,
        prod_dir / "combined_production.pkl",
        has_voice=True,
        model_type="lightgbm"
    )
    
    # 5. Train Random Forest Backend Candidates directly into webapp/models/backend_selected/
    train_production_model(
        "StressID",
        sid_out,
        backend_dir / "stressid_production_rf.pkl",
        has_voice=True,
        model_type="random_forest"
    )
    train_production_model(
        "EmpathicSchool",
        es_out,
        backend_dir / "empathicschool_production_rf.pkl",
        has_voice=False,
        model_type="random_forest"
    )
    train_production_model(
        "WESAD",
        wesad_out,
        backend_dir / "wesad_production_rf.pkl",
        has_voice=False,
        model_type="random_forest"
    )
    train_production_model(
        "Combined",
        combined_out,
        backend_dir / "combined_production_rf.pkl",
        has_voice=True,
        model_type="random_forest"
    )
    
    print("Production training completed successfully.")

if __name__ == "__main__":
    main()
