import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def eval_model(dataset_name, pq_path, model_path):
    if not pq_path.exists() or not model_path.exists():
        print(f"Missing files for {dataset_name}")
        return
        
    df = pd.read_parquet(pq_path)
    
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
        
    model = payload["model"]
    feat_cols = payload["feature_cols"]
    
    X = df[feat_cols].values
    y = df["binary_stress"].values
    
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, average='macro', zero_division=0)
    auc = roc_auc_score(y, probs)
    
    print(f"\n--- Production Model Train-Set Fitting Verification ({dataset_name}) ---")
    print(f"[NOTE: These are fitting sanity checks on train data, NOT generalizable metrics]")
    print(f"Fitting Accuracy:  {acc:.4f}")
    print(f"Fitting Precision: {prec:.4f}")
    print(f"Fitting Recall:    {rec:.4f}")
    print(f"Fitting F1-Score:  {f1:.4f}")
    print(f"Fitting AUC-ROC:   {auc:.4f}")

def main():
    base_dir = Path(__file__).resolve().parents[3]
    
    # StressID
    eval_model(
        "StressID",
        base_dir / "pipeline" / "data" / "stressid" / "normalized_windows.parquet",
        base_dir / "pipeline" / "models" / "production" / "stressid_production.pkl"
    )
    
    # EmpathicSchool
    eval_model(
        "EmpathicSchool",
        base_dir / "pipeline" / "data" / "empathicschool" / "normalized_windows.parquet",
        base_dir / "pipeline" / "models" / "production" / "empathicschool_production.pkl"
    )
    
    # Combined
    eval_model(
        "Combined",
        base_dir / "pipeline" / "data" / "combined" / "normalized_windows.parquet",
        base_dir / "pipeline" / "models" / "production" / "combined_production.pkl"
    )

if __name__ == "__main__":
    main()
