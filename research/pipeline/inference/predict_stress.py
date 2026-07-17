import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress cosmetic warnings
warnings.filterwarnings('ignore')

def load_production_model(model_path):
    """
    Loads the trained production model and its feature columns.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Production model not found at {model_path}")
        
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
        
    return payload

def predict_stress(model_payload, df_windows):
    """
    Runs stress prediction on input windows.
    
    Parameters:
    - model_payload: dict loaded from load_production_model()
    - df_windows: pandas DataFrame containing input feature columns
    
    Returns:
    - predictions: np.ndarray of shape [N] (binary 0 or 1 labels)
    - probabilities: np.ndarray of shape [N] (float between 0 and 1)
    """
    model = model_payload["model"]
    req_features = model_payload["feature_cols"]
    
    # Check if all required features are present in the input DataFrame
    missing_cols = [c for c in req_features if c not in df_windows.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")
        
    X = df_windows[req_features].values
    
    # Run prediction
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    return preds, probs

def main():
    # Simple self-test code
    base_dir = Path(__file__).resolve().parents[3]
    model_path = base_dir / "pipeline" / "models" / "production" / "stressid_production.pkl"
    sample_pq = base_dir / "pipeline" / "data" / "stressid" / "normalized_windows.parquet"
    
    if model_path.exists() and sample_pq.exists():
        print("Running inference self-test...")
        payload = load_production_model(model_path)
        df_sample = pd.read_parquet(sample_pq).head(5)
        
        preds, probs = predict_stress(payload, df_sample)
        print("Self-test Predictions:", preds)
        print("Self-test Probabilities:", probs)
        print("Inference self-test PASSED.")

if __name__ == "__main__":
    main()
