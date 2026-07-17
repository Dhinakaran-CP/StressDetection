import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set determinism
set_determinism()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model Zoo execution device: {DEVICE}")

# MLP Model Architecture
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 1D CNN-GRU Temporal Architecture
class TemporalModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Transpose back: [B, D_conv, T_conv] -> [B, T_conv, D_conv]
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_pytorch_model(model, X_train, y_train, X_test, epochs=3, batch_size=256, is_seq=False):
    model.to(DEVICE)
    model.train()
    
    # Impute NaNs with 0.0 (mean)
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    
    # Datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_clean),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
    # Inference
    model.eval()
    with torch.no_grad():
        test_x_t = torch.FloatTensor(X_test_clean).to(DEVICE)
        logits = model(test_x_t).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.flatten()

def compute_metrics(y_true, y_prob):
    # Ensure binary predictions
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Handle single class edge cases in evaluation fold
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5
            
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc
    }

def train_and_eval_loso(dataset_name, data_dir, folds):
    print(f"\n--- Training on {dataset_name} ---")
    
    # Load combined normalized flat window parquets
    df = pd.read_parquet(data_dir / "normalized_windows.parquet")
    
    # Load combined normalized sequence array
    seq_data = np.load(data_dir / "normalized_sequences.npy")
    
    # Identify non-metadata feature columns
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "binary_stress", "voice_available"]
    feat_cols = [c for c in df.columns if c not in meta_keys]
    
    # Initialize metric logs
    model_types = ["logistic_regression", "lightgbm", "mlp", "temporal"]
    results = {m: [] for m in model_types}
    
    for fold in tqdm(folds, desc=f"LOSO Folds ({dataset_name})"):
        test_sub = fold["test_subject"]
        
        # Split indexes
        train_idx = df[df["subject_id"] != test_sub].index.values
        test_idx = df[df["subject_id"] == test_sub].index.values
        
        if len(test_idx) == 0:
            continue
            
        # Target labels
        y_train = df.loc[train_idx, "binary_stress"].values
        y_test = df.loc[test_idx, "binary_stress"].values
        
        # Flat feature matrices
        X_train_flat = df.loc[train_idx, feat_cols].values
        X_test_flat = df.loc[test_idx, feat_cols].values
        
        # Sequential feature matrices
        X_train_seq = seq_data[train_idx]
        X_test_seq = seq_data[test_idx]
        
        # 1. Logistic Regression
        try:
            # Impute NaNs with 0.0 (mean) for Logistic Regression
            X_tr_lr = np.nan_to_num(X_train_flat, nan=0.0)
            X_te_lr = np.nan_to_num(X_test_flat, nan=0.0)
            
            lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
            lr_model.fit(X_tr_lr, y_train)
            lr_prob = lr_model.predict_proba(X_te_lr)[:, 1]
            results["logistic_regression"].append(compute_metrics(y_test, lr_prob))
        except Exception as e:
            print(f"LR failed for {test_sub}: {e}")
            
        # 2. LightGBM (handles NaNs natively)
        try:
            lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
            lgb_model.fit(X_train_flat, y_train)
            lgb_prob = lgb_model.predict_proba(X_test_flat)[:, 1]
            results["lightgbm"].append(compute_metrics(y_test, lgb_prob))
        except Exception as e:
            print(f"LGBM failed for {test_sub}: {e}")
            
        # 3. MLP
        try:
            mlp_model = MLPModel(input_dim=len(feat_cols))
            mlp_prob = train_pytorch_model(mlp_model, X_train_flat, y_train, X_test_flat, epochs=3, is_seq=False)
            results["mlp"].append(compute_metrics(y_test, mlp_prob))
        except Exception as e:
            print(f"MLP failed for {test_sub}: {e}")
            
        # 4. Temporal Model (CNN-GRU)
        try:
            temp_model = TemporalModel(feature_dim=X_train_seq.shape[-1])
            temp_prob = train_pytorch_model(temp_model, X_train_seq, y_train, X_test_seq, epochs=3, is_seq=True)
            results["temporal"].append(compute_metrics(y_test, temp_prob))
        except Exception as e:
            print(f"Temporal failed for {test_sub}: {e}")
            
    # Compute average metrics across folds
    summary = {}
    for m in model_types:
        metrics_df = pd.DataFrame(results[m])
        summary[m] = {
            "accuracy": float(metrics_df["accuracy"].mean()),
            "precision": float(metrics_df["precision"].mean()),
            "recall": float(metrics_df["recall"].mean()),
            "f1": float(metrics_df["f1"].mean()),
            "roc_auc": float(metrics_df["roc_auc"].mean())
        }
        
    return results, summary

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    
    splits_path = base_dir / "pipeline" / "logs" / "loso_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"LOSO split registry missing at {splits_path}")
        
    splits = read_json(splits_path)
    sid_folds = splits["datasets"]["stressid"]["folds"]
    es_folds = splits["datasets"]["empathicschool"]["folds"]
    
    # 1. StressID model training
    sid_results, sid_summary = train_and_eval_loso("StressID", sid_out, sid_folds)
    
    # 2. EmpathicSchool model training
    es_results, es_summary = train_and_eval_loso("EmpathicSchool", es_out, es_folds)
    
    # Save combined zoo report
    report = {
        "datasets": {
            "stressid": {
                "summary": sid_summary,
                "fold_details": sid_results
            },
            "empathicschool": {
                "summary": es_summary,
                "fold_details": es_results
            }
        }
    }
    
    report_path = base_dir / "pipeline" / "logs" / "model_zoo_metrics.json"
    write_json(report, report_path)
    
    # Print summary tables
    print("\n=== Model Zoo Performance Summaries ===")
    for ds_name, sum_data in [("StressID", sid_summary), ("EmpathicSchool", es_summary)]:
        print(f"\nDataset: {ds_name}")
        print(f"{'Model Archetype':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'AUC-ROC':<8}")
        print("-" * 80)
        for m_name, metrics in sum_data.items():
            print(f"{m_name:<22} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<8.4f} | {metrics['f1']:<8.4f} | {metrics['roc_auc']:<8.4f}")
            
    print("\nModel zoo training completed successfully.")

if __name__ == "__main__":
    main()
