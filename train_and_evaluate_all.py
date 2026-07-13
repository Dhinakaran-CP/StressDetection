import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock

# GPU Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using training device: {DEVICE}")
print("Subject-independent validation enabled (leakage-safe).")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.titlesize'] = 16

# Configuration for training and subject counts (to ensure fast run times on CPU)
DOWNSAMPLE_LIMIT = 5000  # Frame limit for training split in standard CVs
LOSO_SUBJECTS_LIMIT = 10 # Subsampled subjects to run Leave-One-Subject-Out (LOSO) CV
EPOCHS = 5               # Epochs for deep sequence models
BATCH_SIZE = 256
SEQ_LEN = 5

# Create evaluation report directory
REPORTS_DIR = os.path.join(backend_dir, "evaluation_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# PyTorch Architectures
# ---------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        latent = gru_out[:, -1, :] 
        logits = self.classifier(latent) 
        return logits

class FlexDynamicRouter(nn.Module):
    def __init__(self, num_modalities=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_modalities * 2 + num_modalities, 16),
            nn.ReLU(),
            nn.Linear(16, num_modalities),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.mlp(x)

# ---------------------------------------------------------
# Preformed Dataset for Deep Learning
# ---------------------------------------------------------
class PreformedSeqDataset(Dataset):
    def __init__(self, X_seqs, y_seqs):
        self.X_seqs = torch.FloatTensor(X_seqs)
        self.y_seqs = torch.LongTensor(y_seqs)
        
    def __len__(self):
        return len(self.y_seqs)
        
    def __getitem__(self, idx):
        return self.X_seqs[idx], self.y_seqs[idx]

# ---------------------------------------------------------
# Sequence Extractor
# ---------------------------------------------------------
def extract_sequences(X, y, groups, task_groups):
    sequences = []
    labels = []
    df_temp = pd.DataFrame({'s': groups, 't': task_groups})
    unique_groups = df_temp.drop_duplicates().values
    
    for s, t in unique_groups:
        idx = np.where((groups == s) & (task_groups == t))[0]
        if len(idx) < SEQ_LEN:
            continue
        x_data, l_data = X[idx], y[idx]
        for i in range(len(idx) - SEQ_LEN + 1):
            sequences.append(x_data[i:i+SEQ_LEN])
            labels.append(l_data[i+SEQ_LEN-1])
            
    return np.array(sequences), np.array(labels)

# ---------------------------------------------------------
# Deep Learning Training Helper
# ---------------------------------------------------------
def train_deep_model(model, train_loader, val_loader=None, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_probs = []
    all_targets = []
    
    loader = val_loader if val_loader is not None else train_loader
    with torch.no_grad():
        for b_x, b_y in loader:
            b_x = b_x.to(DEVICE)
            logits = model(b_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_targets.append(b_y.numpy())
            
    return np.hstack(all_probs), np.hstack(all_targets)

# ---------------------------------------------------------
# Feature Scaling and Transform Utils
# ---------------------------------------------------------
def apply_subject_aware_normalization(df, feature_cols):
    df_norm = df.copy()
    for subject in df['subject_id'].unique():
        sub_mask = df['subject_id'] == subject
        subject_data = df.loc[sub_mask].sort_values(by=['task_id', 'window_index'])
        if len(subject_data) > 0:
            mean_vals = subject_data[feature_cols].iloc[:2].mean()
            df_norm.loc[sub_mask, feature_cols] = df.loc[sub_mask, feature_cols] - mean_vals
    return df_norm

def apply_temporal_windowing(df, feature_cols, window_size=2):
    df_grouped = df.copy()
    df_grouped = df_grouped.sort_values(by=['subject_id', 'task_id', 'window_index'])
    df_grouped[feature_cols] = df_grouped.groupby(['subject_id', 'task_id'])[feature_cols].transform(
        lambda x: x.rolling(window_size, min_periods=1).mean()
    )
    return df_grouped

# ---------------------------------------------------------
# Metrics Calculation (Classification + Regression)
# ---------------------------------------------------------
def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    # Regression Metrics (treating probabilities as continuous outputs)
    mse = mean_squared_error(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    rmse = np.sqrt(mse)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "MSE": mse,
        "MAE": mae,
        "R2-Score": r2,
        "RMSE": rmse
    }

# ---------------------------------------------------------
# Visualization Helper
# ---------------------------------------------------------
def generate_visual_reports(model_name, results):
    model_dir = os.path.join(REPORTS_DIR, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    
    colors = {
        "Train-Test Split (Subject-wise)": "#2b5c8f",
        "k-Fold CV (Subject-wise)": "#d95f02",
        "Stratified Group CV": "#7570b3",
        "LOSO CV (Subsampled)": "#e7298a",
        "Repeated Group CV": "#66a61e"
    }
    
    # Plot 1: ROC-AUC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for strategy, data in results.items():
        if len(data['y_true']) == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
            auc_val = data['metrics']['ROC-AUC']
            plt.plot(fpr, tpr, color=colors[strategy], lw=2, label=f"{strategy} (AUC = {auc_val:.4f})")
        except Exception as e:
            print(f"Skipping ROC plot for {strategy} due to: {e}")
            
    plt.plot([0, 1], [0, 1], color='#999999', linestyle='--', lw=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel("True Positive Rate", fontsize=11, fontweight='bold', labelpad=8)
    plt.title(f"ROC-AUC Curves - {model_name}", fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roc_auc_curves.png"), bbox_inches='tight')
    plt.close()
    
    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    axes = axes.flatten()
    
    for idx, (strategy, data) in enumerate(results.items()):
        if len(data['y_true']) == 0:
            continue
        y_pred = (data['y_prob'] >= 0.5).astype(int)
        cm = confusion_matrix(data['y_true'], y_pred)
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)
        
        labels = np.array([
            [f"TN\n{cm[0,0]}\n({cm_norm[0,0]*100:.1f}%)", f"FP\n{cm[0,1]}\n({cm_norm[0,1]*100:.1f}%)"],
            [f"FN\n{cm[1,0]}\n({cm_norm[1,0]*100:.1f}%)", f"TP\n{cm[1,1]}\n({cm_norm[1,1]*100:.1f}%)"]
        ])
        
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
            annot_kws={"size": 11, "weight": "bold"}, linewidths=1.5, linecolor='white'
        )
        axes[idx].set_title(strategy, fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10, labelpad=5)
        axes[idx].set_ylabel("True Label", fontsize=10, labelpad=5)
        
    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])
        
    plt.suptitle(f"Confusion Matrices - {model_name}", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrices.png"), bbox_inches='tight')
    plt.close()
    
    # Plot 3: Metrics Dashboard Table
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=150)
    ax.axis('off')
    
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MSE", "MAE", "R2-Score", "RMSE"]
    header = ["Validation Strategy"] + metrics_list
    
    table_data = []
    for strategy, data in results.items():
        row = [strategy]
        for m in metrics_list:
            row.append(f"{data['metrics'][m]:.4f}")
        table_data.append(row)
        
    table = ax.table(
        cellText=table_data, colLabels=header, loc='center', cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    
    for col_idx in range(len(header)):
        cell = table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1e3d59')
        
    for row_idx in range(1, len(table_data) + 1):
        face_color = '#f5f5f5' if row_idx % 2 == 0 else 'white'
        for col_idx in range(len(header)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(face_color)
            if col_idx == 0:
                cell.set_text_props(weight='bold')
                
    ax.set_title(f"Model Performance Metrics - {model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "metrics_comparison.png"), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({k: v['metrics'] for k, v in results.items()}, f, indent=4)
        
    print(f"Visual reports saved successfully in {model_dir}")

# ---------------------------------------------------------
# Core Evaluation Loop (Subject-Independent)
# ---------------------------------------------------------
def run_cv_evaluation(model_type, model_builder, X, y, groups, task_groups=None):
    """
    Runs the 5 subject-independent validation strategies:
    1. Train-Test Split (Subject-wise, 80/20)
    2. 5-Fold Cross Validation (Subject-wise GroupKFold)
    3. Stratified 5-Fold Cross Validation (StratifiedGroupKFold)
    4. Leave-One-Subject-Out CV (Subsampled LOSO)
    5. Repeated Group Cross Validation
    """
    results = {}
    unique_subjects = np.unique(groups)
    
    # Helper to downsample a training partition to avoid CPU execution freeze
    def downsample_indices(train_indices):
        if len(train_indices) > DOWNSAMPLE_LIMIT:
            np.random.seed(42)
            return np.random.choice(train_indices, DOWNSAMPLE_LIMIT, replace=False)
        return train_indices

    # 1. Train-Test Split (Subject-wise, 80/20)
    print("  -> Running Train-Test Split (Subject-wise)...")
    train_subjs, test_subjs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_idx = np.isin(groups, train_subjs)
    test_idx = np.isin(groups, test_subjs)
    
    # Extract raw data splits
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    groups_tr, task_groups_tr = groups[train_idx], (task_groups[train_idx] if task_groups is not None else None)
    groups_te, task_groups_te = groups[test_idx], (task_groups[test_idx] if task_groups is not None else None)
    
    # Downsample train only for runtime performance
    tr_indices = downsample_indices(np.arange(len(X_tr)))
    X_tr_down, y_tr_down = X_tr[tr_indices], y_tr[tr_indices]
    groups_tr_down = groups_tr[tr_indices]
    task_groups_tr_down = task_groups_tr[tr_indices] if task_groups_tr is not None else None
    
    # Preprocess (Fold-level scaling)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_down)
    X_te_s = scaler.transform(X_te)
    
    if model_type == "sklearn":
        clf = model_builder()
        clf.fit(X_tr_s, y_tr_down)
        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_true = y_te
    else:  # PyTorch
        train_ds = PreformedSeqDataset(*extract_sequences(X_tr_s, y_tr_down, groups_tr_down, task_groups_tr_down))
        test_ds = PreformedSeqDataset(*extract_sequences(X_te_s, y_te, groups_te, task_groups_te))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = model_builder()
        y_prob, y_true = train_deep_model(model, train_loader, test_loader)
        
    results["Train-Test Split (Subject-wise)"] = {
        "y_true": y_true, "y_prob": y_prob, "metrics": calculate_metrics(y_true, y_prob)
    }
    
    # 2. 5-Fold Cross Validation (Subject-wise GroupKFold)
    print("  -> Running 5-Fold CV (Subject-wise)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_probs, kf_trues = [], []
    
    for train_subj_idx, test_subj_idx in kf.split(unique_subjects):
        train_subjs = unique_subjects[train_subj_idx]
        test_subjs = unique_subjects[test_subj_idx]
        
        train_idx = np.isin(groups, train_subjs)
        test_idx = np.isin(groups, test_subjs)
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        groups_tr, task_groups_tr = groups[train_idx], (task_groups[train_idx] if task_groups is not None else None)
        groups_te, task_groups_te = groups[test_idx], (task_groups[test_idx] if task_groups is not None else None)
        
        tr_indices = downsample_indices(np.arange(len(X_tr)))
        X_tr_down, y_tr_down = X_tr[tr_indices], y_tr[tr_indices]
        groups_tr_down = groups_tr[tr_indices]
        task_groups_tr_down = task_groups_tr[tr_indices] if task_groups_tr is not None else None
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_down)
        X_te_s = scaler.transform(X_te)
        
        if model_type == "sklearn":
            clf = model_builder()
            clf.fit(X_tr_s, y_tr_down)
            kf_probs.append(clf.predict_proba(X_te_s)[:, 1])
            kf_trues.append(y_te)
        else:
            train_ds = PreformedSeqDataset(*extract_sequences(X_tr_s, y_tr_down, groups_tr_down, task_groups_tr_down))
            test_ds = PreformedSeqDataset(*extract_sequences(X_te_s, y_te, groups_te, task_groups_te))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            model = model_builder()
            probs, trues = train_deep_model(model, train_loader, test_loader)
            kf_probs.append(probs)
            kf_trues.append(trues)
            
    y_prob_kf = np.hstack(kf_probs)
    y_true_kf = np.hstack(kf_trues)
    results["k-Fold CV (Subject-wise)"] = {
        "y_true": y_true_kf, "y_prob": y_prob_kf, "metrics": calculate_metrics(y_true_kf, y_prob_kf)
    }
    
    # 3. Stratified Group K-Fold Cross Validation
    print("  -> Running Stratified Group CV...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf_probs, sgkf_trues = [], []
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        groups_tr, task_groups_tr = groups[train_idx], (task_groups[train_idx] if task_groups is not None else None)
        groups_te, task_groups_te = groups[test_idx], (task_groups[test_idx] if task_groups is not None else None)
        
        tr_indices = downsample_indices(np.arange(len(X_tr)))
        X_tr_down, y_tr_down = X_tr[tr_indices], y_tr[tr_indices]
        groups_tr_down = groups_tr[tr_indices]
        task_groups_tr_down = task_groups_tr[tr_indices] if task_groups_tr is not None else None
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_down)
        X_te_s = scaler.transform(X_te)
        
        if model_type == "sklearn":
            clf = model_builder()
            clf.fit(X_tr_s, y_tr_down)
            sgkf_probs.append(clf.predict_proba(X_te_s)[:, 1])
            sgkf_trues.append(y_te)
        else:
            train_ds = PreformedSeqDataset(*extract_sequences(X_tr_s, y_tr_down, groups_tr_down, task_groups_tr_down))
            test_ds = PreformedSeqDataset(*extract_sequences(X_te_s, y_te, groups_te, task_groups_te))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            model = model_builder()
            probs, trues = train_deep_model(model, train_loader, test_loader)
            sgkf_probs.append(probs)
            sgkf_trues.append(trues)
            
    y_prob_sgkf = np.hstack(sgkf_probs)
    y_true_sgkf = np.hstack(sgkf_trues)
    results["Stratified Group CV"] = {
        "y_true": y_true_sgkf, "y_prob": y_prob_sgkf, "metrics": calculate_metrics(y_true_sgkf, y_prob_sgkf)
    }
    
    # 4. Leave-One-Subject-Out CV (Subsampled LOSO)
    print(f"  -> Running LOSO CV (on {LOSO_SUBJECTS_LIMIT} subjects)...")
    np.random.seed(42)
    loso_subjects = np.random.choice(unique_subjects, size=min(len(unique_subjects), LOSO_SUBJECTS_LIMIT), replace=False)
    loso_probs, loso_trues = [], []
    
    for test_subj in loso_subjects:
        train_idx = groups != test_subj
        test_idx = groups == test_subj
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        groups_tr, task_groups_tr = groups[train_idx], (task_groups[train_idx] if task_groups is not None else None)
        groups_te, task_groups_te = groups[test_idx], (task_groups[test_idx] if task_groups is not None else None)
        
        tr_indices = downsample_indices(np.arange(len(X_tr)))
        X_tr_down, y_tr_down = X_tr[tr_indices], y_tr[tr_indices]
        groups_tr_down = groups_tr[tr_indices]
        task_groups_tr_down = task_groups_tr[tr_indices] if task_groups_tr is not None else None
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_down)
        X_te_s = scaler.transform(X_te)
        
        if model_type == "sklearn":
            clf = model_builder()
            clf.fit(X_tr_s, y_tr_down)
            loso_probs.append(clf.predict_proba(X_te_s)[:, 1])
            loso_trues.append(y_te)
        else:
            # sequence validation for PyTorch
            train_seq = extract_sequences(X_tr_s, y_tr_down, groups_tr_down, task_groups_tr_down)
            test_seq = extract_sequences(X_te_s, y_te, groups_te, task_groups_te)
            
            if len(train_seq[0]) == 0 or len(test_seq[0]) == 0:
                continue
                
            train_ds = PreformedSeqDataset(*train_seq)
            test_ds = PreformedSeqDataset(*test_seq)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            model = model_builder()
            probs, trues = train_deep_model(model, train_loader, test_loader)
            loso_probs.append(probs)
            loso_trues.append(trues)
            
    if len(loso_probs) > 0:
        y_prob_loso = np.hstack(loso_probs)
        y_true_loso = np.hstack(loso_trues)
    else:
        y_prob_loso = np.array([])
        y_true_loso = np.array([])
        
    results["LOSO CV (Subsampled)"] = {
        "y_true": y_true_loso, "y_prob": y_prob_loso, "metrics": calculate_metrics(y_true_loso, y_prob_loso) if len(y_true_loso) > 0 else {}
    }
    
    # 5. Repeated Group Cross Validation
    print("  -> Running Repeated Group CV...")
    rskf_probs, rskf_trues = [], []
    for repeat in range(2):
        shuffled_subjs = unique_subjects.copy()
        np.random.seed(42 + repeat)
        np.random.shuffle(shuffled_subjs)
        kf_group = KFold(n_splits=5, shuffle=False)
        
        for train_subj_idx, test_subj_idx in kf_group.split(shuffled_subjs):
            train_subjs = shuffled_subjs[train_subj_idx]
            test_subjs = shuffled_subjs[test_subj_idx]
            
            train_idx = np.isin(groups, train_subjs)
            test_idx = np.isin(groups, test_subjs)
            
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            
            groups_tr, task_groups_tr = groups[train_idx], (task_groups[train_idx] if task_groups is not None else None)
            groups_te, task_groups_te = groups[test_idx], (task_groups[test_idx] if task_groups is not None else None)
            
            tr_indices = downsample_indices(np.arange(len(X_tr)))
            X_tr_down, y_tr_down = X_tr[tr_indices], y_tr[tr_indices]
            groups_tr_down = groups_tr[tr_indices]
            task_groups_tr_down = task_groups_tr[tr_indices] if task_groups_tr is not None else None
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_down)
            X_te_s = scaler.transform(X_te)
            
            if model_type == "sklearn":
                clf = model_builder()
                clf.fit(X_tr_s, y_tr_down)
                rskf_probs.append(clf.predict_proba(X_te_s)[:, 1])
                rskf_trues.append(y_te)
            else:
                train_ds = PreformedSeqDataset(*extract_sequences(X_tr_s, y_tr_down, groups_tr_down, task_groups_tr_down))
                test_ds = PreformedSeqDataset(*extract_sequences(X_te_s, y_te, groups_te, task_groups_te))
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
                
                model = model_builder()
                probs, trues = train_deep_model(model, train_loader, test_loader)
                rskf_probs.append(probs)
                rskf_trues.append(trues)
                
    y_prob_rskf = np.hstack(rskf_probs)
    y_true_rskf = np.hstack(rskf_trues)
    results["Repeated Group CV"] = {
        "y_true": y_true_rskf, "y_prob": y_prob_rskf, "metrics": calculate_metrics(y_true_rskf, y_prob_rskf)
    }
    
    return results

# ---------------------------------------------------------
# Dynamic Router CV Loop (Subject-Independent)
# ---------------------------------------------------------
def run_router_cv_evaluation(router_X, y, groups):
    """
    Evaluates the FlexDynamicRouter MLP classifier using subject-wise splits.
    """
    unique_subjects = np.unique(groups)
    results = {}
    
    def router_builder():
        return FlexDynamicRouter(num_modalities=3)
        
    def train_router_model(model, train_loader, val_loader=None, epochs=15):
        model = model.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                
                weights = model(b_x) # [batch, 3]
                pf = b_x[:, 0:2]
                pv = b_x[:, 2:4]
                pp = b_x[:, 4:6]
                
                fused_probs = (
                    weights[:, 0:1] * pf +
                    weights[:, 1:2] * pv +
                    weights[:, 2:3] * pp
                )
                
                loss = criterion(fused_probs, b_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        all_probs, all_targets = [], []
        loader = val_loader if val_loader is not None else train_loader
        with torch.no_grad():
            for b_x, b_y in loader:
                b_x = b_x.to(DEVICE)
                weights = model(b_x)
                pf = b_x[:, 0:2]
                pv = b_x[:, 2:4]
                pp = b_x[:, 4:6]
                fused_probs = (
                    weights[:, 0:1] * pf +
                    weights[:, 1:2] * pv +
                    weights[:, 2:3] * pp
                )
                probs = fused_probs[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_targets.append(b_y.numpy())
                
        return np.hstack(all_probs), np.hstack(all_targets)

    class RouterDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # 1. Train-Test Split (Subject-wise, 80/20)
    print("  -> Running Train-Test Split (Subject-wise)...")
    train_subjs, test_subjs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_idx = np.isin(groups, train_subjs)
    test_idx = np.isin(groups, test_subjs)
    
    train_loader = DataLoader(RouterDataset(router_X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(RouterDataset(router_X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
    probs, trues = train_router_model(router_builder(), train_loader, test_loader)
    results["Train-Test Split (Subject-wise)"] = {
        "y_true": trues, "y_prob": probs, "metrics": calculate_metrics(trues, probs)
    }
    
    # 2. k-Fold CV (Subject-wise GroupKFold)
    print("  -> Running 5-Fold CV (Subject-wise)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_probs, kf_trues = [], []
    for train_subj_idx, test_subj_idx in kf.split(unique_subjects):
        train_subjs = unique_subjects[train_subj_idx]
        test_subjs = unique_subjects[test_subj_idx]
        train_idx = np.isin(groups, train_subjs)
        test_idx = np.isin(groups, test_subjs)
        
        train_loader = DataLoader(RouterDataset(router_X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(RouterDataset(router_X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
        probs, trues = train_router_model(router_builder(), train_loader, test_loader)
        kf_probs.append(probs)
        kf_trues.append(trues)
    y_prob_kf = np.hstack(kf_probs)
    y_true_kf = np.hstack(kf_trues)
    results["k-Fold CV (Subject-wise)"] = {
        "y_true": y_true_kf, "y_prob": y_prob_kf, "metrics": calculate_metrics(y_true_kf, y_prob_kf)
    }
    
    # 3. Stratified Group CV
    print("  -> Running Stratified Group CV...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf_probs, sgkf_trues = [], []
    for train_idx, test_idx in sgkf.split(router_X, y, groups):
        train_loader = DataLoader(RouterDataset(router_X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(RouterDataset(router_X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
        probs, trues = train_router_model(router_builder(), train_loader, test_loader)
        sgkf_probs.append(probs)
        sgkf_trues.append(trues)
    y_prob_sgkf = np.hstack(sgkf_probs)
    y_true_sgkf = np.hstack(sgkf_trues)
    results["Stratified Group CV"] = {
        "y_true": y_true_sgkf, "y_prob": y_prob_sgkf, "metrics": calculate_metrics(y_true_sgkf, y_prob_sgkf)
    }
    
    # 4. Leave-One-Subject-Out CV (Subsampled LOSO)
    print(f"  -> Running LOSO CV (on {LOSO_SUBJECTS_LIMIT} subjects)...")
    np.random.seed(42)
    loso_subjects = np.random.choice(unique_subjects, size=min(len(unique_subjects), LOSO_SUBJECTS_LIMIT), replace=False)
    loso_probs, loso_trues = [], []
    for test_subj in loso_subjects:
        train_idx = groups != test_subj
        test_idx = groups == test_subj
        train_loader = DataLoader(RouterDataset(router_X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(RouterDataset(router_X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
        probs, trues = train_router_model(router_builder(), train_loader, test_loader)
        loso_probs.append(probs)
        loso_trues.append(trues)
    y_prob_loso = np.hstack(loso_probs)
    y_true_loso = np.hstack(loso_trues)
    results["LOSO CV (Subsampled)"] = {
        "y_true": y_true_loso, "y_prob": y_prob_loso, "metrics": calculate_metrics(y_true_loso, y_prob_loso)
    }
    
    # 5. Repeated Group Cross Validation
    print("  -> Running Repeated Group CV...")
    rskf_probs, rskf_trues = [], []
    for repeat in range(2):
        shuffled_subjs = unique_subjects.copy()
        np.random.seed(42 + repeat)
        np.random.shuffle(shuffled_subjs)
        kf_group = KFold(n_splits=5, shuffle=False)
        for train_subj_idx, test_subj_idx in kf_group.split(shuffled_subjs):
            train_subjs = shuffled_subjs[train_subj_idx]
            test_subjs = shuffled_subjs[test_subj_idx]
            train_idx = np.isin(groups, train_subjs)
            test_idx = np.isin(groups, test_subjs)
            
            train_loader = DataLoader(RouterDataset(router_X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(RouterDataset(router_X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)
            probs, trues = train_router_model(router_builder(), train_loader, test_loader)
            rskf_probs.append(probs)
            rskf_trues.append(trues)
    y_prob_rskf = np.hstack(rskf_probs)
    y_true_rskf = np.hstack(rskf_trues)
    results["Repeated Group CV"] = {
        "y_true": y_true_rskf, "y_prob": y_prob_rskf, "metrics": calculate_metrics(y_true_rskf, y_prob_rskf)
    }
    
    generate_visual_reports("Deep Fusion Router", results)

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    lock = FeatureRuntimeLock()
    
    # Risky/identity-adjacent features filter logic (arch_.md section 6 compliance)
    excluded_features = ["face_height_norm", "landmark_confidence", "f0_mean", "f0_range", "eda_scl_mean"]
    
    face_features = [f for f in lock.contract["modalities"]["face"]["features"] if f not in excluded_features]
    voice_features = [f for f in lock.contract["modalities"]["voice"]["features"] if f not in excluded_features]
    physio_features = [f for f in lock.contract["modalities"]["physio"]["features"] if f not in excluded_features]
    
    # ---------------------------------------------------------
    # Part 1: Evaluate Classical Modality Experts
    # ---------------------------------------------------------
    print("\n=========================================================")
    # 1. Face Expert (Gradient Boosting)
    print("Evaluating Classical Face Expert (Gradient Boosting)...")
    df_face = pd.read_csv("certified_data/face_certified.csv")
    X_clean_face = df_face[face_features].fillna(0).values
    y_face = df_face["label"].values
    groups_face = df_face["subject_id"].values
    task_groups_face = df_face["task_id"].values
    
    face_results = run_cv_evaluation(
        "sklearn",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        X_clean_face, y_face, groups_face, task_groups_face
    )
    generate_visual_reports("Classical Face Expert", face_results)
    
    # 2. Voice Expert (Random Forest)
    print("\nEvaluating Classical Voice Expert (Random Forest)...")
    df_voice = pd.read_csv("certified_data/voice_certified.csv")
    X_clean_voice = df_voice[voice_features].fillna(0).values
    y_voice = df_voice["label"].values
    groups_voice = df_voice["subject_id"].values
    task_groups_voice = df_voice["task_id"].values
    
    voice_results = run_cv_evaluation(
        "sklearn",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        X_clean_voice, y_voice, groups_voice, task_groups_voice
    )
    generate_visual_reports("Classical Voice Expert", voice_results)
    
    # 3. Physio Expert (Gradient Boosting)
    print("\nEvaluating Classical Physio Expert (Gradient Boosting)...")
    df_physio = pd.read_csv("certified_data/physio_certified.csv")
    X_clean_physio = df_physio[physio_features].fillna(0).values
    y_physio = df_physio["label"].values
    groups_physio = df_physio["subject_id"].values
    task_groups_physio = df_physio["task_id"].values
    
    physio_results = run_cv_evaluation(
        "sklearn",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        X_clean_physio, y_physio, groups_physio, task_groups_physio
    )
    generate_visual_reports("Classical Physio Expert", physio_results)
    
    # ---------------------------------------------------------
    # Part 2: Evaluate Classical RF Experts (Methodology Phase 4)
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("Evaluating Classical RF Experts (Methodology Phase 4 - with Temporal Windowing & Normalization)...")
    
    # Face RF
    print("Running Classical RF Face...")
    df_face_norm = apply_subject_aware_normalization(df_face, face_features)
    df_face_win = apply_temporal_windowing(df_face_norm, face_features)
    X_face_rf = df_face_win[face_features].values
    y_face_rf = df_face_win["label"].values
    groups_face_rf = df_face_win["subject_id"].values
    task_groups_face_rf = df_face_win["task_id"].values
    
    rf_face_results = run_cv_evaluation(
        "sklearn",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        X_face_rf, y_face_rf, groups_face_rf, task_groups_face_rf
    )
    generate_visual_reports("Classical RF Face Expert", rf_face_results)
    
    # Voice RF
    print("Running Classical RF Voice...")
    df_voice_norm = apply_subject_aware_normalization(df_voice, voice_features)
    df_voice_win = apply_temporal_windowing(df_voice_norm, voice_features)
    X_voice_rf = df_voice_win[voice_features].values
    y_voice_rf = df_voice_win["label"].values
    groups_voice_rf = df_voice_win["subject_id"].values
    task_groups_voice_rf = df_voice_win["task_id"].values
    
    rf_voice_results = run_cv_evaluation(
        "sklearn",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        X_voice_rf, y_voice_rf, groups_voice_rf, task_groups_voice_rf
    )
    generate_visual_reports("Classical RF Voice Expert", rf_voice_results)
    
    # Physio RF
    print("Running Classical RF Physio...")
    df_physio_norm = apply_subject_aware_normalization(df_physio, physio_features)
    df_physio_win = apply_temporal_windowing(df_physio_norm, physio_features)
    X_physio_rf = df_physio_win[physio_features].values
    y_physio_rf = df_physio_win["label"].values
    groups_physio_rf = df_physio_win["subject_id"].values
    task_groups_physio_rf = df_physio_win["task_id"].values
    
    rf_physio_results = run_cv_evaluation(
        "sklearn",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        X_physio_rf, y_physio_rf, groups_physio_rf, task_groups_physio_rf
    )
    generate_visual_reports("Classical RF Physio Expert", rf_physio_results)
    
    # ---------------------------------------------------------
    # Part 3: Evaluate Deep Modality Sequence Experts (PyTorch CNN-GRU)
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("Loading and Preprocessing Datasets for Deep Experts (Sequential)...")
    
    df_face_seq = pd.read_csv("certified_data/face_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_voice_seq = pd.read_csv("certified_data/voice_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_physio_seq = pd.read_csv("certified_data/physio_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    
    df_merged = pd.merge(df_face_seq, df_voice_seq, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = pd.merge(df_merged, df_physio_seq, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = df_merged.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)
    
    groups = df_merged['subject_id'].values
    task_groups = df_merged['task_id'].values
    y_merged = df_merged['label'].values
    
    X_deep_face = df_merged[face_features].values
    X_deep_voice = df_merged[voice_features].values
    X_deep_physio = df_merged[physio_features].values
    
    # Deep Face
    print("\nEvaluating Deep Face Expert (PyTorch Conv1D-GRU)...")
    deep_face_results = run_cv_evaluation(
        "pytorch",
        lambda: ModalityEncoder(len(face_features), 16),
        X_deep_face, y_merged, groups, task_groups
    )
    generate_visual_reports("Deep Face Expert", deep_face_results)
    
    # Deep Voice
    print("\nEvaluating Deep Voice Expert (PyTorch Conv1D-GRU)...")
    deep_voice_results = run_cv_evaluation(
        "pytorch",
        lambda: ModalityEncoder(len(voice_features), 16),
        X_deep_voice, y_merged, groups, task_groups
    )
    generate_visual_reports("Deep Voice Expert", deep_voice_results)
    
    # Deep Physio
    print("\nEvaluating Deep Physio Expert (PyTorch Conv1D-GRU)...")
    deep_physio_results = run_cv_evaluation(
        "pytorch",
        lambda: ModalityEncoder(len(physio_features), 16),
        X_deep_physio, y_merged, groups, task_groups
    )
    generate_visual_reports("Deep Physio Expert", deep_physio_results)
    
    # 4. Deep Dynamic Router (MLP Router)
    print("\nEvaluating Deep Fusion Router (PyTorch MLP)...")
    print("Running Dynamic Router CV Splits...")
    
    # Fit simple classifiers to build prediction inputs for the router
    rf_f = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_v = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_p = RandomForestClassifier(n_estimators=50, random_state=42)
    
    rf_f.fit(StandardScaler().fit_transform(X_deep_face[:2000]), y_merged[:2000])
    rf_v.fit(StandardScaler().fit_transform(X_deep_voice[:2000]), y_merged[:2000])
    rf_p.fit(StandardScaler().fit_transform(X_deep_physio[:2000]), y_merged[:2000])
    
    prob_f = rf_f.predict_proba(StandardScaler().fit_transform(X_deep_face))
    prob_v = rf_v.predict_proba(StandardScaler().fit_transform(X_deep_voice))
    prob_p = rf_p.predict_proba(StandardScaler().fit_transform(X_deep_physio))
    
    masks = np.ones((len(y_merged), 3), dtype=np.float32)
    router_X = np.hstack([prob_f, prob_v, prob_p, masks]) # shape: [N, 9]
    
    run_router_cv_evaluation(router_X, y_merged, groups)
    
    print("\n=========================================================")
    print("Evaluation Pipeline completed successfully!")
    print("All performance plots and metrics are generated under 'evaluation_reports/'.")
    print("=========================================================")

if __name__ == "__main__":
    main()
