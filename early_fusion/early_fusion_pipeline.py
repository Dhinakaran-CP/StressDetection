import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, confusion_matrix
)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Configuration
EPOCHS = 8
BATCH_SIZE = 256
SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS_DIR = "early_fusion/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
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
# 1. Data Loader & Preprocessing
# ---------------------------------------------------------
EXCLUDED_FEATURES = ["face_height_norm", "landmark_confidence", "f0_mean", "f0_range", "eda_scl_mean"]

class EarlyFusionDataset(Dataset):
    def __init__(self, df, face_scaler=None, voice_scaler=None, physio_scaler=None, seq_len=5):
        self.seq_len = seq_len
        
        # Identify feature columns using certified FeatureRuntimeLock schema
        from backend.core.feature_runtime_lock import FeatureRuntimeLock
        lock = FeatureRuntimeLock()
        face_cols = [f for f in lock.contract["modalities"]["face"]["features"] if f not in EXCLUDED_FEATURES]
        voice_cols = [f for f in lock.contract["modalities"]["voice"]["features"] if f not in EXCLUDED_FEATURES]
        physio_cols = [f for f in lock.contract["modalities"]["physio"]["features"] if f not in EXCLUDED_FEATURES]
        
        # Scale
        self.face_scaler = face_scaler or StandardScaler()
        self.voice_scaler = voice_scaler or StandardScaler()
        self.physio_scaler = physio_scaler or StandardScaler()
        
        X_face_raw = df[face_cols].values
        X_voice_raw = df[voice_cols].values
        X_physio_raw = df[physio_cols].values
        
        if face_scaler is None:
            X_face = self.face_scaler.fit_transform(X_face_raw)
        else:
            X_face = self.face_scaler.transform(X_face_raw)
            
        if voice_scaler is None:
            X_voice = self.voice_scaler.fit_transform(X_voice_raw)
        else:
            X_voice = self.voice_scaler.transform(X_voice_raw)
            
        if physio_scaler is None:
            X_physio = self.physio_scaler.fit_transform(X_physio_raw)
        else:
            X_physio = self.physio_scaler.transform(X_physio_raw)
            
        self.face_feats = X_face
        self.voice_feats = X_voice
        self.physio_feats = X_physio
        
        self.labels = df['label'].values
        self.subjects = df['subject_id'].values
        self.tasks = df['task_id'].values
        
        self.sequences = []
        self.seq_labels = []
        
        df_groups = pd.DataFrame({'s': self.subjects, 't': self.tasks})
        unique_groups = df_groups.drop_duplicates().values
        
        for s, t in unique_groups:
            idx = np.where((self.subjects == s) & (self.tasks == t))[0]
            if len(idx) < self.seq_len:
                continue
            
            for i in range(len(idx) - self.seq_len + 1):
                window_idx = idx[i:i+self.seq_len]
                self.sequences.append({
                    "face": self.face_feats[window_idx],
                    "voice": self.voice_feats[window_idx],
                    "physio": self.physio_feats[window_idx]
                })
                self.seq_labels.append(self.labels[window_idx[-1]])
                
    def __len__(self):
        return len(self.seq_labels)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.FloatTensor(seq["face"]),
            torch.FloatTensor(seq["voice"]),
            torch.FloatTensor(seq["physio"]),
            torch.LongTensor([self.seq_labels[idx]])[0]
        )

def load_synchronized_data(data_dir="certified_data"):
    print("Loading certified datasets...")
    df_face = pd.read_csv(os.path.join(data_dir, "face_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_voice = pd.read_csv(os.path.join(data_dir, "voice_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_physio = pd.read_csv(os.path.join(data_dir, "physio_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    
    # Normalization
    for df in [df_face, df_voice, df_physio]:
        for col in ['subject_id', 'task_id']:
            df[col] = df[col].astype(str).str.lower().str.strip()
        df['window_index'] = df['window_index'].astype(int)
        
    # Subtract calm state averages per subject (calibration normalization)
    for df, modality, cols in [
        (df_face, 'face', ['left_ear', 'right_ear', 'avg_ear', 'blink_velocity', 'brow_descent_left', 'brow_descent_right', 'brow_asymmetry', 'lip_compression', 'jaw_tension', 'mouth_corner_pull', 'forehead_tension', 'face_height_norm', 'head_tilt', 'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio', 'landmark_confidence', 'nose_wrinkle']),
        (df_voice, 'voice', ['f0_mean', 'f0_std', 'f0_range', 'jitter_percent', 'shimmer_db', 'hnr', 'speaking_rate_proxy', 'voice_intensity', 'high_freq_ratio', 'spectral_flux', 'pause_ratio', 'voiced_fraction']),
        (df_physio, 'physio', ['ecg_rate_mean', 'ecg_hrv_rmssd', 'ecg_hrv_sdnn', 'eda_scl_mean', 'resp_rate_mean'])
    ]:
        features = [c for c in cols if c not in EXCLUDED_FEATURES]
        df[features] = df[features].fillna(0)
        for subj, subj_df in df.groupby('subject_id'):
            calm_df = subj_df[subj_df['label'] == 0]
            mean_calm = calm_df[features].mean().values if len(calm_df) > 0 else subj_df[features].mean().values
            idx = df[df['subject_id'] == subj].index
            df.loc[idx, features] = df.loc[idx, features] - mean_calm
            
    # Merge on aligned frame keys
    df_merged = pd.merge(df_face, df_voice, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = pd.merge(df_merged, df_physio, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = df_merged.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)
    
    print(f"Synchronized aligned rows: {len(df_merged)}")
    return df_merged

# ---------------------------------------------------------
# 2. Early Fusion Models
# ---------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x[:, -1, :]  # [batch, hidden_dim]

class ModalityBank(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.missing_face = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.missing_voice = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.missing_physio = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        
    def forward(self, f, v, p, f_m, v_m, p_m):
        batch_size = f.size(0)
        f_placeholder = self.missing_face.repeat(batch_size, 1)
        v_placeholder = self.missing_voice.repeat(batch_size, 1)
        p_placeholder = self.missing_physio.repeat(batch_size, 1)
        
        f_mask = f_m.view(batch_size, 1).float()
        v_mask = v_m.view(batch_size, 1).float()
        p_mask = p_m.view(batch_size, 1).float()
        
        out_f = f * f_mask + f_placeholder * (1.0 - f_mask)
        out_v = v * v_mask + v_placeholder * (1.0 - v_mask)
        out_p = p * p_mask + p_placeholder * (1.0 - p_mask)
        return out_f, out_v, out_p

class EarlyFusionClassifier(nn.Module):
    def __init__(self, face_dim=16, voice_dim=10, physio_dim=4, latent_dim=16):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 2)
        )
    def forward(self, face, voice, physio):
        return self.classifier(torch.cat([self.face_enc(face), self.voice_enc(voice), self.physio_enc(physio)], dim=1))

class GatedFusionClassifier(nn.Module):
    def __init__(self, face_dim=16, voice_dim=10, physio_dim=4, latent_dim=16):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 2)
        )
    def forward(self, face, voice, physio):
        ef = self.face_enc(face)
        ev = self.voice_enc(voice)
        ep = self.physio_enc(physio)
        w = self.gate(torch.cat([ef, ev, ep], dim=1))
        fused = w[:, 0:1] * ef + w[:, 1:2] * ev + w[:, 2:3] * ep
        return self.classifier(fused)

class CrossAttentionFusionClassifier(nn.Module):
    def __init__(self, face_dim=16, voice_dim=10, physio_dim=4, latent_dim=16):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)
        self.v = nn.Linear(latent_dim, latent_dim)
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 2)
        )
    def forward(self, face, voice, physio):
        stacked = torch.stack([self.face_enc(face), self.voice_enc(voice), self.physio_enc(physio)], dim=1)
        Q = self.q(stacked)
        K = self.k(stacked)
        V = self.v(stacked)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.latent_dim)
        attn = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn, V).view(stacked.size(0), -1)
        return self.classifier(attended)

class FlexiModalMoE(nn.Module):
    def __init__(self, face_dim=16, voice_dim=10, physio_dim=4, latent_dim=16, num_experts=4, top_k=2):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        self.modality_bank = ModalityBank(latent_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * latent_dim, 2 * latent_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2 * latent_dim, 3 * latent_dim))
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(nn.Linear(3 * latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, num_experts))
        self.classifier = nn.Sequential(nn.Linear(3 * latent_dim, latent_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(latent_dim, 2))
        self.top_k = min(top_k, num_experts)
        
    def forward(self, face, voice, physio, f_mask=None, v_mask=None, p_mask=None):
        if f_mask is None: f_mask = torch.ones(face.size(0), device=face.device)
        if v_mask is None: v_mask = torch.ones(voice.size(0), device=voice.device)
        if p_mask is None: p_mask = torch.ones(physio.size(0), device=physio.device)
        
        ef = self.face_enc(face)
        ev = self.voice_enc(voice)
        ep = self.physio_enc(physio)
        
        ef, ev, ep = self.modality_bank(ef, ev, ep, f_mask, v_mask, p_mask)
        joint = torch.cat([ef, ev, ep], dim=1)
        
        routing_logits = self.router(joint)
        probs = F.softmax(routing_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Vectorized sparse MoE gating
        mask = torch.zeros_like(probs).scatter_(1, topk_indices, 1.0)
        masked_probs = probs * mask
        gating_weights = masked_probs / (torch.sum(masked_probs, dim=1, keepdim=True) + 1e-8) # [batch_size, num_experts]
        
        # Compute all experts in parallel for the entire batch
        expert_outs = torch.stack([exp(joint) for exp in self.experts], dim=1) # [batch_size, num_experts, 3 * latent_dim]
        
        # Dynamic weighted sum of expert representations
        expert_outputs = torch.sum(gating_weights.unsqueeze(2) * expert_outs, dim=1) # [batch_size, 3 * latent_dim]
        
        return self.classifier(expert_outputs)

# ---------------------------------------------------------
# 3. Validation Harness & Report Generator
# ---------------------------------------------------------
def generate_loso_plots(model_name, y_true, y_prob, folds_metrics):
    model_dir = os.path.join(REPORTS_DIR, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(6, 5), dpi=150)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='#1b4f72', lw=2, label=f"LOSO CV (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color='#999999', linestyle='--', lw=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.title(f"ROC-AUC Curve - {model_name}", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roc_auc_curves.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(5, 4), dpi=150)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        [f"TN\n{cm[0,0]}\n({cm_norm[0,0]*100:.1f}%)", f"FP\n{cm[0,1]}\n({cm_norm[0,1]*100:.1f}%)"],
        [f"FN\n{cm[1,0]}\n({cm_norm[1,0]*100:.1f}%)", f"TP\n{cm[1,1]}\n({cm_norm[1,1]*100:.1f}%)"]
    ])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, annot_kws={"size": 11, "weight": "bold"})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrices.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Summary Table
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.axis('off')
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MSE", "MAE", "R2-Score", "RMSE"]
    header = ["Fold"] + metrics_list
    
    table_data = []
    for idx, fold in enumerate(folds_metrics):
        row = [f"Fold {idx+1}"]
        for m in metrics_list:
            row.append(f"{fold[m]:.4f}")
        table_data.append(row)
        
    mean_row = ["Mean"]
    for m in metrics_list:
        mean_row.append(f"{np.mean([f[m] for f in folds_metrics]):.4f}")
    table_data.append(mean_row)
    
    table = ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    
    for col_idx in range(len(header)):
        cell = table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1a5276')
        
    for row_idx in range(1, len(table_data) + 1):
        face_color = '#f2f4f4' if row_idx % 2 == 0 else 'white'
        if row_idx == len(table_data):
            face_color = '#d5f5e3'
        for col_idx in range(len(header)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(face_color)
            if col_idx == 0 or row_idx == len(table_data):
                cell.set_text_props(weight='bold')
                
    ax.set_title(f"Model Performance Metrics - {model_name}", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "metrics_comparison.png"), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(folds_metrics, f, indent=4)
    print(f"  -> Saved results to {model_dir}")

def evaluate_early_fusion_model(model_name, model_builder, df_merged, groups, gkf):
    print(f"\nEvaluating: {model_name}...")
    results = []
    preds_y_true = []
    preds_y_prob = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df_merged, df_merged['label'], groups)):
        print(f"  -> Fold {fold+1}/5 evaluating...")
        
        train_ds = EarlyFusionDataset(df_merged.iloc[train_idx])
        test_ds = EarlyFusionDataset(df_merged.iloc[test_idx],
                                     face_scaler=train_ds.face_scaler,
                                     voice_scaler=train_ds.voice_scaler,
                                     physio_scaler=train_ds.physio_scaler)
                                     
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = model_builder().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(EPOCHS):
            model.train()
            for b_face, b_voice, b_physio, b_y in train_loader:
                b_face, b_voice, b_physio, b_y = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(b_face, b_voice, b_physio), b_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        fold_probs = []
        fold_trues = []
        with torch.no_grad():
            for b_face, b_voice, b_physio, b_y in test_loader:
                b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
                logits = model(b_face, b_voice, b_physio)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                fold_probs.append(probs)
                fold_trues.append(b_y.numpy())
                
        fold_probs = np.hstack(fold_probs)
        fold_trues = np.hstack(fold_trues)
        results.append(calculate_metrics(fold_trues, fold_probs))
        preds_y_true.append(fold_trues)
        preds_y_prob.append(fold_probs)
        
    preds_y_true = np.hstack(preds_y_true)
    preds_y_prob = np.hstack(preds_y_prob)
    generate_loso_plots(model_name, preds_y_true, preds_y_prob, results)
    
    mean_acc = np.mean([f["Accuracy"] for f in results])
    print(f"Completed {model_name}. Average Accuracy: {mean_acc:.4f}")
    return results

def main():
    df_merged = load_synchronized_data()
    groups = df_merged['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    
    evaluate_early_fusion_model("Early Fusion Classifier", lambda: EarlyFusionClassifier(), df_merged, groups, gkf)
    evaluate_early_fusion_model("Gated Fusion Classifier", lambda: GatedFusionClassifier(), df_merged, groups, gkf)
    evaluate_early_fusion_model("Cross Attention Fusion Classifier", lambda: CrossAttentionFusionClassifier(), df_merged, groups, gkf)
    evaluate_early_fusion_model("FlexiModal MoE Classifier", lambda: FlexiModalMoE(), df_merged, groups, gkf)
    
    print("\nAll early fusion evaluation tasks completed successfully!")

if __name__ == "__main__":
    main()
