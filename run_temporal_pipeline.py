import os
import sys
import time
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Directory Setup
# ---------------------------------------------------------
backend_dir = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(backend_dir, "pipeline_runs")
os.makedirs(os.path.join(PIPELINE_DIR, "configs"), exist_ok=True)
os.makedirs(os.path.join(PIPELINE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PIPELINE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(PIPELINE_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(PIPELINE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(PIPELINE_DIR, "reports"), exist_ok=True)
for scale in ["2sec", "5sec", "10sec"]:
    os.makedirs(os.path.join(PIPELINE_DIR, "outputs", scale), exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Hardware Acceleration Device: {DEVICE}")

# Set seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ---------------------------------------------------------
# Step 2: Temporal Deep Architectures
# ---------------------------------------------------------
class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Last time step
        return self.fc(out)

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        # Input shape: [batch, seq_len, input_dim]
        # Conv1d expects [batch, channels, seq_len]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, 32]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class TemporalTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        # 2-layer dilated TCN block
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(input_dim, hidden_dim, 1) if input_dim != hidden_dim else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        res = self.proj(x)
        
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        # Crop padding to preserve temporal causality
        x = x[:, :, :res.size(2)]
        
        x = self.dropout2(self.relu(self.bn2(self.conv2(x))))
        x = x[:, :, :res.size(2)]
        
        out = self.relu(x + res)
        out = out.mean(dim=2)  # Global average pooling
        return self.fc(out)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 5, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Temporal pooling
        return self.fc(x)

# ---------------------------------------------------------
# Step 3: Modality Slice Helper (factor=2 for dual representation)
# ---------------------------------------------------------
def get_modality_slices(df, dual=True):
    suffix = "" if not dual else "_abs"
    
    eye_cols = [f"face_ear_mean{suffix}"]
    mouth_cols = [f"face_mar_mean{suffix}"]
    gface_cols = [f"face_brow_mean{suffix}"] + [f"face_deep_embed_{i}{suffix}" for i in range(1, 513) if f"face_deep_embed_{i}{suffix}" in df.columns]
    prosody_cols = [f"voice_rms_mean{suffix}", f"voice_zcr_mean{suffix}", f"voice_pitch_mean{suffix}", f"voice_pitch_std{suffix}"]
    spectral_cols = [f"voice_mfcc_{i}{suffix}" for i in range(1, 14) if f"voice_mfcc_{i}{suffix}" in df.columns]
    quality_cols = [f"quality_score{suffix}", f"face_confidence{suffix}", f"physio_continuity_flag{suffix}"]
    cardio_cols = [f"ecg_hr{suffix}", f"ecg_mean{suffix}", f"ecg_std{suffix}", f"eda_tonic_mean{suffix}", f"eda_phasic_mean{suffix}"]
    motion_cols = [f"resp_rate_mean{suffix}", f"resp_std{suffix}"]
    
    def safe_slice(cols, target_dim):
        existing = [c for c in cols if c in df.columns]
        if len(existing) == 0:
            return np.zeros((len(df), target_dim), dtype=np.float32)
        arr = df[existing].fillna(0).values
        if arr.shape[1] < target_dim:
            pad = np.zeros((len(df), target_dim - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        return arr[:, :target_dim]

    factor = 2 if dual else 1
    eye_arr = safe_slice(eye_cols, 5 * factor)
    mouth_arr = safe_slice(mouth_cols, 3 * factor)
    gface_arr = safe_slice(gface_cols, 8 * factor)
    prosody_arr = safe_slice(prosody_cols, 3 * factor)
    spectral_arr = safe_slice(spectral_cols, 2 * factor)
    quality_arr = safe_slice(quality_cols, 5 * factor)
    cardio_arr = safe_slice(cardio_cols, 3 * factor)
    motion_arr = safe_slice(motion_cols, 1 * factor)
    
    return eye_arr, mouth_arr, gface_arr, prosody_arr, spectral_arr, quality_arr, cardio_arr, motion_arr

# ---------------------------------------------------------
# Step 4: PyTorch Sequence Datasets
# ---------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def make_sequences(arr, seq_len=5):
    N, D = arr.shape
    seqs = []
    for i in range(N):
        if i < seq_len - 1:
            pad = np.repeat(arr[i:i+1], seq_len - 1 - i, axis=0)
            seq = np.vstack([pad, arr[0:i+1]])
        else:
            seq = arr[i - seq_len + 1 : i + 1]
        seqs.append(seq)
    return np.array(seqs)

# ---------------------------------------------------------
# Step 5: PyTorch Deep Learning Training Helper (with Early Stopping)
# ---------------------------------------------------------
def train_deep_model(model_fn, train_seqs, train_labels, val_seqs, val_labels, epochs=30, batch_size=256, patience=10):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = SequenceDataset(train_seqs, train_labels)
    val_dataset = SequenceDataset(val_seqs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    history_train_loss, history_val_loss = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x.to(DEVICE))
            loss = criterion(logits, batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)
            
        epoch_loss /= len(train_dataset)
        history_train_loss.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x.to(DEVICE))
                loss = criterion(logits, batch_y.to(DEVICE))
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_dataset)
        history_val_loss.append(val_loss)
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, history_train_loss, history_val_loss

def eval_deep_model(model, seqs):
    model.eval()
    dataset = SequenceDataset(seqs, np.zeros(len(seqs)))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            
    return np.array(all_preds), np.array(all_probs)

# ---------------------------------------------------------
# Step 6: Main Pipeline Loop
# ---------------------------------------------------------
def main():
    # Setup XGBoost GPU fallback
    xgb_clf = None
    if HAS_XGBOOST:
        try:
            xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="hist", device="cuda")
            xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
            print("[INFO] XGBoost running on GPU.")
        except Exception:
            try:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="gpu_hist")
                xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
                print("[INFO] XGBoost running on GPU (gpu_hist).")
            except Exception:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                print("[INFO] XGBoost running on CPU.")
    else:
        xgb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        print("[INFO] Using sklearn Gradient Boosting.")

    scales_configs = [
        ("2sec", "stress_features_fusion_2s.csv"),
        ("5sec", "stress_features_fusion_5s.csv"),
        ("10sec", "stress_features_fusion_10s.csv")
    ]
    
    global_results = []
    
    for scale, filename in scales_configs:
        print(f"\n==========================================================")
        print(f"  RUNNING TIMEFRAME SCALE: {scale} ({filename})")
        print(f"==========================================================\n")
        
        # Load features from organized results folder
        file_path = os.path.join(backend_dir, "loso_evaluation_results", filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(backend_dir, filename)
            if not os.path.exists(file_path):
                print(f"[WARNING] Feature store {filename} not found. Skipping scale {scale}.")
                continue
                
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        
        subj_list = df["subject_id"].unique().tolist()
        labels = df["label"].astype(int).values
        subjects = df["subject_id"].values
        
        # Split features for tabular classical models
        exclude_cols = ["subject_id", "task_id", "window_index", "label"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_abs")]
        
        # Extract multimodal sequence slices (factor=2 for dual calibration representation)
        eye, mouth, gface, prosody, spectral, quality, cardio, motion = get_modality_slices(df, dual=True)
        
        # Early concatenate the modal arrays to construct sequence inputs of shape [batch, timesteps, features]
        seq_len = 5
        seq_eye = make_sequences(eye, seq_len)
        seq_mouth = make_sequences(mouth, seq_len)
        seq_gface = make_sequences(gface, seq_len)
        seq_prosody = make_sequences(prosody, seq_len)
        seq_spectral = make_sequences(spectral, seq_len)
        seq_quality = make_sequences(quality, seq_len)
        seq_cardio = make_sequences(cardio, seq_len)
        seq_motion = make_sequences(motion, seq_len)
        
        # Concatenate modal slices along the last dimension (total features = 60)
        X_sequence = np.concatenate([
            seq_eye, seq_mouth, seq_gface, seq_prosody, seq_spectral, seq_quality, seq_cardio, seq_motion
        ], axis=-1)
        
        # Standard 5-Fold GroupKFold validation to avoid subject leakage
        cv = GroupKFold(n_splits=5)
        splits = list(cv.split(df, labels, groups=subjects))
        
        # Save split config
        config_path = os.path.join(PIPELINE_DIR, "configs", f"config_{scale}.json")
        with open(config_path, "w") as f:
            json.dump({
                "scale": scale,
                "dataset_records": len(df),
                "features_count": len(feature_cols),
                "sequence_features_count": X_sequence.shape[-1],
                "split_folds": 5,
                "subjects_list": subj_list
            }, f, indent=4)
            
        models_to_evaluate = {
            # Classical Models
            "LogisticRegression": ("classical", "CPU"),
            "SVM": ("classical", "CPU"),
            "RandomForest": ("classical", "CPU"),
            "XGBoost": ("classical", "CPU"),
            "KNN": ("classical", "CPU"),
            
            # Temporal Deep Models (GPU accelerated)
            "GRU": ("temporal_deep", "GPU"),
            "LSTM": ("temporal_deep", "GPU"),
            "CNN-LSTM": ("temporal_deep", "GPU"),
            "TCN": ("temporal_deep", "GPU"),
            "Transformer": ("temporal_deep", "GPU")
        }
        
        for model_name, (category, hardware) in models_to_evaluate.items():
            print(f"--> Benchmarking {model_name} ({category} on {hardware})...")
            start_time = time.time()
            
            all_targets, all_preds, all_probs = [], [], []
            fold_results = []
            
            # To plot loss curves for deep learning models
            avg_train_loss_curve, avg_val_loss_curve = [], []
            
            # Loop over cross validation splits
            fold_idx = 1
            for train_idx, val_idx in splits:
                # Strictly isolate scaling configuration to training set only (No Leakage)
                scaler_classical = StandardScaler()
                X_class_train = scaler_classical.fit_transform(df[feature_cols].iloc[train_idx].fillna(0).values)
                X_class_val = scaler_classical.transform(df[feature_cols].iloc[val_idx].fillna(0).values)
                
                # Normalize temporal sequence inputs
                seq_train = X_sequence[train_idx]
                seq_val = X_sequence[val_idx]
                
                N_tr, S_tr, F_tr = seq_train.shape
                N_va, S_va, F_va = seq_val.shape
                
                scaler_seq = StandardScaler()
                seq_train_flat = scaler_seq.fit_transform(seq_train.reshape(-1, F_tr))
                seq_val_flat = scaler_seq.transform(seq_val.reshape(-1, F_va))
                
                seq_train_norm = seq_train_flat.reshape(N_tr, S_tr, F_tr)
                seq_val_norm = seq_val_flat.reshape(N_va, S_va, F_va)
                
                t_fold = labels[val_idx]
                
                if hardware == "CPU":
                    # Instantiation of classical models
                    if model_name == "LogisticRegression":
                        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    elif model_name == "SVM":
                        clf = SVC(probability=True, class_weight="balanced", max_iter=2000, cache_size=2000)
                    elif model_name == "RandomForest":
                        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1)
                    elif model_name == "XGBoost":
                        clf = xgb_clf
                    elif model_name == "KNN":
                        clf = KNeighborsClassifier(n_neighbors=5)
                        
                    clf.fit(X_class_train, labels[train_idx])
                    preds = clf.predict(X_class_val)
                    probs = clf.predict_proba(X_class_val)[:, 1]
                    
                    # Classic models do not have epochs, save flat zero loss curves
                    avg_train_loss_curve = [0, 0]
                    avg_val_loss_curve = [0, 0]
                else:
                    # Instantiation of Deep temporal architectures
                    input_dim = F_tr
                    if model_name == "GRU":
                        make_fn = lambda: TemporalGRU(input_dim=input_dim)
                    elif model_name == "LSTM":
                        make_fn = lambda: TemporalLSTM(input_dim=input_dim)
                    elif model_name == "CNN-LSTM":
                        make_fn = lambda: CNNLSTMModel(input_dim=input_dim)
                    elif model_name == "TCN":
                        make_fn = lambda: TemporalTCN(input_dim=input_dim)
                    elif model_name == "Transformer":
                        make_fn = lambda: TemporalTransformer(input_dim=input_dim)
                        
                    model, tr_loss, va_loss = train_deep_model(
                        make_fn, seq_train_norm, labels[train_idx],
                        seq_val_norm, labels[val_idx],
                        epochs=25, batch_size=256, patience=10
                    )
                    preds, probs = eval_deep_model(model, seq_val_norm)
                    
                    # Accumulate loss history
                    if len(avg_train_loss_curve) == 0:
                        avg_train_loss_curve = tr_loss
                        avg_val_loss_curve = va_loss
                    else:
                        # Align length if early stopping fired early
                        min_len = min(len(avg_train_loss_curve), len(tr_loss))
                        avg_train_loss_curve = [a + b for a, b in zip(avg_train_loss_curve[:min_len], tr_loss[:min_len])]
                        avg_val_loss_curve = [a + b for a, b in zip(avg_val_loss_curve[:min_len], va_loss[:min_len])]
                
                # Compute fold metrics
                fold_acc = accuracy_score(t_fold, preds)
                fold_f1 = f1_score(t_fold, preds, average="binary", zero_division=0)
                try:
                    fold_auc = roc_auc_score(t_fold, probs)
                except ValueError:
                    fold_auc = 0.5
                    
                fold_results.append({
                    "Fold": fold_idx,
                    "Accuracy": fold_acc,
                    "F1-Score": fold_f1,
                    "ROC-AUC": fold_auc
                })
                
                all_targets.extend(t_fold)
                all_preds.extend(preds)
                all_probs.extend(probs)
                fold_idx += 1
                
            elapsed_time = time.time() - start_time
            
            # Complete dataset calculations
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            
            acc = accuracy_score(all_targets, all_preds)
            prec = precision_score(all_targets, all_preds, average="binary", zero_division=0)
            rec = recall_score(all_targets, all_preds, average="binary", zero_division=0)
            f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
            bal_acc = balanced_accuracy_score(all_targets, all_preds)
            try:
                auc = roc_auc_score(all_targets, all_probs)
            except ValueError:
                auc = 0.5
                
            # Create model folder
            model_out_dir = os.path.join(PIPELINE_DIR, "outputs", scale, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            
            # Save CSV reports
            pd.DataFrame({
                "Actual": all_targets,
                "Predicted": all_preds,
                "Probability": all_probs
            }).to_csv(os.path.join(model_out_dir, "predictions.csv"), index=False)
            
            pd.DataFrame(fold_results).to_csv(os.path.join(model_out_dir, "fold_results.csv"), index=False)
            
            pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced-Accuracy", "ROC-AUC", "Runtime-Seconds"],
                "Value": [acc, prec, rec, f1, bal_acc, auc, elapsed_time]
            }).to_csv(os.path.join(model_out_dir, "metrics.csv"), index=False)
            
            # Write configuration
            with open(os.path.join(model_out_dir, "config.json"), "w") as cfg_f:
                json.dump({
                    "model_name": model_name,
                    "scale": scale,
                    "category": category,
                    "execution_hardware": hardware,
                    "input_dimensions": X_sequence.shape[-1] if hardware == "GPU" else len(feature_cols),
                    "optimizer": "Adam" if hardware == "GPU" else "N/A",
                    "learning_rate": 0.001 if hardware == "GPU" else "N/A"
                }, cfg_f, indent=4)
                
            # Plot Confusion Matrix
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Calm', 'Stress'])
            plt.yticks(tick_marks, ['Calm', 'Stress'])
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "confusion_matrix.png"))
            plt.close()
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "roc_curve.png"))
            plt.close()
            
            # Plot Precision-Recall Curve
            pr_y, pr_x, _ = precision_recall_curve(all_targets, all_probs)
            plt.figure()
            plt.plot(pr_x, pr_y, color='darkgreen', lw=2, label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "precision_recall_curve.png"))
            plt.close()
            
            # Plot Learning Curve
            plt.figure()
            if hardware == "GPU" and len(avg_train_loss_curve) > 0:
                # Divide by fold count (5) to compute average loss
                plt.plot([l / 5.0 for l in avg_train_loss_curve], label='Train Loss')
                plt.plot([l / 5.0 for l in avg_val_loss_curve], label='Val Loss')
                plt.xlabel('Epochs')
                plt.ylabel('CrossEntropy Loss')
            else:
                plt.plot([0, 1], [0, 0], label='N/A (No epochs for classical models)')
            plt.title(f'Learning Curve - {model_name}')
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "learning_curve.png"))
            plt.close()
            
            # Plot Fold Metrics (Bar Chart)
            df_folds = pd.DataFrame(fold_results)
            plt.figure()
            x_idx = np.arange(len(df_folds))
            width = 0.25
            plt.bar(x_idx - width, df_folds["Accuracy"], width, label='Accuracy')
            plt.bar(x_idx, df_folds["F1-Score"], width, label='F1-Score')
            plt.bar(x_idx + width, df_folds["ROC-AUC"], width, label='ROC-AUC')
            plt.xticks(x_idx, [f'Fold {i+1}' for i in range(len(df_folds))])
            plt.ylabel('Score')
            plt.title(f'Fold metrics - {model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "fold_metrics.png"))
            plt.close()
            
            # Plot Class Distribution (Bar Chart)
            plt.figure()
            unique_c, count_c = np.unique(all_targets, return_counts=True)
            plt.bar(['Calm (0)', 'Stress (1)'], count_c, color=['green', 'red'], width=0.5)
            plt.ylabel('Sample Count')
            plt.title('Target Class Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "class_distribution.png"))
            plt.close()
            
            # Plot Calibration Curve
            prob_true, prob_pred = calibration_curve(all_targets, all_probs, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "calibration_curve.png"))
            plt.close()
            
            # Save Model Scorecard
            with open(os.path.join(model_out_dir, "summary.md"), "w") as sc_f:
                sc_f.write(f"# Benchmarking Scorecard: {model_name} ({scale})\n\n")
                sc_f.write(f"* **Model Type:** {category}\n")
                sc_f.write(f"* **Execution Hardware:** {hardware}\n")
                sc_f.write(f"* **Runtime (seconds):** {elapsed_time:.2f} s\n\n")
                sc_f.write(f"### Performance Metrics\n")
                sc_f.write(f"| Metric | Score |\n")
                sc_f.write(f"| --- | --- |\n")
                sc_f.write(f"| Accuracy | {acc:.4f} |\n")
                sc_f.write(f"| Precision | {prec:.4f} |\n")
                sc_f.write(f"| Recall | {rec:.4f} |\n")
                sc_f.write(f"| F1-Score | {f1:.4f} |\n")
                sc_f.write(f"| Balanced Accuracy | {bal_acc:.4f} |\n")
                sc_f.write(f"| ROC-AUC | {auc:.4f} |\n")
                
            global_results.append({
                "Scale": scale,
                "Model Name": model_name,
                "Category": category,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "Runtime-Seconds": elapsed_time
            })
            
    # Save comparison reports
    df_global = pd.DataFrame(global_results)
    df_global = df_global.sort_values(by=["Scale", "Accuracy"], ascending=[True, False]).reset_index(drop=True)
    df_global.to_csv(os.path.join(PIPELINE_DIR, "metrics", "pipeline_leaderboard.csv"), index=False)
    
    report_path = os.path.join(PIPELINE_DIR, "reports", "comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# Master Model Pipeline Comparison Report\n\n")
        f.write("This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures.\n\n")
        
        # Build markdown table manually to avoid tabulate dependency
        headers = list(df_global.columns)
        md_table = "| " + " | ".join(headers) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df_global.iterrows():
            vals = []
            for val in row:
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            md_table += "| " + " | ".join(vals) + " |\n"
            
        f.write(md_table)
        f.write("\n\n*All plots and detailed reports have been categorized into the outputs/ directory.*")
        
    print(f"\n[SUCCESS] Pipeline runs complete. Consolidated report saved to: {report_path}")

if __name__ == "__main__":
    main()
