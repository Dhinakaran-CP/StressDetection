import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json
from pipeline.models.professional import HybridMoEAttentionModel

# Suppress warnings
warnings.filterwarnings('ignore')

# Set determinism
set_determinism()

# Force CUDNN determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model Zoo execution device: {DEVICE}")

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

# Softmax Focal Loss for 2-class softmax models
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        log_probs = torch.log_softmax(inputs, dim=1)
        probs = torch.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        nll_loss = self.nll(log_probs, targets)
        loss = nll_loss * ((1.0 - p_t) ** self.gamma)
        return loss.mean()

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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Custom Dataset for PyTorch sequence loading
class SeqMultimodalDataset(Dataset):
    def __init__(self, seq_tensors, labels, subjects=None):
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.subjects = torch.LongTensor(subjects) if subjects is not None else None
        self.seq_tensors = torch.FloatTensor(seq_tensors)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {
            "x": self.seq_tensors[idx],
            "label": self.labels[idx]
        }
        if self.subjects is not None:
            item["subject"] = self.subjects[idx]
        return item

def train_pytorch_model(model_name, model, X_train, y_train, train_subs_idx, epochs=8, batch_size=256, alpha=0.25, pos_rate=0.42):
    model.to(DEVICE)
    model.train()
    
    # Impute NaNs for training
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    
    # Build dataset
    if model_name in ["mlp", "temporal"]:
        train_dataset = SeqMultimodalDataset(X_train_clean, y_train)
    else: # Professional models
        train_dataset = SeqMultimodalDataset(X_train_clean, y_train, subjects=train_subs_idx)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    if model_name in ["mlp", "temporal"]:
        criterion_stress = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        weights = torch.FloatTensor([pos_rate, 1.0 - pos_rate]).to(DEVICE)
        criterion_stress = SoftmaxFocalLoss(weight=weights, gamma=2.0)
        
    criterion_subj = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            batch_x = batch["x"].to(DEVICE)
            targets = batch["label"].to(DEVICE)
            
            if model_name == "ssvb_casa_ais":
                subjects = batch["subject"].to(DEVICE)
                
                # Slice inputs for sub-modality experts
                eye = batch_x[:, :, 0:10]
                mouth = batch_x[:, :, 10:16]
                gface = batch_x[:, :, 16:34]
                prosody = batch_x[:, :, 34:41]
                spectral = batch_x[:, :, 41:46]
                quality = batch_x[:, :, 46:58]
                cardio = batch_x[:, :, 58:66]
                motion = batch_x[:, :, 66:72]
                
                stress_logits, subj_logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                loss = criterion_stress(stress_logits, targets.squeeze(1).long()) + 0.02 * criterion_subj(subj_logits, subjects)
            elif model_name == "vbc_casa_is":
                eye = batch_x[:, :, 0:10]
                mouth = batch_x[:, :, 10:16]
                gface = batch_x[:, :, 16:34]
                prosody = batch_x[:, :, 34:41]
                spectral = batch_x[:, :, 41:46]
                quality = batch_x[:, :, 46:58]
                cardio = batch_x[:, :, 58:66]
                motion = batch_x[:, :, 66:72]
                
                stress_logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                loss = criterion_stress(stress_logits, targets.squeeze(1).long())
            else: # mlp or temporal
                pred = model(batch_x)
                if pred.dim() == 3:
                    pred = pred.squeeze(-1)
                loss = criterion_stress(pred, targets)
                
            loss.backward()
            optimizer.step()

def predict_pytorch_model(model_name, model, X_test):
    model.eval()
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    test_tensor = torch.FloatTensor(X_test_clean).to(DEVICE)
    
    with torch.no_grad():
        if model_name == "ssvb_casa_ais":
            eye = test_tensor[:, :, 0:10]
            mouth = test_tensor[:, :, 10:16]
            gface = test_tensor[:, :, 16:34]
            prosody = test_tensor[:, :, 34:41]
            spectral = test_tensor[:, :, 41:46]
            quality = test_tensor[:, :, 46:58]
            cardio = test_tensor[:, :, 58:66]
            motion = test_tensor[:, :, 66:72]
            stress_logits, _ = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
            probs = torch.softmax(stress_logits, dim=1)[:, 1].cpu().numpy().flatten()
        elif model_name == "vbc_casa_is":
            eye = test_tensor[:, :, 0:10]
            mouth = test_tensor[:, :, 10:16]
            gface = test_tensor[:, :, 16:34]
            prosody = test_tensor[:, :, 34:41]
            spectral = test_tensor[:, :, 41:46]
            quality = test_tensor[:, :, 46:58]
            cardio = test_tensor[:, :, 58:66]
            motion = test_tensor[:, :, 66:72]
            stress_logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
            probs = torch.softmax(stress_logits, dim=1)[:, 1].cpu().numpy().flatten()
        else:
            stress_logits = model(test_tensor)
            probs = torch.sigmoid(stress_logits).cpu().numpy().flatten()
            
    return probs

# Threshold Tuning Grid-Search using out-of-fold inner CV
def optimize_threshold(model_type, X_train, y_train, train_subjects, make_model_fn=None, is_seq=False, alpha=0.25, pos_rate=0.42):
    from sklearn.model_selection import GroupKFold
    inner_gkf = GroupKFold(n_splits=3)
    
    oof_probs = []
    oof_labels = []
    
    # Identify unique subjects for index mapping in inner splits
    unique_subs = sorted(list(set(train_subjects)))
    sub_to_idx = {s: i for i, s in enumerate(unique_subs)}
    
    for inner_train_idx, inner_val_idx in inner_gkf.split(X_train, y_train, groups=train_subjects):
        y_inner_tr = y_train[inner_train_idx]
        y_inner_val = y_train[inner_val_idx]
        
        if not is_seq:
            # Flatten flat features scaler fitting
            inner_scaler = StandardScaler()
            X_tr_flat = np.nan_to_num(X_train[inner_train_idx], nan=0.0)
            X_val_flat = np.nan_to_num(X_train[inner_val_idx], nan=0.0)
            
            X_tr_flat_scaled = inner_scaler.fit_transform(X_tr_flat)
            X_val_flat_scaled = inner_scaler.transform(X_val_flat)
            
            if model_type == "logistic_regression":
                inner_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, class_weight='balanced')
                inner_model.fit(X_tr_flat_scaled, y_inner_tr)
                probs = inner_model.predict_proba(X_val_flat_scaled)[:, 1]
            elif model_type == "lightgbm":
                n_pos = sum(y_inner_tr)
                n_neg = len(y_inner_tr) - n_pos
                spw = n_neg / (n_pos + 1e-8)
                inner_model = lgb.LGBMClassifier(n_estimators=30, random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=spw)
                inner_model.fit(X_train[inner_train_idx], y_inner_tr)
                probs = inner_model.predict_proba(X_train[inner_val_idx])[:, 1]
            elif model_type == "xgb":
                n_pos = sum(y_inner_tr)
                n_neg = len(y_inner_tr) - n_pos
                spw = n_neg / (n_pos + 1e-8)
                inner_model = XGBClassifier(n_estimators=30, random_state=42, n_jobs=-1, verbosity=0, scale_pos_weight=spw)
                inner_model.fit(X_train[inner_train_idx], y_inner_tr)
                probs = inner_model.predict_proba(X_train[inner_val_idx])[:, 1]
            elif model_type == "rf":
                inner_model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1, class_weight='balanced')
                inner_model.fit(X_train[inner_train_idx], y_inner_tr)
                probs = inner_model.predict_proba(X_train[inner_val_idx])[:, 1]
            elif model_type == "mlp":
                inner_model = make_model_fn()
                # MLP expects flat input, but PyTorch training expects SeqDataset
                # Wrap flat inputs to add pseudo sequence dim [B, 1, D]
                X_tr_seq = X_tr_flat_scaled[:, np.newaxis, :]
                X_val_seq = X_val_flat_scaled[:, np.newaxis, :]
                train_pytorch_model(model_type, inner_model, X_tr_seq, y_inner_tr, None, epochs=3, alpha=alpha, pos_rate=pos_rate)
                probs = predict_pytorch_model(model_type, inner_model, X_val_seq)
        else:
            # Sequence features scaling
            N_tr, T, D = X_train[inner_train_idx].shape
            seq_scaler = StandardScaler()
            X_tr_flat_seq = X_train[inner_train_idx].reshape(-1, D)
            X_tr_flat_seq_scaled = seq_scaler.fit_transform(X_tr_flat_seq)
            X_tr_seq_scaled = X_tr_flat_seq_scaled.reshape(N_tr, T, D)
            
            N_val, _, _ = X_train[inner_val_idx].shape
            X_val_flat_seq = X_train[inner_val_idx].reshape(-1, D)
            X_val_flat_seq_scaled = seq_scaler.transform(X_val_flat_seq)
            X_val_seq_scaled = X_val_flat_seq_scaled.reshape(N_val, T, D)
            
            inner_model = make_model_fn()
            inner_train_subs = train_subjects[inner_train_idx]
            inner_subs_idx = np.array([sub_to_idx.get(s, 0) for s in inner_train_subs])
            
            train_pytorch_model(model_type, inner_model, X_tr_seq_scaled, y_inner_tr, inner_subs_idx, epochs=3, alpha=alpha, pos_rate=pos_rate)
            probs = predict_pytorch_model(model_type, inner_model, X_val_seq_scaled)
            
        oof_probs.extend(probs)
        oof_labels.extend(y_inner_val)
        
    best_f1 = -1.0
    best_thresh = 0.5
    oof_probs = np.array(oof_probs)
    oof_labels = np.array(oof_labels)
    
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (oof_probs >= thresh).astype(int)
        f1 = f1_score(oof_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle single class edge cases in evaluation fold
    if len(np.unique(y_true)) < 2:
        auc_score = 0.5
        pr_auc = 0.0
    else:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_score = 0.5
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(rec, prec)
        except Exception:
            pr_auc = 0.0
            
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "roc_auc": auc_score,
        "pr_auc": pr_auc
    }

def train_and_eval_loso(dataset_name, data_dir, folds):
    print(f"\n==========================================")
    print(f"--- Training on {dataset_name} ---")
    print(f"==========================================")
    
    # Load combined features Parquet
    df = pd.read_parquet(data_dir / "normalized_windows.parquet")
    seq_data = np.load(data_dir / "normalized_sequences.npy")
    
    if seq_data.shape[-1] == 48:
        print(f"[{dataset_name}] Sequence has 48 channels. Padding voice channel to 72 channels on the fly.")
        N_s, T_s, _ = seq_data.shape
        f_s = seq_data[:, :, :34]
        p_s = seq_data[:, :, 34:]
        v_s = np.zeros((N_s, T_s, 24), dtype=np.float32)
        seq_data = np.concatenate([f_s, v_s, p_s], axis=-1)
        
    pos_rate = float(np.mean(df["binary_stress"].values))
    alpha_loss = 1.0 - pos_rate
    print(f"[{dataset_name}] Positive class rate: {pos_rate:.4f} (Imbalance {1.0/pos_rate:.2f}:1)")
    print(f"[{dataset_name}] Focal Loss alpha: {alpha_loss:.4f}")
    
    meta_keys = ["subject_id", "dataset_source", "task_name", "window_id", "face_available", "physio_available", "voice_available", "binary_stress"]
    feat_cols = [c for c in df.columns if c not in meta_keys]
    
    model_types = ["logistic_regression", "lightgbm", "xgb", "rf", "mlp", "temporal", "vbc_casa_is", "ssvb_casa_ais"]
    results = {m: [] for m in model_types}
    
    unique_subs = sorted(list(df["subject_id"].unique()))
    
    for fold in tqdm(folds, desc=f"LOSO Folds ({dataset_name})"):
        test_sub = fold["test_subject"]
        
        train_idx = df[df["subject_id"] != test_sub].index.values
        test_idx = df[df["subject_id"] == test_sub].index.values
        
        if len(test_idx) == 0:
            continue
            
        y_train = df.loc[train_idx, "binary_stress"].values
        y_test = df.loc[test_idx, "binary_stress"].values
        
        X_train_flat = df.loc[train_idx, feat_cols].values
        X_test_flat = df.loc[test_idx, feat_cols].values
        
        X_train_seq = seq_data[train_idx]
        X_test_seq = seq_data[test_idx]
        
        train_subjects = df.loc[train_idx, "subject_id"].values
        
        # Mapping subject strings to indices for adversarial model
        subj_to_idx = {s: idx for idx, s in enumerate(sorted(list(set(train_subjects))))}
        train_subs_idx = np.array([subj_to_idx[s] for s in train_subjects])
        num_subjects = len(subj_to_idx)
        
        # Scale flat features strictly on train split
        scaler = StandardScaler()
        X_tr_flat_clean = np.nan_to_num(X_train_flat, nan=0.0)
        X_te_flat_clean = np.nan_to_num(X_test_flat, nan=0.0)
        
        X_tr_flat_scaled = scaler.fit_transform(X_tr_flat_clean)
        X_te_flat_scaled = scaler.transform(X_te_flat_clean)
        
        # Scale sequence features channel-wise on train split
        N_tr, T, D = X_train_seq.shape
        seq_scaler = StandardScaler()
        X_tr_flat_seq = X_train_seq.reshape(-1, D)
        X_tr_flat_seq_scaled = seq_scaler.fit_transform(X_tr_flat_seq)
        X_train_seq_scaled = X_tr_flat_seq_scaled.reshape(N_tr, T, D)
        
        N_te, _, _ = X_test_seq.shape
        X_te_flat_seq = X_test_seq.reshape(-1, D)
        X_te_flat_seq_scaled = seq_scaler.transform(X_te_flat_seq)
        X_test_seq_scaled = X_te_flat_seq_scaled.reshape(N_te, T, D)
        
        # 1. Logistic Regression
        try:
            opt_t = optimize_threshold("logistic_regression", X_train_flat, y_train, train_subjects, alpha=alpha_loss, pos_rate=pos_rate)
            lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, class_weight='balanced')
            lr_model.fit(X_tr_flat_scaled, y_train)
            lr_prob = lr_model.predict_proba(X_te_flat_scaled)[:, 1]
            results["logistic_regression"].append(compute_metrics(y_test, lr_prob, threshold=opt_t))
        except Exception as e:
            print(f"LR failed for {test_sub}: {e}")
            
        # 2. LightGBM
        try:
            opt_t = optimize_threshold("lightgbm", X_train_flat, y_train, train_subjects, alpha=alpha_loss, pos_rate=pos_rate)
            n_pos = sum(y_train)
            n_neg = len(y_train) - n_pos
            spw = n_neg / (n_pos + 1e-8)
            lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=spw)
            lgb_model.fit(X_train_flat, y_train)
            lgb_prob = lgb_model.predict_proba(X_test_flat)[:, 1]
            results["lightgbm"].append(compute_metrics(y_test, lgb_prob, threshold=opt_t))
        except Exception as e:
            print(f"LGBM failed for {test_sub}: {e}")
            
        # 3. XGBoost
        try:
            opt_t = optimize_threshold("xgb", X_train_flat, y_train, train_subjects, alpha=alpha_loss, pos_rate=pos_rate)
            n_pos = sum(y_train)
            n_neg = len(y_train) - n_pos
            spw = n_neg / (n_pos + 1e-8)
            xgb_model = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbosity=0, scale_pos_weight=spw)
            xgb_model.fit(X_train_flat, y_train)
            xgb_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
            results["xgb"].append(compute_metrics(y_test, xgb_prob, threshold=opt_t))
        except Exception as e:
            print(f"XGB failed for {test_sub}: {e}")
            
        # 4. Random Forest
        try:
            opt_t = optimize_threshold("rf", X_train_flat, y_train, train_subjects, alpha=alpha_loss, pos_rate=pos_rate)
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
            rf_model.fit(X_train_flat, y_train)
            rf_prob = rf_model.predict_proba(X_test_flat)[:, 1]
            results["rf"].append(compute_metrics(y_test, rf_prob, threshold=opt_t))
        except Exception as e:
            print(f"RF failed for {test_sub}: {e}")
            
        # 5. MLP Model
        try:
            make_model_fn = lambda: MLPModel(input_dim=len(feat_cols))
            opt_t = optimize_threshold("mlp", X_train_flat, y_train, train_subjects, make_model_fn=make_model_fn, alpha=alpha_loss, pos_rate=pos_rate)
            mlp_model = make_model_fn()
            
            # Warp 2D flat scaled data to pseudo sequence format [B, 1, D]
            X_tr_seq = X_tr_flat_scaled[:, np.newaxis, :]
            X_te_seq = X_te_flat_scaled[:, np.newaxis, :]
            
            train_pytorch_model("mlp", mlp_model, X_tr_seq, y_train, None, epochs=8, alpha=alpha_loss, pos_rate=pos_rate)
            mlp_prob = predict_pytorch_model("mlp", mlp_model, X_te_seq)
            results["mlp"].append(compute_metrics(y_test, mlp_prob, threshold=opt_t))
        except Exception as e:
            print(f"MLP failed for {test_sub}: {e}")
            
        # 6. Temporal CNN-GRU Model
        try:
            make_model_fn = lambda: TemporalModel(feature_dim=D)
            opt_t = optimize_threshold("temporal", X_train_seq, y_train, train_subjects, make_model_fn=make_model_fn, is_seq=True, alpha=alpha_loss, pos_rate=pos_rate)
            temp_model = make_model_fn()
            train_pytorch_model("temporal", temp_model, X_train_seq_scaled, y_train, None, epochs=8, alpha=alpha_loss, pos_rate=pos_rate)
            temp_prob = predict_pytorch_model("temporal", temp_model, X_test_seq_scaled)
            results["temporal"].append(compute_metrics(y_test, temp_prob, threshold=opt_t))
        except Exception as e:
            print(f"Temporal failed for {test_sub}: {e}")
            
        # 7. VBC-CASA-IS (HybridMoEAttentionModel without adversarial Head)
        try:
            make_model_fn = lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=num_subjects, adversarial=False, dual_representation=False)
            opt_t = optimize_threshold("vbc_casa_is", X_train_seq, y_train, train_subjects, make_model_fn=make_model_fn, is_seq=True, alpha=alpha_loss, pos_rate=pos_rate)
            vbc_model = make_model_fn()
            train_pytorch_model("vbc_casa_is", vbc_model, X_train_seq_scaled, y_train, None, epochs=8, alpha=alpha_loss, pos_rate=pos_rate)
            vbc_prob = predict_pytorch_model("vbc_casa_is", vbc_model, X_test_seq_scaled)
            results["vbc_casa_is"].append(compute_metrics(y_test, vbc_prob, threshold=opt_t))
        except Exception as e:
            print(f"VBC failed for {test_sub}: {e}")
            
        # 8. SSVB-CASA-AIS (HybridMoEAttentionModel with GRL adversarial Head)
        try:
            make_model_fn = lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=num_subjects, adversarial=True, dual_representation=False)
            opt_t = optimize_threshold("ssvb_casa_ais", X_train_seq, y_train, train_subjects, make_model_fn=make_model_fn, is_seq=True, alpha=alpha_loss, pos_rate=pos_rate)
            ssvb_model = make_model_fn()
            train_pytorch_model("ssvb_casa_ais", ssvb_model, X_train_seq_scaled, y_train, train_subs_idx, epochs=8, alpha=alpha_loss, pos_rate=pos_rate)
            ssvb_prob = predict_pytorch_model("ssvb_casa_ais", ssvb_model, X_test_seq_scaled)
            results["ssvb_casa_ais"].append(compute_metrics(y_test, ssvb_prob, threshold=opt_t))
        except Exception as e:
            print(f"SSVB failed for {test_sub}: {e}")
            
    summary = {}
    for m in model_types:
        metrics_df = pd.DataFrame(results[m])
        summary[m] = {
            "accuracy": float(metrics_df["accuracy"].mean()),
            "balanced_accuracy": float(metrics_df["balanced_accuracy"].mean()),
            "precision": float(metrics_df["precision"].mean()),
            "recall": float(metrics_df["recall"].mean()),
            "f1": float(metrics_df["f1"].mean()),
            "roc_auc": float(metrics_df["roc_auc"].mean()),
            "pr_auc": float(metrics_df["pr_auc"].mean()),
            "f1_std": float(metrics_df["f1"].std())
        }
        
    return results, summary

def main():
    base_dir = Path(__file__).resolve().parents[3]
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    combined_out = base_dir / "pipeline" / "data" / "combined"
    
    splits_path = base_dir / "pipeline" / "logs" / "loso_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"LOSO split registry missing at {splits_path}")
        
    splits = read_json(splits_path)
    sid_folds = splits["datasets"]["stressid"]["folds"]
    es_folds = splits["datasets"]["empathicschool"]["folds"]
    combined_folds = splits["datasets"]["combined"]["folds"]
    
    # Checkpoint paths
    sid_chk = base_dir / "pipeline" / "logs" / "checkpoint_stressid.json"
    es_chk = base_dir / "pipeline" / "logs" / "checkpoint_empathicschool.json"
    combined_chk = base_dir / "pipeline" / "logs" / "checkpoint_combined.json"
    
    # 1. StressID model training
    if sid_chk.exists():
        print(f"\n[INFO] Loading cached StressID results from checkpoint...")
        checkpoint = read_json(sid_chk)
        sid_results = checkpoint["fold_details"]
        sid_summary = checkpoint["summary"]
    else:
        sid_results, sid_summary = train_and_eval_loso("StressID", sid_out, sid_folds)
        write_json({"summary": sid_summary, "fold_details": sid_results}, sid_chk)
        
    # 2. EmpathicSchool model training
    if es_chk.exists():
        print(f"\n[INFO] Loading cached EmpathicSchool results from checkpoint...")
        checkpoint = read_json(es_chk)
        es_results = checkpoint["fold_details"]
        es_summary = checkpoint["summary"]
    else:
        es_results, es_summary = train_and_eval_loso("EmpathicSchool", es_out, es_folds)
        write_json({"summary": es_summary, "fold_details": es_results}, es_chk)
        
    # 3. Combined 95-subject model training
    if combined_chk.exists():
        print(f"\n[INFO] Loading cached Combined results from checkpoint...")
        checkpoint = read_json(combined_chk)
        combined_results = checkpoint["fold_details"]
        combined_summary = checkpoint["summary"]
    else:
        combined_results, combined_summary = train_and_eval_loso("Combined", combined_out, combined_folds)
        write_json({"summary": combined_summary, "fold_details": combined_results}, combined_chk)
        
    report = {
        "datasets": {
            "stressid": {
                "summary": sid_summary,
                "fold_details": sid_results
            },
            "empathicschool": {
                "summary": es_summary,
                "fold_details": es_results
            },
            "combined": {
                "summary": combined_summary,
                "fold_details": combined_results
            }
        }
    }
    
    report_path = base_dir / "pipeline" / "logs" / "model_zoo_metrics.json"
    write_json(report, report_path)
    
    # Clean up checkpoints on successful final completion
    for chk in [sid_chk, es_chk, combined_chk]:
        if chk.exists():
            try:
                chk.unlink()
            except Exception:
                pass
    
    # Print summary tables
    print("\n=== Model Zoo Performance Summaries ===")
    for ds_name, sum_data in [("StressID", sid_summary), ("EmpathicSchool", es_summary), ("Combined", combined_summary)]:
        print(f"\nDataset: {ds_name}")
        print(f"{'Model Archetype':<20} | {'Acc':<6} | {'Bal Acc':<7} | {'Recall':<6} | {'F1-Score':<8} | {'AUC-ROC':<7} | {'PR-AUC':<6}")
        print("-" * 80)
        for m_name, metrics in sum_data.items():
            print(f"{m_name:<20} | {metrics['accuracy']:<6.4f} | {metrics['balanced_accuracy']:<7.4f} | {metrics['recall']:<6.4f} | {metrics['f1']:<8.4f} | {metrics['roc_auc']:<7.4f} | {metrics['pr_auc']:<6.4f}")
            
    print("\nModel zoo training completed successfully.")

if __name__ == "__main__":
    main()
