import os
import sys
import time
import json
import torch
import argparse
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
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Directory Setup
# ---------------------------------------------------------
backend_dir = r"c:\Users\StressProject\Desktop\StressDetectionUsingML"
loso_dir = os.path.join(backend_dir, "research", "Phase_1_Baseline_LOSO")

# We run inside Phase_6_Expert_Gating folder for separation
RUN_DIR = os.path.join(backend_dir, "research", "Phase_6_Expert_Gating")
OUTPUTS_DIR = os.path.join(RUN_DIR, "outputs", "expert_pipeline")
os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "reports"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Hardware Acceleration Device: {DEVICE}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------------------------------------
# Step 2: Expert Architectures (Lightweight 1-Layer GRUs)
# ---------------------------------------------------------
class SubpartExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, out_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Last timestep hidden state
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------
# Step 3: Gating Router (Softmax weighting)
# ---------------------------------------------------------
class GatingRouter(nn.Module):
    def __init__(self, context_dim=60, num_experts=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, context):
        # context shape: [batch, context_dim]
        raw_weights = self.net(context)
        weights = self.softmax(raw_weights)
        return weights

# ---------------------------------------------------------
# Step 4: Joint Mixture-of-Experts (MoE) Pipeline Model
# ---------------------------------------------------------
class ExpertPipelineModel(nn.Module):
    def __init__(self, subpart_dims, hidden_dim=16):
        super().__init__()
        self.experts = nn.ModuleList([
            SubpartExpert(input_dim=dim, hidden_dim=hidden_dim) for dim in subpart_dims
        ])
        # The router takes the full 60-dimensional context vector at the current step
        self.router = GatingRouter(context_dim=sum(subpart_dims), num_experts=len(subpart_dims))
        
    def forward(self, subpart_inputs):
        # subpart_inputs: list of 8 tensors, each of shape [batch, seq_len, subpart_dim]
        # 1. Evaluate all experts
        expert_logits = [expert(x) for expert, x in zip(self.experts, subpart_inputs)]
        # Stack logits: [batch, num_experts, 2]
        expert_logits_stacked = torch.stack(expert_logits, dim=1)
        
        # 2. Get router weights based on current frame context (last timestep of all inputs concatenated)
        # Concatenate last timestep of all inputs along features
        context = torch.cat([x[:, -1, :] for x in subpart_inputs], dim=-1)
        weights = self.router(context)  # [batch, num_experts]
        
        # 3. Weighted sum of logits: [batch, 2]
        # unsqueeze weights to [batch, num_experts, 1] for multiplication
        fused_logits = torch.sum(expert_logits_stacked * weights.unsqueeze(-1), dim=1)
        return fused_logits, weights, expert_logits

# ---------------------------------------------------------
# Step 5: Dataset Helper for Subparts
# ---------------------------------------------------------
class SubpartsSequenceDataset(Dataset):
    def __init__(self, subpart_sequences, labels):
        # subpart_sequences: list of 8 numpy arrays, each of shape [N, seq_len, subpart_dim]
        self.subparts = [torch.FloatTensor(s) for s in subpart_sequences]
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return [s[idx] for s in self.subparts], self.labels[idx]

# Modality Slice Helper matching original code
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
# Step 6: MoE Pipeline Training Loop
# ---------------------------------------------------------
def train_moe_pipeline(model, train_loader, val_loader, epochs=25, lr=0.001, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    history_train_loss, history_val_loss = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_xs, batch_y in train_loader:
            optimizer.zero_grad()
            xs = [x.to(DEVICE) for x in batch_xs]
            logits, _, _ = model(xs)
            loss = criterion(logits, batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)
            
        epoch_loss /= len(train_loader.dataset)
        history_train_loss.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_xs, batch_y in val_loader:
                xs = [x.to(DEVICE) for x in batch_xs]
                logits, _, _ = model(xs)
                loss = criterion(logits, batch_y.to(DEVICE))
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_loader.dataset)
        history_val_loss.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    if best_model_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        
    return history_train_loss, history_val_loss

def eval_moe_pipeline(model, val_loader):
    model.eval()
    all_preds, all_probs = [], []
    all_expert_preds = [[] for _ in range(8)]
    all_expert_probs = [[] for _ in range(8)]
    all_weights = []
    
    with torch.no_grad():
        for batch_xs, _ in val_loader:
            xs = [x.to(DEVICE) for x in batch_xs]
            logits, weights, expert_logits = model(xs)
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_weights.extend(weights.cpu().numpy())
            
            for k, el in enumerate(expert_logits):
                ep = el.argmax(dim=1).cpu().numpy()
                eprobs = torch.softmax(el, dim=1)[:, 1].cpu().numpy()
                all_expert_preds[k].extend(ep)
                all_expert_probs[k].extend(eprobs)
                
    return (
        np.array(all_preds), np.array(all_probs), np.array(all_weights),
        [np.array(p) for p in all_expert_preds], [np.array(pb) for pb in all_expert_probs]
    )

# ---------------------------------------------------------
# Step 7: Execution and Evaluation
# ---------------------------------------------------------
def run_expert_scale(scale, filename):
    print(f"\n==========================================================")
    print(f"  RUNNING TIMEFRAME SCALE: {scale} ({filename})")
    print(f"==========================================================\n")
    
    file_path = os.path.join(backend_dir, "data", "features", filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(loso_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(backend_dir, filename)
            if not os.path.exists(file_path):
                print(f"[WARNING] Feature store {filename} not found. Skipping scale {scale}.")
                return []
            
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    subj_list = df["subject_id"].unique().tolist()
    labels = df["label"].astype(int).values
    subjects = df["subject_id"].values
    
    # Slice the 8 subparts
    subparts_data = get_modality_slices(df, dual=True)
    subpart_names = [
        "eye_expert", "mouth_expert", "face_embed_expert",
        "prosody_expert", "spectral_expert",
        "cardio_expert", "respiration_expert", "quality_expert"
    ]
    subpart_dims = [s.shape[1] for s in subparts_data]
    print(f"Subpart dimensions: {list(zip(subpart_names, subpart_dims))}")
    
    # Construct sequence inputs of shape [N, 5, subpart_dim]
    subparts_seq = [make_sequences(s, seq_len=5) for s in subparts_data]
    
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(df, labels, groups=subjects))
    
    scale_dir = os.path.join(OUTPUTS_DIR, scale)
    os.makedirs(scale_dir, exist_ok=True)
    
    # Global variables to accumulate cross-validation outputs
    fused_targets, fused_preds, fused_probs = [], [], []
    fused_weights = []
    
    expert_targets = [[] for _ in range(8)]
    expert_preds = [[] for _ in range(8)]
    expert_probs = [[] for _ in range(8)]
    
    fold_results = []
    avg_train_loss, avg_val_loss = [], []
    
    fold_idx = 1
    start_time = time.time()
    for train_idx, val_idx in splits:
        # Scale each feature subpart separately fit on train only
        scaled_subparts_train = []
        scaled_subparts_val = []
        
        for k in range(8):
            tr_seq = subparts_seq[k][train_idx]
            val_seq = subparts_seq[k][val_idx]
            
            N_tr, S_tr, F_tr = tr_seq.shape
            N_va, S_va, F_va = val_seq.shape
            
            scaler = StandardScaler()
            tr_flat = scaler.fit_transform(tr_seq.reshape(-1, F_tr))
            val_flat = scaler.transform(val_seq.reshape(-1, F_va))
            
            scaled_subparts_train.append(tr_flat.reshape(N_tr, S_tr, F_tr))
            scaled_subparts_val.append(val_flat.reshape(N_va, S_va, F_va))
            
        train_dataset = SubpartsSequenceDataset(scaled_subparts_train, labels[train_idx])
        val_dataset = SubpartsSequenceDataset(scaled_subparts_val, labels[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        model = ExpertPipelineModel(subpart_dims).to(DEVICE)
        
        tr_loss, val_l = train_moe_pipeline(
            model, train_loader, val_loader, epochs=25, lr=0.001, patience=10
        )
        
        # Eval
        preds, probs, weights, exp_preds, exp_probs = eval_moe_pipeline(model, val_loader)
        
        fused_targets.extend(labels[val_idx])
        fused_preds.extend(preds)
        fused_probs.extend(probs)
        fused_weights.extend(weights)
        
        for k in range(8):
            expert_targets[k].extend(labels[val_idx])
            expert_preds[k].extend(exp_preds[k])
            expert_probs[k].extend(exp_probs[k])
            
        fold_acc = accuracy_score(labels[val_idx], preds)
        fold_f1 = f1_score(labels[val_idx], preds, average="binary", zero_division=0)
        try:
            fold_auc = roc_auc_score(labels[val_idx], probs)
        except ValueError:
            fold_auc = 0.5
            
        fold_results.append({
            "Fold": fold_idx,
            "Accuracy": fold_acc,
            "F1-Score": fold_f1,
            "ROC-AUC": fold_auc
        })
        
        # Accumulate loss
        if len(avg_train_loss) == 0:
            avg_train_loss = tr_loss
            avg_val_loss = val_l
        else:
            min_l = min(len(avg_train_loss), len(tr_loss))
            avg_train_loss = [a + b for a, b in zip(avg_train_loss[:min_l], tr_loss[:min_l])]
            avg_val_loss = [a + b for a, b in zip(avg_val_loss[:min_l], val_l[:min_l])]
            
        fold_idx += 1
        
    elapsed_time = time.time() - start_time
    
    fused_targets = np.array(fused_targets)
    fused_preds = np.array(fused_preds)
    fused_probs = np.array(fused_probs)
    fused_weights = np.array(fused_weights)
    
    # 1. Compute fused metrics
    acc = accuracy_score(fused_targets, fused_preds)
    prec = precision_score(fused_targets, fused_preds, average="binary", zero_division=0)
    rec = recall_score(fused_targets, fused_preds, average="binary", zero_division=0)
    f1 = f1_score(fused_targets, fused_preds, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(fused_targets, fused_preds)
    try:
        auc = roc_auc_score(fused_targets, fused_probs)
    except ValueError:
        auc = 0.5
        
    # Create required layout folders
    fusion_dir = os.path.join(scale_dir, "fusion")
    metrics_dir = os.path.join(scale_dir, "metrics")
    plots_dir = os.path.join(scale_dir, "plots")
    reports_dir = os.path.join(scale_dir, "reports")
    
    os.makedirs(fusion_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save Model State
    torch.save(model.state_dict(), os.path.join(fusion_dir, "model.pt"))
    
    # Save fold results and predictions
    pd.DataFrame({
        "Actual": fused_targets,
        "Predicted": fused_preds,
        "Probability": fused_probs
    }).to_csv(os.path.join(fusion_dir, "predictions.csv"), index=False)
    
    pd.DataFrame(fold_results).to_csv(os.path.join(fusion_dir, "fold_results.csv"), index=False)
    
    # Compile individual expert metrics
    expert_f1_scores = []
    expert_acc_scores = []
    
    for k in range(8):
        exp_t = np.array(expert_targets[k])
        exp_p = np.array(expert_preds[k])
        exp_pb = np.array(expert_probs[k])
        
        e_acc = accuracy_score(exp_t, exp_p)
        e_prec = precision_score(exp_t, exp_p, average="binary", zero_division=0)
        e_rec = recall_score(exp_t, exp_p, average="binary", zero_division=0)
        e_f1 = f1_score(exp_t, exp_p, average="binary", zero_division=0)
        try:
            e_auc = roc_auc_score(exp_t, exp_pb)
        except ValueError:
            e_auc = 0.5
            
        expert_f1_scores.append(e_f1)
        expert_acc_scores.append(e_acc)
        
        # Save individual expert folders
        # Layout: outputs/expert_pipeline/5sec/face/eye_expert/, etc.
        if k < 3:
            exp_folder = os.path.join(scale_dir, "face", subpart_names[k])
        elif k < 5:
            exp_folder = os.path.join(scale_dir, "voice", subpart_names[k])
        else:
            exp_folder = os.path.join(scale_dir, "physio", subpart_names[k])
            
        os.makedirs(exp_folder, exist_ok=True)
        pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Value": [e_acc, e_prec, e_rec, e_f1, e_auc]
        }).to_csv(os.path.join(exp_folder, "metrics.csv"), index=False)
        
    # Write consolidated metrics CSV
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced-Accuracy", "ROC-AUC", "Runtime-Seconds"],
        "Value": [acc, prec, rec, f1, bal_acc, auc, elapsed_time]
    }).to_csv(os.path.join(metrics_dir, "metrics.csv"), index=False)
    
    # 2. REQUIRED PLOTS
    # Confusion Matrix
    cm = confusion_matrix(fused_targets, fused_preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(f'Confusion Matrix - MoE Pipeline')
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
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(fused_targets, fused_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - MoE Pipeline')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()
    
    # PR curve
    pr_y, pr_x, _ = precision_recall_curve(fused_targets, fused_probs)
    plt.figure()
    plt.plot(pr_x, pr_y, color='darkgreen', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - MoE Pipeline')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "precision_recall_curve.png"))
    plt.close()
    
    # Fold metric chart (Bar Chart)
    df_folds = pd.DataFrame(fold_results)
    plt.figure()
    x_idx = np.arange(len(df_folds))
    width = 0.25
    plt.bar(x_idx - width, df_folds["Accuracy"], width, label='Accuracy')
    plt.bar(x_idx, df_folds["F1-Score"], width, label='F1-Score')
    plt.bar(x_idx + width, df_folds["ROC-AUC"], width, label='ROC-AUC')
    plt.xticks(x_idx, [f'Fold {i+1}' for i in range(len(df_folds))])
    plt.ylabel('Score')
    plt.title(f'Fold metrics - MoE Pipeline')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fold_metrics.png"))
    plt.close()
    
    # Router Weight Distribution (Bar Chart)
    mean_weights = np.mean(fused_weights, axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(subpart_names, mean_weights, color='teal', width=0.5)
    plt.ylabel('Average Router Weight')
    plt.title('Gating Router Weight Distribution')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "router_weight_distribution.png"))
    plt.close()
    
    # Expert Contribution Chart (F1-score comparison)
    plt.figure(figsize=(10, 4))
    chart_names = subpart_names + ["Fused MoE"]
    chart_scores = expert_f1_scores + [f1]
    colors = ['skyblue'] * 8 + ['gold']
    plt.bar(chart_names, chart_scores, color=colors, width=0.5)
    plt.ylabel('F1-Score')
    plt.title('Expert vs Fused MoE System Performance')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "expert_contribution_chart.png"))
    plt.close()
    
    # Calibration Plot
    prob_true, prob_pred = calibration_curve(fused_targets, fused_probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='MoE Pipeline')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "calibration_curve.png"))
    plt.close()
    
    # Save config files
    with open(os.path.join(scale_dir, "config.json"), "w") as cfg_f:
        json.dump({
            "scale": scale,
            "subparts": subpart_names,
            "dimensions": subpart_dims,
            "gating_router_input": "last_timestep_concat",
            "device": "cuda",
            "seed": 42
        }, cfg_f, indent=4)
        
    # Model scorecard
    with open(os.path.join(reports_dir, "summary.md"), "w") as sc_f:
        sc_f.write(f"# Benchmarking Scorecard: Expert MoE Pipeline ({scale})\n\n")
        sc_f.write(f"* **Execution Hardware:** GPU\n")
        sc_f.write(f"* **Runtime (seconds):** {elapsed_time:.2f} s\n\n")
        sc_f.write(f"### Performance Metrics\n")
        sc_f.write(f"| Metric | Score |\n")
        sc_f.write(f"| --- | --- |\n")
        sc_f.write(f"| Accuracy | {acc:.4f} |\n")
        sc_f.write(f"| Precision | {prec:.4f} |\n")
        sc_f.write(f"| Recall | {rec:.4f} |\n")
        sc_f.write(f"| F1-Score | {f1:.4f} |\n")
        sc_f.write(f"| Balanced Accuracy | {bal_acc:.4f} |\n")
        sc_f.write(f"| ROC-AUC | {auc:.4f} |\n\n")
        sc_f.write(f"### Expert Models F1-Scores\n")
        for k in range(8):
            sc_f.write(f"* **{subpart_names[k]}:** {expert_f1_scores[k]:.4f}\n")
            
    return {
        "Scale": scale,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Runtime-Seconds": elapsed_time
    }

def main():
    parser = argparse.ArgumentParser(description="Expert Submodality Gating Pipeline")
    parser.add_argument("--scale", type=str, default="all", choices=["2sec", "5sec", "10sec", "all"],
                        help="Timeframe scale to run (default: all)")
    args = parser.parse_args()
    
    scales_configs = [
        ("2sec", "stress_features_fusion_2s.csv"),
        ("5sec", "stress_features_fusion_5s.csv"),
        ("10sec", "stress_features_fusion_10s.csv")
    ]
    
    if args.scale != "all":
        scales_configs = [item for item in scales_configs if item[0] == args.scale]
        
    global_results = []
    
    for scale, filename in scales_configs:
        res = run_expert_scale(scale, filename)
        global_results.append(res)
        
    if len(global_results) > 0:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv(os.path.join(RUN_DIR, "reports", "expert_pipeline_leaderboard.csv"), index=False)
        
        # Build consolidated report
        report_path = os.path.join(RUN_DIR, "reports", "expert_comparison_report.md")
        with open(report_path, "w") as f:
            f.write("# Expert Gating MoE Pipeline Comparison Report\n\n")
            f.write("This report compiles performance comparisons for the Expert Gating Mixture of Experts (MoE) pipeline across all window scales (2s, 5s, 10s).\n\n")
            
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
            f.write("\n\n*All plots and detailed reports have been categorized into the outputs/expert_pipeline/ directory.*")
            
        print(f"\n[SUCCESS] Expert pipeline runs complete. Consolidated report saved to: {report_path}")

if __name__ == "__main__":
    main()
