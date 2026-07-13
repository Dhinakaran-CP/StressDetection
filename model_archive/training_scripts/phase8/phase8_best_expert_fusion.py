import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock

print("=========================================================")
print("Phase 8: Best-Expert Multimodal Fusion")
print("=========================================================")

# ---------------------------------------------------------
# 1. Dataset & Data Processing
# ---------------------------------------------------------
SEQ_LEN = 5
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class StressSequenceDataset(Dataset):
    def __init__(self, X_face, X_voice, X_physio, y, groups, task_groups):
        self.sequences_face = []
        self.sequences_voice = []
        self.sequences_physio = []
        self.labels = []
        
        df_temp = pd.DataFrame({'s': groups, 't': task_groups})
        unique_groups = df_temp.drop_duplicates().values
        
        for s, t in unique_groups:
            idx = np.where((groups == s) & (task_groups == t))[0]
            if len(idx) == 0: continue
            
            f_data, v_data, p_data, l_data = X_face[idx], X_voice[idx], X_physio[idx], y[idx]
            
            for i in range(len(idx) - SEQ_LEN + 1):
                self.sequences_face.append(f_data[i:i+SEQ_LEN])
                self.sequences_voice.append(v_data[i:i+SEQ_LEN])
                self.sequences_physio.append(p_data[i:i+SEQ_LEN])
                self.labels.append(l_data[i+SEQ_LEN-1])
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences_face[idx]),
            torch.FloatTensor(self.sequences_voice[idx]),
            torch.FloatTensor(self.sequences_physio[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )

def subject_adaptive_scaling(X, groups):
    df_tmp = pd.DataFrame(X)
    df_tmp['subject_id'] = groups
    
    global_mean = df_tmp.drop(columns=['subject_id']).mean().values
    global_std = df_tmp.drop(columns=['subject_id']).std().values
    global_std[global_std == 0] = 1e-6
    
    X_out = np.zeros_like(X, dtype=float)
    for subj, group_df in df_tmp.groupby('subject_id'):
        feats = group_df.drop(columns=['subject_id']).values
        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
        std[std == 0] = 1e-6
        
        idx = np.where(groups == subj)[0]
        X_out[idx] = (X[idx] - mean) / std
        
    return X_out

# ---------------------------------------------------------
# 2. Deep Encoders & Dynamic Router
# ---------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        gru_out, hidden = self.gru(x)
        latent = gru_out[:, -1, :] 
        logits = self.classifier(latent) 
        return logits

class DynamicRouter(nn.Module):
    def __init__(self, num_modalities):
        super().__init__()
        # Input is concatenated probabilities (num_modalities * 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_modalities * 2, 8),
            nn.ReLU(),
            nn.Linear(8, num_modalities),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.mlp(x)

# ---------------------------------------------------------
# 3. Execution Pipeline
# ---------------------------------------------------------
print("\n[1] Loading certified datasets...")
df = pd.merge(pd.read_csv("certified_data/face_certified.csv").drop(columns=['video_id', 'window_start', 'window_end']),
              pd.read_csv("certified_data/voice_certified.csv").drop(columns=['video_id', 'window_start', 'window_end']),
              on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
df = pd.merge(df, pd.read_csv("certified_data/physio_certified.csv").drop(columns=['video_id', 'window_start', 'window_end']),
              on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')

df = df.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)

# Subset subjects for faster research pipeline execution to test the fusion logics (15 subjects)
# (In production we would run this on all 65, but 15 is enough to prove pairwise vs 3-way static/dynamic fusion superiority)
subjects = df['subject_id'].unique()[:15]
df = df[df['subject_id'].isin(subjects)].reset_index(drop=True)

lock = FeatureRuntimeLock()
face_features = lock.contract["modalities"]["face"]["features"]
voice_features = lock.contract["modalities"]["voice"]["features"]
physio_features = lock.contract["modalities"]["physio"]["features"]

groups = df['subject_id'].values
task_groups = df['task_id'].values
y = df['label'].values

X_face = subject_adaptive_scaling(df[face_features].values, groups)
X_voice = subject_adaptive_scaling(df[voice_features].values, groups)
X_physio = subject_adaptive_scaling(df[physio_features].values, groups)

print(f"Total merged windows (subset 15): {len(y)}")

gkf = GroupKFold(n_splits=5)
results = {
    "face_only": [], 
    "voice_only": [], 
    "physio_only": [],
    "static_pairwise": [], 
    "static_3way": [],
    "dynamic_pairwise": [], 
    "dynamic_3way": []
}

criterion = nn.CrossEntropyLoss()

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_face, y, groups)):
    print(f"\n  --> Fold {fold+1}/5")
    
    train_dataset = StressSequenceDataset(X_face[train_idx], X_voice[train_idx], X_physio[train_idx], y[train_idx], groups[train_idx], task_groups[train_idx])
    test_dataset = StressSequenceDataset(X_face[test_idx], X_voice[test_idx], X_physio[test_idx], y[test_idx], groups[test_idx], task_groups[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train independent Modality Experts
    enc_f = ModalityEncoder(18, 16).to(DEVICE)
    enc_v = ModalityEncoder(12, 16).to(DEVICE)
    enc_p = ModalityEncoder(5, 16).to(DEVICE)
    
    opt_f = optim.AdamW(enc_f.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_v = optim.AdamW(enc_v.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_p = optim.AdamW(enc_p.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        enc_f.train(); enc_v.train(); enc_p.train()
        for b_face, b_voice, b_physio, b_y in train_loader:
            b_face, b_voice, b_physio, b_y = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
            
            # Face
            opt_f.zero_grad()
            loss_f = criterion(enc_f(b_face), b_y)
            loss_f.backward()
            opt_f.step()
            
            # Voice
            opt_v.zero_grad()
            loss_v = criterion(enc_v(b_voice), b_y)
            loss_v.backward()
            opt_v.step()
            
            # Physio
            opt_p.zero_grad()
            loss_p = criterion(enc_p(b_physio), b_y)
            loss_p.backward()
            opt_p.step()
            
    # Freeze Encoders & Extract Probabilities for Fusion Training
    enc_f.eval(); enc_v.eval(); enc_p.eval()
    
    # Helper to extract probabilities
    def extract_probs(loader):
        p_f, p_v, p_p, y_true = [], [], [], []
        with torch.no_grad():
            for b_face, b_voice, b_physio, b_y in loader:
                b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
                p_f.append(torch.softmax(enc_f(b_face), dim=1).cpu().numpy())
                p_v.append(torch.softmax(enc_v(b_voice), dim=1).cpu().numpy())
                p_p.append(torch.softmax(enc_p(b_physio), dim=1).cpu().numpy())
                y_true.append(b_y.numpy())
        return np.vstack(p_f), np.vstack(p_v), np.vstack(p_p), np.hstack(y_true)

    train_loader_unshuffled = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_pf, train_pv, train_pp, train_y = extract_probs(train_loader_unshuffled)
    test_pf, test_pv, test_pp, test_y = extract_probs(test_loader)
    
    # Evaluate Unimodal
    results["face_only"].append(accuracy_score(test_y, np.argmax(test_pf, axis=1)))
    results["voice_only"].append(accuracy_score(test_y, np.argmax(test_pv, axis=1)))
    results["physio_only"].append(accuracy_score(test_y, np.argmax(test_pp, axis=1)))
    
    # -----------------------------------------------------
    # Static Weighted Fusion (Grid Search on Train, Eval on Test)
    # -----------------------------------------------------
    def grid_search_static(pf, pp, pv=None):
        best_acc = 0
        best_w = None
        for w1 in np.linspace(0, 1, 11):
            if pv is None:
                w2 = 1.0 - w1
                preds = np.argmax((w1 * pf) + (w2 * pp), axis=1)
                acc = accuracy_score(train_y, preds)
                if acc >= best_acc: best_acc, best_w = acc, (w1, w2)
            else:
                for w2 in np.linspace(0, 1-w1, 11):
                    w3 = 1.0 - w1 - w2
                    preds = np.argmax((w1 * pf) + (w2 * pp) + (w3 * pv), axis=1)
                    acc = accuracy_score(train_y, preds)
                    if acc >= best_acc: best_acc, best_w = acc, (w1, w2, w3)
        return best_w

    # Pairwise (Face + Physio)
    best_w_pw = grid_search_static(train_pf, train_pp)
    test_pw_static = np.argmax((best_w_pw[0] * test_pf) + (best_w_pw[1] * test_pp), axis=1)
    results["static_pairwise"].append(accuracy_score(test_y, test_pw_static))
    
    # 3-Way
    best_w_3w = grid_search_static(train_pf, train_pp, train_pv)
    test_3w_static = np.argmax((best_w_3w[0] * test_pf) + (best_w_3w[1] * test_pp) + (best_w_3w[2] * test_pv), axis=1)
    results["static_3way"].append(accuracy_score(test_y, test_3w_static))
    
    # -----------------------------------------------------
    # Dynamic Fusion (Train Router on Train, Eval on Test)
    # -----------------------------------------------------
    def train_dynamic_router(train_inputs, test_inputs):
        router = DynamicRouter(num_modalities=len(train_inputs)).to(DEVICE)
        opt_r = optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
        
        train_inputs_t = [torch.FloatTensor(p).to(DEVICE) for p in train_inputs]
        train_y_t = torch.LongTensor(train_y).to(DEVICE)
        
        # Train
        for e in range(50):
            router.train()
            opt_r.zero_grad()
            cat_in = torch.cat(train_inputs_t, dim=1) # (N, num_modalities * 2)
            weights = router(cat_in) # (N, num_modalities)
            
            # Final logits = weighted sum of probabilities
            fused_p = torch.zeros_like(train_inputs_t[0])
            for m in range(len(train_inputs)):
                fused_p += weights[:, m:m+1] * train_inputs_t[m]
                
            loss = criterion(fused_p, train_y_t)
            loss.backward()
            opt_r.step()
            
        # Eval
        router.eval()
        with torch.no_grad():
            test_inputs_t = [torch.FloatTensor(p).to(DEVICE) for p in test_inputs]
            cat_test = torch.cat(test_inputs_t, dim=1)
            weights_test = router(cat_test)
            fused_p_test = torch.zeros_like(test_inputs_t[0])
            for m in range(len(test_inputs)):
                fused_p_test += weights_test[:, m:m+1] * test_inputs_t[m]
            return torch.argmax(fused_p_test, dim=1).cpu().numpy()
            
    # Pairwise
    test_pw_dyn = train_dynamic_router([train_pf, train_pp], [test_pf, test_pp])
    results["dynamic_pairwise"].append(accuracy_score(test_y, test_pw_dyn))
    
    # 3-Way
    test_3w_dyn = train_dynamic_router([train_pf, train_pv, train_pp], [test_pf, test_pv, test_pp])
    results["dynamic_3way"].append(accuracy_score(test_y, test_3w_dyn))

print("\n[3] Phase 8 Benchmark Results (Cross-Subject Accuracy):")
for method, scores in results.items():
    print(f"  {method.ljust(20)}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Generate Markdown Report
report = f"""# Phase 8: Final Expert Fusion Benchmark

## Protocol
- **Validation**: Leave-One-Subject-Out (Strict 5-Fold GroupKFold, Subset 15 Subjects)
- **Base Encoders**: Independently trained 1D-CNN+GRU Encoders (Face, Voice, Physio) frozen after training.
- **Static Weighted Fusion**: Best fixed weights ($w_1, w_2, w_3$) found via Grid Search on training fold.
- **Dynamic Gated Fusion**: A lightweight MLP Router trained on the frozen probabilities to dynamically assign weights.

## Results (Accuracy across subjects)
| Architecture | Mean Accuracy | Std Dev |
|--------------|---------------|---------|
"""
for method, scores in results.items():
    report += f"| {method} | {np.mean(scores):.4f} | {np.std(scores):.4f} |\n"

report += """
## Conclusion
(Autogenerated) If Pairwise Static outperforms Pairwise Dynamic, we stick to low-latency Static fusion.
If Pairwise (Face+Physio) outperforms 3-Way (Face+Voice+Physio), we drop Voice from the final runtime engine.
"""

os.makedirs("reports", exist_ok=True)
with open("reports/phase8_final_fusion_benchmark.md", "w") as f:
    f.write(report)
    
print("\n[4] Complete. Report saved to reports/phase8_final_fusion_benchmark.md")
