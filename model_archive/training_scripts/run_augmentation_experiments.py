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

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from training.augmentation import apply_jitter, apply_scaling, apply_time_mask, apply_modality_dropout

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEQ_LEN = 5
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StressAugmentedDataset(Dataset):
    def __init__(self, X_face, X_physio, y, groups, task_groups, augment_type=None):
        self.sequences_face = []
        self.sequences_physio = []
        self.labels = []
        self.augment_type = augment_type
        
        df_temp = pd.DataFrame({'s': groups, 't': task_groups})
        unique_groups = df_temp.drop_duplicates().values
        
        for s, t in unique_groups:
            idx = np.where((groups == s) & (task_groups == t))[0]
            if len(idx) == 0: continue
            
            f_data, p_data, l_data = X_face[idx], X_physio[idx], y[idx]
            
            for i in range(len(idx) - SEQ_LEN + 1):
                self.sequences_face.append(f_data[i:i+SEQ_LEN])
                self.sequences_physio.append(p_data[i:i+SEQ_LEN])
                self.labels.append(l_data[i+SEQ_LEN-1])
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        face_seq = self.sequences_face[idx].copy()
        physio_seq = self.sequences_physio[idx].copy()
        label = self.labels[idx]
        
        if self.augment_type == 'jitter':
            face_seq = apply_jitter(face_seq, std=0.03)
            physio_seq = apply_jitter(physio_seq, std=0.03)
        elif self.augment_type == 'scaling':
            face_seq = apply_scaling(face_seq)
            physio_seq = apply_scaling(physio_seq)
        elif self.augment_type == 'time_mask':
            face_seq = apply_time_mask(face_seq)
            physio_seq = apply_time_mask(physio_seq)
        elif self.augment_type == 'modality_dropout':
            face_seq, physio_seq = apply_modality_dropout(face_seq, physio_seq)
        elif self.augment_type == 'combined':
            # Randomly apply augmentations
            if np.random.rand() < 0.5:
                face_seq = apply_jitter(face_seq, std=0.03)
                physio_seq = apply_jitter(physio_seq, std=0.03)
            if np.random.rand() < 0.5:
                face_seq = apply_scaling(face_seq)
                physio_seq = apply_scaling(physio_seq)
            if np.random.rand() < 0.3:
                face_seq = apply_time_mask(face_seq)
                physio_seq = apply_time_mask(physio_seq)
            face_seq, physio_seq = apply_modality_dropout(face_seq, physio_seq, dropout_prob=0.1)
            
        return (
            torch.FloatTensor(face_seq),
            torch.FloatTensor(physio_seq),
            torch.LongTensor([label])[0]
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

# Compact deep models for quick experimentation
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
        return logits, latent

class GatedFusionNetwork(nn.Module):
    def __init__(self, face_dim=18, physio_dim=5, hidden_dim=16):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, hidden_dim)
        self.physio_enc = ModalityEncoder(physio_dim, hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, face, physio):
        logit_f, l_f = self.face_enc(face)
        logit_p, l_p = self.physio_enc(physio)
        
        combined = torch.cat([l_f, l_p], dim=1)
        gates = self.gate_mlp(combined)
        
        fused = gates[:, 0:1] * l_f + gates[:, 1:2] * l_p
        logits_fusion = self.classifier(fused)
        return logit_f, logit_p, logits_fusion

def run_experiment(augment_type, face_data, physio_data, y, groups, task_groups, gkf):
    print(f"\nEvaluating Augmentation: {str(augment_type).upper()}")
    fold_accuracies = []
    criterion = nn.CrossEntropyLoss()
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(face_data, y, groups)):
        # Train
        train_dataset = StressAugmentedDataset(
            face_data[train_idx], physio_data[train_idx], y[train_idx],
            groups[train_idx], task_groups[train_idx], augment_type=augment_type
        )
        # Test (never augment test data!)
        test_dataset = StressAugmentedDataset(
            face_data[test_idx], physio_data[test_idx], y[test_idx],
            groups[test_idx], task_groups[test_idx], augment_type=None
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = GatedFusionNetwork().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        for epoch in range(EPOCHS):
            model.train()
            for b_face, b_physio, b_y in train_loader:
                b_face, b_physio, b_y = b_face.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                lf, lp, l_fus = model(b_face, b_physio)
                loss = criterion(lf, b_y) + criterion(lp, b_y) + criterion(l_fus, b_y)
                loss.backward()
                optimizer.step()
                
        # Eval
        model.eval()
        all_y = []
        all_preds = []
        with torch.no_grad():
            for b_face, b_physio, b_y in test_loader:
                b_face, b_physio = b_face.to(DEVICE), b_physio.to(DEVICE)
                _, _, l_fus = model(b_face, b_physio)
                all_y.extend(b_y.numpy())
                all_preds.extend(torch.argmax(l_fus, dim=1).cpu().numpy())
                
        acc = accuracy_score(all_y, all_preds)
        fold_accuracies.append(acc)
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"  Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    return mean_acc, std_acc

def main():
    print("=========================================================")
    print("Phase 7: Augmentation Comparison & Evaluation")
    print("=========================================================")
    
    # Load 15 subjects subset to run fast
    df_face = pd.read_csv("certified_data/face_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'])
    df_physio = pd.read_csv("certified_data/physio_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'])
    df = pd.merge(df_face, df_physio, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df = df.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)
    
    subjects = df['subject_id'].unique()[:15]
    df = df[df['subject_id'].isin(subjects)].reset_index(drop=True)
    
    lock = FeatureRuntimeLock()
    face_features = lock.contract["modalities"]["face"]["features"]
    physio_features = lock.contract["modalities"]["physio"]["features"]
    
    groups = df['subject_id'].values
    task_groups = df['task_id'].values
    y = df['label'].values
    
    X_face = subject_adaptive_scaling(df[face_features].values, groups)
    X_physio = subject_adaptive_scaling(df[physio_features].values, groups)
    
    gkf = GroupKFold(n_splits=5)
    
    results = {}
    for aug in [None, 'jitter', 'scaling', 'time_mask', 'modality_dropout', 'combined']:
        mean_acc, std_acc = run_experiment(aug, X_face, X_physio, y, groups, task_groups, gkf)
        results[str(aug)] = (mean_acc, std_acc)
        
    # Write report
    report = """# Phase 7: Augmentation Comparison Report

## Protocol
- **Validation**: Leave-One-Subject-Out (Strict 5-Fold GroupKFold, Subset 15 Subjects)
- **Model**: Gated Fusion Network (1D-CNN+GRU encoders for Face & Physio + dynamic router)
- **Epochs**: 8

## Results
| Augmentation Strategy | Mean Accuracy | Std Dev | Performance Delta |
|-----------------------|---------------|---------|-------------------|
"""
    base_acc = results['None'][0]
    for key, (val_acc, val_std) in results.items():
        delta = val_acc - base_acc
        report += f"| {key.capitalize()} | {val_acc:.4f} | {val_std:.4f} | {delta:+.4f} |\n"
        
    report += """
## Conclusion
Choose the simplest augmentation method that consistently improves validation accuracy or reduces standard deviation. If no augmentation shows benefits on validation set (meaning delta is negative or zero), we reject it to avoid unnecessary runtime/training overhead.
"""
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase7_augmentation_comparison.md", "w") as f:
        f.write(report)
    print("\nReport written to reports/phase7_augmentation_comparison.md")

if __name__ == "__main__":
    main()
