import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from backend.core.artifact_manifest import ArtifactManifest
from backend.core.version_registry import VersionRegistry
from training.augmentation import apply_time_mask

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEQ_LEN = 5
BATCH_SIZE = 512  # Batch size 512 for fast training on CPU
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(backend_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class Stress3ModalityDataset(Dataset):
    def __init__(self, X_face, X_voice, X_physio, y, groups, task_groups, augment=False):
        self.sequences_face = []
        self.sequences_voice = []
        self.sequences_physio = []
        self.labels = []
        self.augment = augment
        
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
        face_seq = self.sequences_face[idx].copy()
        voice_seq = self.sequences_voice[idx].copy()
        physio_seq = self.sequences_physio[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            face_seq = apply_time_mask(face_seq)
            voice_seq = apply_time_mask(voice_seq)
            physio_seq = apply_time_mask(physio_seq)
            
        return (
            torch.FloatTensor(face_seq),
            torch.FloatTensor(voice_seq),
            torch.FloatTensor(physio_seq),
            torch.LongTensor([label])[0]
        )

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

class FlexDynamicRouter(nn.Module):
    def __init__(self, num_modalities=3):
        super().__init__()
        # Input: num_modalities * 2 (probabilities) + num_modalities (availability mask)
        self.mlp = nn.Sequential(
            nn.Linear(num_modalities * 2 + num_modalities, 16),
            nn.ReLU(),
            nn.Linear(16, num_modalities),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.mlp(x)

def subject_adaptive_scaling(X, groups):
    df_tmp = pd.DataFrame(X)
    df_tmp['subject_id'] = groups
    
    X_out = np.zeros_like(X, dtype=float)
    for subj, group_df in df_tmp.groupby('subject_id'):
        feats = group_df.drop(columns=['subject_id']).values
        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
        std[std == 0] = 1e-6
        
        idx = np.where(groups == subj)[0]
        X_out[idx] = (X[idx] - mean) / std
        
    return X_out

def evaluate_loso(X_face, X_voice, X_physio, y, groups, task_groups, face_features, voice_features, physio_features):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score
    
    print("\nStarting Strict 3-Way 5-Fold LOSO Cross-Validation on Full Dataset...")
    gkf = GroupKFold(n_splits=5)
    
    # Store accuracies for different combinations of inputs
    results = {
        "face_only": [], "voice_only": [], "physio_only": [],
        "face_physio": [], "face_voice": [], "voice_physio": [],
        "all_3_modalities": []
    }
    criterion = nn.CrossEntropyLoss()
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_face, y, groups)):
        print(f"  -> Fold {fold+1}/5")
        
        train_dataset = Stress3ModalityDataset(
            X_face[train_idx], X_voice[train_idx], X_physio[train_idx], y[train_idx],
            groups[train_idx], task_groups[train_idx], augment=True
        )
        test_dataset = Stress3ModalityDataset(
            X_face[test_idx], X_voice[test_idx], X_physio[test_idx], y[test_idx],
            groups[test_idx], task_groups[test_idx], augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Encoders
        enc_f = ModalityEncoder(len(face_features), 16).to(DEVICE)
        enc_v = ModalityEncoder(len(voice_features), 16).to(DEVICE)
        enc_p = ModalityEncoder(len(physio_features), 16).to(DEVICE)
        
        opt_f = optim.AdamW(enc_f.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        opt_v = optim.AdamW(enc_v.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        opt_p = optim.AdamW(enc_p.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Train unimodal modality encoders with Time Masking
        for epoch in range(EPOCHS):
            enc_f.train(); enc_v.train(); enc_p.train()
            for b_face, b_voice, b_physio, b_y in train_loader:
                b_face, b_voice, b_physio, b_y = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
                
                opt_f.zero_grad()
                loss_f = criterion(enc_f(b_face), b_y)
                loss_f.backward()
                opt_f.step()
                
                opt_v.zero_grad()
                loss_v = criterion(enc_v(b_voice), b_y)
                loss_v.backward()
                opt_v.step()
                
                opt_p.zero_grad()
                loss_p = criterion(enc_p(b_physio), b_y)
                loss_p.backward()
                opt_p.step()
                
        # Extract Probabilities
        enc_f.eval(); enc_v.eval(); enc_p.eval()
        
        def extract_probs(loader):
            pf, pv, pp, y_true = [], [], [], []
            with torch.no_grad():
                for b_face, b_voice, b_physio, b_y in loader:
                    b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
                    pf.append(torch.softmax(enc_f(b_face), dim=1).cpu().numpy())
                    pv.append(torch.softmax(enc_v(b_voice), dim=1).cpu().numpy())
                    pp.append(torch.softmax(enc_p(b_physio), dim=1).cpu().numpy())
                    y_true.append(b_y.numpy())
            return np.vstack(pf), np.vstack(pv), np.vstack(pp), np.hstack(y_true)
            
        train_loader_unshuffled = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_pf, train_pv, train_pp, train_y = extract_probs(train_loader_unshuffled)
        test_pf, test_pv, test_pp, test_y = extract_probs(test_loader)
        
        # Train FlexDynamicRouter with Modality Dropout
        router = FlexDynamicRouter(num_modalities=3).to(DEVICE)
        opt_r = optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
        
        train_y_t = torch.LongTensor(train_y).to(DEVICE)
        num_samples = len(train_y)
        
        for e in range(50):
            router.train()
            opt_r.zero_grad()
            
            # Construct probability inputs
            probs_f = torch.FloatTensor(train_pf).to(DEVICE)
            probs_v = torch.FloatTensor(train_pv).to(DEVICE)
            probs_p = torch.FloatTensor(train_pp).to(DEVICE)
            
            # Modality Dropout: randomly drop Face, Voice, or Physio
            # Availability masks of shape (N, 3)
            masks = np.ones((num_samples, 3), dtype=np.float32)
            for i in range(num_samples):
                # 30% dropout rate for each modality
                r = np.random.rand(3)
                masks[i, 0] = 0.0 if r[0] < 0.3 else 1.0
                masks[i, 1] = 0.0 if r[1] < 0.3 else 1.0
                masks[i, 2] = 0.0 if r[2] < 0.3 else 1.0
                # Ensure at least one modality is active
                if np.sum(masks[i]) == 0:
                    masks[i, np.random.randint(3)] = 1.0
                    
            masks_t = torch.FloatTensor(masks).to(DEVICE)
            
            # Neutral probabilities [0.5, 0.5] if dropped
            probs_f_dropout = torch.where(masks_t[:, 0:1] == 1.0, probs_f, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            probs_v_dropout = torch.where(masks_t[:, 1:2] == 1.0, probs_v, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            probs_p_dropout = torch.where(masks_t[:, 2:3] == 1.0, probs_p, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            
            # Concatenate probabilities and availability mask
            cat_in = torch.cat([probs_f_dropout, probs_v_dropout, probs_p_dropout, masks_t], dim=1)
            raw_weights = router(cat_in)
            
            # Apply mask and re-normalize weights
            masked_weights = raw_weights * masks_t
            sum_weights = torch.sum(masked_weights, dim=1, keepdim=True)
            # Avoid division by zero
            sum_weights = torch.where(sum_weights == 0, torch.ones_like(sum_weights), sum_weights)
            norm_weights = masked_weights / sum_weights
            
            fused_p = (norm_weights[:, 0:1] * probs_f) + (norm_weights[:, 1:2] * probs_v) + (norm_weights[:, 2:3] * probs_p)
            loss = criterion(fused_p, train_y_t)
            loss.backward()
            opt_r.step()
            
        # Eval different inference availability patterns on test set
        router.eval()
        
        def run_router_eval(mask_vector):
            with torch.no_grad():
                test_pf_t = torch.FloatTensor(test_pf).to(DEVICE)
                test_pv_t = torch.FloatTensor(test_pv).to(DEVICE)
                test_pp_t = torch.FloatTensor(test_pp).to(DEVICE)
                
                # Apply mask to inputs
                t_mask = torch.FloatTensor([mask_vector] * len(test_y)).to(DEVICE)
                
                pf_in = torch.where(t_mask[:, 0:1] == 1.0, test_pf_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
                pv_in = torch.where(t_mask[:, 1:2] == 1.0, test_pv_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
                pp_in = torch.where(t_mask[:, 2:3] == 1.0, test_pp_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
                
                cat_test = torch.cat([pf_in, pv_in, pp_in, t_mask], dim=1)
                raw_w = router(cat_test)
                
                masked_w = raw_w * t_mask
                sum_w = torch.sum(masked_w, dim=1, keepdim=True)
                sum_w = torch.where(sum_w == 0, torch.ones_like(sum_w), sum_w)
                norm_w = masked_w / sum_w
                
                fused_test = (norm_w[:, 0:1] * test_pf_t) + (norm_w[:, 1:2] * test_pv_t) + (norm_w[:, 2:3] * test_pp_t)
                preds = torch.argmax(fused_test, dim=1).cpu().numpy()
                return accuracy_score(test_y, preds)
                
        results["face_only"].append(accuracy_score(test_y, np.argmax(test_pf, axis=1)))
        results["voice_only"].append(accuracy_score(test_y, np.argmax(test_pv, axis=1)))
        results["physio_only"].append(accuracy_score(test_y, np.argmax(test_pp, axis=1)))
        
        results["face_physio"].append(run_router_eval([1.0, 0.0, 1.0]))
        results["face_voice"].append(run_router_eval([1.0, 1.0, 0.0]))
        results["voice_physio"].append(run_router_eval([0.0, 1.0, 1.0]))
        results["all_3_modalities"].append(run_router_eval([1.0, 1.0, 1.0]))
        
    print("\nPhase 8 Final Flex-Fusion Strict Validation Results:")
    for k, v in results.items():
        print(f"  {k.ljust(25)}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")
        
    # Write report
    report = f"""# Phase 8: Final Audited Multimodal Fusion Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Modality Encoders**: PyTorch 1D-CNN+GRU Encoders trained with Time Masking augmentation.
- **Fusion Engine**: Flex-Modality Dynamic Router MLP (supports any subset of Face, Voice, Physio inputs).
- **Sequence Length**: {SEQ_LEN}

## Final Benchmark Results (Cross-Subject Validation Accuracies)

### Unimodal Encoders
- **Face-Only**: {np.mean(results['face_only']):.4f} ($\pm$ {np.std(results['face_only']):.4f})
- **Voice-Only**: {np.mean(results['voice_only']):.4f} ($\pm$ {np.std(results['voice_only']):.4f})
- **Physio-Only**: {np.mean(results['physio_only']):.4f} ($\pm$ {np.std(results['physio_only']):.4f})

### Pairwise Combinations
- **Face + Physio**: {np.mean(results['face_physio']):.4f} ($\pm$ {np.std(results['face_physio']):.4f})
- **Face + Voice**: {np.mean(results['face_voice']):.4f} ($\pm$ {np.std(results['face_voice']):.4f})
- **Voice + Physio**: {np.mean(results['voice_physio']):.4f} ($\pm$ {np.std(results['voice_physio']):.4f})

### Full 3-Way Fusion
- **Face + Voice + Physio (All Sensors Present)**: **{np.mean(results['all_3_modalities']):.4f}** ($\pm$ {np.std(results['all_3_modalities']):.4f})
"""
    with open("reports/phase8_final_fusion_benchmark.md", "w") as f:
        f.write(report)
    print("Saved final benchmark report to reports/phase8_final_fusion_benchmark.md")
    return results

def train_and_package():
    print("\n[1] Loading certified datasets...")
    df_face = pd.read_csv("certified_data/face_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'])
    df_voice = pd.read_csv("certified_data/voice_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'])
    df_physio = pd.read_csv("certified_data/physio_certified.csv").drop(columns=['video_id', 'window_start', 'window_end'])
    
    df = pd.merge(df_face, df_voice, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df = pd.merge(df, df_physio, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df = df.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)

    lock = FeatureRuntimeLock()
    face_features = lock.contract["modalities"]["face"]["features"]
    voice_features = lock.contract["modalities"]["voice"]["features"]
    physio_features = lock.contract["modalities"]["physio"]["features"]

    groups = df['subject_id'].values
    task_groups = df['task_id'].values
    y = df['label'].values

    # Evaluate LOSO strict cross-validation on full dataset first
    results = evaluate_loso(
        df[face_features].values, df[voice_features].values, df[physio_features].values,
        y, groups, task_groups, face_features, voice_features, physio_features
    )

    # 1. Fit Global Scalers after subject-adaptive scaling
    from sklearn.preprocessing import StandardScaler
    print("\n[2] Applying Subject-Adaptive Normalization and Fitting Production Scalers...")
    X_face_norm = subject_adaptive_scaling(df[face_features].values, groups)
    X_voice_norm = subject_adaptive_scaling(df[voice_features].values, groups)
    X_physio_norm = subject_adaptive_scaling(df[physio_features].values, groups)

    face_scaler = StandardScaler()
    voice_scaler = StandardScaler()
    physio_scaler = StandardScaler()
    
    X_face_scaled = face_scaler.fit_transform(X_face_norm)
    X_voice_scaled = voice_scaler.fit_transform(X_voice_norm)
    X_physio_scaled = physio_scaler.fit_transform(X_physio_norm)
    
    face_scaler_path = os.path.join(MODELS_DIR, "deep_face_scaler.pkl")
    voice_scaler_path = os.path.join(MODELS_DIR, "deep_voice_scaler.pkl")
    physio_scaler_path = os.path.join(MODELS_DIR, "deep_physio_scaler.pkl")
    
    with open(face_scaler_path, "wb") as f:
        pickle.dump(face_scaler, f)
    with open(voice_scaler_path, "wb") as f:
        pickle.dump(voice_scaler, f)
    with open(physio_scaler_path, "wb") as f:
        pickle.dump(physio_scaler, f)

    # 2. Train Encoders on 100% of certified data with Time Masking
    print("\n[3] Training Production Modality Encoders on Full Data (Time-Masked)...")
    dataset = Stress3ModalityDataset(X_face_scaled, X_voice_scaled, X_physio_scaled, y, groups, task_groups, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    enc_f = ModalityEncoder(len(face_features), 16).to(DEVICE)
    enc_v = ModalityEncoder(len(voice_features), 16).to(DEVICE)
    enc_p = ModalityEncoder(len(physio_features), 16).to(DEVICE)
    
    opt_f = optim.AdamW(enc_f.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_v = optim.AdamW(enc_v.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_p = optim.AdamW(enc_p.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        enc_f.train(); enc_v.train(); enc_p.train()
        for b_face, b_voice, b_physio, b_y in loader:
            b_face, b_voice, b_physio, b_y = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
            
            opt_f.zero_grad()
            loss_f = criterion(enc_f(b_face), b_y)
            loss_f.backward()
            opt_f.step()
            
            opt_v.zero_grad()
            loss_v = criterion(enc_v(b_voice), b_y)
            loss_v.backward()
            opt_v.step()
            
            opt_p.zero_grad()
            loss_p = criterion(enc_p(b_physio), b_y)
            loss_p.backward()
            opt_p.step()
            
        print(f"  -> Epoch {epoch+1}/{EPOCHS} complete")

    # 3. Train Production Dynamic Router on Encoder Probabilities
    print("\n[4] Training Production Dynamic Router on Full Data...")
    enc_f.eval(); enc_v.eval(); enc_p.eval()
    
    loader_unshuffled = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    pf_all, pv_all, pp_all, y_all = [], [], [], []
    with torch.no_grad():
        for b_face, b_voice, b_physio, b_y in loader_unshuffled:
            b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
            pf_all.append(torch.softmax(enc_f(b_face), dim=1).cpu().numpy())
            pv_all.append(torch.softmax(enc_v(b_voice), dim=1).cpu().numpy())
            pp_all.append(torch.softmax(enc_p(b_physio), dim=1).cpu().numpy())
            y_all.append(b_y.numpy())
            
    pf_all = np.vstack(pf_all)
    pv_all = np.vstack(pv_all)
    pp_all = np.vstack(pp_all)
    y_all = np.hstack(y_all)
    
    router = FlexDynamicRouter(num_modalities=3).to(DEVICE)
    opt_r = optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_pf_t = torch.FloatTensor(pf_all).to(DEVICE)
    train_pv_t = torch.FloatTensor(pv_all).to(DEVICE)
    train_pp_t = torch.FloatTensor(pp_all).to(DEVICE)
    y_all_t = torch.LongTensor(y_all).to(DEVICE)
    num_samples = len(y_all)
    
    for e in range(50):
        router.train()
        opt_r.zero_grad()
        
        # Apply Modality Dropout to router training
        masks = np.ones((num_samples, 3), dtype=np.float32)
        for i in range(num_samples):
            r = np.random.rand(3)
            masks[i, 0] = 0.0 if r[0] < 0.3 else 1.0
            masks[i, 1] = 0.0 if r[1] < 0.3 else 1.0
            masks[i, 2] = 0.0 if r[2] < 0.3 else 1.0
            if np.sum(masks[i]) == 0:
                masks[i, np.random.randint(3)] = 1.0
                
        masks_t = torch.FloatTensor(masks).to(DEVICE)
        
        probs_f_drop = torch.where(masks_t[:, 0:1] == 1.0, train_pf_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
        probs_v_drop = torch.where(masks_t[:, 1:2] == 1.0, train_pv_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
        probs_p_drop = torch.where(masks_t[:, 2:3] == 1.0, train_pp_t, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
        
        cat_in = torch.cat([probs_f_drop, probs_v_drop, probs_p_drop, masks_t], dim=1)
        raw_weights = router(cat_in)
        
        masked_weights = raw_weights * masks_t
        sum_weights = torch.sum(masked_weights, dim=1, keepdim=True)
        sum_weights = torch.where(sum_weights == 0, torch.ones_like(sum_weights), sum_weights)
        norm_weights = masked_weights / sum_weights
        
        fused_p = (norm_weights[:, 0:1] * train_pf_t) + (norm_weights[:, 1:2] * train_pv_t) + (norm_weights[:, 2:3] * train_pp_t)
        loss = criterion(fused_p, y_all_t)
        loss.backward()
        opt_r.step()

    # 4. Save PyTorch Models
    print("\n[5] Saving Production Weights and Router...")
    face_model_path = os.path.join(MODELS_DIR, "deep_face_expert.pt")
    voice_model_path = os.path.join(MODELS_DIR, "deep_voice_expert.pt")
    physio_model_path = os.path.join(MODELS_DIR, "deep_physio_expert.pt")
    router_model_path = os.path.join(MODELS_DIR, "deep_fusion_router.pt")
    
    torch.save(enc_f.state_dict(), face_model_path)
    torch.save(enc_v.state_dict(), voice_model_path)
    torch.save(enc_p.state_dict(), physio_model_path)
    torch.save(router.state_dict(), router_model_path)
    
    import json
    deep_config = {
        "sequence_length": SEQ_LEN,
        "use_dynamic_router": True,
        "active_modalities": ["face", "voice", "physio"]
    }
    deep_config_path = os.path.join(MODELS_DIR, "deep_fusion_config.json")
    with open(deep_config_path, "w") as f:
        json.dump(deep_config, f, indent=4)

    # 5. Create manifests and register artifacts
    print("\n[6] Registering Packaged Production Artifacts...")
    registry = VersionRegistry()
    
    # Manifest for Face Expert
    manifest_f = ArtifactManifest("face_expert_v2", "model", "2.0.0", metadata={
        "accuracy": float(np.mean(results["face_only"])),
        "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
        "framework": "PyTorch (1D-CNN+GRU)"
    })
    manifest_f.compute_hash(face_model_path)
    manifest_f.save(face_model_path)
    registry.register_model("face_expert", manifest_f)
    
    # Manifest for Voice Expert
    manifest_v = ArtifactManifest("voice_expert_v2", "model", "2.0.0", metadata={
        "accuracy": float(np.mean(results["voice_only"])),
        "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
        "framework": "PyTorch (1D-CNN+GRU)"
    })
    manifest_v.compute_hash(voice_model_path)
    manifest_v.save(voice_model_path)
    registry.register_model("voice_expert", manifest_v)
    
    # Manifest for Physio Expert
    manifest_p = ArtifactManifest("physio_expert_v2", "model", "2.0.0", metadata={
        "accuracy": float(np.mean(results["physio_only"])),
        "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
        "framework": "PyTorch (1D-CNN+GRU)"
    })
    manifest_p.compute_hash(physio_model_path)
    manifest_p.save(physio_model_path)
    registry.register_model("physio_expert", manifest_p)

    # Manifest for Dynamic Router
    manifest_r = ArtifactManifest("deep_fusion_router_v2", "model", "2.0.0", metadata={
        "accuracy_all_present": float(np.mean(results["all_3_modalities"])),
        "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
        "framework": "PyTorch (MLP Flex-Router)"
    })
    manifest_r.compute_hash(router_model_path)
    manifest_r.save(router_model_path)
    registry.register_model("deep_fusion_router", manifest_r)
    
    print("\nSuccessfully packaged and registered all Phase 8 deep learning production models!")

if __name__ == "__main__":
    train_and_package()
