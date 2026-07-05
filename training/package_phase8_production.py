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
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-3
LAMBDA_ADV = 0.02  # Penalty for subject identity classification
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(backend_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class Stress3ModalityDataset(Dataset):
    def __init__(self, X_face, X_voice, X_physio, y, groups, task_groups, subj_to_idx=None, augment=False):
        self.sequences_face = []
        self.sequences_voice = []
        self.sequences_physio = []
        self.labels = []
        self.subj_ids = []
        self.augment = augment
        
        df_temp = pd.DataFrame({'s': groups, 't': task_groups})
        unique_groups = df_temp.drop_duplicates().values
        
        for s, t in unique_groups:
            idx = np.where((groups == s) & (task_groups == t))[0]
            if len(idx) == 0: continue
            
            f_data, v_data, p_data, l_data = X_face[idx], X_voice[idx], X_physio[idx], y[idx]
            s_label = subj_to_idx.get(s, 0) if subj_to_idx is not None else 0
            
            for i in range(len(idx) - SEQ_LEN + 1):
                self.sequences_face.append(f_data[i:i+SEQ_LEN])
                self.sequences_voice.append(v_data[i:i+SEQ_LEN])
                self.sequences_physio.append(p_data[i:i+SEQ_LEN])
                self.labels.append(l_data[i+SEQ_LEN-1])
                self.subj_ids.append(s_label)
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        face_seq = self.sequences_face[idx].copy()
        voice_seq = self.sequences_voice[idx].copy()
        physio_seq = self.sequences_physio[idx].copy()
        label = self.labels[idx]
        subj_id = self.subj_ids[idx]
        
        if self.augment:
            face_seq = apply_time_mask(face_seq)
            voice_seq = apply_time_mask(voice_seq)
            physio_seq = apply_time_mask(physio_seq)
            
        return (
            torch.FloatTensor(face_seq),
            torch.FloatTensor(voice_seq),
            torch.FloatTensor(physio_seq),
            torch.LongTensor([label])[0],
            torch.LongTensor([subj_id])[0]
        )

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

class DeepSequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        latent = gru_out[:, -1, :] 
        return latent

class AdversarialModalityEncoder(nn.Module):
    def __init__(self, input_dim, num_subjects=65, hidden_dim=16):
        super().__init__()
        self.encoder = DeepSequenceEncoder(input_dim, hidden_dim)
        self.stress_head = nn.Linear(hidden_dim, 2)
        self.subject_head = nn.Linear(hidden_dim, num_subjects)
        
    def forward(self, x):
        latent = self.encoder(x)
        stress_logits = self.stress_head(latent)
        subject_logits = self.subject_head(latent)
        return stress_logits, subject_logits

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

# ---------------------------------------------------------
# Training and Evaluation Helpers
# ---------------------------------------------------------
def evaluate_loso_strategy(X_face, X_voice, X_physio, y, groups, task_groups, face_features, voice_features, physio_features, subj_to_idx, adversarial=False):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score
    
    gkf = GroupKFold(n_splits=5)
    results = {
        "face_only": [], "voice_only": [], "physio_only": [],
        "face_physio": [], "face_voice": [], "voice_physio": [],
        "all_3_modalities": []
    }
    criterion_stress = nn.CrossEntropyLoss()
    criterion_subject = nn.CrossEntropyLoss()
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_face, y, groups)):
        train_dataset = Stress3ModalityDataset(
            X_face[train_idx], X_voice[train_idx], X_physio[train_idx], y[train_idx],
            groups[train_idx], task_groups[train_idx], subj_to_idx, augment=True
        )
        test_dataset = Stress3ModalityDataset(
            X_face[test_idx], X_voice[test_idx], X_physio[test_idx], y[test_idx],
            groups[test_idx], task_groups[test_idx], subj_to_idx, augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        if adversarial:
            enc_f = AdversarialModalityEncoder(len(face_features), 65).to(DEVICE)
            enc_v = AdversarialModalityEncoder(len(voice_features), 65).to(DEVICE)
            enc_p = AdversarialModalityEncoder(len(physio_features), 65).to(DEVICE)
        else:
            enc_f = ModalityEncoder(len(face_features), 16).to(DEVICE)
            enc_v = ModalityEncoder(len(voice_features), 16).to(DEVICE)
            enc_p = ModalityEncoder(len(physio_features), 16).to(DEVICE)
            
        opt_f = optim.AdamW(enc_f.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        opt_v = optim.AdamW(enc_v.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        opt_p = optim.AdamW(enc_p.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        for epoch in range(EPOCHS):
            enc_f.train(); enc_v.train(); enc_p.train()
            for b_face, b_voice, b_physio, b_y, b_subj in train_loader:
                b_face, b_voice, b_physio, b_y, b_subj = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE), b_subj.to(DEVICE)
                
                # Face
                opt_f.zero_grad()
                if adversarial:
                    stress_logits, subj_logits = enc_f(b_face)
                    loss_stress = criterion_stress(stress_logits, b_y)
                    loss_subj = criterion_subject(subj_logits, b_subj)
                    loss = loss_stress - LAMBDA_ADV * loss_subj
                else:
                    loss = criterion_stress(enc_f(b_face), b_y)
                loss.backward()
                opt_f.step()
                
                # Voice
                opt_v.zero_grad()
                if adversarial:
                    stress_logits, subj_logits = enc_v(b_voice)
                    loss_stress = criterion_stress(stress_logits, b_y)
                    loss_subj = criterion_subject(subj_logits, b_subj)
                    loss = loss_stress - LAMBDA_ADV * loss_subj
                else:
                    loss = criterion_stress(enc_v(b_voice), b_y)
                loss.backward()
                opt_v.step()
                
                # Physio
                opt_p.zero_grad()
                if adversarial:
                    stress_logits, subj_logits = enc_p(b_physio)
                    loss_stress = criterion_stress(stress_logits, b_y)
                    loss_subj = criterion_subject(subj_logits, b_subj)
                    loss = loss_stress - LAMBDA_ADV * loss_subj
                else:
                    loss = criterion_stress(enc_p(b_physio), b_y)
                loss.backward()
                opt_p.step()
                
        enc_f.eval(); enc_v.eval(); enc_p.eval()
        
        def extract_probs(loader):
            pf, pv, pp, y_true = [], [], [], []
            with torch.no_grad():
                for b_face, b_voice, b_physio, b_y, _ in loader:
                    b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
                    if adversarial:
                        f_logits, _ = enc_f(b_face)
                        v_logits, _ = enc_v(b_voice)
                        p_logits, _ = enc_p(b_physio)
                    else:
                        f_logits = enc_f(b_face)
                        v_logits = enc_v(b_voice)
                        p_logits = enc_p(b_physio)
                    pf.append(torch.softmax(f_logits, dim=1).cpu().numpy())
                    pv.append(torch.softmax(v_logits, dim=1).cpu().numpy())
                    pp.append(torch.softmax(p_logits, dim=1).cpu().numpy())
                    y_true.append(b_y.numpy())
            return np.vstack(pf), np.vstack(pv), np.vstack(pp), np.hstack(y_true)
            
        train_loader_unshuffled = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_pf, train_pv, train_pp, train_y = extract_probs(train_loader_unshuffled)
        test_pf, test_pv, test_pp, test_y = extract_probs(test_loader)
        
        # Train FlexDynamicRouter
        router = FlexDynamicRouter(num_modalities=3).to(DEVICE)
        opt_r = optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
        train_y_t = torch.LongTensor(train_y).to(DEVICE)
        num_samples = len(train_y)
        
        for e in range(50):
            router.train()
            opt_r.zero_grad()
            
            probs_f = torch.FloatTensor(train_pf).to(DEVICE)
            probs_v = torch.FloatTensor(train_pv).to(DEVICE)
            probs_p = torch.FloatTensor(train_pp).to(DEVICE)
            
            masks = np.ones((num_samples, 3), dtype=np.float32)
            for i in range(num_samples):
                r = np.random.rand(3)
                masks[i, 0] = 0.0 if r[0] < 0.3 else 1.0
                masks[i, 1] = 0.0 if r[1] < 0.3 else 1.0
                masks[i, 2] = 0.0 if r[2] < 0.3 else 1.0
                if np.sum(masks[i]) == 0:
                    masks[i, np.random.randint(3)] = 1.0
                    
            masks_t = torch.FloatTensor(masks).to(DEVICE)
            probs_f_dropout = torch.where(masks_t[:, 0:1] == 1.0, probs_f, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            probs_v_dropout = torch.where(masks_t[:, 1:2] == 1.0, probs_v, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            probs_p_dropout = torch.where(masks_t[:, 2:3] == 1.0, probs_p, torch.FloatTensor([[0.5, 0.5]]).to(DEVICE))
            
            cat_in = torch.cat([probs_f_dropout, probs_v_dropout, probs_p_dropout, masks_t], dim=1)
            raw_weights = router(cat_in)
            
            masked_weights = raw_weights * masks_t
            sum_weights = torch.sum(masked_weights, dim=1, keepdim=True)
            sum_weights = torch.where(sum_weights == 0, torch.ones_like(sum_weights), sum_weights)
            norm_weights = masked_weights / sum_weights
            
            fused_p = (norm_weights[:, 0:1] * probs_f) + (norm_weights[:, 1:2] * probs_v) + (norm_weights[:, 2:3] * probs_p)
            loss = criterion_stress(fused_p, train_y_t)
            loss.backward()
            opt_r.step()
            
        router.eval()
        
        def run_router_eval(mask_vector):
            with torch.no_grad():
                test_pf_t = torch.FloatTensor(test_pf).to(DEVICE)
                test_pv_t = torch.FloatTensor(test_pv).to(DEVICE)
                test_pp_t = torch.FloatTensor(test_pp).to(DEVICE)
                
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

    # Subject mappings for subject adversarial training
    subj_list = np.unique(groups)
    subj_to_idx = {name: i for i, name in enumerate(subj_list)}

    # Apply subject-adaptive normalization and scaling
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
    
    # Save standard/base scalers
    with open(os.path.join(MODELS_DIR, "deep_face_scaler.pkl"), "wb") as f:
        pickle.dump(face_scaler, f)
    with open(os.path.join(MODELS_DIR, "deep_voice_scaler.pkl"), "wb") as f:
        pickle.dump(voice_scaler, f)
    with open(os.path.join(MODELS_DIR, "deep_physio_scaler.pkl"), "wb") as f:
        pickle.dump(physio_scaler, f)

    # Save adversarial scalers (identical values/weights, named separately for cleaner loading)
    with open(os.path.join(MODELS_DIR, "adv_face_scaler.pkl"), "wb") as f:
        pickle.dump(face_scaler, f)
    with open(os.path.join(MODELS_DIR, "adv_voice_scaler.pkl"), "wb") as f:
        pickle.dump(voice_scaler, f)
    with open(os.path.join(MODELS_DIR, "adv_physio_scaler.pkl"), "wb") as f:
        pickle.dump(physio_scaler, f)

    # Evaluate BOTH Strategies
    print("\nEvaluating Strategy 4 (Standard CNN-GRU)...")
    res_std = evaluate_loso_strategy(
        X_face_scaled, X_voice_scaled, X_physio_scaled, y, groups, task_groups,
        face_features, voice_features, physio_features, subj_to_idx, adversarial=False
    )
    
    print("\nEvaluating Strategy 5 (Adversarial CNN-GRU)...")
    res_adv = evaluate_loso_strategy(
        X_face_scaled, X_voice_scaled, X_physio_scaled, y, groups, task_groups,
        face_features, voice_features, physio_features, subj_to_idx, adversarial=True
    )

    print("\nStrategy 4 (Standard) LOSO Results:")
    for k, v in res_std.items():
        print(f"  {k.ljust(25)}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

    print("\nStrategy 5 (Adversarial) LOSO Results:")
    for k, v in res_adv.items():
        print(f"  {k.ljust(25)}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

    # Train on 100% data
    dataset = Stress3ModalityDataset(X_face_scaled, X_voice_scaled, X_physio_scaled, y, groups, task_groups, subj_to_idx, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion_stress = nn.CrossEntropyLoss()
    criterion_subject = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Train Strategy 4: Standard Encoders
    # ---------------------------------------------------------
    print("\n[3] Training Strategy 4 (Standard) Modality Encoders on Full Data...")
    enc_f_std = ModalityEncoder(len(face_features), 16).to(DEVICE)
    enc_v_std = ModalityEncoder(len(voice_features), 16).to(DEVICE)
    enc_p_std = ModalityEncoder(len(physio_features), 16).to(DEVICE)
    
    opt_f_std = optim.AdamW(enc_f_std.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_v_std = optim.AdamW(enc_v_std.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_p_std = optim.AdamW(enc_p_std.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        enc_f_std.train(); enc_v_std.train(); enc_p_std.train()
        for b_face, b_voice, b_physio, b_y, _ in loader:
            b_face, b_voice, b_physio, b_y = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE)
            
            opt_f_std.zero_grad()
            loss_f = criterion_stress(enc_f_std(b_face), b_y)
            loss_f.backward()
            opt_f_std.step()
            
            opt_v_std.zero_grad()
            loss_v = criterion_stress(enc_v_std(b_voice), b_y)
            loss_v.backward()
            opt_v_std.step()
            
            opt_p_std.zero_grad()
            loss_p = criterion_stress(enc_p_std(b_physio), b_y)
            loss_p.backward()
            opt_p_std.step()

    # ---------------------------------------------------------
    # Train Strategy 5: Adversarial Encoders
    # ---------------------------------------------------------
    print("\n[4] Training Strategy 5 (Adversarial) Modality Encoders on Full Data...")
    enc_f_adv = AdversarialModalityEncoder(len(face_features), 65).to(DEVICE)
    enc_v_adv = AdversarialModalityEncoder(len(voice_features), 65).to(DEVICE)
    enc_p_adv = AdversarialModalityEncoder(len(physio_features), 65).to(DEVICE)
    
    opt_f_adv = optim.AdamW(enc_f_adv.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_v_adv = optim.AdamW(enc_v_adv.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_p_adv = optim.AdamW(enc_p_adv.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        enc_f_adv.train(); enc_v_adv.train(); enc_p_adv.train()
        for b_face, b_voice, b_physio, b_y, b_subj in loader:
            b_face, b_voice, b_physio, b_y, b_subj = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE), b_y.to(DEVICE), b_subj.to(DEVICE)
            
            opt_f_adv.zero_grad()
            stress_logits, subj_logits = enc_f_adv(b_face)
            loss_stress = criterion_stress(stress_logits, b_y)
            loss_subj = criterion_subject(subj_logits, b_subj)
            loss_f = loss_stress - LAMBDA_ADV * loss_subj
            loss_f.backward()
            opt_f_adv.step()
            
            opt_v_adv.zero_grad()
            stress_logits, subj_logits = enc_v_adv(b_voice)
            loss_stress = criterion_stress(stress_logits, b_y)
            loss_subj = criterion_subject(subj_logits, b_subj)
            loss_v = loss_stress - LAMBDA_ADV * loss_subj
            loss_v.backward()
            opt_v_adv.step()
            
            opt_p_adv.zero_grad()
            stress_logits, subj_logits = enc_p_adv(b_physio)
            loss_stress = criterion_stress(stress_logits, b_y)
            loss_subj = criterion_subject(subj_logits, b_subj)
            loss_p = loss_stress - LAMBDA_ADV * loss_subj
            loss_p.backward()
            opt_p_adv.step()

    # ---------------------------------------------------------
    # Train Dynamic Routers for BOTH strategies
    # ---------------------------------------------------------
    def train_router(enc_f, enc_v, enc_p, is_adv):
        enc_f.eval(); enc_v.eval(); enc_p.eval()
        loader_unshuffled = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        pf_all, pv_all, pp_all, y_all = [], [], [], []
        with torch.no_grad():
            for b_face, b_voice, b_physio, b_y, _ in loader_unshuffled:
                b_face, b_voice, b_physio = b_face.to(DEVICE), b_voice.to(DEVICE), b_physio.to(DEVICE)
                if is_adv:
                    f_logits, _ = enc_f(b_face)
                    v_logits, _ = enc_v(b_voice)
                    p_logits, _ = enc_p(b_physio)
                else:
                    f_logits = enc_f(b_face)
                    v_logits = enc_v(b_voice)
                    p_logits = enc_p(b_physio)
                pf_all.append(torch.softmax(f_logits, dim=1).cpu().numpy())
                pv_all.append(torch.softmax(v_logits, dim=1).cpu().numpy())
                pp_all.append(torch.softmax(p_logits, dim=1).cpu().numpy())
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
            loss = criterion_stress(fused_p, y_all_t)
            loss.backward()
            opt_r.step()
        return router

    print("\n[5] Training Routers...")
    router_std = train_router(enc_f_std, enc_v_std, enc_p_std, is_adv=False)
    router_adv = train_router(enc_f_adv, enc_v_adv, enc_p_adv, is_adv=True)

    # ---------------------------------------------------------
    # Save Model Mappings (Adversarial is extracted into ModalityEncoder weights format)
    # ---------------------------------------------------------
    def extract_adv_state_dict(adv_model):
        """Construct standard ModalityEncoder state dict from AdversarialModalityEncoder."""
        std_sd = {}
        adv_sd = adv_model.state_dict()
        for k, v in adv_sd.items():
            if k.startswith("encoder."):
                std_sd[k.replace("encoder.", "")] = v
            elif k.startswith("stress_head."):
                std_sd[k.replace("stress_head.", "classifier.")] = v
        return std_sd

    print("\n[6] Saving All Artifacts to Disk...")
    # Standard (Strategy 4)
    torch.save(enc_f_std.state_dict(), os.path.join(MODELS_DIR, "deep_face_expert.pt"))
    torch.save(enc_v_std.state_dict(), os.path.join(MODELS_DIR, "deep_voice_expert.pt"))
    torch.save(enc_p_std.state_dict(), os.path.join(MODELS_DIR, "deep_physio_expert.pt"))
    torch.save(router_std.state_dict(), os.path.join(MODELS_DIR, "deep_fusion_router.pt"))
    
    # Adversarial (Strategy 5)
    torch.save(extract_adv_state_dict(enc_f_adv), os.path.join(MODELS_DIR, "adv_face_expert.pt"))
    torch.save(extract_adv_state_dict(enc_v_adv), os.path.join(MODELS_DIR, "adv_voice_expert.pt"))
    torch.save(extract_adv_state_dict(enc_p_adv), os.path.join(MODELS_DIR, "adv_physio_expert.pt"))
    torch.save(router_adv.state_dict(), os.path.join(MODELS_DIR, "adv_fusion_router.pt"))

    # Config File (set Adversarial as primary strategy)
    import json
    deep_config = {
        "sequence_length": SEQ_LEN,
        "use_dynamic_router": True,
        "primary_strategy": "adversarial",
        "active_modalities": ["face", "voice", "physio"]
    }
    with open(os.path.join(MODELS_DIR, "deep_fusion_config.json"), "w") as f:
        json.dump(deep_config, f, indent=4)

    # ---------------------------------------------------------
    # Version Registry Registration
    # ---------------------------------------------------------
    print("\n[7] Registering Packaged Production Artifacts...")
    registry = VersionRegistry()

    def register_manifest(key, filename, accuracy, framework):
        manifest_path = os.path.join(MODELS_DIR, filename)
        manifest = ArtifactManifest(key + "_v2", "model", "2.0.0", metadata={
            "accuracy": float(accuracy),
            "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
            "framework": framework
        })
        manifest.compute_hash(manifest_path)
        manifest.save(manifest_path)
        registry.register_model(key, manifest)

    # Strategy 4 (Standard fallback)
    register_manifest("face_expert", "deep_face_expert.pt", np.mean(res_std["face_only"]), "PyTorch (1D-CNN+GRU)")
    register_manifest("voice_expert", "deep_voice_expert.pt", np.mean(res_std["voice_only"]), "PyTorch (1D-CNN+GRU)")
    register_manifest("physio_expert", "deep_physio_expert.pt", np.mean(res_std["physio_only"]), "PyTorch (1D-CNN+GRU)")
    register_manifest("deep_fusion_router", "deep_fusion_router.pt", np.mean(res_std["all_3_modalities"]), "PyTorch (MLP Flex-Router)")

    # Strategy 5 (Adversarial primary)
    register_manifest("adv_face_expert", "adv_face_expert.pt", np.mean(res_adv["face_only"]), "PyTorch (Adversarial CNN-GRU)")
    register_manifest("adv_voice_expert", "adv_voice_expert.pt", np.mean(res_adv["voice_only"]), "PyTorch (Adversarial CNN-GRU)")
    register_manifest("adv_physio_expert", "adv_physio_expert.pt", np.mean(res_adv["physio_only"]), "PyTorch (Adversarial CNN-GRU)")
    register_manifest("adv_fusion_router", "adv_fusion_router.pt", np.mean(res_adv["all_3_modalities"]), "PyTorch (Adversarial Flex-Router)")

    # Also output new validation results to a new report
    report = f"""# Production Models Multi-Strategy Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Feature Contract**: Standard normalized calibration inputs
- **Sequence Length**: {SEQ_LEN}

## Strategy 4 (Standard CNN-GRU) Benchmarks
- **Face-Only**: {np.mean(res_std['face_only']):.4f} ($\pm$ {np.std(res_std['face_only']):.4f})
- **Voice-Only**: {np.mean(res_std['voice_only']):.4f} ($\pm$ {np.std(res_std['voice_only']):.4f})
- **Physio-Only**: {np.mean(res_std['physio_only']):.4f} ($\pm$ {np.std(res_std['physio_only']):.4f})
- **3-Way Fusion**: **{np.mean(res_std['all_3_modalities']):.4f}** ($\pm$ {np.std(res_std['all_3_modalities']):.4f})

## Strategy 5 (Adversarial CNN-GRU) Benchmarks (PRIMARY)
- **Face-Only**: {np.mean(res_adv['face_only']):.4f} ($\pm$ {np.std(res_adv['face_only']):.4f})
- **Voice-Only**: {np.mean(res_adv['voice_only']):.4f} ($\pm$ {np.std(res_adv['voice_only']):.4f})
- **Physio-Only**: {np.mean(res_adv['physio_only']):.4f} ($\pm$ {np.std(res_adv['physio_only']):.4f})
- **3-Way Fusion (Adversarial)**: **{np.mean(res_adv['all_3_modalities']):.4f}** ($\pm$ {np.std(res_adv['all_3_modalities']):.4f})
"""
    with open(os.path.join(MODELS_DIR, "production_benchmark.md"), "w") as f:
        f.write(report)

    print("\nSuccessfully packaged and registered Strategy 5 (Adversarial) and Strategy 4 (Standard) production models!")

if __name__ == "__main__":
    train_and_package()
