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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Directory Setup
# ---------------------------------------------------------
backend_dir = r"c:\Users\StressProject\Desktop\StressDetectionUsingML"
loso_dir = os.path.join(backend_dir, "research", "Phase_1_Baseline_LOSO")

# We run inside Phase_5_GAN_Augmentation folder for separation
RUN_DIR = os.path.join(backend_dir, "research", "Phase_5_GAN_Augmentation")
OUTPUTS_DIR = os.path.join(RUN_DIR, "outputs")
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

# Check XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ---------------------------------------------------------
# Step 2: Temporal Deep Architectures (Identical to baseline)
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
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, 32]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class TemporalTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
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
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        res = self.proj(x)
        
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = x[:, :, :res.size(2)]
        
        x = self.dropout2(self.relu(self.bn2(self.conv2(x))))
        x = x[:, :, :res.size(2)]
        
        out = self.relu(x + res)
        out = out.mean(dim=2)
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
        x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
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
def train_deep_model(model_fn, train_seqs, train_labels, val_seqs, val_labels, epochs=25, batch_size=256, patience=10):
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
# Step 6: WGAN-GP Architecture and Training Logic
# ---------------------------------------------------------
class TabularGenerator(nn.Module):
    def __init__(self, z_dim, class_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + class_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        return self.net(x)

class TabularCritic(nn.Module):
    def __init__(self, feature_dim, class_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + class_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

def compute_gradient_penalty(critic, real_samples, fake_samples, labels_onehot, device):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels_onehot)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(real_features, real_labels, z_dim=64, class_dim=2, epochs=20, batch_size=256, device="cuda"):
    N, feature_dim = real_features.shape
    generator = TabularGenerator(z_dim, class_dim, feature_dim).to(device)
    critic = TabularCritic(feature_dim, class_dim).to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))
    
    labels_onehot_all = np.zeros((N, class_dim), dtype=np.float32)
    labels_onehot_all[np.arange(N), real_labels] = 1.0
    
    actual_batch_size = min(batch_size, N)
    
    dataset = TensorDataset(
        torch.FloatTensor(real_features),
        torch.FloatTensor(labels_onehot_all),
        torch.LongTensor(real_labels)
    )
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=(N >= actual_batch_size))
    
    g_losses = []
    c_losses = []
    
    n_critic = 5
    lambda_gp = 10.0
    
    for epoch in range(epochs):
        for i, (real_x, real_y_oh, real_y) in enumerate(loader):
            curr_bs = real_x.size(0)
            real_x = real_x.to(device)
            real_y_oh = real_y_oh.to(device)
            
            # Train Critic
            opt_c.zero_grad()
            z = torch.randn(curr_bs, z_dim, device=device)
            fake_x = generator(z, real_y_oh)
            
            real_validity = critic(real_x, real_y_oh)
            fake_validity = critic(fake_x, real_y_oh)
            
            gp = compute_gradient_penalty(critic, real_x, fake_x, real_y_oh, device)
            c_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp
            
            c_loss.backward()
            opt_c.step()
            c_losses.append(c_loss.item())
            
            # Train Generator
            if i % n_critic == 0:
                opt_g.zero_grad()
                z = torch.randn(curr_bs, z_dim, device=device)
                gen_x = generator(z, real_y_oh)
                g_loss = -critic(gen_x, real_y_oh).mean()
                g_loss.backward()
                opt_g.step()
                g_losses.append(g_loss.item())
                
    return generator, g_losses, c_losses

# ---------------------------------------------------------
# Step 7: GAN Quality Check Helpers
# ---------------------------------------------------------
def generate_synthetic_samples(generator, num_samples, feature_dim, target_class=1, z_dim=64, device="cuda"):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        y_oh = torch.zeros(num_samples, 2, device=device)
        y_oh[:, target_class] = 1.0
        synthetic_x = generator(z, y_oh).cpu().numpy()
    return synthetic_x

def evaluate_synthetic_quality(real_samples, synthetic_samples, scale, save_dir):
    sample_size = min(1000, len(real_samples), len(synthetic_samples))
    indices_real = np.random.choice(len(real_samples), sample_size, replace=False)
    indices_synth = np.random.choice(len(synthetic_samples), sample_size, replace=False)
    
    r_sub = real_samples[indices_real]
    s_sub = synthetic_samples[indices_synth]
    
    # 1. Feature similarity mean vs mean, std vs std
    real_mean = np.mean(real_samples, axis=0)
    synth_mean = np.mean(synthetic_samples, axis=0)
    real_std = np.std(real_samples, axis=0)
    synth_std = np.std(synthetic_samples, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(real_mean, synth_mean, alpha=0.5, color='blue')
    plt.plot([min(real_mean), max(real_mean)], [min(real_mean), max(real_mean)], 'r--')
    plt.xlabel("Real Feature Means")
    plt.ylabel("Synthetic Feature Means")
    plt.title("Feature Means (Real vs Synthetic)")
    
    plt.subplot(1, 2, 2)
    plt.scatter(real_std, synth_std, alpha=0.5, color='green')
    plt.plot([min(real_std), max(real_std)], [min(real_std), max(real_std)], 'r--')
    plt.xlabel("Real Feature Stds")
    plt.ylabel("Synthetic Feature Stds")
    plt.title("Feature Stds (Real vs Synthetic)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_similarity.png"))
    plt.close()
    
    # 2. Distribution overlap (histogram for top 3 features with highest variance in real data)
    variances = np.var(real_samples, axis=0)
    top_indices = np.argsort(variances)[-3:]
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(top_indices):
        plt.subplot(1, 3, i + 1)
        plt.hist(real_samples[:, idx], bins=30, alpha=0.5, label='Real', density=True)
        plt.hist(synthetic_samples[:, idx], bins=30, alpha=0.5, label='Synthetic', density=True)
        plt.title(f"Feature {idx} Distribution")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "generated_vs_real_distribution.png"))
    plt.close()
    
    # 3. Nearest neighbor check (sample_quality_check)
    nn_real = NearestNeighbors(n_neighbors=2)
    nn_real.fit(r_sub)
    distances_real_real, _ = nn_real.kneighbors(r_sub)
    real_real_dist = distances_real_real[:, 1]
    
    nn_synth = NearestNeighbors(n_neighbors=1)
    nn_synth.fit(r_sub)
    distances_synth_real, _ = nn_synth.kneighbors(s_sub)
    synth_real_dist = distances_synth_real[:, 0]
    
    plt.figure(figsize=(6, 5))
    plt.hist(real_real_dist, bins=30, alpha=0.5, label='Real-Real Dist', density=True, color='purple')
    plt.hist(synth_real_dist, bins=30, alpha=0.5, label='Synth-Real Dist', density=True, color='orange')
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Sample Quality Check: Nearest Neighbors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_quality_check.png"))
    plt.close()
    
    corr_matrix = np.corrcoef(real_mean, synth_mean)
    mean_correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
    quality_passed = mean_correlation > 0.5
    
    return {
        "mean_correlation": mean_correlation,
        "quality_passed": bool(quality_passed),
        "mean_real_real_distance": float(np.mean(real_real_dist)),
        "mean_synth_real_distance": float(np.mean(synth_real_dist))
    }

# ---------------------------------------------------------
# Step 8: Benchmarking & Pipeline Core
# ---------------------------------------------------------
def run_pipeline(scale, filename):
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
    
    exclude_cols = ["subject_id", "task_id", "window_index", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_abs")]
    
    eye, mouth, gface, prosody, spectral, quality, cardio, motion = get_modality_slices(df, dual=True)
    
    seq_len = 5
    seq_eye = make_sequences(eye, seq_len)
    seq_mouth = make_sequences(mouth, seq_len)
    seq_gface = make_sequences(gface, seq_len)
    seq_prosody = make_sequences(prosody, seq_len)
    seq_spectral = make_sequences(spectral, seq_len)
    seq_quality = make_sequences(quality, seq_len)
    seq_cardio = make_sequences(cardio, seq_len)
    seq_motion = make_sequences(motion, seq_len)
    
    X_sequence = np.concatenate([
        seq_eye, seq_mouth, seq_gface, seq_prosody, seq_spectral, seq_quality, seq_cardio, seq_motion
    ], axis=-1)
    
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(df, labels, groups=subjects))
    
    models_to_evaluate = {
        "LogisticRegression": ("classical", "CPU"),
        "SVM": ("classical", "CPU"),
        "RandomForest": ("classical", "CPU"),
        "XGBoost": ("classical", "CPU"),
        "KNN": ("classical", "CPU"),
        "GRU": ("temporal_deep", "GPU"),
        "LSTM": ("temporal_deep", "GPU"),
        "CNN-LSTM": ("temporal_deep", "GPU"),
        "TCN": ("temporal_deep", "GPU"),
        "Transformer": ("temporal_deep", "GPU")
    }
    
    xgb_clf = None
    if HAS_XGBOOST:
        try:
            xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="hist", device="cuda")
            xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
        except Exception:
            try:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="gpu_hist")
                xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
            except Exception:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        xgb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        
    scale_results = []
    
    for experiment_mode in ["real_only", "gan_augmented"]:
        print(f"\n>>> Running Mode: {experiment_mode} for {scale} <<<\n")
        
        mode_dir = os.path.join(OUTPUTS_DIR, experiment_mode, scale)
        os.makedirs(mode_dir, exist_ok=True)
        
        gan_curve_saved = False
        quality_checked = False
        
        for model_name, (category, hardware) in models_to_evaluate.items():
            print(f"--> Benchmarking {model_name} in {experiment_mode}...")
            start_time = time.time()
            
            all_targets, all_preds, all_probs = [], [], []
            fold_results = []
            avg_train_loss_curve, avg_val_loss_curve = [], []
            
            gan_g_losses_all, gan_c_losses_all = [], []
            
            fold_idx = 1
            for train_idx, val_idx in splits:
                scaler_classical = StandardScaler()
                X_class_train = scaler_classical.fit_transform(df[feature_cols].iloc[train_idx].fillna(0).values)
                X_class_val = scaler_classical.transform(df[feature_cols].iloc[val_idx].fillna(0).values)
                
                seq_train = X_sequence[train_idx]
                seq_val = X_sequence[val_idx]
                N_tr, S_tr, F_tr = seq_train.shape
                N_va, S_va, F_va = seq_val.shape
                
                scaler_seq = StandardScaler()
                seq_train_flat = scaler_seq.fit_transform(seq_train.reshape(-1, F_tr))
                seq_val_flat = scaler_seq.transform(seq_val.reshape(-1, F_va))
                
                seq_train_norm = seq_train_flat.reshape(N_tr, S_tr, F_tr)
                seq_val_norm = seq_val_flat.reshape(N_va, S_va, F_va)
                
                train_labels = labels[train_idx]
                val_labels = labels[val_idx]
                
                if experiment_mode == "gan_augmented":
                    c0_idx = np.where(train_labels == 0)[0]
                    c1_idx = np.where(train_labels == 1)[0]
                    num_to_generate = len(c0_idx) - len(c1_idx)
                    
                    if num_to_generate > 0:
                        if hardware == "CPU":
                            gan_epochs = 15
                            gen, g_losses, c_losses = train_wgan_gp(
                                X_class_train, train_labels, z_dim=64, epochs=gan_epochs, batch_size=512, device=DEVICE
                            )
                            synth_features = generate_synthetic_samples(
                                gen, num_to_generate, X_class_train.shape[1], target_class=1, z_dim=64, device=DEVICE
                            )
                            gan_g_losses_all.extend(g_losses)
                            gan_c_losses_all.extend(c_losses)
                            
                            if not quality_checked and fold_idx == 1:
                                os.makedirs(os.path.join(mode_dir, "CTGAN"), exist_ok=True)
                                qc_res = evaluate_synthetic_quality(
                                    X_class_train[c1_idx], synth_features, scale, os.path.join(mode_dir, "CTGAN")
                                )
                                with open(os.path.join(mode_dir, "CTGAN", "synthetic_quality.json"), "w") as qc_f:
                                    json.dump(qc_res, qc_f, indent=4)
                                quality_checked = True
                                
                            X_class_train = np.vstack([X_class_train, synth_features])
                            train_labels_aug = np.concatenate([train_labels, [1] * num_to_generate])
                        else:
                            seq_train_flat_norm = seq_train_norm.reshape(N_tr, -1)
                            gan_epochs = 15
                            gen, g_losses, c_losses = train_wgan_gp(
                                seq_train_flat_norm, train_labels, z_dim=64, epochs=gan_epochs, batch_size=512, device=DEVICE
                            )
                            synth_flat = generate_synthetic_samples(
                                gen, num_to_generate, seq_train_flat_norm.shape[1], target_class=1, z_dim=64, device=DEVICE
                            )
                            gan_g_losses_all.extend(g_losses)
                            gan_c_losses_all.extend(c_losses)
                            
                            synth_seqs = synth_flat.reshape(num_to_generate, S_tr, F_tr)
                            seq_train_norm = np.concatenate([seq_train_norm, synth_seqs], axis=0)
                            train_labels_aug = np.concatenate([train_labels, [1] * num_to_generate])
                    else:
                        train_labels_aug = train_labels
                else:
                    train_labels_aug = train_labels
                
                if experiment_mode == "gan_augmented" and not gan_curve_saved and fold_idx == 1 and len(gan_g_losses_all) > 0:
                    plt.figure()
                    plt.plot(gan_g_losses_all, label="Generator Loss")
                    plt.plot(gan_c_losses_all, label="Critic Loss")
                    plt.xlabel("Iteration Step")
                    plt.ylabel("Loss")
                    plt.title(f"WGAN-GP Training Loss - {scale}")
                    plt.legend()
                    plt.tight_layout()
                    os.makedirs(os.path.join(mode_dir, "CTGAN"), exist_ok=True)
                    plt.savefig(os.path.join(mode_dir, "CTGAN", "gan_loss_curve.png"))
                    plt.close()
                    gan_curve_saved = True

                if hardware == "CPU":
                    if model_name == "LogisticRegression":
                        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    elif model_name == "SVM":
                        clf = SVC(probability=True, class_weight="balanced", max_iter=500, tol=0.01, cache_size=2000)
                    elif model_name == "RandomForest":
                        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1)
                    elif model_name == "XGBoost":
                        clf = xgb_clf
                    elif model_name == "KNN":
                        clf = KNeighborsClassifier(n_neighbors=5)
                        
                    if model_name == "SVM":
                        if len(X_class_train) > 3000:
                            np.random.seed(42)
                            sub_idx = np.random.choice(len(X_class_train), 3000, replace=False)
                            X_train_svm = X_class_train[sub_idx]
                            y_train_svm = train_labels_aug[sub_idx]
                        else:
                            X_train_svm = X_class_train
                            y_train_svm = train_labels_aug
                        clf.fit(X_train_svm, y_train_svm)
                    else:
                        clf.fit(X_class_train, train_labels_aug)
                    preds = clf.predict(X_class_val)
                    probs = clf.predict_proba(X_class_val)[:, 1]
                    
                    avg_train_loss_curve = [0, 0]
                    avg_val_loss_curve = [0, 0]
                else:
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
                        make_fn, seq_train_norm, train_labels_aug,
                        seq_val_norm, val_labels,
                        epochs=25, batch_size=256, patience=10
                    )
                    preds, probs = eval_deep_model(model, seq_val_norm)
                    
                    if len(avg_train_loss_curve) == 0:
                        avg_train_loss_curve = tr_loss
                        avg_val_loss_curve = va_loss
                    else:
                        min_len = min(len(avg_train_loss_curve), len(tr_loss))
                        avg_train_loss_curve = [a + b for a, b in zip(avg_train_loss_curve[:min_len], tr_loss[:min_len])]
                        avg_val_loss_curve = [a + b for a, b in zip(avg_val_loss_curve[:min_len], va_loss[:min_len])]
                        
                fold_acc = accuracy_score(val_labels, preds)
                fold_f1 = f1_score(val_labels, preds, average="binary", zero_division=0)
                try:
                    fold_auc = roc_auc_score(val_labels, probs)
                except ValueError:
                    fold_auc = 0.5
                    
                fold_results.append({
                    "Fold": fold_idx,
                    "Accuracy": fold_acc,
                    "F1-Score": fold_f1,
                    "ROC-AUC": fold_auc
                })
                
                all_targets.extend(val_labels)
                all_preds.extend(preds)
                all_probs.extend(probs)
                fold_idx += 1
                
            elapsed_time = time.time() - start_time
            
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
                
            if experiment_mode == "gan_augmented":
                model_out_dir = os.path.join(mode_dir, "CTGAN", model_name)
            else:
                model_out_dir = os.path.join(mode_dir, model_name)
                
            os.makedirs(model_out_dir, exist_ok=True)
            
            if hardware == "CPU":
                joblib.dump(clf, os.path.join(model_out_dir, "model.pkl"))
            else:
                torch.save(model.state_dict(), os.path.join(model_out_dir, "model.pt"))
                
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
            
            with open(os.path.join(model_out_dir, "config.json"), "w") as cfg_f:
                json.dump({
                    "model_name": model_name,
                    "scale": scale,
                    "experiment_mode": experiment_mode,
                    "category": category,
                    "execution_hardware": hardware,
                    "input_dimensions": X_sequence.shape[-1] if hardware == "GPU" else len(feature_cols),
                    "optimizer": "Adam" if hardware == "GPU" else "N/A",
                    "learning_rate": 0.001 if hardware == "GPU" else "N/A"
                }, cfg_f, indent=4)
                
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
            
            plt.figure()
            if hardware == "GPU" and len(avg_train_loss_curve) > 0:
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
            
            scale_results.append({
                "Scale": scale,
                "Experiment Mode": experiment_mode,
                "Model Name": model_name,
                "Category": category,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "Runtime-Seconds": elapsed_time
            })
            
    return scale_results

def main():
    parser = argparse.ArgumentParser(description="GAN Enhanced Temporal Model Training Pipeline")
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
        scale_results = run_pipeline(scale, filename)
        global_results.extend(scale_results)
        
    if len(global_results) > 0:
        df_global = pd.DataFrame(global_results)
        df_global = df_global.sort_values(by=["Scale", "Experiment Mode", "Accuracy"], ascending=[True, True, False]).reset_index(drop=True)
        
        leaderboard_path = os.path.join(RUN_DIR, "reports", "gan_pipeline_leaderboard.csv")
        df_global.to_csv(leaderboard_path, index=False)
        print(f"\n[INFO] Saved consolidated leaderboard to: {leaderboard_path}")
        
        report_path = os.path.join(RUN_DIR, "reports", "gan_comparison_report.md")
        with open(report_path, "w") as f:
            f.write("# GAN Experiment Comparison Report\n\n")
            f.write("This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures, comparing the Real-Only baseline with the GAN-Augmented training splits.\n\n")
            
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
            f.write("\n\n*All plots and detailed reports have been categorized into the gan_pipeline_run/outputs/ directory.*")
            
        print(f"\n[SUCCESS] Pipeline runs complete. Consolidated report saved to: {report_path}")

if __name__ == "__main__":
    main()
