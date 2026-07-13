import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock

# Feature Sub-groups Configuration (Excluding risky identity features)
EXCLUDED_FEATURES = ["face_height_norm", "landmark_confidence", "f0_mean", "f0_range", "eda_scl_mean"]

EYE_FEATURES = ['left_ear', 'right_ear', 'avg_ear', 'blink_velocity', 'eye_openness_ratio']
MOUTH_FEATURES = ['lip_compression', 'jaw_tension', 'mouth_corner_pull']
GLOBAL_FACE_FEATURES = ['brow_descent_left', 'brow_descent_right', 'brow_asymmetry', 'forehead_tension', 
                        'head_tilt', 'temporal_x_var', 'temporal_y_var', 'nose_wrinkle']

PROSODY_FEATURES = ['speaking_rate_proxy', 'pause_ratio', 'voiced_fraction']
SPECTRAL_FEATURES = ['spectral_flux', 'high_freq_ratio']
QUALITY_FEATURES = ['f0_std', 'jitter_percent', 'shimmer_db', 'hnr', 'voice_intensity']

CARDIO_FEATURES = ['ecg_rate_mean', 'ecg_hrv_rmssd', 'ecg_hrv_sdnn']
MOTION_FEATURES = ['resp_rate_mean']

class MultimodalExpertDataset(Dataset):
    def __init__(self, df, face_scaler=None, voice_scaler=None, physio_scaler=None, seq_len=5):
        self.seq_len = seq_len
        
        # Extract features
        lock = FeatureRuntimeLock()
        face_cols = [f for f in lock.contract["modalities"]["face"]["features"] if f not in EXCLUDED_FEATURES]
        voice_cols = [f for f in lock.contract["modalities"]["voice"]["features"] if f not in EXCLUDED_FEATURES]
        physio_cols = [f for f in lock.contract["modalities"]["physio"]["features"] if f not in EXCLUDED_FEATURES]
        
        X_face_raw = df[face_cols].values
        X_voice_raw = df[voice_cols].values
        X_physio_raw = df[physio_cols].values
        
        # Scale
        self.face_scaler = face_scaler or StandardScaler()
        self.voice_scaler = voice_scaler or StandardScaler()
        self.physio_scaler = physio_scaler or StandardScaler()
        
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
            
        # Re-map scaled arrays to DataFrames for easy subsetting
        self.df_face_scaled = pd.DataFrame(X_face, columns=face_cols)
        self.df_voice_scaled = pd.DataFrame(X_voice, columns=voice_cols)
        self.df_physio_scaled = pd.DataFrame(X_physio, columns=physio_cols)
        
        # Sub-group expert slices
        self.feats_eye = self.df_face_scaled[EYE_FEATURES].values
        self.feats_mouth = self.df_face_scaled[MOUTH_FEATURES].values
        self.feats_global_face = self.df_face_scaled[GLOBAL_FACE_FEATURES].values
        
        self.feats_prosody = self.df_voice_scaled[PROSODY_FEATURES].values
        self.feats_spectral = self.df_voice_scaled[SPECTRAL_FEATURES].values
        self.feats_quality = self.df_voice_scaled[QUALITY_FEATURES].values
        
        self.feats_cardio = self.df_physio_scaled[CARDIO_FEATURES].values
        self.feats_motion = self.df_physio_scaled[MOTION_FEATURES].values
        
        self.labels = df['label'].values
        self.subjects = df['subject_id'].values
        self.tasks = df['task_id'].values
        
        # Generate temporal sequences grouped by subject and task
        self.sequences = []
        self.seq_labels = []
        self.seq_subj_ids = []
        
        df_groups = pd.DataFrame({'s': self.subjects, 't': self.tasks})
        unique_groups = df_groups.drop_duplicates().values
        
        # Map subjects to indices
        subj_list = np.unique(self.subjects)
        self.subj_to_idx = {name: i for i, name in enumerate(subj_list)}
        
        for s, t in unique_groups:
            idx = np.where((self.subjects == s) & (self.tasks == t))[0]
            if len(idx) < self.seq_len:
                continue
            
            for i in range(len(idx) - self.seq_len + 1):
                window_idx = idx[i:i+self.seq_len]
                
                # Slices for this window index sequence
                item = {
                    "eye": self.feats_eye[window_idx],
                    "mouth": self.feats_mouth[window_idx],
                    "global_face": self.feats_global_face[window_idx],
                    "prosody": self.feats_prosody[window_idx],
                    "spectral": self.feats_spectral[window_idx],
                    "quality": self.feats_quality[window_idx],
                    "cardio": self.feats_cardio[window_idx],
                    "motion": self.feats_motion[window_idx]
                }
                self.sequences.append(item)
                self.seq_labels.append(self.labels[window_idx[-1]])
                self.seq_subj_ids.append(self.subj_to_idx[s])
                
    def __len__(self):
        return len(self.seq_labels)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.seq_labels[idx]
        subj_id = self.seq_subj_ids[idx]
        
        return (
            torch.FloatTensor(seq["eye"]),
            torch.FloatTensor(seq["mouth"]),
            torch.FloatTensor(seq["global_face"]),
            torch.FloatTensor(seq["prosody"]),
            torch.FloatTensor(seq["spectral"]),
            torch.FloatTensor(seq["quality"]),
            torch.FloatTensor(seq["cardio"]),
            torch.FloatTensor(seq["motion"]),
            torch.LongTensor([label])[0],
            torch.LongTensor([subj_id])[0]
        )

def load_and_align_data(data_dir="certified_data"):
    print("Loading datasets...")
    df_face = pd.read_csv(os.path.join(data_dir, "face_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_voice = pd.read_csv(os.path.join(data_dir, "voice_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    df_physio = pd.read_csv(os.path.join(data_dir, "physio_certified.csv")).drop(columns=['video_id', 'window_start', 'window_end'], errors='ignore')
    
    # Casing normalization
    for df in [df_face, df_voice, df_physio]:
        for col in ['subject_id', 'task_id']:
            df[col] = df[col].astype(str).str.lower().str.strip()
        df['window_index'] = df['window_index'].astype(int)
        
    # Calibration baseline normalization (Subtract subject's calm average)
    lock = FeatureRuntimeLock()
    for df, modality in [(df_face, 'face'), (df_voice, 'voice'), (df_physio, 'physio')]:
        features = [f for f in lock.contract["modalities"][modality]["features"] if f not in EXCLUDED_FEATURES]
        df[features] = df[features].fillna(0)
        
        for subj, subj_df in df.groupby('subject_id'):
            calm_df = subj_df[subj_df['label'] == 0]
            if len(calm_df) > 0:
                mean_calm = calm_df[features].mean().values
            else:
                mean_calm = subj_df[features].mean().values
            
            idx = df[df['subject_id'] == subj].index
            df.loc[idx, features] = df.loc[idx, features] - mean_calm
            
    # Merge datasets
    df_merged = pd.merge(df_face, df_voice, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = pd.merge(df_merged, df_physio, on=['subject_id', 'task_id', 'window_index', 'label'], how='outer')
    df_merged = df_merged.dropna(subset=['label']).sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True).fillna(0)
    
    print(f"Synchronized rows: {len(df_merged)}")
    return df_merged
