import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile
import librosa
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
FS_PHYSIO = 500  # 500 Hz physiological sampling rate
WINDOW_SEC = 5   # 5 seconds
STEP_SEC = 2.5   # 50% overlap (2.5 seconds step)
WINDOW_LEN_PHYSIO = int(WINDOW_SEC * FS_PHYSIO)
STEP_LEN_PHYSIO = int(STEP_SEC * FS_PHYSIO)

# Device Configuration for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Feature Extraction Acceleration Device: {DEVICE}")

# MediaPipe face mesh index groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
BROW = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

# ---------------------------------------------------------
# GPU Deep Feature Encoders (PyTorch CUDA models)
# ---------------------------------------------------------
class DeepAudioCNN(nn.Module):
    """Deep CNN for extracting acoustic embeddings on GPU."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
    def forward(self, x):
        # Input shape: [batch, 1, mel_bins, time_steps]
        return self.conv(x)

class DeepVideoCNN(nn.Module):
    """Deep CNN for extracting facial motion embeddings on GPU."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
    def forward(self, x):
        # Input shape: [batch, 3, height, width]
        return self.conv(x)

# ---------------------------------------------------------
# Modality Feature Extractors
# ---------------------------------------------------------
def extract_ecg_features(ecg_signal):
    """Extract HR, RMSSD, and SDNN features from ECG segment."""
    try:
        nyq = 0.5 * FS_PHYSIO
        b, a = signal.butter(3, [0.5 / nyq, 45.0 / nyq], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Minimum peak height and distance of 0.5 sec (120 bpm limit)
        peaks, _ = signal.find_peaks(filtered, height=np.max(filtered)*0.5, distance=int(0.5 * FS_PHYSIO))
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / FS_PHYSIO
            hr = 60.0 / np.mean(rr_intervals)
            sdnn = np.std(rr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        else:
            hr, sdnn, rmssd = np.nan, np.nan, np.nan
    except Exception:
        hr, sdnn, rmssd = np.nan, np.nan, np.nan
        
    return {
        "ecg_hr": hr,
        "ecg_sdnn": sdnn,
        "ecg_rmssd": rmssd,
        "ecg_mean": np.mean(ecg_signal),
        "ecg_std": np.std(ecg_signal)
    }

def extract_eda_features(eda_signal):
    """Extract SCL (tonic) and SCR (phasic) features from EDA segment."""
    try:
        # Low-pass filter for SCL tonic component (cutoff 0.5 Hz)
        nyq = 0.5 * FS_PHYSIO
        b_tonic, a_tonic = signal.butter(2, 0.5 / nyq, btype='low')
        tonic = signal.filtfilt(b_tonic, a_tonic, eda_signal)
        
        # Phasic component is the residual
        phasic = eda_signal - tonic
        
        # Count SCR peaks on phasic signal
        scr_peaks, _ = signal.find_peaks(phasic, height=0.01, distance=int(1.0 * FS_PHYSIO))
        scr_count = len(scr_peaks)
        scr_amp = np.max(phasic[scr_peaks]) if scr_count > 0 else 0.0
    except Exception:
        tonic = eda_signal
        phasic = np.zeros_like(eda_signal)
        scr_count, scr_amp = 0.0, 0.0
        
    return {
        "eda_tonic_mean": np.mean(tonic),
        "eda_tonic_std": np.std(tonic),
        "eda_phasic_mean": np.mean(phasic),
        "eda_phasic_std": np.std(phasic),
        "eda_scr_count": float(scr_count),
        "eda_scr_amplitude": float(scr_amp)
    }

def extract_resp_features(resp_signal):
    """Extract respiration rate and variance features from respiration segment."""
    try:
        # Zero-crossings rate to estimate respiration rate
        zero_crossings = np.where(np.diff(np.sign(resp_signal - np.mean(resp_signal))))[0]
        resp_rate = (len(zero_crossings) / 2.0) / WINDOW_SEC
    except Exception:
        resp_rate = np.nan
        
    return {
        "resp_rate_mean": resp_rate,
        "resp_std": np.std(resp_signal),
        "resp_mean": np.mean(resp_signal)
    }

def extract_audio_features(wav_path, start_sec, audio_encoder):
    """Extract speech features (MFCC, pitch) and GPU deep acoustic embeddings."""
    features = {}
    try:
        y, sr = librosa.load(wav_path, sr=None, offset=start_sec, duration=WINDOW_SEC)
        if len(y) == 0:
            return features
        
        # Classical Handcrafted features
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        
        # Estimate pitch contour
        pitches, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        valid_pitches = pitches[~np.isnan(pitches)]
        pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0.0
        pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0.0
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        
        features = {
            "voice_rms_mean": float(rms),
            "voice_zcr_mean": float(zcr),
            "voice_pitch_mean": float(pitch_mean),
            "voice_pitch_std": float(pitch_std)
        }
        for i, val in enumerate(mfccs):
            features[f"voice_mfcc_{i+1}"] = float(val)
            
        # GPU Accelerated Deep Feature Extraction (Wav2Vec-style CNN wrapper)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to PyTorch tensor and move to GPU
        spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, 64, time_steps]
        with torch.no_grad():
            deep_embeddings = audio_encoder(spec_tensor).squeeze(0).cpu().numpy()
            
        for i, val in enumerate(deep_embeddings):
            features[f"voice_deep_embed_{i+1}"] = float(val)
            
    except Exception:
        pass
    return features

def extract_video_features(mp4_path, start_sec, face_mesh, video_encoder):
    """Extract face landmarks (EAR/MAR) and GPU deep visual CNN features."""
    features = {}
    try:
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        ears, mars, brow_dists = [], [], []
        frames_list = []
        
        for offset in range(int(WINDOW_SEC)):
            frame_num = int((start_sec + offset) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Collect frames for Deep GPU processing (resize for efficiency)
            resized_frame = cv2.resize(frame, (128, 128))
            frames_list.append(resized_frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Compute EAR (Eye Aspect Ratio)
                def get_ear(eye_indices):
                    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
                    v_dist = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
                    h_dist = np.linalg.norm(p[0] - p[3])
                    return v_dist / (2.0 * h_dist + 1e-8)
                
                ear_l = get_ear(LEFT_EYE)
                ear_r = get_ear(RIGHT_EYE)
                ears.append((ear_l + ear_r) / 2.0)
                
                # Compute MAR (Mouth Aspect Ratio)
                p_lips = [np.array([landmarks[i].x, landmarks[i].y]) for i in LIPS]
                mar = np.linalg.norm(p_lips[2] - p_lips[7]) / (np.linalg.norm(p_lips[0] - p_lips[5]) + 1e-8)
                mars.append(mar)
                
                # Compute Brow-to-Eye distance
                p_brow = np.array([landmarks[BROW[2]].x, landmarks[BROW[2]].y])
                p_eye = np.array([landmarks[LEFT_EYE[1]].x, landmarks[LEFT_EYE[1]].y])
                brow_dists.append(np.linalg.norm(p_brow - p_eye))
                
        cap.release()
        
        if len(ears) > 0:
            features = {
                "face_ear_mean": float(np.mean(ears)),
                "face_ear_std": float(np.std(ears)),
                "face_mar_mean": float(np.mean(mars)),
                "face_mar_std": float(np.std(mars)),
                "face_brow_mean": float(np.mean(brow_dists)),
                "face_brow_std": float(np.std(brow_dists))
            }
            
        # GPU Accelerated Deep Video feature extraction
        if len(frames_list) > 0:
            # Convert to PyTorch tensor [batch, 3, 128, 128]
            tensor_frames = np.stack(frames_list, axis=0).transpose(0, 3, 1, 2)
            tensor_frames = torch.FloatTensor(tensor_frames / 255.0).to(DEVICE)
            
            with torch.no_grad():
                deep_embeddings = video_encoder(tensor_frames).mean(dim=0).cpu().numpy()
                
            for i, val in enumerate(deep_embeddings):
                features[f"face_deep_embed_{i+1}"] = float(val)
                
    except Exception:
        pass
    return features

def main():
    dataset_dir = "StressID Dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: Folder '{dataset_dir}' not found.")
        sys.exit(1)
        
    print("Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    # Initialize GPU models
    audio_encoder = DeepAudioCNN().to(DEVICE).eval()
    video_encoder = DeepVideoCNN().to(DEVICE).eval()
    
    # Load labels
    labels_path = os.path.join(dataset_dir, "labels.csv")
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        labels_map = dict(zip(df_labels['subject/task'], df_labels['binary-stress']))
    else:
        labels_map = {}
        print("Warning: labels.csv not found. Extracted features will not have stress targets.")

    # Find physiological files
    physio_files = glob.glob(os.path.join(dataset_dir, "Physiological", "**", "*.txt"), recursive=True)
    print(f"Found {len(physio_files)} physiological session files.")
    
    records = []
    
    # Process each subject and task
    for f_path in tqdm(physio_files, desc="Extracting Features"):
        filename = os.path.basename(f_path)
        # Expected filename: subject_task.txt
        name_part = filename.replace(".txt", "")
        parts = name_part.split("_")
        if len(parts) < 2:
            continue
        subj_id = parts[0]
        task_name = "_".join(parts[1:])
        subj_task = f"{subj_id}_{task_name}"
        
        # Load physio data
        try:
            df_physio = pd.read_csv(f_path)
        except Exception:
            continue
            
        if not {"ECG", "EDA", "RR"}.issubset(df_physio.columns):
            continue
            
        n_rows = len(df_physio)
        n_windows = (n_rows - WINDOW_LEN_PHYSIO) // STEP_LEN_PHYSIO + 1
        if n_windows <= 0:
            continue
            
        # Audio / Video path references
        audio_path = os.path.join(dataset_dir, "Audio", subj_id, f"{subj_task}.wav")
        video_path = os.path.join(dataset_dir, "Videos", subj_id, f"{subj_task}.mp4")
        
        has_audio = os.path.exists(audio_path)
        has_video = os.path.exists(video_path)
        
        # Process each window
        for w_idx in range(n_windows):
            start_row = w_idx * STEP_LEN_PHYSIO
            end_row = start_row + WINDOW_LEN_PHYSIO
            start_sec = start_row / FS_PHYSIO
            
            # Slice physio signals
            segment = df_physio.iloc[start_row:end_row]
            
            # 1. Physio extraction (CPU is optimal here)
            f_ecg = extract_ecg_features(segment["ECG"].values)
            f_eda = extract_eda_features(segment["EDA"].values)
            f_resp = extract_resp_features(segment["RR"].values)
            
            # 2. Audio extraction (uses GPU for deep audio CNN)
            f_audio = {}
            if has_audio:
                f_audio = extract_audio_features(audio_path, start_sec, audio_encoder)
                
            # 3. Video extraction (uses GPU for deep video CNN)
            f_video = {}
            if has_video:
                f_video = extract_video_features(video_path, start_sec, face_mesh, video_encoder)
                
            # Merge features
            record = {
                "subject_id": subj_id,
                "task_id": task_name,
                "window_index": w_idx,
                "label": labels_map.get(subj_task, 0)
            }
            record.update(f_ecg)
            record.update(f_eda)
            record.update(f_resp)
            record.update(f_audio)
            record.update(f_video)
            
            # Set absolute proxy columns for future dual-relative calibration
            for k, v in list(record.items()):
                if k not in ["subject_id", "task_id", "window_index", "label"]:
                    record[f"{k}_abs"] = v
                    
            records.append(record)

    if len(records) == 0:
        print("No records extracted. Please check dataset paths.")
        return

    df_out = pd.DataFrame(records)
    
    # Baseline Calibration (z-score normalization using Breathing or Relax tasks)
    print("Calibrating features against subject calm baselines...")
    calibrated_cols = [c for c in df_out.columns if c not in ["subject_id", "task_id", "window_index", "label"] and not c.endswith("_abs")]
    
    for subj in df_out["subject_id"].unique():
        subj_mask = df_out["subject_id"] == subj
        # Find neutral calm task
        calm_mask = subj_mask & df_out["task_id"].str.lower().str.contains("breathing|relax")
        if calm_mask.sum() == 0:
            calm_mask = subj_mask  # fallback to subject mean
            
        for col in calibrated_cols:
            calm_mean = df_out.loc[calm_mask, col].mean()
            calm_std = df_out.loc[calm_mask, col].std() or 1.0
            
            # Replace primary feature with baseline relative deviation
            # Keep raw under '_abs' suffix
            df_out.loc[subj_mask, col] = (df_out.loc[subj_mask, col] - calm_mean) / (calm_std + 1e-8)

    # Save outputs
    out_path = "stress_features_store.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSuccessfully extracted {len(df_out)} synchronized records.")
    print(f"Features saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
