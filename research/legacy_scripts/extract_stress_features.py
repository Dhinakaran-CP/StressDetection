import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
import librosa
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Device Configuration for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Feature Extraction Acceleration Device: {DEVICE}")

# MediaPipe face mesh index groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
BROW = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

# ---------------------------------------------------------
# FaceMeshWrapper to support legacy and modern MediaPipe Tasks API
# ---------------------------------------------------------
USE_LEGACY_MEDIAPIPE = False
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    USE_LEGACY_MEDIAPIPE = True
except (ImportError, AttributeError):
    USE_LEGACY_MEDIAPIPE = False

class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None
        
        if self.use_tasks:
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
            if not os.path.exists(self.model_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                print("Downloading Face Landmarker model asset for Tasks API...")
                urllib.request.urlretrieve(url, self.model_path)
            
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            import mediapipe.solutions.face_mesh as mp_face_mesh
            self.fm = mp_face_mesh.FaceMesh(
                static_image_mode=self.static_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def process(self, rgb_image):
        if self.use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            res = self.detector.detect(mp_image)
            
            class LegacyLandmarkResult:
                def __init__(self, landmarks):
                    self.landmark = landmarks
                    
            class LegacyResult:
                def __init__(self, face_landmarks):
                    self.multi_face_landmarks = [LegacyLandmarkResult(face_landmarks[0])]
                    
            class LegacyResultEmpty:
                multi_face_landmarks = None

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                return LegacyResult(res.face_landmarks)
            else:
                return LegacyResultEmpty()
        else:
            return self.fm.process(rgb_image)

    def close(self):
        if self.fm:
            self.fm.close()
        if self.detector:
            self.detector.close()


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
        return self.conv(x)


# ---------------------------------------------------------
# Feature Extraction Pipeline Helper
# ---------------------------------------------------------
def process_single_session(phys_path, audio_path, video_path, labels_map, face_mesh, audio_encoder, video_encoder):
    filename = os.path.basename(phys_path)
    name_part = filename.replace(".txt", "")
    parts = name_part.split("_")
    if len(parts) < 2:
        return None
    subj_id = parts[0]
    task_name = "_".join(parts[1:])
    subj_task = f"{subj_id}_{task_name}"
    
    # 1. Load Raw Physiology Signals (500 Hz)
    try:
        df_physio = pd.read_csv(phys_path)
    except Exception:
        return None
        
    if not {"ECG", "EDA", "RR"}.issubset(df_physio.columns):
        return None
        
    ecg = df_physio["ECG"].values
    eda = df_physio["EDA"].values
    resp = df_physio["RR"].values
    duration_sec = len(df_physio) / 500.0
    
    # 2. Load and Pre-process Audio (GPU-accelerated chunks)
    has_audio = os.path.exists(audio_path)
    audio_data, audio_sr = None, 16000
    if has_audio:
        try:
            audio_data, audio_sr = librosa.load(audio_path, sr=None)
        except Exception:
            has_audio = False

    # 3. Load and Pre-process Video (downsampled image extraction)
    has_video = os.path.exists(video_path)
    video_frames = []
    video_fps = 30.0
    if has_video:
        try:
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Read frame timestamps roughly once per second
            for sec in range(int(duration_sec)):
                frame_num = int(sec * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break
                # Keep rgb image and thumbnail for deep visual CNN
                video_frames.append({
                    "sec": sec,
                    "rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    "resized": cv2.resize(frame, (128, 128))
                })
            cap.release()
        except Exception:
            has_video = False

    # 4. Modality Feature Mapping Loop (1-second steps)
    timeline_features = []
    
    for sec in range(int(duration_sec)):
        # 1-Second Physio Stats
        start_idx = int(sec * 500)
        end_idx = min(len(df_physio), start_idx + 500)
        if start_idx >= len(df_physio):
            break
            
        phys_slice = df_physio.iloc[start_idx:end_idx]
        ecg_slice = phys_slice["ECG"].values
        eda_slice = phys_slice["EDA"].values
        resp_slice = phys_slice["RR"].values
        
        # Audio Segment Features (1-second step)
        rms, zcr, pitch_mean, pitch_std = np.nan, np.nan, np.nan, np.nan
        mfccs = [np.nan] * 13
        deep_audio_feat = [0.0] * 512
        
        if has_audio and audio_data is not None:
            try:
                a_start = int(sec * audio_sr)
                a_end = min(len(audio_data), a_start + audio_sr)
                y_sec = audio_data[a_start:a_end]
                if len(y_sec) > 256:
                    rms = librosa.feature.rms(y=y_sec).mean()
                    zcr = librosa.feature.zero_crossing_rate(y=y_sec).mean()
                    pitches, _, _ = librosa.pyin(y_sec, fmin=100, fmax=500, sr=audio_sr)
                    valid_p = pitches[~np.isnan(pitches)]
                    pitch_mean = np.mean(valid_p) if len(valid_p) > 0 else 0.0
                    pitch_std = np.std(valid_p) if len(valid_p) > 0 else 0.0
                    mfccs = librosa.feature.mfcc(y=y_sec, sr=audio_sr, n_mfcc=13).mean(axis=1).tolist()
                    
                    # GPU Deep feature extraction
                    mel = librosa.feature.melspectrogram(y=y_sec, sr=audio_sr, n_mels=64)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    spec_t = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        deep_audio_feat = audio_encoder(spec_t).squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass
                
        # Video Segment Features (1-second frame match)
        ear, mar, brow_dist = np.nan, np.nan, np.nan
        deep_video_feat = [0.0] * 512
        face_detected = 0.0
        
        if has_video and sec < len(video_frames):
            frame_item = video_frames[sec]
            try:
                # MediaPipe Landmarks (CPU inference wrapper)
                res = face_mesh.process(frame_item["rgb"])
                if res.multi_face_landmarks:
                    face_detected = 1.0
                    landmarks = res.multi_face_landmarks[0].landmark
                    
                    # EAR
                    p_le = [np.array([landmarks[i].x, landmarks[i].y]) for i in LEFT_EYE]
                    ear_l = (np.linalg.norm(p_le[1] - p_le[5]) + np.linalg.norm(p_le[2] - p_le[4])) / (2.0 * np.linalg.norm(p_le[0] - p_le[3]) + 1e-8)
                    p_re = [np.array([landmarks[i].x, landmarks[i].y]) for i in RIGHT_EYE]
                    ear_r = (np.linalg.norm(p_re[1] - p_re[5]) + np.linalg.norm(p_re[2] - p_re[4])) / (2.0 * np.linalg.norm(p_re[0] - p_re[3]) + 1e-8)
                    ear = (ear_l + ear_r) / 2.0
                    
                    # MAR
                    p_lips = [np.array([landmarks[i].x, landmarks[i].y]) for i in LIPS]
                    mar = np.linalg.norm(p_lips[2] - p_lips[7]) / (np.linalg.norm(p_lips[0] - p_lips[5]) + 1e-8)
                    
                    # Brow Distance
                    p_brow = np.array([landmarks[BROW[2]].x, landmarks[BROW[2]].y])
                    p_eye = np.array([landmarks[LEFT_EYE[1]].x, landmarks[LEFT_EYE[1]].y])
                    brow_dist = np.linalg.norm(p_brow - p_eye)
                
                # GPU Deep Video CNN feature extractor
                img_t = torch.FloatTensor(frame_item["resized"].transpose(2, 0, 1) / 255.0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    deep_video_feat = video_encoder(img_t).squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass
                
        # Consolidate 1-second step features
        sec_features = {
            "sec": sec,
            "ecg_val": np.mean(ecg_slice),
            "ecg_std": np.std(ecg_slice),
            "eda_val": np.mean(eda_slice),
            "eda_std": np.std(eda_slice),
            "resp_val": np.mean(resp_slice),
            "resp_std": np.std(resp_slice),
            "voice_rms": rms,
            "voice_zcr": zcr,
            "voice_pitch": pitch_mean,
            "voice_pitch_std": pitch_std,
            "face_ear": ear,
            "face_mar": mar,
            "face_brow": brow_dist,
            "face_detected": face_detected,
            "deep_audio": deep_audio_feat,
            "deep_video": deep_video_feat
        }
        for i, m_val in enumerate(mfccs):
            sec_features[f"voice_mfcc_{i+1}"] = m_val
            
        timeline_features.append(sec_features)
        
    return {
        "subject_id": subj_id,
        "task_id": task_name,
        "subj_task": subj_task,
        "label": labels_map.get(subj_task, 0),
        "timeline": timeline_features
    }


# ---------------------------------------------------------
# Dynamic Multiscale Window Aggregator
# ---------------------------------------------------------
def aggregate_windows(sessions, window_sec):
    step_sec = window_sec / 2.0  # 50% overlap
    records = []
    
    for sess in sessions:
        timeline = sess["timeline"]
        n_secs = len(timeline)
        
        # Determine sliding windows
        n_windows = int((n_secs - window_sec) // step_sec) + 1
        if n_windows <= 0:
            continue
            
        for w_idx in range(n_windows):
            start_sec = int(w_idx * step_sec)
            end_sec = start_sec + window_sec
            
            slice_timeline = timeline[start_sec:end_sec]
            
            # --- 1. ECG Features ---
            ecg_vals = [s["ecg_val"] for s in slice_timeline if not np.isnan(s["ecg_val"])]
            ecg_hr = len(ecg_vals) * 60.0 / window_sec if len(ecg_vals) > 0 else np.nan
            ecg_mean = np.mean(ecg_vals) if len(ecg_vals) > 0 else 0.0
            ecg_std = np.std(ecg_vals) if len(ecg_vals) > 0 else 0.0
            
            # --- 2. EDA Features (Tonic/Phasic filters) ---
            eda_vals = np.array([s["eda_val"] for s in slice_timeline if not np.isnan(s["eda_val"])])
            eda_tonic, eda_phasic = 0.0, 0.0
            if len(eda_vals) > 2:
                # lowpass tonic proxy
                eda_tonic = np.mean(eda_vals)
                # phasic residual proxy
                eda_phasic = np.mean(np.abs(eda_vals - eda_tonic))
                
            # --- 3. Respiration Features ---
            resp_vals = [s["resp_val"] for s in slice_timeline if not np.isnan(s["resp_val"])]
            resp_mean = np.mean(resp_vals) if len(resp_vals) > 0 else 0.0
            resp_std = np.std(resp_vals) if len(resp_vals) > 0 else 0.0
            
            # --- 4. Handcrafted Audio Features ---
            voice_rms = np.nanmean([s["voice_rms"] for s in slice_timeline])
            voice_zcr = np.nanmean([s["voice_zcr"] for s in slice_timeline])
            voice_pitch = np.nanmean([s["voice_pitch"] for s in slice_timeline])
            voice_pitch_std = np.nanmean([s["voice_pitch_std"] for s in slice_timeline])
            
            # --- 5. Handcrafted Video Features ---
            face_ear = np.nanmean([s["face_ear"] for s in slice_timeline])
            face_mar = np.nanmean([s["face_mar"] for s in slice_timeline])
            face_brow = np.nanmean([s["face_brow"] for s in slice_timeline])
            
            # Face confidence score
            face_conf = np.mean([s["face_detected"] for s in slice_timeline])
            
            # --- 6. GPU Embeddings Aggregation ---
            deep_audios = np.array([s["deep_audio"] for s in slice_timeline])
            deep_videos = np.array([s["deep_video"] for s in slice_timeline])
            
            mean_deep_audio = deep_audios.mean(axis=0) if len(deep_audios) > 0 else np.zeros(512)
            mean_deep_video = deep_videos.mean(axis=0) if len(deep_videos) > 0 else np.zeros(512)
            
            # Quality scores & continuity checks (§7.4)
            quality_score = 1.0
            if np.isnan(face_ear) or face_conf < 0.2:
                quality_score -= 0.3  # penalize face tracking loss
            if np.isnan(voice_rms):
                quality_score -= 0.3  # penalize silent audio
            if np.isnan(ecg_hr):
                quality_score -= 0.4  # heavy penalty on missing physiological signal
                
            physio_continuity = 1.0 if not np.isnan(ecg_hr) else 0.0
            
            # Pack record
            record = {
                "subject_id": sess["subject_id"],
                "task_id": sess["task_id"],
                "window_index": w_idx,
                "label": sess["label"],
                "ecg_hr": ecg_hr,
                "ecg_mean": ecg_mean,
                "ecg_std": ecg_std,
                "eda_tonic_mean": eda_tonic,
                "eda_phasic_mean": eda_phasic,
                "resp_rate_mean": resp_mean,
                "resp_std": resp_std,
                "voice_rms_mean": voice_rms,
                "voice_zcr_mean": voice_zcr,
                "voice_pitch_mean": voice_pitch,
                "voice_pitch_std": voice_pitch_std,
                "face_ear_mean": face_ear,
                "face_mar_mean": face_mar,
                "face_brow_mean": face_brow,
                "face_confidence": face_conf,
                "quality_score": max(0.0, quality_score),
                "physio_continuity_flag": physio_continuity
            }
            
            # Add deep audio embeddings
            for i, val in enumerate(mean_deep_audio):
                record[f"voice_deep_embed_{i+1}"] = float(val)
            # Add deep face embeddings
            for i, val in enumerate(mean_deep_video):
                record[f"face_deep_embed_{i+1}"] = float(val)
                
            # Populate absolute copy columns for dual-relative calibration
            for k, v in list(record.items()):
                if k not in ["subject_id", "task_id", "window_index", "label"]:
                    record[f"{k}_abs"] = v
                    
            records.append(record)
            
    # Apply baseline calibration z-scores (Layer E)
    if len(records) > 0:
        df = pd.DataFrame(records)
        calibrated_cols = [c for c in df.columns if c not in ["subject_id", "task_id", "window_index", "label"] and not c.endswith("_abs")]
        
        for subj in df["subject_id"].unique():
            subj_mask = df["subject_id"] == subj
            calm_mask = subj_mask & df["task_id"].str.lower().str.contains("breathing|relax")
            if calm_mask.sum() == 0:
                calm_mask = subj_mask
                
            for col in calibrated_cols:
                mean_val = df.loc[calm_mask, col].mean()
                std_val = df.loc[calm_mask, col].std() or 1.0
                df.loc[subj_mask, col] = (df.loc[subj_mask, col] - mean_val) / (std_val + 1e-8)
        return df
    return pd.DataFrame()


def main():
    dataset_dir = "StressID Dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: Folder '{dataset_dir}' not found.")
        sys.exit(1)
        
    print("Initializing MediaPipe Face Mesh...")
    face_mesh = FaceMeshWrapper(static_mode=True)
    
    # Initialize GPU models
    audio_encoder = DeepAudioCNN().to(DEVICE).eval()
    video_encoder = DeepVideoCNN().to(DEVICE).eval()
    
    # Load labels mapping
    labels_path = os.path.join(dataset_dir, "labels.csv")
    labels_map = {}
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        labels_map = dict(zip(df_labels['subject/task'], df_labels['binary-stress']))
    
    physio_files = glob.glob(os.path.join(dataset_dir, "Physiological", "**", "*.txt"), recursive=True)
    # Ignore resource forks starting with "._"
    physio_files = [f for f in physio_files if not os.path.basename(f).startswith("._")]
    print(f"Found {len(physio_files)} physiological session files to process.")
    
    extracted_sessions = []
    
    # Extract session signals at 1-second steps (single pass)
    for f_path in tqdm(physio_files, desc="Processing Sessions"):
        filename = os.path.basename(f_path)
        name_part = filename.replace(".txt", "")
        parts = name_part.split("_")
        if len(parts) < 2:
            continue
        subj_id = parts[0]
        task_name = "_".join(parts[1:])
        subj_task = f"{subj_id}_{task_name}"
        
        audio_path = os.path.join(dataset_dir, "Audio", subj_id, f"{subj_task}.wav")
        video_path = os.path.join(dataset_dir, "Videos", subj_id, f"{subj_task}.mp4")
        
        res = process_single_session(
            phys_path=f_path,
            audio_path=audio_path,
            video_path=video_path,
            labels_map=labels_map,
            face_mesh=face_mesh,
            audio_encoder=audio_encoder,
            video_encoder=video_encoder
        )
        if res is not None:
            extracted_sessions.append(res)
            
    # Generate multi-scale window stores (2s, 5s, 10s) (§6.2)
    for w_sec in [2, 5, 10]:
        print(f"\nAggregating and calibrating {w_sec}s windows...")
        df_win = aggregate_windows(extracted_sessions, w_sec)
        if not df_win.empty:
            # 1. Fusion Store (Concatenated Raw + Calibrated)
            fusion_path = f"stress_features_fusion_{w_sec}s.csv"
            df_win.to_csv(fusion_path, index=False)
            print(f"Saved Fusion Store to: {os.path.abspath(fusion_path)}")
            
            # Legacy compatibility file
            store_path = f"stress_features_store_{w_sec}s.csv"
            df_win.to_csv(store_path, index=False)
            
            # 2. Raw Store (Absolute Features only)
            raw_cols = [c for c in df_win.columns if c.endswith("_abs") or c in ["subject_id", "task_id", "window_index", "label", "quality_score", "physio_continuity_flag"]]
            df_raw = df_win[raw_cols].copy()
            # Rename absolute columns to remove the "_abs" suffix for clean usage
            rename_map = {c: c[:-4] for c in df_raw.columns if c.endswith("_abs")}
            df_raw.rename(columns=rename_map, inplace=True)
            raw_path = f"stress_features_raw_{w_sec}s.csv"
            df_raw.to_csv(raw_path, index=False)
            print(f"Saved Raw Store to: {os.path.abspath(raw_path)}")
            
            # 3. Calibrated Store (Baseline-relative z-scores only)
            calib_cols = [c for c in df_win.columns if not c.endswith("_abs") or c in ["subject_id", "task_id", "window_index", "label", "quality_score", "physio_continuity_flag"]]
            df_calib = df_win[calib_cols].copy()
            calib_path = f"stress_features_calibrated_{w_sec}s.csv"
            df_calib.to_csv(calib_path, index=False)
            print(f"Saved Calibrated Store to: {os.path.abspath(calib_path)}")
            
    print("\nExtraction process completed successfully!")


if __name__ == "__main__":
    main()
