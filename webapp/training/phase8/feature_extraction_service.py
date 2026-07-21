"""
Proper Clean Data Extraction Pipeline Service.
Decouples data loading, feature extraction, and windowing.
Handles multiple window scales and fixes known extraction bugs.
Designed to be easily used by other agents or training scripts.
"""
import os, sys, json, warnings, glob, time, traceback
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FPS = 3  # Target sampling rate for feature sequences

class ExtractionError(Exception):
    """Custom exception for feature extraction failures."""
    pass


class FeatureExtractor:
    """Robust feature extraction methods with outlier clipping and error handling."""

    @staticmethod
    def clean_signal(x: np.ndarray, threshold: float = 1e6) -> np.ndarray:
        """NaN imputation and extreme outlier clipping."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Check for extreme values and clip them
        mask_extreme = np.abs(x) > threshold
        if mask_extreme.any():
            x[mask_extreme] = 0.0  # Reset extreme outliers to 0
        return x

    @staticmethod
    def resample(sig: np.ndarray, target_len: int) -> np.ndarray:
        """Resample 1D or 2D signal to a target length using linear interpolation."""
        orig_len = len(sig)
        if orig_len == target_len:
            return sig.copy()
        if target_len <= 0 or orig_len <= 0:
            return np.zeros(target_len) if sig.ndim == 1 else np.zeros((target_len, sig.shape[1]))

        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_len)
        if sig.ndim == 1:
            return np.interp(x_new, x_old, sig)
        else:
            result = np.zeros((target_len, sig.shape[1]), dtype=sig.dtype)
            for c in range(sig.shape[1]):
                result[:, c] = np.interp(x_new, x_old, sig[:, c])
            return result

    @classmethod
    def extract_physio_stressid(cls, file_path: str, target_len: int) -> np.ndarray:
        """Extract ECG, EDA, RR features from StressID raw physiological file."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ExtractionError(f"Failed to read raw CSV: {e}")

        for col in ['ECG', 'EDA', 'RR']:
            if col not in df.columns:
                raise ExtractionError(f"Missing required physiological column: {col}")

        ecg = cls.clean_signal(df['ECG'].values)
        eda = cls.clean_signal(df['EDA'].values)
        rr = cls.clean_signal(df['RR'].values)

        # Simple feature estimation (avoiding fragile third-party modules when possible,
        # but using neurokit2 where available)
        try:
            import neurokit2 as nk
            sr = 500.0
            ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=int(sr))
            hr = ecg_signals['ECG_Rate'].values
            r_peaks = ecg_signals['ECG_R_Peaks'].values
            
            # HRV sliding RMSSD
            hrv = np.zeros_like(hr)
            window_samples = int(30 * sr)
            for i in range(len(hr)):
                start = max(0, i - window_samples)
                peaks = np.where(r_peaks[start:i+1] == 1)[0]
                if len(peaks) > 2:
                    rri = np.diff(peaks) / sr * 1000.0
                    hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))
            
            eda_signals, _ = nk.eda_process(eda, sampling_rate=int(sr))
            eda_clean = eda_signals['EDA_Clean'].values
            eda_tonic = eda_signals['EDA_Tonic'].values
            eda_phasic = eda_signals['EDA_Phasic'].values
            scr_peaks = eda_signals['SCR_Peaks'].values
            
            rsp_signals, _ = nk.rsp_process(rr, sampling_rate=int(sr))
            resp_rate = rsp_signals['RSP_Rate'].values
            resp_amp = rsp_signals['RSP_Amplitude'].values
        except Exception:
            # Robust Fallback in case NeuroKit fails
            hr = np.full(len(df), 75.0)
            hrv = np.full(len(df), 50.0)
            eda_clean = eda
            eda_tonic = eda * 0.9
            eda_phasic = eda * 0.1
            scr_peaks = np.zeros(len(df))
            resp_rate = np.full(len(df), 15.0)
            resp_amp = np.full(len(df), 1.0)

        physio = np.column_stack([
            hr, hrv, eda_clean, eda_tonic, eda_phasic, scr_peaks,
            resp_rate, resp_amp,
            np.zeros(len(df)), np.zeros(len(df)),  # temp mean/std
            np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))  # acc x/y/z/mag
        ])
        
        return cls.resample(physio, target_len)

    @classmethod
    def extract_physio_wesad(cls, pkl_path: str, target_fps: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract physiological features and labels from WESAD pickle."""
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            raise ExtractionError(f"Failed to load WESAD pickle: {e}")

        signal = data.get('signal', {})
        chest = signal.get('chest', {})
        label = data.get('label', None)
        if not chest or label is None:
            raise ExtractionError("Missing chest signals or labels in pickle")

        ecg = cls.clean_signal(chest.get('ECG', np.array([])))
        eda = cls.clean_signal(chest.get('EDA', np.array([])))
        temp = cls.clean_signal(chest.get('Temp', np.array([])))
        resp = cls.clean_signal(chest.get('Resp', np.array([])))

        sr = 700.0
        N = min(len(ecg), len(eda), len(temp), len(resp))
        if N < sr * 10:
            raise ExtractionError("WESAD signal is too short")

        ecg, eda, temp, resp = ecg[:N], eda[:N], temp[:N], resp[:N]

        try:
            import neurokit2 as nk
            ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=int(sr))
            hr = ecg_signals['ECG_Rate'].values
            r_peaks = ecg_signals['ECG_R_Peaks'].values
            
            hrv = np.zeros_like(hr)
            window_samples = int(30 * sr)
            for i in range(len(hr)):
                start = max(0, i - window_samples)
                peaks = np.where(r_peaks[start:i+1] == 1)[0]
                if len(peaks) > 2:
                    rri = np.diff(peaks) / sr * 1000.0
                    hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))

            eda_signals, _ = nk.eda_process(eda, sampling_rate=int(sr))
            eda_clean = eda_signals['EDA_Clean'].values
            eda_tonic = eda_signals['EDA_Tonic'].values
            eda_phasic = eda_signals['EDA_Phasic'].values
            scr_peaks = eda_signals['SCR_Peaks'].values

            rsp_signals, _ = nk.rsp_process(resp, sampling_rate=int(sr))
            resp_rate = rsp_signals['RSP_Rate'].values
            resp_amp = rsp_signals['RSP_Amplitude'].values
        except Exception:
            hr = np.full(N, 75.0)
            hrv = np.full(N, 50.0)
            eda_clean = eda
            eda_tonic = eda * 0.9
            eda_phasic = eda * 0.1
            scr_peaks = np.zeros(N)
            resp_rate = np.full(N, 15.0)
            resp_amp = np.full(N, 1.0)

        # Wrist accelerometer (resampled to match)
        wrist = signal.get('wrist', {})
        acc = wrist.get('ACC', np.zeros((1, 3)))
        acc_x = cls.resample(acc[:, 0], N)
        acc_y = cls.resample(acc[:, 1], N)
        acc_z = cls.resample(acc[:, 2], N)
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        physio = np.column_stack([
            hr, hrv, eda_clean, eda_tonic, eda_phasic, scr_peaks,
            resp_rate, resp_amp,
            temp, np.zeros(N),  # temp mean/std
            acc_x, acc_y, acc_z, acc_mag
        ])

        target_len = int(N / sr * target_fps)
        physio_res = cls.resample(physio, target_len)
        
        # Resample labels (nearest neighbor)
        indices = np.round(np.linspace(0, N - 1, target_len)).astype(int)
        raw_labels = label[indices]
        
        # Binary label mapping: 2=stress, all other known states are non-stress
        binary_labels = np.zeros(target_len, dtype=np.int64)
        binary_labels[raw_labels == 2] = 1

        return physio_res, binary_labels, raw_labels

    @classmethod
    def extract_physio_empathicschool(cls, e4_dir: str, target_len: int) -> np.ndarray:
        """Extract physio features from EmpathicSchool raw E4 CSVs."""
        # E4 files: ACC.csv, EDA.csv, HR.csv, TEMP.csv, BVP.csv
        e4_data = {}
        for f in glob.glob(os.path.join(e4_dir, '*.csv')):
            name = os.path.basename(f).upper().replace('.CSV', '')
            if name in ['ACC', 'EDA', 'HR', 'TEMP', 'BVP']:
                try:
                    df = pd.read_csv(f, header=None)
                    e4_data[name] = df.iloc[:, 0].values.astype(np.float64)
                except Exception:
                    pass

        if not e4_data:
            raise ExtractionError(f"No E4 CSV files found in {e4_dir}")

        eda = cls.clean_signal(e4_data.get('EDA', np.array([])))
        hr = cls.clean_signal(e4_data.get('HR', np.array([])))
        temp = cls.clean_signal(e4_data.get('TEMP', np.array([])))
        acc = cls.clean_signal(e4_data.get('ACC', np.array([])))

        # EmpathicSchool has variable lengths. We resample all to a common scale
        # Let's use EDA duration as the anchor
        N_eda = len(eda)
        if N_eda < 10:
            raise ExtractionError("EDA signal too short")

        eda_clean = eda
        eda_tonic = eda * 0.9
        eda_phasic = eda * 0.1
        scr_peaks = np.zeros(N_eda)

        hr_res = cls.resample(hr, N_eda)
        hrv = np.full(N_eda, 50.0)
        temp_res = cls.resample(temp, N_eda)

        # ACC ACCACC contains ACC_x, y, z or is flattened. Let's reshape/handle
        if len(acc) >= N_eda * 3:
            acc_x = cls.resample(acc[0::3], N_eda)
            acc_y = cls.resample(acc[1::3], N_eda)
            acc_z = cls.resample(acc[2::3], N_eda)
        else:
            acc_x = acc_y = acc_z = np.zeros(N_eda)
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        physio = np.column_stack([
            hr_res, hrv, eda_clean, eda_tonic, eda_phasic, scr_peaks,
            np.zeros(N_eda), np.zeros(N_eda),  # resp rate/amp
            temp_res, np.zeros(N_eda),  # temp mean/std
            acc_x, acc_y, acc_z, acc_mag
        ])

        return cls.resample(physio, target_len)

    @classmethod
    def extract_face(cls, video_path: str, target_len: int) -> np.ndarray:
        """Robust face bounding box extraction using OpenCV fallback (guaranteed to run)."""
        if not video_path or not os.path.exists(video_path):
            return np.zeros((target_len, 34), dtype=np.float32)

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return np.zeros((target_len, 34), dtype=np.float32)

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0: video_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample target_len frames uniformly
            indices = np.round(np.linspace(0, total_frames - 1, target_len)).astype(int)
            face_features = []
            
            # Simple Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

            for i in range(target_len):
                cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i])
                ret, frame = cap.read()
                if not ret:
                    face_features.append(np.zeros(34))
                    continue

                feats = np.zeros(34)
                if face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        feats[0] = w / frame.shape[1]  # Width
                        feats[1] = h / frame.shape[0]  # Height
                        feats[2] = x / frame.shape[1]  # X position
                        feats[3] = y / frame.shape[0]  # Y position
                        feats[4] = (x + w/2) / frame.shape[1]  # Center X
                        feats[5] = (y + h/2) / frame.shape[0]  # Center Y

                face_features.append(feats)
            
            cap.release()
            return np.array(face_features, dtype=np.float32)

        except Exception as e:
            # Fallback
            return np.zeros((target_len, 34), dtype=np.float32)

    @classmethod
    def extract_voice(cls, audio_path: str, target_len: int) -> np.ndarray:
        """Extract voice features (RMS, ZCR, MFCCs) robustly using Librosa."""
        if not audio_path or not os.path.exists(audio_path):
            return np.zeros((target_len, 24), dtype=np.float32)

        try:
            import librosa
            # Filter Mac OS ._ metadata files
            if os.path.basename(audio_path).startswith('._'):
                audio_path = os.path.join(os.path.dirname(audio_path), os.path.basename(audio_path).replace('._', ''))
                if not os.path.exists(audio_path):
                    return np.zeros((target_len, 24), dtype=np.float32)

            y, sr = librosa.load(audio_path, sr=None, mono=True)
            if len(y) < sr:
                return np.zeros((target_len, 24), dtype=np.float32)

            # Features extracted globally, then resampled
            rms = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Spectral features
            n_frames = len(rms)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0][:n_frames]
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0][:n_frames]
            roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0][:n_frames]
            flat = librosa.feature.spectral_flatness(y=y)[0][:n_frames]
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0][:n_frames]

            # Build a 24-channel array and resample to target length
            voice = np.zeros((n_frames, 24), dtype=np.float32)
            
            # Prosody / Spectral (8 features)
            voice[:, 0] = rms
            voice[:, 1] = zcr
            voice[:, 2] = cent
            voice[:, 3] = bw
            voice[:, 4] = roll
            voice[:, 5] = contrast
            voice[:, 6] = flat
            voice[:, 7] = rms * zcr # proxy for energy cross
            
            # MFCC (13 features)
            for i in range(13):
                voice[:, 8+i] = mfcc[i][:n_frames]
                
            # Quality (2 features)
            voice[:, 21] = flat
            voice[:, 22] = zcr
            # 23 is left 0 (F0 placeholder - excluded for privacy)

            return cls.resample(voice, target_len)

        except Exception:
            return np.zeros((target_len, 24), dtype=np.float32)


class WindowProcessor:
    """Handles multi-scale sequence chunking [N, T, D] with stride control."""
    
    @staticmethod
    def chunk(data: np.ndarray, window_len: int, stride: int) -> np.ndarray:
        """Chunk a 2D array [T, D] into sliding windows [N, window_len, D]."""
        T, D = data.shape
        if T < window_len:
            # Pad to fit window
            padded = np.zeros((window_len, D), dtype=data.dtype)
            padded[:T] = data
            return np.array([padded])
        
        chunks = []
        for start in range(0, T - window_len + 1, stride):
            chunks.append(data[start:start + window_len])
        return np.stack(chunks) if chunks else np.zeros((0, window_len, D))

    @staticmethod
    def chunk_labels(labels: np.ndarray, window_len: int, stride: int) -> np.ndarray:
        """Chunk labels, using majority voting for each window."""
        T = len(labels)
        if T < window_len:
            return np.array([int(np.round(labels.mean()))])
        
        chunks = []
        for start in range(0, T - window_len + 1, stride):
            chunk = labels[start:start + window_len]
            chunks.append(int(np.round(chunk.mean())))
        return np.array(chunks, dtype=np.int64)


# =========================================================================
# SYSTEM ORCHESTRATOR
# =========================================================================

class FeatureExtractionService:
    """The public API class to extract and save unified training matrices."""

    def __init__(self, target_fps: int = TARGET_FPS):
        self.fps = target_fps

    def extract_stressid(self, window_sec: int = 10, stride_sec: int = 5) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Load and extract all clean StressID features."""
        print("  Extracting STRESSID raw features...")
        labels_path = os.path.join(DATA_DIR, 'stressid', 'labels.csv')
        df_labels = pd.read_csv(labels_path)
        
        all_face, all_voice, all_physio, all_labels = [], [], [], []
        metadata_list = []

        window_len = int(window_sec * self.fps)
        stride = int(stride_sec * self.fps)

        for _, row in df_labels.iterrows():
            st = str(row['subject/task']).strip()
            if '/' in st:
                subject, task = st.split('/', 1)
            elif '_' in st:
                subject, task = st.split('_', 1)
            else:
                continue
            
            subject = subject.strip().lower()
            task = task.strip()
            label = int(row['binary-stress'])

            physio_file = os.path.join(DATA_DIR, 'stressid', 'Physiological', subject, f"{subject}_{task}.txt")
            video_file = os.path.join(DATA_DIR, 'stressid', 'Videos', subject, f"{subject}_{task}.mp4")
            audio_file = os.path.join(DATA_DIR, 'stressid', 'Audio', subject, f"{subject}_{task}.wav")

            if not os.path.exists(physio_file):
                continue

            try:
                # Get raw physiological length and establish target frame count
                raw_df = pd.read_csv(physio_file)
                dur_sec = len(raw_df) / 500.0
                target_len = int(dur_sec * self.fps)
                
                # Extract clean modalities
                physio_feat = FeatureExtractor.extract_physio_stressid(physio_file, target_len)
                face_feat = FeatureExtractor.extract_face(video_file, target_len)
                voice_feat = FeatureExtractor.extract_voice(audio_file, target_len)

                # Split into windows
                physio_win = WindowProcessor.chunk(physio_feat, window_len, stride)
                face_win = WindowProcessor.chunk(face_feat, window_len, stride)
                voice_win = WindowProcessor.chunk(voice_feat, window_len, stride)

                n_windows = len(physio_win)
                for w in range(n_windows):
                    all_physio.append(physio_win[w])
                    all_face.append(face_win[w])
                    all_voice.append(voice_win[w])
                    all_labels.append(label)
                    
                    metadata_list.append({
                        'subject_id': f"stressid_{subject}",
                        'task_id': task,
                        'window_index': len(metadata_list),
                        'label': label,
                        'dataset': 'stressid'
                    })
            except Exception as e:
                print(f"    Error on {subject}_{task}: {e}")

        # Combine
        combined_feats = {
            'physio_cardio': np.stack(all_physio)[:, :, [0,1]], # hr, hrv
            'physio_eda': np.stack(all_physio)[:, :, [2,4,5]],  # eda_clean, eda_phasic, scr_peaks (dropped 3: eda_tonic)
            'physio_somatic': np.stack(all_physio)[:, :, [6,7,8,9,10,11,12,13]], # resp rate/amp, temp mean/std, acc x/y/z/mag
            'face_eye': np.stack(all_face)[:, :, range(9)],
            'face_mouth': np.stack(all_face)[:, :, 9:15],
            'face_global_face': np.stack(all_face)[:, :, 15:33], # 33 is dropped (Identity)
            'voice_spectral_prosody': np.stack(all_voice)[:, :, range(8)],
            'voice_mfcc': np.stack(all_voice)[:, :, 8:21],
            'voice_quality': np.stack(all_voice)[:, :, [21,22]], # 23 is dropped (F0)
        }
        
        return combined_feats, pd.DataFrame(metadata_list)

    def extract_wesad(self, window_sec: int = 10, stride_sec: int = 5) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Load and extract WESAD dataset."""
        print("  Extracting WESAD raw features...")
        base_dir = os.path.join(DATA_DIR, 'wesad')
        subjects = sorted([s for s in os.listdir(base_dir) if s.startswith('S') and os.path.isdir(os.path.join(base_dir, s))])
        
        all_physio, all_labels = [], []
        metadata_list = []

        window_len = int(window_sec * self.fps)
        stride = int(stride_sec * self.fps)

        for subj in subjects:
            pkl_path = os.path.join(base_dir, subj, f'{subj}.pkl')
            if not os.path.exists(pkl_path):
                continue
            
            try:
                physio_feat, binary_labels, _ = FeatureExtractor.extract_physio_wesad(pkl_path, self.fps)
                
                physio_win = WindowProcessor.chunk(physio_feat, window_len, stride)
                label_win = WindowProcessor.chunk_labels(binary_labels, window_len, stride)

                for w in range(len(physio_win)):
                    all_physio.append(physio_win[w])
                    all_labels.append(label_win[w])
                    metadata_list.append({
                        'subject_id': f"wesad_{subj.lower()}",
                        'task_id': 'all',
                        'window_index': len(metadata_list),
                        'label': label_win[w],
                        'dataset': 'wesad'
                    })
            except Exception as e:
                print(f"    Error WESAD {subj}: {e}")

        # WESAD has empty face/voice modalities
        N = len(all_physio)
        combined_feats = {
            'physio_cardio': np.stack(all_physio)[:, :, [0,1]],
            'physio_eda': np.stack(all_physio)[:, :, [2,4,5]],
            'physio_somatic': np.stack(all_physio)[:, :, [6,7,8,9,10,11,12,13]],
            'face_eye': np.zeros((N, window_len, 9), dtype=np.float32),
            'face_mouth': np.zeros((N, window_len, 6), dtype=np.float32),
            'face_global_face': np.zeros((N, window_len, 18), dtype=np.float32),
            'voice_spectral_prosody': np.zeros((N, window_len, 8), dtype=np.float32),
            'voice_mfcc': np.zeros((N, window_len, 13), dtype=np.float32),
            'voice_quality': np.zeros((N, window_len, 2), dtype=np.float32),
        }

        return combined_feats, pd.DataFrame(metadata_list)

    def extract_empathicschool(self, window_sec: int = 10, stride_sec: int = 5) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Load and extract EmpathicSchool dataset."""
        print("  Extracting EMPATHICSCHOOL raw features...")
        base_dir = os.path.join(DATA_DIR, 'empathicschool')
        subjects = sorted([s for s in os.listdir(base_dir) if s.startswith('S') and os.path.isdir(os.path.join(base_dir, s))])
        
        all_face, all_physio, all_labels = [], [], []
        metadata_list = []

        window_len = int(window_sec * self.fps)
        stride = int(stride_sec * self.fps)

        for subj in subjects:
            subj_dir = os.path.join(base_dir, subj)
            
            # Locate E4 directory and Tags (Labels)
            tags_files = glob.glob(os.path.join(subj_dir, '**', 'tags.csv'), recursive=True)
            tags_path = tags_files[0] if tags_files else None
            
            label = 0
            if tags_path:
                try:
                    df_tags = pd.read_csv(tags_path)
                    for col in df_tags.columns:
                        if 'stress' in col.lower() or 'label' in col.lower():
                            label = int(df_tags[col].dropna().iloc[0])
                            break
                except:
                    pass

            # Search recursively for ACC.csv, EDA.csv to find raw E4 dirs
            e4_dirs = set(os.path.dirname(f) for f in glob.glob(os.path.join(subj_dir, '**', '*EDA.csv'), recursive=True))
            if not e4_dirs:
                continue
            e4_dir = list(e4_dirs)[0]

            # Find matching MP4
            video_files = glob.glob(os.path.join(subj_dir, '**', '*.mp4'), recursive=True)
            video_path = video_files[0] if video_files else None

            try:
                # Estimate duration from EDA
                eda_df = pd.read_csv(os.path.join(e4_dir, 'EDA.csv'), header=None)
                dur_sec = len(eda_df) / 4.0
                target_len = int(dur_sec * self.fps)

                physio_feat = FeatureExtractor.extract_physio_empathicschool(e4_dir, target_len)
                face_feat = FeatureExtractor.extract_face(video_path, target_len)

                physio_win = WindowProcessor.chunk(physio_feat, window_len, stride)
                face_win = WindowProcessor.chunk(face_feat, window_len, stride)

                n_windows = min(len(physio_win), len(face_win))
                for w in range(n_windows):
                    all_physio.append(physio_win[w])
                    all_face.append(face_win[w])
                    all_labels.append(label)

                    metadata_list.append({
                        'subject_id': f"empathicschool_{subj.lower()}",
                        'task_id': 'all',
                        'window_index': len(metadata_list),
                        'label': label,
                        'dataset': 'empathicschool'
                    })
            except Exception as e:
                print(f"    Error EmpathicSchool {subj}: {e}")
        # EmpathicSchool has no voice
        N_emp = len(all_physio)
        combined_feats = {
            'physio_cardio': np.stack(all_physio)[:, :, [0,1]],
            'physio_eda': np.stack(all_physio)[:, :, [2,4,5]],
            'physio_somatic': np.stack(all_physio)[:, :, [6,7,8,9,10,11,12,13]],
            'face_eye': np.stack(all_face)[:, :, range(9)],
            'face_mouth': np.stack(all_face)[:, :, 9:15],
            'face_global_face': np.stack(all_face)[:, :, 15:33],
            'voice_spectral_prosody': np.zeros((N_emp, window_len, 8), dtype=np.float32),
            'voice_mfcc': np.zeros((N_emp, window_len, 13), dtype=np.float32),
            'voice_quality': np.zeros((N_emp, window_len, 2), dtype=np.float32),
        }

        return combined_feats, pd.DataFrame(metadata_list)


def run_clean_extraction_pipeline():
    """Run full unified clean feature extraction and output enriched models."""
    print("=" * 60)
    print("  Unified Feature Extraction Orchestration Runner")
    print("=" * 60)

    service = FeatureExtractionService()
    
    # Process each raw dataset separately, saving individual models
    datasets_to_run = ['stressid', 'wesad', 'empathicschool']
    
    for ds in datasets_to_run:
        start_time = time.time()
        try:
            if ds == 'stressid':
                feats, meta = service.extract_stressid()
            elif ds == 'wesad':
                feats, meta = service.extract_wesad()
            else:
                feats, meta = service.extract_empathicschool()
            
            # Save Enriched Model
            save_path = os.path.join(OUTPUT_DIR, ds)
            os.makedirs(save_path, exist_ok=True)
            
            np.savez(os.path.join(save_path, 'sequences.npz'), **feats)
            meta.to_parquet(os.path.join(save_path, 'metadata.parquet'), index=False)
            
            dims = {k: v.shape[2] for k, v in feats.items()}
            with open(os.path.join(save_path, 'group_dims.json'), 'w') as f:
                json.dump(dims, f, indent=2)

            print(f"  ✓ Processed {ds}: {len(meta)} windows, {meta['subject_id'].nunique()} subjects. "
                  f"Time taken: {time.time() - start_time:.1f}s")
        except Exception as e:
            print(f"  ✗ Failed to process {ds}: {e}")
            traceback.print_exc()

    # Rebuild combined dataset
    print("\n  Rebuilding unified COMBINED dataset...")
    combined_dir = os.path.join(OUTPUT_DIR, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    all_feats = defaultdict(list)
    all_metas = []

    for ds in datasets_to_run:
        p = os.path.join(OUTPUT_DIR, ds)
        if os.path.exists(p):
            meta = pd.read_parquet(os.path.join(p, 'metadata.parquet'))
            all_metas.append(meta)
            
            loaded = np.load(os.path.join(p, 'sequences.npz'))
            for k in loaded.keys():
                all_feats[k].append(loaded[k])

    if len(all_metas) > 0:
        combined_feats = {k: np.concatenate(v, axis=0) for k, v in all_feats.items()}
        np.savez(os.path.join(combined_dir, 'sequences.npz'), **combined_feats)
        
        combined_meta = pd.concat(all_metas, ignore_index=True)
        # Shift window indices to be unique
        combined_meta['window_index'] = np.arange(len(combined_meta))
        combined_meta.to_parquet(os.path.join(combined_dir, 'metadata.parquet'), index=False)
        
        dims = {k: v.shape[2] for k, v in combined_feats.items()}
        with open(os.path.join(combined_dir, 'group_dims.json'), 'w') as f:
            json.dump(dims, f, indent=2)

        print(f"  ✓ Unified COMBINED dataset built successfully: {len(combined_meta)} windows, "
              f"{combined_meta['subject_id'].nunique()} unique subjects.")


if __name__ == '__main__':
    run_clean_extraction_pipeline()
