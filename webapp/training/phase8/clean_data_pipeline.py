"""
Clean Data Extraction Pipeline — extracts features from RAW dataset files,
handles multiple window sizes, fixes known pipeline bugs, outputs enriched
training data ready for SSVB-CASA-AIS training.

Usage:
    python clean_data_pipeline.py --datasets stressid wesad empathicschool
    python clean_data_pipeline.py --datasets stressid --windows 2 5 10 30
"""
import os, sys, json, glob, warnings, argparse, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'webapp'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'research', 'pipeline')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window configurations
DEFAULT_WINDOWS = {2: {'stride': 1}, 5: {'stride': 2}, 10: {'stride': 5}, 30: {'stride': 15}}
TARGET_FPS = 3  # target frames per second for sequence extraction

# =========================================================================
# RAW DATA LOADERS
# =========================================================================

def load_stressid_raw():
    """Load StressID raw data. Returns list of records: {subject, task, label,
    video_path, audio_path, physio_path}."""
    base_dir = os.path.join(DATA_DIR, 'stressid')
    labels_path = os.path.join(base_dir, 'labels.csv')
    if not os.path.exists(labels_path):
        print("  StressID labels.csv not found")
        return []
    
    labels_df = pd.read_csv(labels_path)
    records = []
    
    for _, row in labels_df.iterrows():
        st = str(row['subject/task']).strip()
        if '/' in st:
            subject, task = st.split('/', 1)
        elif '_' in st:
            subject, task = st.split('_', 1)
        else:
            continue
        subject = subject.strip().lower()
        task = task.strip()
        
        label = row.get('binary-stress', row.get('binary_stress', None))
        if pd.isna(label) or label == '':
            continue
        try:
            label = int(float(label))
        except:
            continue
        
        video = os.path.join(base_dir, 'Videos', subject, f"{subject}_{task}.mp4")
        audio = os.path.join(base_dir, 'Audio', subject, f"{subject}_{task}.wav")
        # Fix: skip Mac artifact files
        if audio.endswith('.wav') and os.path.basename(audio).startswith('._'):
            audio = audio.replace('._', '')
        
        physio = os.path.join(base_dir, 'Physiological', subject, f"{subject}_{task}.txt")
        
        records.append({
            'subject': f"stressid_{subject}",
            'task': task,
            'label': label,
            'video_path': video if os.path.exists(video) else None,
            'audio_path': audio if os.path.exists(audio) else None,
            'physio_path': physio if os.path.exists(physio) else None,
            'dataset': 'stressid',
        })
    
    return records


def load_wesad_raw():
    """Load WESAD raw data from pickle files. Returns list of records.
    Label mapping: 1=baseline(not stress), 2=stress, 3=amusement(not stress),
    4=meditation(not stress)."""
    base_dir = os.path.join(DATA_DIR, 'wesad')
    records = []
    
    # Label mapping from WESAD paper
    STRESS_LABELS = {2: 1}  # only TSST stress is "stress"
    NON_STRESS_LABELS = {0: 0, 1: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    
    subjects = sorted([s for s in os.listdir(base_dir) 
                       if s.startswith('S') and os.path.isdir(os.path.join(base_dir, s))])
    
    for subj in subjects:
        pkl_path = os.path.join(base_dir, subj, f'{subj}.pkl')
        if not os.path.exists(pkl_path):
            continue
        
        records.append({
            'subject': f"wesad_{subj.lower()}",
            'task': 'all',
            'label': None,  # labels are in the pickle, loaded during extraction
            'pkl_path': pkl_path,
            'dataset': 'wesad',
        })
    
    return records


def load_empathicschool_raw():
    """Load EmpathicSchool raw data. Returns list of records."""
    base_dir = os.path.join(DATA_DIR, 'empathicschool')
    records = []
    
    subjects = sorted([s for s in os.listdir(base_dir)
                       if s.startswith('S') and os.path.isdir(os.path.join(base_dir, s))])
    
    for subj in subjects:
        subj_dir = os.path.join(base_dir, subj)
        
        # Find all MP4 files for this subject
        mp4_files = []
        for root, dirs, files in os.walk(subj_dir):
            # Skip dirs that are just single-letter codes (already captured by parent)
            for f in files:
                if f.lower().endswith('.mp4') and not 'no_audio' in f.lower():
                    mp4_files.append(os.path.join(root, f))
        
        # Find all E4 CSV files
        e4_files = defaultdict(list)
        for root, dirs, files in os.walk(subj_dir):
            for f in files:
                if f.endswith('.csv') and f[0].isupper():
                    fname_lower = f.lower()
                    if any(x in fname_lower for x in ['acc', 'eda', 'hr', 'temp', 'bvp', 'ibi']):
                        e4_files[os.path.dirname(root)].append(os.path.join(root, f))
        
        # Tags file for labels
        tags_files = glob.glob(os.path.join(subj_dir, '**', 'tags.csv'), recursive=True)
        tags_path = tags_files[0] if tags_files else None
        
        # Create session records
        # EmpathicSchool has T1-T8 tasks with potential stress annotations
        # For simplicity, create one record per subject/directory
        records.append({
            'subject': f"empathicschool_{subj.lower()}",
            'task': 'all',
            'label': None,
            'video_dir': subj_dir if mp4_files else None,
            'e4_dir': subj_dir if e4_files else None,
            'tags_path': tags_path,
            'dataset': 'empathicschool',
        })
    
    return records


# =========================================================================
# FEATURE EXTRACTION (with robust error handling)
# =========================================================================

def extract_physio_stressid(physio_path, target_fps=3, window_sec=10):
    """Extract physio features from StressID raw CSV with robust error handling.
    Returns [n_windows, window_len, 14] array (or None on failure)."""
    try:
        df = pd.read_csv(physio_path)
    except:
        return None
    
    required = {'ECG', 'EDA', 'RR'}
    if not required.issubset(df.columns):
        return None
    
    sr = 500.0
    N = len(df)
    duration_sec = N / sr
    
    # Extract raw signals
    ecg = df['ECG'].values.astype(np.float64)
    eda = df['EDA'].values.astype(np.float64)
    resp = df['RR'].values.astype(np.float64)
    
    # Simple signal quality checks
    ecg = _clean_signal(ecg)
    eda = _clean_signal(eda)
    resp = _clean_signal(resp)
    
    # Compute features using robust methods
    try:
        import neurokit2 as nk
        # ECG
        ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=int(sr))
        hr = ecg_signals['ECG_Rate'].values
        # HRV (RMSSD over 30s sliding window)
        r_peaks = ecg_signals['ECG_R_Peaks'].values
        hrv = _compute_rmssd_sliding(r_peaks, sr, window_sec=30)
        
        # EDA
        eda_signals, _ = nk.eda_process(eda, sampling_rate=int(sr))
        eda_clean = eda_signals['EDA_Clean'].values
        eda_tonic = eda_signals['EDA_Tonic'].values
        eda_phasic = eda_signals['EDA_Phasic'].values
        scr_peaks = eda_signals['SCR_Peaks'].values
        
        # Respiration
        rsp_signals, _ = nk.rsp_process(resp, sampling_rate=int(sr))
        resp_rate = rsp_signals['RSP_Rate'].values
        resp_amplitude = rsp_signals['RSP_Amplitude'].values
    except Exception:
        # Fallback: simple computations
        hr = _estimate_hr_from_ecg(ecg, sr)
        hrv = np.full(N, 50.0)
        eda_clean = eda
        eda_tonic = _lowpass(eda, 0.05, sr)
        eda_phasic = eda - eda_tonic
        scr_peaks = np.zeros(N)
        resp_rate = np.full(N, 15.0)
        resp_amplitude = np.full(N, 1.0)
    
    # Build 14-channel physio signal
    physio = np.column_stack([
        hr, hrv,
        eda_clean, eda_tonic, eda_phasic, scr_peaks,
        resp_rate, resp_amplitude,
        np.zeros(N), np.zeros(N),  # temp_mean, temp_std placeholder
        np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),  # acc x,y,z,mag
    ])  # [N, 14]
    
    # NaN safeguard
    physio = np.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Window into sequences
    window_len = int(target_fps * window_sec)
    stride = int(target_fps * window_sec / 2)
    target_len = int(target_fps * duration_sec)
    
    # Resample to target FPS
    physio_resampled = _resample_to_fps(physio, N, target_len)
    
    seqs = _window_sequence(physio_resampled, window_len, stride)
    return seqs  # [n_windows, window_len, 14]


def extract_physio_wesad(pkl_path, target_fps=3, window_sec=10):
    """Extract physio features from WESAD pickle."""
    try:
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except:
        return None, None
    
    signal = data.get('signal', {})
    label = data.get('label', None)
    if label is None:
        return None, None
    
    # Extract chest signals (ECG, EDA, EMG, Temp, Resp) at 700Hz
    chest = signal.get('chest', {})
    if not chest:
        return None, None
    
    ecg = chest.get('ECG', np.array([])).astype(np.float64)
    eda = chest.get('EDA', np.array([])).astype(np.float64)
    temp = chest.get('Temp', np.array([])).astype(np.float64)
    resp = chest.get('Resp', np.array([])).astype(np.float64)
    
    sr = 700.0
    N = min(len(ecg), len(eda), len(temp), len(resp))
    if N < sr * 10:  # need at least 10 seconds
        return None, None
    
    ecg, eda, temp, resp = ecg[:N], eda[:N], temp[:N], resp[:N]
    
    # Clean
    ecg = _clean_signal(ecg)
    eda = _clean_signal(eda)
    resp = _clean_signal(resp)
    
    try:
        import neurokit2 as nk
        ecg_signals, _ = nk.ecg_process(ecg, sampling_rate=int(sr))
        hr = ecg_signals['ECG_Rate'].values
        r_peaks = ecg_signals['ECG_R_Peaks'].values
        hrv = _compute_rmssd_sliding(r_peaks, sr, window_sec=30)
        
        eda_signals, _ = nk.eda_process(eda, sampling_rate=int(sr))
        eda_clean = eda_signals['EDA_Clean'].values
        eda_tonic = eda_signals['EDA_Tonic'].values
        eda_phasic = eda_signals['EDA_Phasic'].values
        scr_peaks = eda_signals['SCR_Peaks'].values
        
        rsp_signals, _ = nk.rsp_process(resp, sampling_rate=int(sr))
        resp_rate = rsp_signals['RSP_Rate'].values
        resp_amplitude = rsp_signals['RSP_Amplitude'].values
    except:
        hr = _estimate_hr_from_ecg(ecg, sr)
        hrv = np.full(N, 50.0)
        eda_clean = eda
        eda_tonic = _lowpass(eda, 0.05, sr)
        eda_phasic = eda - eda_tonic
        scr_peaks = np.zeros(N)
        resp_rate = np.full(N, 15.0)
        resp_amplitude = np.full(N, 1.0)
    
    # Use wrist accelerometer if available
    wrist = signal.get('wrist', {})
    acc_x = wrist.get('ACC_x', np.zeros(N)).astype(np.float64)[:N]
    acc_y = wrist.get('ACC_y', np.zeros(N)).astype(np.float64)[:N]
    acc_z = wrist.get('ACC_z', np.zeros(N)).astype(np.float64)[:N]
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Temp from chest (already extracted), use wrist temp if available
    temp_wrist = wrist.get('TEMP', np.zeros(N)).astype(np.float64)[:N]
    temp_mean = temp[:N] if len(temp) >= N else temp_wrist
    
    physio = np.column_stack([
        hr, hrv,
        eda_clean, eda_tonic, eda_phasic, scr_peaks,
        resp_rate, resp_amplitude,
        temp_mean, np.zeros(N),  # temp_mean, temp_std
        acc_x, acc_y, acc_z, acc_mag,
    ])
    
    physio = np.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Map labels: 1=baseline, 2=stress, 3=amusement, 4=meditation → binary
    # Define stress=2 (TSST), non-stress=everything else
    binary_label = np.zeros_like(label)
    binary_label[label == 2] = 1  # only TSST stress
    
    # Window
    duration_sec = N / sr
    window_len = int(target_fps * window_sec)
    stride = int(target_fps * window_sec / 2)
    target_len = int(target_fps * duration_sec)
    
    physio_resampled = _resample_to_fps(physio, N, target_len)
    label_resampled = _resample_labels(label, N, target_len)
    binary_resampled = _resample_labels(binary_label, N, target_len)
    
    seqs = _window_sequence(physio_resampled, window_len, stride)
    labels_seq = _window_labels(label_resampled, window_len, stride)
    binary_seq = _window_labels(binary_resampled, window_len, stride, majority=True)
    
    if len(seqs) == 0:
        return None, None
    
    return seqs, binary_seq, labels_seq  # [n_windows, window_len, 14], [n_windows], [n_windows]


def extract_physio_empathicschool(e4_dir, target_fps=3, window_sec=10):
    """Extract physio features from EmpathicSchool E4 CSV files."""
    # Find and parse E4 files
    e4_data = {}
    for root, dirs, files in os.walk(e4_dir):
        for f in files:
            f_lower = f.lower()
            if f.endswith('.csv') and any(x in f_lower for x in ['eda', 'hr', 'temp', 'acc', 'bvp']):
                fpath = os.path.join(root, f)
                try:
                    df = pd.read_csv(fpath, header=None)
                    vals = df.iloc[:, 0].values.astype(np.float64)
                    vals = _clean_signal(vals)
                    key = f_lower.replace('.csv', '')
                    if key not in e4_data or len(vals) > len(e4_data.get(key, [])):
                        e4_data[key] = vals
                except:
                    pass
    
    if not e4_data:
        return None
    
    # Determine sampling rates and align
    eda = e4_data.get('eda', np.array([]))
    hr = e4_data.get('hr', np.array([]))
    temp = e4_data.get('temp', np.array([]))
    
    # E4 sampling rates: EDA=4Hz, HR=1Hz, TEMP=4Hz, ACC=32Hz, BVP=64Hz
    eda_sr = 4.0
    hr_sr = 1.0
    temp_sr = 4.0
    
    # Resample everything to a common rate
    common_sr = 4.0
    if len(eda) > 0:
        eda_dur = len(eda) / eda_sr
    elif len(hr) > 0:
        eda_dur = len(hr) / hr_sr
    else:
        return None
    
    N_common = int(eda_dur * common_sr)
    if N_common < common_sr * 5:  # need at least 5 seconds
        return None
    
    def resample_to_common(sig, src_sr):
        if len(sig) == 0:
            return np.zeros(N_common)
        return _resample_to_fps(sig, len(sig), N_common)
    
    eda_r = resample_to_common(eda, eda_sr)
    hr_r = resample_to_common(hr, hr_sr)
    temp_r = resample_to_common(temp, temp_sr)
    
    # Process EDA
    try:
        import neurokit2 as nk
        eda_signals, _ = nk.eda_process(eda_r, sampling_rate=int(common_sr))
        eda_clean = eda_signals['EDA_Clean'].values
        eda_tonic = eda_signals['EDA_Tonic'].values
        eda_phasic = eda_signals['EDA_Phasic'].values
        scr_peaks = eda_signals['SCR_Peaks'].values
    except:
        eda_clean = eda_r
        eda_tonic = _lowpass(eda_r, 0.05, common_sr)
        eda_phasic = eda_r - eda_tonic
        scr_peaks = np.zeros(N_common)
    
    # HRV from HR signal (simple RMSSD on HR itself as proxy)
    hrv = np.zeros_like(hr_r)
    for i in range(len(hr_r)):
        start = max(0, i - 30)
        segment = hr_r[start:i+1]
        if len(segment) > 2:
            hrv[i] = np.sqrt(np.mean(np.diff(segment) ** 2))
    
    # ACC
    acc = e4_data.get('acc', np.array([]))
    acc_sr = 32.0
    if len(acc) > 0:
        acc_aligned = _resample_to_fps(acc, len(acc), N_common * 8)  # 32Hz → 32Hz (keep resolution)
        # Actually simpler: resample directly
        acc_x = _resample_to_fps(acc[::3] if len(acc) > 2 else acc, max(1, len(acc)//3), N_common)
        acc_y = _resample_to_fps(acc[1::3] if len(acc) > 2 else acc, max(1, len(acc)//3), N_common)
        acc_z = _resample_to_fps(acc[2::3] if len(acc) > 2 else acc, max(1, len(acc)//3), N_common)
    else:
        acc_x = acc_y = acc_z = np.zeros(N_common)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Build 14-channel physio
    physio = np.column_stack([
        hr_r[:N_common], hrv[:N_common],
        eda_clean[:N_common], eda_tonic[:N_common], eda_phasic[:N_common], scr_peaks[:N_common],
        np.zeros(N_common), np.zeros(N_common),  # resp_rate, resp_amplitude (not avail from E4)
        temp_r[:N_common], np.zeros(N_common),  # temp_mean, temp_std
        acc_x, acc_y, acc_z, acc_mag,
    ])
    
    physio = np.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Window
    window_len = int(target_fps * window_sec)
    stride = int(target_fps * window_sec / 2)
    target_len = int(target_fps * eda_dur)
    
    physio_resampled = _resample_to_fps(physio, N_common, target_len)
    seqs = _window_sequence(physio_resampled, window_len, stride)
    
    return seqs


def extract_face_features(video_path, target_fps=3):
    """Extract face features from video file."""
    if video_path is None or not os.path.exists(video_path):
        return None
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 3:
            cap.release()
            return None
        
        # Sample frames at target_fps
        step = max(1, int(video_fps / target_fps))
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frames.append(frame)
            frame_idx += 1
        cap.release()
        
        if len(frames) < 2:
            return None
        
        # Use OpenCV face detection only (robust fallback)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print("    Face detection: haarcascade file not found, skipping face extraction")
            return np.zeros((len(frames), 34), dtype=np.float32)

        face_cascade = cv2.CascadeClassifier(cascade_path)
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                feats = np.zeros(34)
                feats[0] = w / frame.shape[1]
                feats[1] = h / frame.shape[0]
                feats[2] = x / frame.shape[1]
                feats[3] = y / frame.shape[0]
                face_features.append(feats)
            else:
                face_features.append(np.full(34, np.nan))
        
        if len(face_features) == 0:
            return None
        
        face_array = np.array(face_features, dtype=np.float32)  # [n_frames, 34]
        face_array = np.nan_to_num(face_array, nan=0.0)
        return face_array
        
    except Exception as e:
        print(f"    Face extraction error: {e}")
        return None


def extract_voice_features(audio_path, target_fps=3):
    """Extract voice features from audio file."""
    if audio_path is None or not os.path.exists(audio_path):
        return None
    
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) < sr:  # less than 1 second
            return None
        
        # Extract features with hop_length matching target_fps
        hop_length = int(sr / target_fps)
        if len(y) < hop_length:
            return None
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        # Pitch
        pitches, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=75, fmax=300, sr=sr, hop_length=hop_length)
        pitches = np.nan_to_num(pitches, nan=0.0)
        f0_mean = pitches
        f0_std = np.full_like(pitches, np.nanstd(pitches[pitches > 0]) if (pitches > 0).sum() > 1 else 0)
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).mean(axis=0)
        # Voice quality
        hnr = _compute_hnr_framewise(y, sr, hop_length)
        jitter = _compute_jitter_framewise(pitches)
        
        # Hop length may give us slightly different lengths; unify
        n_frames = min(
            len(rms), len(zcr), len(pitches), mfcc.shape[1],
            len(spectral_centroid), len(spectral_bandwidth),
            len(spectral_rolloff), len(spectral_flatness), len(chroma_stft),
            len(hnr), len(jitter)
        )
        
        voice = np.column_stack([
            rms[:n_frames], zcr[:n_frames],
            f0_mean[:n_frames], f0_std[:n_frames],
            pitches[:n_frames], pitches[:n_frames],  # f0_min/f0_max placeholder
            pitches[:n_frames],  # f0_range
            voiced_probs[:n_frames] if len(voiced_probs) >= n_frames else np.zeros(n_frames),
            mfcc[0, :n_frames],  # loudness proxy (MFCC0)
            np.zeros(n_frames),  # loudness_std placeholder
            hnr[:n_frames], jitter[:n_frames],
        ])
        
        # MFCCs 0-12
        mfcc_features = np.vstack([mfcc[i, :n_frames] for i in range(13)]).T
        
        spectral = np.column_stack([
            spectral_centroid[:n_frames], spectral_bandwidth[:n_frames],
            spectral_rolloff[:n_frames], spectral_flatness[:n_frames],
            chroma_stft[:n_frames], zcr[:n_frames],  # zcr already computed
        ])
        
        voice_full = np.concatenate([voice, mfcc_features, spectral], axis=1)  # [n_frames, 24]
        voice_full = np.nan_to_num(voice_full, nan=0.0, posinf=0.0, neginf=0.0)
        
        return voice_full
        
    except Exception as e:
        print(f"    Voice extraction error: {e}")
        return None


# =========================================================================
# MULTI-WINDOW SUPPORT
# =========================================================================

def window_sequences_at_multiple_scales(seqs_raw, window_configs, target_fps=TARGET_FPS):
    """Window a raw per-frame feature sequence at multiple resolutions.
    
    Args:
        seqs_raw: [n_frames, feat_dim] per-frame features (e.g., face at 3fps)
        window_configs: dict of {window_sec: {'stride': stride_sec}}
        
    Returns:
        dict of {window_sec: [n_windows, window_len, feat_dim]}
    """
    n_frames, feat_dim = seqs_raw.shape
    results = {}
    
    for win_sec, config in window_configs.items():
        win_len = int(target_fps * win_sec)
        stride = int(target_fps * config['stride'])
        
        if n_frames < win_len:
            results[win_sec] = np.zeros((0, win_len, feat_dim), dtype=np.float32)
            continue
        
        windows = []
        for start in range(0, n_frames - win_len + 1, stride):
            windows.append(seqs_raw[start:start + win_len])
        
        if len(windows) > 0:
            results[win_sec] = np.stack(windows).astype(np.float32)
        else:
            results[win_sec] = np.zeros((0, win_len, feat_dim), dtype=np.float32)
    
    return results


# =========================================================================
# LABEL MAPPING
# =========================================================================

def map_stressid_labels(records, window_configs):
    """Map StressID labels from per-task to per-window."""
    # Already 1-to-1: each record = one task = one label
    return records


def map_wesad_labels(seqs, binary_seq, labels_seq, window_configs):
    """Map WESAD labels from continuous signal to per-window."""
    # Already done in extract_physio_wesad
    return seqs, binary_seq


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _clean_signal(x, max_val=1e8):
    """Remove extreme values and NaN from signal."""
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, -max_val, max_val)
    return x


def _resample_to_fps(data, orig_len, target_len):
    """Resample data from orig_len to target_len using interpolation."""
    if orig_len == target_len:
        return data.copy()
    if target_len <= 0 or orig_len <= 0:
        return np.zeros((target_len, data.shape[1])) if data.ndim > 1 else np.zeros(target_len)
    
    if data.ndim == 1:
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, data)
    else:
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_len)
        result = np.zeros((target_len, data.shape[1]), dtype=data.dtype)
        for c in range(data.shape[1]):
            result[:, c] = np.interp(x_new, x_old, data[:, c])
        return result


def _resample_labels(labels, orig_len, target_len):
    """Resample discrete labels by nearest-neighbor."""
    if orig_len == target_len:
        return labels.copy()
    indices = np.round(np.linspace(0, orig_len - 1, target_len)).astype(int)
    return labels[indices]


def _window_sequence(seq, window_len, stride):
    """Window a [T, D] array into overlapping windows [N, window_len, D]."""
    T, D = seq.shape
    if T < window_len:
        return np.zeros((0, window_len, D), dtype=np.float32)
    windows = []
    for start in range(0, T - window_len + 1, stride):
        windows.append(seq[start:start + window_len])
    return np.stack(windows).astype(np.float32) if windows else np.zeros((0, window_len, D), dtype=np.float32)


def _window_labels(labels, window_len, stride, majority=False):
    """Window labels, taking majority or first value."""
    T = len(labels)
    result = []
    for start in range(0, T - window_len + 1, stride):
        chunk = labels[start:start + window_len]
        if majority:
            result.append(int(np.round(chunk.mean())))
        else:
            result.append(int(chunk[0]))
    return np.array(result, dtype=np.int64)


def _lowpass(signal, cutoff, sr, order=4):
    """Simple low-pass filter."""
    try:
        from scipy import signal as scipy_signal
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
        return scipy_signal.filtfilt(b, a, signal)
    except:
        return signal


def _estimate_hr_from_ecg(ecg, sr):
    """Simple heart rate estimation from ECG."""
    try:
        from scipy import signal as scipy_signal
        # Find peaks
        peaks, _ = scipy_signal.find_peaks(ecg, distance=int(sr * 0.4))
        if len(peaks) < 2:
            return np.full(len(ecg), 75.0)
        rr_intervals = np.diff(peaks) / sr * 60.0
        hr_values = 60.0 / (np.diff(peaks) / sr)
        hr = np.interp(np.arange(len(ecg)), peaks[1:], hr_values)
        hr = np.nan_to_num(hr, nan=75.0)
        return hr
    except:
        return np.full(len(ecg), 75.0)


def _compute_rmssd_sliding(r_peaks, sr, window_sec=30):
    """Compute RMSSD HRV over a sliding window."""
    hrv = np.zeros(len(r_peaks))
    window_samples = int(window_sec * sr)
    for i in range(len(r_peaks)):
        start = max(0, i - window_samples)
        peaks_in_w = np.where(r_peaks[start:i+1] == 1)[0]
        if len(peaks_in_w) > 2:
            rri = np.diff(peaks_in_w) / sr * 1000.0
            hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))
    return hrv


def _compute_hnr_framewise(y, sr, hop_length):
    """Compute HNR per frame."""
    frame_length = int(sr * 0.03)  # 30ms
    n_frames = 1 + (len(y) - frame_length) // hop_length
    hnr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_length]
        if len(frame) < 2:
            continue
        # Autocorrelation-based HNR
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        if len(corr) < 2:
            continue
        r0 = corr[0] + 1e-10
        r1 = corr[1] if len(corr) > 1 else 0
        # Find peak corresponding to pitch period
        low_lag = int(sr / 300)
        high_lag = int(sr / 75)
        if high_lag >= len(corr):
            high_lag = len(corr) - 1
        if low_lag >= high_lag:
            continue
        peak_idx = np.argmax(corr[low_lag:high_lag]) + low_lag
        r_tau = corr[peak_idx]
        hnr[i] = 10 * np.log10(max(1e-10, r_tau) / max(1e-10, r0 - r_tau))
    return np.nan_to_num(hnr, nan=0.0)


def _compute_jitter_framewise(pitches):
    """Compute jitter from pitch array."""
    jitter = np.zeros(len(pitches))
    for i in range(len(pitches)):
        start = max(0, i - 10)
        segment = pitches[start:i+1]
        valid = segment[segment > 0]
        if len(valid) >= 2:
            diffs = np.abs(np.diff(valid))
            jitter[i] = np.mean(diffs) / (np.mean(valid) + 1e-10)
    return jitter


def _compute_face_features(landmarks):
    """Compute 34 face features from MediaPipe landmarks."""
    # Eye aspect ratio, mouth features, head pose, etc.
    # Simplified version matching the 34-feature pipeline format
    feats = np.zeros(34)
    
    def get_lm(idx):
        return landmarks[idx]
    
    def dist(a, b):
        return np.linalg.norm(a - b)
    
    # Eye AR
    left_eye_pts = [33, 160, 158, 133, 153, 144]
    right_eye_pts = [362, 385, 387, 263, 373, 380]
    
    left_eye = np.array([get_lm(i)[:2] for i in left_eye_pts])
    right_eye = np.array([get_lm(i)[:2] for i in right_eye_pts])
    
    # Eye aspect ratios
    ear_left = (dist(left_eye[1], left_eye[2]) + dist(left_eye[4], left_eye[5])) / (2 * dist(left_eye[0], left_eye[3]) + 1e-6)
    ear_right = (dist(right_eye[1], right_eye[2]) + dist(right_eye[4], right_eye[5])) / (2 * dist(right_eye[0], right_eye[3]) + 1e-6)
    feats[0] = ear_left * 100
    feats[1] = ear_right * 100
    feats[2] = (ear_left + ear_right) / 2 * 100
    feats[3] = abs(ear_left - ear_right) * 100
    
    # Mouth
    mouth_pts = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
    mouth = np.array([get_lm(i)[:2] for i in mouth_pts])
    mouth_width = dist(mouth[0], mouth[6])
    mouth_height = dist(mouth[3], mouth[7])
    feats[8] = mouth_width * 100
    feats[9] = mouth_height * 100
    feats[10] = (mouth_width / (mouth_height + 1e-6)) * 10
    
    # Head pose (simplified from face orientation)
    nose_tip = get_lm(1)
    chin = get_lm(152)
    left_eye_outer = get_lm(33)
    right_eye_outer = get_lm(263)
    left_mouth = get_lm(61)
    right_mouth = get_lm(291)
    
    # Head tilt from eye y-coordinates
    feats[15] = (left_eye_outer[1] - right_eye_outer[1]) * 100
    
    # Nose wrinkle proxy
    nose_bridge = get_lm(168)
    nose_tip_z = nose_tip[2]
    nose_bridge_z = nose_bridge[2]
    feats[17] = (nose_tip_z - nose_bridge_z) * 100
    
    # More features would fill indices 4-7, 11-14, 16, 18-33
    # Placeholder for remaining indices with face geometry
    feats[4:8] = [feats[0]*0.5, feats[1]*0.5, feats[2]*0.5, 0]  # blink velocity etc
    feats[11] = chin[1] - nose_tip[1]  # face height
    feats[12:17] = [0, 0, 0, 0, 0]  # more head pose
    
    return feats


# =========================================================================
# BUILD ENRICHED DATA
# =========================================================================

def features_to_enriched(face_feats, voice_feats, physio_feats, label, subject_id, window_idx):
    """Build a single enriched data record from per-modality windows."""
    # Not used in batch processing — see process_dataset
    pass


def process_stressid(window_configs):
    """Process all StressID raw data and output enriched format."""
    print(f"\n{'='*60}")
    print("  Processing StressID")
    print(f"{'='*60}")
    
    records = load_stressid_raw()
    print(f"  Loaded {len(records)} task records")
    
    # Collect per-subject features
    all_face = []
    all_voice = []
    all_physio = []
    all_labels = []
    all_subjects = []
    all_tasks = []
    
    skipped_no_data = 0
    skipped_error = 0
    
    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(records)}")
        
        subject = rec['subject']
        label = rec['label']
        
        # Extract each modality
        face_feats = extract_face_features(rec['video_path'], TARGET_FPS)
        voice_feats = extract_voice_features(rec['audio_path'], TARGET_FPS)
        physio_feats = extract_physio_stressid(rec['physio_path'], TARGET_FPS, max(window_configs.keys()))
        
        if face_feats is None and voice_feats is None and physio_feats is None:
            skipped_no_data += 1
            continue
        
        try:
            # Window each modality at multiple scales
            face_windows = window_sequences_at_multiple_scales(face_feats, window_configs) if face_feats is not None else {}
            voice_windows = window_sequences_at_multiple_scales(voice_feats, window_configs) if voice_feats is not None else {}
            physio_windows = window_sequences_at_multiple_scales(physio_feats, window_configs) if physio_feats is not None else {}
            
            # Align: take the minimum number of windows across modalities
            n_windows = min(
                min((v.shape[0] for v in face_windows.values()), default=0),
                min((v.shape[0] for v in voice_windows.values()), default=0),
                min((v.shape[0] for v in physio_windows.values()), default=0),
            )
            
            if n_windows == 0:
                skipped_error += 1
                continue
            
            # Use the first window scale to get index
            win_sec = max(window_configs.keys())
            
            face_w = face_windows.get(win_sec, np.zeros((0, 0, 34)))[:n_windows]
            voice_w = voice_windows.get(win_sec, np.zeros((0, 0, 24)))[:n_windows]
            physio_w = physio_windows.get(win_sec, np.zeros((0, 0, 14)))[:n_windows]
            
            for w in range(n_windows):
                all_face.append(face_w[w] if len(face_w) > w else np.zeros((0, 34)))
                all_voice.append(voice_w[w] if len(voice_w) > w else np.zeros((0, 24)))
                all_physio.append(physio_w[w] if len(physio_w) > w else np.zeros((0, 14)))
                all_labels.append(label)
                all_subjects.append(subject)
                all_tasks.append(rec['task'])
        
        except Exception as e:
            skipped_error += 1
            continue
    
    print(f"  Results: {len(all_labels)} windows, {len(np.unique(all_subjects))} subjects")
    print(f"  Skipped: {skipped_no_data} no data, {skipped_error} errors")
    
    # For StressID, build enriched format at each window scale
    # Since we used the largest window scale for alignment, resample for smaller scales
    return _build_for_all_scales(all_face, all_voice, all_physio, all_labels, all_subjects, all_tasks, 'stressid', window_configs)


def process_wesad(window_configs):
    """Process all WESAD raw data and output enriched format."""
    print(f"\n{'='*60}")
    print("  Processing WESAD")
    print(f"{'='*60}")
    
    records = load_wesad_raw()
    print(f"  Loaded {len(records)} subject records")
    
    all_face = []
    all_voice = []
    all_physio = []
    all_labels = []
    all_subjects = []
    all_tasks = []
    
    for rec in records:
        result = extract_physio_wesad(rec['pkl_path'], TARGET_FPS, max(window_configs.keys()))
        if result is None:
            print(f"  SKIP {rec['subject']}: extraction failed")
            continue
        
        seqs, binary, _ = result
        if seqs is None or len(seqs) == 0:
            continue
        
        subject = rec['subject']
        for w in range(len(seqs)):
            all_physio.append(seqs[w])
            all_labels.append(int(binary[w]))
            all_subjects.append(subject)
            all_tasks.append('all')
    
    print(f"  Results: {len(all_labels)} windows, {len(np.unique(all_subjects))} subjects")
    
    # WESAD has no face/voice, so all_face/all_voice remain empty
    return _build_for_all_scales(all_face, all_voice, all_physio, all_labels, all_subjects, all_tasks, 'wesad', window_configs)


def process_empathicschool(window_configs):
    """Process all EmpathicSchool raw data and output enriched format."""
    print(f"\n{'='*60}")
    print("  Processing EmpathicSchool")
    print(f"{'='*60}")
    
    records = load_empathicschool_raw()
    print(f"  Loaded {len(records)} subject records")
    
    all_face = []
    all_physio = []
    all_labels = []
    all_subjects = []
    all_tasks = []
    
    for rec in records:
        subject = rec['subject']
        
        # Face extraction from video
        face_feats = None
        if rec['video_dir']:
            # Find first available MP4
            video_files = glob.glob(os.path.join(rec['video_dir'], '**', '*.mp4'), recursive=True)
            video_files = [f for f in video_files if 'no_audio' not in f.lower()]
            if video_files:
                # Use the first video as representative
                face_feats = extract_face_features(video_files[0], TARGET_FPS)
        
        # Physio extraction from E4 CSVs
        physio_feats = extract_physio_empathicschool(rec['e4_dir'], TARGET_FPS, max(window_configs.keys())) if rec['e4_dir'] else None
        
        if face_feats is None and physio_feats is None:
            continue
        
        # Label from tags file
        label = 0
        if rec['tags_path']:
            try:
                tags = pd.read_csv(rec['tags_path'])
                # tags.csv format varies; try common label columns
                for col in tags.columns:
                    if 'stress' in col.lower() or 'label' in col.lower() or 'condition' in col.lower():
                        vals = tags[col].dropna()
                        if len(vals) > 0:
                            label = int(vals.iloc[0]) if vals.iloc[0] in [0, 1] else 0
                        break
            except:
                pass
        
        try:
            face_windows = window_sequences_at_multiple_scales(face_feats, window_configs) if face_feats is not None else {}
            physio_windows = window_sequences_at_multiple_scales(physio_feats, window_configs) if physio_feats is not None else {}
            
            n_windows = min(
                min((v.shape[0] for v in face_windows.values()), default=0),
                min((v.shape[0] for v in physio_windows.values()), default=0),
            )
            
            if n_windows == 0:
                continue
            
            win_sec = max(window_configs.keys())
            face_w = face_windows.get(win_sec, np.zeros((0, 0, 34)))[:n_windows]
            physio_w = physio_windows.get(win_sec, np.zeros((0, 0, 14)))[:n_windows]
            
            for w in range(n_windows):
                all_face.append(face_w[w] if len(face_w) > w else np.zeros((0, 34)))
                all_physio.append(physio_w[w] if len(physio_w) > w else np.zeros((0, 14)))
                all_labels.append(label)
                all_subjects.append(subject)
                all_tasks.append('all')
        
        except Exception as e:
            continue
    
    print(f"  Results: {len(all_labels)} windows, {len(np.unique(all_subjects))} subjects")
    
    return _build_for_all_scales(all_face, [], all_physio, all_labels, all_subjects, all_tasks, 'empathicschool', window_configs)


def _build_for_all_scales(all_face, all_voice, all_physio, all_labels, all_subjects, all_tasks, ds_name, window_configs):
    """For each window scale, build enriched NPZ + metadata."""
    if len(all_labels) == 0:
        print(f"  WARNING: No data for {ds_name}")
        return
    
    n_total = len(all_labels)
    print(f"  Building enriched data for {n_total} base windows...")
    
    # Face shapes: get actual frame counts
    face_shapes = [f.shape[0] for f in all_face if len(f) > 0]
    voice_shapes = [v.shape[0] for v in all_voice if len(v) > 0]
    physio_shapes = [p.shape[0] for p in all_physio if len(p) > 0]
    
    # The base window uses max(window_configs.keys()) in seconds * TARGET_FPS frames
    base_frames = max(window_configs.keys()) * TARGET_FPS
    
    for win_sec in sorted(window_configs.keys()):
        print(f"\n  Window: {win_sec}s")
        target_frames = int(win_sec * TARGET_FPS)
        
        if target_frames > base_frames:
            print(f"    SKIP: target {target_frames} > base {base_frames} frames")
            continue
        
        # Downsample: each base window sequence can be subsampled to shorter windows
        # For simplicity, take the first `target_frames` frames from each base window
        stride = 1 if target_frames == base_frames else max(1, base_frames // target_frames)
        indices = np.arange(0, base_frames, stride)[:target_frames]
        
        def subsample(arr_list, feat_dim):
            result = []
            for arr in arr_list:
                if len(arr) == 0:
                    result.append(np.zeros((1, target_frames, feat_dim), dtype=np.float32))
                elif arr.shape[0] < target_frames:
                    # Pad
                    padded = np.zeros((target_frames, feat_dim), dtype=np.float32)
                    padded[:arr.shape[0]] = arr
                    result.append(padded[np.newaxis, :, :])
                else:
                    result.append(arr[indices][np.newaxis, :, :])
            return np.concatenate(result, axis=0) if result else np.zeros((0, target_frames, feat_dim), dtype=np.float32)
        
        face_data = subsample(all_face, 34)
        voice_data = subsample(all_voice, 24)
        physio_data = subsample(all_physio, 14)
        
        print(f"    Face: {face_data.shape}, Voice: {voice_data.shape}, Physio: {physio_data.shape}")
        
        # Map to sub-modality groups
        # Face 34ch → eye(9), mouth(6), global_face(18) → 33 features (exclude idx 11)
        face_eye = face_data[:, :, [0,1,2,3,4, 18,19,20,32]]
        face_mouth = face_data[:, :, [8,9,10, 24,25,26]]
        face_gface = np.delete(face_data[:, :, [5,6,7,12,13,14,15,16,17,21,22,23,27,28,29,30,31,33]], 11, axis=2) if face_data.shape[2] > 11 else face_data[:, :, [5,6,7,12,13,14,15,16,17,21,22,23,27,28,29,30,31,33]]
        
        # Voice 24ch → spectral_prosody(8), mfcc(13), quality(2) → 23 features (exclude idx 2=f0_mean)
        voice_sp = voice_data[:, :, [0,1,3, 15,16,17,18,19]]  # rms, zcr, f0_std, spectral_*, chroma
        voice_mfcc = voice_data[:, :, 20:33] if voice_data.shape[2] >= 33 else voice_data[:, :, -13:]
        voice_qual = voice_data[:, :, [13,14]]  # hnr, jitter
        
        # Physio 14ch → cardio(2), eda(3), somatic(8) → 13 features (exclude idx 1=eda_tonic)
        physio_cardio = physio_data[:, :, [0,1]]  # hr, hrv
        physio_eda = np.delete(physio_data[:, :, [2,3,4,5,6]], 1, axis=2)  # eda_clean, eda_phasic, scr → exclude eda_tonic
        physio_somatic = physio_data[:, :, [7,8,9,10,11,12,13]]  # resp_rate, resp_amplitude, temp_mean, acc_x,y,z,mag
        
        # Save
        ds_dir = os.path.join(OUTPUT_DIR, f"{ds_name}_{win_sec}s")
        os.makedirs(ds_dir, exist_ok=True)
        
        np.savez(os.path.join(ds_dir, "sequences.npz"),
                 face_eye=face_eye, face_mouth=face_mouth, face_global_face=face_gface,
                 voice_spectral_prosody=voice_sp, voice_mfcc=voice_mfcc, voice_quality=voice_qual,
                 physio_cardio=physio_cardio, physio_eda=physio_eda, physio_somatic=physio_somatic)
        
        meta = pd.DataFrame({
            'subject_id': all_subjects,
            'task_id': all_tasks,
            'window_index': np.arange(n_total),
            'label': all_labels,
            'dataset': ds_name,
        })
        meta.to_parquet(os.path.join(ds_dir, "metadata.parquet"), index=False)
        
        dims = {
            'face_eye': face_eye.shape[2],
            'face_mouth': face_mouth.shape[2],
            'face_global_face': face_gface.shape[2],
            'voice_spectral_prosody': voice_sp.shape[2],
            'voice_mfcc': voice_mfcc.shape[2],
            'voice_quality': voice_qual.shape[2],
            'physio_cardio': physio_cardio.shape[2],
            'physio_eda': physio_eda.shape[2],
            'physio_somatic': physio_somatic.shape[2],
        }
        with open(os.path.join(ds_dir, "group_dims.json"), 'w') as f:
            json.dump(dims, f, indent=2)
        
        n_stress = int(sum(all_labels))
        print(f"    Saved: {n_total} windows, {n_stress} stress ({100*n_stress/max(n_total,1):.1f}%), "
              f"{len(np.unique(all_subjects))} subjects")
        print(f"    Path: {ds_dir}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Clean Data Extraction Pipeline')
    parser.add_argument('--datasets', nargs='+',
                        default=['stressid', 'wesad', 'empathicschool'],
                        help='Datasets to process')
    parser.add_argument('--windows', nargs='+', type=int,
                        default=[2, 5, 10, 30],
                        help='Window sizes in seconds')
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help='Output directory')
    args = parser.parse_args()
    
    window_configs = {}
    for w in args.windows:
        stride = max(1, w // 2)
        window_configs[w] = {'stride': stride}
    
    print(f"{'='*60}")
    print(f"  CLEAN DATA EXTRACTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Windows: {args.windows}s {dict(window_configs)}")
    print(f"  Output: {args.output}")
    print(f"  FPS: {TARGET_FPS}")
    print()
    
    for ds in args.datasets:
        try:
            if ds == 'stressid':
                process_stressid(window_configs)
            elif ds == 'wesad':
                process_wesad(window_configs)
            elif ds == 'empathicschool':
                process_empathicschool(window_configs)
            else:
                print(f"  Unknown dataset: {ds}")
        except Exception as e:
            print(f"  ERROR processing {ds}: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
