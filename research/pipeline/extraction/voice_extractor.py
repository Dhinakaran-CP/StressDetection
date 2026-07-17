import os
import json
import glob
import cv2
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import write_json, read_json

# Set determinism first
set_determinism()

def compute_hnr(y, sr, fmin=75, fmax=300):
    corr = np.correlate(y, y, mode='full')
    corr = corr[len(corr)//2:]
    
    low_lag = int(sr / fmax)
    high_lag = int(sr / fmin)
    
    if len(corr) <= high_lag:
        return 0.0
        
    peak_lag = np.argmax(corr[low_lag:high_lag]) + low_lag
    r_0 = corr[0] + 1e-8
    r_tau = corr[peak_lag]
    
    hnr = 10 * np.log10(max(1e-8, r_tau) / max(1e-8, r_0 - r_tau))
    return max(0.0, hnr)

def compute_jitter(pitches):
    valid = pitches[~np.isnan(pitches)]
    if len(valid) < 2:
        return 0.0
    diffs = np.abs(np.diff(valid))
    mean_diff = np.mean(diffs)
    mean_val = np.mean(valid) + 1e-8
    return mean_diff / mean_val

def compute_24_features(chunk, sr):
    if len(chunk) < 256:
        return None
        
    try:
        rms = librosa.feature.rms(y=chunk).mean()
        zcr = librosa.feature.zero_crossing_rate(y=chunk).mean()
        
        # Pitch tracking
        pitches, voiced_flag, voiced_probs = librosa.pyin(chunk, fmin=75, fmax=300, sr=sr)
        valid_pitches = pitches[~np.isnan(pitches)]
        f0_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else np.nan
        f0_std = np.std(valid_pitches) if len(valid_pitches) > 0 else np.nan
        
        # MFCCs (13)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        mfccs = mfcc.mean(axis=1).tolist()
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr, roll_percent=0.85).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=chunk).mean()
        chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr).mean()
        
        # Voice quality features
        hnr = compute_hnr(chunk, sr)
        jitter = compute_jitter(pitches)
        
        feat = {
            "rms": rms,
            "zcr": zcr,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flatness": spectral_flatness,
            "chroma_stft": chroma_stft,
            "hnr": hnr,
            "jitter": jitter
        }
        
        for i, m_val in enumerate(mfccs):
            feat[f"mfcc_{i+1}"] = m_val
            
        return feat
    except Exception:
        return None

def extract_windows_and_sequences(features_list, subject_id, task_name, binary_stress, window_size=30, stride=15):
    flat_records = []
    sequences_list = []
    
    n_frames = len(features_list)
    n_windows = int((n_frames - window_size) // stride) + 1
    
    fields = [
        "rms", "zcr", "f0_mean", "f0_std",
        "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "chroma_stft", "hnr", "jitter"
    ]
    
    for w_idx in range(n_windows):
        start = w_idx * stride
        end = start + window_size
        window_feats = features_list[start:end]
        
        valid_feats = [f for f in window_feats if f is not None]
        
        # Check voice detection availability (>50% frames have valid features and RMS above silence threshold)
        non_silent_feats = [f for f in valid_feats if f["rms"] > 0.001 and not np.isnan(f["f0_mean"])]
        voice_available = 1 if len(non_silent_feats) > (window_size / 2) else 0
        
        window_id = f"{subject_id}_{task_name}_W{w_idx}"
        
        flat_record = {
            "subject_id": subject_id,
            "dataset_source": "stressid",
            "task_name": task_name,
            "window_id": window_id,
            "voice_available": voice_available,
            "binary_stress": binary_stress
        }
        
        sequence_matrix = np.zeros((window_size, len(fields)), dtype=np.float32)
        
        for i_f, f in enumerate(window_feats):
            if f is not None:
                for col_idx, field in enumerate(fields):
                    sequence_matrix[i_f, col_idx] = f[field]
            else:
                sequence_matrix[i_f, :] = np.nan
                
        if voice_available:
            for field in fields:
                vals = [f[field] for f in valid_feats if not np.isnan(f[field])]
                if vals:
                    flat_record[f"{field}_mean"] = np.mean(vals)
                    flat_record[f"{field}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                    flat_record[f"{field}_min"] = np.min(vals)
                    flat_record[f"{field}_max"] = np.max(vals)
                    flat_record[f"{field}_range"] = np.max(vals) - np.min(vals)
                else:
                    flat_record[f"{field}_mean"] = np.nan
                    flat_record[f"{field}_std"] = np.nan
                    flat_record[f"{field}_min"] = np.nan
                    flat_record[f"{field}_max"] = np.nan
                    flat_record[f"{field}_range"] = np.nan
        else:
            for field in fields:
                flat_record[f"{field}_mean"] = np.nan
                flat_record[f"{field}_std"] = np.nan
                flat_record[f"{field}_min"] = np.nan
                flat_record[f"{field}_max"] = np.nan
                flat_record[f"{field}_range"] = np.nan
                
        flat_records.append(flat_record)
        sequences_list.append(sequence_matrix)
        
    return flat_records, sequences_list

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    config_path = base_dir / "pipeline" / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
        
    stressid_raw = base_dir / config["datasets"]["stressid"]["raw_path"]
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    
    log_file = base_dir / "pipeline" / "logs" / "voice_extraction.log"
    if log_file.exists():
        log_file.unlink()
        
    labels_csv_path = stressid_raw / "labels.csv"
    df_labels = pd.read_csv(labels_csv_path)
    label_map = {row['subject/task']: int(row['binary-stress']) for _, row in df_labels.iterrows()}
    
    subjects = sorted([x for x in os.listdir(stressid_raw / "Videos") if not x.startswith('.')])
    
    flat_records_all = []
    sequences_all = []
    window_meta_all = []
    
    total_processed_files = 0
    total_silent_frames = 0
    total_windows_produced = 0
    
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write("--- StressID Voice Extraction ---\n")
        
    for sub in tqdm(subjects, desc="StressID Voice"):
        video_dir = stressid_raw / "Videos" / sub
        video_files = list(video_dir.glob("*.mp4"))
        
        for video_path in video_files:
            task_name = video_path.stem.replace(f"{sub}_", "")
            key = f"{sub}_{task_name}"
            if key not in label_map:
                continue
            lbl = label_map[key]
            
            # Read video properties to get canonical frame count N
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(fps / 3.0))
            frame_idx = 0
            n_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    n_frames += 1
                frame_idx += 1
            cap.release()
            
            if n_frames == 0:
                continue
                
            # Audio path
            audio_path = stressid_raw / "Audio" / sub / f"{sub}_{task_name}.wav"
            
            features_list = []
            
            if audio_path.exists():
                try:
                    # Load audio
                    y, sr = librosa.load(str(audio_path), sr=None)
                    
                    # Split into exactly n_frames equal duration chunks to match face extractor
                    chunk_size = len(y) / n_frames
                    
                    for i in range(n_frames):
                        start_sample = int(i * chunk_size)
                        end_sample = min(len(y), int((i + 1) * chunk_size))
                        chunk = y[start_sample:end_sample]
                        
                        feat = compute_24_features(chunk, sr)
                        features_list.append(feat)
                        
                        if feat is None or feat["rms"] <= 0.001 or np.isnan(feat["f0_mean"]):
                            total_silent_frames += 1
                            
                    total_processed_files += 1
                except Exception as e:
                    print(f"Error reading audio {audio_path}: {e}")
                    features_list = [None] * n_frames
            else:
                # No audio file for non-speaking tasks
                features_list = [None] * n_frames
                
            flat_rec, seqs = extract_windows_and_sequences(features_list, sub, task_name, lbl)
            
            flat_records_all.extend(flat_rec)
            for fr, seq in zip(flat_rec, seqs):
                global_idx = len(sequences_all)
                sequences_all.append(seq)
                window_meta_all.append({
                    "window_id": fr["window_id"],
                    "sequence_index": global_idx
                })
                total_windows_produced += 1
                
            with open(log_file, "a", encoding="utf-8") as f_log:
                f_log.write(f"Subject: {sub}, Task: {task_name}, Frames: {n_frames}, Windows: {len(flat_rec)}\n")
                
    # Save outputs
    if flat_records_all:
        df_flat = pd.DataFrame(flat_records_all)
        df_flat.to_parquet(sid_out / "voice_windows.parquet")
        
        np.save(sid_out / "voice_sequences.npy", np.array(sequences_all, dtype=np.float32))
        pd.DataFrame(window_meta_all).to_parquet(sid_out / "voice_sequences_index.parquet")
        
    # Self-verification check
    # Metadata columns: subject_id, dataset_source, task_name, window_id, voice_available, binary_stress (6 cols)
    # Features: 24 features * 5 aggregations = 120 columns
    # Total = 126 columns
    
    issues = []
    pq_file = sid_out / "voice_windows.parquet"
    if not pq_file.exists():
        issues.append("voice_windows.parquet missing")
    else:
        df = pd.read_parquet(pq_file)
        if len(df.columns) != 126:
            issues.append(f"voice_windows.parquet columns count mismatch: expected 126, got {len(df.columns)}")
            
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("Voice extraction verification PASSED.")

if __name__ == "__main__":
    main()
