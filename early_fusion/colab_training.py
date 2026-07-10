"""
Google Colab Multimodal Stress Detection Expert Training Script (Updated & Corrected).
This script is self-contained. Copy-paste it into a Google Colab notebook cell
or run it directly in Colab after mounting Google Drive.

UPDATED FEATURES:
1. Window-based Feature Extraction: Instead of averaging features across the entire video or audio,
   features are extracted at the window level (1.0-second window, 0.5-second stride), aligning
   perfectly with the schema contract.
2. Comprehensive Metadata Preservation: Every extracted row preserves `subject_id`, `task_id`,
   `video_id`, `window_index`, `window_start`, and `window_end`.
3. Support for Temporal/Blink features: Computes standard deviations of eye openness (blink velocity)
   and landmarks over the temporal window.

IMPORTANT: Ensure you run this first in Colab to install dependencies:
!pip install mediapipe opencv-python librosa imbalanced-learn scikit-learn pandas numpy
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import io
import sys
import csv
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Check if running in Google Colab environment
if os.path.exists('/content/drive'):
    print("Google Drive mount detected. Using Colab paths...")
    BASE_DRIVE_DIR = '/content/drive/MyDrive/Multimodal_stress_Detection'
    STRESSID_ROOT = os.path.join(BASE_DRIVE_DIR, 'StressID', 'StressID Dataset')
else:
    print("Google Drive mount not detected. Running in local environment...")
    # Map to local repo root
    BASE_DRIVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    STRESSID_ROOT = os.path.join(BASE_DRIVE_DIR, 'certified_data') # locally fallback

# Configure paths based on BASE_DRIVE_DIR
FACES_ROOT = os.path.join(BASE_DRIVE_DIR, 'facesData')
STRESSID_AUDIO = os.path.join(STRESSID_ROOT, 'Audio')

# Setup Outputs
OUTPUT_DIR = os.path.join(BASE_DRIVE_DIR, 'backend', 'expert_models')
DATASET_OUT_DIR = os.path.join(BASE_DRIVE_DIR, 'dataset_extracted')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_OUT_DIR, exist_ok=True)

FACE_TRAIN_CSV = os.path.join(BASE_DRIVE_DIR, 'backend', 'training', 'face_indicators_train.csv')
FACE_TEST_CSV  = os.path.join(BASE_DRIVE_DIR, 'backend', 'training', 'face_indicators_test.csv')
os.makedirs(os.path.dirname(FACE_TRAIN_CSV), exist_ok=True)

FACE_STRESSID_CSV = os.path.join(DATASET_OUT_DIR, 'face_indicators_stressid.csv')
VOICE_STRESSID_CSV = os.path.join(DATASET_OUT_DIR, 'voice_indicators_stressid.csv')

FACE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'face_expert_lightweight.pkl')
FACE_SCALER_PATH = os.path.join(OUTPUT_DIR, 'face_scaler_lightweight.pkl')

VOICE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'voice_expert_lightweight.pkl')
VOICE_SCALER_PATH = os.path.join(OUTPUT_DIR, 'voice_scaler_lightweight.pkl')

FORCE_EXTRACTION = True  # Set to True to overwrite existing CSVs and Models

# -----------------------------------------------------------------------------
# 1. STANDALONE FEATURE EXTRACTORS
# -----------------------------------------------------------------------------

# --- Face Feature Extractor (using mediapipe) ---
MEDIAPIPE_AVAILABLE = False
USE_LEGACY_MEDIAPIPE = False

try:
    import cv2
    import mediapipe as mp
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
        USE_LEGACY_MEDIAPIPE = True
        MEDIAPIPE_AVAILABLE = True
        print("Using Legacy MediaPipe solutions API.")
    except (ImportError, AttributeError):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        USE_LEGACY_MEDIAPIPE = False
        MEDIAPIPE_AVAILABLE = True
        print("Using Modern MediaPipe Tasks API fallback.")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"Warning: mediapipe or opencv-python import failed: {e}. Face mesh feature extraction will not work.")

class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None

        if self.use_tasks:
            self.model_path = "face_landmarker.task"
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
                refine_landmarks=False,
                min_detection_confidence=0.5
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))

def compute_18_face_indicators(image_path):
    """Extract 18 face landmarks from a static image path using Python MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe not installed.")

    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    with FaceMeshWrapper(static_mode=True) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm  = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm])

    faceH = dist(pts, 10, 152) + 1e-6
    faceW = dist(pts, 234, 454) + 1e-6
    iod   = dist(pts, 33, 263) + 1e-6

    earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
    earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
    avgEAR = (earL + earR) / 2

    return [
        earL, earR, avgEAR,
        0.0,                                                    # blink_velocity
        dist(pts, 55, 159) / faceH,                             # brow_descent_left
        dist(pts, 285, 386) / faceH,                            # brow_descent_right
        abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,    # brow_asymmetry
        dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
        dist(pts, 4, 152) / iod,                                # jaw_tension
        (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
        dist(pts, 10, 151) / faceH,                             # forehead_tension
        faceH / iod,                                            # face_height_norm
        0.0,                                                    # head_tilt
        0.0,                                                    # temporal_x_var
        0.0,                                                    # temporal_y_var
        avgEAR,                                                 # eye_openness_ratio
        0.9,                                                    # landmark_confidence
        dist(pts, 4, 50) / faceH                                # nose_wrinkle
    ]

# --- Voice Feature Extractor (using librosa) ---
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa is not installed. Voice feature extraction will fail.")

def extract_voice_stress_indicators_from_slice(y, sr):
    """
    Extract 12 acoustic stress biomarkers from a preloaded audio slice/window.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa not installed.")

    if len(y) < sr * 0.5:
        return None

    indicators = {}
    EPS = 1e-10

    # 1-3: F0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=min(len(y), 2048)
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        indicators['f0_mean']  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        indicators['f0_std']   = float(np.std(f0_voiced))   if len(f0_voiced) > 0 else 0.0
        indicators['f0_range'] = float(np.ptp(f0_voiced))   if len(f0_voiced) > 0 else 0.0
    except Exception:
        indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0

    # 4-5: Jitter and Shimmer
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)

    try:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        periods = []
        amplitudes = []
        for frame in frames.T:
            ac = np.correlate(frame, frame, mode='full')[frame_len - 1:]
            ac = ac / (ac[0] + EPS)
            min_lag = int(sr / 500)
            max_lag = int(sr / 60)
            if max_lag < len(ac):
                peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
                periods.append(peak_idx)
            amplitudes.append(np.sqrt(np.mean(frame ** 2)))

        periods = np.array(periods, dtype=float)
        amplitudes = np.array(amplitudes, dtype=float)

        jitter  = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS)) if len(periods) > 1 else 0.0
        shimmer = float(np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + EPS)) if len(amplitudes) > 1 else 0.0
    except Exception:
        jitter, shimmer = 0.0, 0.0

    indicators['jitter_percent'] = min(jitter * 100, 10.0)
    indicators['shimmer_db']     = min(shimmer * 20, 5.0)

    # 6: HNR
    try:
        ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
        ac_norm = ac_full / (ac_full[0] + EPS)
        min_period = int(sr / 400)
        max_period = int(sr / 80)
        if max_period < len(ac_norm):
            peak_val = np.max(ac_norm[min_period:max_period])
            hnr = 10 * np.log10(peak_val / (1 - peak_val + EPS) + EPS)
        else:
            hnr = 0.0
    except Exception:
        hnr = 0.0
    indicators['hnr'] = float(np.clip(hnr, -20, 30))

    # 7: Zero Crossing Rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['speaking_rate_proxy'] = float(np.mean(zcr))
    except Exception:
        indicators['speaking_rate_proxy'] = 0.0

    # 8: RMS
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['voice_intensity'] = float(np.mean(rms))
    except Exception:
        indicators['voice_intensity'] = 0.0
        rms = np.array([0.0])

    # 9: High frequency ratio
    try:
        stft = np.abs(librosa.stft(y, n_fft=min(len(y), 512), hop_length=hop_len))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=min(len(y), 512))
        high_mask = freqs >= 3000
        total_energy = np.sum(stft) + EPS
        indicators['high_freq_ratio'] = float(np.sum(stft[high_mask]) / total_energy)
    except Exception:
        indicators['high_freq_ratio'] = 0.0
        stft = np.zeros((257, 1))

    # 10: Spectral flux
    try:
        spectral_flux = np.mean(np.diff(stft, axis=1) ** 2) if stft.shape[1] > 1 else 0.0
        indicators['spectral_flux'] = float(np.clip(spectral_flux, 0, 1))
    except Exception:
        indicators['spectral_flux'] = 0.0

    # 11: Pause ratio
    try:
        silence_thresh = 0.01 * np.max(np.abs(y))
        pause_frames = np.sum(rms < silence_thresh)
        indicators['pause_ratio'] = float(pause_frames / (len(rms) + EPS))
    except Exception:
        indicators['pause_ratio'] = 0.0

    # 12: Voiced fraction
    try:
        voiced_frac = float(np.sum(voiced_flag) / (len(voiced_flag) + EPS)) if 'voiced_flag' in locals() else 0.5
    except Exception:
        voiced_frac = 0.5
    indicators['voiced_fraction'] = voiced_frac

    return [
        indicators['f0_mean'],
        indicators['f0_std'],
        indicators['f0_range'],
        indicators['jitter_percent'],
        indicators['shimmer_db'],
        indicators['hnr'],
        indicators['speaking_rate_proxy'],
        indicators['voice_intensity'],
        indicators['high_freq_ratio'],
        indicators['spectral_flux'],
        indicators['pause_ratio'],
        indicators['voiced_fraction'],
    ]

# -----------------------------------------------------------------------------
# 2. OFFLINE DATA EXTRACTION
# -----------------------------------------------------------------------------

def run_face_extraction():
    """Extract features from the Western lab faces static image dataset (legacy support)"""
    print("\n" + "="*50)
    print("STAGE 1: Face Landmark Feature Extraction (Static Images)")
    print("="*50)

    if not os.path.exists(FACES_ROOT):
        print(f"Error: facesData root directory not found at: {FACES_ROOT}")
        return False

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is not installed.")
        return False

    def process_split(split_name, output_csv):
        if os.path.exists(output_csv) and not FORCE_EXTRACTION:
            print(f"Cache found for Face {split_name} split: {output_csv}. Skipping extraction...")
            return True

        rows = []
        for label_name, label_val in [('stress', 1), ('nostress', 0)]:
            folder = os.path.join(FACES_ROOT, split_name, label_name)
            if not os.path.exists(folder):
                continue

            images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(images)} images in {split_name}/{label_name}...")

            for i, img_name in enumerate(images):
                path = os.path.join(folder, img_name)
                try:
                    indicators = compute_18_face_indicators(path)
                    if indicators is not None:
                        # Dummy subject/task/window info for static facesData split to keep columns consistent
                        subject_id = "static_sub"
                        task_id = "static_task"
                        video_id = f"static_{split_name}_{label_name}"
                        window_index = i
                        window_start = float(i)
                        window_end = float(i + 1)
                        
                        rows.append([subject_id, task_id, video_id, window_index, window_start, window_end, label_val] + indicators)
                except Exception:
                    pass
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(images)} images...")

        if len(rows) == 0:
            return False

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'subject_id', 'task_id', 'video_id', 'window_index', 'window_start', 'window_end', 'label',
                'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
                'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
                'lip_compression', 'jaw_tension', 'mouth_corner_pull',
                'forehead_tension', 'face_height_norm', 'head_tilt',
                'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
                'landmark_confidence', 'nose_wrinkle'
            ]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Saved {len(rows)} samples to {output_csv}")
        return True

    print("Extracting training split...")
    train_success = process_split('train', FACE_TRAIN_CSV)
    print("Extracting testing split...")
    test_success = process_split('test', FACE_TEST_CSV)

    return train_success and test_success

def extract_face_windows_from_video(video_path, subject_id, task_id, video_id, label_val, fm, window_len=1.0, stride=0.5):
    """
    Extract face indicator windows (1.0s length, 0.5s stride) from video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    frame_data = []  # List of tuples: (timestamp, indicators)
    frame_count = 0
    
    # Process frames at 10 Hz (sample rate of 10 frames per second to balance computation and accuracy)
    sample_rate = 10.0
    sample_interval = max(1, int(fps / sample_rate))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            t = frame_count / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([[l.x * w, l.y * h] for l in lm])

                faceH = dist(pts, 10, 152) + 1e-6
                faceW = dist(pts, 234, 454) + 1e-6
                iod   = dist(pts, 33, 263) + 1e-6

                earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
                earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
                avgEAR = (earL + earR) / 2

                indicators = [
                    earL, earR, avgEAR,
                    0.0,                                                    # blink_velocity
                    dist(pts, 55, 159) / faceH,                             # brow_descent_left
                    dist(pts, 285, 386) / faceH,                            # brow_descent_right
                    abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,    # brow_asymmetry
                    dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
                    faceW / faceH,                                          # jaw_tension
                    (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
                    dist(pts, 10, 151) / faceH,                             # forehead_tension
                    faceH / iod,                                            # face_height_norm
                    0.0,                                                    # head_tilt
                    0.0,                                                    # temporal_x_var
                    0.0,                                                    # temporal_y_var
                    avgEAR,                                                 # eye_openness_ratio
                    0.9,                                                    # landmark_confidence
                    dist(pts, 4, 50) / faceH                                # nose_wrinkle
                ]
                frame_data.append((t, indicators))
        frame_count += 1
    cap.release()
    
    if not frame_data:
        return []
        
    max_t = frame_data[-1][0]
    windows = []
    window_index = 0
    
    while True:
        window_start = window_index * stride
        window_end = window_start + window_len
        if window_start > max_t:
            break
            
        # Select frames within the window
        window_frames = [ind for t, ind in frame_data if window_start <= t <= window_end]
        if len(window_frames) >= 1:
            # Average indicators over the window
            avg_indicators = list(np.mean(window_frames, axis=0))
            
            # Extract standard deviation metrics for dynamics
            ears = [ind[2] for ind in window_frames]
            blink_vel = float(np.std(ears)) if len(ears) > 1 else 0.0
            avg_indicators[3] = blink_vel
            
            brow_ds = [ind[4] for ind in window_frames]
            mouth_cp = [ind[9] for ind in window_frames]
            avg_indicators[13] = float(np.std(brow_ds)) if len(brow_ds) > 1 else 0.0
            avg_indicators[14] = float(np.std(mouth_cp)) if len(mouth_cp) > 1 else 0.0
            
            row = [subject_id, task_id, video_id, window_index, window_start, window_end, label_val] + avg_indicators
            windows.append(row)
            
        window_index += 1
        
    return windows

def run_face_extraction_stressid_videos():
    print("\n" + "="*50)
    print("STAGE 1: Face Landmark Feature Extraction (StressID Videos - Windowed)")
    print("="*50)

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is not installed.")
        return False

    labels_csv = os.path.join(STRESSID_ROOT, 'labels.csv')
    if not os.path.exists(labels_csv):
        labels_csv_fallback = os.path.join(STRESSID_ROOT, '._labels.csv')
        if os.path.exists(labels_csv_fallback):
            labels_csv = labels_csv_fallback

    if not os.path.exists(labels_csv):
        print(f"Error: labels.csv not found at {labels_csv}")
        return False

    try:
        labels_df = pd.read_csv(labels_csv)
    except Exception as e:
        print(f"Error reading labels CSV: {e}")
        return False

    output_csv = FACE_STRESSID_CSV
    if os.path.exists(output_csv) and not FORCE_EXTRACTION:
        print(f"Cache found for StressID Face features: {output_csv}. Skipping extraction...")
        return True

    rows = []
    print(f"Processing videos from {STRESSID_ROOT} based on labels...")

    with FaceMeshWrapper(static_mode=False) as fm:
        for idx, row in labels_df.iterrows():
            subject_task = row.get('subject/task')
            if not subject_task:
                continue
            
            parts = subject_task.split('_')
            subject = parts[0]
            task = parts[1] if len(parts) > 1 else "Unknown"
            
            label_val = row.get('binary-stress')
            if label_val is None or pd.isna(label_val):
                continue

            video_path = os.path.join(STRESSID_ROOT, 'Videos', subject, f"{subject_task}.mp4")
            if not os.path.exists(video_path):
                continue

            print(f"Extracting windowed features from: {subject_task}.mp4 ...")
            video_windows = extract_face_windows_from_video(video_path, subject, task, subject_task, int(label_val), fm)
            if video_windows:
                rows.extend(video_windows)
                print(f"  > Success: extracted {len(video_windows)} synchronized temporal windows.")

    if len(rows) == 0:
        print("Error: No facial features extracted from videos.")
        return False

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'subject_id', 'task_id', 'video_id', 'window_index', 'window_start', 'window_end', 'label',
            'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
            'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
            'lip_compression', 'jaw_tension', 'mouth_corner_pull',
            'forehead_tension', 'face_height_norm', 'head_tilt',
            'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
            'landmark_confidence', 'nose_wrinkle'
        ]
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {output_csv}")
    return True

def extract_voice_windows_from_audio(fpath, subject_id, task_id, video_id, label_val, sr_target=16000, window_len=1.0, stride=0.5):
    """
    Extract Voice indicator windows (1.0s length, 0.5s stride) from an audio recording.
    """
    if not os.path.exists(fpath):
        return []
        
    try:
        y, sr = librosa.load(fpath, sr=sr_target, mono=True)
    except Exception as e:
        print(f"Error loading audio file {fpath}: {e}")
        return []
        
    duration = len(y) / sr
    windows = []
    window_index = 0
    
    while True:
        window_start = window_index * stride
        window_end = window_start + window_len
        if window_end > duration:
            break
            
        start_sample = int(window_start * sr)
        end_sample = int(window_end * sr)
        y_window = y[start_sample:end_sample]
        
        if len(y_window) < sr * 0.5:
            break
            
        res = extract_voice_stress_indicators_from_slice(y_window, sr)
        if res is not None:
            row = [subject_id, task_id, video_id, window_index, window_start, window_end, label_val] + res
            windows.append(row)
            
        window_index += 1
        
    return windows

# -----------------------------------------------------------------------------
# 3. TRAINING MODELS
# -----------------------------------------------------------------------------

def train_face_expert():
    print("\n" + "="*50)
    print("STAGE 2: Training Lightweight Face Expert")
    print("="*50)

    if os.path.exists(FACE_STRESSID_CSV):
        df = pd.read_csv(FACE_STRESSID_CSV)
        print(f"Loaded StressID Face features dataset: {FACE_STRESSID_CSV}")
    elif os.path.exists(FACE_TRAIN_CSV) and os.path.exists(FACE_TEST_CSV):
        df_train = pd.read_csv(FACE_TRAIN_CSV)
        df_test  = pd.read_csv(FACE_TEST_CSV)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print("Loaded facesData (train/test) dataset")
    else:
        print("Error: Extracted face CSV files missing. Run extraction first.")
        return

    FEATURES = [
        'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
        'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
        'lip_compression', 'jaw_tension', 'mouth_corner_pull',
        'forehead_tension', 'face_height_norm', 'head_tilt',
        'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
        'landmark_confidence', 'nose_wrinkle',
    ]

    X = df[FEATURES].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f'Train size: {len(X_train)}  |  Test size: {len(X_test)}')
    print(f'Class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}')

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    print("Fitting Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train_res)

    y_pred = model.predict(X_test_scaled)
    print('\nTest Results:')
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_res, cv=cv, scoring='f1')
    print(f'5-Fold CV F1-Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    with open(FACE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(FACE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {FACE_MODEL_PATH}")
    print(f"Saved: {FACE_SCALER_PATH}")

def train_voice_expert():
    print("\n" + "="*50)
    print("STAGE 3: Training Lightweight Voice Expert")
    print("="*50)

    FEATURE_NAMES = [
        'subject_id', 'task_id', 'video_id', 'window_index', 'window_start', 'window_end', 'label',
        'f0_mean', 'f0_std', 'f0_range', 'jitter_percent', 'shimmer_db',
        'hnr', 'speaking_rate_proxy', 'voice_intensity', 'high_freq_ratio',
        'spectral_flux', 'pause_ratio', 'voiced_fraction'
    ]

    if os.path.exists(VOICE_STRESSID_CSV) and not FORCE_EXTRACTION:
        print(f"Cache found for StressID Voice features: {VOICE_STRESSID_CSV}. Loading cached data...")
        df = pd.read_csv(VOICE_STRESSID_CSV)
    else:
        if not os.path.exists(STRESSID_AUDIO):
            print(f"Error: StressID audio path not found at: {STRESSID_AUDIO}")
            return

        labels_csv = os.path.join(STRESSID_ROOT, 'labels.csv')
        if not os.path.exists(labels_csv):
            labels_csv_fallback = os.path.join(STRESSID_ROOT, '._labels.csv')
            if os.path.exists(labels_csv_fallback):
                labels_csv = labels_csv_fallback

        use_csv_labels = False
        label_map = {}
        if os.path.exists(labels_csv):
            try:
                labels_df = pd.read_csv(labels_csv)
                for _, row in labels_df.iterrows():
                    st = row.get('subject/task')
                    lbl = row.get('binary-stress')
                    if st and lbl is not None and not pd.isna(lbl):
                        label_map[str(st).strip().lower()] = int(lbl)
                use_csv_labels = True
                print(f"Loaded {len(label_map)} labels from {labels_csv} for audio mapping.")
            except Exception as e:
                print(f"Warning: Could not read labels CSV for audio mapping: {e}")

        # Fallback keyword matching if not found in CSV labels
        STRESS_CONDITIONS = ['public_speaking', 'mental_math', 'stroop', 'math', 'speaking', 'stress']
        CALM_CONDITIONS   = ['rest', 'baseline', 'relax', 'calm', 'breathing', 'reading', 'video', 'nostress']

        rows = []
        audio_files = []
        for root, dirs, files in os.walk(STRESSID_AUDIO):
            for fname in files:
                if fname.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    audio_files.append((root, fname))

        print(f"Found {len(audio_files)} audio samples. Commencing windowed feature extraction...")

        for idx, (root, fname) in enumerate(audio_files):
            fpath = os.path.join(root, fname)
            base_name = os.path.splitext(fname)[0].strip()
            base_name_lower = base_name.lower()

            parts = base_name.split('_')
            subject = parts[0]
            task = parts[1] if len(parts) > 1 else "Unknown"

            label = None
            if use_csv_labels and base_name_lower in label_map:
                label = label_map[base_name_lower]
            else:
                for sc in STRESS_CONDITIONS:
                    if sc in base_name_lower:
                        label = 1
                        break
                if label is None:
                    for cc in CALM_CONDITIONS:
                        if cc in base_name_lower:
                            label = 0
                            break
            
            if label is None:
                label = 0  # Default fallback

            try:
                audio_windows = extract_voice_windows_from_audio(fpath, subject, task, base_name, int(label))
                if audio_windows:
                    rows.extend(audio_windows)
            except Exception:
                pass

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(audio_files)} audio files...")

        if len(rows) == 0:
            print("Error: No voice features extracted.")
            return

        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        df.to_csv(VOICE_STRESSID_CSV, index=False)
        print(f"Saved {len(df)} extracted voice samples to cached CSV: {VOICE_STRESSID_CSV}")

    print(f'Using voice dataset with {len(df)} samples.')
    print(df['label'].value_counts())

    # Training feature subset
    FEATURES = [
        'f0_mean', 'f0_std', 'f0_range', 'jitter_percent', 'shimmer_db',
        'hnr', 'speaking_rate_proxy', 'voice_intensity', 'high_freq_ratio',
        'spectral_flux', 'pause_ratio', 'voiced_fraction'
    ]

    X = df[FEATURES].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res_scaled  = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    print("Fitting Voice Gradient Boosting Expert...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_res_scaled, y_res)

    y_pred = model.predict(X_test_scaled)
    print('\nTest Results:')
    print(classification_report(y_test, y_pred, target_names=['Calm', 'Stress']))

    with open(VOICE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VOICE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {VOICE_MODEL_PATH}")
    print(f"Saved: {VOICE_SCALER_PATH}")

# -----------------------------------------------------------------------------
# 4. SCRIPT RUNNER
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("GOOGLE COLAB / DRIVE MULTIMODAL STRESS EXPERTS TRAINING PIPELINE")
    print("="*60)

    start_time = time.time()

    face_model_exists = os.path.exists(FACE_MODEL_PATH) and os.path.exists(FACE_SCALER_PATH)
    if face_model_exists and not FORCE_EXTRACTION:
        print("\n>>> Face expert model already exists. Skipping face extraction and training.")
    else:
        print("\nAttempting extraction from StressID video files...")
        face_extracted = run_face_extraction_stressid_videos()

        if not face_extracted:
            print("StressID Videos extraction skipped or failed. Falling back to facesData images...")
            face_extracted = run_face_extraction()

        if face_extracted:
            train_face_expert()

    voice_model_exists = os.path.exists(VOICE_MODEL_PATH) and os.path.exists(VOICE_SCALER_PATH)
    if voice_model_exists and not FORCE_EXTRACTION:
        print("\n>>> Voice expert model already exists. Skipping voice extraction.")
    else:
        train_voice_expert()

    total_m = (time.time() - start_time) / 60.0
    print("\n" + "="*50)
    print(f"Pipeline executed successfully in {total_m:.2f} minutes.")
    print(f"Trained models saved in Google Drive folder: {OUTPUT_DIR}")
    print("="*50)
