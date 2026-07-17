import os
import json
import glob
import urllib.request
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import read_csv_or_xls, write_json, read_json

# Set determinism first
set_determinism()

# MediaPipe face mesh index groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
BROW = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

mp_to_dlib = {
    33: 36, 133: 39, 263: 45, 362: 42,
    159: 37, 145: 41, 386: 43, 374: 47,
    158: 38, 153: 40, 385: 44, 380: 46,
    55: 19, 285: 24, 13: 62, 14: 66,
    61: 48, 291: 54, 10: 27, 151: 28,
    152: 8, 4: 30, 50: 29, 172: 2, 397: 14
}

# 3D model points of standard face
model_pts = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)

def estimate_head_pose(pts_6):
    focal_length = 640
    center = (320, 240)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_pts, pts_6, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0
    rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    return float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])

def compute_18_features(pt_func, format_type='mediapipe'):
    def get_pt(idx):
        if format_type == 'dlib':
            dlib_idx = mp_to_dlib.get(idx, None)
            if dlib_idx is None:
                if idx == 10: return pt_func(27)
                if idx == 151: return pt_func(28)
                return pt_func(0)
            return pt_func(dlib_idx)
        else:
            return pt_func(idx)

    def dist(a, b):
        return np.linalg.norm(a - b)

    try:
        forehead = get_pt(10)
        chin = get_pt(152)
        faceH = dist(forehead, chin) + 1e-6
        
        eyeL_outer = get_pt(33)
        eyeR_outer = get_pt(263)
        iod = dist(eyeL_outer, eyeR_outer) + 1e-6

        # EAR
        earL = (dist(get_pt(159), get_pt(145)) + dist(get_pt(158), get_pt(153))) / (2 * dist(eyeL_outer, get_pt(133)) + 1e-6)
        earR = (dist(get_pt(386), get_pt(374)) + dist(get_pt(385), get_pt(380))) / (2 * dist(get_pt(362), eyeR_outer) + 1e-6)
        avgEAR = (earL + earR) / 2.0

        # Brow descent
        browDescL = dist(get_pt(55), get_pt(159)) / faceH
        browDescR = dist(get_pt(285), get_pt(386)) / faceH
        browAsym = abs(browDescL - browDescR)

        # Lip compression
        lipGap = dist(get_pt(13), get_pt(14))
        lipWidth = dist(get_pt(61), get_pt(291)) + 1e-6
        lipCompression = lipGap / lipWidth

        # Jaw tension
        jawDisplacement = dist(get_pt(4), chin) / iod
        jawAngleWidth = dist(get_pt(172), get_pt(397)) / iod

        # Mouth corner pull
        noseTip = get_pt(4)
        mcPull = (dist(get_pt(61), noseTip) + dist(get_pt(291), noseTip)) / (2 * faceH)

        # Forehead tension
        foreheadTension = dist(forehead, get_pt(151)) / faceH

        # Head tilt
        headTilt = np.abs(np.arctan2(eyeR_outer[1] - eyeL_outer[1], eyeR_outer[0] - eyeL_outer[0]) * (180.0 / np.pi))

        # Nose wrinkle
        nose_wrinkle = dist(noseTip, get_pt(50)) / faceH

        # 3D pose using 6 landmarks
        pts_6 = np.array([
            noseTip, chin, eyeL_outer, eyeR_outer, get_pt(61), get_pt(291)
        ], dtype=np.float32)
        pitch, yaw, roll = estimate_head_pose(pts_6)

        return {
            "left_ear": earL,
            "right_ear": earR,
            "avg_ear": avgEAR,
            "blink_velocity": 0.0, # Filled temporally later
            "eye_openness_ratio": avgEAR,
            "brow_descent_left": browDescL,
            "brow_descent_right": browDescR,
            "brow_asymmetry": browAsym,
            "lip_compression": lipCompression,
            "jaw_tension": jawAngleWidth,
            "mouth_corner_pull": mcPull,
            "forehead_tension": foreheadTension,
            "nose_wrinkle": nose_wrinkle,
            "face_height_norm": faceH / iod,
            "head_tilt": headTilt,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }
    except Exception:
        return None

# Setup detector wrapper
class MediaPipeDetector:
    def __init__(self):
        self.model_path = Path("pipeline/config/face_landmarker.task")
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print("Downloading face_landmarker.task...")
            urllib.request.urlretrieve(url, self.model_path)
            
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame_rgb):
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.detector.detect(mp_image)
        if res and res.face_landmarks and len(res.face_landmarks) > 0:
            return res.face_landmarks[0]
        return None

def process_frames_to_features(frame_landmarks_list, format_type):
    """
    Computes 18 static + 16 temporal delta features for a list of frames.
    """
    features_list = []
    prev_ear = None
    
    for i, landmarks in enumerate(frame_landmarks_list):
        if landmarks is None:
            features_list.append(None)
            prev_ear = None
            continue
            
        def get_landmark_coords(idx):
            if format_type == 'dlib':
                pt = landmarks[idx]
                return np.array([float(pt['x']), float(pt['y'])])
            else:
                pt = landmarks[idx]
                # Scale by standard size to match pixel distances
                return np.array([pt.x * 640.0, pt.y * 480.0])
                
        feat = compute_18_features(get_landmark_coords, format_type)
        if feat is None:
            features_list.append(None)
            prev_ear = None
            continue
            
        # Blink velocity
        if prev_ear is not None:
            feat["blink_velocity"] = abs(feat["avg_ear"] - prev_ear) / 0.333
        else:
            feat["blink_velocity"] = 0.0
        prev_ear = feat["avg_ear"]
        
        # Add temporal delta features (16 features)
        delta_fields = [
            "left_ear", "right_ear", "blink_velocity", "brow_descent_left",
            "brow_descent_right", "brow_asymmetry", "lip_compression", "jaw_tension",
            "mouth_corner_pull", "forehead_tension", "head_tilt", "pitch",
            "yaw", "roll", "eye_openness_ratio", "nose_wrinkle"
        ]
        
        if i > 0 and features_list[i-1] is not None:
            prev_feat = features_list[i-1]
            for f in delta_fields:
                feat[f"delta_{f}"] = feat[f] - prev_feat[f]
        else:
            for f in delta_fields:
                feat[f"delta_{f}"] = 0.0
                
        features_list.append(feat)
        
    return features_list

def extract_windows_and_sequences(features_list, subject_id, dataset_source, task_name, binary_stress, window_size=30, stride=15):
    """
    Builds flat parquets and sequence matrices using 10-second windows (30 frames at 3 fps).
    """
    flat_records = []
    sequences_list = []
    
    n_frames = len(features_list)
    n_windows = int((n_frames - window_size) // stride) + 1
    
    fields = [
        "left_ear", "right_ear", "avg_ear", "blink_velocity", "eye_openness_ratio",
        "brow_descent_left", "brow_descent_right", "brow_asymmetry", "lip_compression",
        "jaw_tension", "mouth_corner_pull", "forehead_tension", "nose_wrinkle",
        "face_height_norm", "head_tilt", "pitch", "yaw", "roll",
        "delta_left_ear", "delta_right_ear", "delta_blink_velocity", "delta_brow_descent_left",
        "delta_brow_descent_right", "delta_brow_asymmetry", "delta_lip_compression", "delta_jaw_tension",
        "delta_mouth_corner_pull", "delta_forehead_tension", "delta_head_tilt", "delta_pitch",
        "delta_yaw", "delta_roll", "delta_eye_openness_ratio", "delta_nose_wrinkle"
    ]
    
    for w_idx in range(n_windows):
        start = w_idx * stride
        end = start + window_size
        window_feats = features_list[start:end]
        
        # Check face detection availability (>50% frames)
        valid_feats = [f for f in window_feats if f is not None]
        face_available = 1 if len(valid_feats) > (window_size / 2) else 0
        
        window_id = f"{subject_id}_{task_name}_W{w_idx}"
        
        flat_record = {
            "subject_id": subject_id,
            "dataset_source": dataset_source,
            "task_name": task_name,
            "window_id": window_id,
            "face_available": face_available,
            "binary_stress": binary_stress
        }
        
        sequence_matrix = np.zeros((window_size, len(fields)), dtype=np.float32)
        
        for i_f, f in enumerate(window_feats):
            if f is not None:
                for col_idx, field in enumerate(fields):
                    sequence_matrix[i_f, col_idx] = f[field]
            else:
                sequence_matrix[i_f, :] = np.nan
                
        if face_available:
            # Flatten to mean, std, min, max, range
            for field in fields:
                vals = [f[field] for f in valid_feats]
                flat_record[f"{field}_mean"] = np.mean(vals)
                flat_record[f"{field}_std"] = np.std(vals)
                flat_record[f"{field}_min"] = np.min(vals)
                flat_record[f"{field}_max"] = np.max(vals)
                flat_record[f"{field}_range"] = np.max(vals) - np.min(vals)
        else:
            # Fill with NaN
            for field in fields:
                flat_record[f"{field}_mean"] = np.nan
                flat_record[f"{field}_std"] = np.nan
                flat_record[f"{field}_min"] = np.nan
                flat_record[f"{field}_max"] = np.nan
                flat_record[f"{field}_range"] = np.nan
                
        flat_records.append(flat_record)
        sequences_list.append(sequence_matrix)
        
    return flat_records, sequences_list

def run_stressid_extraction(detector, raw_path, output_dir, log_file):
    print("Running StressID Face extraction...")
    labels_csv_path = raw_path / "labels.csv"
    df_labels = pd.read_csv(labels_csv_path)
    
    label_map = {row['subject/task']: int(row['binary-stress']) for _, row in df_labels.iterrows()}
    
    # List subjects
    subjects = sorted([x for x in os.listdir(raw_path / "Videos") if not x.startswith('.')])
    
    flat_records_all = []
    sequences_all = []
    window_meta_all = []
    
    total_processed_frames = 0
    total_missed_faces = 0
    total_windows_produced = 0
    
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write("--- StressID Face Extraction ---\n")
        
    for sub in tqdm(subjects, desc="StressID Face"):
        video_dir = raw_path / "Videos" / sub
        video_files = list(video_dir.glob("*.mp4"))
        
        for video_path in video_files:
            task_name = video_path.stem.replace(f"{sub}_", "")
            key = f"{sub}_{task_name}"
            if key not in label_map:
                continue
            lbl = label_map[key]
            
            # Read video and sample at 3 FPS (every 10th frame at 30 fps)
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(fps / 3.0))
            
            frame_idx = 0
            landmarks_list = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lm = detector.detect_landmarks(rgb)
                    landmarks_list.append(lm)
                    total_processed_frames += 1
                    if lm is None:
                        total_missed_faces += 1
                frame_idx += 1
            cap.release()
            
            if not landmarks_list:
                continue
                
            features = process_frames_to_features(landmarks_list, format_type='mediapipe')
            flat_rec, seqs = extract_windows_and_sequences(features, sub, "stressid", task_name, lbl)
            
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
                f_log.write(f"Subject: {sub}, Task: {task_name}, Frames: {len(landmarks_list)}, Windows: {len(flat_rec)}\n")
                
    # Save outputs
    if flat_records_all:
        df_flat = pd.DataFrame(flat_records_all)
        df_flat.to_parquet(output_dir / "face_windows.parquet")
        
        np.save(output_dir / "face_sequences.npy", np.array(sequences_all, dtype=np.float32))
        pd.DataFrame(window_meta_all).to_parquet(output_dir / "face_sequences_index.parquet")
        
    print(f"StressID Completed. Windows: {total_windows_produced}")

def run_empathicschool_extraction(raw_path, output_dir, log_file):
    print("Running EmpathicSchool Face extraction...")
    
    # We load labels map from survey Excel files recursively
    # First, let's find the survey xlsx file per subject
    subjects = [f"S{i}" for i in range(1, 31)]
    
    flat_records_all = []
    sequences_all = []
    window_meta_all = []
    
    total_processed_frames = 0
    total_missed_faces = 0
    total_windows_produced = 0
    
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write("\n--- EmpathicSchool Face Extraction ---\n")
        
    for sub in tqdm(subjects, desc="EmpathicSchool Face"):
        sub_dir = raw_path / sub
        if not sub_dir.exists():
            continue
            
        # Search for survey xlsx file
        xlsx_files = list(sub_dir.glob("**/*.xlsx"))
        xlsx_files = [x for x in xlsx_files if "xception" not in x.name.lower() and not x.name.startswith("~$")]
        
        survey_map = {}
        if xlsx_files:
            try:
                df_survey = pd.read_excel(xlsx_files[0])
                for _, row in df_survey.iterrows():
                    task_name = str(row['Task']).strip()
                    if pd.isna(row.get('Mental Demand')):
                        continue
                    mental = float(row.get('Mental Demand', 0))
                    physical = float(row.get('Physical demand', 0))
                    temporal = float(row.get('Temporal demand', 0))
                    perf = float(row.get('Performance', 0))
                    effort = float(row.get('Effort', 0))
                    frust = float(row.get('Frustration', 0))
                    
                    nasa_tlx = (mental + physical + temporal + perf + effort + frust) * (100.0 / 120.0)
                    binary_stress = 1 if nasa_tlx >= 50.0 else 0
                    survey_map[task_name] = binary_stress
            except Exception as e:
                print(f"Error parsing labels for subject {sub}: {e}")
                
        # Find all Landmarks json files
        landmark_files = list(sub_dir.glob("**/*Landmark*.json"))
        landmark_files = [x for x in landmark_files if "frame" not in x.name]
        
        for l_file in landmark_files:
            # Task name is derived from filename
            # e.g. T1Landmarks.json -> T1
            t_name = l_file.name.replace("Landmarks.json", "").replace("Landmark.json", "").replace("landmark.json", "")
            
            # Map task folder to label key
            task_mapping = {
                "T1": "Preparing presentation",
                "T2": "Presentation",
                "T3": "Iqtest",
                "T4": "Watching video",
                "T5": "Watching Video",
                "T6": "No Video"
            }
            mapped_task = task_mapping.get(t_name, None)
            if not mapped_task or mapped_task not in survey_map:
                lbl = 0  # Fallback/default label
            else:
                lbl = survey_map[mapped_task]
                
            try:
                d = read_json(l_file)
                frame_keys = sorted([k for k in d.keys() if k.isdigit()], key=int)
                
                # Sample at 3 fps (every 10th frame from 30 fps)
                sampled_keys = [k for idx, k in enumerate(frame_keys) if idx % 10 == 0]
                landmarks_list = [d[k] for k in sampled_keys]
                
                total_processed_frames += len(landmarks_list)
                
                features = process_frames_to_features(landmarks_list, format_type='dlib')
                flat_rec, seqs = extract_windows_and_sequences(features, sub, "empathicschool", t_name, lbl)
                
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
                    f_log.write(f"Subject: {sub}, Task: {t_name}, Frames: {len(landmarks_list)}, Windows: {len(flat_rec)}\n")
            except Exception as e:
                print(f"Error processing landmark file {l_file}: {e}")
                
    if flat_records_all:
        df_flat = pd.DataFrame(flat_records_all)
        df_flat.to_parquet(output_dir / "face_windows.parquet")
        
        np.save(output_dir / "face_sequences.npy", np.array(sequences_all, dtype=np.float32))
        pd.DataFrame(window_meta_all).to_parquet(output_dir / "face_sequences_index.parquet")
        
    print(f"EmpathicSchool Completed. Windows: {total_windows_produced}")

def main():
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "pipeline" / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
        
    # Set up paths
    stressid_raw = base_dir / config["datasets"]["stressid"]["raw_path"]
    empathic_raw = base_dir / config["datasets"]["empathicschool"]["raw_path"]
    
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    
    log_file = base_dir / "pipeline" / "logs" / "face_extraction.log"
    if log_file.exists():
        log_file.unlink()
        
    # 1. StressID Face Feature Extraction
    detector = MediaPipeDetector()
    run_stressid_extraction(detector, stressid_raw, sid_out, log_file)
    detector.detector.close()
    
    # 2. EmpathicSchool Face Feature Extraction (reads landmarks JSON)
    run_empathicschool_extraction(empathic_raw, es_out, log_file)
    
    # Self-verification check
    # Columns count verification: 170 feature columns + metadata columns
    # Metadata: subject_id, dataset_source, task_name, window_id, face_available, binary_stress (6 cols)
    # Total cols = 176
    
    issues = []
    for out_path, name in [(sid_out, "StressID"), (es_out, "EmpathicSchool")]:
        pq_file = out_path / "face_windows.parquet"
        if not pq_file.exists():
            issues.append(f"{name} face_windows.parquet missing")
        else:
            df = pd.read_parquet(pq_file)
            if len(df.columns) != 176:
                issues.append(f"{name} columns count mismatch: expected 176, got {len(df.columns)}")
                
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("Face extraction verification PASSED.")

if __name__ == "__main__":
    main()
