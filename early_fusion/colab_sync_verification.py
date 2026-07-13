"""
Google Colab Raw Multimodal Data Synchronization Verification Script.
This script performs a synchronization audit directly on the raw multimodal files
(Videos, Audios, and Physiological files) stored in Google Drive.

It maps files by subject and task, measures their recording durations,
and determines if they are aligned and ready for temporal window-level extraction.

Ensure you install these dependencies in Colab before running:
!pip install opencv-python pandas numpy
"""

import os
import sys
import glob
import wave
import pandas as pd
import numpy as np

# ==========================================
# 1. Configuration & Google Drive Mount
# ==========================================
IN_COLAB = 'google.colab' in sys.modules
GOOGLE_DRIVE_MOUNT_POINT = '/content/drive'

# Base drive folders to search
COLAB_BASE_PATH = '/content/drive/MyDrive/Multimodal_stress_Detection/'
LOCAL_BASE_PATH = './'

if IN_COLAB:
    print("Detected Google Colab environment. Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount(GOOGLE_DRIVE_MOUNT_POINT)
        base_path = COLAB_BASE_PATH
    except Exception as e:
        print(f"Warning: Failed to mount Google Drive ({e}). Falling back to local path.")
        base_path = LOCAL_BASE_PATH
else:
    print("Detected local environment. Using local workspace path...")
    base_path = LOCAL_BASE_PATH

print(f"Scanning base directory: {os.path.abspath(base_path)}\n")

# ==========================================
# 2. Helper Functions for Duration Measurement
# ==========================================
def get_video_duration(path):
    """Get video duration using OpenCV"""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps if fps > 0 else 0.0
        cap.release()
        return float(duration)
    except Exception:
        return 0.0

def get_audio_duration(path):
    """Get wave audio duration using wave module (fast, standard library)"""
    try:
        with wave.open(path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return float(frames) / float(rate)
    except Exception:
        # Fallback to librosa if available
        try:
            import librosa
            return float(librosa.get_duration(path=path))
        except Exception:
            return 0.0

def get_physio_duration(path, fs=500):
    """Get physiological duration based on line count in txt/csv file"""
    try:
        # Fast line counting
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            num_lines = sum(1 for _ in f)
        # Deduct 1 for header
        return float(max(0, num_lines - 1)) / float(fs)
    except Exception:
        return 0.0

# ==========================================
# 3. Recursively Scan for Raw Files
# ==========================================
print("Searching for raw files recursively in the base path...")
raw_videos = {}
raw_audios = {}
raw_physios = {}

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')
physio_extensions = ('.txt', '.csv')

# Exclude system folders
exclude_dirs = {'.git', '.pytest_cache', 'venv', '__pycache__', 'facesData'}

for root, dirs, files in os.walk(base_path):
    # Prune directory search
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        # Ignore macOS shadow metadata files and dotfiles
        if file.startswith('._') or file.startswith('.') or file.lower() == 'desktop.ini':
            continue
            
        file_lower = file.lower()
        fpath = os.path.join(root, file)
        root_lower = root.lower()
        
        # Parse subject and task from filename (e.g. Subject01_Stroop.wav -> subject01, stroop)
        name, ext = os.path.splitext(file)
        parts = name.split('_')
        if len(parts) >= 2:
            sub = parts[0].strip().lower()
            task = parts[1].strip().lower()
            key = (sub, task)
        else:
            continue
            
        ext_lower = ext.lower()
        
        # Explicit modality classification based on subfolder hierarchy
        is_video = ("videos" in root_lower or "video" in root_lower or "face" in root_lower) and ext_lower in video_extensions
        is_audio = ("audio" in root_lower or "voice" in root_lower or "speech" in root_lower) and ext_lower in audio_extensions
        is_physio = ("physiological" in root_lower or "physio" in root_lower) and ext_lower in physio_extensions
        
        # Fallback to extension and filename markers if folder structure does not match (helps in test environments)
        if not (is_video or is_audio or is_physio):
            if ext_lower in video_extensions:
                is_video = True
            elif ext_lower in audio_extensions:
                is_audio = True
            elif ext_lower in physio_extensions and ('physio' in file_lower or 'rr' in file_lower or 'eda' in file_lower):
                is_physio = True
                
        if is_video:
            raw_videos[key] = fpath
        elif is_audio:
            raw_audios[key] = fpath
        elif is_physio:
            raw_physios[key] = fpath

print(f"Found:")
print(f" - Raw Video files: {len(raw_videos)}")
print(f" - Raw Audio files: {len(raw_audios)}")
print(f" - Raw Physio files: {len(raw_physios)}")

# ==========================================
# 4. Alignment Audit
# ==========================================
all_keys = set(raw_videos.keys()) | set(raw_audios.keys()) | set(raw_physios.keys())
print(f"\nTotal unique subject-task pairs detected: {len(all_keys)}")

audit_rows = []
fully_synchronized_count = 0
missing_face_count = 0
missing_voice_count = 0
missing_physio_count = 0
duration_mismatch_count = 0

duration_tolerance = 2.0  # Duration difference allowed in seconds

for key in all_keys:
    sub, task = key
    video_file = raw_videos.get(key)
    audio_file = raw_audios.get(key)
    physio_file = raw_physios.get(key)
    
    # Measure durations
    v_dur = get_video_duration(video_file) if video_file else 0.0
    a_dur = get_audio_duration(audio_file) if audio_file else 0.0
    p_dur = get_physio_duration(physio_file) if physio_file else 0.0
    
    has_video = video_file is not None
    has_audio = audio_file is not None
    has_physio = physio_file is not None
    
    status = "Aligned"
    details = []
    
    if not has_video:
        status = "Missing Face"
        missing_face_count += 1
        details.append("No video file")
    if not has_audio:
        status = "Missing Voice"
        missing_voice_count += 1
        details.append("No audio file")
    if not has_physio:
        status = "Missing Physio"
        missing_physio_count += 1
        details.append("No physiological file")
        
    if has_video and has_audio and has_physio:
        # Check duration consistency
        durs = [v_dur, a_dur, p_dur]
        max_diff = max(durs) - min(durs)
        if max_diff > duration_tolerance:
            status = "Duration Mismatch"
            duration_mismatch_count += 1
            details.append(f"Mismatched lengths (Video: {v_dur:.1f}s, Audio: {a_dur:.1f}s, Physio: {p_dur:.1f}s)")
        else:
            fully_synchronized_count += 1
            details.append(f"Aligned ({v_dur:.1f}s)")
            
    audit_rows.append({
        "subject_id": sub,
        "task_id": task,
        "video_exists": has_video,
        "audio_exists": has_audio,
        "physio_exists": has_physio,
        "video_duration": v_dur,
        "audio_duration": a_dur,
        "physio_duration": p_dur,
        "status": status,
        "details": "; ".join(details)
    })

# ==========================================
# 5. Output Summary and Status Decision
# ==========================================
print("\n" + "="*50)
print("RAW DATA SYNCHRONIZATION AUDIT REPORT")
print("="*50)
print(f"Total Subject-Task Pairs Evaluated:  {len(all_keys)}")
print(f"Fully Synchronized (All Modalities): {fully_synchronized_count}")
print(f"Missing Face (Video):                {missing_face_count}")
print(f"Missing Voice (Audio):               {missing_voice_count}")
print(f"Missing Physio (Logs):               {missing_physio_count}")
print(f"Duration Mismatches (> {duration_tolerance}s):   {duration_mismatch_count}")
print("="*50)

# Determine final label
if len(all_keys) == 0:
    final_status = "Not synchronized"
    conclusion = "No raw multimodal files could be parsed or matched in the selected directory."
elif fully_synchronized_count == len(all_keys) and duration_mismatch_count == 0:
    final_status = "Fully synchronized"
    conclusion = "All detected subject-task recordings possess matched video, audio, and physio files with identical lengths."
elif fully_synchronized_count > 0:
    final_status = "Partially synchronized"
    conclusion = f"A subset of {fully_synchronized_count} recordings is fully matched and aligned, but other recordings have missing modalities or length discrepancies."
else:
    final_status = "Not synchronized"
    conclusion = "No recording pairs have complete, matching files across all three modalities."

print(f"\nFinal Synchronization Status: **{final_status}**")
print(f"Conclusion: {conclusion}\n")

# ==========================================
# 6. Save Report
# ==========================================
report_data = [
    {"metric": "total_recordings", "value": len(all_keys), "details": "Total subject-task pairs evaluated"},
    {"metric": "fully_synchronized", "value": fully_synchronized_count, "details": "Recordings with all 3 modalities present and matching lengths"},
    {"metric": "missing_face", "value": missing_face_count, "details": "Recordings missing video data"},
    {"metric": "missing_voice", "value": missing_voice_count, "details": "Recordings missing audio data"},
    {"metric": "missing_physio", "value": missing_physio_count, "details": "Recordings missing physiological logs"},
    {"metric": "duration_mismatches", "value": duration_mismatch_count, "details": "Recordings with length difference > 2.0s"},
    {"metric": "final_status", "value": final_status, "details": "Final classification of dataset synchronization"}
]

report_df = pd.DataFrame(report_data)
output_csv_filename = "synchronization_report.csv"
report_df.to_csv(output_csv_filename, index=False)
print(f"Saved synchronization summary metrics to: {os.path.abspath(output_csv_filename)}")

# Save detailed file audit
detailed_df = pd.DataFrame(audit_rows)
detailed_csv_filename = "raw_files_alignment_audit.csv"
detailed_df.to_csv(detailed_csv_filename, index=False)
print(f"Saved detailed file-level audit to: {os.path.abspath(detailed_csv_filename)}")
