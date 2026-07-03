import os
import sys
import pandas as pd
import numpy as np
import cv2
import glob
from tqdm import tqdm
from model import MultimodalStressDetector

# --- Configuration ---
RAW_DATASET_DIR = r"f:\Multimodal_stress_Detection\datasets\StressID\StressID Dataset"
OUTPUT_DIR = r"f:\Multimodal_stress_Detection\backend\Feature Extraction\Features"
LABELS_PATH = os.path.join(RAW_DATASET_DIR, "labels.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(video_path, detector):
    """
    Extract facial features from video by averaging features across frames.
    Sampling: 1 frame every 30 frames (approx 1 per second) to save time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frame_features_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame (assuming 30fps, this is 1 frame/sec)
        if frame_count % 30 == 0:
            # Save temp frame for model to read (model.py reads from disk)
            temp_img_path = "temp_frame.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            # Extract features (ignoring smile detection for training features)
            feats, _ = detector.extract_facial_features(temp_img_path)
            
            if feats is not None:
                frame_features_list.append(feats)
            
            # Cleanup
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
        frame_count += 1
        
    cap.release()
    
    if not frame_features_list:
        return None
        
    # Average features across the video
    return np.mean(frame_features_list, axis=0)

def process_physio(physio_path, detector):
    """
    Extract physio features.
    Maps ECG -> EEG inputs (proxy)
    Maps EDA -> GSR inputs
    """
    try:
        # Read text file
        df = pd.read_csv(physio_path)
        
        # Check columns
        if 'ECG' not in df.columns or 'EDA' not in df.columns:
            return None
            
        ecg_data = df['ECG'].values
        eda_data = df['EDA'].values
        
        # Pass to model (ECG as Proxy for EEG input in the function signature)
        return detector.extract_physiological_features(eeg_data=ecg_data, gsr_data=eda_data)
        
    except Exception as e:
        print(f"Error processing physio {physio_path}: {e}")
        return None

def main():
    print("="*60)
    print("dataset Integration Script")
    print("Re-processing StressID Raw Dataset for ")
    print("="*60)
    
    # Initialize detector
    detector = MultimodalStressDetector()
    
    # Load Labels
    if not os.path.exists(LABELS_PATH):
        print(f"Error: Labels not found at {LABELS_PATH}")
        return

    labels_df = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(labels_df)} labels")
    
    # Lists to store data
    audio_data = []
    video_data = []
    physio_data = []
    valid_ids = []
    
    # Limit for testing. Set to None for full run.
    LIMIT = None 
    # print(f"NOTE: Processing only {LIMIT} samples for demonstration.")
    # print("To process the full dataset, edit this script and set LIMIT = None\n")
    
    count = 0
    
    # Iterate through potential subjects/tasks in labels
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        if LIMIT and count >= LIMIT:
            break
            
        subject_task = row['subject/task']
        subject = subject_task.split('_')[0]
        
        # Construct Paths
        audio_path = os.path.join(RAW_DATASET_DIR, "Audio", subject, f"{subject_task}.wav")
        video_path = os.path.join(RAW_DATASET_DIR, "Videos", subject, f"{subject_task}.mp4")
        physio_path = os.path.join(RAW_DATASET_DIR, "Physiological", subject, f"{subject_task}.txt")
        
        # Check if all files exist
        if not (os.path.exists(audio_path) and os.path.exists(video_path) and os.path.exists(physio_path)):
            continue
            
        print(f"\nProcessing {subject_task}...")
        
        # 1. Audio
        aud_feats = detector.extract_voice_features(audio_path)
        
        # 2. Video
        vid_feats = process_video(video_path, detector)
        
        # 3. Physio
        phy_feats = process_physio(physio_path, detector)
        
        if aud_feats is not None and vid_feats is not None and phy_feats is not None:
            audio_data.append(aud_feats)
            video_data.append(vid_feats)
            physio_data.append(phy_feats)
            valid_ids.append(subject_task)
            count += 1
            print(f"  > Success")
        else:
            print(f"  > Failed extraction")

    # Create DataFrames
    print("\nCreating DataFrames...")
    
    # Column names? We don't have them, just generic
    # Audio
    df_audio = pd.DataFrame(audio_data, index=valid_ids)
    df_audio.to_csv(os.path.join(OUTPUT_DIR, "integrated_audio.csv"), header=False)
    
    # Video
    df_video = pd.DataFrame(video_data, index=valid_ids)
    df_video.index.name = 'id'
    df_video.columns = [f"feat_{i}" for i in range(df_video.shape[1])] # Dummy headers
    df_video.to_csv(os.path.join(OUTPUT_DIR, "integrated_video.csv"))
    
    # Physio
    df_physio = pd.DataFrame(physio_data, index=valid_ids)
    df_physio.index.name = 'id'
    df_physio.columns = [f"feat_{i}" for i in range(df_physio.shape[1])]
    df_physio.to_csv(os.path.join(OUTPUT_DIR, "integrated_physio.csv"))
    
    print(f"\n[SUCCESS] Processed {len(valid_ids)} samples.")
    print(f"Files saved to {OUTPUT_DIR}")
    print(" - integrated_audio.csv")
    print(" - integrated_video.csv")
    print(" - integrated_physio.csv")

if __name__ == "__main__":
    main()
