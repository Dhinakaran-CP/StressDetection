import os
import glob
import pandas as pd
import numpy as np
import neurokit2 as nk
import warnings

warnings.filterwarnings("ignore")

PHYSIO_DIR = "Physiological"
LABELS_PATH = "backend/training/Dataset/labels.csv"
OUTPUT_PATH = "dataset_certified/physio_certified.csv"
FS = 500  # 500 Hz

def extract_physio_features(raw_file_path, subject_id, task_id, label, window_length=1.0, hop_length=0.5, context_length=10.0):
    """
    Extracts features from a raw physio file using a trailing context window,
    but anchors the window_start/window_end to match the Face/Voice modalities.
    """
    try:
        # Some files might have different headers, but the sample showed "ECG,EDA,RR"
        df_raw = pd.read_csv(raw_file_path)
        if "ECG" not in df_raw.columns:
            print(f"Skipping {raw_file_path}: missing expected columns.")
            return []
            
        # Clean signals over the whole recording (much faster than per-window)
        ecg_signals, info = nk.ecg_process(df_raw['ECG'], sampling_rate=FS)
        eda_signals, _ = nk.eda_process(df_raw['EDA'], sampling_rate=FS)
        rsp_signals, _ = nk.rsp_process(df_raw['RR'], sampling_rate=FS)
        
        r_peaks = info['ECG_R_Peaks']
        
        total_samples = len(df_raw)
        total_seconds = total_samples / FS
        
        rows = []
        # We step by hop_length. The window 'end' time is what aligns with the Face modality.
        window_index = 0
        current_start = 0.0
        
        while current_start + window_length <= total_seconds:
            current_end = current_start + window_length
            
            # For physio, we look back 'context_length' seconds from current_end to get a stable reading.
            context_start = max(0.0, current_end - context_length)
            
            start_idx = int(context_start * FS)
            end_idx = int(current_end * FS)
            
            # Extract slices
            ecg_slice = ecg_signals.iloc[start_idx:end_idx]
            eda_slice = eda_signals.iloc[start_idx:end_idx]
            rsp_slice = rsp_signals.iloc[start_idx:end_idx]
            
            # R-peaks in this context window
            peaks_in_window = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
            rr_intervals = np.diff(peaks_in_window) / FS * 1000.0  # in ms
            
            # Features
            ecg_rate_mean = ecg_slice['ECG_Rate'].mean()
            
            if len(rr_intervals) > 1:
                ecg_hrv_rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
                ecg_hrv_sdnn = np.std(rr_intervals)
            else:
                ecg_hrv_rmssd = 0.0
                ecg_hrv_sdnn = 0.0
                
            eda_scl_mean = eda_slice['EDA_Tonic'].mean()
            resp_rate_mean = rsp_slice['RSP_Rate'].mean()
            
            # Only append if valid values exist
            if not np.isnan(ecg_rate_mean):
                row = {
                    "subject_id": subject_id,
                    "task_id": task_id,
                    "video_id": f"{subject_id}_{task_id}",
                    "window_index": window_index,
                    "window_start": current_start,
                    "window_end": current_end,
                    "label": label,
                    "ecg_rate_mean": ecg_rate_mean,
                    "ecg_hrv_rmssd": ecg_hrv_rmssd,
                    "ecg_hrv_sdnn": ecg_hrv_sdnn,
                    "eda_scl_mean": eda_scl_mean,
                    "resp_rate_mean": resp_rate_mean
                }
                rows.append(row)
                
            current_start += hop_length
            window_index += 1
            
        return rows
    except Exception as e:
        print(f"Error processing {raw_file_path}: {e}")
        return []

def main():
    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_PATH)
    # create mapping dict: "2ea4_Baseline" -> 0
    label_map = {}
    for _, row in labels_df.iterrows():
        label_map[row['subject/task']] = row['binary-stress']
        
    all_rows = []
    
    # Iterate over subjects
    subject_dirs = [d for d in os.listdir(PHYSIO_DIR) if os.path.isdir(os.path.join(PHYSIO_DIR, d))]
    
    print(f"Found {len(subject_dirs)} subjects.")
    
    for subj in subject_dirs:
        subj_path = os.path.join(PHYSIO_DIR, subj)
        files = glob.glob(os.path.join(subj_path, "*.txt"))
        
        for file in files:
            filename = os.path.basename(file).replace(".txt", "")
            # Example filename: "2ea4_Baseline"
            if filename in label_map:
                label = label_map[filename]
                
                parts = filename.split("_")
                subject_id = parts[0]
                task_id = parts[1] if len(parts) > 1 else "Unknown"
                
                print(f"Extracting {filename} (Label: {label})...")
                rows = extract_physio_features(file, subject_id, task_id, label)
                all_rows.extend(rows)
            else:
                print(f"No label found for {filename}, skipping.")

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        # Ensure correct column order
        cols = ["subject_id", "task_id", "video_id", "window_index", "window_start", "window_end", "label",
                "ecg_rate_mean", "ecg_hrv_rmssd", "ecg_hrv_sdnn", "eda_scl_mean", "resp_rate_mean"]
        df_out = df_out[cols]
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully saved {len(df_out)} rows to {OUTPUT_PATH}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
