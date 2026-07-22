import os
import sys
import argparse
import zipfile
import shutil
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process the UBFC-Phys dataset.")
    parser.add_argument("--s3-uri", type=str, default=None, help="AWS S3 URI for the UBFC-Phys dataset (from IEEE DataPort).")
    parser.add_argument("--aws-access-key", type=str, default=None, help="AWS Access Key ID.")
    parser.add_argument("--aws-secret-key", type=str, default=None, help="AWS Secret Access Key.")
    parser.add_argument("--local-dir", type=str, default="data/ubfc_phys", help="Local directory containing/where to download ZIP or CSV files.")
    parser.add_argument("--out-dir", type=str, default="data/processed/ubfc_phys", help="Output directory for processed CSV files.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep the raw ZIP files/folders after processing (default is to delete them).")
    parser.add_argument("--target-fs", type=float, default=4.0, help="Target sampling rate (Hz) for aligned output CSV (default: 4.0 Hz).")
    return parser.parse_args()

def download_from_s3(s3_uri, aws_access_key, aws_secret_key, local_dir):
    """
    Downloads files from AWS S3 using provided credentials, skipping large video files.
    """
    print(f"[AWS S3] Attempting to download from {s3_uri} to {local_dir}...")
    try:
        import boto3
        from urllib.parse import urlparse
    except ImportError:
        print("[ERROR] boto3 is required to download from S3. Please install it using pip.")
        sys.exit(1)

    # Parse bucket and key prefix
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")

    # Initialize client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    os.makedirs(local_dir, exist_ok=True)

    # List objects
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        file_count = 0
        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                
                # Skip directory markers
                if key.endswith("/"):
                    continue
                    
                # EXCLUDE video formats to save disk and network usage
                if key.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv')):
                    continue
                
                # Preserve directory structure relative to the prefix
                if key.startswith(prefix):
                    rel_key = key[len(prefix):].lstrip("/")
                else:
                    rel_key = key
                    
                local_path = os.path.join(local_dir, rel_key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                print(f"[AWS S3] Downloading {key} -> {local_path}...")
                s3.download_file(bucket_name, key, local_path)
                file_count += 1
                
        print(f"[AWS S3] Download completed. Downloaded {file_count} physiological/metadata files.")
    except Exception as e:
        print(f"[ERROR] Failed to download from S3: {e}")
        print("Please check your S3 URI, Access Keys, and Internet connection.")
        sys.exit(1)

def load_empatica_csv(file_path):
    """
    Loads Empatica E4 CSV format file.
    Row 1: start time (epoch)
    Row 2: sample rate (Hz)
    Row 3+: signal values
    """
    try:
        df = pd.read_csv(file_path, header=None)
        if len(df) <= 2:
            return None, None, None
        start_time = float(df.iloc[0, 0])
        fs = float(df.iloc[1, 0])
        values = df.iloc[2:].values.flatten()
        return start_time, fs, values
    except Exception as e:
        print(f"  [WARN] Failed to load {file_path.name}: {e}")
        return None, None, None

def process_signals(eda_path, bvp_path, target_fs=4.0):
    """
    Processes EDA and BVP signals using NeuroKit2 and aligns them to target_fs.
    """
    # Load raw data
    eda_start, eda_fs, eda_vals = load_empatica_csv(eda_path)
    bvp_start, bvp_fs, bvp_vals = load_empatica_csv(bvp_path)

    if eda_vals is None or bvp_vals is None:
        return None

    # Sync and get time duration
    # Empatica E4 EDA is usually 4Hz, BVP is 64Hz
    eda_duration = len(eda_vals) / eda_fs
    bvp_duration = len(bvp_vals) / bvp_fs
    duration_sec = min(eda_duration, bvp_duration)

    # 1. Process EDA
    try:
        eda_signals, _ = nk.eda_process(eda_vals, sampling_rate=int(eda_fs))
        eda_clean = eda_signals["EDA_Clean"].values
        eda_tonic = eda_signals["EDA_Tonic"].values
        eda_phasic = eda_signals["EDA_Phasic"].values
        scr_peaks = eda_signals["SCR_Peaks"].values
        
        # Cumulative SCR peaks in a 30s sliding window
        scr_count = np.zeros_like(eda_clean)
        window_samples = int(30 * eda_fs)
        for i in range(len(eda_clean)):
            start_w = max(0, i - window_samples)
            scr_count[i] = np.sum(scr_peaks[start_w:i+1])
    except Exception as e:
        print(f"  [WARN] NeuroKit2 EDA processing failed: {e}. Using raw/fallback.")
        eda_clean = eda_vals
        eda_tonic = eda_vals
        eda_phasic = np.zeros_like(eda_vals)
        scr_count = np.zeros_like(eda_vals)

    # 2. Process BVP (PPG)
    try:
        bvp_signals, _ = nk.ppg_process(bvp_vals, sampling_rate=int(bvp_fs))
        bvp_hr = bvp_signals["PPG_Rate"].values
        bvp_peaks = bvp_signals["PPG_Peaks"].values
        
        # Cumulative HRV proxy (RMSSD of peak intervals in 30s window)
        bvp_hrv = np.zeros_like(bvp_hr)
        window_samples = int(30 * bvp_fs)
        for i in range(len(bvp_hr)):
            start_w = max(0, i - window_samples)
            peaks_in_w = np.where(bvp_peaks[start_w:i+1] == 1)[0]
            if len(peaks_in_w) > 2:
                rri = np.diff(peaks_in_w) / bvp_fs * 1000.0  # ms
                bvp_hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))
            else:
                bvp_hrv[i] = np.nan
    except Exception as e:
        print(f"  [WARN] NeuroKit2 BVP processing failed: {e}. Using flat fallback.")
        bvp_hr = np.ones(len(bvp_vals)) * 75.0
        bvp_hrv = np.ones(len(bvp_vals)) * 50.0

    # Handle NaNs in HRV
    nan_mask = np.isnan(bvp_hrv)
    if np.all(nan_mask):
        bvp_hrv[:] = 50.0
    elif np.any(nan_mask):
        mean_hrv = np.nanmean(bvp_hrv)
        bvp_hrv[nan_mask] = mean_hrv if not np.isnan(mean_hrv) else 50.0

    # 3. Align signals to the target sampling rate
    total_samples = int(duration_sec * target_fs)
    ts_target = np.arange(total_samples) / target_fs

    # Timestamps for source signals
    ts_eda = np.arange(len(eda_vals)) / eda_fs
    ts_bvp = np.arange(len(bvp_vals)) / bvp_fs

    # Interpolate signals to target timeline
    eda_clean_int = np.interp(ts_target, ts_eda, eda_clean)
    eda_tonic_int = np.interp(ts_target, ts_eda, eda_tonic)
    eda_phasic_int = np.interp(ts_target, ts_eda, eda_phasic)
    scr_count_int = np.interp(ts_target, ts_eda, scr_count)
    hr_int = np.interp(ts_target, ts_bvp, bvp_hr)
    hrv_int = np.interp(ts_target, ts_bvp, bvp_hrv)

    # Build DataFrame
    df_aligned = pd.DataFrame({
        "time_sec": ts_target,
        "eda_clean": eda_clean_int,
        "eda_tonic": eda_tonic_int,
        "eda_phasic": eda_phasic_int,
        "eda_scr_count": scr_count_int,
        "hr": hr_int,
        "hrv": hrv_int
    })
    
    return df_aligned

def scan_and_group_files(local_dir):
    """
    Scans the local directory recursively for BVP and EDA file pairs.
    Returns: dict of subject_id -> list of task_dict
    """
    local_path = Path(local_dir)
    # Recursively find BVP files
    bvp_files = list(local_path.rglob("*bvp*.csv")) + list(local_path.rglob("*BVP*.csv"))
    bvp_files = sorted(list(set(bvp_files)))
    
    subjects = {}
    
    for bvp_f in bvp_files:
        name_lower = bvp_f.name.lower()
        
        # Parse subject ID (e.g. s10 from bvp_s10_T1.csv or s10_zip in parent folders)
        sub_match = re.search(r's\d+', bvp_f.name)
        if sub_match:
            sub_id = sub_match.group(0)
        else:
            parts = bvp_f.parts
            sub_id = None
            for p in reversed(parts):
                sub_match = re.search(r's\d+', p)
                if sub_match:
                    sub_id = sub_match.group(0)
                    break
            if not sub_id:
                sub_id = "unknown_subject"
                
        # Parse task name (e.g., T1, T2, T3)
        task_match = re.search(r'T\d+', bvp_f.name)
        if task_match:
            task_key = task_match.group(0)
        else:
            if "rest" in name_lower or "t1" in name_lower:
                task_key = "T1"
            elif "speech" in name_lower or "t2" in name_lower:
                task_key = "T2"
            elif "arithmetic" in name_lower or "t3" in name_lower:
                task_key = "T3"
            else:
                task_key = "T_Unknown"
                
        # Map task to label and full task name
        if task_key == "T1" or "rest" in name_lower:
            task_name = "T1_Rest"
            label = 0
        elif task_key == "T2" or "speech" in name_lower:
            task_name = "T2_Speech"
            label = 1
        elif task_key == "T3" or "arithmetic" in name_lower:
            task_name = "T3_Arithmetic"
            label = 1
        else:
            task_name = f"{task_key}_Task"
            label = 1
            
        # Find corresponding EDA file in the same folder
        eda_f = None
        parent = bvp_f.parent
        eda_name_cand = bvp_f.name.replace('bvp', 'eda').replace('BVP', 'EDA')
        eda_cand_f = parent / eda_name_cand
        if eda_cand_f.exists():
            eda_f = eda_cand_f
        else:
            # Fallback scan parent folder
            eda_files_in_folder = list(parent.glob("*eda*.csv")) + list(parent.glob("*EDA*.csv"))
            for ec in eda_files_in_folder:
                if task_key.lower() in ec.name.lower():
                    eda_f = ec
                    break
                    
        if eda_f is None:
            print(f"  [WARN] Could not find matching EDA file for BVP file: {bvp_f.name}")
            continue
            
        if sub_id not in subjects:
            subjects[sub_id] = []
            
        subjects[sub_id].append({
            "task_name": task_name,
            "bvp_path": bvp_f,
            "eda_path": eda_f,
            "label": label
        })
        
    return subjects

def main():
    args = parse_args()

    # Create directories
    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Download if S3 parameters are specified
    if args.s3_uri:
        if not args.aws_access_key or not args.aws_secret_key:
            print("[ERROR] AWS Access Key and Secret Key are required when specifying an S3 URI.")
            sys.exit(1)
        download_from_s3(args.s3_uri, args.aws_access_key, args.aws_secret_key, args.local_dir)

    # 2. Scan and group files recursively
    print("[INFO] Scanning local directory for BVP and EDA file pairs...")
    subjects = scan_and_group_files(args.local_dir)

    if not subjects:
        print(f"[INFO] No UBFC-Phys CSV files found in {args.local_dir}.")
        print("Please place the UBFC-Phys data in that directory or provide S3 credentials to download it.")
        sys.exit(0)

    print(f"[INFO] Found {len(subjects)} subjects to process.")
    
    all_subjects_dfs = []
    
    # 3. Process each subject
    for subject_id, tasks in tqdm(subjects.items(), desc="Processing Subjects"):
        subject_dfs = []
        for task in tasks:
            task_name = task["task_name"]
            eda_path = task["eda_path"]
            bvp_path = task["bvp_path"]
            lbl = task["label"]
            
            print(f"  Processing {subject_id} - {task_name}...")
            df_task = process_signals(eda_path, bvp_path, target_fs=args.target_fs)
            if df_task is not None:
                df_task["subject_id"] = subject_id
                df_task["task"] = task_name
                df_task["label"] = lbl
                subject_dfs.append(df_task)
                
        if subject_dfs:
            df_subj = pd.concat(subject_dfs, ignore_index=True)
            # Save subject-level CSV
            subj_out_path = os.path.join(args.out_dir, f"{subject_id}_processed.csv")
            df_subj.to_csv(subj_out_path, index=False)
            print(f"  Saved processed data to {subj_out_path} ({len(df_subj)} rows)")
            
            all_subjects_dfs.append(df_subj)
            
            # Clean up raw files/folders if requested (default is to delete)
            if not args.keep_zips:
                try:
                    dirs_to_delete = set()
                    for task in tasks:
                        dirs_to_delete.add(task["bvp_path"].parent)
                        # Also delete parent zip folder if recursively nested
                        if task["bvp_path"].parent.parent.name.endswith("_zip"):
                            dirs_to_delete.add(task["bvp_path"].parent.parent)
                    
                    for d in dirs_to_delete:
                        if d.exists() and d.is_dir():
                            shutil.rmtree(d)
                            print(f"  Deleted source directory: {d.name}")
                except Exception as e:
                    print(f"  [WARN] Failed to delete source files for {subject_id}: {e}")
        else:
            print(f"  [ERROR] Subject processing failed or skipped for {subject_id}")

    # 4. Consolidate all subjects
    if all_subjects_dfs:
        df_all = pd.concat(all_subjects_dfs, ignore_index=True)
        consolidated_path = os.path.join(args.out_dir, "ubfc_phys_all_subjects.csv")
        df_all.to_csv(consolidated_path, index=False)
        print(f"\n[SUCCESS] Completed processing. Merged CSV stored at: {consolidated_path}")
        print(f"Total Rows: {len(df_all)}, Total Subjects: {df_all['subject_id'].nunique()}")
    else:
        print("[WARN] No subjects were successfully processed.")

if __name__ == "__main__":
    main()
