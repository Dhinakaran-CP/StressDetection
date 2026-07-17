import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import neurokit2 as nk
from pipeline.common.determinism import set_determinism
from pipeline.common.io_utils import read_csv_or_xls, write_json, read_json

# Set determinism first
set_determinism()

def process_stressid_physio(physio_file):
    try:
        df = pd.read_csv(physio_file)
    except Exception:
        return None
        
    if not {"ECG", "EDA", "RR"}.issubset(df.columns):
        return None
        
    ecg = df["ECG"].values
    eda = df["EDA"].values
    resp = df["RR"].values
    
    sr = 500.0
    duration_sec = len(df) / sr
    
    # 1. ECG Process
    try:
        ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=int(sr))
        hr = ecg_signals["ECG_Rate"].values
        # Running RMSSD over 30s window as HRV proxy
        r_peaks = ecg_signals["ECG_R_Peaks"].values
        hrv = np.zeros_like(hr)
        window_samples = int(30 * sr)
        for i in range(len(hr)):
            start_w = max(0, i - window_samples)
            peaks_in_w = np.where(r_peaks[start_w:i+1] == 1)[0]
            if len(peaks_in_w) > 2:
                rri = np.diff(peaks_in_w) / sr * 1000.0  # in ms
                hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))
            else:
                hrv[i] = np.nan
    except Exception:
        hr = np.ones(len(df)) * 75.0
        hrv = np.ones(len(df)) * 50.0

    # 2. EDA Process
    try:
        eda_signals, eda_info = nk.eda_process(eda, sampling_rate=int(sr))
        eda_clean = eda_signals["EDA_Clean"].values
        eda_tonic = eda_signals["EDA_Tonic"].values
        eda_phasic = eda_signals["EDA_Phasic"].values
        scr_peaks = eda_signals["SCR_Peaks"].values
        
        # Cumulative SCR peaks in a 30s sliding window
        scr_count = np.zeros_like(eda_clean)
        window_samples = int(30 * sr)
        for i in range(len(eda_clean)):
            start_w = max(0, i - window_samples)
            scr_count[i] = np.sum(scr_peaks[start_w:i+1])
    except Exception:
        eda_clean = eda
        eda_tonic = eda
        eda_phasic = np.zeros_like(eda)
        scr_count = np.zeros_like(eda)

    # 3. Respiration Process
    try:
        rsp_signals, rsp_info = nk.rsp_process(resp, sampling_rate=int(sr))
        resp_rate = rsp_signals["RSP_Rate"].values
        resp_amplitude = rsp_signals["RSP_Amplitude"].values
    except Exception:
        resp_rate = np.ones(len(df)) * 15.0
        resp_amplitude = np.ones(len(df)) * 1.0
        
    # Interpolate to 3 FPS
    ts_source = np.arange(len(df)) / sr
    ts_target = np.arange(0, duration_sec, 1/3.0)
    
    # Pack features
    feats_3fps = []
    for t in ts_target:
        idx = min(len(df) - 1, int(t * sr))
        feats_3fps.append({
            "ecg_hr": hr[idx],
            "ecg_hrv": hrv[idx] if not np.isnan(hrv[idx]) else 50.0,
            "eda_clean": eda_clean[idx],
            "eda_tonic": eda_tonic[idx],
            "eda_phasic": eda_phasic[idx],
            "eda_scr_count": scr_count[idx],
            "resp_rate": resp_rate[idx],
            "resp_amplitude": resp_amplitude[idx],
            "temp_mean": np.nan,
            "temp_std": np.nan,
            "acc_x": np.nan,
            "acc_y": np.nan,
            "acc_z": np.nan,
            "acc_mag": np.nan
        })
        
    return feats_3fps

def load_empatica_csv(file_path):
    if not file_path.exists():
        return None, None, None
    try:
        df = read_csv_or_xls(file_path)
        if len(df) <= 2:
            return None, None, None
        start_time = float(df.iloc[0, 0])
        fs = float(df.iloc[1, 0])
        values = df.iloc[2:].values.flatten()
        return start_time, fs, values
    except Exception:
        return None, None, None

def process_empathic_physio(eda_path, bvp_path, hr_path, temp_path, acc_path):
    # Load all files
    eda_start, eda_fs, eda_vals = load_empatica_csv(eda_path)
    bvp_start, bvp_fs, bvp_vals = load_empatica_csv(bvp_path)
    hr_start, hr_fs, hr_vals = load_empatica_csv(hr_path)
    temp_start, temp_fs, temp_vals = load_empatica_csv(temp_path)
    acc_start, acc_fs, acc_vals = load_empatica_csv(acc_path)
    
    if eda_vals is None:
        return None
        
    # Standardize to EDA start and duration
    start_time = eda_start
    duration_sec = len(eda_vals) / eda_fs
    
    # 1. Process EDA
    try:
        eda_signals, eda_info = nk.eda_process(eda_vals, sampling_rate=int(eda_fs))
        eda_clean = eda_signals["EDA_Clean"].values
        eda_tonic = eda_signals["EDA_Tonic"].values
        eda_phasic = eda_signals["EDA_Phasic"].values
        scr_peaks = eda_signals["SCR_Peaks"].values
        scr_count = np.zeros_like(eda_clean)
        window_samples = int(30 * eda_fs)
        for i in range(len(eda_clean)):
            start_w = max(0, i - window_samples)
            scr_count[i] = np.sum(scr_peaks[start_w:i+1])
    except Exception:
        eda_clean = eda_vals
        eda_tonic = eda_vals
        eda_phasic = np.zeros_like(eda_vals)
        scr_count = np.zeros_like(eda_vals)
        
    # 2. Process Heart Rate & HRV from BVP or HR file
    hr_interpolated = np.ones(len(eda_vals)) * 75.0
    hrv_interpolated = np.ones(len(eda_vals)) * 50.0
    
    if bvp_vals is not None and len(bvp_vals) > 64:
        try:
            bvp_signals, bvp_info = nk.ppg_process(bvp_vals, sampling_rate=int(bvp_fs))
            bvp_hr = bvp_signals["PPG_Rate"].values
            bvp_peaks = bvp_signals["PPG_Peaks"].values
            bvp_hrv = np.zeros_like(bvp_hr)
            window_samples = int(30 * bvp_fs)
            for i in range(len(bvp_hr)):
                start_w = max(0, i - window_samples)
                peaks_in_w = np.where(bvp_peaks[start_w:i+1] == 1)[0]
                if len(peaks_in_w) > 2:
                    rri = np.diff(peaks_in_w) / bvp_fs * 1000.0
                    bvp_hrv[i] = np.sqrt(np.mean(np.diff(rri) ** 2))
                else:
                    bvp_hrv[i] = np.nan
            # Interpolate to EDA timeline
            ts_bvp = bvp_start + np.arange(len(bvp_vals)) / bvp_fs
            ts_eda = eda_start + np.arange(len(eda_vals)) / eda_fs
            hr_interpolated = np.interp(ts_eda, ts_bvp, bvp_hr, left=75.0, right=75.0)
            hrv_interpolated = np.interp(ts_eda, ts_bvp, bvp_hrv, left=50.0, right=50.0)
        except Exception:
            pass
    elif hr_vals is not None:
        # Fallback to HR file directly
        ts_hr = hr_start + np.arange(len(hr_vals)) / hr_fs
        ts_eda = eda_start + np.arange(len(eda_vals)) / eda_fs
        hr_interpolated = np.interp(ts_eda, ts_hr, hr_vals, left=75.0, right=75.0)
        
    # 3. Process TEMP
    temp_interpolated = np.ones(len(eda_vals)) * 32.0
    if temp_vals is not None:
        ts_temp = temp_start + np.arange(len(temp_vals)) / temp_fs
        ts_eda = eda_start + np.arange(len(eda_vals)) / eda_fs
        temp_interpolated = np.interp(ts_eda, ts_temp, temp_vals, left=32.0, right=32.0)
        
    # 4. Process ACC
    acc_x_int = np.zeros(len(eda_vals))
    acc_y_int = np.zeros(len(eda_vals))
    acc_z_int = np.zeros(len(eda_vals))
    acc_mag_int = np.zeros(len(eda_vals))
    
    if acc_vals is not None and len(acc_vals) >= 3:
        # ACC usually has 3 axes interleaved: x, y, z
        n_samples = len(acc_vals) // 3
        acc_vals = acc_vals[:n_samples * 3]
        acc_reshaped = acc_vals.reshape(-1, 3)
        acc_x = acc_reshaped[:, 0]
        acc_y = acc_reshaped[:, 1]
        acc_z = acc_reshaped[:, 2]
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        ts_acc = acc_start + np.arange(len(acc_reshaped)) / acc_fs
        ts_eda = eda_start + np.arange(len(eda_vals)) / eda_fs
        acc_x_int = np.interp(ts_eda, ts_acc, acc_x, left=0.0, right=0.0)
        acc_y_int = np.interp(ts_eda, ts_acc, acc_y, left=0.0, right=0.0)
        acc_z_int = np.interp(ts_eda, ts_acc, acc_z, left=0.0, right=0.0)
        acc_mag_int = np.interp(ts_eda, ts_acc, acc_mag, left=0.0, right=0.0)
        
    # Interpolate all to 3 FPS
    ts_target = np.arange(0, duration_sec, 1/3.0)
    feats_3fps = []
    
    for t in ts_target:
        idx = min(len(eda_vals) - 1, int(t * eda_fs))
        
        # Temp stats (1/3 second window)
        # 1/3 second corresponds to eda_fs / 3.0 samples
        w_size = max(1, int(eda_fs / 3.0))
        start_w = max(0, idx - w_size)
        temp_window = temp_interpolated[start_w:idx+1]
        t_mean = np.mean(temp_window)
        t_std = np.std(temp_window) if len(temp_window) > 1 else 0.0
        
        feats_3fps.append({
            "ecg_hr": hr_interpolated[idx],
            "ecg_hrv": hrv_interpolated[idx] if not np.isnan(hrv_interpolated[idx]) else 50.0,
            "eda_clean": eda_clean[idx],
            "eda_tonic": eda_tonic[idx],
            "eda_phasic": eda_phasic[idx],
            "eda_scr_count": scr_count[idx],
            "resp_rate": np.nan,
            "resp_amplitude": np.nan,
            "temp_mean": t_mean,
            "temp_std": t_std,
            "acc_x": acc_x_int[idx],
            "acc_y": acc_y_int[idx],
            "acc_z": acc_z_int[idx],
            "acc_mag": acc_mag_int[idx]
        })
        
    return feats_3fps

def extract_windows_and_sequences(features_list, subject_id, dataset_source, task_name, binary_stress, window_size=30, stride=15):
    flat_records = []
    sequences_list = []
    
    n_frames = len(features_list)
    n_windows = int((n_frames - window_size) // stride) + 1
    
    fields = [
        "ecg_hr", "ecg_hrv", "eda_clean", "eda_tonic", "eda_phasic", "eda_scr_count",
        "resp_rate", "resp_amplitude", "temp_mean", "temp_std", "acc_x", "acc_y", "acc_z", "acc_mag"
    ]
    
    for w_idx in range(n_windows):
        start = w_idx * stride
        end = start + window_size
        window_feats = features_list[start:end]
        
        valid_feats = [f for f in window_feats if f is not None]
        
        # Check continuity: if at least 50% frames have valid non-NaN features for required signals
        # For StressID, check ecg_hr. For EmpathicSchool, check eda_clean.
        if dataset_source == "stressid":
            valid_subset = [f for f in valid_feats if not np.isnan(f["ecg_hr"])]
        else:
            valid_subset = [f for f in valid_feats if not np.isnan(f["eda_clean"])]
            
        physio_available = 1 if len(valid_subset) > (window_size / 2) else 0
        
        window_id = f"{subject_id}_{task_name}_W{w_idx}"
        
        flat_record = {
            "subject_id": subject_id,
            "dataset_source": dataset_source,
            "task_name": task_name,
            "window_id": window_id,
            "physio_available": physio_available,
            "binary_stress": binary_stress
        }
        
        sequence_matrix = np.zeros((window_size, len(fields)), dtype=np.float32)
        
        for i_f, f in enumerate(window_feats):
            if f is not None:
                for col_idx, field in enumerate(fields):
                    sequence_matrix[i_f, col_idx] = f[field]
            else:
                sequence_matrix[i_f, :] = np.nan
                
        if physio_available:
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

def run_stressid_extraction(raw_path, output_dir, log_file):
    print("Running StressID Physio extraction...")
    labels_csv_path = raw_path / "labels.csv"
    df_labels = pd.read_csv(labels_csv_path)
    label_map = {row['subject/task']: int(row['binary-stress']) for _, row in df_labels.iterrows()}
    
    subjects = sorted([x for x in os.listdir(raw_path / "Videos") if not x.startswith('.')])
    
    flat_records_all = []
    sequences_all = []
    window_meta_all = []
    
    total_processed = 0
    total_windows = 0
    
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write("--- StressID Physio Extraction ---\n")
        
    for sub in tqdm(subjects, desc="StressID Physio"):
        phys_dir = raw_path / "Physiological" / sub
        phys_files = list(phys_dir.glob("*.txt"))
        
        for phys_path in phys_files:
            task_name = phys_path.stem.replace(f"{sub}_", "")
            key = f"{sub}_{task_name}"
            if key not in label_map:
                continue
            lbl = label_map[key]
            
            # Read physiological file properties to get canonical frame count N
            # Slicing physiological data directly to match face frame count
            feats = process_stressid_physio(phys_path)
            if not feats:
                continue
                
            flat_rec, seqs = extract_windows_and_sequences(feats, sub, "stressid", task_name, lbl)
            
            flat_records_all.extend(flat_rec)
            for fr, seq in zip(flat_rec, seqs):
                global_idx = len(sequences_all)
                sequences_all.append(seq)
                window_meta_all.append({
                    "window_id": fr["window_id"],
                    "sequence_index": global_idx
                })
                total_windows += 1
                
            total_processed += 1
            with open(log_file, "a", encoding="utf-8") as f_log:
                f_log.write(f"Subject: {sub}, Task: {task_name}, Frames: {len(feats)}, Windows: {len(flat_rec)}\n")
                
    if flat_records_all:
        df_flat = pd.DataFrame(flat_records_all)
        df_flat.to_parquet(output_dir / "physio_windows.parquet")
        
        np.save(output_dir / "physio_sequences.npy", np.array(sequences_all, dtype=np.float32))
        pd.DataFrame(window_meta_all).to_parquet(output_dir / "physio_sequences_index.parquet")
        
    print(f"StressID Completed. Windows: {total_windows}")

def run_empathicschool_extraction(raw_path, output_dir, log_file):
    print("Running EmpathicSchool Physio extraction...")
    
    subjects = [f"S{i}" for i in range(1, 31)]
    
    flat_records_all = []
    sequences_all = []
    window_meta_all = []
    
    total_processed = 0
    total_windows = 0
    
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write("\n--- EmpathicSchool Physio Extraction ---\n")
        
    for sub in tqdm(subjects, desc="EmpathicSchool Physio"):
        sub_dir = raw_path / sub
        if not sub_dir.exists():
            continue
            
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
                
        # Find all files recursively using glob
        all_files = [Path(x) for x in glob.glob(str(sub_dir / "**/*"), recursive=True)]
        
        # Check schema Mode A vs Mode B
        eda_files = [x for x in all_files if x.name.endswith("EDA.csv")]
        
        # For S29 and S30 (Mode B):
        is_mode_b = len(eda_files) == 1 and not any(x.parent.name.startswith("T") for x in eda_files)
        
        if is_mode_b:
            # Mode B: Continuous E4 files
            t_dir = eda_files[0].parent
            eda_path = t_dir / "EDA.csv"
            bvp_path = t_dir / "BVP.csv"
            hr_path = t_dir / "HR.csv"
            temp_path = t_dir / "TEMP.csv"
            acc_path = t_dir / "ACC.csv"
            
            feats = process_empathic_physio(eda_path, bvp_path, hr_path, temp_path, acc_path)
            if feats:
                flat_rec, seqs = extract_windows_and_sequences(feats, sub, "empathicschool", "continuous", 0)
                
                flat_records_all.extend(flat_rec)
                for fr, seq in zip(flat_rec, seqs):
                    global_idx = len(sequences_all)
                    sequences_all.append(seq)
                    window_meta_all.append({
                        "window_id": fr["window_id"],
                        "sequence_index": global_idx
                    })
                    total_windows += 1
                total_processed += 1
                with open(log_file, "a", encoding="utf-8") as f_log:
                    f_log.write(f"Subject: {sub}, Task: continuous, Frames: {len(feats)}, Windows: {len(flat_rec)}\n")
        else:
            # Mode A: Task directories T1 to T9
            # Gather unique tasks from EDA filenames
            # e.g. T1EDA.csv -> T1
            t_names = sorted(list({x.parent.name for x in eda_files if x.parent.name.startswith("T")}))
            
            for t_name in t_names:
                t_dir = [x.parent for x in eda_files if x.parent.name == t_name][0]
                
                # Check file names in T*
                eda_path = t_dir / f"{t_name}EDA.csv"
                if not eda_path.exists():
                    eda_path = t_dir / "EDA.csv"
                    
                bvp_path = t_dir / f"{t_name}BVP.csv"
                if not bvp_path.exists():
                    bvp_path = t_dir / "BVP.csv"
                    
                hr_path = t_dir / f"{t_name}HR.csv"
                if not hr_path.exists():
                    hr_path = t_dir / f"{t_name}hr.csv"
                if not hr_path.exists():
                    hr_path = t_dir / "HR.csv"
                    
                temp_path = t_dir / f"{t_name}TEMP.csv"
                if not temp_path.exists():
                    temp_path = t_dir / "TEMP.csv"
                    
                acc_path = t_dir / f"{t_name}ACC.csv"
                if not acc_path.exists():
                    acc_path = t_dir / "ACC.csv"
                    
                task_mapping = {
                    "T1": "Preparing presentation",
                    "T2": "Presentation",
                    "T3": "Iqtest",
                    "T4": "Watching video",
                    "T5": "Watching Video",
                    "T6": "No Video"
                }
                mapped_task = task_mapping.get(t_name, None)
                lbl = survey_map.get(mapped_task, 0)
                
                feats = process_empathic_physio(eda_path, bvp_path, hr_path, temp_path, acc_path)
                if feats:
                    flat_rec, seqs = extract_windows_and_sequences(feats, sub, "empathicschool", t_name, lbl)
                    
                    flat_records_all.extend(flat_rec)
                    for fr, seq in zip(flat_rec, seqs):
                        global_idx = len(sequences_all)
                        sequences_all.append(seq)
                        window_meta_all.append({
                            "window_id": fr["window_id"],
                            "sequence_index": global_idx
                        })
                        total_windows += 1
                    total_processed += 1
                    with open(log_file, "a", encoding="utf-8") as f_log:
                        f_log.write(f"Subject: {sub}, Task: {t_name}, Frames: {len(feats)}, Windows: {len(flat_rec)}\n")
                        
    if flat_records_all:
        df_flat = pd.DataFrame(flat_records_all)
        df_flat.to_parquet(output_dir / "physio_windows.parquet")
        
        np.save(output_dir / "physio_sequences.npy", np.array(sequences_all, dtype=np.float32))
        pd.DataFrame(window_meta_all).to_parquet(output_dir / "physio_sequences_index.parquet")
        
    print(f"EmpathicSchool Completed. Windows: {total_windows}")

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    config_path = base_dir / "pipeline" / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
        
    stressid_raw = base_dir / config["datasets"]["stressid"]["raw_path"]
    empathic_raw = base_dir / config["datasets"]["empathicschool"]["raw_path"]
    
    sid_out = base_dir / "pipeline" / "data" / "stressid"
    es_out = base_dir / "pipeline" / "data" / "empathicschool"
    
    log_file = base_dir / "pipeline" / "logs" / "physio_extraction.log"
    if log_file.exists():
        log_file.unlink()
        
    # 1. StressID Physio Extraction
    run_stressid_extraction(stressid_raw, sid_out, log_file)
    
    # 2. EmpathicSchool Physio Extraction
    run_empathicschool_extraction(empathic_raw, es_out, log_file)
    
    # Self-verification check
    # Metadata columns: 6
    # Feature columns: 14 * 5 = 70
    # Total columns: 76
    
    issues = []
    for out_path, name in [(sid_out, "StressID"), (es_out, "EmpathicSchool")]:
        pq_file = out_path / "physio_windows.parquet"
        if not pq_file.exists():
            issues.append(f"{name} physio_windows.parquet missing")
        else:
            df = pd.read_parquet(pq_file)
            if len(df.columns) != 76:
                issues.append(f"{name} columns count mismatch: expected 76, got {len(df.columns)}")
                
    if issues:
        print("Self-verification FAILED:", issues)
    else:
        print("Physio extraction verification PASSED.")

if __name__ == "__main__":
    main()
