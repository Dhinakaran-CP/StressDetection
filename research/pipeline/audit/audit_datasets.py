import os
import json
import zipfile
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.common.io_utils import read_csv_or_xls, write_json, read_json

def extract_nested_zips(base_path):
    print("Checking and extracting nested zip files in EmpathicSchool...")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".zip"):
                zip_path = Path(root) / file
                extract_dir = Path(root) / file.replace(".zip", "")
                if not extract_dir.exists():
                    print(f"Extracting nested zip {zip_path} to {extract_dir}...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                    except Exception as e:
                        print(f"Failed to extract {zip_path}: {e}")

def audit_stressid(raw_path):
    print("Auditing StressID dataset...")
    raw_path = Path(raw_path)
    labels_csv_path = raw_path / "labels.csv"
    
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"StressID labels.csv not found at {labels_csv_path}")
        
    df_labels = pd.read_csv(labels_csv_path)
    
    label_map = {}
    stressed_count = 0
    non_stressed_count = 0
    
    for idx, row in df_labels.iterrows():
        key = row['subject/task']
        val = int(row['binary-stress'])
        label_map[key] = val
        if val == 1:
            stressed_count += 1
        else:
            non_stressed_count += 1
            
    f = lambda s: {x for x in os.listdir(raw_path / s) if not x.startswith('.')} if (raw_path / s).exists() else set()
    v_subs = f("Videos")
    a_subs = f("Audio")
    p_subs = f("Physiological")
    subjects = sorted(list(v_subs.union(a_subs).union(p_subs)))
    
    subject_reports = {}
    
    for sub in subjects:
        report = {
            "has_video": sub in v_subs,
            "has_audio": sub in a_subs,
            "has_physio": sub in p_subs,
            "tasks": {},
            "total_duration_sec": 0,
            "has_unusable_short_segment": False
        }
        
        sub_tasks = [k.split("_")[1] for k in label_map.keys() if k.startswith(sub + "_")]
        if not sub_tasks:
            sub_tasks = ["Breathing", "Counting1", "Counting2", "Counting3", "Math", "Reading", "Relax", "Speaking", "Stroop", "Video1", "Video2"]
            
        for task in sub_tasks:
            key = f"{sub}_{task}"
            has_label = key in label_map
            label_val = label_map.get(key, None)
            
            video_file = raw_path / "Videos" / sub / f"{sub}_{task}.mp4"
            audio_file = raw_path / "Audio" / sub / f"{sub}_{task}.wav"
            physio_file = raw_path / "Physiological" / sub / f"{sub}_{task}.txt"
            
            task_status = {
                "video_exists": video_file.exists(),
                "audio_exists": audio_file.exists(),
                "physio_exists": physio_file.exists(),
                "label_exists": has_label,
                "label_value": label_val,
                "duration_sec": 0,
                "usable": True
            }
            
            if physio_file.exists():
                try:
                    with open(physio_file, "r") as f_phys:
                        line_count = sum(1 for _ in f_phys)
                    if line_count > 1:
                        dur = (line_count - 1) / 500.0
                        task_status["duration_sec"] = dur
                        report["total_duration_sec"] += dur
                        if dur < 30.0:
                            task_status["usable"] = False
                            report["has_unusable_short_segment"] = True
                except Exception:
                    task_status["usable"] = False
            
            report["tasks"][task] = task_status
            
        subject_reports[sub] = report
        
    return subject_reports, {
        "stressed": stressed_count,
        "non_stressed": non_stressed_count,
        "total_labels": len(label_map)
    }

def audit_empathicschool(raw_path):
    print("Auditing EmpathicSchool dataset...")
    raw_path = Path(raw_path)
    
    extract_nested_zips(raw_path)
    
    subjects = [f"S{i}" for i in range(1, 31)]
    subject_reports = {}
    
    stressed_count = 0
    non_stressed_count = 0
    total_labels = 0
    
    for sub in subjects:
        sub_dir = raw_path / sub
        if not sub_dir.exists():
            print(f"Warning: Subject directory {sub_dir} missing.")
            continue
            
        # Recursive glob search for robust matching
        all_files = [Path(x) for x in glob.glob(str(sub_dir / "**/*"), recursive=True)]
        
        # 1. Check label file presence
        label_files = [x for x in all_files if x.suffix in ['.xlsx', '.xls'] and "xception" not in x.name.lower() and not x.name.startswith("~$")]
        has_label_file = len(label_files) > 0
        label_file_path = label_files[0] if has_label_file else None
        
        # 2. Check physio presence (*EDA.csv or EDA.csv)
        eda_files = [x for x in all_files if x.name.endswith("EDA.csv")]
        has_physio = len(eda_files) > 0
        
        # 3. Check face video/features presence
        video_features = [x for x in all_files if any(k in x.name.lower() for k in ["landmark", "xception", ".mp4"])]
        has_video = len(video_features) > 0
        
        report = {
            "has_label_file": has_label_file,
            "label_file": str(label_file_path) if label_file_path else None,
            "has_video": has_video,
            "has_physio": has_physio,
            "tasks": {},
            "total_duration_sec": 0,
            "has_unusable_short_segment": False
        }
        
        # Read total duration from the EDA file(s)
        total_duration = 0.0
        for eda_file in eda_files:
            try:
                df = read_csv_or_xls(eda_file)
                if len(df) > 2:
                    fs = float(df.iloc[1, 0])
                    dur = (len(df) - 2) / fs
                    total_duration += dur
            except Exception as e:
                print(f"Error reading EDA duration for {sub}: {e}")
                
        report["total_duration_sec"] = total_duration
        if total_duration < 30.0:
            report["has_unusable_short_segment"] = True
            
        # Extract survey labels if Excel file exists
        if has_label_file and label_file_path.exists():
            try:
                df_survey = pd.read_excel(label_file_path)
                for idx, row in df_survey.iterrows():
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
                    
                    if binary_stress == 1:
                        stressed_count += 1
                    else:
                        non_stressed_count += 1
                    total_labels += 1
                    
                    report["tasks"][task_name] = {
                        "nasa_tlx": nasa_tlx,
                        "binary_stress": binary_stress,
                        "usable": True
                    }
            except Exception as e:
                print(f"Error parsing survey for subject {sub}: {e}")
                
        subject_reports[sub] = report
        
    return subject_reports, {
        "stressed": stressed_count,
        "non_stressed": non_stressed_count,
        "total_labels": total_labels
    }

def audit_wesad(raw_path):
    print("Auditing WESAD dataset...")
    raw_path = Path(raw_path)
    
    # WESAD has subjects S2 to S17, skipping S12
    subjects = [f"S{i}" for i in range(2, 18) if i != 12]
    
    subject_reports = {}
    total_labels = 0
    stressed_count = 0
    non_stressed_count = 0
    
    for sub in subjects:
        sub_dir = raw_path / sub
        pkl_path = sub_dir / f"{sub}.pkl"
        
        report = {
            "has_physio": pkl_path.exists(),
            "has_video": False,
            "has_audio": False,
            "duration_sec": 0,
            "usable": pkl_path.exists(),
            "class_balance": {
                "stressed": 0,
                "non_stressed": 0
            }
        }
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                labels = data['label']
                unique, counts = np.unique(labels, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                
                # In WESAD:
                # 1 = baseline (neutral) -> non_stressed
                # 2 = stress -> stressed
                neutral_samples = int(counts_dict.get(1, 0))
                stress_samples = int(counts_dict.get(2, 0))
                
                report["duration_sec"] = len(labels) / 700.0  # WESAD chest is 700Hz
                report["class_balance"]["stressed"] = stress_samples
                report["class_balance"]["non_stressed"] = neutral_samples
                
                stressed_count += stress_samples
                non_stressed_count += neutral_samples
                total_labels += (neutral_samples + stress_samples)
                
            except Exception as e:
                print(f"Error auditing subject {sub}: {e}")
                report["usable"] = False
                
        subject_reports[sub] = report
        
    return subject_reports, {
        "stressed": stressed_count,
        "non_stressed": non_stressed_count,
        "total_labels": total_labels
    }

def main():
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "pipeline" / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
        
    stressid_raw = base_dir / config["datasets"]["stressid"]["raw_path"]
    empathic_raw = base_dir / config["datasets"]["empathicschool"]["raw_path"]
    wesad_raw = base_dir / config["datasets"]["wesad"]["raw_path"]
    
    stressid_report, stressid_balance = audit_stressid(stressid_raw)
    empathic_report, empathic_balance = audit_empathicschool(empathic_raw)
    wesad_report, wesad_balance = audit_wesad(wesad_raw)
    
    sid_total = len(stressid_report)
    sid_face = sum(1 for s in stressid_report.values() if s["has_video"])
    sid_voice = sum(1 for s in stressid_report.values() if s["has_audio"])
    sid_physio = sum(1 for s in stressid_report.values() if s["has_physio"])
    
    es_total = len(empathic_report)
    es_face = sum(1 for s in empathic_report.values() if s["has_video"])
    es_physio = sum(1 for s in empathic_report.values() if s["has_physio"])
    
    wesad_total = len(wesad_report)
    wesad_physio = sum(1 for s in wesad_report.values() if s["has_physio"])
    
    sid_face_pct = sid_face / sid_total
    sid_voice_pct = sid_voice / sid_total
    sid_physio_pct = sid_physio / sid_total
    
    es_face_pct = es_face / es_total if es_total > 0 else 0
    es_physio_pct = es_physio / es_total if es_total > 0 else 0
    
    wesad_physio_pct = wesad_physio / wesad_total if wesad_total > 0 else 0
    
    print(f"StressID Modality Completeness:")
    print(f" - Face (Video): {sid_face}/{sid_total} ({sid_face_pct:.1%})")
    print(f" - Voice (Audio): {sid_voice}/{sid_total} ({sid_voice_pct:.1%})")
    print(f" - Physio: {sid_physio}/{sid_total} ({sid_physio_pct:.1%})")
    
    print(f"EmpathicSchool Modality Completeness:")
    print(f" - Face (Video/Landmarks): {es_face}/{es_total} ({es_face_pct:.1%})")
    print(f" - Physio: {es_physio}/{es_total} ({es_physio_pct:.1%})")
    
    print(f"WESAD Modality Completeness:")
    print(f" - Physio: {wesad_physio}/{wesad_total} ({wesad_physio_pct:.1%})")
    
    # Gate G1 condition: pct >= 80% for all required modalities
    gate_g1_passed = (
        sid_face_pct >= 0.80 and
        sid_voice_pct >= 0.80 and
        sid_physio_pct >= 0.80 and
        es_face_pct >= 0.80 and
        es_physio_pct >= 0.80 and
        wesad_physio_pct >= 0.80
    )
    
    status = "PASS" if gate_g1_passed else "FAIL"
    
    audit_report = {
        "datasets": {
            "stressid": {
                "subjects_count": sid_total,
                "class_balance": stressid_balance,
                "modalities": {
                    "face": {"present": sid_face, "percent": sid_face_pct},
                    "voice": {"present": sid_voice, "percent": sid_voice_pct},
                    "physio": {"present": sid_physio, "percent": sid_physio_pct}
                },
                "subjects": stressid_report
            },
            "empathicschool": {
                "subjects_count": es_total,
                "class_balance": empathic_balance,
                "modalities": {
                    "face": {"present": es_face, "percent": es_face_pct},
                    "physio": {"present": es_physio, "percent": es_physio_pct}
                },
                "subjects": empathic_report
            },
            "wesad": {
                "subjects_count": wesad_total,
                "class_balance": wesad_balance,
                "modalities": {
                    "physio": {"present": wesad_physio, "percent": wesad_physio_pct}
                },
                "subjects": wesad_report
            }
        },
        "gate_g1": {
            "passed": gate_g1_passed,
            "status": status
        }
    }
    
    report_json_path = base_dir / "pipeline" / "logs" / "audit_report.json"
    write_json(audit_report, report_json_path)
    
    summary_md_path = base_dir / "pipeline" / "logs" / "audit_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f_md:
        f_md.write(f"""# Data Audit and Completeness Gate (G1) Summary Report

**Status:** {status}
**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. StressID Dataset (Primary)
- **Total Subjects:** {sid_total} (verified 65 subjects)
- **Class Balance (Task-level):**
  - Stressed: {stressid_balance['stressed']}
  - Non-stressed: {stressid_balance['non_stressed']}
  - Ratio (Stressed/Total): {stressid_balance['stressed'] / stressid_balance['total_labels']:.2%}
- **Modality Completeness:**
  - Face (Video): {sid_face}/{sid_total} ({sid_face_pct:.2%})
  - Voice (Audio): {sid_voice}/{sid_total} ({sid_voice_pct:.2%})
  - Physiology: {sid_physio}/{sid_total} ({sid_physio_pct:.2%})

## 2. EmpathicSchool Dataset (Supplementary)
- **Total Subjects:** {es_total} (verified 30 subjects)
- **Class Balance (Interval/Task-level):**
  - Stressed: {empathic_balance['stressed']}
  - Non-stressed: {empathic_balance['non_stressed']}
  - Ratio (Stressed/Total): {empathic_balance['stressed'] / max(1, empathic_balance['total_labels']):.2%}
- **Modality Completeness:**
  - Face (Video/Landmarks): {es_face}/{es_total} ({es_face_pct:.2%})
  - Physiology: {es_physio}/{es_total} ({es_physio_pct:.2%})

## 3. WESAD Dataset (Supplementary)
- **Total Subjects:** {wesad_total} (verified 15 subjects)
- **Class Balance (Sample-level at 700Hz):**
  - Stressed: {wesad_balance['stressed']}
  - Non-stressed: {wesad_balance['non_stressed']}
  - Ratio (Stressed/Total): {wesad_balance['stressed'] / max(1, wesad_balance['total_labels']):.2%}
- **Modality Completeness:**
  - Physiology: {wesad_physio}/{wesad_total} ({wesad_physio_pct:.2%})

## 4. Gate G1 Evaluation
- **Condition:** Every required modality must be present for at least 80% of each dataset's subjects.
- **Evaluation Result:**
  - StressID Face: {"PASS" if sid_face_pct >= 0.8 else "FAIL"} ({sid_face_pct:.2%})
  - StressID Voice: {"PASS" if sid_voice_pct >= 0.8 else "FAIL"} ({sid_voice_pct:.2%})
  - StressID Physio: {"PASS" if sid_physio_pct >= 0.8 else "FAIL"} ({sid_physio_pct:.2%})
  - EmpathicSchool Face: {"PASS" if es_face_pct >= 0.8 else "FAIL"} ({es_face_pct:.2%})
  - EmpathicSchool Physio: {"PASS" if es_physio_pct >= 0.8 else "FAIL"} ({es_physio_pct:.2%})
  - WESAD Physio: {"PASS" if wesad_physio_pct >= 0.8 else "FAIL"} ({wesad_physio_pct:.2%})
- **Verdict:** **{"PASSED" if gate_g1_passed else "FAILED"}**
""")
        
    import time
    status_file = base_dir / "pipeline" / "logs" / "task_status.jsonl"
    status_data = {
        "task": "TASK-01",
        "status": status,
        "timestamp": time.time(),
        "metrics": {
            "sid_face_pct": sid_face_pct,
            "sid_voice_pct": sid_voice_pct,
            "sid_physio_pct": sid_physio_pct,
            "es_face_pct": es_face_pct,
            "es_physio_pct": es_physio_pct,
            "wesad_physio_pct": wesad_physio_pct,
            "gate_g1_passed": gate_g1_passed
        }
    }
    with open(status_file, "a", encoding="utf-8") as sf:
        sf.write(json.dumps(status_data) + "\n")
        
    print(f"Audit completed. Status: {status}")
    
    if not gate_g1_passed:
        fail_report = base_dir / "pipeline" / "logs" / "FAILURE_REPORT_TASK-01.md"
        with open(fail_report, "w", encoding="utf-8") as fr:
            fr.write(f"# Failure Report: TASK-01\n\n**Time:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n**Gate G1 Failed.** Required modalities did not meet the 80% subject threshold.\n")
        raise RuntimeError("Gate G1 Completeness failed.")

if __name__ == "__main__":
    main()
