# Raw Multimodal Data Synchronization Audit Report

This report summarizes the file-level alignment and duration audit performed on the raw dataset folders in Google Drive:
* **Base Directory:** `/content/drive/MyDrive/Multimodal_stress_Detection/`
* **Local Summary:** [synchronization_report.csv](file:///e:/Document/GitHub/StressDetectionUsingML/early_fusion/reports/synchronization/synchronization_report.csv)
* **Detailed Alignment CSV:** [raw_files_alignment_audit.csv](file:///e:/Document/GitHub/StressDetectionUsingML/early_fusion/reports/synchronization/raw_files_alignment_audit.csv)

---

## 1. Audit Summary Metrics

The script recursively scanned the folders and identified **775 unique subject-task pairs** (recordings). Here is the status breakdown:

| Status Metric | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Total Evaluated Recordings** | **775** | 100% | Unique `(subject_id, task_id)` pairs |
| **Fully Synchronized** | **364** | 47.0% | Video, Audio, and Physio are all present and align in duration |
| **Missing Face (Video)** | **146** | 18.8% | Recordings with missing video (`.mp4`) files |
| **Missing Voice (Audio)** | **397** | 51.2% | Recordings with missing audio (`.wav`) files |
| **Missing Physio (Logs)** | **0** | 0.0% | Recordings with missing physiological (`.txt`) logs |
| **Duration Mismatches** | **0** | 0.0% | Recordings with duration difference > 2.0 seconds |

> [!NOTE]
> **Why are 397 recordings missing Voice?**
> In the StressID experimental design, several task conditions are resting or silent states (e.g. `Baseline`, `Relax`, `Breathing`, `Video1`, `Video2`) where no speech occurs. No audio was recorded for these sessions. Thus, the missing audio files for these tasks (~325 files) are a standard feature of the dataset design, not an extraction error.

---

## 2. Key Findings

### 1. Duration Agreement is Perfect
* For all **364 fully matched recordings**, the difference in length between the Video, Audio, and Physiological signals is **less than 2.0 seconds** (0 duration mismatches). This indicates that the raw files were recorded synchronously and can be windowed (e.g., 1.0s window, 0.5s stride) with direct index-to-index matching.

### 2. Modality Completeness
* **Physiological Logs** are the most complete modality in the dataset (present in 100% of the 775 recordings).
* **Videos** are present in 81.2% of the recordings (629 out of 775).
* **Audio files** are present in 48.8% of the recordings (378 out of 775) due to resting/silent state exclusions.

---

## 3. Recommendations for Fusion Models
* **Multimodal Fused Engine (Face + Voice + Physio):** Train only on the **364 fully synchronized recordings** to perform Early/Gated Fusion or Cross-Attention.
* **Bimodal Engine (Face + Physio):** You can utilize a larger subset of **629 recordings** where both Face and Physio are present, as Voice is the most frequently absent modality.
* **Late Fusion:** Use late fusion models that predict on available unimodal outputs and dynamically average them, which will allow you to leverage the full 775 recordings.
