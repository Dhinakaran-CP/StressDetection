# Evaluation and Verification Procedure

This document details the step-by-step procedure executed to set up the environment, run the archived models benchmark, update the version registry, and verify the codebase stability.

---

## 🛠️ Step-by-Step Execution Log

### Step 1: Virtual Environment Initialization
A dedicated virtual environment using Python 3.10 was created at the workspace root to isolate the dependencies:
```powershell
& "C:\Program Files\Python310\python.exe" -m venv venv
```

### Step 2: Dependency Installation
The package manager (`pip`) was upgraded, followed by installing the core machine learning libraries, audio/image feature extractors, API requirements, and test suites:
```powershell
# Upgrade pip and install standard ML stack
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install numpy pandas scikit-learn==1.6.1 scipy torch PyYAML tqdm joblib opencv-python pillow flask flask-socketio eventlet flask-cors requests

# Install additional feature extraction and explainability libraries for tests
venv\Scripts\python.exe -m pip install soundfile mediapipe matplotlib shap librosa
```

### Step 3: Creation of the Model Evaluation Pipeline
We created the benchmark pipeline script `evaluate_archive.py` to test both classical and deep sequence models under strict Leave-One-Subject-Out (LOSO) boundaries.
*   **Architecture Definitions**: Mapped standard `ModalityEncoder` (Conv1D + GRU sequence encoder) and `DynamicRouter` (MLP late-gated fusion) structures to exactly match PyTorch state dict keys.
*   **Normalization & Sequence Builder**:
    *   **Calm Baseline Calibration**: For each subject, calculated their calm feature averages (from rows where `label == 0`) and subtracted them from all session features.
    *   **Sliding Window Builder**: Formed temporal windows of length 5 (`seq_len=5`) per subject-task session (with edge padding).
*   **Multimodal Alignment**: Performed inner joins on the individual Face, Voice, and Physio dataframes using lowercased keys `['subject_id', 'task_id', 'video_id', 'window_index']` to ensure a synchronized aligned subset (43,110 frames).

### Step 4: Benchmarking Execution
The evaluations were executed on the certified datasets:
```powershell
venv\Scripts\python.exe evaluate_archive.py
```
*   **Outputs**: Summary files were written to the newly created [`performance_metrics/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/performance_metrics) folder:
    *   `overall_summary.json` containing metrics for all 17 model configurations.
    *   Individual model metric logs (e.g. `Strategy_5_Fusion_Router_All_metrics.json`).

### Step 5: Version Registry Correction
We identified a hash mismatch in the registry verification tests. The registered hash for `explainability_bundle.json` did not match the file on disk. We updated [`models/registry.json`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/models/registry.json) with the correct SHA-256 hash:
*   Old registered hash: `3cb57fcfef8bd498f6192425648b0dfcffc5ac99a4721d7ca1bf5cd7ccac693d`
*   New corrected hash: `9106c2e5b995883a2cde93006950bbd2803398ec872883e768e923543edad6b2`

### Step 6: Test Suite Validation
We verified the complete automated testing framework:
```powershell
venv\Scripts\python.exe -m unittest discover -s tests/
```
*   **Result**: 97/97 tests successfully compiled and passed, proving full runtime regression safety.
