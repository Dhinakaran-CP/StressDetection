# Multimodal Stress Detection: Agent Onboarding Guide

Welcome! If you are an AI agent or a new developer onboarding to this project, this document provides a comprehensive, fact-based technical breakdown of the system architecture, the machine learning methodology, and the current critical challenges facing the project regarding data evaluation.

---

## 1. Project Overview
This project is a **Real-Time Multimodal Stress Intelligence Platform**. It detects and quantifies human stress levels by fusing three distinct biological telemetry streams:
1. **Facial Expressions** (via Webcam)
2. **Voice Acoustics** (via Microphone)
3. **Physiological Signals** (EEG/GSR datasets)

The goal is to provide a real-time health monitoring dashboard that not only predicts stress but uses Explainable AI (SHAP) to tell the user exactly *why* they are stressed (e.g., "High Vocal Jitter" or "Left Brow Tension").

---

## 2. System Architecture

The application is built on a decoupled Client-Server architecture designed for high-speed, low-latency streaming.

### Frontend (React)
- **Framework**: React.js with a custom CSS glassmorphism UI.
- **Data Visualization**: Uses `Recharts` to render live radar and bar charts for stress biomarkers.
- **Web Workers**: Uses a background Web Worker (`public/facePostWorker.js`) to offload the heavy POST requests of sending base64 webcam frames to the backend without freezing the UI thread.
- **Event Streaming**: Consumes a Server-Sent Events (SSE) stream from the backend to update the UI instantly.

### Backend (Python + Flask)
- **Framework**: Flask API with `Flask-CORS`.
- **Concurrency**: Uses `Eventlet` and `Socket.IO` to manage asynchronous workers and SSE streaming.
- **Core Files**:
  - `app.py`: The main entry point, API routes, and SSE generator.
  - `realtime_core.py`: Manages the real-time session, accumulates predictions, and yields SSE data.
  - `model.py`: Contains the actual feature extraction (MediaPipe/Librosa) and inference logic for the ML models.
  - `score_buffer.py`: A rolling temporal buffer that applies smoothing to predictions (e.g., applying a 15-second decay buffer for voice).
  - `voice_worker.py`: A dedicated, high-speed vocal feature extraction pipeline.

---

## 3. Machine Learning Methodology

The system uses a **Late-Fusion Ensemble Methodology**. Instead of concatenating raw features into a single massive neural network, we train three distinct "Expert Models" and fuse their probabilistic outputs.

### Modality 1: Facial Expert
- **Extraction**: Uses MediaPipe Tasks API to extract 3D facial landmarks. Calculates 18 high-level geometric Action Units (e.g., Eye Aspect Ratio, Brow Tension, Lip Compression).
- **Model**: Lightweight Gradient Boosting Classifier (`face_expert_lightweight.pkl`).
- **Training Strategy**: Trained on extracted geometries, balanced using SMOTE.

### Modality 2: Voice Expert
- **Extraction**: Uses Librosa to extract 12 vocal biomarkers (Pitch mean/std, Jitter, Shimmer, HNR, Spectral Flux, Pause Ratio, etc.).
- **Model**: Gradient Boosting Classifier (`voice_expert_lightweight.pkl`). 

### Modality 3: Physiological Expert
- **Extraction**: Processes 51 features from EEG (Alpha/Beta power) and GSR (Skin Conductance).
- **Model**: A Calibrated Soft-Voting Ensemble combining Gradient Boosting and Random Forest (`physio_expert.pkl`).

### The Fused Engine
Predictions from the three experts are aggregated via a weighted confidence engine (Face: ~37%, Voice: ~47%, Physio: ~34%). The engine applies temporal smoothing so the real-time UI does not aggressively flicker between states.

---

## 4. The Current Problem: Data Extraction & Evaluation Limitations

We recently discovered and fixed a major **Data Leakage** issue in the offline evaluation scripts (`strict_fused_evaluation.py` and `evaluate_fused_engine_bootstrapped.py`). The scripts were randomly splitting the entire dataset (80/20) and training the base experts, but they failed to hold out the 20 specific synchronized samples used to test the Fused Engine. As a result, the models had already "seen" the test data during training.

While we successfully patched the scripts to cleanly hold out the evaluation samples (revealing a true unseen Fused Accuracy of 45%), **we hit a hard wall when trying to implement advanced validation methodologies.**

### The Core Issue: Missing Metadata in Extracted CSVs
The training script (`backend/training/colab_training.py`) was run in Google Colab to extract features from the raw StressID videos/audios. However, it was flawed in two ways:
1. **Video-Level Averaging (Loss of Temporal Data):** Instead of saving frame-by-frame time-series data, it averaged all frames across an entire video into a single row (`avg_indicators = list(np.mean(frame_indicators, axis=0))`). 
2. **Missing Subject IDs:** It did not save the Subject ID (e.g., `Subject01`) or the Task Name into the output CSVs (`face_indicators_stressid.csv` and `voice_indicators_stressid.csv`).

### Why This Blocks Us:
Because of these two flaws, we currently **cannot** perform the following standard methodologies on the offline Face and Voice data:
- **Leave-One-Subject-Out (LOSO) Cross-Validation**: We cannot group the data by Subject to test model generalization to unseen humans, because we don't know which row belongs to which subject.
- **Temporal Window Evaluation**: We cannot simulate the real-time system's smoothing buffer offline, because there are no consecutive frames saved in the CSVs (only a single average per video).

*(Note: We **can** perform LOSO on the Physio dataset (`integrated_physio.csv`), as its index gracefully preserved the `Subject_Task` strings).*

### The Path Forward
To make the project novel and rigorously tested, **the extraction pipeline must be rerun.**
A developer or agent will need to modify `colab_training.py`, mount the Google Drive containing the raw `.mp4` and `.wav` files, and re-extract the datasets ensuring that:
1. Every row contains a `subject_id` column.
2. Every row represents a single short time-window (e.g., 1 second) rather than an entire video, allowing for temporal buffering simulations.
