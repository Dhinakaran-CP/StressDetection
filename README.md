# Multimodal Stress Intelligence Platform

A real-time stress detection system that uses three physiological modalities — **facial expressions**, **voice acoustics**, and **physiological signals** — to detect and quantify stress levels using machine learning.

---

## 📖 Overview

The **Multimodal Stress Intelligence Platform** aims to accurately identify human stress levels by fusing multiple streams of biological telemetry. By observing users through their webcam and microphone—as well as optionally analyzing uploaded EEG and GSR data—the application acts as an intelligent health monitoring dashboard. It not only predicts the likelihood of stress but also explicitly tells users *why* they are stressed using Explainable AI (XAI).

## ✨ Key Features

- **Real-Time Telemetry Dashboard:** Stream live stress analysis directly from your webcam and microphone.
- **Multimodal Uploads:** Support for batch analysis of images, audio files, and physiological CSV data (EEG/GSR).
- **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to break down the exact biometric features driving your stress score (e.g., "Left Brow Tension" or "Vocal Jitter").
- **Personal Baseline Calibration:** Calibrate the system to your natural resting state before analyzing stress to reduce false positives.
- **Interactive UI:** Smooth React-based frontend with a toggleable Earthy/Cyber theme and responsive visual plots.

---

## 🛠️ Tech Stack

### Frontend
- **Framework:** React.js
- **Visualization:** Recharts (for live radar and bar charts)
- **Styling:** Custom CSS (glassmorphism UI, dynamic themes)
- **Web Workers:** Background thread processing for face posture requests.

### Backend
- **Framework:** Python, Flask, Flask-CORS
- **Concurrency:** Eventlet & Socket.IO (for asynchronous workers and SSE/WebSocket streaming)

### Machine Learning & Data Processing
- **Core ML:** Scikit-Learn (Ensemble methods, SVM, Random Forest, Gradient Boosting, Calibrated Classifiers)
- **Computer Vision:** MediaPipe Tasks API (Face Landmarking), OpenCV (Fallback Haar Cascades)
- **Audio Processing:** Librosa, Custom Fast Autocorrelation
- **Explainability:** SHAP (TreeExplainer)
- **Data Manipulation:** NumPy, Pandas

---

## 🧠 Methodology & Architecture

The system uses a **three-expert fusion** approach:

| Expert | Input | Model | Latency |
|--------|-------|-------|---------|
| **Facial** | Webcam frames → MediaPipe 3D Landmarks | Gradient Boosting | ~8 ms |
| **Voice** | Microphone chunks → 12 Acoustic Biomarkers | Gradient Boosting | ~15 ms |
| **Physiological** | Uploaded EEG/GSR Signals | Soft-Voting Ensemble (GB + RF) | ~5 ms |

### 1. Facial Expression Expert
- Extracts 18 high-level geometric features based on facial action units (Eye Aspect Ratio, Brow Tension, Lip Compression, Jaw Displacement, etc.).
- Gradient Boosting Classifier trained on augmented face landmark geometries, balanced via SMOTE (Synthetic Minority Over-sampling Technique).

### 2. Voice Acoustics Expert
- Extracts 12 specific acoustic biomarkers including Pitch, Jitter (frequency instability), Shimmer (amplitude instability), and Harmonics-to-Noise Ratio (HNR).
- Gradient Boosting Classifier utilizing custom autocorrelation with parabolic peak interpolation, achieving a heavily optimized execution time of **<15ms** (down from 4.6s using standard libraries).

### 3. Physiological Signal Expert
- Analyzes CSV datasets of EEG (Brainwave Alpha/Beta power) and GSR (Skin Conductance rate).
- Soft-Voting Ensemble (Gradient Boosting + Random Forest) wrapped in a `CalibratedClassifierCV` for highly accurate probabilistic stress mapping.

### ⚙️ Fusion Engine
Results from the distinct models are aggregated via a **weighted confidence engine**. It applies temporal smoothing and a 15-second decay buffer for voice (ensuring conversation-style scoring continuity while the user takes breaths or pauses). The output is streamed in real-time to the frontend via Server-Sent Events (SSE).

---

## 📂 Project Structure

```text
StressIntelligencePlatform/
│
├── run.bat                       # One-click launcher (starts backend + frontend)
├── README.md
├── TEST_GUIDE.md                 # Comprehensive testing guide
│
├── backend/                      # Flask API server & ML Inference
│   ├── app.py                    # Main application entry point (API routes, SSE)
│   ├── model.py                  # Feature extraction + inference logic
│   ├── realtime_core.py          # Real-time session management & SSE streams
│   ├── voice_worker.py           # High-speed vocal feature extraction
│   ├── calibration.py            # Per-user baseline calibration engine
│   ├── score_buffer.py           # Rolling score buffer with smoothing
│   ├── requirements.txt          # Python dependencies
│   │
│   ├── expert_models/            # Production lightweight models (~1-2 MB each)
│   ├── tests/                    # API, health, and streaming test scripts
│   └── uploads/                  # Temporary file uploads (auto-cleaned)
│
├── frontend/                     # React application
│   ├── public/
│   │   └── facePostWorker.js     # Web Worker: offloads face POST requests
│   └── src/
│       ├── App.js                # Root Component
│       ├── pages/Dashboard.js    # Main UI Dashboard
│       └── components/           # UI Components (RealtimeMonitor, AnalysisPanel)
│
└── reports/                      # Performance benchmarks & analysis reports
```

---

## 🚀 Getting Started

### Quick Start (Windows)
You can run the entire stack (both frontend and backend) simultaneously using the provided batch script:
```bash
run.bat
```
- **Frontend URL:** http://localhost:3000
- **Backend API URL:** http://localhost:5000

### Manual Setup

**1. Backend**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**2. Frontend**
```bash
cd frontend
npm install
npm start
```