# 🧠 Multimodal Stress Intelligence Platform

A real-time, multimodal stress detection system that fuses **facial expressions**, **voice acoustics**, and **physiological signals** using deep sequence learning — giving each sensor dynamic weight based on availability and signal quality.

---

## ✨ Key Features

- **Real-Time Streaming Dashboard** — Live stress analysis from webcam + microphone via WebSocket
- **3-Way Flex-Modality Fusion** — Works with any combination of Face / Voice / Physio sensors. Missing sensors are gracefully masked; the Dynamic Router re-normalizes weights automatically
- **Deep Sequence Learning** — 1D-CNN + GRU encoders process temporal windows of 5 frames per modality
- **Personal Baseline Calibration** — Calibrates to your natural resting state before analysis to cancel out identity bias
- **Explainable AI (XAI)** — SHAP-based explanations show *which* biometric features drove the stress score (e.g., "Brow Tension +0.18", "Vocal Jitter +0.12")
- **Stress Support Chatbot** — In-app assistant powered by Gemini 2.5 Flash for stress relief guidance
- **Modality Uploads** — Batch analysis via image/audio/physiological CSV uploads

---

## 🏆 Model Performance (Strict LOSO, 65 Subjects)

All models are validated with **Leave-One-Subject-Out (LOSO) 5-Fold GroupKFold** — no subject appears in both train and test sets. This prevents identity leakage.

| Modality | Architecture | LOSO Accuracy | Notes |
|---|---|---|---|
| Face | PyTorch 1D-CNN + GRU | 55.10% ± 4.58% | 18 facial AU + gaze features |
| Voice | PyTorch 1D-CNN + GRU | **61.46% ± 3.14%** | Best single-modality |
| Physio | PyTorch 1D-CNN + GRU | 58.95% ± 4.48% | EDA, HRV, EEG, BVP |
| **3-Way Fusion** | MLP Flex-Router | **58.26% ± 3.03%** | Dynamic weights, Modality Dropout |

> **Identity Leakage Gap**: Classical RF models leaked 18–26% between random-split and LOSO accuracy. The deep CNN-GRU pipeline reduces this to **7.62%** — the model learns stress patterns, not who the person is.

> Full benchmarks, ablation results, and training evidence: see [`model_archive/`](model_archive/README.md)

---

## 🏗️ Architecture

```
Webcam ──► Face CNN-GRU Encoder ──► P(calm), P(stress) ──┐
                                                           │
Microphone ► Voice CNN-GRU Encoder ► P(calm), P(stress) ──► Flex-Modality Dynamic Router MLP
                                                           │   (masks absent sensors, re-normalizes)
EEG/GSR ──► Physio CNN-GRU Encoder ► P(calm), P(stress) ──┘
                                                           │
                                                      Fused Stress Score
                                                           │
                                               Flask/SocketIO → React Dashboard
```

### Dynamic Router Gating

The router receives a **9-dimensional input**:

```
[P_face_calm, P_face_stress, P_voice_calm, P_voice_stress,
 P_physio_calm, P_physio_stress, mask_face, mask_voice, mask_physio]
```

Active weights are re-normalized: `ŵ_m = w_m · mask_m / Σ(w_k · mask_k)`.  
Trained with **Modality Dropout** so it handles any sensor subset at runtime.

---

## 📁 Project Structure

```
StressDetectionUsingML/
│
├── run.bat                         # One-click launcher (backend + frontend)
├── README.md
│
├── backend/                        # Flask API server & ML inference engine
│   ├── app.py                      # Main entry point (API routes, SocketIO)
│   ├── model.py                    # Feature extraction + classical inference
│   ├── realtime_core.py            # Real-time session management
│   ├── voice_worker.py             # High-speed vocal feature extraction
│   ├── calibration.py              # Per-user baseline calibration engine
│   ├── score_buffer.py             # Rolling score buffer with smoothing
│   ├── face_landmarker.task        # MediaPipe face landmark model
│   ├── requirements.txt            # Python dependencies
│   ├── core/                       # Feature runtime lock, dataset certifier
│   ├── runtime/                    # runtime_engine.py — deep model inference
│   ├── explainability/             # SHAP explainability bundle
│   └── monitoring/                 # Model monitoring + rollback logic
│
├── frontend/                       # React application
│   └── src/
│       ├── App.js
│       ├── pages/Dashboard.js      # Main dashboard UI
│       ├── theme.css               # Design system
│       └── components/             # RealtimeMonitor, AnalysisPanel, Chatbot, etc.
│
├── models/                         # Production model artifacts (PyTorch + sklearn)
│   ├── deep_face_expert.pt         # Face CNN-GRU encoder
│   ├── deep_voice_expert.pt        # Voice CNN-GRU encoder
│   ├── deep_physio_expert.pt       # Physio CNN-GRU encoder
│   ├── deep_fusion_router.pt       # Flex-Modality Dynamic Router MLP
│   ├── deep_*_scaler.pkl           # StandardScalers per modality
│   ├── *_expert_lightweight.pkl    # Classical sklearn baseline experts (v1)
│   ├── explainability_bundle.json  # SHAP top-driver data
│   ├── deep_fusion_config.json     # Fusion runtime config
│   └── registry.json              # Model version registry & hash manifest
│
├── configs/                        # Contract YAML schemas
│   ├── feature_contract.yaml       # Canonical feature names & dimensions
│   ├── api_contract.yaml
│   ├── schema_contract.yaml
│   └── performance_contract.yaml
│
├── tests/                          # Production test suite (97 tests, 100% pass)
│   ├── test_api_endpoints.py
│   ├── test_runtime_engine.py
│   ├── test_explainability_bundle.py
│   ├── test_phase5_integration.py
│   └── ...
│
├── certified_data/                 # Pre-processed training datasets (CSV, gitignored)
│   └── *.manifest.json             # Dataset integrity manifests
│
├── training/                       # Production training scripts
│   ├── package_phase8_production.py  # ← Re-train all models from scratch
│   └── augmentation.py              # Data augmentation utilities
│
└── model_archive/                  # 📦 Permanent preservation archive
    ├── README.md                   # Master evidence record (all metrics + hashes)
    ├── deep_models/                # Production PyTorch models + docs
    ├── classical_models/           # Phase 4 sklearn baselines + docs
    ├── training_scripts/           # All 13 training scripts with TRAINING_SCRIPTS.md
    ├── reports/                    # All benchmark reports (Phase 6–8, leakage audit)
    └── docs/                       # All research documentation
```

---

## 🚀 Getting Started

### Quick Start (Windows)
```bash
run.bat
```
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5000

### Manual Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend**
```bash
cd frontend
npm install
npm start
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js, Recharts, Socket.IO client |
| Backend | Python, Flask, Flask-SocketIO, Eventlet |
| Deep Learning | PyTorch (1D-CNN + GRU, MLP) |
| Classical ML | scikit-learn (Random Forest, Gradient Boosting) |
| Computer Vision | MediaPipe Tasks API, OpenCV |
| Audio | Librosa, custom autocorrelation |
| Explainability | SHAP (TreeExplainer) |

---

## 📦 Model Archive

All trained model versions, training scripts, benchmark reports, and research documentation are permanently preserved in [`model_archive/`](model_archive/README.md).

This includes:
- Every model version's accuracy, F1-score, confusion matrix, and SHA-256 hash
- All 8 research phases with decision logs
- Full Python training scripts for reproducibility
- The complete methodology and leakage audit history

---

## 📊 Dataset

**StressID** (LORIA Lab, France) — publicly available multimodal stress dataset.  
65 subjects, 11 task conditions: Baseline, Stroop, Math, Reading, Breathing, Video ×2, Counting ×3, Speaking, Relaxation.

---

## 📜 License

This project is for academic research purposes.