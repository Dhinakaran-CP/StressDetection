# Model Card: Random Forest Backend Classifier

## Model Details
* **Model Name**: Random Forest Backend Classifier
* **Version**: 1.0.0
* **Framework**: scikit-learn (RandomForestClassifier)
* **Hyperparameters**: `n_estimators=50`, `random_state=42`, `class_weight='balanced'`
* **Input Dimension**: 368 conformed flat features (Face: 170, Voice: 120, Physio: 70)
* **Output**: Binary probability score of stressed state (0: Non-Stressed, 1: Stressed)

---

## Intended Use
* **Primary Use Case**: Real-time stress detection on resource-constrained devices, edge platforms, and lightweight backend services.
* **Fallback Mode**: Acts as the primary backend classifier when deep sequence model evaluation (SSVB-CASA-AIS) is disabled or latency constraints require sub-5ms inference.

---

## Preprocessing & Feature Schema
* **Input Alignment**: All missing modalities are initialized to `NaN` and mean-imputed to `0.0`.
* **Feature Scaling**: Flat features are scaled using subject-wise StandardScaler baseline.

---

## Evaluation Metrics (LOSO Cross-Validation)
The following metrics were evaluated under subject-independent Leave-One-Subject-Out (LOSO) cross-validation on the Combined 91-subject dataset:

| Metric | Realized Value |
| :--- | :--- |
| **Accuracy** | **74.30%** |
| **F1-Score (Stressed Class)** | **0.6325** |
| **Average Latency** | **< 5 ms (CPU)** |
| **Memory Footprint** | **~2.5 MB** |

---

## Limitations & Disclaimers
* **Disclaimer**: This model is not a medical diagnostic tool. Stress predictions are for wellness and behavioral analysis.
* **Modality Dependency**: Optimal accuracy is achieved when all physical and vocal modalities are available; performance may degrade gracefully if only physiological signals are present.
