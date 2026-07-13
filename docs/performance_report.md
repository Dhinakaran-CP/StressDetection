# Comprehensive Model Performance & Zoo Audit Report

This report presents a thorough evaluation of the classical, deep, and adversarial models stored in `model_archive` and the current production `models/` directory. All evaluations were run on the `certified_data/` datasets under strict Leave-One-Subject-Out (LOSO) cross-validation and 5-Fold GroupKFold setups.

---

## 📊 Summary of Model Performance Comparison

The table below contrasts the **registered metrics from `MODEL_ZOO.md`** against the **actual evaluated metrics** from this session's benchmark run.

| Model Category & Strategy | Target Modality | Evaluation Setup | Registered Zoo Accuracy | Evaluated Accuracy (Overall) | Evaluated Accuracy (LOSO Mean) | Evaluated F1-Score (Overall) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Face Classical (v1)** | Face | Random Split vs LOSO | 0.6904 | 0.6881 | 0.6875 | 0.6206 |
| **Voice Classical (v1)** | Voice | Random Split vs LOSO | 0.7070 | 0.6084 | 0.6084 | 0.7384 |
| **Physio Classical (v1)** | Physio | Random Split vs LOSO | 0.6722 | 0.5751 | 0.5759 | 0.5987 |
| **Strategy 4 Face Deep** | Face | Strict LOSO CNN-GRU | 0.6614 | 0.5780 | 0.5781 | 0.0714 |
| **Strategy 4 Voice Deep** | Voice | Strict LOSO CNN-GRU | 0.6243 | 0.6548 | 0.6548 | 0.7805 |
| **Strategy 4 Physio Deep**| Physio | Strict LOSO CNN-GRU | 0.6556 | 0.5727 | 0.5734 | 0.0239 |
| **Strategy 5 Face Adv** | Face | Adversarial CNN-GRU | 0.6706 | 0.6458 | 0.6453 | 0.5016 |
| **Strategy 5 Voice Adv** | Voice | Adversarial CNN-GRU | 0.6186 | 0.6935 | 0.6935 | 0.8156 |
| **Strategy 5 Physio Adv** | Physio | Adversarial CNN-GRU | 0.6424 | 0.5995 | 0.5988 | 0.5960 |
| **Strategy 4 Fusion Router**| Multi-Sensor | Standard MLP Router | 0.6724 | 0.2952 | 0.2955 | 0.0853 |
| **Strategy 5 Fusion Router**| Multi-Sensor | Adversarial Router | **0.6736** | **0.6890** | **0.6885** | **0.8015** |

---

## 🔍 In-Depth Analysis & Key Insights

### 1. Proof of Identity Leakage in Classical Models
Evaluating the classical models (`Classical_v1` pickles) under strict subject-wise boundaries yields a significant drop in accuracy for **Voice** (from **70.70%** down to **60.84%**) and **Physio** (from **67.22%** down to **57.51%**). This validates the security and methodology changes in Phase 8: early classical models on random splits were indeed learning subject identity shortcuts (timbre, vocal range, resting heart rate) rather than pure stress indicators.

### 2. Strategy 4 Representation Collapse (Standard Deep)
The standard deep models for Face and Physio demonstrate extreme bias during testing, with F1-scores dropping to **0.0714** and **0.0239** respectively. The network predicted the calm state (`0`) for almost all sequences, explaining why the standard dynamic router collapsed to **29.52%** accuracy. Without identity suppression training, standard backpropagation failed to generalize stress features across different physiological traits.

### 3. Superiority of Strategy 5 Tuned Adversarial Suppression
The adversarial sequence models (Strategy 5) exhibit highly stable generalizeability across subjects:
*   **Adversarial Voice** reaches **69.35%** accuracy and **0.8156** F1-score.
*   **Adversarial Face** stabilizes at **64.58%** accuracy.
*   **Adversarial Physio** stabilizes at **59.95%** accuracy.
*   **Adversarial Fusion Router** achieves a robust **68.90%** overall accuracy and **68.85%** LOSO Mean Accuracy, outperforming the registered zoo baseline of **67.36%**.

---

## 🎛️ Multi-Sensor Robustness Sweep (Strategy 5 Router)

To test the router's behavior under sensor failures, we simulated all 7 input combinations on the synchronized multimodal dataset (43,110 frames). The Strategy 5 router shows strong graceful degradation:

| Active Modalities | Input Mask ($[M_{face}, M_{voice}, M_{physio}]$) | Overall Accuracy | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **Face Only** | $[1, 0, 0]$ | 0.5624 | 0.6511 |
| **Voice Only** | $[0, 1, 0]$ | 0.6982 | 0.8195 |
| **Physio Only** | $[0, 0, 1]$ | 0.6875 | 0.8001 |
| **Face + Physio** | $[1, 0, 1]$ | 0.6871 | 0.7998 |
| **Face + Voice** | $[1, 1, 0]$ | 0.6333 | 0.7512 |
| **Voice + Physio** | $[0, 1, 1]$ | 0.6879 | 0.8004 |
| **All Sensors Present**| $[1, 1, 1]$ | **0.6890** | **0.8015** |

---

## 📁 Artifacts Produced

1.  **Metric Dump Directory**: All per-model metrics have been exported to the [`performance_metrics/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/performance_metrics) folder in standard JSON format:
    *   [`overall_summary.json`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/performance_metrics/overall_summary.json): Aggregated summary metrics.
    *   Individual model metric files: e.g., `Face_Classical_v1_metrics.json`, `Strategy_5_Fusion_Router_All_metrics.json`, etc.
2.  **Venv and Unit Test Validation**: Clean virtual environment established at the workspace root, and 97/97 tests verified passed.
