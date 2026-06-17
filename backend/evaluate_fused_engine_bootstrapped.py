import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Bootstrapping 100 Synchronized Evaluation Samples...")

# Load raw datasets
physio_path = 'training/Feature Extraction/Features/integrated_physio.csv'
face_path = '../dataset_extracted/face_indicators_stressid.csv'
voice_path = '../dataset_extracted/voice_indicators_stressid.csv'

labels_path = 'training/Dataset/labels.csv'
labels = pd.read_csv(labels_path, index_col=0).dropna()
x_physio = pd.read_csv(physio_path, index_col=0)

common_idx = list(x_physio.index.intersection(labels.index))
labels = labels.loc[common_idx]
x_physio = x_physio.loc[common_idx]

physio_stress = x_physio[labels['binary-stress'] == 1].values
physio_calm = x_physio[labels['binary-stress'] == 0].values

x_face = pd.read_csv(face_path)
x_voice = pd.read_csv(voice_path)

face_stress = x_face[x_face['label'] == 1].drop('label', axis=1).values
face_calm = x_face[x_face['label'] == 0].drop('label', axis=1).values

voice_stress = x_voice[x_voice['label'] == 1].drop('label', axis=1).values
voice_calm = x_voice[x_voice['label'] == 0].drop('label', axis=1).values

# We will bootstrap 50 Stress and 50 Calm samples (Total 100)
N = 50
np.random.seed(42)

def sample_modality(data, n):
    idx = np.random.choice(len(data), n, replace=True)
    return data[idx]

X_face_eval = np.vstack([sample_modality(face_calm, N), sample_modality(face_stress, N)])
X_voice_eval = np.vstack([sample_modality(voice_calm, N), sample_modality(voice_stress, N)])
X_physio_eval = np.vstack([sample_modality(physio_calm, N), sample_modality(physio_stress, N)])

y_true = np.array([0]*N + [1]*N)

# Load Models
with open('expert_models/face_expert_lightweight.pkl', 'rb') as f: face_model = pickle.load(f)
with open('expert_models/face_scaler_lightweight.pkl', 'rb') as f: face_scaler = pickle.load(f)
with open('expert_models/voice_expert_lightweight.pkl', 'rb') as f: voice_model = pickle.load(f)
with open('expert_models/voice_scaler_lightweight.pkl', 'rb') as f: voice_scaler = pickle.load(f)
with open('expert_models/physio_expert.pkl', 'rb') as f: physio_model = pickle.load(f)
with open('expert_models/physio_scaler.pkl', 'rb') as f: physio_scaler = pickle.load(f)

# Ensure shapes
if X_face_eval.shape[1] > face_scaler.n_features_in_: X_face_eval = X_face_eval[:, :face_scaler.n_features_in_]
if X_voice_eval.shape[1] > voice_scaler.n_features_in_: X_voice_eval = X_voice_eval[:, :voice_scaler.n_features_in_]
if hasattr(physio_scaler, 'n_features_in_') and X_physio_eval.shape[1] > physio_scaler.n_features_in_: X_physio_eval = X_physio_eval[:, :physio_scaler.n_features_in_]

# Weights
base_weights = {'face': 0.371, 'voice': 0.474, 'physio': 0.338}
norm_weights = {m: w / sum(base_weights.values()) for m, w in base_weights.items()}

fused_preds = []
face_correct, voice_correct, physio_correct, fused_correct = 0, 0, 0, 0

print(f"\n--- FUSED ENGINE EVALUATION REPORT (100 SAMPLES) ---")

for i in range(len(y_true)):
    f_scaled = face_scaler.transform(X_face_eval[i].reshape(1, -1))
    v_scaled = voice_scaler.transform(X_voice_eval[i].reshape(1, -1))
    p_scaled = physio_scaler.transform(X_physio_eval[i].reshape(1, -1))
    
    prob_f = face_model.predict_proba(f_scaled)[0][1]
    prob_v = voice_model.predict_proba(v_scaled)[0][1]
    prob_p = physio_model.predict_proba(p_scaled)[0][1]
    
    pred_f = 1 if prob_f >= 0.5 else 0
    pred_v = 1 if prob_v >= 0.5 else 0
    pred_p = 1 if prob_p >= 0.5 else 0
    
    fused_score = (prob_f * norm_weights['face']) + (prob_v * norm_weights['voice']) + (prob_p * norm_weights['physio'])
    fused_pred = 1 if fused_score >= 0.5 else 0
    fused_preds.append(fused_pred)
    
    actual = y_true[i]
    if pred_f == actual: face_correct += 1
    if pred_v == actual: voice_correct += 1
    if pred_p == actual: physio_correct += 1
    if fused_pred == actual: fused_correct += 1

print("\n--- FINAL METRICS COMPARISON (100 Samples) ---")
print(f"Face Modality Accuracy:   {face_correct / len(y_true) * 100:.1f}%")
print(f"Voice Modality Accuracy:  {voice_correct / len(y_true) * 100:.1f}%")
print(f"Physio Modality Accuracy: {physio_correct / len(y_true) * 100:.1f}%")
print(f"FUSED ENGINE ACCURACY:    {fused_correct / len(y_true) * 100:.1f}%")

print("\n--- FUSED ENGINE CLASSIFICATION REPORT ---")
print(classification_report(y_true, fused_preds, target_names=["No Stress", "Stress"]))
