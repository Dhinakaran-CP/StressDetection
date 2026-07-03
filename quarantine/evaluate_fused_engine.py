import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load evaluation data
face_eval_path = '../evaluation_data/face_eval_samples.csv'
voice_eval_path = '../evaluation_data/voice_eval_samples.csv'
physio_path = 'training/Feature Extraction/Features/integrated_physio.csv'
labels_path = 'training/Dataset/labels.csv'

labels = pd.read_csv(labels_path, index_col=0).dropna()
x_physio = pd.read_csv(physio_path, index_col=0)

common_idx = list(x_physio.index.intersection(labels.index))
labels = labels.loc[common_idx]
x_physio = x_physio.loc[common_idx]

stressed_idx = labels[labels['binary-stress'] == 1].index.tolist()[:10]
calm_idx = labels[labels['binary-stress'] == 0].index.tolist()[:10]
eval_idx = stressed_idx + calm_idx

y_true = labels.loc[eval_idx, 'binary-stress'].values
X_physio = x_physio.loc[eval_idx].values

face_eval_df = pd.read_csv(face_eval_path)
voice_eval_df = pd.read_csv(voice_eval_path)
X_face = face_eval_df.drop('label', axis=1).values
X_voice = voice_eval_df.drop('label', axis=1).values

# Load Models
with open('expert_models/face_expert_lightweight.pkl', 'rb') as f:
    face_model = pickle.load(f)
with open('expert_models/face_scaler_lightweight.pkl', 'rb') as f:
    face_scaler = pickle.load(f)
with open('expert_models/voice_expert_lightweight.pkl', 'rb') as f:
    voice_model = pickle.load(f)
with open('expert_models/voice_scaler_lightweight.pkl', 'rb') as f:
    voice_scaler = pickle.load(f)
with open('expert_models/physio_expert.pkl', 'rb') as f:
    physio_model = pickle.load(f)
with open('expert_models/physio_scaler.pkl', 'rb') as f:
    physio_scaler = pickle.load(f)

if X_face.shape[1] > face_scaler.n_features_in_: X_face = X_face[:, :face_scaler.n_features_in_]
if X_voice.shape[1] > voice_scaler.n_features_in_: X_voice = X_voice[:, :voice_scaler.n_features_in_]
if hasattr(physio_scaler, 'n_features_in_') and X_physio.shape[1] > physio_scaler.n_features_in_: X_physio = X_physio[:, :physio_scaler.n_features_in_]

# Fusion weights (from model.py)
base_weights = {'face': 0.371, 'voice': 0.474, 'physio': 0.338}
norm_weights = {m: w / sum(base_weights.values()) for m, w in base_weights.items()}

fused_preds = []
print("--- FUSED ENGINE EVALUATION REPORT ---")
print(f"Samples: 10 Stressed, 10 Calm\n")

for i in range(len(eval_idx)):
    f_scaled = face_scaler.transform(X_face[i].reshape(1, -1))
    v_scaled = voice_scaler.transform(X_voice[i].reshape(1, -1))
    p_scaled = physio_scaler.transform(X_physio[i].reshape(1, -1))
    
    prob_f = face_model.predict_proba(f_scaled)[0][1]
    prob_v = voice_model.predict_proba(v_scaled)[0][1]
    prob_p = physio_model.predict_proba(p_scaled)[0][1]
    
    fused_score = (prob_f * norm_weights['face']) + (prob_v * norm_weights['voice']) + (prob_p * norm_weights['physio'])
    fused_pred = 1 if fused_score >= 0.5 else 0
    fused_preds.append(fused_pred)
    
    actual = int(y_true[i])
    state = "STRESSED" if actual == 1 else "CALM    "
    print(f"[{state}] ID: {eval_idx[i]:15} | Face: {prob_f:.2f} | Voice: {prob_v:.2f} | Physio: {prob_p:.2f} => FUSED SCORE: {fused_score:.4f} | Pred: {fused_pred} | Match: {'YES' if fused_pred==actual else 'NO'}")

print("\n--- FINAL METRICS ---")
print(classification_report(y_true, fused_preds, target_names=["No Stress", "Stress"]))
print(f"Fused Accuracy: {accuracy_score(y_true, fused_preds) * 100:.1f}%")
