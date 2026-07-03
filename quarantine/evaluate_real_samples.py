import pandas as pd
import numpy as np
import pickle
import os

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Loading Real Data for Evaluation...")

labels_path = r'training/Dataset/labels.csv'
physio_path = r'training/Feature Extraction/Features/integrated_physio.csv'
face_eval_path = r'../evaluation_data/face_eval_samples.csv'
voice_eval_path = r'../evaluation_data/voice_eval_samples.csv'

# Load labels
labels = pd.read_csv(labels_path, index_col=0).dropna()

# Load features for physio
try:
    x_physio = pd.read_csv(physio_path, index_col=0)
except Exception as e:
    print(f"Error loading physio features: {e}")
    exit(1)

common_idx = list(x_physio.index.intersection(labels.index))

if not common_idx:
    print("Error: No common indexes found between modalities and labels.")
    exit(1)

labels = labels.loc[common_idx]
x_physio = x_physio.loc[common_idx]

# Split 10 stressed and 10 calm for Physio
stressed_idx = labels[labels['binary-stress'] == 1].index.tolist()[:10]
calm_idx = labels[labels['binary-stress'] == 0].index.tolist()[:10]
eval_idx = stressed_idx + calm_idx
y_true_physio = labels.loc[eval_idx, 'binary-stress'].values
X_physio = x_physio.loc[eval_idx].values

# Load newly extracted Face and Voice evaluation samples
try:
    face_eval_df = pd.read_csv(face_eval_path)
    voice_eval_df = pd.read_csv(voice_eval_path)
    
    X_face = face_eval_df.drop('label', axis=1).values
    y_true_face = face_eval_df['label'].values
    
    X_voice = voice_eval_df.drop('label', axis=1).values
    y_true_voice = voice_eval_df['label'].values
except Exception as e:
    print(f"Error loading face/voice evaluation samples: {e}")
    exit(1)

# Load Models
try:
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
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Ensure shapes match the models' expectations.
# The face lightweight model expects 18 features. The raw integrated_video might have more.
if X_face.shape[1] > face_scaler.n_features_in_:
    X_face = X_face[:, :face_scaler.n_features_in_]

if X_voice.shape[1] > voice_scaler.n_features_in_:
    X_voice = X_voice[:, :voice_scaler.n_features_in_]

if hasattr(physio_scaler, 'n_features_in_') and X_physio.shape[1] > physio_scaler.n_features_in_:
    X_physio = X_physio[:, :physio_scaler.n_features_in_]

# Evaluate
print(f"\n--- EVALUATION REPORT ({len(eval_idx)} Real Samples) ---")
print(f"Samples: 10 Stressed, 10 Calm")

print("\n--- FACE MODEL RESULTS ---")
face_preds = []
face_correct = 0
for i in range(len(X_face)):
    features = X_face[i].reshape(1, -1)
    scaled = face_scaler.transform(features)
    prob = face_model.predict_proba(scaled)[0][1]
    pred = 1 if prob > 0.5 else 0
    actual = int(y_true_face[i])
    if pred == actual: face_correct += 1
    state = "STRESSED" if actual == 1 else "CALM    "
    print(f"[{state}] EvalSample_{i:02d} | Predicted Score: {prob:.4f} | Pred: {pred} | Match: {'YES' if pred==actual else 'NO'}")

print(f"Face Model Accuracy: {face_correct/len(X_face) * 100:.1f}%")


print("\n--- VOICE MODEL RESULTS ---")
voice_preds = []
voice_correct = 0
for i in range(len(X_voice)):
    features = X_voice[i].reshape(1, -1)
    scaled = voice_scaler.transform(features)
    prob = voice_model.predict_proba(scaled)[0][1]
    pred = 1 if prob > 0.5 else 0
    actual = int(y_true_voice[i])
    if pred == actual: voice_correct += 1
    state = "STRESSED" if actual == 1 else "CALM    "
    print(f"[{state}] EvalSample_{i:02d} | Predicted Score: {prob:.4f} | Pred: {pred} | Match: {'YES' if pred==actual else 'NO'}")

print(f"Voice Model Accuracy: {voice_correct/len(X_voice) * 100:.1f}%")

print("\n--- PHYSIO MODEL RESULTS ---")
physio_preds = []
physio_correct = 0
for i, idx in enumerate(eval_idx):
    features = X_physio[i].reshape(1, -1)
    scaled = physio_scaler.transform(features)
    prob = physio_model.predict_proba(scaled)[0][1]
    pred = 1 if prob > 0.5 else 0
    actual = int(y_true_physio[i])
    if pred == actual: physio_correct += 1
    state = "STRESSED" if actual == 1 else "CALM    "
    print(f"[{state}] ID: {idx:15} | Predicted Score: {prob:.4f} | Pred: {pred} | Match: {'YES' if pred==actual else 'NO'}")

print(f"Physio Model Accuracy: {physio_correct/20 * 100:.1f}%")
