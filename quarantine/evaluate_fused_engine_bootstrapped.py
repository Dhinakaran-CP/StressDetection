import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=====================================================")
print("  STRICT BOOTSTRAPPED EVALUATION (NO DATA LEAKAGE)   ")
print("=====================================================\n")

# ==========================================
# 1. LOAD DATA & EXTRACT EVALUATION SAMPLES
# ==========================================
print("Loading Face, Voice, and Physio Datasets...")
face_df = pd.read_csv('../dataset_extracted/face_indicators_stressid.csv')
voice_df = pd.read_csv('../dataset_extracted/voice_indicators_stressid.csv')

labels = pd.read_csv('training/Dataset/labels.csv', index_col=0).dropna()
physio_raw = pd.read_csv('training/Feature Extraction/Features/integrated_physio.csv', index_col=0)
common_idx = list(physio_raw.index.intersection(labels.index))
labels = labels.loc[common_idx]
physio_raw = physio_raw.loc[common_idx]

print("Extracting 20 Curated Synchronized Samples to prevent Data Leakage...")
real_face = pd.read_csv('../evaluation_data/face_eval_samples.csv')
real_voice = pd.read_csv('../evaluation_data/voice_eval_samples.csv')

stressed_idx = labels[labels['binary-stress'] == 1].index.tolist()[:10]
calm_idx = labels[labels['binary-stress'] == 0].index.tolist()[:10]
eval_idx = stressed_idx + calm_idx

# Remove these samples from the main datasets
face_df_clean = pd.concat([face_df, real_face, real_face]).drop_duplicates(keep=False)
voice_df_clean = pd.concat([voice_df, real_voice, real_voice]).drop_duplicates(keep=False)
physio_clean = physio_raw.drop(eval_idx)
labels_clean = labels.drop(eval_idx)

X_face = face_df_clean.drop('label', axis=1).values
y_face = face_df_clean['label'].values

X_voice = voice_df_clean.drop('label', axis=1).values
y_voice = voice_df_clean['label'].values

X_physio = physio_clean.values[:, :51]
y_physio = labels_clean['binary-stress'].values

# ==========================================
# 2. STRICT 80/20 SPLIT & SMOTE
# ==========================================
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_face, y_face, test_size=0.2, random_state=42, stratify=y_face)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_voice, y_voice, test_size=0.2, random_state=42, stratify=y_voice)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_physio, y_physio, test_size=0.2, random_state=42, stratify=y_physio)

print("Applying SMOTE & Scalers...")
smote = SMOTE(random_state=42)
Xf_train_res, yf_train_res = smote.fit_resample(Xf_train, yf_train)
Xv_train_res, yv_train_res = smote.fit_resample(Xv_train, yv_train)
Xp_train_res, yp_train_res = smote.fit_resample(Xp_train, yp_train)

scaler_f, scaler_v, scaler_p = StandardScaler(), StandardScaler(), StandardScaler()
Xf_train_sc = scaler_f.fit_transform(Xf_train_res)
Xv_train_sc = scaler_v.fit_transform(Xv_train_res)
Xp_train_sc = scaler_p.fit_transform(Xp_train_res)

# ==========================================
# 3. TRAIN MODELS
# ==========================================
print("Training Base Experts...")
face_model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xf_train_sc, yf_train_res)
voice_model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xv_train_sc, yv_train_res)

p_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
p_rf = RandomForestClassifier(n_estimators=100, random_state=42)
p_voting = VotingClassifier(estimators=[('gb', p_gb), ('rf', p_rf)], voting='soft')
physio_model = CalibratedClassifierCV(estimator=p_voting, cv=3).fit(Xp_train_sc, yp_train_res)

# ==========================================
# 4. SYNCHRONIZED BOOTSTRAPPING
# ==========================================
print("\nBootstrapping 100 Synchronized Evaluation Samples...")
Xf_real = real_face.drop('label', axis=1).values[:, :18]
Xv_real = real_voice.drop('label', axis=1).values[:, :12]
Xp_real = physio_raw.loc[eval_idx].values[:, :51]
y_real = real_face['label'].values

# Indices: 0-9 are Stress (label 1), 10-19 are Calm (label 0)
stress_indices = np.where(y_real == 1)[0]
calm_indices = np.where(y_real == 0)[0]

np.random.seed(42)
N = 50
boot_calm_idx = np.random.choice(calm_indices, N, replace=True)
boot_stress_idx = np.random.choice(stress_indices, N, replace=True)
boot_idx = np.concatenate([boot_calm_idx, boot_stress_idx])

X_face_eval = scaler_f.transform(Xf_real[boot_idx])
X_voice_eval = scaler_v.transform(Xv_real[boot_idx])
X_physio_eval = scaler_p.transform(Xp_real[boot_idx])
y_true = y_real[boot_idx]

# ==========================================
# 5. FUSED ENGINE EVALUATION
# ==========================================
base_weights = {'face': 0.371, 'voice': 0.474, 'physio': 0.338}
norm_weights = {m: w / sum(base_weights.values()) for m, w in base_weights.items()}

fused_preds = []
face_correct, voice_correct, physio_correct, fused_correct = 0, 0, 0, 0

print(f"\n--- FUSED ENGINE EVALUATION REPORT (100 BOOTSTRAPPED SYNCHRONIZED SAMPLES) ---")

for i in range(len(y_true)):
    prob_f = face_model.predict_proba(X_face_eval[i].reshape(1, -1))[0][1]
    prob_v = voice_model.predict_proba(X_voice_eval[i].reshape(1, -1))[0][1]
    prob_p = physio_model.predict_proba(X_physio_eval[i].reshape(1, -1))[0][1]
    
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
