import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("==================================================")
print("  TRAINING SOFT-VOTING ENSEMBLES (FACE & VOICE)   ")
print("==================================================\n")

# Paths
face_path = '../dataset_extracted/face_indicators_stressid.csv'
voice_path = '../dataset_extracted/voice_indicators_stressid.csv'

# 1. Load Data
face_df = pd.read_csv(face_path)
voice_df = pd.read_csv(voice_path)

X_face = face_df.drop('label', axis=1).values
y_face = face_df['label'].values

X_voice = voice_df.drop('label', axis=1).values
y_voice = voice_df['label'].values

# 2. Split Data (80/20)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_face, y_face, test_size=0.2, random_state=42, stratify=y_face)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_voice, y_voice, test_size=0.2, random_state=42, stratify=y_voice)

# 3. Apply Advanced SMOTE (SMOTETomek cleans noisy boundaries)
from imblearn.combine import SMOTETomek
smote = SMOTETomek(random_state=42)
Xf_train_res, yf_train_res = smote.fit_resample(Xf_train, yf_train)
Xv_train_res, yv_train_res = smote.fit_resample(Xv_train, yv_train)

# 4. Scale Data with RobustScaler to handle outliers
from sklearn.preprocessing import RobustScaler
scaler_face = RobustScaler()
Xf_train_scaled = scaler_face.fit_transform(Xf_train_res)
Xf_test_scaled = scaler_face.transform(Xf_test)

scaler_voice = RobustScaler()
Xv_train_scaled = scaler_voice.fit_transform(Xv_train_res)
Xv_test_scaled = scaler_voice.transform(Xv_test)

# 5. Define Ensemble Models with Custom Weights
def create_ensemble():
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    svc = SVC(probability=True, kernel='rbf', C=2.0, random_state=42)
    
    # Give more weight to Tree-based models which handle this tabular data better
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('svc', svc)],
        voting='soft',
        weights=[2, 1.5, 1]
    )
    return ensemble

face_ensemble = create_ensemble()
voice_ensemble = create_ensemble()

# 6. Train Models
print("--> Training Face Soft-Voting Ensemble (GB + RF + SVM)...")
face_ensemble.fit(Xf_train_scaled, yf_train_res)
print("    Done.\n")

print("--> Training Voice Soft-Voting Ensemble (GB + RF + SVM)...")
voice_ensemble.fit(Xv_train_scaled, yv_train_res)
print("    Done.\n")

# 7. Evaluate Models
print("==================================================")
print("              EVALUATION RESULTS                  ")
print("==================================================\n")

print("--- FACE ENSEMBLE ---")
yf_pred = face_ensemble.predict(Xf_test_scaled)
print(f"Face Accuracy: {accuracy_score(yf_test, yf_pred) * 100:.2f}%\n")
print(classification_report(yf_test, yf_pred, target_names=["No Stress", "Stress"]))

print("--- VOICE ENSEMBLE ---")
yv_pred = voice_ensemble.predict(Xv_test_scaled)
print(f"Voice Accuracy: {accuracy_score(yv_test, yv_pred) * 100:.2f}%\n")
print(classification_report(yv_test, yv_pred, target_names=["No Stress", "Stress"]))
