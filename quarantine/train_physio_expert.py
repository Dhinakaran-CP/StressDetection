import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configuration
LABELS_CSV = "Dataset/labels.csv"
PHYSIO_CSV = "Feature Extraction/Features/integrated_physio.csv"
MODEL_SAVE_PATH = os.path.join("..", "expert_models", "physio_expert.pkl")
SCALER_SAVE_PATH = os.path.join("..", "expert_models", "physio_scaler.pkl")

def train_physio_expert():
    print("Starting Physio Expert training...")
    
    # Ensure working directory is the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.exists(LABELS_CSV) or not os.path.exists(PHYSIO_CSV):
        print(f"Error: Missing CSV files ({LABELS_CSV} or {PHYSIO_CSV}).")
        return
        
    labels = pd.read_csv(LABELS_CSV, index_col=0).dropna()
    x_phys = pd.read_csv(PHYSIO_CSV, index_col=0)
    
    common_idx = list(x_phys.index.intersection(labels.index))
    print(f"Found {len(common_idx)} common samples between labels and physio features.")
    
    if len(common_idx) == 0:
        return
        
    X_full = x_phys.loc[common_idx].values
    y = labels.loc[common_idx, 'binary-stress'].values
    
    # Truncate to the 51 active features (42 EEG + 9 GSR) used by our real-time pipeline.
    # The older integrated_physio.csv has 133 columns due to zero-padding in legacy code.
    X_active = X_full[:, :51]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_active, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # SMOTE balancing
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Balanced train class distribution: {pd.Series(y_res).value_counts().to_dict()}")

    # Fit Scaler
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    # Train Soft-Voting Ensemble Model
    print("Training Physio Expert Soft-Voting Ensemble (Gradient Boosting + Random Forest)...")
    clf = VotingClassifier(
        estimators=[
            ('gb',  GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)),
            ('rf',  RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))
        ],
        voting='soft',
        n_jobs=-1
    )
    
    print("Fitting Calibrated Classifier on the Voting Ensemble...")
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv=3)
    calibrated_clf.fit(X_res_scaled, y_res)

    # Evaluate
    print("\nEvaluating on Test Set...")
    y_pred = calibrated_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

    # Save Model & Scaler
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(calibrated_clf, f)
        
    print(f"Saving scaler to {SCALER_SAVE_PATH}...")
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training process finished successfully!")

if __name__ == "__main__":
    train_physio_expert()
