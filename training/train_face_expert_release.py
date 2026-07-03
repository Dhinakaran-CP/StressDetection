import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from training.release_expert_model import package_and_release_expert

def train_face_expert():
    print("==============================================")
    print("STAGE 4: Training Face Expert (LOSO Protocol)")
    print("==============================================")
    
    DATA_PATH = "certified_data/face_certified.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Certified dataset not found: {DATA_PATH}. Run Phase 2 first.")
        
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded certified dataset: {len(df)} rows.")
    
    # 1. Prepare Features through Runtime Lock (for dimensions/missing values)
    lock = FeatureRuntimeLock()
    
    # Extract feature column names (assumes the CSV columns start from index 7 after metadata)
    # The contract says 18 features. We will extract exactly those listed in the contract.
    feature_names = lock.contract["modalities"]["face"]["features"]
    
    X_raw = df[feature_names].values
    y = df["label"].values
    groups = df["subject_id"].values
    
    # Fill missing values exactly as runtime will
    X_clean = []
    for row in X_raw:
        X_clean.append(lock.process_face_features(row, scaler=None)[0])
    X = np.array(X_clean)
    
    # 2. Leave-One-Subject-Out via GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # We will pick the last fold as our "hold-out test set" to package, but in true LOSO
    # you'd average metrics. For simplicity of artifact creation, we do one explicit split.
    train_idx, test_idx = list(gkf.split(X, y, groups))[-1]
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]
    
    # Prove no leakage
    train_subjects = set(groups_train)
    test_subjects = set(groups_test)
    assert len(train_subjects.intersection(test_subjects)) == 0, "DATA LEAKAGE DETECTED!"
    
    print(f"Train subjects: {len(train_subjects)} | Test subjects: {len(test_subjects)}")
    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train
    print("Fitting Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 5. Release
    package_and_release_expert("face", model, scaler, X_test_scaled, y_test)

if __name__ == "__main__":
    train_face_expert()
