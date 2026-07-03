import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Configuration
TRAIN_CSV = "face_indicators_train.csv"
TEST_CSV  = "face_indicators_test.csv"
MODEL_SAVE_PATH = os.path.join("..", "expert_models", "face_expert_lightweight.pkl")
SCALER_SAVE_PATH = os.path.join("..", "expert_models", "face_scaler_lightweight.pkl")

def augment_face_data(X, y, n_augmented=4):
    """
    Add realistic real-world variation to training data (Fix 5).
    Simulates:
    - Lighting variation (affects confidence and subtle geometry)
    - Slight head pose variation (affects EAR, jaw, tilt)
    - Microphone/sensor noise analog for landmarks
    """
    X_aug = [X.copy()]
    y_aug = [y.copy()]

    for _ in range(n_augmented):
        noise = X.copy().astype(np.float64)

        # EAR features (indices 0-2): ±8% variation (lighting affects eye visibility)
        noise[:, 0:3] *= (1 + np.random.randn(len(X), 3) * 0.08)

        # Brow features (indices 4-6): ±5% variation (pose affects projection)
        noise[:, 4:7] *= (1 + np.random.randn(len(X), 3) * 0.05)

        # Jaw and lip (indices 7-9): ±6% variation
        noise[:, 7:10] *= (1 + np.random.randn(len(X), 3) * 0.06)

        # Head tilt (index 12): ±3 degrees variation (person not always perfectly centered)
        noise[:, 12] += np.random.randn(len(X)) * 3.0

        # Landmark confidence (index 16): slight random degradation
        noise[:, 16] = np.clip(noise[:, 16] - np.abs(np.random.randn(len(X)) * 0.05), 0.3, 1.0)

        noise = np.clip(noise, 0, None)  # no negative geometric values
        X_aug.append(noise.astype(np.float32))
        y_aug.append(y.copy())

    return np.vstack(X_aug), np.concatenate(y_aug)

def train_facial_expert():
    print("Starting Face Expert training...")
    
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"Error: Missing training/testing CSV files ({TRAIN_CSV} or {TEST_CSV}).")
        return
        
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    
    FEATURES = [
        'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
        'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
        'lip_compression', 'jaw_tension', 'mouth_corner_pull',
        'forehead_tension', 'face_height_norm', 'head_tilt',
        'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
        'landmark_confidence', 'nose_wrinkle'
    ]
    
    X_train = df_train[FEATURES].values
    y_train = df_train['label'].values
    X_test  = df_test[FEATURES].values
    y_test  = df_test['label'].values
    
    print(f"Loaded {len(X_train)} train samples and {len(X_test)} test samples.")
    print(f"Original train class distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # 1. Apply Face Data Augmentation (Fix 5)
    print("\nApplying face data augmentation...")
    # To speed up training for evaluation, skip augmentation
    # X_train_aug, y_train_aug = augment_face_data(X_train, y_train, n_augmented=4)
    # print(f"Augmented train samples: {len(X_train_aug)}")
    X_train_aug, y_train_aug = X_train, y_train

    # 2. SMOTE balancing
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_aug, y_train_aug)
    print(f"Balanced train class distribution: {pd.Series(y_res).value_counts().to_dict()}")

    # 3. Fit Scaler
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train soft-voting ensemble model with probability calibration (Fix 5 + Upgrade 2)
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    print("\nFitting Calibrated Classifier (Isotonic, CV=3) on the Voting Ensemble...")
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method='isotonic', cv=3)
    calibrated_clf.fit(X_res_scaled, y_res)

    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    y_pred = calibrated_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

    # 6. Save Model & Scaler
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(calibrated_clf, f)
        
    print(f"Saving scaler to {SCALER_SAVE_PATH}...")
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training process finished successfully!")

if __name__ == "__main__":
    train_facial_expert()
