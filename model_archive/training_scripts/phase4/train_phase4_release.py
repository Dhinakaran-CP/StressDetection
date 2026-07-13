import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from training.release_expert_model import package_and_release_expert

def subject_aware_normalization(df, feature_cols):
    df_norm = df.copy()
    for subject in df['subject_id'].unique():
        sub_mask = df['subject_id'] == subject
        # Assume first 2 windows of a subject's session as 'calibration' baseline
        # (This mimics the runtime engine's behavior of calibrating on first 2 frames)
        subject_data = df.loc[sub_mask].sort_values(by=['task_id', 'window_index'])
        if len(subject_data) > 0:
            mean_vals = subject_data[feature_cols].iloc[:2].mean()
            df_norm.loc[sub_mask, feature_cols] = df.loc[sub_mask, feature_cols] - mean_vals
    return df_norm

def apply_temporal_windowing(df, feature_cols, window_size=2):
    """Averages features over consecutive windows within the same task."""
    df_grouped = df.copy()
    df_grouped = df_grouped.sort_values(by=['subject_id', 'task_id', 'window_index'])
    df_grouped[feature_cols] = df_grouped.groupby(['subject_id', 'task_id'])[feature_cols].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    return df_grouped

def train_expert(modality):
    print("==============================================")
    print(f"Training {modality.capitalize()} Expert (Phase 4 Methodology)")
    print("==============================================")
    
    DATA_PATH = f"certified_data/{modality}_certified.csv"
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}. Skipping.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # 1. Prepare Features through Runtime Lock 
    lock = FeatureRuntimeLock()
    feature_names = lock.contract["modalities"][modality]["features"]
    
    # Pre-process raw features via lock to handle NaN and exact dimension ordering
    X_raw = df[feature_names].values
    X_clean = []
    if modality == 'face':
        process_func = lock.process_face_features
    elif modality == 'voice':
        process_func = lock.process_voice_features
    else:
        process_func = lock.process_physio_features
        
    for row in X_raw:
        X_clean.append(process_func(row, scaler=None)[0])
    df[feature_names] = np.array(X_clean)
    
    # Apply Phase 4 Transformations
    df_norm = subject_aware_normalization(df, feature_names)
    df_win = apply_temporal_windowing(df_norm, feature_names, window_size=2)
    
    X = df_win[feature_names].values
    y = df_win["label"].values
    groups = df_win["subject_id"].values
    
    # 2. Leave-One-Subject-Out via GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    train_idx, test_idx = list(gkf.split(X, y, groups))[-1]
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Random Forest (Phase 4 configs)
    print("Fitting Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=4, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Release
    package_and_release_expert(modality, model, scaler, X_test_scaled, y_test)

if __name__ == "__main__":
    train_expert("face")
    train_expert("voice")
    train_expert("physio")
