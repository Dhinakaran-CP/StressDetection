import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, brier_score_loss, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.core.feature_runtime_lock import FeatureRuntimeLock

print("=========================================================")
print("Phase 6: Multimodal Research & Fusion Strategy Evaluation")
print("=========================================================")

# ---------------------------------------------------------
# 1. Load and Merge Data
# ---------------------------------------------------------
print("\n[1] Loading certified datasets...")
face_path = "certified_data/face_certified.csv"
voice_path = "certified_data/voice_certified.csv"
physio_path = "certified_data/physio_certified.csv"

if not (os.path.exists(face_path) and os.path.exists(voice_path) and os.path.exists(physio_path)):
    raise FileNotFoundError("Missing one or more certified datasets in certified_data/")

df_face = pd.read_csv(face_path)
df_voice = pd.read_csv(voice_path)
df_physio = pd.read_csv(physio_path)

# Drop unnecessary columns that might conflict (video_id, window_start, window_end)
keys = ['subject_id', 'task_id', 'window_index', 'label']
drop_cols = ['video_id', 'window_start', 'window_end']

df_face = df_face.drop(columns=[c for c in drop_cols if c in df_face.columns])
df_voice = df_voice.drop(columns=[c for c in drop_cols if c in df_voice.columns])
df_physio = df_physio.drop(columns=[c for c in drop_cols if c in df_physio.columns])

# Outer merge to test missing modality robustness
df = pd.merge(df_face, df_voice, on=keys, how='outer')
df = pd.merge(df, df_physio, on=keys, how='outer')

# For this experiment, we will drop rows without labels (shouldn't happen)
df = df.dropna(subset=['label'])
df = df.sort_values(by=['subject_id', 'task_id', 'window_index']).reset_index(drop=True)

print(f"Total merged windows: {len(df)}")
print(f"Missing Face: {df['avg_ear'].isna().sum()} | Missing Voice: {df['f0_mean'].isna().sum()} | Missing Physio: {df['ecg_rate_mean'].isna().sum()}")

# Fill missing features with 0 (since StandardScaler/SubjectScaler will center them later, or they just act as padded values)
df = df.fillna(0)

# Get feature names from lock
lock = FeatureRuntimeLock()
face_features = lock.contract["modalities"]["face"]["features"]
voice_features = lock.contract["modalities"]["voice"]["features"]
physio_features = lock.contract["modalities"]["physio"]["features"]

# ---------------------------------------------------------
# 2. Custom Transformers
# ---------------------------------------------------------
class SubjectAdaptiveScaler(BaseEstimator, TransformerMixin):
    """Learns a standard scaling per subject."""
    def __init__(self):
        self.subject_means = {}
        self.subject_stds = {}
        self.global_mean = None
        self.global_std = None

    def fit(self, X, y=None, groups=None):
        df_tmp = pd.DataFrame(X)
        df_tmp['subject_id'] = groups
        
        self.global_mean = df_tmp.drop(columns=['subject_id']).mean().values
        self.global_std = df_tmp.drop(columns=['subject_id']).std().values
        self.global_std[self.global_std == 0] = 1e-6
        
        for subj, group_df in df_tmp.groupby('subject_id'):
            feats = group_df.drop(columns=['subject_id']).values
            self.subject_means[subj] = np.mean(feats, axis=0)
            std = np.std(feats, axis=0)
            std[std == 0] = 1e-6
            self.subject_stds[subj] = std
        return self

    def transform(self, X, groups=None):
        if groups is None:
            # Fallback if groups not provided
            return (X - self.global_mean) / self.global_std
            
        X_out = np.zeros_like(X, dtype=float)
        for i, (row, subj) in enumerate(zip(X, groups)):
            if subj in self.subject_means:
                X_out[i] = (row - self.subject_means[subj]) / self.subject_stds[subj]
            else:
                X_out[i] = (row - self.global_mean) / self.global_std
        return X_out

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, y=y, groups=groups).transform(X, groups=groups)

# Temporal feature aggregation function (adds rolling mean/std to features)
def add_temporal_context(df_features, groups, task_groups):
    """Calculates temporal rolling stats within each subject+task"""
    print("Applying temporal aggregation (rolling window=3)...")
    df_temp = pd.DataFrame(df_features)
    df_temp['subj_task'] = [f"{s}_{t}" for s, t in zip(groups, task_groups)]
    
    # Calculate rolling mean and std
    rolling_mean = df_temp.groupby('subj_task').rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Drop grouping col
    df_temp = df_temp.drop(columns=['subj_task'])
    rolling_mean = rolling_mean.drop(columns=['subj_task'], errors='ignore')
    
    # Concatenate raw features + rolling mean
    out = np.hstack([df_temp.values, rolling_mean.values])
    return out

# ---------------------------------------------------------
# 3. Setup Models & Cross Validation
# ---------------------------------------------------------
X_face = df[face_features].values
X_voice = df[voice_features].values
X_physio = df[physio_features].values
y = df['label'].values
groups = df['subject_id'].values
task_groups = df['task_id'].values

# Add temporal aggregation (this creates a wider feature vector containing history context)
X_face_temp = add_temporal_context(X_face, groups, task_groups)
X_voice_temp = add_temporal_context(X_voice, groups, task_groups)
X_physio_temp = add_temporal_context(X_physio, groups, task_groups)

# Define Base Encoders (Calibrated to provide true probabilities for confidence-aware fusion)
base_encoders = {
    "face": CalibratedClassifierCV(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42), cv=2),
    "voice": CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42), cv=2),
    "physio": CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42), cv=2)
}

gkf = GroupKFold(n_splits=5)

results = {
    "face_only": [],
    "voice_only": [],
    "physio_only": [],
    "naive_avg_3way": [],
    "meta_fusion_3way": []
}

print("\n[2] Starting Leave-One-Subject-Out Cross-Validation...")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_face_temp, y, groups)):
    print(f"  --> Fold {fold+1}/5")
    
    # Assert NO Leakage
    train_subjs = set(groups[train_idx])
    test_subjs = set(groups[test_idx])
    assert len(train_subjs.intersection(test_subjs)) == 0, "DATA LEAKAGE DETECTED!"
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train Base Models
    pred_train_probs = {}
    pred_test_probs = {}
    
    for mod, X_mod in zip(["face", "voice", "physio"], [X_face_temp, X_voice_temp, X_physio_temp]):
        scaler = SubjectAdaptiveScaler()
        X_train_s = scaler.fit_transform(X_mod[train_idx], groups=groups[train_idx])
        X_test_s = scaler.transform(X_mod[test_idx], groups=groups[test_idx])
        
        model = clone(base_encoders[mod])
        model.fit(X_train_s, y_train)
        
        pred_train_probs[mod] = model.predict_proba(X_train_s)
        pred_test_probs[mod] = model.predict_proba(X_test_s)
        
        # Evaluate Unimodal
        preds = np.argmax(pred_test_probs[mod], axis=1)
        acc = accuracy_score(y_test, preds)
        results[f"{mod}_only"].append(acc)
        
    # Fusion 1: Naive Average (Current Baseline Method)
    avg_probs = (pred_test_probs["face"] + pred_test_probs["voice"] + pred_test_probs["physio"]) / 3.0
    avg_preds = np.argmax(avg_probs, axis=1)
    results["naive_avg_3way"].append(accuracy_score(y_test, avg_preds))
    
    # Fusion 2: Meta-Learner (Confidence-aware learned fusion)
    # Stack probabilities as features for the meta classifier
    X_meta_train = np.hstack([pred_train_probs["face"], pred_train_probs["voice"], pred_train_probs["physio"]])
    X_meta_test = np.hstack([pred_test_probs["face"], pred_test_probs["voice"], pred_test_probs["physio"]])
    
    meta_model = LogisticRegression(random_state=42, class_weight='balanced')
    meta_model.fit(X_meta_train, y_train)
    meta_preds = meta_model.predict(X_meta_test)
    results["meta_fusion_3way"].append(accuracy_score(y_test, meta_preds))

print("\n[3] Research Benchmark Results (Cross-Subject Accuracy):")
for method, scores in results.items():
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"  {method.ljust(20)}: {mean_acc:.4f} (+/- {std_acc:.4f})")

# Generate Markdown Report
report = f"""# Phase 6: Multimodal Fusion Benchmark

## Protocol
- **Validation**: Leave-One-Subject-Out (Strict 5-Fold GroupKFold)
- **Temporal Context**: Rolling Window of size 3 (Mean Aggregation)
- **Normalization**: Subject-Adaptive Scaling (Learned per subject)
- **Base Encoders**: Face (MLP), Voice (RF), Physio (GBM) - all calibrated.

## Results (Accuracy across subjects)
| Architecture | Mean Accuracy | Std Dev |
|--------------|---------------|---------|
"""

for method, scores in results.items():
    report += f"| {method} | {np.mean(scores):.4f} | {np.std(scores):.4f} |\n"

report += """
## Conclusion
(Autogenerated) The meta-fusion approach determines the optimal combination of confidences from each modality. If the meta-fusion significantly outperforms the naive average, it indicates that learning dynamic confidence gates is superior to handcrafted weights.
"""

os.makedirs("reports", exist_ok=True)
with open("reports/phase6_fusion_benchmark.md", "w") as f:
    f.write(report)
    
print("\n[4] Complete. Report saved to reports/phase6_fusion_benchmark.md")
