import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from model import MultimodalStressDetector
import os
import sys
import pickle

# Set working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_dataset():
    """
    Load the dataset based on your existing structure
    """
    print("Loading dataset...")
    
    # Load labels
    labels_path = r'f:\Multimodal_stress_Detection\datasets\StressID\StressID Dataset\labels.csv'
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {os.path.abspath(labels_path)}")
        return None, None, None, None
    
    labels = pd.read_csv(labels_path, sep=",", header=0, index_col=0).dropna()
    print(f"Loaded {len(labels)} labels")
    
    # Load features for each modality
    phys_features_path = r'f:\Multimodal_stress_Detection\backend\Feature Extraction\Features\integrated_physio.csv'
    video_features_path = r'f:\Multimodal_stress_Detection\backend\Feature Extraction\Features\integrated_video.csv'
    audio_features_path = r'f:\Multimodal_stress_Detection\backend\Feature Extraction\Features\integrated_audio.csv'
    
    if not os.path.exists(phys_features_path) or not os.path.exists(video_features_path) or not os.path.exists(audio_features_path):
        print("Error: One or more feature files missing.")
        return None, None, None, None
    
    x_phys = pd.read_csv(phys_features_path, sep=",", header=0, index_col=0)
    x_video = pd.read_csv(video_features_path, sep=",", header=0, index_col=0)
    x_audio = pd.read_csv(audio_features_path, sep=",", header=None, index_col=0)
    
    x_audio.index = [i.split('.')[0] for i in list(x_audio.index)]
    common_idx = list(x_phys.index.intersection(x_video.index).intersection(x_audio.index).intersection(labels.index))
    
    if len(common_idx) == 0:
        print("Error: No common samples found!")
        return None, None, None, None
    
    x_phys = x_phys.loc[common_idx]
    x_video = x_video.loc[common_idx]
    x_audio = x_audio.loc[common_idx]
    y = labels.loc[common_idx]['binary-stress']
    
    return x_phys.values, x_video.values, x_audio.values, y.values

def train_model():
    X_phys, X_video, X_audio, y = load_dataset()
    if X_phys is None: return
    
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_phys_train, X_phys_test = X_phys[train_idx], X_phys[test_idx]
    X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_combined_train = np.hstack([X_phys_train, X_audio_train])
    smote = SMOTE(random_state=42)
    X_combined_balanced, y_balanced = smote.fit_resample(X_combined_train, y_train)
    
    n_phys = X_phys_train.shape[1]
    X_phys_train_balanced = X_combined_balanced[:, :n_phys]
    X_audio_train_balanced = X_combined_balanced[:, n_phys:]

    print("\nTraining Voice and Physio Expert Models (Random Forest)...")

    # 1. Voice Expert
    voice_scaler = StandardScaler()
    X_audio_scaled = voice_scaler.fit_transform(X_audio_train_balanced)
    voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
    voice_model.fit(X_audio_scaled, y_balanced)
    
    with open('voice_expert_model.pkl', 'wb') as f:
        pickle.dump({'model': voice_model, 'scaler': voice_scaler}, f)

    # 2. Physio Expert
    phys_scaler = StandardScaler()
    X_phys_scaled = phys_scaler.fit_transform(X_phys_train_balanced)
    phys_model = RandomForestClassifier(n_estimators=100, random_state=42)
    phys_model.fit(X_phys_scaled, y_balanced)

    with open('physio_expert_model.pkl', 'wb') as f:
        pickle.dump({'model': phys_model, 'scaler': phys_scaler}, f)
    
    print("\n[Evaluation]")
    v_acc = accuracy_score(y_test, voice_model.predict(voice_scaler.transform(X_audio_test)))
    print(f"Voice RF Accuracy: {v_acc:.2%}")
    p_acc = accuracy_score(y_test, phys_model.predict(phys_scaler.transform(X_phys_test)))
    print(f"Physio RF Accuracy: {p_acc:.2%}")

if __name__ == "__main__":
    train_model()
