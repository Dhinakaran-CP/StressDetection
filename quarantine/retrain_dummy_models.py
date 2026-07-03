import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Set working directory to this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_voice_model():
    print("Generating calibrated voice model...")
    np.random.seed(42)
    
    # Generate 500 samples of calm voice
    # Calm voice typically has F0 ~150Hz, low intensity, low high-freq ratio, etc.
    calm_f0 = np.random.normal(150.0, 15.0, 500)
    calm_f0_std = np.random.normal(5.0, 2.0, 500)
    calm_f0_range = np.random.normal(20.0, 5.0, 500)
    calm_jitter = np.random.normal(0.5, 0.2, 500)
    calm_shimmer = np.random.normal(0.1, 0.05, 500)
    calm_hnr = np.random.normal(20.0, 3.0, 500)
    calm_speak_rate = np.random.normal(0.02, 0.005, 500)
    calm_intensity = np.random.normal(0.05, 0.02, 500)
    calm_high_freq = np.random.normal(0.01, 0.005, 500)
    calm_flux = np.random.normal(0.02, 0.01, 500)
    calm_pause = np.random.normal(0.15, 0.05, 500)
    calm_voiced_frac = np.random.normal(0.8, 0.1, 500)
    
    X_calm = np.column_stack([
        calm_f0, calm_f0_std, calm_f0_range, calm_jitter, calm_shimmer,
        calm_hnr, calm_speak_rate, calm_intensity, calm_high_freq, calm_flux,
        calm_pause, calm_voiced_frac
    ])
    y_calm = np.zeros(500)
    
    # Generate 500 samples of stressed voice
    stress_f0 = np.random.normal(250.0, 25.0, 500)
    stress_f0_std = np.random.normal(20.0, 5.0, 500)
    stress_f0_range = np.random.normal(60.0, 15.0, 500)
    stress_jitter = np.random.normal(3.5, 1.0, 500)
    stress_shimmer = np.random.normal(1.0, 0.3, 500)
    stress_hnr = np.random.normal(8.0, 2.0, 500)
    stress_speak_rate = np.random.normal(0.08, 0.02, 500)
    stress_intensity = np.random.normal(0.25, 0.1, 500)
    stress_high_freq = np.random.normal(0.15, 0.05, 500)
    stress_flux = np.random.normal(0.1, 0.05, 500)
    stress_pause = np.random.normal(0.02, 0.01, 500)
    stress_voiced_frac = np.random.normal(0.5, 0.15, 500)
    
    X_stress = np.column_stack([
        stress_f0, stress_f0_std, stress_f0_range, stress_jitter, stress_shimmer,
        stress_hnr, stress_speak_rate, stress_intensity, stress_high_freq, stress_flux,
        stress_pause, stress_voiced_frac
    ])
    y_stress = np.ones(500)
    
    X = np.vstack([X_calm, X_stress])
    y = np.concatenate([y_calm, y_stress])
    
    voice_scaler = StandardScaler()
    X_scaled = voice_scaler.fit_transform(X)
    
    base_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    voice_expert = CalibratedClassifierCV(base_clf, cv=5, method='isotonic')
    voice_expert.fit(X_scaled, y)
    
    with open('../expert_models/voice_expert_lightweight.pkl', 'wb') as f:
        pickle.dump(voice_expert, f)
    with open('../expert_models/voice_scaler_lightweight.pkl', 'wb') as f:
        pickle.dump(voice_scaler, f)
        
    print(f"Saved voice model. Scaler means: {voice_scaler.mean_}")

def generate_physio_model():
    print("Generating 51-feature physio model...")
    np.random.seed(42)
    
    # 51 features: 42 EEG + 9 GSR
    X_calm = np.random.randn(500, 51)
    X_stress = np.random.randn(500, 51) + 1.0  # simple shift
    
    X = np.vstack([X_calm, X_stress])
    y = np.concatenate([np.zeros(500), np.ones(500)])
    
    physio_scaler = StandardScaler()
    X_scaled = physio_scaler.fit_transform(X)
    
    base_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    physio_expert = CalibratedClassifierCV(base_clf, cv=5, method='isotonic')
    physio_expert.fit(X_scaled, y)
    
    with open('../expert_models/physio_expert.pkl', 'wb') as f:
        pickle.dump(physio_expert, f)
    with open('../expert_models/physio_scaler.pkl', 'wb') as f:
        pickle.dump(physio_scaler, f)
        
    print(f"Saved physio model with n_features_in_={physio_scaler.n_features_in_}")

if __name__ == '__main__':
    generate_voice_model()
    generate_physio_model()
