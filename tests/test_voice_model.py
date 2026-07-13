import pickle
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
expert_dir = os.path.join(BASE_DIR, 'models')

try:
    with open(os.path.join(expert_dir, 'voice_expert_lightweight.pkl'), 'rb') as f:
        voice_expert = pickle.load(f)
    with open(os.path.join(expert_dir, 'voice_scaler_lightweight.pkl'), 'rb') as f:
        voice_scaler = pickle.load(f)
        
    print("Voice Expert Type:", type(voice_expert))
    print("Voice Scaler Type:", type(voice_scaler))
    print("Scaler Mean:", voice_scaler.mean_)
    print("Scaler Scale:", voice_scaler.scale_)
    
    # Try a dummy zero vector
    dummy_vec = np.zeros((1, 12))
    dummy_scaled = voice_scaler.transform(dummy_vec)
    print("Scaled zero vector:", dummy_scaled)
    prob_zero = voice_expert.predict_proba(dummy_scaled)[0]
    print("Predict prob for zero vector:", prob_zero)
    
    # Try scaler mean vector
    mean_scaled = voice_scaler.transform(voice_scaler.mean_.reshape(1, -1))
    prob_mean = voice_expert.predict_proba(mean_scaled)[0]
    print("Predict prob for mean vector:", prob_mean)

except Exception as e:
    import traceback
    traceback.print_exc()
