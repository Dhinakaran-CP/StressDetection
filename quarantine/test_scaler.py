import pickle
import numpy as np

with open('expert_models/face_scaler_lightweight.pkl', 'rb') as f:
    scaler = pickle.load(f)
    print("Face scaler expects", getattr(scaler, 'n_features_in_', 'Unknown'), "features")
