import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

with open('expert_models/face_expert_lightweight.pkl', 'rb') as f:
    face_model = pickle.load(f)
    print("Face model expects", face_model.n_features_in_, "features")

with open('expert_models/voice_expert_lightweight.pkl', 'rb') as f:
    voice_model = pickle.load(f)
    print("Voice model expects", voice_model.n_features_in_, "features")

with open('expert_models/physio_expert.pkl', 'rb') as f:
    physio_model = pickle.load(f)
    print("Physio model expects", physio_model.n_features_in_, "features")
