import numpy as np
from model import MultimodalStressDetector

model = MultimodalStressDetector()
model.load_model()

# Test with zeros
face = np.zeros(18)
voice = np.zeros(12)

res = model.predict(facial_features=face, voice_features=voice)
print("Result:", res)
