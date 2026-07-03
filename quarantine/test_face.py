from model import MultimodalStressDetector
import cv2
import numpy as np

model = MultimodalStressDetector()

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imwrite('dummy.jpg', img)

features, bbox = model.extract_facial_features('dummy.jpg')
print("Features:", type(features), "length:", len(features) if features is not None else "None")
